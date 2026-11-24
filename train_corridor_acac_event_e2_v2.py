"""事件触发版 ACAC：四方法 × 四种子。此入口不修改共享物理或旧实验。"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
import pickle
from pathlib import Path
import shutil
import time
import traceback

import train_eval_corridor_uagmc_happo_mappo as base
from corridor_dispatch import PriorityCorridor
from corridor_env import Request, validate_requests

UPSTREAM = "5ca6afb5e00f281c2be1f4cf98f3493c80714d4e"
STEPS = 1_996_800
SEEDS = (2101, 2102, 2103, 2104)
MARKS = (204800, 501760, 1003520, 1597440, 1699840, 1802240, 1904640, STEPS)
RATES = (1.0, 1.1, 1.2, 1.3, 1.4)
METHODS = {"FULL": (True, True, True), "NO_TE": (False, True, True),
           "NO_SA": (True, False, True), "MICRO_GAE": (True, True, False)}
LANES, LENGTH = 5, 2048
CONTEXT_DIM = 3 + 4*8


class EventPhysics(PriorityCorridor):
    """只替换推进边界与调机调用门。生命周期、可行性和服务事件仍由父类处理。"""

    def advance_to(self, tick):
        if not self.now < tick <= self.cutoff:
            raise ValueError("推进边界必须在未来且不越过截止")
        pieces = []
        while self.now < tick and not self.done:
            end = min(tick, self._events[0][0]) if self._events else tick
            if end <= self.now:
                raise RuntimeError("当前时刻事件应在推进前清空")
            people = len(self.passengers) - self.finished
            pieces.append((self.now, end, people))
            self.area_ticks += (end - self.now) * people
            self.now = end
            self._drain()
        return pieces

    def choose_returns(self, requested, stat):
        # 只撤去分钟门，原候选、顺序、容量、投影、回退与统计不变。
        if not self.policy_returns:
            raise RuntimeError("规则空返模式禁止策略调用")
        legal = self.targets()
        stat["air_decisions"] += 1
        stat["air_no_opportunity"] += not legal
        stat["air_multi_target"] += len(legal) >= 2
        stat["requested_" + requested] += 1
        if not legal:
            return dict(requested=requested, available=[], applied=None, dispatch=[], fallback="no_target")
        remaining = {s: len(self.queue[s]) for s in self.dep}
        natural = max(self.dep, key=lambda s: self._lq_key(s, remaining))
        priority = requested if requested in legal else natural if natural in legal else legal[0]
        stat["air_priority_projection"] += priority != requested
        stat["applied_" + priority] += 1
        ledger = []
        for plane in self._return_planes():
            possible = [s for s in self.dep if remaining[s] and self._can_return(plane, s)]
            if not possible:
                continue
            dest = priority if priority in possible else max(possible, key=lambda s: self._lq_key(s, remaining))
            result = self._dispatch(plane, dest)
            ledger.append(dict(aircraft=plane.id, destination=dest, status=result,
                               fallback=None if dest == priority else "priority_ineligible_or_batch_demand_met"))
            if result == "charging":
                stat["air_charge_only"] += 1
            if result != "reserved":
                continue
            remaining[dest] = max(0, remaining[dest] - self.cfg.seats)
            stat["air_dispatched"] += 1
            stat["executed_" + dest] += 1
            stat["air_requested_executed"] += dest == requested
            self.totals["priority_match" if dest == priority else "priority_fallback"] += 1
        return dict(requested=requested, available=list(legal), applied=priority, dispatch=ledger,
                    fallback=None if requested == priority else "priority_projection")


class Lane:
    """决策事件与只读分钟计量点分开；分钟计量点从不调用 Actor。"""

    def __init__(self, gym, np, cfg, rows):
        self.rows = rows
        self.view = base.build_gym(gym, np)(cfg, rows, "MAPPO")
        self.view.env = EventPhysics(cfg, policy_returns=True)
        self.env = self.view.env
        self.minute = self.env._ticks(cfg.decision_min)
        self.reset()

    def reset(self):
        self.view.reset(seed=123)
        self.meter = self.minute
        self.air_tick = -1
        self.air_signature = None
        self.focal_id = None
        self._focal_tick = -1
        self.pending = None
        self.event_id = 0

    def snapshot(self):
        return dict(env=self.env, view_history=self.view.history, view_stat=self.view.stat,
                    route_index=self.view.route_index, meter=self.meter, air_tick=self.air_tick,
                    air_signature=self.air_signature, focal_id=self.focal_id, pending=self.pending,
                    event_id=self.event_id, _focal_tick=self._focal_tick)

    def restore(self, state):
        self.env = self.view.env = state["env"]
        self.view.history, self.view.stat = state["view_history"], state["view_stat"]
        self.view.route_index = state["route_index"]
        for key in ("meter", "air_tick", "air_signature", "focal_id", "pending", "event_id", "_focal_tick"):
            setattr(self, key, state[key])

    def critic_context(self):
        # 只看已揭示乘客和当前飞机状态。不遍历未来事件队列，不编码绝对时间。
        e = self.env
        access = Counter(p.dep for p in e.passengers.values() if p.state == "access")
        waiting = sum(p.state == "route" for p in e.passengers.values())
        ready = Counter(a.station for a in e.planes.values() if a.state == "ready")
        inbound = Counter(a.target for a in e.planes.values() if a.target is not None)
        fleet = e.cfg.fleet
        features = [len(e.passengers)/300, (len(e.passengers)-e.finished)/300, waiting/300]
        for s in (*e.dep, *e.arr):
            features.extend((len(e.queue.get(s, ()))/300, access[s]/300, ready[s]/fleet,
                             inbound[s]/fleet, e.local[s]/fleet, e.pa[s]/fleet,
                             e.turn_active[s]/fleet, e.charge_active[s]/fleet))
        if len(features) != CONTEXT_DIM:
            raise ValueError("集中式当前状态仅适配冻结 E2 的四站")
        return features

    def signature(self, targets):
        # 同一批调机不会因剩余队列仍非空就不断重派。等队列或可行飞机集合真实变化。
        e = self.env
        return (tuple((s, len(e.queue[s])) for s in e.dep),
                tuple((p.id, tuple(s for s in targets if e._can_return(p, s)))
                      for p in e._return_planes()))

    def next(self):
        if self.pending is not None:
            raise RuntimeError("有一个待执行的合法决策")
        e = self.env
        pieces = []
        while True:
            if e.done:
                # 回合尾端不足一分钟仍算一次旧版 advance，保留真实 tick 与原奖励。
                return dict(kind="terminal", tick=e.now, pieces=pieces, complete=e.finished == e.request_count)
            if e.now == self.meter:
                self.meter += self.minute
                frame = self.view.snapshot(self.view.focal(), False)[0]
                self.view.history = self.view.history[1:] + [frame]
                return dict(kind="minute", tick=e.now, pieces=pieces)
            if e.now != self.air_tick:
                # focal 仅可引用本物理时刻已经选站的最后一位乘客。
                if getattr(self, "_focal_tick", None) != e.now:
                    self.focal_id = None
            route = min((p.req.id for p in e.passengers.values() if p.state == "route"), default=None)
            if route is not None:
                self.view.focal = lambda: e.passengers[route]
                try:
                    x = self.view.observation()
                finally:
                    del self.view.focal
                return self.offer(0, route, (True, True, True), x, pieces)
            legal = e.targets()
            if legal and self.air_tick != e.now and self.signature(legal) != self.air_signature:
                focal = e.passengers.get(self.focal_id)
                if focal is None:
                    self.view.route_index = 0
                mask = tuple(s in legal for s in base.STATIONS)
                return self.offer(1, None, mask, self.view.observation(post=True, focal=focal), pieces)
            if not legal:
                self.air_signature = None
            tick = min(self.meter, e.cutoff, e._events[0][0] if e._events else e.cutoff)
            pieces.extend(e.advance_to(tick))

    def offer(self, role, passenger, mask, obs, pieces):
        self.pending = (role, passenger, mask)
        self.event_id += 1
        return dict(kind="action", tick=self.env.now, pieces=pieces, event_id=self.event_id,
                    role=role, mask=mask, obs=obs)

    def act(self, a):
        if self.pending is None:
            raise RuntimeError("没有待执行的合法决策")
        role, pid, mask = self.pending
        if not 0 <= a < len(mask) or not mask[a]:
            raise ValueError(f"策略请求了非法机场：{a}; {mask}")
        self.pending = None
        e = self.env
        dest = base.STATIONS[a]
        if role == 0:
            self.view.route_index = a
            self.view.stat["route_" + dest] += 1
            self.focal_id, self._focal_tick = pid, e.now
            e._route(e.passengers[pid], dest)
            record = dict(requested=dest, available=list(base.STATIONS), applied=dest,
                          passenger=pid, dispatch=[], fallback=None)
        else:
            self.air_tick = e.now
            record = e.choose_returns(dest, self.view.stat)
        e._drain()
        if role == 1:
            self.air_signature = self.signature(e.targets())
        return record


def discount_rewards(pieces, start, tick_min, gamma):
    # 每个 tick 在区间左端计入人时，gamma 的指数以物理分钟计。
    tick_gamma = gamma**tick_min
    denom = 1 - tick_gamma
    return sum(-people*tick_min*gamma**((a-start)*tick_min) *
               ((1-tick_gamma**(b-a))/denom if denom else b-a) for a, b, people in pieces)


def gae(values, starts, ends, discounts, continuations, rewards, lam):
    # ends 是对应下一状态的目标网络估值；真终止传零，时间截断仍 bootstrap。
    assert len(values) == len(starts) == len(ends) == len(discounts) == len(continuations) == len(rewards)
    result = [0.] * len(values)
    carry = 0.
    for k in reversed(range(len(values))):
        carry = rewards[k] + discounts[k]*ends[k] - values[k] + lam*discounts[k]*continuations[k]*carry
        result[k] = carry
    return result


def credit(records, tick_min, gamma, lam, macro):
    """输入以一个 rollout 或 episode 末端 probe 收尾；不跨 reset 传递优势。"""
    # 每条记录的 pieces 是上一条记录到本条记录的增量。同 tick 的顺序动作增量为零。
    event_indices = [i for i, r in enumerate(records) if r["kind"] == "action"]
    if not event_indices:
        return []
    delta = lambda i, j: (records[j]["tick"] - records[i]["tick"]) * tick_min

    def cost(i, j):
        return discount_rewards((x for k in range(i+1, j+1) for x in records[k]["pieces"]),
                                records[i]["tick"], tick_min, gamma if macro else 1.)

    if macro:
        starts = event_indices
        ends = starts[1:] + [len(records)-1]
    else:
        starts = [i for i, r in enumerate(records[:-1]) if r["kind"] in ("start", "minute")]
        ends = starts[1:] + [len(records)-1]
    end_values = [0. if records[j].get("complete", False) else records[j]["target_value"] for j in ends]
    adv = gae([records[i]["target_value"] for i in starts], starts, end_values,
              [gamma**delta(i, j) if macro else gamma for i, j in zip(starts, ends)],
              [j != len(records)-1 for j in ends],
              [cost(i, j) for i, j in zip(starts, ends)], lam)
    if macro:
        return [(records[i], records[i]["target_value"]+a, a) for i, a in zip(starts, adv)]
    result = []
    pos = 0
    for i in event_indices:
        while pos + 1 < len(starts) and starts[pos+1] <= i:
            pos += 1
        # 把分钟目标移至该动作时刻：去掉该动作前已经产生的奖励。无需对同分钟动作重复递推 lambda。
        a = starts[pos]
        ret = records[a]["target_value"] + adv[pos] - cost(a, i)
        result.append((records[i], ret, ret-records[i]["target_value"]))
    return result


def model_classes(torch):
    from torch import nn
    from torch.nn import functional as F
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    def linear(inp, out, act="leaky_relu"):
        m = nn.Linear(inp, out)
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(act))
        nn.init.zeros_(m.bias)
        return m

    class Encoder(nn.Module):
        def __init__(self, dim, te, width):
            super().__init__()
            self.te = te
            self.net = nn.Sequential(linear(dim+4*te, width), nn.LeakyReLU(),
                                     linear(width, width), nn.LeakyReLU())
            self.gru = nn.GRU(width, 64, batch_first=True)
            self.register_buffer("time_freq", torch.tensor([1., .01]))

        def embed(self, x, clock):
            if self.te:
                phase = clock[..., None] * self.time_freq
                x = torch.cat((x, torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)), -1)
            return self.net(x)

        def one(self, x, time, h):
            return self.gru(self.embed(x, time)[:, None], h[None])[0][:, 0]

        def sequence(self, x, time, mask_np, h):
            # 按角色压紧到自己的事件序列，向后展开时沿用最近一次输出。
            mask = torch.as_tensor(mask_np, device=x.device)
            count = mask.long().cumsum(1)
            ii, jj = mask.nonzero(as_tuple=True)
            b, t = mask.shape
            if not len(ii):
                return h[:, None].expand(-1, t, -1)
            packed = x.new_zeros((b, t, self.gru.input_size))
            packed[ii, count[ii, jj]-1] = self.embed(x[ii, jj], time[ii, jj])
            lens = mask_np.sum(1).clip(1)
            compressed = pack_padded_sequence(packed, lens, batch_first=True, enforce_sorted=False)
            h_out = self.gru(compressed, h[None])[0]
            h_out = pad_packed_sequence(h_out, batch_first=True, total_length=t)[0]
            return torch.cat((h[:, None], h_out), 1).gather(1, count[..., None].expand(-1, -1, 64))

    class Actor(nn.Module):
        def __init__(self, dim, te):
            super().__init__()
            self.history = Encoder(dim, te, 128)
            self.head = nn.Sequential(linear(64, 64), nn.LeakyReLU(), nn.LayerNorm(64), linear(64, 3, "linear"))

    class PopArt(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.head = nn.Linear(width, 1)
            self.register_buffer("moving", torch.zeros(3))

        def moments(self):
            mass = self.moving[2].clamp(min=1e-5)
            mean = self.moving[0]/mass
            sd = (self.moving[1]/mass-mean.square()).clamp(min=.01).sqrt()
            return mean, sd

        @torch.no_grad()
        def update(self, y):
            old_mu, old_sigma = self.moments()
            self.moving.lerp_(torch.stack((y.mean(), y.square().mean(), y.new_tensor(1.))), 1e-5)
            mu, sigma = self.moments()
            self.head.weight.mul_(old_sigma/sigma)
            self.head.bias.mul_(old_sigma).add_(old_mu-mu).div_(sigma)

        def forward(self, x):
            return self.head(x).squeeze(-1)

        def raw(self, z):
            mu, sigma = self.moments()
            return mu + z*sigma

        def normalize(self, y):
            mu, sigma = self.moments()
            return (y-mu)/sigma

    class Critic(nn.Module):
        def __init__(self, dim, te, attention):
            super().__init__()
            self.history = nn.ModuleList(Encoder(dim, te, 128) for _ in range(2))
            self.current = nn.Sequential(linear(CONTEXT_DIM, 64), nn.LeakyReLU())
            self.attention = attention
            if attention:
                self.position = nn.Parameter(torch.empty(2, 64))
                nn.init.normal_(self.position, std=.02)
                # 对齐一层 GPT2 角色融合拓扑；两个角色互相可见，禁用因果遮罩。
                self.norm_a, self.norm_b, self.norm_out = (nn.LayerNorm(64) for _ in range(3))
                self.sa = nn.MultiheadAttention(64, 2, batch_first=True)
                self.ff = nn.Sequential(nn.Linear(64, 256), nn.GELU(approximate="tanh"), nn.Linear(256, 64))
            else:
                self.mix = nn.Sequential(linear(128, 64), nn.LeakyReLU())
            self.value = PopArt(64)

        def forward(self, h, current):
            shape = h.shape[:-2]
            # 探测和真实动作走同一个估值头，当前状态分支在真实动作样本上学习。
            x = h.reshape(-1, 2, 64) + self.current(current.reshape(-1,CONTEXT_DIM))[:,None]
            if self.attention:
                x = x + self.position
                n = self.norm_a(x)
                x = x + self.sa(n, n, n, need_weights=False)[0]
                x = self.norm_out(x+self.ff(self.norm_b(x))).mean(1)
            else:
                x = self.mix(x.flatten(1))
            return self.value(x).reshape(shape)

    return Actor, Critic


def distribution(torch, actor, h, mask):
    logits = actor.head(h)
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -torch.inf))


class Collector:
    def __init__(self, gym, np, torch, cfg, rows, lanes, actors, critic, target, normalizer):
        self.np, self.torch = np, torch
        self.lanes = [Lane(gym, np, cfg, rows) for _ in range(lanes)]
        self.actors, self.critic, self.target, self.norm = actors, critic, target, normalizer
        self.device = next(critic.parameters()).device
        self.ah = torch.zeros((lanes, 2, 64), device=self.device)
        self.ch, self.th = torch.zeros_like(self.ah), torch.zeros_like(self.ah)
        self.hist = [[[], []] for _ in range(lanes)]
        self.next_norm = None

    def snapshot(self):
        return dict(lanes=[x.snapshot() for x in self.lanes], ah=self.ah.cpu(), ch=self.ch.cpu(),
                    th=self.th.cpu(), hist=self.hist)

    def restore(self, state):
        for lane, s in zip(self.lanes, state["lanes"], strict=True):
            lane.restore(s)
        for k in ("ah", "ch", "th"):
            setattr(self, k, state[k].to(self.device))
        self.hist = state["hist"]
        self.next_norm = None

    def normalize(self, raw):
        return self.np.clip((raw-self.norm.mean)/self.np.sqrt(self.norm.var+1e-8),-10.,10.).astype(self.np.float32)

    def refresh(self):
        # 参数改变后重新编码本回合已见历史，避免把旧网络状态交给新网络。
        if self.next_norm is not None:
            self.norm = self.next_norm
            self.next_norm = None
        torch, np = self.torch, self.np
        with torch.no_grad():
            for i, pair in enumerate(self.hist):
                for role, history in enumerate(pair):
                    if not history:
                        continue
                    x = torch.as_tensor(self.normalize(np.stack([row[0] for row in history])), device=self.device)[None]
                    t = torch.as_tensor([row[1] for row in history], device=self.device)[None]
                    for net, h in ((self.actors[role].history, self.ah), (self.critic.history[role], self.ch),
                                   (self.target.history[role], self.th)):
                        h[i,role] = net.gru(net.embed(x,t))[1][0,0]
    def probe(self, pending):
        # 批量只读当前状态：不采样动作，不更新任何持久隐藏态，也不推进世界。
        if not pending:
            return []
        torch, np = self.torch, self.np
        idx = [i for i,_ in pending]
        state = torch.as_tensor(np.asarray([self.lanes[i].critic_context() for i in idx],np.float32),device=self.device)
        with torch.no_grad():
            result = torch.stack((self.critic.value.raw(self.critic(self.ch[idx],state)),
                                  self.target.value.raw(self.target(self.th[idx],state))),-1).cpu().numpy()
        return [dict(item, value=float(v), target_value=float(t)) for (_,item),(v,t) in zip(pending,result,strict=True)]

    def decision(self, offers, training):
        torch, np = self.torch, self.np
        ix = [i for i, _ in offers]
        clock = np.asarray([r["tick"]*self.lanes[i].env.cfg.tick_min + 1. for i, r in offers], np.float32)
        raw = np.stack([r["obs"] for _, r in offers])
        if training:
            self.next_norm.update(raw)
        obs = self.normalize(raw)
        current = np.asarray([self.lanes[i].critic_context() for i in ix],np.float32) if training else None
        to = lambda x: torch.as_tensor(x, device=self.device)
        x, t = to(obs), to(clock)
        with torch.no_grad():
            prev_actor = torch.stack([self.ah[i, r["role"]] for i, r in offers])
            previous = torch.cat((prev_actor[:, None], self.ch[ix]), 1).cpu().numpy()
            act = torch.empty((len(ix), 2), device=self.device)
            for role in range(2):
                batch = [k for k, (_, r) in enumerate(offers) if r["role"] == role]
                if not batch:
                    continue
                ids = [ix[k] for k in batch]
                state = self.actors[role].history.one(x[batch], t[batch], self.ah[ids, role])
                self.ah[ids, role] = state
                if training:
                    self.ch[ids, role] = self.critic.history[role].one(x[batch], t[batch], self.ch[ids, role])
                    self.th[ids, role] = self.target.history[role].one(x[batch], t[batch], self.th[ids, role])
                dist = distribution(torch, self.actors[role], state, to([offers[k][1]["mask"] for k in batch]))
                a = dist.sample()
                act[batch, 0], act[batch, 1] = a.float(), dist.log_prob(a)
            value = self.critic.value.raw(self.critic(self.ch[ix],to(current))) if training else act.new_zeros(len(ix))
            target_value = self.target.value.raw(self.target(self.th[ix],to(current))) if training else value
            results = torch.column_stack((act, value, target_value)).cpu().numpy()
        for k, (i, r) in enumerate(offers):
            action, logp, v, tv = results[k]
            r.update(obs=obs[k], clock=float(clock[k]), actor_before=previous[k, 0],
                     critic_before=previous[k, 1:], action=int(action), logp=float(logp),
                     value=float(v), target_value=float(tv))
            if training:
                r["current"] = current[k]
                self.hist[i][r["role"]].append((raw[k].copy(), float(clock[k])))

    def collect(self, quota, gamma, lam, macro):
        blocks, examples = [], []
        counts, sample = Counter(), {}
        done = [0] * len(self.lanes)
        seen = [None] * len(self.lanes)
        role_seen = [[None, None] for _ in self.lanes]
        # 一轮固定行为尺度，下一尺度只在更新边界生效；logp 与 PPO 输入严格配对。
        self.next_norm = copy.deepcopy(self.norm)
        def start(ids):
            return self.probe([(i,dict(kind="start",tick=self.lanes[i].env.now,pieces=[])) for i in ids])
        record = [[r] for r in start(range(len(self.lanes)))]
        while min(done) < quota:
            offers = []
            probes, resets = [], []
            for i, lane in enumerate(self.lanes):
                if done[i] == quota:
                    continue
                item = lane.next()
                for a, b, people in item["pieces"]:
                    counts["physical_ticks"] += b-a
                    counts["person_ticks"] += (b-a)*people
                if item["kind"] == "action":
                    offers.append((i, item))
                    continue
                probes.append((i,item))
            for (i,_),item in zip(probes,self.probe(probes),strict=True):
                lane = self.lanes[i]
                record[i].append(item)
                done[i] += 1
                counts["legacy_equivalent_steps"] += 1
                terminal = item["kind"] == "terminal"
                if terminal or done[i] == quota:
                    blocks.append(credit(record[i], lane.env.cfg.tick_min, gamma, lam, macro))
                    if terminal:
                        counts["terminated" if item["complete"] else "truncated"] += 1
                        lane.env.metrics()  # 原物理在这里验证累计人时守恒。
                        lane.reset()
                        self.ah[i].zero_()
                        self.ch[i].zero_()
                        self.th[i].zero_()
                        self.hist[i] = [[], []]
                        seen[i] = None
                        role_seen[i] = [None, None]
                        if done[i]<quota:
                            resets.append(i)
            for i,r in zip(resets,start(resets),strict=True):
                record[i] = [r]
            if not offers:
                continue
            self.decision(offers, True)
            for i, item in offers:
                lane = self.lanes[i]
                effect = lane.act(item["action"])
                record[i].append(item)
                counts["passenger_events" if item["role"] == 0 else "aircraft_events"] += 1
                counts["joint_events"] += 1
                counts["noninteger_events"] += item["tick"] % lane.minute != 0
                if seen[i] is not None:
                    gap = item["tick"] - seen[i]
                    counts[f"joint_gap_ticks_{gap}"] += 1
                seen[i] = item["tick"]
                role = item["role"]
                if role_seen[i][role] is not None:
                    counts[f"role{role}_gap_ticks_{item['tick']-role_seen[i][role]}"] += 1
                role_seen[i][role] = item["tick"]
                for dispatch in effect["dispatch"]:
                    counts["dispatch_"+dispatch["status"]] += 1
                if item["role"] not in sample:
                    sample[item["role"]] = dict(effect, event_id=item["event_id"], tick=item["tick"],
                        role=item["role"], mask=item["mask"], action_log_prob=item["logp"],
                        critic_value_before_action=item["value"], observation=item["obs"].tolist(),
                        hidden_before=item["actor_before"].tolist(),
                        hidden_after=self.ah[i, item["role"]].cpu().tolist())
        # 段内不跨回合。截断反传初值来自在线采样时真实的各角色状态。
        for block in blocks:
            examples.extend(block[j:j+32] for j in range(0, len(block), 32))
        return examples, counts, sample


def learn(torch, np, examples, actors, critic, opts, args):
    if not examples:
        raise RuntimeError("整轮没有合法决策事件，不能训练")
    device = next(critic.parameters()).device
    to = lambda x: torch.as_tensor(x, device=device)
    returns = to(np.asarray([ret for chunk in examples for _, ret, _ in chunk], np.float32))
    critic.value.update(returns)
    reports = []
    for _ in range(args["ppo_epoch"]):
        for ids in np.array_split(np.random.permutation(len(examples)), min(len(examples), args["actor_num_mini_batch"])):
            chunks = [examples[i] for i in ids]
            b, t, dim = len(chunks), max(map(len, chunks)), len(chunks[0][0][0]["obs"])
            obs = np.zeros((b,t,dim), np.float32)
            current = np.zeros((b,t,CONTEXT_DIM),np.float32)
            clock, lp, ret, adv = [np.zeros((b,t), np.float32) for _ in range(4)]
            action = np.zeros((b,t), np.int64)
            roles = np.full((b,t), -1, np.int8)
            mask = np.ones((b,t,3), bool)
            ch = np.stack([chunk[0][0]["critic_before"] for chunk in chunks])
            ah = np.zeros((b,2,64), np.float32)
            for i, chunk in enumerate(chunks):
                counted = set()
                for j, (r, v, a) in enumerate(chunk):
                    role = r["role"]
                    if role not in counted:
                        ah[i,role] = r["actor_before"]
                        counted.add(role)
                    roles[i,j], obs[i,j], clock[i,j], mask[i,j] = role, r["obs"], r["clock"], r["mask"]
                    current[i,j] = r["current"]
                    action[i,j], lp[i,j], ret[i,j], adv[i,j] = r["action"], r["logp"], v, a
            x, clock = to(obs), to(clock)
            masks = [roles == i for i in range(2)]
            values = critic(torch.stack([critic.history[i].sequence(x, clock, masks[i], to(ch[:,i]))
                                         for i in range(2)], -2),to(current))[to(roles >= 0)]
            target = critic.value.normalize(to(ret[roles>=0]))
            loss = (values-target).square().mean()
            optimize(torch, opts[2], critic, loss, args)
            with torch.no_grad():
                ev = 1-(values-target).var(unbiased=False)/(target.var(unbiased=False)+1e-8)
            metrics = dict(value_loss=loss.item(), explained_variance=ev.item())
            for i, model in enumerate(actors):
                hit = masks[i]
                if not hit.any():
                    continue
                feature = model.history.sequence(x, clock, hit, to(ah[:,i]))[to(hit)]
                dist = distribution(torch, model, feature, to(mask[hit]))
                logratio = dist.log_prob(to(action[hit])) - to(lp[hit])
                ratio = logratio.exp()
                clip = args["clip_param"]
                a = to(adv[hit])
                policy = -torch.minimum(ratio*a, ratio.clamp(1-clip,1+clip)*a).mean()
                entropy = dist.entropy().mean()
                optimize(torch, opts[i], model, policy - args["entropy_coef"]*entropy, args)
                metrics.update({f"policy_loss_{i}": policy.item(), f"entropy_{i}": entropy.item(),
                    f"approx_kl_{i}": (ratio-1-logratio).mean().item(),
                    f"clip_fraction_{i}": ((ratio-1).abs()>clip).float().mean().item()})
            reports.append(metrics)
    return dict(minibatches=len(reports),role_minibatches=[sum(f"policy_loss_{i}" in r for r in reports) for i in range(2)],
                **{k: sum(r[k] for r in reports if k in r)/sum(k in r for r in reports)
                   for k in set().union(*(r.keys() for r in reports))})


def optimize(torch, opt, net, loss, args):
    if not torch.isfinite(loss):
        raise FloatingPointError("ACAC 损失出现 NaN/Inf")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), args["max_grad_norm"], error_if_nonfinite=True)
    opt.step()


def canonical(x):
    return json.loads(json.dumps(x, allow_nan=False))


def demand_sha(rows):
    return hashlib.sha256(json.dumps([asdict(r) for r in rows], sort_keys=True).encode()).hexdigest()


def same_source(path, sha):
    lf = path.read_bytes().replace(b"\r\n", b"\n")
    return sha in [hashlib.sha256(s).hexdigest() for s in (lf, lf.replace(b"\n", b"\r\n"))]


def bank_for(cli, versions):
    root = cli.base_output
    protocol = base.read(root / "PROTOCOL.json")
    if (protocol["schema"], protocol["dispatch_protocol"], protocol["steps_per_cell"],
            protocol["rollout"], protocol["dev_marks"]) != (
            "corridor_legacy_interface_v2", base.DISPATCH_PROTOCOL, STEPS, 10240, list(MARKS)):
        raise ValueError("不是旧式分钟调机 v2 的四种子对照合同")
    folder = Path(base.__file__).parent
    for key, path in (("entry_sha", Path(base.__file__)),
                      ("environment_sha", folder / "corridor_env.py"),
                      ("dispatch_sha", folder / "corridor_dispatch.py")):
        if not same_source(path, protocol[key]):
            raise ValueError(f"源码与 MAPPO 基线不一致：{path.name}")
    for k, v in versions.items():
        if str(protocol[k]) != v:
            raise ValueError(f"运行库与基线不一致：{k}: {protocol[k]} != {v}")
    cfg = base.config(cli.config, "E2")
    rows = base.trace(base.config(cli.config, "E1"), cli.train_trace)
    if canonical(asdict(cfg)) != protocol["configs"]["E2"] or demand_sha(rows) != protocol["train_trace_sha256"]:
        raise ValueError("物理或训练需求与原 MAPPO 不一致")
    raw = base.read(root / "DEMAND_BANK.json")
    bank = []
    for term in protocol["bank"]:
        name = term["name"]
        trace = tuple(Request(r["id"], r["time_min"], tuple(r["origin"]), tuple(r["destination"]))
                      for r in raw[name])
        if demand_sha(trace) != term["sha256"]:
            raise ValueError(f"评估需求哈希不符：{name}")
        bank.append((name, term["role"], trace))
    if Counter(role for _, role, _ in bank) != {"validation": 4, "final": 4}:
        raise ValueError("评估库不是四条验证与四条最终需求")
    return cfg, rows, bank, protocol


def physical_test(gym, np, cfg, rows):
    make = base.build_gym(gym, np)
    ref, candidate = (make(cfg, rows, "MAPPO") for _ in range(2))
    candidate.env = EventPhysics(cfg, policy_returns=True)
    ref.reset(seed=123)
    candidate.reset(seed=123)
    def fingerprint(e):
        return (
            e.now, e.done, e.area_ticks, e.finished, e.passengers, e.planes, e.queue,
            e.chargeq, e.turnq, e.local, e.pa, e.charge_active, e.turn_active, e.totals,
            [(t, seq, fn.__name__, args) for t, seq, fn, args in e._events],
            {s: vars(cal) for s, cal in e.pads.items()},
        )
    count = dispatched = 0
    while not ref.env.done:
        ap, aa = count % 3, (count//3) % 3
        for env in (ref, candidate):
            env.step(ap)
        e = candidate.env
        dest = base.STATIONS[aa]
        n = e.area_ticks
        plan = e.choose_returns(dest, candidate.stat)
        dispatched += len(plan["dispatch"])
        e._drain()
        pieces = [] if e.done else e.advance_to(min(e.now+e._ticks(cfg.decision_min), e.cutoff))
        candidate.phase = 0
        _, r, _, _, _ = ref.step(aa)
        integral = sum((b-a)*q for a,b,q in pieces)
        if (fingerprint(e) != fingerprint(ref.env) or ref.stat != candidate.stat
                or integral != e.area_ticks-n or not math.isclose(r, -integral*cfg.tick_min)):
            fields = [i for i, (x,y) in enumerate(zip(fingerprint(e),fingerprint(ref.env))) if x!=y]
            raise AssertionError(f"同动作同分钟时钟的物理或奖励不一致：第 {count} 步，"
                                 f"字段={fields}，奖励={r}/{-integral*cfg.tick_min}，"
                                 f"控制统计一致={ref.stat==candidate.stat}")
        count += 1
    if not dispatched or ref.env.metrics() != candidate.env.metrics():
        raise AssertionError("没有覆盖实际调机或回合指标不同")
    # 专门覆盖同 tick 多乘客：用正式压力回放的 1.4× 协议，训练 trace 本身不改。
    t0 = min(r.time_min for r in rows)
    dense = tuple(validate_requests([replace(r,time_min=t0+(r.time_min-t0)/1.4) for r in rows]))
    lane = Lane(gym, np, cfg, dense)
    last = None
    audit = Counter()
    times = [[], []]
    while True:
        q = lane.next()
        if q["kind"] == "terminal":
            lane.env.metrics()
            break
        if q["kind"] != "action":
            continue
        tick, role = q["tick"], q["role"]
        if last and tick == last[0]:
            audit["same_tick_ordered"] += 1
            if role < last[1]:
                raise AssertionError("同 tick 航空抢在乘客之前")
            audit["passenger_then_passenger" if role==0 else "passenger_then_aircraft"] += 1
        audit["fractional_tick"] += tick % lane.minute != 0
        times[role].append(tick)
        last = tick, role
        if not any(q["mask"]):
            raise AssertionError("无合法动作却调用 Actor")
        lane.act(next(i for i, yes in enumerate(q["mask"]) if yes))
    hist = [{str(k):v for k,v in Counter(b-a for a,b in zip(seq,seq[1:])).items()} for seq in times]
    if not audit["same_tick_ordered"] or not audit["fractional_tick"] or not all(times) or hist[0] == hist[1]:
        raise AssertionError(f"测试需求没有覆盖真实异步事件：{dict(audit)}，角色次数={[len(x) for x in times]}")
    return dict(matched_minutes=count, checked_aircraft_orders=dispatched, role_gap_ticks=hist, **audit)


def cutoff_fixture(gym,np,cfg):
    # 合成三乘客同 tick；V0 无本地飞机，零接驳后必须入队并触发 V2 批量空返。
    # 只用于验收，正式物理配置和需求不取此值。
    src, dst = cfg.stations["V0"].xy, cfg.stations["V2"].xy
    rows = tuple(Request(i,.2,src,dst) for i in ("C","A","B"))
    return Lane(gym,np,replace(cfg,initial_fleet={"V2":cfg.fleet},initial_soc=1.,
                               parking=False,cutoff_min=.7),rows)


def event_boundary_test(gym,np,cfg):
    lane=cutoff_fixture(gym,np,cfg)
    packets=[]
    for j,pid in enumerate(("A","B","C"),1):
        item=lane.next()
        assert (item["tick"],item["role"],lane.pending[1])==(2,0,pid)
        packets.append(item)
        lane.act(0)
        assert list(lane.env.queue["V0"])==list(("A","B","C")[:j])
        assert lane.env.area_ticks==0
    item=lane.next()
    assert (item["tick"],item["role"],item["mask"])==(2,1,(True,False,False))
    effect=lane.act(0)
    assert effect["applied"]=="V0" and len(effect["dispatch"])==3
    assert all(lane.env.planes[r["aircraft"]].state in ("reserved","airborne") and
               r["status"]=="reserved" and r["destination"]=="V0" for r in effect["dispatch"])
    tail=lane.next()
    assert tail["kind"]=="terminal" and not tail["complete"] and tail["tick"]==7
    assert sum((b-a)*n for a,b,n in tail["pieces"])==lane.env.area_ticks==15
    assert abs(lane.env.metrics()["fixed_cutoff_cost_min"]-.5)<1e-9
    assert tuple(p["event_id"] for p in packets)+(item["event_id"],)==(1,2,3,4)
    return dict(passenger_order=["A","B","C"],same_tick_dispatches=3,cutoff_tick=7,person_ticks=15)


def numeric_test():
    # gamma=.5，-2 人时连续 2 tick；左端折扣后的总额为 -3。
    assert math.isclose(discount_rewards([(0,2,2)], 0, 1., .5), -3.)
    # 同 tick 操作、跨 2 分钟、真实终止；最后一项 bootstrap 为零。
    assert gae([1,2,1], [0,1,2], [2,1,0], [1,.25,.5], [True,True,False],
               [0.,-3.,-2.], .5) == [-1.5625,-5.125,-3.]
    assert gae([4], [0], [8], [.5], [False], [-3], .5) == [-3]
    r = [dict(tick=k, kind=kind, pieces=p, target_value=v, complete=end) for k,kind,p,v,end in
         [(0,"start",[],0,False),(0,"action",[],0,False),(1,"minute",[(0,1,2)],0,False),
          (1,"action",[],0,False),(2,"terminal",[(1,2,4)],0,True)]]
    for macro in (True, False):
        assert [x[2] for x in credit(r, 1., .5, .5, macro)] == [-3.,-4.]
    r.insert(2, dict(tick=0, kind="action", pieces=[], target_value=0, complete=False))
    full = [x[2] for x in credit(r, 1., .5, .5, True)]
    micro = [x[2] for x in credit(r, 1., .5, .5, False)]
    assert full == [-1.5,-3.,-4.] and micro == [-3.,-3.,-4.]
    # tick=.5 分钟，前两个动作同 tick，分钟 probe 跨过后真实截断；手算值包含 cutoff bootstrap。
    r=[dict(tick=k,kind=kind,pieces=p,target_value=v,complete=False) for k,kind,p,v in
       [(0,"start",[],0),(1,"action",[(0,1,2)],2),(1,"action",[],3),
        (2,"minute",[(1,2,4)],4),(2,"action",[],5),(3,"terminal",[(2,3,6)],8)]]
    assert [x[2] for x in credit(r,.5,.25,.5,True)]==[-.75,-3.5,-4.]
    assert [x[2] for x in credit(r,.5,.25,.5,False)]==[-3.625,-4.625,-6.]


def probe_test(coll, gym, np, torch, cfg):
    original = coll.lanes[0]
    coll.lanes[0] = lane = cutoff_fixture(gym,np,cfg)
    def snap(kind):
        return coll.probe([(0,dict(kind=kind,tick=lane.env.now,pieces=[]))])[0]
    try:
        hidden=[h.clone() for h in (coll.ah,coll.ch,coll.th)]
        first=snap("start")
        pending=lane.next()
        assert pending["tick"]==2 and pending["role"]==0
        before=lane.snapshot()
        current=snap("minute")
        reference=torch.as_tensor([lane.critic_context()],device=coll.device)
        with torch.no_grad():
            value=coll.target.value.raw(coll.target(coll.th[:1],reference)).item()
        assert math.isclose(current["target_value"],value,abs_tol=1e-6)
        assert current["target_value"]!=first["target_value"]
        assert lane.env.now==2 and before==lane.snapshot()
        assert all(torch.equal(a,b) for a,b in zip(hidden,(coll.ah,coll.ch,coll.th),strict=True))
        pieces=[]
        while True:
            lane.act(next(i for i,ok in enumerate(pending["mask"]) if ok))
            pending=lane.next()
            pieces.extend(pending["pieces"])
            if pending["kind"]=="terminal":
                break
        end=coll.probe([(0,pending)])[0]
        start=dict(current,kind="action",pieces=[])
        amount=discount_rewards(pieces,start["tick"],cfg.tick_min,.99)
        target=credit([dict(start,kind="start"),start,end],cfg.tick_min,.99,.95,True)[0][1]
        assert math.isclose(target,amount+.99**.5*end["target_value"],abs_tol=1e-8)
        return dict(minute_value_changed=True, terminal_bootstrap=end["target_value"],terminal_reward=amount)
    finally:
        coll.lanes[0]=original


def seed_models(torch, cfg, gym, np, rows, method, device, args, lanes):
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    te, attention, _ = METHODS[method]
    shape = base.build_gym(gym, np)(cfg, rows, "MAPPO").observation_space.shape
    Actor, Critic = model_classes(torch)
    actors = [Actor(shape[0], te).to(device) for _ in range(2)]
    critic = Critic(shape[0], te, attention).to(device)
    target = copy.deepcopy(critic).eval().requires_grad_(False)
    for net in [*actors, critic, target]:
        for sub in net.modules():
            if isinstance(sub, torch.nn.GRU):
                sub.flatten_parameters()
    optics = [torch.optim.Adam(net.parameters(), lr=args["lr" if i<2 else "critic_lr"],
                              eps=args["opti_eps"], weight_decay=args["weight_decay"])
              for i, net in enumerate([*actors, critic])]
    norm = RunningMeanStd(shape=shape)
    return Collector(gym,np,torch,cfg,rows,lanes,actors,critic,target,norm), optics


def state_of(coll, opts, np, torch, real, job, audit):
    return dict(real_steps=real, job=job, audit=dict(audit),
                actors=[a.state_dict() for a in coll.actors], critic=coll.critic.state_dict(),
                target=coll.target.state_dict(), optimizers=[o.state_dict() for o in opts],
                # 环境对象单独封装；只读权重做回放时无需反序列化训练进程的物理对象。
                rng=base.rng_snapshot(np, torch), rollout_blob=pickle.dumps(coll.snapshot(),protocol=5))


def restore(coll, opts, state, norm, np, torch):
    for n, w in zip([*coll.actors, coll.critic, coll.target], [*state["actors"],state["critic"],state["target"]], strict=True):
        n.load_state_dict(w)
    for opt, old in zip(opts, state["optimizers"], strict=True):
        opt.load_state_dict(old)
    coll.norm = norm
    coll.restore(pickle.loads(state["rollout_blob"]))
    # 原帮助函数按字符串 cuda 分流；这里必须传设备类型而不是 cuda:0。
    base.rng_restore(state["rng"], coll.device.type, np, torch)


def save_norm(norm):
    class Writer:
        def save(self, path):
            with open(path, "wb") as file:
                pickle.dump(norm, file, protocol=5)
    return Writer()


def validate_acceptance(cli, manifest, gym, np, torch, cfg, rows, args):
    started = time.monotonic()
    numeric_test()
    physics = physical_test(gym, np, cfg, rows)
    physics.update(event_boundary_test(gym,np,cfg))
    folder = cli.output / "acceptance_v2"
    folder.mkdir(parents=True, exist_ok=True)
    reports = {}
    for method in METHODS:
        base.set_seed(2101,np,torch)
        coll, opts = seed_models(torch,cfg,gym,np,rows,method,cli.device,args,LANES)
        probe_audit=probe_test(coll,gym,np,torch,cfg)
        t0 = time.monotonic()
        chunks, counts, sample = coll.collect(LENGTH, args["gamma"], args["gae_lambda"], METHODS[method][2])
        steps=LANES*LENGTH
        if counts["legacy_equivalent_steps"]!=steps:
            raise AssertionError("验收未跑满正式规格")
        if set(sample) != {0,1} or not counts["noninteger_events"]:
            raise AssertionError("短测没有触发两个角色和非整分钟时钟")
        # 对事件记录检查：当前角色状态变动、另一角色隐藏态未参与其 GRU 输入。
        if any(x["hidden_before"] == x["hidden_after"] for x in sample.values()):
            raise AssertionError("有角色的有效决策没有更新历史")
        # PopArt 更新不能改变同一输入的反归一化预测。
        with torch.no_grad():
            z = coll.ch.detach()
            ctx = torch.as_tensor([l.critic_context() for l in coll.lanes],device=cli.device)
            before = coll.critic.value.raw(coll.critic(z,ctx))
            coll.critic.value.update(torch.tensor([-20.,-10.], device=cli.device))
            torch.testing.assert_close(before, coll.critic.value.raw(coll.critic(z,ctx)), atol=3e-6, rtol=1e-4)
        before = [[p.detach().clone() for p in n.parameters()] for n in [*coll.actors,coll.critic]]
        print(f"{method}：已采满 {steps} 步，开始正式规格 PPO 更新",flush=True)
        stats = learn(torch,np,chunks,coll.actors,coll.critic,opts,args)
        expected=args["actor_num_mini_batch"]*args["ppo_epoch"]
        if stats["minibatches"]!=expected or stats["role_minibatches"]!=[expected,expected]:
            raise AssertionError(f"未覆盖全部双角色 minibatch：{stats}")
        for net, old in zip([*coll.actors,coll.critic],before,strict=True):
            if not any(not torch.equal(p,q) for p,q in zip(net.parameters(),old,strict=True)):
                raise AssertionError("短测中发现未更新网络")
        trial = folder/method
        trial.mkdir(exist_ok=True)
        # 与生产一致：发布新尺度、重建两套 Critic 与 Actor 的隐藏状态，再存恢复点。
        coll.refresh()
        checkpoints = state_of(coll,opts,np,torch,steps,method,counts)
        base.save_generation(trial,steps,steps,(steps,),save_norm(coll.norm),lambda p:torch.save(checkpoints,p),".pt")
        path, npath = base.verified(trial,base.read(trial/"RESUME.json"))
        restored, new_opt = seed_models(torch,cfg,gym,np,rows,method,cli.device,args,LANES)
        with npath.open("rb") as file:
            restore(restored,new_opt,torch.load(path,map_location="cpu",weights_only=False),pickle.load(file),np,torch)
        x = coll.collect(6,args["gamma"],args["gae_lambda"],METHODS[method][2])
        with npath.open("rb") as file:
            restore(restored,new_opt,torch.load(path,map_location="cpu",weights_only=False),pickle.load(file),np,torch)
        y = restored.collect(6,args["gamma"],args["gae_lambda"],METHODS[method][2])
        def actions(chunks):
            return sorted((r["tick"],r["event_id"],r["role"],r["action"],tuple(r["mask"]))
                          for chunk in chunks for r,_,_ in chunk)
        if x[1] != y[1] or actions(x[0]) != actions(y[0]) or x[2].keys()!=y[2].keys():
            raise AssertionError(f"恢复后的物理路径或随机动作不同：{x[1]} / {y[1]}")
        for role in x[2]:
            for name in ("critic_value_before_action","action_log_prob","hidden_before","hidden_after","observation"):
                np.testing.assert_allclose(x[2][role][name],y[2][role][name],rtol=1e-4,atol=2e-5,
                                           err_msg=f"恢复后 {name} 不一致")
        reports[method] = dict(metrics=stats, counts=dict(counts), probe=probe_audit,
                               fps=steps/max(.001,time.monotonic()-t0))
        print(f"验收通过：{method}，步计数 {counts['legacy_equivalent_steps']}", flush=True)
    base.save(cli.output/"ACCEPTANCE.json", dict(manifest=manifest, device=cli.device, physical_test=physics,
                                                lanes=LANES, rollout_per_lane=LENGTH,
                                                four_arms=reports, seconds=time.monotonic()-started))


def train_cell(cli, method, seed, cell, job, gym, np, torch, cfg, rows, args):
    base.set_seed(seed,np,torch)
    coll, opts = seed_models(torch,cfg,gym,np,rows,method,cli.device,args,LANES)
    ptr = cell/"RESUME.json"
    real = 0
    audit = Counter()
    if ptr.exists():
        meta = base.read(ptr)
        if any(Path(meta[k]).name != meta[k] for k in ("model", "norm")):
            raise ValueError("恢复指针路径异常")
        model, normal = base.verified(cell,meta)
        state = torch.load(model,map_location="cpu",weights_only=False)
        real = state["real_steps"]
        if (real != meta["real_steps"] or state["job"] != job or real % 10240
                or not 0 <= real <= STEPS):
            raise ValueError("恢复点不属于当前 cell")
        with normal.open("rb") as file:
            restore(coll,opts,state,pickle.load(file),np,torch)
        audit.update(state["audit"])
    start = real
    began = time.monotonic()
    while real < STEPS:
        if (cli.output/"STOP_REQUESTED").exists() or shutil.disk_usage(cli.output).free < 5*2**30:
            raise InterruptedError("暂停或磁盘小于 5 GiB；保留上一更新边界")
        for i, opt in enumerate(opts):
            opt.param_groups[0]["lr"] = args["lr" if i<2 else "critic_lr"]*(1-real/STEPS)
        chunks, counts, sample = coll.collect(LENGTH,args["gamma"],args["gae_lambda"],METHODS[method][2])
        if counts["legacy_equivalent_steps"] != 10240:
            raise RuntimeError("旧计数合同不符")
        measures = learn(torch,np,chunks,coll.actors,coll.critic,opts,args)
        real += 10240
        # 官方硬更新按回合计数；这里以固定 rollout 为更新时钟，四臂一致。
        audit.update(counts)
        episodes=audit["terminated"]+audit["truncated"]
        elapsed_episodes=episodes-audit["episodes_at_target_sync"]
        did_sync=(real//10240)%4==0
        if did_sync:
            coll.target.load_state_dict(coll.critic.state_dict())
            audit["episodes_at_target_sync"]=episodes
            audit["target_syncs"]+=1
        coll.refresh()
        state = state_of(coll,opts,np,torch,real,job,audit)
        base.save_generation(cell,real,STEPS,MARKS,save_norm(coll.norm),lambda p:torch.save(state,p),".pt")
        elapsed = time.monotonic()-began
        live = dict(method=method, seed=seed, real_steps=real, real_fps=(real-start)/max(.001,elapsed),
                    elapsed=elapsed, simulated_minutes=audit["physical_ticks"]*cfg.tick_min,
                    learning_rate=opts[0].param_groups[0]["lr"], counters=dict(audit),
                    target_synced=did_sync, episodes_since_previous_target_sync=elapsed_episodes, **measures)
        base.save(cell/"LIVE.json",live)
        base.save(cell/"updates"/f"{real:09d}.json",dict(live, sampled_event_records=sample))
    return coll


def replay(cli, cell, job, mark, coll, bank, np, torch):
    rows_out = []
    pointer = base.read(cell/"checkpoints"/f"{mark:09d}.json")
    if pointer["real_steps"] != mark or any(Path(pointer[k]).name != pointer[k] for k in ("model", "norm")):
        raise ValueError("评估指针与里程碑不一致")
    model, normal = base.verified(cell,pointer)
    state = torch.load(model,map_location="cpu",weights_only=False)
    if state["job"] != job or state["real_steps"] != mark:
        raise ValueError("checkpoint 与回放合同不符")
    for a,w in zip(coll.actors,state["actors"],strict=True):
        a.load_state_dict(w)
    coll.critic.load_state_dict(state["critic"])
    coll.target.load_state_dict(state["target"])
    with normal.open("rb") as file:
        coll.norm = pickle.load(file)
    lane = coll.lanes[0]
    rates = (1.,) if mark != STEPS else RATES
    for rate in rates:
        for j,(name,role,requests) in enumerate(bank):
            if role=="final" and mark!=STEPS or role=="validation" and rate!=1.:
                continue
            t0 = min(r.time_min for r in requests)
            dense = tuple(validate_requests([replace(r,time_min=t0+(r.time_min-t0)/rate) for r in requests]))
            for seed in ((123,124,125) if role=="final" else (123,124)):
                if (cli.output/"STOP_REQUESTED").exists():
                    raise InterruptedError("评估已暂停")
                file = cell/"evaluation"/f"{mark}_{rate:.1f}_{j}_{seed}.json"
                if file.exists():
                    row = base.read(file)
                    if row["weight_sha"]!=pointer["model_sha"]:
                        raise ValueError("评估缓存来自另一存档")
                else:
                    base.set_seed(seed,np,torch)
                    lane.view.rows = dense
                    lane.reset()
                    coll.ah.zero_()
                    coll.ch.zero_()
                    coll.th.zero_()
                    while True:
                        item=lane.next()
                        if item["kind"]=="terminal":
                            break
                        if item["kind"]=="action":
                            coll.decision([(0,item)],False)
                            lane.act(item["action"])
                    metrics=lane.env.metrics()
                    if metrics["all_requests_finished"]:
                        att=[sum(p.phases.values())*lane.env.cfg.tick_min for p in lane.env.passengers.values()]
                        for q in (90,95):
                            metrics[f"p{q}_att_min"]=float(np.percentile(att,q))
                    else:
                        metrics.update(p90_att_min=None,p95_att_min=None)
                    metrics["awt_min"] = (sum(metrics["phase_min_per_generated"][k] for k in ("queue","pad_wait"))
                                          if metrics["all_requests_finished"] else None)
                    row=dict(step=mark,rate=rate,trace=name,role=role,policy_seed=seed,metrics=metrics,
                             control=dict(lane.view.stat),weight_sha=pointer["model_sha"])
                    base.save(file,row)
                rows_out.append(row)
    expected = 8 if mark != STEPS else 8+5*12
    if len(rows_out)!=expected:
        raise RuntimeError(f"回放格不全：{len(rows_out)}/{expected}")
    return rows_out


def summarize(matrix):
    scores={}
    for method, seeds in matrix.items():
        scores[method]={}
        for rate in RATES:
            per_seed={}
            for seed, rows in seeds.items():
                stats=[r["metrics"] for r in rows if r["role"]=="final" and r["rate"]==rate]
                names=("mean_att_min","completion","fixed_cutoff_cost_min","awt_min","p90_att_min","p95_att_min")
                per_seed[str(seed)]={k:(sum(s[k] for s in stats)/len(stats)
                                       if stats and all(s[k] is not None for s in stats) else None) for k in names}
            scores[method][str(rate)]=per_seed
    comparisons={}
    if "FULL" in scores:
        for method,z in scores.items():
            if method=="FULL":
                continue
            comparisons[method]={r:{k:(stats["mean_att_min"]-scores["FULL"][r][k]["mean_att_min"]
                                       if None not in (stats["mean_att_min"],scores["FULL"][r][k]["mean_att_min"]) else None)
                                    for k,stats in seeds.items()} for r,seeds in z.items()}
    return dict(by_method_rate_seed=scores, paired_att_minus_full=comparisons)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("config","train-trace","base-output","harl-dir","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--phase",choices=("physics-test","acceptance","all"),default="all")
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    cli=p.parse_args(argv)
    cli.output=cli.output.resolve()
    ref=cli.base_output.resolve()
    if ref==cli.output or ref in cli.output.parents or cli.output in ref.parents:
        p.error("派生实验不得写入原基线")
    import gymnasium as gym
    import numpy as np
    if cli.phase=="physics-test":
        numeric_test()
        cfg=base.config(cli.config,"E2")
        audit=physical_test(gym,np,cfg,base.trace(base.config(cli.config,"E1"),cli.train_trace))
        print(json.dumps(dict(audit, **event_boundary_test(gym,np,cfg))))
        return 0
    import torch
    import stable_baselines3 as sb3
    torch.set_num_threads(1)
    if cli.device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA 不可用")
    if cli.phase=="all" and cli.device!="cuda":
        p.error("正式训练要求 CUDA")
    versions=dict(torch=str(torch.__version__),gymnasium=gym.__version__, numpy=np.__version__,sb3=sb3.__version__)
    cfg,rows,bank,reference=bank_for(cli, versions if cli.phase=="all" else {})
    args,_,harl=base.harl_vendor(cli.harl_dir.resolve(),np,torch)
    if (args["n_rollout_threads"],args["episode_length"],args["gamma"],args["gae_lambda"],
            args["actor_num_mini_batch"],args["critic_num_mini_batch"],args["ppo_epoch"])!=(5,2048,.99,.95,20,20,5):
        p.error("训练合同与冻结版不一致")
    manifest=canonical(dict(revision="acac_event_e2_v2", code_sha=base.digest(Path(__file__)),
        upstream=UPSTREAM, reference=reference, config_sha=base.digest(cli.config), trace_sha=base.digest(cli.train_trace),
        harl_commit=base.HARL_COMMIT,harl=harl, args=args,runtime=versions,device=cli.device, methods=METHODS,
        steps_per_cell=STEPS,steps_total=16*STEPS,seeds=SEEDS,rates=RATES,marks=MARKS,evaluation_device="cpu",
        timing="每个角色事件单独排序；整分钟计量点不调用 Actor；终止残段按一次旧 advance 计",
        target="每 4 rollout 硬同步，统计对应实际回合数量；分钟和截断值通过当前状态与最新历史重新计算",
        state_probe="共享 Critic 已揭示物理状态分支；计量点不改变任何 GRU 或观测统计",
        normalization="每 rollout 固定行为尺度；边界发布下一统计量并用 raw 历史重建隐藏态",
        optimizer="独立 Adam、PPO Clip、PopArt 平方误差，沿用走廊 LR/Clip/熵/梯度上限；无作者 epsilon 日程",
        attention="单层非因果 GPT 风格角色交互、mean_linear；无 GPT2 代码依赖与 dropout",
        official=dict(clip_ratio=.01,ppo_epochs=5,gae_lambda=.95,target_update_episodes=32,epsilon_exploration=True),
        time_embedding=dict(dim=4, frequencies_per_min=[1.,.01]),
        scope="UAM 原子决策适配，不是持久宏动作；事件时钟和合法动作遮罩均改变，不能单因子对比旧 ACAC"))
    with base.lock_run(cli.output):
        pf=cli.output/"PROTOCOL.json"
        gate=cli.output/"ACCEPTANCE.json"
        for existing in (pf,gate):
            if existing.exists():
                prior=base.read(existing)
                if (prior.get("manifest", prior).get("revision"))!="acac_event_e2_v2":
                    p.error("v1 或其他实验目录不得混用")
        if pf.exists() and base.read(pf)!=manifest:
            p.error("正式输出已绑定另一合同")
        if cli.phase=="acceptance":
            validate_acceptance(cli,manifest,gym,np,torch,cfg,rows,args)
            return 0
        a=base.read(gate) if gate.exists() else {}
        if (a.get("manifest")!=manifest or set(a.get("four_arms",{}))!=set(METHODS)
                or (a.get("lanes"),a.get("rollout_per_lane"))!=(LANES,LENGTH)):
            p.error("同一源码、CUDA、配置的四臂正式规格验收未通过；先运行 --phase acceptance")
        base.save(pf,manifest)
        base.save(cli.output/"DEMAND_BANK.json",dict(train=[asdict(r) for r in rows],
                  **{name:[asdict(r) for r in group] for name,_,group in bank}))
        matrix={}
        errors=[]
        for method in METHODS:
            method_rows={}
            for seed in SEEDS:
                cell=cli.output/"cells"/f"ACAC_EVENT_{method}__s{seed}"
                job=dict(contract=base.digest(pf),method=method,seed=seed)
                try:
                    if (cli.output/"STOP_REQUESTED").exists():
                        return 130
                    jf=cell/"JOB.json"
                    if jf.exists() and base.read(jf)!=job:
                        raise ValueError("训练格身份不一致")
                    base.save(jf,job)
                    print(f"{cell.name}：开始或继续",flush=True)
                    train_cell(cli,method,seed,cell,job,gym,np,torch,cfg,rows,args)
                    # 与旧五档评估相同：冻结策略在 CPU 上回放，不做优化。
                    eval_coll,_=seed_models(torch,cfg,gym,np,rows,method,"cpu",args,1)
                    data=[row for mark in MARKS for row in replay(cli,cell,job,mark,eval_coll,bank,np,torch)]
                    base.save(cell/"EVALUATION.json",dict(rows=data))
                    base.save(cell/"STATUS.json",dict(job=job,real_steps=STEPS,evaluated_through_final=True))
                    method_rows[seed]=data
                except (InterruptedError,KeyboardInterrupt):
                    return 130
                except Exception:
                    detail=traceback.format_exc()
                    base.save(cell/"ERROR.json",dict(job=job,error=detail))
                    print(detail,flush=True)
                    errors.append(cell.name)
            if len(method_rows)!=4:
                base.save(cli.output/"SUMMARY.json",dict(result=summarize(matrix),failures=errors))
                return 1
            matrix[method]=method_rows
            base.save(cli.output/"SUMMARY.json",dict(result=summarize(matrix),failures=errors,
                                                     completed=sum(map(len,matrix.values()))))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
