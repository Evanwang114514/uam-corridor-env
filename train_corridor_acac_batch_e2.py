"""E2 双线矩阵：原 ACAC Full 的分钟批次排序与步内信用。"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import gzip
import hashlib
import math
from pathlib import Path
import pickle
import random
import shutil
import time
import traceback

import train_corridor_acac_mappo_e2 as old
import train_eval_corridor_uagmc_happo_mappo as base
from corridor_env import validate_requests


STEPS, SEEDS, MARKS, RATES = old.STEPS, old.SEEDS, old.MARKS, old.RATES
ORDER = ("A0_ORIGINAL", "A1_RANDOM", "A2_READY", "A3_STATIC", "A4_DYNAMIC")
CREDIT = ("B0_SHARED", "B1_EVENT_GAE", "B2_PREFIX_Q")
TRAIN = (*ORDER[-2:], *CREDIT)
POLICY_SEEDS = (123, 124, 125)
VALUE_KEYS = old.VALUE_KEYS


def contract(path, value):
    value = old.canonical(value)
    if path.is_file() and base.read(path) != value:
        raise ValueError(f"已有合同不一致：{path}；请使用新输出目录")
    if not path.is_file():
        base.save(path, value)
    return base.digest(path)


def pressure_trace(rows, rate):
    t0 = min(r.time_min for r in rows)
    return validate_requests([replace(r, time_min=t0+(r.time_min-t0)/rate) for r in rows])


def burst_trace(rows, pairs, triples):
    # 移动小组中的后续请求，均匀铺开；保留首尾时间、总需求、ID 和 OD。
    # 必须有三人批次，否则第一次排序后只剩一位，动态重算不可能改变后续顺序。
    rows = validate_requests(list(rows))
    count=pairs+triples
    slack=len(rows)-2-2*pairs-3*triples
    if pairs<0 or triples<1 or slack<0:
        raise ValueError("突发批次数不合法（至少一个三人批次，且不能触及首尾或复用请求）")
    out = list(rows)
    consumed=0
    for q in range(count):
        n=2+int((q+1)*triples//count>q*triples//count)
        i=1+consumed+(q+1)*slack//(count+1)
        if any(rows[j+1].time_min-rows[j].time_min<.99 for j in range(i,i+n-1)):
            raise ValueError("原始需求轨迹不是分钟间隔轨迹")
        for j in range(i+1,i+n):
            out[j]=replace(rows[j],time_min=rows[i].time_min)
        consumed+=n
    if [(r.id, r.origin, r.destination) for r in out] != [(r.id, r.origin, r.destination) for r in rows]:
        raise AssertionError("突发轨迹不允许改变 OD")
    return validate_requests(out)


def uniform(*key):
    # 排序的随机数与原 Actor 的 Categorical 隔离，不改变回放策略随机流。
    bits = int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "little")
    return (bits >> 11) / (1 << 53)


class MinuteBatchLane:
    def __init__(self, view, index):
        self.view, self.index = view, index
        self.episode, self.clock = -1, 0
        self.remaining = []
        self.pending = ()
        self.mid = None
        self.reset()

    @property
    def env(self):
        return self.view.env

    def reset(self):
        self.view.reset(seed=123)
        self.episode += 1
        self.clock = 1
        self.remaining = []
        self.pending = ()
        self.mid = None

    @contextmanager
    def focal(self, pid):
        local = "focal" in self.view.__dict__
        before = self.view.__dict__.get("focal")
        self.view.focal = lambda: self.env.passengers[pid]
        try:
            yield
        finally:
            if local:
                self.view.focal = before
            else:
                del self.view.focal

    def begin_tick(self):
        if self.view.phase != 0 or self.remaining or self.env.done:
            raise RuntimeError("只能从有效分钟边界开始批次")
        self.pending = tuple(p.req.id for p in self.env.passengers.values() if p.state == "route")
        self.remaining = list(self.pending)
        self.now = self.env.now

    def observation(self, pid=None):
        if pid is None:
            return self.view.observation()
        with self.focal(pid):
            return self.view.observation()

    def commit(self, pid, action):
        if pid is None:
            if self.remaining:
                raise AssertionError("有待选站乘客时不允许空决定")
            self.mid, r, end, trunc, info = self.view.step(action)
        else:
            self.remaining.remove(pid)
            passenger = self.env.passengers[pid]
            if passenger.state != "route":
                raise AssertionError("只允许处理已揭示且未选站的乘客")
            if self.remaining:
                # 前 m−1 人只登记接驳预约；不提前调用 _drain、服务或推进时钟。
                station = base.STATIONS[action]
                self.env._route(passenger, station)
                self.view.stat["route_"+station] += 1
                if self.env.now != self.now or self.view.phase != 0:
                    raise AssertionError("额外乘客动作推进了物理时钟")
                return
            with self.focal(pid):
                self.mid, r, end, trunc, info = self.view.step(action)
        if r or end or trunc or info["physical_increment"] != 0 or self.env.now != self.now:
            raise AssertionError("最后一次选站不能推进/计奖")

    def advance(self, action):
        if self.remaining or self.view.phase != 1:
            raise AssertionError("必须处理完当前乘客批次才能调机")
        _, reward, done, truncated, info = self.view.step(action)
        if (info["physical_increment"] != 1 or self.env.now < self.now
                or (self.env.now == self.now and not (done or truncated))):
            raise AssertionError("每分钟只能推进一次外层调度时钟")
        self.clock += 1
        self.mid = None
        return float(reward), bool(done), bool(truncated)

    def priority(self, pid):
        # 用已揭示 OD、当前队列、READY 和已承诺接驳；优先处理备选站差距大的乘客。
        env = self.env
        p = env.passengers[pid]
        costs = []
        for station in base.STATIONS:
            ready = sum(a.state == "ready" and a.station == station for a in env.planes.values())
            inbound = sum(x.state == "access" and x.dep == station for x in env.passengers.values())
            drive = env._drive(p.req.origin, env.cfg.stations[station].xy, env.cfg.access_kmh)*env.cfg.tick_min
            queue = len(env.queue[station])+inbound
            costs.append(drive + queue/(1+ready)*env.cfg.decision_min)
        x, y, _ = sorted(costs)
        return y-x

    def rule_order(self, method, seed):
        if method == "A1_RANDOM":
            return sorted(self.pending, key=lambda pid: uniform(seed, self.index, self.episode, self.now, pid))
        if method == "A2_READY":
            return sorted(self.pending, key=self.priority, reverse=True)
        # 原始插入顺序与到达先后相同；FCFS 与最长等待优先采用相同的并列规则。
        if sorted(self.pending, key=lambda pid: self.env.passengers[pid].req.time_min) != list(self.pending):
            raise AssertionError("当前环境的 FCFS 不等于原始顺序，不能复用对照")
        return list(self.pending)


def model_types(torch):
    nn = torch.nn

    class Order(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(dim, 64), nn.LeakyReLU(),
                                         nn.Linear(64, 64), nn.LeakyReLU())
            self.score = nn.Linear(64, 1)
            self.value = nn.Sequential(nn.Linear(129, 64), nn.LeakyReLU(), nn.Linear(64, 1))
            # 值输出初始为原模型的归一化均值，排序网络正常初始化。
            nn.init.zeros_(self.value[-1].weight)
            nn.init.zeros_(self.value[-1].bias)

        def scores(self, obs):
            return self.score(self.encoder(obs)).squeeze(-1)

        def set_value(self, obs, mask):
            enc = self.encoder(obs)
            n = mask.sum(-1, keepdim=True)
            valid = mask.unsqueeze(-1)
            mean = (enc*valid).sum(-2)/n.clamp_min(1)
            maximum = enc.masked_fill(~valid, -torch.inf).amax(-2)
            maximum = torch.where(n > 0, maximum, 0.)
            return self.value(torch.cat((mean, maximum, n.to(obs.dtype)), -1)).squeeze(-1)

    class PrefixQ(nn.Module):
        def __init__(self, dim, width):
            super().__init__()
            self.prefix = nn.GRU(dim+3, 32, batch_first=True)
            self.head = nn.Sequential(nn.Linear(2*width+dim+32+1, 64), nn.LeakyReLU(), nn.Linear(64, 1))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

        def forward(self, anchor, obs, actions):
            onehot = torch.nn.functional.one_hot(actions, 3).to(obs.dtype)
            hidden = self.prefix(torch.cat((obs, onehot), -1))[0]
            lengths = torch.arange(1, obs.shape[1]+1, device=obs.device, dtype=obs.dtype)
            lengths = lengths[None, :, None].expand(obs.shape[0], -1, -1)
            return self.head(torch.cat((anchor[:, None].expand(-1, obs.shape[1], -1), hidden, lengths), -1)).squeeze(-1)

    return Order, PrefixQ


@dataclass
class Event:
    role: int
    pid: str | None
    q: int
    obs: object
    ah_before: object
    ah_after: object
    ch_before: object
    action: int
    logp: float
    value: float
    adv: float = 0.


@dataclass
class Minute:
    lane: int
    clock: int
    physical_tick: int
    ah: object
    ch: object
    passengers: list
    aircraft: Event | None = None
    reward: float = 0.
    terminal: bool = False
    trunc: bool = False
    value: float = 0.
    adv: float = 0.
    target: float = 0.
    keep: bool = False
    anchor: object = None
    order_choices: list | None = None
    order_value: float = 0.
    order_adv: float = 0.

    def events(self):
        return self.passengers + ([self.aircraft] if self.aircraft is not None else [])


@dataclass
class Choice:
    ids: tuple
    obs: object
    picked: int
    logp: float


def choose_order(torch, model, lane, t, obs_for, method, rng_seed, static):
    ids = tuple(lane.remaining)
    if len(ids) == 1:
        # 最后一个条件选择概率恒为 1，仍执行其动作但不重复加损失。
        return ids[0], None
    if method == "A3_STATIC":
        full_obs, full_scores = static
        ix = [lane.pending.index(pid) for pid in ids]
        obs, logits = full_obs[ix], full_scores[ix]
    else:
        obs = obs_for(ids)
        logits = model.scores(t(obs))
    logp = logits.log_softmax(dim=-1)
    cum = logp.exp().cumsum(0).cpu().numpy()
    u = uniform("order", rng_seed, lane.index, lane.episode, lane.now, len(lane.pending)-len(ids))
    picked = min(len(ids)-1, int(cum.searchsorted(u, side="right")))
    return ids[picked], Choice(ids, obs, picked, float(logp[picked]))


class Collector:
    def __init__(self, env_type, cfg, rows, actors, critic, norm, torch, np, device, method, seed):
        self.lanes = [MinuteBatchLane(env_type(cfg, rows, "MAPPO"), i) for i in range(base.LANES)]
        self.actors, self.critic, self.norm = actors, critic, norm
        self.torch, self.np, self.device, self.method, self.seed = torch, np, device, method, seed
        self.ah = torch.zeros((base.LANES, 2, 64), device=device)
        self.ch = torch.zeros_like(self.ah)

    def t(self, data):
        return self.torch.as_tensor(data, device=self.device)

    def observations(self, lane, ids):
        return self.norm.normalize_obs(self.np.stack([lane.observation(pid) for pid in ids]))

    def tick_values(self, lanes):
        np, torch = self.np, self.torch
        ix = [ln.index for ln in lanes]
        raw = self.norm.normalize_obs(np.stack([ln.observation() for ln in lanes]))
        clocks = self.t(np.asarray([ln.clock for ln in lanes], np.float32))
        legal = self.t(np.asarray([ln.view.focal() is not None for ln in lanes], bool))
        p = self.critic.history[0].one(self.t(raw), clocks, legal, self.ch[ix, 0])
        a = self.ch[ix, 1]
        return self.critic(p, a)[0].cpu().numpy(), torch.cat((self.t(raw), p, a), -1).cpu().numpy()

    def draw(self, role, offers):
        if not offers:
            return []
        np, torch = self.np, self.torch
        ix = [ln.index for ln, _, _ in offers]
        obs = self.norm.normalize_obs(np.stack([x for _, _, x in offers]))
        x = self.t(obs)
        clock = self.t(np.asarray([ln.clock for ln, _, _ in offers], np.float32))
        ah, ch = self.ah[ix, role], self.ch[ix, role]
        true = torch.ones(len(offers), device=self.device, dtype=torch.bool)
        ah_out = self.actors[role].history.one(x, clock, true, ah)
        ch_out = self.critic.history[role].one(x, clock, true, ch)
        dist = torch.distributions.Categorical(logits=self.actors[role].head(ah_out))
        action = dist.sample()
        self.ah[ix, role], self.ch[ix, role] = ah_out, ch_out
        value = self.critic(self.ch[ix, 0], self.ch[ix, 1])[0] if self.method == "B1_EVENT_GAE" else clock*0
        arrays = (obs, ah.cpu().numpy(), ah_out.cpu().numpy(), ch.cpu().numpy(),
                  action.cpu().numpy(), dist.log_prob(action).cpu().numpy(), value.cpu().numpy())
        return [Event(role, pid, len(ln.pending)-len(ln.remaining) if role == 0 else -1,
                      o, h, h2, c, int(act), float(lp), float(v))
                for (ln, pid, _), (o, h, h2, c, act, lp, v) in zip(offers, zip(*arrays), strict=True)]

    def collect(self, length, order_model):
        np = self.np
        ticks = [[] for _ in self.lanes]
        for _ in range(length):
            for ln in self.lanes:
                ln.begin_tick()
            v, anchors = self.tick_values(self.lanes)
            ah, ch = self.ah.cpu().numpy().copy(), self.ch.cpu().numpy().copy()
            record = [Minute(ln.index, ln.clock, ln.now, ah[i], ch[i], [], value=float(v[i]), anchor=anchors[i])
                      for i, ln in enumerate(self.lanes)]
            initial = {}
            for ln, mt in zip(self.lanes, record, strict=True):
                if order_model is not None and len(ln.pending) > 1:
                    arr = self.observations(ln, ln.pending)
                    initial[ln.index] = (arr, order_model.scores(self.t(arr))) if self.method == "A3_STATIC" else None
                    mt.order_choices = []
                    mt.order_value = float(order_model.set_value(self.t(arr)[None],
                                           self.t(np.ones((1, len(arr)), bool)))[0])
                if not ln.pending:
                    ln.commit(None, 0)
            while True:
                offers = []
                for ln, mt in zip(self.lanes, record, strict=True):
                    if not ln.remaining:
                        continue
                    if ln.index in initial:
                        pid, choice = choose_order(self.torch, order_model, ln, self.t,
                            lambda ids, lane=ln: self.observations(lane, ids), self.method, self.seed, initial[ln.index])
                        if choice is not None:
                            mt.order_choices.append(choice)
                    else:
                        pid = ln.remaining[0]
                    offers.append((ln, pid, ln.observation(pid)))
                if not offers:
                    break
                for (ln, pid, _), ev in zip(offers, self.draw(0, offers), strict=True):
                    record[ln.index].passengers.append(ev)
                    ln.commit(pid, ev.action)
            offers = [(ln, None, ln.mid) for ln in self.lanes if ln.view.eligible()]
            for (ln, _, _), ev in zip(offers, self.draw(1, offers), strict=True):
                record[ln.index].aircraft = ev
            for ln, mt in zip(self.lanes, record, strict=True):
                air = mt.aircraft
                mt.reward, mt.terminal, mt.trunc = ln.advance(air.action if air is not None else 0)
                ticks[ln.index].append(mt)
                if mt.terminal or mt.trunc:
                    ln.reset()
                    self.ah[ln.index] = 0
                    self.ch[ln.index] = 0
        return ticks, self.tick_values(self.lanes)[0]


def tick_advantages(np, ticks, bootstrap, vn, args):
    length, lanes = len(ticks[0]), len(ticks)
    reward = np.empty((length, lanes), np.float32)
    value = np.empty((length+1, lanes), np.float32)
    done, trunc, legal = [np.empty((length, lanes), bool) for _ in range(3)]
    for i, lane in enumerate(ticks):
        for j, mt in enumerate(lane):
            reward[j, i], value[j, i] = mt.reward, mt.value
            done[j, i], trunc[j, i] = mt.terminal or mt.trunc, mt.trunc
            legal[j, i] = bool(mt.passengers or mt.aircraft is not None)
    value[-1] = bootstrap
    raw = vn.denormalize(value.reshape(-1, 1)).reshape(value.shape)
    adv, keep = old.advantages(np, reward, raw, done, trunc, legal,
                                args["gamma"], args["gae_lambda"], True)
    for i, lane in enumerate(ticks):
        for j, mt in enumerate(lane):
            mt.adv, mt.target, mt.keep = float(adv[j, i]), float(adv[j, i]+raw[j, i]), bool(keep[j, i])
            for ev in mt.passengers:
                ev.adv = mt.adv
    return raw


def event_gae(np, ticks, raw, vn, args):
    """分钟调度保持原样；仅让 lambda 在每个合法原子动作之间传播。"""
    gamma, lam = args["gamma"], args["gae_lambda"]
    for i, lane in enumerate(ticks):
        start = 0
        for stop in [j+1 for j,m in enumerate(lane) if m.terminal or m.trunc] + [len(lane)]:
            if stop <= start:
                continue
            seg = [(j, e) for j in range(start,stop) for e in lane[j].events()]
            vs = vn.denormalize(np.asarray([e.value for _,e in seg], np.float32).reshape(-1,1)).ravel() if seg else []
            right = stop-1 if lane[stop-1].trunc else stop
            next_v = 0. if lane[stop-1].terminal else float(raw[right,i])
            carry = 0.
            for (j,e), v in zip(reversed(seg), reversed(vs), strict=True):
                if right < j:
                    raise AssertionError("事件时钟逆序")
                reward = sum(gamma**(k-j)*lane[k].reward for k in range(j,right))
                discount = gamma**(right-j)
                carry = reward + discount*(next_v+lam*carry) - float(v)
                if e.role == 0:
                    e.adv = carry
                next_v, right = float(v), j
            start = stop


def batches(np, n, epochs, count, rng=None):
    rng=np.random if rng is None else rng
    for _ in range(epochs):
        for ids in np.array_split(rng.permutation(n), min(n, count)):
            if len(ids):
                yield ids


class Metrics:
    def __init__(self):
        self.values = defaultdict(list)

    def add(self, **kw):
        for k, v in kw.items():
            v = float(v)
            if not math.isfinite(v):
                raise FloatingPointError(f"非有限更新诊断：{k}")
            self.values[k].append(v)

    def report(self):
        return {k: sum(v)/len(v) for k,v in self.values.items()}


def ev(torch, predicted, target):
    return 1.-(predicted-target).var(unbiased=False)/(target.var(unbiased=False)+1e-8)


def update_policy(torch, net, opt, logratio, adv, entropy, args, metrics, prefix):
    ratio = logratio.exp()
    clip = args["clip_param"]
    loss = -torch.minimum(ratio*adv, ratio.clamp(1-clip,1+clip)*adv).mean()
    old.optimize(torch, opt, net, loss-args["entropy_coef"]*entropy.mean(), args)
    metrics.add(**{prefix+"_loss":loss.item(),prefix+"_entropy":entropy.mean().item(),
                   prefix+"_kl":(ratio-1-logratio).mean().item(),
                   prefix+"_clip_fraction":((ratio-1.).abs()>clip).float().mean().item()})


def chunks_for(ticks, block):
    result = []
    for lane in ticks:
        chunk = []
        for m in lane:
            chunk.append(m)
            if len(chunk) == block or m.terminal or m.trunc:
                result.append(chunk)
                chunk = []
        if chunk:
            result.append(chunk)
    return result


def pack_events(np, chunks, dim):
    # 第一列是每个分钟的预动作 V 读出点；无乘客时插入不更新 GRU 的占位。
    span = [sum(max(1,len(m.passengers)) + int(m.aircraft is not None) for m in chunk) for chunk in chunks]
    shape = (len(chunks), max(span))
    obs = np.zeros((*shape,dim),np.float32)
    clock, lp, adv, value, target = (np.zeros(shape,np.float32) for _ in range(5))
    action = np.zeros(shape,np.int64)
    role = np.full(shape,-1,np.int8)
    vmask = np.zeros(shape,bool)
    for i, chunk in enumerate(chunks):
        j = 0
        for m in chunk:
            vmask[i,j], value[i,j], target[i,j] = m.keep, m.value, m.target
            for e in (m.passengers or [None])+([m.aircraft] if m.aircraft is not None else []):
                if e is not None:
                    role[i,j],obs[i,j],action[i,j],lp[i,j],adv[i,j] = e.role,e.obs,e.action,e.logp,e.adv
                clock[i,j] = m.clock
                j += 1
    return dict(obs=obs,clock=clock,roles=role,lp=lp,action=action,adv=adv,vmask=vmask,value=value,
                target=target,ah=np.stack([c[0].ah for c in chunks]),ch=np.stack([c[0].ch for c in chunks]))


def rebuild(torch, np, t, history, data, role, k):
    mask = data["roles"] == role
    return history.sequence(t(data["obs"]), t(data["clock"]), t(mask), t(data[k][:,role]),
                            t(np.argwhere(mask)), torch.as_tensor(mask.sum(1).clip(1)))


def credit_update(torch, np, t, ticks, actor, critic, actor_opt, critic_opt, vn, vc, args, audit):
    chunks = chunks_for(ticks,int(args["data_chunk_length"]))
    dim = len(ticks[0][0].anchor)-128
    all_adv = np.asarray([ev.adv for lane in ticks for m in lane for ev in m.passengers],np.float32)
    if not len(all_adv) or not np.isfinite(all_adv).all():
        raise RuntimeError("乘客原子样本为空或信用不是有限数")
    center, scale = all_adv.mean(), all_adv.std()+1e-5
    metrics = Metrics()
    audit_value = audit
    for ids in batches(np,len(chunks),args["ppo_epoch"],args["actor_num_mini_batch"]):
        d = pack_events(np,[chunks[i] for i in ids],dim)
        hit = d["roles"]==0
        if not hit.any():
            continue
        h = rebuild(torch,np,t,actor.history,d,0,"ah")
        dist = torch.distributions.Categorical(logits=actor.head(h[t(hit)]))
        logratio = dist.log_prob(t(d["action"][hit]))-t(d["lp"][hit])
        if audit:
            error = logratio.abs().max().item()
            if error > 3e-4:
                raise AssertionError(f"原子动作的旧 logp 与重建 GRU 不匹配：{error}")
            metrics.add(behavior_logp_max_error=error)
            audit = False
        update_policy(torch,actor,actor_opt,logratio,t((d["adv"][hit]-center)/scale),
                      dist.entropy(),args,metrics,"passenger")
    for ids in batches(np,len(chunks),args["critic_epoch"],args["critic_num_mini_batch"]):
        d = pack_events(np,[chunks[i] for i in ids],dim)
        hit = d["vmask"]
        if not hit.any():
            continue
        p,a = (rebuild(torch,np,t,critic.history[i],d,i,"ch") for i in range(2))
        prediction = critic(p[t(hit)],a[t(hit)])[0]
        # 在每分钟第一条记录读 V；此时本分钟的 Aircraft 事件尚未进入历史。
        if audit_value:
            error = (prediction - t(d["value"][hit])).abs().max().item()
            if error > 3e-4:
                raise AssertionError(f"重建的分钟预动作 V 与采样值不一致：{error}")
            metrics.add(behavior_value_max_error=error)
            audit_value = False
        y = t(d["target"][hit])[:,None]
        loss = vc.cal_value_loss(prediction[:,None],t(d["value"][hit])[:,None],y,vn)
        old.optimize(torch,critic_opt,critic,args["value_loss_coef"]*loss,args)
        with torch.no_grad():
            metrics.add(value_loss=loss.item(),explained_variance=ev(torch,prediction,vn.normalize(y).flatten()).item())
    return metrics.report()


def prefix_arrays(np, picked):
    n, span, dim = len(picked),max(len(m.passengers) for m in picked),len(picked[0].passengers[0].obs)
    obs = np.zeros((n,span,dim),np.float32)
    action = np.zeros((n,span),np.int64)
    mask = np.zeros((n,span),bool)
    for i,m in enumerate(picked):
        for j,e in enumerate(m.passengers):
            obs[i,j],action[i,j],mask[i,j]=e.obs,e.action,True
    return np.stack([m.anchor for m in picked]),obs,action,mask


def prefix_credit(torch,np,t,net,opt,ticks,args,value_scale,rng):
    # 残差形式 Q_q = V_t + s*f(prefix)，尺度固定为来源模型的 ValueNorm 标准差。
    todo = [m for lane in ticks for m in lane if m.passengers]
    metrics = Metrics()
    # 使用采样时已有的 Q 估计信用；先全部冻结优势、再拟合本 rollout 的回报。
    with torch.no_grad():
        for k in range(0,len(todo),256):
            group=todo[k:k+256]
            anchor,obs,action,_=prefix_arrays(np,group)
            q=net(t(anchor),t(obs),t(action)).cpu().numpy()*value_scale
            for i,m in enumerate(group):
                if len(m.passengers) == 1:
                    # 单乘客时没有步内信用歧义，保持 B0 原优势。
                    continue
                v=q[i,:len(m.passengers)]
                delta=np.diff(np.r_[0.,v])
                metrics.add(prefix_telescope_error=abs(float(delta.sum()-v[-1])),
                            prefix_abs_credit=float(np.abs(delta).mean()))
                for e,adv in zip(m.passengers,delta,strict=True):
                    e.adv=float(adv)
    for ids in batches(np,len(todo),args["critic_epoch"],args["critic_num_mini_batch"],rng):
        group=[todo[i] for i in ids]
        anchor,obs,action,mask=prefix_arrays(np,group)
        raw=net(t(anchor),t(obs),t(action))
        target=t(np.asarray([m.adv/value_scale for m in group],np.float32))[:,None].expand_as(raw)
        loss=torch.nn.functional.smooth_l1_loss(raw[t(mask)],target[t(mask)])
        old.optimize(torch,opt,net,args["value_loss_coef"]*loss,args)
        metrics.add(prefix_value_loss=loss.item(),prefix_explained_variance=ev(torch,raw[t(mask)],target[t(mask)]).item())
    return metrics.report()


def pad_sets(np, options):
    b,n,dim=len(options),max(len(x) for x in options),len(options[0][0])
    obs=np.zeros((b,n,dim),np.float32)
    mask=np.zeros((b,n),bool)
    for i,arr in enumerate(options):
        obs[i,:len(arr)]=arr
        mask[i,:len(arr)]=True
    return obs,mask


def order_update(torch,np,t,net,opt,ticks,vn,vc,args,audit):
    group=[m for lane in ticks for m in lane if m.order_choices]
    if not group:
        raise RuntimeError("零合法多乘客排序样本，拒绝空训练")
    values_norm=np.asarray([m.order_value for m in group],np.float32)
    values=vn.denormalize(values_norm[:,None]).ravel()
    target=np.asarray([m.target for m in group],np.float32)
    advantage=target-values
    advantage=(advantage-advantage.mean())/(advantage.std()+1e-5)
    metrics=Metrics()
    for ids in batches(np,len(group),args["ppo_epoch"],args["actor_num_mini_batch"]):
        batch=[group[i] for i in ids]
        choices=[c for m in batch for c in m.order_choices]
        rows=np.repeat(np.arange(len(batch)),[len(m.order_choices) for m in batch])
        obs,mask=pad_sets(np,[c.obs for c in choices])
        logits=net.scores(t(obs)).masked_fill(~t(mask),-torch.inf)
        d=torch.distributions.Categorical(logits=logits)
        lp=d.log_prob(t(np.asarray([c.picked for c in choices],np.int64)))
        diff=lp-t(np.asarray([c.logp for c in choices],np.float32))
        joint=lp.new_zeros(len(batch)).index_add(0,t(rows),diff)
        joint_entropy=lp.new_zeros(len(batch)).index_add(0,t(rows),d.entropy())
        if audit:
            error=joint.abs().max().item()
            if error>3e-4:
                raise AssertionError(f"排列联合 logp 重建不符：{error}")
            metrics.add(order_logp_max_error=error)
            audit=False
        update_policy(torch,net,opt,joint,t(advantage[ids]),joint_entropy,args,metrics,"order")
    for ids in batches(np,len(group),args["critic_epoch"],args["critic_num_mini_batch"]):
        obs,mask=pad_sets(np,[group[i].order_choices[0].obs for i in ids])
        pred=net.set_value(t(obs),t(mask))[:,None]
        y=t(target[ids])[:,None]
        loss=vc.cal_value_loss(pred,t(values_norm[ids])[:,None],y,vn)
        old.optimize(torch,opt,net,args["value_loss_coef"]*loss,args)
        with torch.no_grad():
            metrics.add(order_value_loss=loss.item(),order_explained_variance=ev(torch,pred,vn.normalize(y)).item())
    return dict(order_minutes=len(group),**metrics.report())


def tensor_hash(torch, net):
    h=hashlib.sha256()
    for name,x in sorted(net.state_dict().items()):
        v=x.detach().cpu().contiguous()
        h.update(repr((name,tuple(v.shape),str(v.dtype))).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def load_source(cli, folder, torch, device, env_type, cfg, rows, expected):
    from harl.common.valuenorm import ValueNorm

    if not cli.allow_trusted_checkpoints:
        raise ValueError("只加载自己生成的 PyTorch/归一化存档；请显式确认 --allow-trusted-checkpoints")
    p=expected["pointer"]
    model,normpath=old.checked_pointer(folder,p,STEPS)
    state=torch.load(model,map_location="cpu",weights_only=False)
    if (state["real_steps"],state["job"]) != (STEPS,expected["job"]):
        raise ValueError("原 ACAC Full 权重身份不符")
    with normpath.open("rb") as stream:
        norm=pickle.load(stream)
    if not norm.norm_obs or norm.norm_reward:
        raise ValueError("来源 ObsNorm 合同不符")
    norm.training=False
    dim=norm.observation_space.shape[0]
    Actor,Critic,width=old.model_types(torch)
    actors=[Actor(dim,True).to(device) for _ in range(2)]
    critic=Critic(dim,True,True).to(device)
    audit={}
    names=("passenger","aircraft","critic")
    for name,net,weights in zip(names,[*actors,critic],state["weights"],strict=True):
        net.load_state_dict(weights,strict=True)
        audit[name]=dict(sha=tensor_hash(torch,net),
                         loaded={k:list(v.shape) for k,v in weights.items()},missing=[],unexpected=[])
    vn=ValueNorm(1,device=device)
    for k in VALUE_KEYS:
        getattr(vn,k).copy_(state["value_statistics"][k].to(device))
    return actors,critic,vn,norm,dim,width,audit


def stats_state(vn):
    return {k:getattr(vn,k).detach().cpu() for k in VALUE_KEYS}


class FrozenNormFile:
    def __init__(self, source):
        self.source=source

    def save(self, output):
        # 这是冻结的原始归一化。直接复制已核验源文件，不重新 pickle 脱离 VecEnv 的对象。
        shutil.copyfile(self.source,output)


def guard(cli):
    if (cli.output/"STOP_REQUESTED").exists() or shutil.disk_usage(cli.output).free<5*2**30:
        raise InterruptedError("暂停或剩余磁盘不足 5 GiB；保留上一合法 rollout 恢复点")


def retain_data(cell, step, ticks, bootstrap):
    # 真正保存完整决策记录，不只保存抽样指标；默认只保留最终 rollout，防止占满共享磁盘。
    target=cell/"rollouts"/f"{step:09d}.pkl.gz"
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_suffix(".pending")
    with gzip.open(tmp,"wb",compresslevel=1) as out:
        pickle.dump(dict(ticks=[[asdict(m) for m in lane] for lane in ticks],bootstrap=bootstrap),out,protocol=5)
    tmp.replace(target)


def train(cli, cell, job, method, seed, cfg, rows, env_type, source, args, np, torch):
    import cloudpickle
    from harl.common.valuenorm import ValueNorm
    from harl.algorithms.critics.v_critic import VCritic

    order=method in ORDER
    device=torch.device(cli.device)
    base.set_seed(seed,np,torch)
    actors,critic,vn,norm,dim,width,loaded=load_source(cli,source["folder"],torch,device,env_type,cfg,rows,source)
    frozen_norm=FrozenNormFile(source["folder"]/source["pointer"]["norm"])
    actors[0].requires_grad_(not order)
    actors[1].requires_grad_(False)
    critic.requires_grad_(not order)
    OrderActor,PrefixQ=model_types(torch)
    extra=(OrderActor(dim).to(device) if order else PrefixQ(dim,width).to(device) if method=="B2_PREFIX_Q" else None)
    value_scale=max(1.,float(vn.denormalize(np.ones((1,1),np.float32))[0,0]
                            -vn.denormalize(np.zeros((1,1),np.float32))[0,0]))
    order_norm=ValueNorm(1,device=device) if order else None
    if order:
        for k in VALUE_KEYS:
            getattr(order_norm,k).copy_(getattr(vn,k))
    modules={"passenger":actors[0],"aircraft":actors[1],"critic":critic}
    if extra is not None:
        modules["order" if order else "prefix_q"]=extra
    rates={k:args["critic_lr" if k in ("critic","prefix_q") else "lr"] for k in modules}
    opts={k:torch.optim.Adam(m.parameters(),lr=rates[k],eps=args["opti_eps"],
                            weight_decay=args["weight_decay"])
          for k,m in modules.items() if any(p.requires_grad for p in m.parameters())}
    vc=VCritic(args,norm.observation_space,device=device)
    frozen={k:loaded[k]["sha"] for k in loaded if k not in opts}
    audit=dict(provenance={k:v for k,v in source.items() if k not in ("folder","evaluation")},
               strict_layers=loaded,frozen=frozen,
               updated={k:[n for n,p in modules[k].named_parameters() if p.requires_grad] for k in opts},
               new_module=(None if extra is None else dict(seed=seed,initialization="PyTorch 默认；价值输出层零初始化",
                           sha=tensor_hash(torch,extra))),obsnorm_frozen=True,prefix_value_scale=value_scale)
    contract(cell/"LOAD_AUDIT.json",audit)
    coll=Collector(env_type,cfg,rows,actors,critic,norm,torch,np,device,method,seed)
    real,counts=0,Counter()
    base.set_seed(seed,np,torch)
    resume=cell/"RESUME.json"
    if resume.is_file():
        meta=base.read(resume)
        real=int(meta["real_steps"])
        if not 0<real<=STEPS or real%base.ROLLOUT:
            raise ValueError("恢复点不在合法 rollout 边界")
        mp,_=old.checked_pointer(cell,meta,real)
        if meta["norm_sha"]!=source["pointer"]["norm_sha"]:
            raise ValueError("恢复点未保持冻结 ObsNorm")
        state=torch.load(mp,map_location="cpu",weights_only=False)
        if state["job"]!=job or state["real_steps"]!=real:
            raise ValueError("恢复点身份不匹配")
        for k,m in modules.items():
            m.load_state_dict(state["models"][k],strict=True)
        for k,opt in opts.items():
            opt.load_state_dict(state["optimizers"][k])
        for k in VALUE_KEYS:
            getattr(vn,k).copy_(state["values"][k])
            if order:
                getattr(order_norm,k).copy_(state["order_values"][k])
        coll.lanes=cloudpickle.loads(state["lanes"])
        coll.ah,coll.ch=state["ah"].to(device),state["ch"].to(device)
        counts.update(state["counts"])
        base.rng_restore(state["rng"],device,np,torch)
        print(f"{cell.name}：从 {real:,} 步续跑",flush=True)
    start,began=real,time.monotonic()
    try:
        while real<STEPS:
            guard(cli)
            for k,opt in opts.items():
                opt.param_groups[0]["lr"]=rates[k]*(1.-real/STEPS)
            with torch.no_grad():
                ticks,bootstrap=coll.collect(base.LENGTH,extra if order else None)
                raw=tick_advantages(np,ticks,bootstrap,vn,args)
                if method=="B1_EVENT_GAE":
                    event_gae(np,ticks,raw,vn,args)
            flat=[m for lane in ticks for m in lane]
            if len(flat)!=base.ROLLOUT:
                raise AssertionError("正式预算只能按物理分钟计数")
            multi=[m for m in flat if len(m.passengers)>1]
            triples=sum(len(m.passengers)>2 for m in multi)
            if not triples:
                raise RuntimeError("正式采样没有三人以上批次，动态优先级机制没有有效训练机会")
            stats={}
            if order:
                stats.update(order_update(torch,np,coll.t,extra,opts["order"],ticks,order_norm,vc,args,real==start))
            else:
                if method=="B2_PREFIX_Q":
                    # 辅助头打乱使用独立随机流，不扰动三种信用方法共用的 PPO minibatch 随机流。
                    rng=np.random.default_rng(np.random.SeedSequence([seed,real,0x5051]))
                    stats.update(prefix_credit(torch,np,coll.t,extra,opts["prefix_q"],ticks,args,value_scale,rng))
                stats.update(credit_update(torch,np,coll.t,ticks,actors[0],critic,opts["passenger"],
                                           opts["critic"],vn,vc,args,real==start))
            current=dict(physical_minutes=len(flat),passenger_events=sum(len(m.passengers) for m in flat),
                         aircraft_events=sum(m.aircraft is not None for m in flat),multi_passenger_minutes=len(multi),
                         three_plus_passenger_minutes=triples,
                         terminated=sum(m.terminal for m in flat),truncated=sum(m.trunc for m in flat))
            counts.update(current)
            real+=len(flat)
            # 与旧训练一致：rollout 边界携带采样隐藏态；恢复额外保存完整环境及随机流。
            is_mark=real in MARKS
            if is_mark or real==start+base.ROLLOUT:
                for k,v in frozen.items():
                    if tensor_hash(torch,modules[k])!=v:
                        raise AssertionError(f"不应更新的模块发生变化：{k}")
            if (cli.retain_rollouts=="all" or cli.retain_rollouts=="marks" and is_mark
                    or cli.retain_rollouts=="final" and real==STEPS):
                retain_data(cell,real,ticks,bootstrap)
            state=dict(job=job,real_steps=real,models={k:m.state_dict() for k,m in modules.items()},
                       optimizers={k:o.state_dict() for k,o in opts.items()},values=stats_state(vn),
                       order_values=stats_state(order_norm) if order else None,counts=dict(counts),
                       lanes=cloudpickle.dumps(coll.lanes),ah=coll.ah.detach().cpu(),ch=coll.ch.detach().cpu(),
                       rng=base.rng_snapshot(np,torch))
            base.save_generation(cell,real,STEPS,MARKS,frozen_norm,lambda path:torch.save(state,path),".pt")
            audit_row=dict(real_steps=real,elapsed_seconds=time.monotonic()-began,**current,**stats)
            audit_row["real_fps"]=(real-start)/max(1.,audit_row["elapsed_seconds"])
            base.save(cell/"updates"/f"{real:09d}.json",audit_row)
            base.save(cell/"LIVE.json",dict(audit_row,counts=dict(counts)))
        base.save(cell/"TRAINED.json",dict(real_steps=real,counts=dict(counts),
                  frozen_verified=frozen, final_sha={k:tensor_hash(torch,m) for k,m in modules.items()}))
    finally:
        for ln in coll.lanes:
            ln.view.close()


def eval_bundle(cli, env_type, cfg, rows, source, cell, job, mark, method, torch):
    actors,critic,vn,norm,dim,width,_=load_source(cli,source["folder"],torch,"cpu",env_type,cfg,rows,source)
    order_model=None
    if mark and method in TRAIN:
        ptr=base.read(cell/"checkpoints"/f"{mark:09d}.json")
        mf,_=old.checked_pointer(cell,ptr,mark)
        state=torch.load(mf,map_location="cpu",weights_only=False)
        if (state["job"],state["real_steps"])!=(job,mark):
            raise ValueError("回放读取的权重身份不符")
        actors[0].load_state_dict(state["models"]["passenger"],strict=True)
        actors[1].load_state_dict(state["models"]["aircraft"],strict=True)
        if method in ORDER:
            order_model=model_types(torch)[0](dim).eval()
            order_model.load_state_dict(state["models"]["order"],strict=True)
        sig=ptr["model_sha"]
    else:
        sig=source["pointer"]["model_sha"]
    return [a.eval() for a in actors],norm,width,order_model,sig


def replay(env_type,cfg,rows,seed,method,bundle,np,torch):
    actors,norm,width,order_model,_=bundle
    base.set_seed(seed,np,torch)
    lane=MinuteBatchLane(env_type(cfg,rows,"MAPPO"),0)
    hidden=[torch.zeros((1,width)) for _ in range(2)]
    t=torch.as_tensor
    audit=Counter()

    def act(role,x,valid):
        x=norm.normalize_obs(x[None])
        minute=lane.env.now//lane.env._ticks(cfg.decision_min)+1
        hidden[role]=actors[role].history.one(t(x),t([minute]),t([valid]),hidden[role])
        # 复刻历史回放：无机会也抽一次，但不更新 GRU 或采入训练样本。
        return int(torch.distributions.Categorical(logits=actors[role].head(hidden[role])).sample().item())

    try:
        with torch.no_grad():
            while not lane.env.done:
                lane.begin_tick()
                if len(lane.pending)>1:
                    audit["multi_passenger_minutes"]+=1
                if len(lane.pending)>2:
                    audit["three_plus_passenger_minutes"]+=1
                obs_for=lambda ids: norm.normalize_obs(np.stack([lane.observation(pid) for pid in ids]))
                static=None
                if order_model is not None and method=="A3_STATIC" and len(lane.pending)>1:
                    arr=obs_for(lane.pending)
                    static=(arr,order_model.scores(t(arr)))
                if order_model is None:
                    pending=iter(lane.rule_order(method,seed))
                first=None
                if not lane.pending:
                    lane.commit(None,act(0,lane.observation(),False))
                while lane.remaining:
                    pid=(next(pending) if order_model is None else
                         choose_order(torch,order_model,lane,t,obs_for,method,seed,static)[0])
                    if first is None:
                        first=pid
                    lane.commit(pid,act(0,lane.observation(pid),True))
                    audit["passenger_events"]+=1
                if first is not None and first!=lane.pending[0]:
                    audit["first_passenger_changed_minutes"]+=1
                lane.advance(act(1,lane.mid,lane.view.eligible()))
                audit["physical_minutes"]+=1
        return dict(metrics=lane.env.metrics(),control=dict(lane.view.stat),audit=dict(audit))
    finally:
        lane.view.close()


def source_cells(cli, source_protocol):
    result={}
    for seed in SEEDS:
        folder=cli.acac_source/"cells"/f"ACAC_FULL__s{seed}"
        job=base.read(folder/"JOB.json")
        if (job["method"],job["seed"],job["steps"],job["protocol_sha"])!=(
                "ACAC_FULL",seed,STEPS,base.digest(source_protocol)):
            raise ValueError(f"来源不是约定的 ACAC Full：{folder}")
        p=base.read(folder/"checkpoints"/f"{STEPS:09d}.json")
        old.checked_pointer(folder,p,STEPS)
        ep=folder/"EVALUATION.json"
        e=base.read(ep)
        if not base.read(folder/"STATUS.json")["evaluated_through_final"]:
            raise ValueError(f"来源评估未完成：{folder}")
        result[seed]=dict(folder=folder,job=job,pointer=p,evaluation=e,evaluation_sha=base.digest(ep))
    return result


def cached_pressure(cli, source, bank, seed, tag):
    expected={(name,p) for name,role,_ in bank if role=="final" for p in POLICY_SEEDS}
    if tag in ("A0_ORIGINAL","MAPPO_REUSED"):
        src=source["evaluation"]["pressure"]
        if [r["rate"] for r in src]!=list(RATES):
            raise ValueError("来源压力档位不匹配")
        for r in src:
            keys={(str(t),p) for _,t,p in (z["key"] for z in r["pairs"])}
            if keys!=expected or len(r["pairs"])!=len(expected) or any(z["key"][0]!=r["rate"] for z in r["pairs"]):
                raise ValueError("原 ACAC/MAPPO 压力回放的配对键不匹配")
        k="model" if tag=="A0_ORIGINAL" else "mappo"
        return [dict(rate=r["rate"],**r[k]) for r in src]
    ans=[]
    root=cli.transfer_output/"pressure"/f"E2__HAPPO__s{seed}"
    done=root/"DONE.json"
    if not done.is_file():
        raise FileNotFoundError(f"找不到已跑过的 HAPPO 压力回放完成标记：{done}")
    for rate in RATES:
        rows=[base.read(root/f"{rate:.1f}"/f"{trace}_{p}.json") for trace,p in sorted(expected)]
        if any(row["rate"]!=rate for row in rows) or {(row["trace"],row["policy_seed"]) for row in rows}!=expected:
            raise ValueError("原 HAPPO 压力回放配对键不匹配")
        ans.append(dict(rate=rate,**old.summarize(rows)))
    return ans


def alignment(cli,cfg,bank,env_type,sources,np,torch):
    # 生产启动内嵌一个冻结回合的验收；对照真实归档，不用交接稿中的舍入数字。
    p=cli.output/"BATCH_ALIGNMENT.json"
    s=sources[SEEDS[0]]
    signature=dict(model=s["pointer"]["model_sha"],evaluation=s["evaluation_sha"],
                   code=base.digest(Path(__file__)))
    if p.is_file():
        if base.read(p)["signature"]!=signature:
            raise ValueError("批次验收与当前源码/权重不一致")
        return
    name,_,rows=next(x for x in bank if x[0]=="final_8301" and x[1]=="final")
    rate,pseed=1.4,123
    dense=pressure_trace(rows,rate)
    bundle=eval_bundle(cli,env_type,cfg,dense,s,None,None,0,"A0_ORIGINAL",torch)
    sample=replay(env_type,cfg,dense,pseed,"A0_ORIGINAL",bundle,np,torch)
    target=next(x["model"] for r in s["evaluation"]["pressure"] for x in r["pairs"]
                if x["key"]==[rate,name,pseed])
    actual=old.scalar_metrics(sample["metrics"])
    if min(sample["audit"]["multi_passenger_minutes"],sample["audit"]["passenger_events"])<=0:
        raise ValueError("验收回合未实际覆盖同分钟多个乘客")
    if any(actual[k]!=target[k] and (None in (actual[k],target[k]) or
                                     not math.isclose(actual[k],target[k],rel_tol=0.,abs_tol=1e-7))
           for k in old.METRICS):
        raise RuntimeError(f"批次与真实归档不一致：{actual} / {target}")
    base.save(p,dict(signature=signature,passed=True,trace=name,policy_seed=pseed,
                     rate=rate,expected=target,actual=actual,audit=sample["audit"]))
    print(f"批次接口验收通过：{name}，{rate}×，ATT={actual['mean_att_min']:.9f}",flush=True)


def burst_eval(cli,cell,job,method,cfg,bank,env_type,bundle,np,torch):
    # 保留标准五档原始需求不动；另用同量突发组检验静态/动态的真实差异。
    signature=dict(job=job,model_sha=bundle[-1],pairs=cli.burst_pairs,triples=cli.burst_triples)
    rows=[]
    for name,role,original in bank:
        if role!="final":
            continue
        dense=burst_trace(original,cli.burst_pairs,cli.burst_triples)
        for ps in POLICY_SEEDS:
            guard(cli)
            path=cell/"burst_1x"/f"{name}_{ps}.json"
            if path.is_file():
                x=base.read(path)
                if x["signature"]!=signature or (x["trace"],x["policy_seed"])!=(name,ps):
                    raise ValueError("突发评估缓存合同不匹配")
            else:
                x=dict(signature=signature,trace=name,policy_seed=ps,
                       **replay(env_type,cfg,dense,ps,method,bundle,np,torch))
                base.save(path,x)
            if x["audit"].get("three_plus_passenger_minutes",0)<cli.burst_triples:
                raise ValueError("突发评估没有覆盖约定的三人批次")
            rows.append(x)
    return old.summarize(rows)


def eval_cell(cli,cell,job,method,cfg,bank,env_type,source,np,torch):
    trained=method in TRAIN
    standard={(r,name):pressure_trace(rows,r) for name,role,rows in bank if role=="final" for r in RATES}
    metrics,native=[],[]
    for mark in (MARKS if trained else (0,)):
        guard(cli)
        bundle=eval_bundle(cli,env_type,cfg,bank[0][2],source,cell,job,mark,method,torch)
        signature=dict(job=job,model_sha=bundle[-1],norm_sha=source["pointer"]["norm_sha"])

        def record(path,rows,trace,ps,rate):
            guard(cli)
            meta=dict(step=mark,rate=rate,trace=trace,policy_seed=ps)
            if path.is_file():
                x=base.read(path)
                if x["signature"]!=signature or any(x[k]!=v for k,v in meta.items()):
                    raise ValueError(f"回放缓存合同不匹配：{path}")
                return x
            x=dict(signature=signature,**meta,**replay(env_type,cfg,rows,ps,method,bundle,np,torch))
            base.save(path,x)
            return x

        if trained:
            for name,role,rows in bank:
                if role=="validation":
                    for ps in (123,124):
                        path=cell/"validation"/f"{mark:09d}"/f"{name}_{ps}.json"
                        native.append(record(path,rows,name,ps,1.))
        if mark not in (0,STEPS):
            continue
        for rate in RATES:
            sample=[record(cell/"pressure"/f"{rate:.1f}"/f"{name}_{p}.json",
                           standard[(rate,name)],name,p,rate)
                    for name,role,_ in bank if role=="final" for p in POLICY_SEEDS]
            metrics.append(dict(rate=rate,**old.summarize(sample)))
            print(f"{cell.name}：压力 {rate:.1f}× 回放结束",flush=True)
    if len(metrics)!=5 or trained and len(native)!=8*len(MARKS):
        raise AssertionError("最终回放或验证检查点数量不完整")
    burst=burst_eval(cli,cell,job,method,cfg,bank,env_type,bundle,np,torch)
    base.save(cell/"EVALUATION.json",dict(pressure=metrics,burst_1x=burst,validation=native,weights_unchanged=True))
    return dict(job=job,real_steps=STEPS if trained else 0,
                evaluation_sha=base.digest(cell/"EVALUATION.json"),rates=metrics,burst_1x=burst)


def write_report(cli,baseline,data,errors):
    items=[]
    for tag,seeds in {**baseline,**data}.items():
        for seed,row in seeds.items():
            for score in row["rates"]:
                rate=score["rate"]
                i=list(RATES).index(rate)
                ctrl=baseline["A0_ORIGINAL"][seed]["rates"][i]
                b0=(data.get("B0_SHARED",{}).get(seed,{}).get("rates") or [None]*5)[i]
                att=score["mean_att_min"]
                delta=lambda b: att-b["mean_att_min"] if b is not None and None not in (att,b["mean_att_min"]) else None
                items.append(dict(method=tag,seed=seed,**score,delta_to_A0=delta(ctrl),delta_to_B0=delta(b0)))
    path=cli.output/"PAIRED_SUMMARY.csv"
    tmp=path.with_suffix(".tmp")
    if items:
        with tmp.open("w",encoding="utf-8-sig",newline="") as stream:
            w=csv.DictWriter(stream,fieldnames=list(items[0]))
            w.writeheader()
            w.writerows(items)
        tmp.replace(path)
    base.save(cli.output/"SUMMARY.json",dict(baselines=baseline,methods=data,failures=errors))
    lines=["# E2 分钟批次结果", "", "ATT/AWT 单位为分钟；先在每个训练种子的匹配回合上取均值，再跨种子聚合。",
           "A0、HAPPO、MAPPO 复用原始五档回放；B 线的主要归因对照是 B0。", "",
           "|方法|压力|种子数|ATT|AWT|完成率|ATT−A0|ATT−B0|",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for tag in dict.fromkeys(x["method"] for x in items):
        for rate in RATES:
            group=[x for x in items if x["method"]==tag and x["rate"]==rate]
            if not group:
                continue
            def fmt(key):
                xs=[x[key] for x in group]
                return "-" if None in xs else f"{sum(xs)/len(xs):.3f}"
            lines.append("|"+"|".join([tag,str(rate),str(len(group)),*(fmt(k) for k in
                         ("mean_att_min","awt_min","completion","delta_to_A0","delta_to_B0"))])+"|")
    lines+=["","另列：1.0× 同量突发需求（含三人批次）；不与标准五档混表。","",
            "|方法|种子|ATT|AWT|完成率|ATT−A0|ATT−B0|","|---|---:|---:|---:|---:|---:|---:|"]
    for tag,seeds in {**baseline,**data}.items():
        for s,row in seeds.items():
            if "burst_1x" in row:
                scores=row["burst_1x"]
                values=[scores[k] for k in ("mean_att_min","awt_min","completion")]
                for ref in (baseline["A0_ORIGINAL"][s],data.get("B0_SHARED",{}).get(s,{})):
                    score=ref.get("burst_1x",{}).get("mean_att_min")
                    values.append(values[0]-score if None not in (values[0],score) else None)
                lines.append("|"+"|".join([tag,s,*("-" if x is None else f"{x:.3f}" for x in values)])+"|")
    lines+=["",f"本次失败：{len(errors)}。逐种子配对见 PAIRED_SUMMARY.csv。"]
    dest=cli.output/("ORDER_RESULTS.md" if cli.line=="order" else "CREDIT_RESULTS.md")
    draft=dest.with_suffix(".tmp")
    draft.write_text("\n".join(lines)+"\n",encoding="utf-8")
    draft.replace(dest)


def run(cli):
    import gymnasium as gym
    import numpy as np
    import stable_baselines3 as sb3
    import torch

    torch.set_num_threads(1)
    if cli.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定的 CUDA 不可用；不会自动换设备")
    versions=dict(torch=str(torch.__version__),numpy=np.__version__,
                  sb3=sb3.__version__,gymnasium=gym.__version__)
    cfg,original,bank,base_protocol=old.bank_for(cli,versions)
    args,_,vendor=base.harl_vendor(cli.harl_dir,np,torch)
    for k in ("use_policy_active_masks","use_proper_time_limits","use_gae","use_valuenorm"):
        if not args[k]:
            raise ValueError(f"原 PPO 合同缺少 {k}")
    if (base.LANES,base.LENGTH,args["gamma"],args["gae_lambda"],args["actor_num_mini_batch"],
            args["critic_num_mini_batch"])!=(5,2048,.99,.95,20,20):
        raise ValueError("原 ACAC PPO 或真实步合同不一致")
    pt=cli.acac_source/"PROTOCOL.json"
    src=base.read(pt)
    if (src["schema"]!="acac_e2_legal_decision_event_mappo_v1" or src["base_protocol"]!=base_protocol
            or src["harl_args"]!=old.canonical(args) or src["harl_files"]!=vendor
            or not old.same_source(Path(old.__file__),src["source_sha"])):
        raise ValueError("来源不是已验证的 ACAC Full 同一源码、物理和 PPO 合同")
    transfer=base.read(cli.transfer_output/"PROTOCOL.json")
    if transfer["source_protocol"]!=base_protocol:
        raise ValueError("不能复用其他历史合同的 HAPPO 结果")
    sources=source_cells(cli,pt)
    burst=burst_trace(original,cli.burst_pairs,cli.burst_triples)
    decision_bins=lambda rows: Counter(math.ceil(r.time_min/cfg.decision_min-1e-9) for r in rows)
    old_bins,new_bins=decision_bins(original),decision_bins(burst)
    if (max(old_bins.values())!=1 or max(new_bins.values())!=3
            or sum(n==2 for n in new_bins.values())!=cli.burst_pairs
            or sum(n==3 for n in new_bins.values())!=cli.burst_triples):
        raise ValueError("没有构造出约定数目的合法双人/三人分钟")
    manifest=dict(schema="corridor_acac_minute_batch_v1",line=cli.line,
        code_sha=base.digest(Path(__file__)),base_protocol_sha=base.digest(cli.base_output/"PROTOCOL.json"),
        source_protocol_sha=base.digest(pt),transfer_protocol_sha=base.digest(cli.transfer_output/"PROTOCOL.json"),
        references={s:{k:v for k,v in r.items() if k not in ("folder","evaluation")} for s,r in sources.items()},
        env_source_sha=base_protocol["environment_sha"],dispatch_sha=base_protocol["dispatch_sha"],
        config=asdict(cfg),versions=versions,train_source=old.demand_sha(original),train_burst=old.demand_sha(burst),
        burst_pairs=cli.burst_pairs,burst_triples=cli.burst_triples,requests=len(burst),
        first_min=burst[0].time_min,last_min=burst[-1].time_min,
        eval_contract=base_protocol["bank"],seeds=SEEDS,rates=RATES,marks=MARKS,steps=STEPS,
        ppo=args, device=cli.device,retain_rollouts=cli.retain_rollouts,
        common="1.0× 总量与首尾不变；同一分钟先逐人选站，最后一次走原接口，再调机一次",
        shared="B0–B2 的 Critic 只用原物理 Macro-GAE；只更换乘客优势，冻结 Aircraft 与 ObsNorm",
        credit="B2 在多乘客分钟用 Q 前缀差分，单乘客仍使用 B0；同一批的回报训练各前缀",
        order="A3/A4 只更新同构排序网络与集合值头；单 tick 用完整排列联合 log-prob",
        ordering_aliases={"FCFS":"A0_ORIGINAL","LONGEST_WAIT_FIRST":"A0_ORIGINAL"},
        rollout_state="完整分钟边界环境、GRU、RNG、优化器快照；各线独立；正式训练步只计真实分钟",
        deferred={"A5":"等 A3/A4 后定义条件 Order-Q",
                  "B3/B4":"需要固定随机流的正式反事实分支及额外计算预算，不以 V 或伪标签替代"})
    digest=contract(cli.output/"PROTOCOL.json",manifest)
    contract(cli.output/"DEMAND_BANK.json",dict(train_original=[asdict(r) for r in original],
             train=[asdict(r) for r in burst],**{name:[asdict(r) for r in rows] for name,_,rows in bank}))
    env_type,_=old.observed_env(gym,np,cfg)
    baseline={}
    for tag in ("HAPPO_REUSED","MAPPO_REUSED","A0_ORIGINAL"):
        baseline[tag]={str(s):dict(reused=True,real_steps=0,
                             rates=cached_pressure(cli,sources[s],bank,s,tag)) for s in SEEDS}
    alignment(cli,cfg,bank,env_type,sources,np,torch)
    for seed in SEEDS:
        source=sources[seed]
        bundle=eval_bundle(cli,env_type,cfg,bank[0][2],source,None,None,0,"A0_ORIGINAL",torch)
        baseline["A0_ORIGINAL"][str(seed)]["burst_1x"]=burst_eval(cli,cli.output/"reference"/f"A0__s{seed}",
            {"protocol_sha":digest,"seed":seed}, "A0_ORIGINAL",cfg,bank,env_type,bundle,np,torch)
    base.save(cli.output/"REUSED_BASELINES.json",baseline)
    print(f"{cli.line}：{len(burst)} 人、{cli.burst_pairs} 个双人批次、{cli.burst_triples} 个三人批次；"
          "复用 HAPPO/MAPPO/A0 各四种子的原五档",flush=True)
    universe=ORDER[1:] if cli.line=="order" else CREDIT
    selected=cli.methods or universe
    data,errors={},[]
    try:
        for method in universe:
            for seed in SEEDS:
                cell=cli.output/"cells"/f"{method}__s{seed}"
                status=cell/"STATUS.json"
                job=dict(protocol_sha=digest,method=method,seed=seed,steps=STEPS if method in TRAIN else 0)
                if method not in selected and not status.is_file():
                    continue
                try:
                    guard(cli)
                    contract(cell/"JOB.json",job)
                    if status.is_file():
                        row=base.read(status)
                        if row["job"]!=job or row["evaluation_sha"]!=base.digest(cell/"EVALUATION.json"):
                            raise ValueError("完成格评估记录的身份不符")
                    else:
                        print(f"{cell.name}：开始或继续",flush=True)
                        if method in TRAIN:
                            train(cli,cell,job,method,seed,cfg,burst,env_type,sources[seed],args,np,torch)
                        row=eval_cell(cli,cell,job,method,cfg,bank,env_type,sources[seed],np,torch)
                        base.save(status,row)
                    data.setdefault(method,{})[str(seed)]=row
                    print(f"{cell.name}：完成",flush=True)
                except (KeyboardInterrupt,InterruptedError):
                    raise
                except Exception:
                    err=dict(method=method,seed=seed,error=traceback.format_exc())
                    base.save(cell/"ERROR.json",err)
                    print(err["error"],flush=True)
                    errors.append(err)
                write_report(cli,baseline,data,errors)
    finally:
        write_report(cli,baseline,data,errors)
    if not errors and all(len(data.get(m,{}))==len(SEEDS) for m in universe):
        base.save(cli.output/"RUN_COMPLETE.json",dict(protocol_sha=digest,line=cli.line,
                  cells=sum(len(x) for x in data.values()),real_steps=sum(r["real_steps"] for s in data.values()
                  for r in s.values()),rates=list(RATES),failures=0))
    return int(bool(errors))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line",required=True,choices=("order","credit"))
    for name in ("config","train-trace","harl-dir","base-output","transfer-output","acac-source","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--methods",nargs="+",choices=(*ORDER[1:],*CREDIT))
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--burst-pairs",type=int,default=60)
    parser.add_argument("--burst-triples",type=int,default=24)
    parser.add_argument("--retain-rollouts",choices=("final","marks","all"),default="final")
    parser.add_argument("--allow-trusted-checkpoints",action="store_true",
                        help="仅在权重和 pickle 都是本账号已核验存档时启用")
    cli=parser.parse_args(argv)
    allowed=set(ORDER[1:] if cli.line=="order" else CREDIT)
    if cli.methods and not set(cli.methods)<=allowed:
        parser.error("所选方法不属于这条实验线")
    if not cli.allow_trusted_checkpoints:
        parser.error("必须明确允许读取自有来源模型 --allow-trusted-checkpoints")
    for k in ("config","train_trace","harl_dir","base_output","transfer_output","acac_source","output"):
        setattr(cli,k,getattr(cli,k).resolve())
    cli.output=cli.output/cli.line
    for p in (cli.base_output,cli.transfer_output,cli.acac_source):
        if cli.output==p or p in cli.output.parents or cli.output in p.parents:
            parser.error("新产物目录与已有实验重叠，拒绝写入")
    with base.lock_run(cli.output):
        try:
            return run(cli)
        except (KeyboardInterrupt,InterruptedError) as exc:
            print(f"暂停：{exc}",flush=True)
            return 130
        except Exception:
            base.save(cli.output/"RUN_ERROR.json",dict(error=traceback.format_exc()))
            raise


if __name__=="__main__":
    raise SystemExit(main())
