"""最初 K3 均匀 Lookahead–HAPPO 的 L0–L5：新走廊从零训练和评估。"""

from __future__ import annotations

import argparse
import copy
import time
import types
import ast
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
from types import SimpleNamespace

from corridor_env import (Config, CorridorEnv, Region, Station, baseline_episode,
                          cav_direct_episode, make_requests, validate_requests)

STATIONS = ("V0", "V1", "V3")
LAYERS = {"E1": (False, False), "E2": (True, False), "E3": (True, True)}

HARL_COMMIT = "b1af98b0dbab72a2eee9d160751cd09aedbb8ce2"
LANES, LENGTH = 5, 2048
ROLLOUT = LANES * LENGTH
PACK = 168 + 4 + 38 + 15 + 2 * 3 * 364 + 1


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def config(path, layer):
    d = read(path)
    d = dict(d.get("config", d))
    d["stations"] = {s: Station(**{**p, "xy": tuple(p["xy"])}) for s, p in d["stations"].items()}
    for k in ("origins", "destinations"):
        d[k] = tuple(Region(**{**p, "box": tuple(p["box"])}) for p in d[k])
    cfg = Config(**d)
    if (cfg.departures(), cfg.arrivals(), cfg.reposition) != (STATIONS, ("V2",), "lq"):
        raise ValueError("基准配置必须是 V0/V1/V3 → V2，默认空返为 LQ")
    if (cfg.decision_min != 1 or cfg.lateral or not cfg.tlof or not cfg.egress
            or cfg.fleet != 48 or cfg.seats != 1 or cfg.boarding != "immediate"):
        raise ValueError("需要冻结的走廊配置：1 分钟决策、48 架单座、TLOF 和末段开启、无横向调机")
    if (cfg.allocation() != dict.fromkeys(STATIONS, 16)
            or tuple(cfg.stations[s].chargers for s in (*STATIONS, "V2")) != (11, 11, 11, 24)
            or any(v.turnaround_min != 3 for v in cfg.stations.values())):
        raise ValueError("需要 16/16/16 初始机队、3 分钟周转参数及 11/11/11/24 充电位的 E3 配置")
    turn, charge = LAYERS[layer]
    return replace(cfg, turnaround=turn, finite_charging=charge, demand_csv=None).validate()


def trace(cfg, path):
    rows = make_requests(replace(cfg, demand_csv=str(path)))
    if len(rows) > 300 or max(q.time_min for q in rows) >= cfg.cutoff_min:
        raise ValueError("基线观测可容纳至多 300 个请求；所有请求须在截止前揭示")
    return rows


def derived_trace(rows, cfg, seed):
    # 沿用标定轨迹的到达时刻和经验起点分布；独立抽取起点和配置内的终点。
    # 这是新生成的评估需求，不是单条标定轨迹的重复回放。
    rng = random.Random(seed)
    return validate_requests([
        replace(q, origin=rng.choice(rows).origin,
                destination=(rng.uniform(z.box[0], z.box[1]), rng.uniform(z.box[2], z.box[3])))
        for q in rows
        for z in (rng.choices(cfg.destinations, weights=[r.weight for r in cfg.destinations])[0],)
    ])


class PriorityCorridor(CorridorEnv):
    """空返沿用走廊 LQ 的供需判定；联合策略只重排合法目的站。"""

    priority = None

    def needs(self, plane, ready, inbound, station):
        return (self._eligible(plane, station) and self._park_available(station)
                and len(self.queue[station]) > (ready[station] + inbound[station]) * self.cfg.seats
                and (not self.cfg.energy or self._soc_cost(plane.station, station)
                     + self.cfg.reserve_soc <= 1 + 1e-9))

    def supply_counts(self):
        return (Counter(p.station for p in self.planes.values() if p.state == "ready"),
                Counter(p.target for p in self.planes.values() if p.target in self.dep))

    def targets(self):
        ready, inbound = self.supply_counts()
        return {s for s in self.dep for p in self.planes.values()
                if p.station in self.arr and self.needs(p, ready, inbound, s)}

    def _reposition_auto(self):
        if self.priority is None:
            return super()._reposition_auto()
        ready, inbound = self.supply_counts()
        for p in self.planes.values():
            if p.state != "ready" or p.station not in self.arr:
                continue
            candidates = [s for s in self.dep if self.needs(p, ready, inbound, s)]
            candidates.sort(key=lambda s: (s != self.priority, -len(self.queue[s]), self._air(p.station, s), s))
            if not candidates:
                continue
            s = candidates[0]
            action = self._dispatch(p, s)
            if action in {"reserved", "charging"}:
                ready[p.station] -= 1
            if action == "reserved":
                inbound[s] += 1
                self.totals["priority_match" if s == self.priority else "priority_fallback"] += 1


def build_gym(gym, np):
    class LegacyPhase(gym.Env):
        """旧观测排列，新的物理状态；一个物理分钟只产生一次人时奖励。"""

        def __init__(self, cfg, rows, method):
            self.c = cfg
            self.rows = rows
            self.single = method == "UAGMC"
            self.env = PriorityCorridor(cfg)
            self.action_space = gym.spaces.Discrete(3)
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, (6, 28) if self.single else (PACK + 8,), dtype=np.float32)

        def focal(self):
            return next((p for p in self.env.passengers.values() if p.state == "route"), None)

        def snapshot(self, focal, full=True):
            e, c = self.env, self.c
            minute = c.tick_min
            access = {s: [] for s in STATIONS}
            land, charge, turn = {}, {}, {}
            for when, _, fn, args in e._events:
                name = fn.__name__
                if name == "_arrive_queue":
                    access[args[0].dep].append((when-e.now)*minute)
                elif name in ("_land", "_charged", "_turned"):
                    ({"_land": land, "_charged": charge, "_turned": turn}[name])[args[0].id] = when
                elif name == "_launch" and args[0].target in STATIONS:
                    # 已预订但尚未起飞的返程也属于已承诺运力。
                    land[args[0].id] = when + sum(args[1:4])
            od = list(focal.req.origin + focal.req.destination) if focal is not None else [0.] * 4
            frame, resources, tdm, times, masks = od.copy(), [], [], [], []
            for s in STATIONS:
                local = [a for a in e.planes.values() if a.station == s]
                inbound = [a for a in e.planes.values() if a.target == s and a.id in land]
                station = c.stations[s]
                # 沿用旧字段：等待、已选站在途、充电、当地机数、座位、充电时间、飞行时间。
                # 新增生命周期不造旧字段：排队充电/周转飞机不计 READY；名义 ETA 不假装知道释放时间。
                active_charge = sum(a.state == "charging" for a in local)
                charge_left = [max(0, charge.get(a.id, e.now) - e.now) * minute for a in local]
                flight_left = [max(0, land.get(a.id, e.now) - e.now) * minute for a in local]
                mean = lambda xs: sum(xs)/len(xs) if xs else 0.
                q, incoming = len(e.queue[s]), sorted(access[s])
                frame.extend([q, len(incoming), active_charge, len(local), len(local)*c.seats,
                              mean(charge_left), min(charge_left, default=0.), mean(flight_left)])
                if not full:
                    continue
                events, batteries = [], []
                idle = sum(a.state == "ready" for a in local)
                for a in local + [a for a in inbound if a.station != s]:
                    if a.riders:
                        continue
                    nominal_charge = (max(0., c.target_soc-a.soc)/station.charge_soc_per_min
                                      if c.energy else 0.)
                    if a.id in land and a.target == s:
                        soc = max(0., a.soc - e._soc_cost(a.station, s)) if a.state == "reserved" else a.soc
                        nominal_charge = max(0., c.target_soc-soc)/station.charge_soc_per_min if c.energy else 0.
                        eta = (land[a.id]-e.now)*minute + (station.turnaround_min if c.turnaround else 0.) + nominal_charge
                    elif a.state == "ready":
                        eta = 0.
                    elif a.id in charge:
                        eta = (charge[a.id]-e.now)*minute
                    elif a.id in turn:
                        eta = (turn[a.id]-e.now)*minute + nominal_charge
                    else:
                        continue
                    events.append(eta)
                    batteries.append(a.soc)
                events.sort()
                for values, slots in ((incoming, 300), (events, 64)):
                    if len(values) > slots:
                        raise RuntimeError("已承诺事件超出旧观测槽，不静默截断")
                    times.extend([v/60. for v in values] + [0.]*(slots-len(values)))
                    masks.extend([1.]*len(values) + [0.]*(slots-len(values)))
                tau = e._drive(focal.req.origin, station.xy, c.access_kmh)*minute if focal else 0.
                demand = q + sum(t <= tau for t in incoming) + int(focal is not None)
                ready = sum(t <= tau for t in events)
                tdm.extend([tau/60., demand/300., len(incoming)/300., ready/20., demand/(1+ready)/300.])
                resources.extend([q/300., len(incoming)/300., idle/20., active_charge/20., len(inbound)/20.,
                                  min(events, default=600.)/60., mean(events)/60. if events else 10.,
                                  mean(batteries), station.xy[0]/60., station.xy[1]/60.])
            local = [a for a in e.planes.values() if a.station == "V2"]
            resources.extend([sum(a.state == "ready" for a in local)/20.,
                              sum(a.state == "charging" for a in local)/20., len(e.planes)/100.,
                              len(e.passengers)/300., e.finished/300., e.now*minute/600., 1., 1.])
            return frame, od+resources+tdm+times+masks+[float(focal is not None)]

        def observation(self, post=False, focal=None):
            if not post:
                focal = self.focal()
            frame, extras = self.snapshot(focal, not self.single)
            history = (self.history[1:] + [frame]) if post else self.history
            if self.single:
                return np.asarray(history, dtype=np.float32)
            # 延迟门未启用；保持旧版 8 个尾字段，不塞进新的资源或未来需求。
            tail = [int(post), *(int(post and self.route_index == i) for i in range(3)), 0., 0., 0., 0.]
            obs = np.asarray([x for f in history for x in f]+extras+tail, dtype=np.float32)
            if obs.shape != (PACK+8,) or not np.isfinite(obs).all():
                raise RuntimeError("联合观测维数或数值不符")
            return obs

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.env.priority = None
            self.env.reset(self.rows)
            frame, _ = self.snapshot(self.focal(), False)
            self.history = [frame]*6
            self.phase, self.route_index = 0, 0
            self.stat = Counter()
            return self.observation(), {}

        def advance(self, routes=None):
            _, reward, done, cutoff, info = self.env.step(routes)
            self.phase = 0
            frame, _ = self.snapshot(self.focal(), False)
            self.history = self.history[1:]+[frame]
            if done or cutoff:
                info["control"] = dict(self.stat)
            info["physical_increment"] = 1
            return self.observation(), reward, done, cutoff, info

        def step(self, action):
            action = int(action)
            if not self.action_space.contains(action):
                raise ValueError("动作必须是 V0/V1/V3 的索引")
            if self.phase == 0:
                p = self.focal()
                self.route_index = action if p is not None else 0
                if p is not None:
                    self.stat["route_"+STATIONS[action]] += 1
                if self.single:
                    return self.advance({p.req.id: STATIONS[action]} if p is not None else None)
                if p is not None:
                    self.env._route(p, STATIONS[action])
                # 同一时刻先处理已到达事件与载客需求，不提前触发空返或推进时钟。
                import heapq
                e = self.env
                while True:
                    while e._events and e._events[0][0] == e.now:
                        _, _, fn, args = heapq.heappop(e._events)
                        fn(*args)
                    e._local_service()
                    if not e._events or e._events[0][0] != e.now:
                        break
                self.phase = 1
                return self.observation(post=True, focal=p), 0., False, False, {"physical_increment": 0}
            targets = self.env.targets()
            requested = STATIONS[action]
            self.stat["air_decisions"] += 1
            self.stat["air_no_opportunity"] += not targets
            self.stat["air_multi_target"] += len(targets) > 1
            self.stat["air_priority_projection"] += bool(targets) and requested not in targets
            if requested not in targets:
                # 无可行目标时退回默认 LQ；不把没有合法机会计成主动不调机。
                requested = None
            self.env.priority = requested
            self.stat["requested_"+STATIONS[action]] += 1
            return self.advance()

    return LegacyPhase

def harl_vendor(path, np, torch):
    import yaml

    # 优先使用现有只读作者副本；只抽取作者 HAPPO 更新，避免导入星际等无关环境。
    yml = path/"harl/configs/algos_cfgs/happo.yaml"
    runner = path/"harl/runners/on_policy_ha_runner.py"
    if not yml.is_file() or not runner.is_file():
        raise FileNotFoundError(f"HARL 作者源码目录不完整：{path}")
    if (path/".git").exists():
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if head != HARL_COMMIT or subprocess.check_output(
                ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
            raise ValueError(f"HARL 请使用干净的作者提交 {HARL_COMMIT}")
    hashes = {p.relative_to(path).as_posix(): digest(p) for p in sorted((path/"harl").rglob("*.py"))}
    hashes[yml.relative_to(path).as_posix()] = digest(yml)
    sys.path.insert(0, str(path))
    raw = yaml.safe_load(yml.read_text(encoding="utf-8"))
    args = {**raw["model"], **raw["algo"], **raw["train"]}
    args.update(episode_length=LENGTH, n_rollout_threads=LANES, actor_num_mini_batch=20,
                critic_num_mini_batch=20, gamma=.99, gae_lambda=.95, use_recurrent_policy=False,
                use_naive_recurrent_policy=False)
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OnPolicyHARunner")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    ns = dict(np=np, torch=torch, _t2n=lambda x: x.detach().cpu().numpy())
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(runner), "exec"), ns)
    return args, ns["train"], hashes


def set_seed(seed, np, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def verified(folder, meta):
    model, norm = folder/meta["model"], folder/meta["norm"]
    if digest(model) != meta["model_sha"] or digest(norm) != meta["norm_sha"]:
        raise ValueError(f"存档与归一化校验失败：{folder}")
    return model, norm


def rng_snapshot(np, torch):
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def rng_restore(state, device, np, torch):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if str(device) == "cuda" and state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])

def lock_run(root):
    # 使用操作系统文件锁；进程退出后自动释放，异常时不用手工清理旧锁。
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/"matrix.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise RuntimeError("输出目录已有训练或评估进程，请不要启动重复任务")
    return lock


def eval_rules(out, layer, cfg, bank):
    # 每个需求在原走廊规则和 CAV 下各跑一次；不复制成多个随机种子。
    path = out/"RULES"/f"{layer}.json"
    if path.is_file():
        return
    rows = []
    for name, role, requests in bank:
        for rule in ("SPF_SRC", "STTF", "QTTI", "READY", "BOTH_V0", "BOTH_V1", "BOTH_V3", "CAV_DIRECT"):
            if (out/"STOP_REQUESTED").exists():
                raise InterruptedError("规则评估暂停")
            forced = rule.removeprefix("BOTH_") if rule.startswith("BOTH_") else None
            settings = replace(cfg, initial_fleet={forced: cfg.fleet}) if forced else cfg
            result = cav_direct_episode(cfg, requests) if rule == "CAV_DIRECT" else baseline_episode(
                settings, requests, "FORCED" if forced else rule,
                dict.fromkeys(STATIONS, 3), dict.fromkeys(STATIONS, 30), forced)
            rows.append(dict(trace=name, role=role, rule=rule, **result))
    save(path, rows)

SHADOW_H=20
SHADOW_N=128
WARM=2
RING=8
SAMPLES=2048


# 元组是虚拟更新轮数、真实环境用不用影子、价值估计是否用 V-trace。
ARMS={
    'L0_HAPPO':(0,False,False),
    'L1_FULL':(1,True,True),
    'L2_SHADOW_2':(2,True,True),
    'L3_SHADOW_4':(4,True,True),
    'L4_DISCARD_SHADOW':(1,False,True),
    'L5_NO_VTRACE':(1,True,False),
}

def vtrace(reward,value,bootstrap,continues,discount,logp,enabled=True):
    """价值校正用联合比率；返回不含角色比率的 Q−V，供独立 PPO 权重使用。"""
    import numpy as np
    if not all(np.isfinite(x).all() for x in (reward,value,bootstrap,logp)):
        raise ValueError('V-trace 输入包含非有限数值')
    ratio=np.exp(np.clip(logp,-40,40))
    c=np.minimum(1.,ratio) if enabled else np.ones_like(ratio)
    corrected=np.empty_like(value)
    delta=reward+discount*bootstrap-value
    carry=np.zeros_like(value[0])
    for i in range(len(value)-1,-1,-1):
        carry=c[i]*delta[i]+discount[i]*continues[i]*c[i]*carry
        corrected[i]=value[i]+carry
    # 切开 episode 后的 reset 样本；最后一个 rollout 步使用真实 next value。
    nxt=bootstrap.copy()
    nxt[:-1]=np.where(continues[:-1],corrected[1:],bootstrap[:-1])
    advantage=reward+discount*nxt-value
    if not np.isfinite(corrected).all() or not np.isfinite(advantage).all():
        raise ValueError('V-trace 目标非有限')
    return corrected,advantage,ratio


def ratio_parts(new,anchor,own_beh,prefix):
    """PPO 裁剪只看新/旧正式策略，旧正式/行为是外部权重。"""
    return new-anchor,anchor-own_beh+prefix

def policy_logits(actor,obs,zeros,masks,actions):
    """通过作者接口拿分布；不重写 actor 网络。"""
    _,ent,dist=actor.evaluate_actions(obs,zeros,actions,masks,None,None)
    if not hasattr(dist,'logits'):
        raise ValueError('本轮必须是 K3 单离散 Categorical')
    return dist.logits,ent


def offpolicy_update(actors,critic,cb,vn,buffs,adv,anchor,order,np,torch):
    """本项目的新顺序更新层，不修改 vendor；保留作者 actor、critic、优化器。"""
    advantage=(adv-adv.mean())/(adv.std()+1e-5)
    flat=advantage.reshape(-1,1)
    prefix=np.zeros_like(flat)
    report=[]
    for i in order:
        actor,b=actors[i],buffs[i]
        n=flat.shape[0]
        obs=b.obs[:-1].reshape(n,-1)
        zs=b.rnn_states[:-1].reshape(n,*b.rnn_states.shape[2:])
        act=b.actions.reshape(n,1).astype(np.int64)
        masks=b.masks[:-1].reshape(n,1)
        mu=b.action_log_probs.reshape(n,1)
        pi0=anchor[i].reshape(n,1)
        sums=[]
        # 前序权重当前角色训练期间固定；不把联合 V-trace 比率重复乘到目标。
        w=np.exp(np.clip(pi0-mu+prefix,-10,10)).astype(np.float32)
        width=n//actor.actor_num_mini_batch
        if n%actor.actor_num_mini_batch:
            raise ValueError('正式 minibatch 不整除')
        for _ in range(actor.ppo_epoch):
            for ix in torch.randperm(n).numpy().reshape(-1,width):
                raw,ent,_=actor.evaluate_actions(obs[ix],zs[ix],act[ix],masks[ix],None,None)
                target=(torch.as_tensor(flat[ix],**actor.tpdv))
                ratio=(raw-torch.as_tensor(pi0[ix],**actor.tpdv)).clamp(-20,20).exp()
                weight=torch.as_tensor(w[ix],**actor.tpdv)
                term=torch.minimum(ratio*target,ratio.clamp(1-actor.clip_param,1+actor.clip_param)*target)
                loss=-(weight*term).mean()
                actor.actor_optimizer.zero_grad(set_to_none=True)
                (loss-actor.entropy_coef*ent).backward()
                grad=torch.nn.utils.clip_grad_norm_(actor.actor.parameters(),actor.max_grad_norm,
                                                    error_if_nonfinite=True)
                actor.actor_optimizer.step()
                sums.append([loss.item(),ent.item(),grad.item()])
        with torch.no_grad():
            new=[]
            for j in range(0,n,2048):
                z=slice(j,j+2048)
                lp,_,_=actor.evaluate_actions(obs[z],zs[z],act[z],masks[z],None,None)
                new.append(lp.cpu().numpy())
        prefix+=np.concatenate(new)-mu
        report.append(dict(role=i,losses=np.mean(sums,axis=0),mean_weight=float(w.mean()),
                           max_weight=float(w.max())))
    cm=critic.train(cb,vn)
    return report,cm


def wm_model(dim,torch):
    from torch import nn
    class Dynamics(nn.Module):
        """共享当前状态代理；单独预测乘客选站后观测与整分钟转移、奖励。"""
        def __init__(self):
            super().__init__()
            def net(ni,no):
                return nn.Sequential(nn.Linear(ni,192),nn.SiLU(),nn.Linear(192,192),
                                     nn.SiLU(),nn.Linear(192,no))
            core=dim-8
            self.post=net(dim+3,2*core+3)
            self.next=net(dim+6,2*(core+1))

        def post_parameters(self,x,p):
            return self.post(torch.cat((x,nn.functional.one_hot(p,3).float()),-1))

        def next_parameters(self,x,p,a):
            return self.next(torch.cat((x,nn.functional.one_hot(p,3).float(),
                                        nn.functional.one_hot(a,3).float()),-1))
    return Dynamics()


def gaussian(params,torch):
    mean,log_std=params.chunk(2,dim=-1)
    return mean,log_std.clamp(-5,1)


def fit_world(model,opt,ring,norm,torch,np,rng,device):
    data={k:np.concatenate([block[k] for block in ring])
          for k in ('x','post','next','p','executed_p','a','r','done','age')}
    # 不把自动 reset 引入学习目标；留出真实样本由固定下标切出。
    ids=np.arange(len(data['r']))
    train=ids[ids%8!=0]; valid=ids[ids%8==0]
    if not len(train) or not len(valid):
        raise ValueError('真实回放池不足')
    pack=norm.observation_space.shape[0]-8
    reward_unit=100.
    def tx(x,dtype=torch.float32):
        return torch.as_tensor(x,dtype=dtype,device=device)
    def err(index):
        x=tx(norm.normalize_obs(data['x'][index]))
        before=tx(norm.normalize_obs(data['post'][index])[:,:pack])
        after=tx(norm.normalize_obs(data['next'][index])[:,:pack])
        p,a=[tx(data[k][index],torch.long) for k in ('p','a')]
        reward=tx(data['r'][index]/reward_unit).reshape(-1)
        mask=tx(~data['done'][index]).reshape(-1)
        post_parameters=model.post_parameters(x,p)
        u,lu=gaussian(post_parameters[:,:-3],torch)
        logits=post_parameters[:,-3:]
        v,lv=gaussian(model.next_parameters(x,p,a),torch)
        def nll(y,z,l):
            return .5*((y-z)*torch.exp(-l)).square()+l
        e_post=nll(before,u,lu).mean()
        e_next=(nll(after,v[:,:pack],lv[:,:pack]).mean(-1)*mask).sum()/mask.sum().clamp_min(1.)
        e_rew=nll(reward,v[:,-1],lv[:,-1]).mean()
        applied=tx(data['executed_p'][index],torch.long)
        e_applied=torch.nn.functional.cross_entropy(logits,applied)
        return e_post+e_next+e_rew+e_applied,(before,u,after,v,mask,reward,x,logits,applied)
    model.train()
    for _ in range(48):
        ix=rng.choice(train,256,replace=len(train)<256)
        loss,_=err(ix)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        opt.step()
    model.eval()
    with torch.no_grad():
        loss,(post,post_hat,nxt,y,mask,r,x,logits,applied)=err(valid[:1024])
        report=dict(nll=float(loss),post_rmse=float((post-post_hat).square().mean().sqrt()),
                    next_rmse=float((((nxt-y[:,:pack]).square().mean(-1)*mask).sum()
                                    /mask.sum().clamp_min(1)).sqrt()),
                    persistence_rmse=float((((nxt-x[:,:pack]).square().mean(-1)*mask).sum()
                                    /mask.sum().clamp_min(1)).sqrt()),
                    executed_passenger_accuracy=float((logits.argmax(-1)==applied).float().mean()),
                    reward_mae=float((r-y[:,-1]).abs().mean()*reward_unit),n=len(valid[:1024]))
    return data,report


def shadow_team(real,vc,vn):
    # deepcopy 整个训练对象以保持参数与各自优化器引用；任何虚拟更新都不写回。
    return copy.deepcopy((real,vc,vn))


def value_probe(vn):
    return {k:getattr(vn,k).detach().cpu() for k in ('running_mean','running_mean_sq','debiasing_term')}


def runner_for(args,actors,critic,an,cb,vn):
    return types.SimpleNamespace(algo_args={'train':args},actor=actors,critic=critic,actor_buffer=an,
                                 critic_buffer=cb,value_normalizer=vn,state_type='EP',
                                 fixed_order=False,num_agents=2,action_aggregation='prod')


def imagine(team,args,space,action_space,model,norm,dataset,depth,author,Buffer,CB,np,torch,rng,device,cutoff):
    actors,critic,vn=team
    shape=dict(args,episode_length=SHADOW_H,n_rollout_threads=SHADOW_N)
    for a in actors:
        a.actor_num_mini_batch=10
        a.entropy_coef=.001
    critic.critic_num_mini_batch=10
    bs=[Buffer(shape,space,action_space) for _ in actors]; cb=CB(shape,space)
    runner=runner_for(shape,actors,critic,bs,cb,vn)
    zero=np.zeros((SHADOW_N,args['recurrent_n'],args['hidden_sizes'][-1]),np.float32)
    ones=np.ones((SHADOW_N,1),np.float32)
    d=space.shape[0]; core=d-8
    starts=np.arange(len(dataset['r']))
    # 用走廊截止时间替换旧环境写死的 480 分钟起点边界。
    starts=starts[(starts%8!=0)&(dataset['age'] + SHADOW_H < cutoff)&(~dataset['done'])]
    if not len(starts):
        raise RuntimeError('没有合法的虚拟起点')
    def array(t): return t.detach().cpu().numpy()
    def sample(param):
        mean,log_std=gaussian(param,torch)
        return mean+torch.randn_like(mean)*log_std.exp()
    raw_tail=np.zeros((4,d),np.float32)
    raw_tail[1:,-8]=1.
    raw_tail[1:,-7:-4]=np.eye(3)
    tails=norm.normalize_obs(raw_tail)[:,-8:]
    reports=[]
    for _ in range(depth):
        obs=norm.normalize_obs(dataset['x'][rng.choice(starts,SHADOW_N,replace=True)])
        for a in actors: a.prep_rollout()
        critic.prep_rollout()
        reward_avg=[]
        with torch.no_grad():
            for k in range(SHADOW_H):
                values,_=critic.get_values(obs,zero,ones)
                p,lp,_=actors[0].get_actions(obs,zero,ones)
                x=torch.as_tensor(obs,device=device,dtype=torch.float32); pi=p.long().flatten()
                pp=model.post_parameters(x,pi)
                y=sample(pp[:,:-3]).clamp(-10.,10.)
                # 无待选乘客时真实环境把请求动作映射为 0；在影子内用学习到的执行类别。
                executed=torch.distributions.Categorical(logits=pp[:,-3:]).sample()
                mid=np.concatenate((array(y),tails[1+array(executed)]),axis=-1)
                a,la,_=actors[1].get_actions(mid,zero,ones)
                out=sample(model.next_parameters(x,pi,a.long().flatten()))
                nxt=np.concatenate((array(out[:,:core].clamp(-10.,10.)),np.repeat(tails[:1],SHADOW_N,axis=0)),axis=-1)
                r=array((out[:,-1:]*100.).clamp(-300.,0.))
                reward_avg.append(float(r.mean()))
                bs[0].obs[k]=obs; bs[1].obs[k]=mid; cb.share_obs[k]=obs
                for b,act,logp in zip(bs,(p,a),(lp,la)):
                    b.insert(nxt,zero,array(act),array(logp),ones,ones)
                cb.insert(nxt,zero,array(values),r,ones,ones)
                obs=nxt
            v,_=critic.get_values(obs,zero,ones)
            cb.compute_returns(array(v),vn)
        for a in actors: a.prep_training()
        critic.prep_training()
        am,cm=author(runner)
        reports.append(dict(mean_imagined_reward=float(np.mean(reward_avg)),actor=am,critic=cm))
    return reports


def probe_logits(actors,bs,torch,ix):
    out=[]
    with torch.no_grad():
        for a,b in zip(actors,bs):
            n=b.actions.shape[0]*b.actions.shape[1]
            z=b.rnn_states[:-1].reshape(n,*b.rnn_states.shape[2:])
            logits,_=policy_logits(a,b.obs[:-1].reshape(n,-1)[ix],z[ix],
                                   b.masks[:-1].reshape(n,1)[ix],b.actions.reshape(n,1)[ix])
            out.append(logits.detach().cpu().numpy())
    return out


def build_pair(args, space, action_space, device):
    from harl.algorithms.actors.happo import HAPPO
    from harl.algorithms.critics.v_critic import VCritic
    from harl.common.valuenorm import ValueNorm
    return ([HAPPO(args, space, action_space, device=device) for _ in range(2)],
            VCritic(args, space, device=device), ValueNorm(1, device=device))


def atomic_write(path, writer):
    tmp = path.with_suffix(path.suffix + ".pending")
    writer(tmp)
    tmp.replace(path)


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def checkpoint(folder, real, marks, previous, state, extra, norm, torch):
    # 评估点只存正式团队和归一化；大回放池仅存于训练恢复槽。
    tag = "rolling_b" if (previous or {}).get("tag") == "rolling_a" else "rolling_a"
    stem = f"step_{real:09d}" if real in marks else tag
    model, npath = folder/(stem+".pt"), folder/(stem+".norm.pkl")
    atomic_write(model, lambda p: torch.save(state, p))
    atomic_write(npath, lambda p: norm.save(str(p)))
    meta = dict(real_steps=real, model=model.name, norm=npath.name, model_sha=digest(model),
                norm_sha=digest(npath), tag=tag, imagined_steps=state["imagined_steps"],
                model_updates=state["model_updates"])
    if real in marks:
        save(folder/"checkpoints"/f"{real:09d}.json", meta)
    if extra is not None:
        path = folder/(tag+".world.pt")
        atomic_write(path, lambda p: torch.save(extra, p))
        meta.update(world=path.name, world_sha=digest(path))
    save(folder/"RESUME.json", meta)
    return meta


def train_cell(folder, cli, arm, seed, cfg, rows, marks, env_class, args, author, gym, np, torch):
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer as Buffer
    from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP as CB

    set_seed(seed, np, torch)
    ptr = read(folder/"RESUME.json") if (folder/"RESUME.json").is_file() else None
    real = ptr["real_steps"] if ptr else 0
    if real == cli.steps:
        return
    if real > cli.steps or real % ROLLOUT:
        raise ValueError("恢复点不在本合同的 rollout 边界")
    vec = SubprocVecEnv([lambda: env_class(cfg, rows, "HAPPO") for _ in range(LANES)], start_method="spawn")
    norm = None
    try:
        if ptr:
            model_file, norm_file = verified(folder, ptr)
            norm = VecNormalize.load(str(norm_file), vec)
        else:
            norm = VecNormalize(vec, norm_obs=True, norm_reward=False, gamma=.99)
        norm.training = True
        space, action_space = norm.observation_space, gym.spaces.Discrete(3)
        device = torch.device(cli.device)
        actors, critic, vn = build_pair(args, space, action_space, device)
        depth, use_shadow, use_vtrace = ARMS[arm]
        rng = np.random.default_rng(61811+seed)
        rs = rng_snapshot(np, torch)
        dynamics = wm_model(space.shape[0], torch).to(device) if depth else None
        opt = torch.optim.Adam(dynamics.parameters(), lr=3e-4) if depth else None
        rng_restore(rs, cli.device, np, torch)
        ring, updates, imag, fits = [], 0, 0, 0
        if ptr:
            state = torch.load(model_file, map_location="cpu", weights_only=False)
            if state["real_steps"] != real:
                raise ValueError("模型训练步数与指针不一致")
            for actor, w, o in zip(actors, state["actors"], state["optimizers"]):
                actor.actor.load_state_dict(w)
                actor.actor_optimizer.load_state_dict(o)
            critic.critic.load_state_dict(state["critic"])
            critic.critic_optimizer.load_state_dict(state["critic_optimizer"])
            vn.load_state_dict(state["value_norm"])
            for k, v in state["value_statistics"].items():
                getattr(vn, k).copy_(v)
            updates, imag, fits = state["updates"], state["imagined_steps"], state["model_updates"]
            if depth:
                path = folder/ptr["world"]
                if digest(path) != ptr["world_sha"]:
                    raise ValueError("世界模型与回放池存档校验失败")
                extra = torch.load(path, map_location="cpu", weights_only=False)
                dynamics.load_state_dict(extra["wm"])
                opt.load_state_dict(extra["opt"])
                ring = extra["ring"]
                rng.bit_generator.state = extra["rng"]
                del extra
            rng_restore(state["rng"], cli.device, np, torch)
            del state

        n, t = LANES, LENGTH
        bs = [Buffer(args, space, action_space) for _ in actors]
        cb = CB(args, space)
        runner = runner_for(args, actors, critic, bs, cb, vn)
        z = np.zeros((n, args["recurrent_n"], args["hidden_sizes"][-1]), np.float32)
        ones = np.ones((n, 1), np.float32)
        masks = ones.copy()
        obs, age = norm.reset(), np.zeros(n, np.int32)
        initial, since = real, time.monotonic()
        stop = cli.output/"STOP_REQUESTED"

        def array(x):
            return x.detach().cpu().numpy()

        def live(phase):
            save(folder/"LIVE.json", dict(phase=phase, real_steps=real, updates=updates,
                 imagined_steps=imag, model_updates=fits, real_fps=(real-initial)/max(1., time.monotonic()-since)))

        def capture():
            state = dict(real_steps=real, updates=updates, imagined_steps=imag, model_updates=fits,
                         actors=[a.actor.state_dict() for a in actors],
                         optimizers=[a.actor_optimizer.state_dict() for a in actors],
                         critic=critic.critic.state_dict(), critic_optimizer=critic.critic_optimizer.state_dict(),
                         value_norm=vn.state_dict(), value_statistics=value_probe(vn), rng=rng_snapshot(np, torch))
            extra = (dict(wm=dynamics.state_dict(), opt=opt.state_dict(), ring=ring,
                          rng=rng.bit_generator.state) if depth and real < cli.steps else None)
            result = checkpoint(folder, real, marks, ptr, state, extra, norm, torch)
            if real == cli.steps:
                # 正式最终存档不依赖大回放池；只清理本格两个已失效的恢复缓存。
                for name in ("rolling_a.world.pt", "rolling_b.world.pt"):
                    (folder/name).unlink(missing_ok=True)
            print(f"{folder.name}：真实 {real:,} / {cli.steps:,}，虚拟 {imag:,}，拟合更新 {fits:,}", flush=True)
            return result

        while real < cli.steps:
            if stop.exists():
                if real > (ptr["real_steps"] if ptr else 0):
                    ptr = capture()
                raise InterruptedError("已在更新边界暂停")
            for actor in actors:
                actor.prep_rollout()
                actor.lr_decay(real, cli.steps)
            critic.prep_rollout()
            critic.lr_decay(real, cli.steps)
            active = bool(depth and updates >= WARM)
            shadow, audit = None, {}
            if active:
                live("fit_and_imagine")
                rs = rng_snapshot(np, torch)
                dataset, report = fit_world(dynamics, opt, ring, norm, torch, np, rng, device)
                shadow = shadow_team(actors, critic, vn)
                audit.update(model=report, shadow=imagine(shadow, args, space, action_space, dynamics, norm,
                    dataset, depth, author, Buffer, CB, np, torch, rng, device, cfg.cutoff_min))
                del dataset
                rng_restore(rs, cli.device, np, torch)
                imag += depth*SHADOW_H*SHADOW_N
                fits += 48
            behavior = shadow[0] if active and use_shadow else actors
            for actor in behavior:
                actor.prep_rollout()
            for b in bs:
                b.step, b.factor = 0, None
            cb.step = 0
            anchor = np.zeros((2,t,n,1), np.float32)
            ends = np.full((t,n,1), np.nan, np.float32)
            positions = {}
            if depth:
                for ix in sorted(rng.choice(t*n,SAMPLES,replace=False)):
                    positions.setdefault(ix//n, []).append(ix%n)
            block = {k: [] for k in ("x","post","next","p","executed_p","a","r","done","age")}
            for k in range(t):
                ix = positions.get(k)
                if ix is not None:
                    x = norm.get_original_obs()[ix].copy()
                with torch.no_grad():
                    vals, _ = critic.get_values(obs, z, masks)
                    p, lp, _ = behavior[0].get_actions(obs, z, masks)
                    anchor[0,k] = array(actors[0].evaluate_actions(obs,z,array(p),masks)[0] if use_shadow and active else lp)
                mid, reward, done, infos = norm.step(array(p).ravel())
                if np.any(reward) or np.any(done) or any(i["physical_increment"] for i in infos):
                    raise RuntimeError("选站准备不能推进物理分钟或产生奖励")
                if ix is not None:
                    post = norm.get_original_obs()[ix].copy()
                with torch.no_grad():
                    a, la, _ = behavior[1].get_actions(mid, z, masks)
                    anchor[1,k] = array(actors[1].evaluate_actions(mid,z,array(a),masks)[0] if use_shadow and active else la)
                nxt, r, done, infos = norm.step(array(a).ravel())
                if any(info["physical_increment"] != 1 for info in infos):
                    raise RuntimeError("真实步计数不符")
                nm = (~done).astype(np.float32)[:,None]
                bad = np.asarray([[0. if i.get("TimeLimit.truncated") else 1.] for i in infos], np.float32)
                if active:
                    for j,i in enumerate(infos):
                        if done[j] and not bad[j,0]:
                            with torch.no_grad():
                                end, _ = critic.get_values(np.asarray(i["terminal_observation"])[None],z[:1],ones[:1])
                            ends[k,j] = vn.denormalize(array(end))[0]
                bs[0].obs[k], bs[1].obs[k], cb.share_obs[k] = obs,mid,obs
                bs[0].masks[k], bs[1].masks[k] = masks,masks
                for b,act,log in zip(bs,(p,a),(lp,la)):
                    b.insert(nxt,z,array(act),array(log),nm,ones)
                cb.insert(nxt,z,array(vals),r[:,None],nm,bad)
                if ix is not None:
                    ps = np.argmax(post[:,-7:-4],axis=1)
                    for key,value in zip(block,(x,post,norm.get_original_obs()[ix],array(p).ravel()[ix],
                                                 ps,array(a).ravel()[ix],r[ix],done[ix],age[ix])):
                        block[key].append(value.copy())
                age = np.where(done,0,age+1)
                obs,masks = nxt,nm
                real += n
                if (k+1)%256 == 0:
                    live("collect_real")
            with torch.no_grad():
                v_next,_ = critic.get_values(obs,z,masks)
            if depth:
                ring = (ring+[{k:np.concatenate(v) for k,v in block.items()}])[-RING:]
            ix = np.linspace(0,n*t-1,256,dtype=int)
            if active:
                old_logits, shadow_logits = probe_logits(actors,bs,torch,ix), probe_logits(shadow[0],bs,torch,ix)
            for actor in actors:
                actor.prep_training()
            critic.prep_training()
            if active:
                cb.value_preds[-1] = array(v_next)
                vals = vn.denormalize(cb.value_preds)
                cont = cb.masks[1:].astype(bool)
                discount = args["gamma"]*(cont|(~cb.bad_masks[1:].astype(bool)))
                bootstrap = np.where(np.isfinite(ends),ends,vals[1:])
                gain = (anchor-np.stack([b.action_log_probs for b in bs])).sum(axis=0)
                cb.returns[:-1],adv,ratio = vtrace(cb.rewards,vals[:-1],bootstrap,cont,discount,gain,enabled=use_vtrace)
                order = torch.randperm(2).tolist()
                am,cm = offpolicy_update(actors,critic,cb,vn,bs,adv,anchor,order,np,torch)
                new = probe_logits(actors,bs,torch,ix)
                def kl(p,q):
                    return float(np.mean(np.sum(np.exp(p)*(p-q),axis=-1)))
                audit.update(actor_order=order,kl_new_vs_old=[kl(a,b) for a,b in zip(new,old_logits)],
                             kl_new_vs_shadow=[kl(a,b) for a,b in zip(new,shadow_logits)],
                             joint_rho_mean=float(ratio.mean()),joint_rho_above_one=float((ratio>1).mean()),
                             joint_rho_relative_ess=float(ratio.mean()**2/np.square(ratio.astype(np.float64)).mean()))
            else:
                cb.compute_returns(array(v_next),vn)
                am,cm = author(runner)
            updates += 1
            audit.update(real_steps=real,updates=updates,imagined_steps=imag,model_updates=fits,actor=am,critic=cm)
            save(folder/"updates"/f"{real:09d}.json",plain(audit))
            live("trained_rollout")
            if real in marks or updates%cli.save_every == 0 or stop.exists():
                ptr = capture()
            for b in bs:
                b.after_update()
            cb.after_update()
    finally:
        (norm if norm is not None else vec).close()


def eval_cell(folder, a, method, cfg, bank, marks, env_class, args, np, torch):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    final = []
    for mark in marks:
        ck = folder/"checkpoints"/f"{mark:09d}.json"
        result_path = folder/"eval"/f"{mark:09d}.json"
        if not ck.is_file():
            continue
        if result_path.is_file():
            final.extend(read(result_path))
            continue
        model_path, norm_path = verified(folder, read(ck))
        vec = DummyVecEnv([lambda: env_class(cfg, bank[0][2], "HAPPO")])
        norm = None
        try:
            norm = VecNormalize.load(str(norm_path), vec)
            norm.training = False
            norm.norm_reward = False
            device = torch.device(a.device)
            pair, _, _ = build_pair(args, norm.observation_space, norm.action_space, device)
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            if state["real_steps"] != mark:
                raise ValueError("评估权重与里程碑不符")
            for actor, weights in zip(pair, state["actors"]):
                actor.actor.load_state_dict(weights)
                actor.prep_rollout()
            del state
            zeros, ones = np.zeros((1, 1, 128), np.float32), np.ones((1, 1), np.float32)
            results = []
            for name, role, rows in bank:
                if role == "final" and mark != a.steps:
                    continue
                for seed in ((123, 124, 125) if role == "final" else (123, 124)):
                    if (a.output/"STOP_REQUESTED").exists():
                        raise InterruptedError("评估暂停；再次运行从未完成的存档继续")
                    set_seed(seed, np, torch)
                    env = env_class(cfg, rows, "HAPPO")
                    x, _ = env.reset(seed=123)
                    while not env.env.done:
                        obs = norm.normalize_obs(x[None])
                        with torch.no_grad():
                            z, _, _ = pair[env.phase].get_actions(obs, zeros, ones)
                            action = int(z.item())
                        x, _, _, _, _ = env.step(action)
                    metrics = env.env.metrics()
                    results.append(dict(trace=name, role=role, policy_seed=seed, step=mark,
                                        metrics=metrics, control=dict(env.stat), method=method))
            save(result_path, results)
            final.extend(results)
            print(f"{folder.name}：已评估 {mark:,} 步存档，{len(results)} 个回合", flush=True)
        finally:
            (norm if norm is not None else vec).close()
    if final:
        save(folder/"EVALUATION.json", dict(rows=final, weights_unchanged=True))
    return final


def paired_summary(statuses):
    # 先在各训练种子内对齐，再跨种子汇总；回放次数不当训练种子数。
    rows = {tuple(row["cell"].split("__")): row for row in statuses}
    pairs = []
    for layer in LAYERS:
        for arm, base in [(name, "L0_HAPPO") for name in ARMS if name != "L0_HAPPO"] + [
                ("L1_FULL","L5_NO_VTRACE"), ("L5_NO_VTRACE","L4_DISCARD_SHADOW")]:
            matched = []
            for (env, method, seed), row in sorted(rows.items()):
                anchor = rows.get((env, base, seed))
                if env != layer or method != arm or not anchor:
                    continue
                if not (row["evaluated_through_final"] and anchor["evaluated_through_final"]):
                    continue
                result = {"seed": int(seed[1:])}
                for split in ("late_validation", "final"):
                    a, b = row[split], anchor[split]
                    if a["n"] != b["n"] or not a["n"]:
                        continue
                    result[split] = {key: a[key]-b[key] if a[key] is not None and b[key] is not None else None
                                     for key in ("completion", "att_min", "cutoff_cost_min")}
                matched.append(result)
            if matched:
                pairs.append(dict(layer=layer, method=arm, reference=base, n=len(matched),
                                  differences=matched, direction="当前组减参照；时间负值更好"))
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path, help="E3 的 11/11/11/24 基准 manifest 或配置")
    p.add_argument("--train-trace", required=True, type=Path)
    p.add_argument("--validation-traces", type=Path, nargs="+", help="自备验证 CSV；不传则按固定种子独立生成")
    p.add_argument("--final-traces", type=Path, nargs="+", help="自备最终 CSV；只评最终存档")
    p.add_argument("--harl-dir", type=Path, required=True, help="既有 HARL 作者源码目录")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--phase", choices=("train", "eval", "all"), default="all")
    p.add_argument("--arms", choices=tuple(ARMS), nargs="+", default=tuple(ARMS))
    p.add_argument("--layers", choices=LAYERS, nargs="+", default=tuple(LAYERS))
    p.add_argument("--seeds", type=int, nargs="+", default=(2101, 2102, 2103, 2104))
    p.add_argument("--steps", type=int, default=1_996_800, help="每格真实步数，必须是 10240 的整数倍")
    p.add_argument("--save-every", type=int, default=5, help="每隔多少轮更新保存非评估恢复点")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a = p.parse_args()
    if a.steps <= 0 or a.steps % ROLLOUT or a.save_every <= 0:
        p.error("训练长度须为正且对齐 10,240 步 rollout；保存间隔必须为正")

    import gymnasium as gym
    import numpy as np
    import torch
    import stable_baselines3 as sb3

    torch.set_num_threads(1)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境无可用 CUDA；如需 CPU，请显式传 --device cpu")
    env_class = build_gym(gym, np)
    joint_args, update, vendor_files = harl_vendor(a.harl_dir.resolve(), np, torch)
    configs = {layer: config(a.config, layer) for layer in LAYERS}
    basis = configs["E1"]
    train = trace(basis, a.train_trace)
    bank = []
    for role, paths, base_seed in (("validation", a.validation_traces, 4301), ("final", a.final_traces, 8301)):
        if paths:
            bank.extend((f"{role}_{i+1}_{x.stem}", role, trace(basis, x)) for i, x in enumerate(paths))
        else:
            bank.extend((f"{role}_{s}", role, derived_trace(train, basis, s)) for s in range(base_seed, base_seed+4))
    def trace_hash(rows):
        return hashlib.sha256(json.dumps([asdict(q) for q in rows], sort_keys=True).encode()).hexdigest()
    def demand_signature(rows):
        # 比较实际需求，防止改文件名或请求编号后重复使用同一轨迹。
        return tuple(sorted((q.time_min, q.origin, q.destination) for q in rows))
    signatures = [demand_signature(train), *(demand_signature(rows) for _, _, rows in bank)]
    if len(set(signatures)) != len(signatures):
        raise ValueError("训练、验证和最终评估不可复用相同需求轨迹")
    marks = sorted({math.ceil(a.steps*f/ROLLOUT)*ROLLOUT for f in (.1, .25, .5, .8, .85, .9, .95, 1.)})
    tasks = []
    for seed in dict.fromkeys(a.seeds):
        methods = list(dict.fromkeys(a.arms))
        random.Random(14112027 + seed).shuffle(methods)
        for layer in dict.fromkeys(a.layers):
            tasks.extend((layer, m, seed) for m in methods)
    contract = dict(schema="corridor_lookahead_happo_l0l5_v1", arms=ARMS,
                    shadow=dict(horizon=SHADOW_H, trajectories=SHADOW_N, warmup=WARM, replay_blocks=RING,
                                samples_per_rollout=SAMPLES, fit_updates=48, fit_batch=256, rho_clip=1, c_clip=1), environment_sha=digest(Path(__file__).with_name("corridor_env.py")),
                    entry_sha=digest(Path(__file__)), configs={k: asdict(v) for k, v in configs.items()},
                    steps_per_cell=a.steps, rollout=ROLLOUT, dev_marks=marks, train_trace_sha256=trace_hash(train),
                    bank=[dict(name=n, role=r, sha256=trace_hash(t)) for n, r, t in bank],
                    torch=torch.__version__, sb3=sb3.__version__, gymnasium=gym.__version__, numpy=np.__version__,
                    boundary="全新训练；不加载旧权重；车队和约束不变。奖励为门到门生成未完成人时。")
    failures = 0
    with lock_run(a.output):
        path = a.output/"PROTOCOL.json"
        # JSON 解析会把元组写成数组；按序列化后的内容比较。
        canonical = json.loads(json.dumps(contract, ensure_ascii=False))
        if path.is_file() and read(path) != canonical:
            raise ValueError("当前配置、版本或需求与既有输出不一致，请使用新输出路径")
        save(path, canonical)
        save(a.output/"DEMAND_BANK.json", {
            "train": [asdict(q) for q in train],
            **{n: [asdict(q) for q in rows] for n, _, rows in bank}})
        print(f"{len(tasks)} 格；每格 {a.steps:,} 真实步；训练预算 {len(tasks)*a.steps:,} 步", flush=True)
        for layer, method, seed in tasks:
            if (a.output/"STOP_REQUESTED").exists():
                print("已暂停；移除 STOP_REQUESTED 后重跑原命令。", flush=True)
                break
            folder = a.output/"cells"/f"{layer}__{method}__s{seed}"
            meta = dict(layer=layer, method=method, seed=seed, harl_args=joint_args, harl_files=vendor_files, harl_pin=HARL_COMMIT,
                        shadow_depth=ARMS[method][0], shadow_behavior=ARMS[method][1], vtrace=ARMS[method][2])
            try:
                job = folder/"JOB.json"
                if job.is_file() and read(job) != meta:
                    raise ValueError("本格合同与恢复时不同")
                if a.phase != "eval":
                    save(job, meta)
                    train_cell(folder, a, method, seed, configs[layer], train, marks,
                               env_class, joint_args, update, gym, np, torch)
                evaluated = []
                if a.phase != "train":
                    evaluated = eval_cell(folder, a, method, configs[layer], bank, marks,
                                          env_class, joint_args, np, torch)
                    if evaluated:
                        eval_rules(a.output, layer, configs[layer], bank)
                ptr = read(folder/"RESUME.json") if (folder/"RESUME.json").exists() else {}
                real = ptr.get("real_steps", 0)
                score = [x["metrics"] for x in evaluated if x["role"] == "validation" and x["step"] >= .8*a.steps]
                final = [x["metrics"] for x in evaluated if x["role"] == "final"]
                def aggregate(rows):
                    def mean(k):
                        v = [x[k] for x in rows]
                        return sum(v)/len(v) if v and all(x is not None for x in v) else None
                    return dict(n=len(rows), completion=mean("completion"), att_min=mean("mean_att_min"),
                                cutoff_cost_min=mean("fixed_cutoff_cost_min"))
                save(folder/"STATUS.json", dict(real_steps=real, imagined_steps=ptr.get("imagined_steps",0),
                     model_updates=ptr.get("model_updates",0), trained=real == a.steps,
                     evaluated_through_final=bool(final), late_validation=aggregate(score), final=aggregate(final)))
            except (KeyboardInterrupt, InterruptedError):
                print("已停止，保留最近完整存档；可用原命令继续。", flush=True)
                break
            except Exception:
                failures += 1
                detail = traceback.format_exc()
                save(folder/"ERROR.json", dict(error=detail))
                print(f"{folder.name} 失败，继续下一格：\n{detail}", file=sys.stderr, flush=True)
        rows = [{"cell": f.parent.name, **read(f)} for f in sorted(a.output.glob("cells/*/STATUS.json"))]
        save(a.output/"SUMMARY.json", dict(cells=rows, paired_differences=paired_summary(rows),
                                           failures_this_call=failures))
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
