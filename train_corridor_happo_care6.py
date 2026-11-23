#!/usr/bin/env python3
"""冻结 HAPPO 的 CARE 式机制适配；不是原论文完整训练流程的复现。"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, replace
import hashlib
import heapq
import json
import math
from pathlib import Path
import random
import sys
import traceback

ARMS = ("CARE_FULL", "TOTAL_ONLY", "A_PLUS_D", "BRANCH_UTILITIES",
        "UNMATCHED", "NO_CALIB")
TRAINED_ARMS = ARMS[:-1]
SEEDS = (2101, 2102, 2103, 2104)
RATES = (1.0, 1.1, 1.2, 1.3, 1.4)
STEP_CONTRACT = 1_996_800
SCHEMA = "corridor_happo_care6_v2_batch"
QLEVELS = (.10, .50, .90)
QUANTILES = 3
STATIONS = ("V0", "V1", "V3")


def store(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                              default=str), encoding="utf-8")
    tmp.replace(path)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash_matches(path, expected):
    """仅允许 LF/CRLF 换行差异，不放宽真实源码差异。"""
    raw = Path(path).read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return expected in {hashlib.sha256(x).hexdigest() for x in (raw,lf,lf.replace(b"\n",b"\r\n"))}


def demand_hash(rows):
    return hashlib.sha256(json.dumps([asdict(q) for q in rows],
                                     sort_keys=True).encode()).hexdigest()


def compressed_trace(rows, pressure, np):
    if not pressure > 0:
        raise ValueError("pressure must be positive")
    from corridor_env import validate_requests
    start = min(q.time_min for q in rows)
    changed = [replace(q, time_min=start + (q.time_min-start)/pressure) for q in rows]
    return validate_requests(changed)


def checkpoint(base_root, seed):
    root = Path(base_root)
    folder = root/"cells"/f"E2__HAPPO__s{seed}"
    final = folder/"checkpoints"/f"{STEP_CONTRACT:09d}.json"
    if final.is_file():
        meta = load(final)
    elif (folder/"RESUME.json").is_file():
        meta = load(folder/"RESUME.json")
    else:
        raise FileNotFoundError(f"HAPPO checkpoint absent: {folder}")
    if int(meta.get("real_steps", -1)) != STEP_CONTRACT:
        raise RuntimeError(f"HAPPO seed {seed} not at {STEP_CONTRACT:,} real steps")
    if any(not isinstance(meta.get(k),str) or Path(meta[k]).name != meta[k] for k in ("model","norm")):
        raise ValueError("存档文件名异常")
    model, norm = folder/meta["model"], folder/meta["norm"]
    if not model.is_file() or not norm.is_file():
        raise FileNotFoundError(f"HAPPO weights/norm missing in {folder}")
    if sha(model) != meta["model_sha"] or sha(norm) != meta["norm_sha"]:
        raise RuntimeError(f"HAPPO checkpoint SHA verification failed: {folder}")
    return model, norm, folder


def source_check(base_root, cfg, train_rows, repo):
    protocol = load(Path(base_root)/"PROTOCOL.json")
    if protocol.get("schema") != "corridor_legacy_interface_v2":
        raise ValueError("Base source must be original train_eval_corridor_uagmc_happo_mappo.py")
    if int(protocol["steps_per_cell"]) != STEP_CONTRACT:
        raise ValueError("Base HAPPO training steps differ from frozen contract")
    if protocol.get("train_trace_sha256") != demand_hash(train_rows):
        raise ValueError("Training trace differs from baseline HAPPO trace")
    for field, path in (("environment_sha", repo/"corridor_env.py"),
                        ("dispatch_sha", repo/"corridor_dispatch.py"),
                        ("entry_sha",repo/"train_eval_corridor_uagmc_happo_mappo.py")):
        if not source_hash_matches(path, protocol[field]):
            raise ValueError(f"Source physics mismatch: {path}; do NOT mix checkpoints")
    if protocol["configs"]["E2"] != json.loads(json.dumps(asdict(cfg))):
        raise ValueError("E2 configuration differs from frozen HAPPO source")
    return protocol


def clone_known_only(wrapper):
    """冻结 HAPPO 的 CARE 式机制适配；不是原论文完整训练流程的复现。"""
    e = wrapper.env
    events, rows = e._events, wrapper.rows
    try:
        e._events = [x for x in events if x[2].__name__ != "_reveal"]
        heapq.heapify(e._events)
        wrapper.rows = ()
        branch = copy.deepcopy(wrapper)
    finally:
        e._events, wrapper.rows = events, rows
    branch.env.request_count = len(branch.env.passengers) + 1  # sentinel; no phantom passenger
    if any(fn.__name__ == "_reveal" for _, _, fn, _ in branch.env._events):
        raise AssertionError("future requests leaked into snapshot")
    return branch


class DemandModel:
    """冻结 HAPPO 的 CARE 式机制适配；不是原论文完整训练流程的复现。"""
    def __init__(self, rows, duration=300., width=30.):
        self.duration = float(duration)
        self.width = float(width)
        self.nbins = max(1, int(math.ceil(self.duration/self.width)))
        self.counts = [0]*self.nbins
        self.origins = []
        self.dests = []
        for p in rows:
            i = min(self.nbins-1, max(0, int(p.time_min/self.width)))
            self.counts[i] += 1
            self.origins.append(tuple(p.origin))
            self.dests.append(tuple(p.destination))
        self.per_min = [n/self.width for n in self.counts]
        if not self.origins:
            raise ValueError("Training trace empty")

    def attach(self, wrapper, seed, horizon, np):
        from corridor_env import Request
        e = wrapper.env
        rng = np.random.default_rng(seed)
        now = e.now*e.cfg.tick_min
        n_synth = 0
        for dt in range(1, horizon+1):
            t = now+dt*e.cfg.decision_min
            if t >= e.cfg.cutoff_min:
                break
            i = min(self.nbins-1, max(0, int(t/self.width)))
            count = int(rng.poisson(self.per_min[i]*e.cfg.decision_min))
            for z in range(count):
                j = int(rng.integers(len(self.origins)))
                m = int(rng.integers(len(self.dests)))
                req = Request(f"_care_s{seed}_{dt}_{z}", t,
                              self.origins[j], self.dests[m])
                e._push(e._ticks(t), e._reveal, req)
                n_synth += 1
        e.request_count = len(e.passengers) + n_synth + 1
        return n_synth


def load_model_context(a, seed, cfg, rows, legacy, np, torch, gym):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    model_path, norm_path, _ = checkpoint(a.base_output, seed)
    legacy_maker = legacy.build_gym(gym, np)
    Base = type(legacy_maker(cfg, rows, "HAPPO"))
    class CapturedPhase(Base):
        def reset(self, **kwargs):
            result = super().reset(**kwargs)
            self.care_obs = result[0]
            return result
        def step(self, action):
            result = super().step(action)
            self.care_obs = result[0]
            return result
    def env_maker(c, r, method):
        if method != "HAPPO":
            raise ValueError("CARE requires HAPPO backbone")
        return CapturedPhase(c, r, method)
    vec = DummyVecEnv([lambda: env_maker(cfg, rows, "HAPPO")])
    norm = VecNormalize.load(str(norm_path), vec)
    norm.training = False
    norm.norm_reward = False
    device = torch.device(a.device)
    args, _, vendor = legacy.harl_vendor(Path(a.harl_dir).resolve(), np, torch)
    job=load(Path(a.base_output)/"cells"/f"E2__HAPPO__s{seed}"/"JOB.json")
    if job.get("harl_args")!=args or job.get("harl_files")!=vendor:
        raise ValueError("HARL 参数或作者源码与训练存档不符")
    pair, critic, vn = legacy.build_pair("HAPPO", args,
                                         norm.observation_space, gym.spaces.Discrete(3), device)
    weights = torch.load(model_path, map_location="cpu", weights_only=False)
    if weights["real_steps"] != STEP_CONTRACT:
        raise ValueError("Model and baseline step count mismatch")
    for actor, state in zip(pair, weights["actors"]):
        actor.actor.load_state_dict(state)
        actor.prep_rollout()
    critic.critic.load_state_dict(weights["critic"])
    critic.prep_rollout()
    # CUDA 训练时统计张量可能未被注册；原训练器另存了完整统计，按原恢复语义加载。
    stats = weights.get("value_statistics")
    required = ("running_mean", "running_mean_sq", "debiasing_term")
    if stats is not None:
        if not isinstance(stats, dict) or set(stats) != set(required):
            raise ValueError("价值归一化统计字段不完整，拒绝使用默认统计")
        with torch.no_grad():
            for key in required:
                target = getattr(vn, key)
                source = torch.as_tensor(stats[key], device=target.device, dtype=target.dtype)
                if source.shape != target.shape or not torch.isfinite(source).all():
                    raise ValueError(f"价值归一化统计形状或数值异常：{key}")
                target.copy_(source)
    else:
        vn.load_state_dict(weights["value_norm"])
    z = np.zeros((1, 1, 128), dtype=np.float32)
    masks = np.ones((1, 1), dtype=np.float32)
    return dict(norm=norm, actors=pair, critic=critic, vn=vn,
                device=device, env_maker=env_maker, zeros=z, masks=masks)


def normal_obs(ctx, raw, np):
    return ctx["norm"].normalize_obs(np.asarray(raw, dtype=np.float32)[None])[0]


def actor_prob(ctx, role, raw, np, torch):
    a = ctx["actors"][role]
    x = normal_obs(ctx, raw, np)[None]
    with torch.no_grad():
        placeholder = torch.zeros((1, 1), dtype=torch.long, device=ctx["device"])
        distribution = a.evaluate_actions(x, ctx["zeros"], placeholder,
                                          ctx["masks"], None, None)[2]
        p = distribution.probs[0].detach().cpu().numpy().reshape(-1)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError(f"Unexpected HAPPO action probs: {p}")
    return p / p.sum()


def nominal(ctx, role, raw, seed, t, np, torch):
    probs = actor_prob(ctx, role, raw, np, torch)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(t), int(role)]))
    u = float(rng.random())
    return min(2, int(np.searchsorted(np.cumsum(probs), u, side="right")))


def valid_actions(wrapper):
    if wrapper.phase == 0:
        return list(range(3)) if wrapper.focal() is not None else []
    return [i for i, x in enumerate(STATIONS) if x in wrapper.env.targets()]


def roll_branch(state, ctx, demand, seed, target_role, first,
                replay_peer, horizon, np, torch, *, record=False):
    """冻结 HAPPO 的 CARE 式机制适配；不是原论文完整训练流程的复现。"""
    env = copy.deepcopy(state)
    demand.attach(env, seed, horizon, np)
    peer_actions = {}
    elapsed = 0
    power = 1.0
    utility = 0.0
    trajectory_one_min = None
    start_now = env.env.now
    bad_projection = 0
    prev_projection = int(env.stat.get("air_priority_projection", 0))
    steps = 0
    while elapsed < horizon and not env.env.done:
        role = env.phase
        key = (env.env.now-start_now, role)
        raw = env.care_obs
        if steps == 0:
            action = int(first)
        elif replay_peer is not None and role != target_role and key in replay_peer:
            action = replay_peer[key]
        else:
            action = nominal(ctx, role, raw, seed, env.env.now, np, torch)
        if record and role != target_role:
            peer_actions[key] = action
        _, reward, _, _, info = env.step(action)
        if role == 1:
            projection = int(env.stat.get("air_priority_projection", 0))
            bad_projection += max(0, projection-prev_projection)
            prev_projection = projection
        if info.get("physical_increment", -1) == 1:
            utility += power * float(reward)
            power *= .99
            elapsed += 1
            if trajectory_one_min is None:
                trajectory_one_min = normal_obs(ctx, env.care_obs, np)
        else:
            if info.get("physical_increment") != 0:
                raise RuntimeError("Physical-minute contract broken in CARE branch")
        steps += 1
        if steps > 2*horizon+2:
            raise RuntimeError("CARE branch got stuck in nonphysical phase")
    if not env.env.done and elapsed:
        with torch.no_grad():
            raw = env.care_obs
            vals, _ = ctx["critic"].get_values(normal_obs(ctx, raw, np)[None],
                                                 ctx["zeros"], ctx["masks"])
            # HARL 原版返回 NumPy 数组，同时兼容返回张量的实现。
            restored = ctx["vn"].denormalize(vals)
            if torch.is_tensor(restored):
                restored = restored.detach().cpu().numpy()
            v = float(np.asarray(restored).reshape(-1)[0])
        utility += power * v
    if trajectory_one_min is None:
        trajectory_one_min = normal_obs(ctx, env.care_obs, np)
    max_queue = max(map(len, env.env.queue.values()), default=0)
    return utility, peer_actions, trajectory_one_min, (bad_projection, max_queue)


def sample_family(wrapper, ctx, demand, seed, horizon, np, torch):
    """冻结 HAPPO 的 CARE 式机制适配；不是原论文完整训练流程的复现。"""
    from corridor_env import Config
    if wrapper.env.done:
        return None, 0
    options = valid_actions(wrapper)
    if not options:
        return None, 0
    role = wrapper.phase
    raw = wrapper.care_obs
    x = normal_obs(ctx, raw, np)
    formal = nominal(ctx, role, raw, seed, wrapper.env.now, np, torch)
    if role == 1 and formal not in options:
        formal = max(options, key=lambda i: actor_prob(ctx, role, raw, np, torch)[i])
    if len(options) < 2:
        return None, 0
    start_queue = max(map(len, wrapper.env.queue.values()), default=0)
    snap = clone_known_only(wrapper)
    base_seed = (int(seed)*9973 + wrapper.env.now*53 + role*7) % (2**31-1)
    y0r, peer_trace, next_x, _ = roll_branch(snap, ctx, demand, base_seed, role,
                                              formal, None, horizon, np, torch, record=True)
    y0p, _, _, _ = roll_branch(snap, ctx, demand, base_seed, role,
                               formal, peer_trace, horizon, np, torch)
    if abs(y0r-y0p) > 1e-5 * max(1., abs(y0r)):
        raise RuntimeError(f"Nominal reactive/replay mismatch: {y0r}, {y0p}")
    samples = []
    simulated = 2*horizon
    for candidate in options:
        if candidate == formal:
            continue
        ykr, _, _, diag = roll_branch(snap, ctx, demand, base_seed, role,
                                       candidate, None, horizon, np, torch)
        ykp, _, _, _ = roll_branch(snap, ctx, demand, base_seed, role,
                                    candidate, peer_trace, horizon, np, torch)
        simulated += 2*horizon
        a, d = ykr-y0r, ykp-y0p
        samples.append(dict(x=x, next_x=next_x, role=role, candidate=candidate,
                            formal=formal, A=a, D=d, R=a-d,
                            y0r=y0r, ykr=ykr, y0p=y0p, ykp=ykp,
                            adverse=float(diag[0]>0 or diag[1]-start_queue>=5),
                            group_seed=int(seed)))
    return samples, simulated


def collect(a, seed, cfg, train_rows, ctx, legacy, np, torch):
    """冻结 HAPPO 的 CARE 式机制适配；不是原论文完整训练流程的复现。"""
    path = a.output/"data"/f"s{seed}.npz"
    meta_path = path.with_suffix(".json")
    if path.is_file():
        meta = load(meta_path)
        return np.load(path, allow_pickle=False), meta
    demand = DemandModel(train_rows, duration=cfg.cutoff_min)
    maker = ctx["env_maker"]
    train, cal = [], []
    desired = a.snapshots + a.calibration_snapshots
    branch_minutes = 0
    episodes = 0
    limits = {"train": a.snapshots, "cal": a.calibration_snapshots}
    while len(train)<limits["train"] or len(cal)<limits["cal"]:
        split = "cal" if episodes%5==4 else "train"
        if len(cal) >= limits["cal"]:
            split="train"
        if len(train) >= limits["train"]:
            split="cal"
        source = legacy.derived_trace(train_rows, cfg, 100000+seed*100+episodes)
        env = maker(cfg, source, "HAPPO")
        env.reset(seed=episodes+seed)
        per_episode=0
        while not env.env.done:
            role, now = env.phase, env.env.now
            options = valid_actions(env)
            sampling = np.random.default_rng(np.random.SeedSequence([seed,episodes,now,role,3]))
            remaining = limits[split]-len(cal if split=="cal" else train)
            if (len(options)>=2 and remaining>0 and per_episode < a.max_per_episode
                    and float(sampling.random()) < a.snapshot_probability):
                rows, cost = sample_family(env, ctx, demand,
                                           seed+episodes*123457, a.branch_horizon, np, torch)
                if rows:
                    dst = train if split=="train" else cal
                    dst.extend(rows[:remaining])
                    branch_minutes += cost
                    per_episode += 1
            raw = env.care_obs
            act = nominal(ctx, role, raw, seed+episodes*10000, now, np, torch)
            env.step(act)
        episodes += 1
        if episodes >= a.max_collect_episodes:
            raise RuntimeError(f"Insufficient CARE snapshots: train={len(train)} cal={len(cal)}; "
                               "increase --max-collect-episodes or --snapshot-probability")
    rows = train+cal
    keys = ["x", "next_x", "role", "candidate", "formal", "A", "D", "R",
            "y0r", "ykr", "y0p", "ykp", "adverse", "group_seed"]
    arrays={k:np.asarray([row[k] for row in rows], dtype=np.float32 if k not in
                          ("role", "candidate", "formal", "group_seed") else np.int64)
            for k in keys}
    arrays["split"] = np.asarray([0]*len(train)+[1]*len(cal), dtype=np.uint8)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path, **arrays)
    meta = dict(seed=seed, train_samples=len(train), cal_samples=len(cal), episodes=episodes,
                approx_counterfactual_physical_minutes=branch_minutes,
                demand_model="time_binned_30min_Poisson_OD_train_only",
                reactive_replay_horizon=a.branch_horizon,
                future_real_events_excluded=True)
    store(meta_path,meta)
    print(f"seed={seed} CARE dataset train {len(train)} cal {len(cal)} "
          f"virtual-minutes~{branch_minutes:,}",flush=True)
    return np.load(path,allow_pickle=False),meta


def architecture(dim, torch):
    from torch import nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            if dim < 177:
                raise ValueError("E2 legacy obs must contain 6x28 history")
            self.gru=nn.GRU(28,64,batch_first=True)
            self.body=nn.Sequential(nn.Linear(dim-168,128),nn.ReLU(),
                                    nn.Linear(128,64),nn.ReLU())
            self.fuse=nn.Sequential(nn.Linear(128,128),nn.ReLU(),
                                    nn.Linear(128,64),nn.ReLU())
            self.predict=nn.Linear(64,36)  # next latest-frame(28)+phase-tail(8)

        def forward(self,x):
            _,h=self.gru(x[:,:168].reshape(-1,6,28))
            z=self.body(x[:,168:])
            return self.fuse(torch.cat((h[0],z),-1))

    class Head(nn.Module):
        def __init__(self, variant):
            super().__init__()
            self.variant=variant
            output={"CARE_FULL":3,"TOTAL_ONLY":1,"A_PLUS_D":2,
                    "BRANCH_UTILITIES":4,"UNMATCHED":3}[variant]
            self.output=output
            self.net=nn.Sequential(nn.Linear(64+3+2+3,128),nn.ReLU(),
                                   nn.Linear(128,128),nn.ReLU(),
                                   nn.Linear(128,output*QUANTILES))

        def forward(self,z,act,role,formal):
            import torch.nn.functional as F
            features=torch.cat((z,F.one_hot(act.long(),3).float(),
                                F.one_hot(role.long(),2).float(),
                                F.one_hot(formal.long(),3).float()),-1)
            return self.net(features).reshape(-1,self.output,QUANTILES)
    return Encoder(),Head


def targets(data, variant, indices, device, torch, np):
    keys={"CARE_FULL":("A","D","R"),"TOTAL_ONLY":("A",),
          "A_PLUS_D":("A","D"),
          "BRANCH_UTILITIES":("y0r","ykr","y0p","ykp"),
          "UNMATCHED":("A","D","R")}[variant]
    ys = np.stack([data[k][indices] for k in keys],axis=1).astype(np.float32)
    return torch.from_numpy(ys/100.).to(device)


def pinball(pred, target, torch):
    qs=torch.as_tensor(QLEVELS,device=pred.device).reshape(1,1,-1)
    err=target.unsqueeze(-1)-pred
    return torch.maximum(qs*err,(qs-1.)*err).mean()


def train_models(a, seed, data, np, torch):
    from torch import nn
    import torch.nn.functional as F
    root=a.output/"models"/f"s{seed}"
    root.mkdir(parents=True,exist_ok=True)
    device=torch.device(a.device)
    x=data["x"]
    ix=np.where(data["split"]==0)[0]
    rng=np.random.default_rng(seed+9499)
    encoder,Head=architecture(x.shape[1],torch)
    encoder.to(device)
    encpath=root/"encoder.pt"
    if encpath.is_file():
        encoder.load_state_dict(torch.load(encpath,map_location=device,weights_only=True))
    else:
        opt=torch.optim.Adam(encoder.parameters(),lr=a.lr)
        enc_x=torch.as_tensor(x[ix],dtype=torch.float32,device=device)
        nxs=torch.as_tensor(data["next_x"][ix],dtype=torch.float32,device=device)
        goal=torch.cat((nxs[:,140:168],nxs[:,-8:]),-1)
        encoder.train()
        for epoch in range(a.encoder_epochs):
            perm = rng.permutation(len(ix))
            for cut in range(0,len(ix),a.batch_size):
                ids=perm[cut:cut+a.batch_size]
                z=encoder(enc_x[ids])
                loss=F.smooth_l1_loss(encoder.predict(z),goal[ids])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(encoder.parameters(),1.)
                opt.step()
        torch.save(encoder.state_dict(),encpath)
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad_(False)
    with torch.no_grad():
        allz=encoder(torch.as_tensor(x,dtype=torch.float32,device=device)).detach()
    adverse=torch.as_tensor(data["adverse"],dtype=torch.float32,device=device)
    class Risk(nn.Module):
        def __init__(self):
            super().__init__()
            self.f=nn.Sequential(nn.Linear(64+3+2,64),nn.ReLU(),nn.Linear(64,1))
        def forward(self,z,act,role):
            return self.f(torch.cat((z,F.one_hot(act.long(),3).float(),
                                     F.one_hot(role.long(),2).float()),-1)).flatten()
    risk=Risk().to(device)
    risk_file=root/"risk.pt"
    act=torch.as_tensor(data["candidate"],dtype=torch.long,device=device)
    role=torch.as_tensor(data["role"],dtype=torch.long,device=device)
    if risk_file.is_file():
        risk.load_state_dict(torch.load(risk_file,map_location=device,weights_only=True))
    else:
        opt=torch.optim.Adam(risk.parameters(),lr=a.lr)
        for epoch in range(a.head_epochs):
            for cut in range(0,len(ix),a.batch_size):
                ids=rng.choice(ix,size=min(a.batch_size,len(ix)),replace=False)
                logit=risk(allz[ids],act[ids],role[ids])
                loss=F.binary_cross_entropy_with_logits(logit,adverse[ids])
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        torch.save(risk.state_dict(),risk_file)
    risk.eval()
    for arm in TRAINED_ARMS:
        if arm != ("CARE_FULL" if a.current_arm=="NO_CALIB" else a.current_arm):
            continue
        path=root/f"{arm}.pt"
        if path.is_file():
            continue
        head=Head(arm).to(device)
        opt=torch.optim.Adam(head.parameters(),lr=a.lr)
        mis=None
        if arm=="UNMATCHED":
            mis=np.asarray(data["D"],dtype=np.float32).copy()
            for r in (0,1):
                ids=np.where((data["role"]==r)&(data["split"]==0))[0]
                mis[ids]=mis[rng.permutation(ids)]
        head.train()
        for ep in range(a.head_epochs):
            for cut in range(0,len(ix),a.batch_size):
                ids=rng.choice(ix,size=min(a.batch_size,len(ix)),replace=False)
                y=targets(data,arm,ids,device,torch,np)
                if mis is not None:
                    y[:,2]=torch.as_tensor((data["A"][ids]-mis[ids])/100.,
                                           dtype=torch.float32,device=device)
                formal=torch.as_tensor(data["formal"][ids],dtype=torch.long,device=device)
                pred=head(allz[ids],act[ids],role[ids],formal)
                loss=pinball(pred,y,torch)
                if arm in ("CARE_FULL","UNMATCHED"):
                    loss=loss+.03*F.smooth_l1_loss(pred[:,0,1],pred[:,1,1]+pred[:,2,1])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(),1.)
                opt.step()
        torch.save(head.state_dict(),path)
    return dict(encoder_file=str(encpath),models=len(TRAINED_ARMS),risk_file=str(risk_file))


def scorer(a, seed, ctx, np, torch):
    root=a.output/"models"/f"s{seed}"
    enc,Head=architecture(ctx["norm"].observation_space.shape[0],torch)
    dev=torch.device(a.device)
    enc.to(dev);enc.load_state_dict(torch.load(root/"encoder.pt",map_location=dev,weights_only=True));enc.eval()
    heads={}
    for arm in TRAINED_ARMS:
        if arm != ("CARE_FULL" if a.current_arm=="NO_CALIB" else a.current_arm):
            continue
        model=Head(arm).to(dev)
        model.load_state_dict(torch.load(root/f"{arm}.pt",map_location=dev,weights_only=True))
        model.eval();heads[arm]=model
    if a.current_arm=="NO_CALIB":
        heads={"NO_CALIB":heads["CARE_FULL"]}
    from torch import nn
    class Risk(nn.Module):
        def __init__(self):
            super().__init__()
            self.f=nn.Sequential(nn.Linear(64+3+2,64),nn.ReLU(),nn.Linear(64,1))
        def forward(self,z,act,role):
            import torch.nn.functional as F
            return self.f(torch.cat((z,F.one_hot(act.long(),3).float(),
                                     F.one_hot(role.long(),2).float()),-1)).flatten()
    risk=Risk().to(dev)
    risk.load_state_dict(torch.load(root/"risk.pt",map_location=dev,weights_only=True))
    risk.eval()
    return enc,heads,risk


def predict_low(encoder,head,x,role,actions,torch,np,formal):
    device=next(encoder.parameters()).device
    z=encoder(torch.as_tensor(np.repeat(x[None],len(actions),axis=0),
                              dtype=torch.float32,device=device))
    cand=torch.as_tensor(actions,dtype=torch.long,device=device)
    who=torch.full_like(cand,int(role))
    out=head(z,cand,who,torch.full_like(cand,int(formal))).detach().cpu().numpy()
    if head.variant=="BRANCH_UTILITIES":
        return (out[:,1,0]-out[:,0,2])*100.
    return out[:,0,0]*100.


def family_calibration(a, seed, data, encoder, heads, np, torch):
    target=a.output/"models"/f"s{seed}"/f"CALIBRATION_{a.current_arm}.json"
    if target.is_file():
        return load(target)
    ids=np.where(data["split"]==1)[0]
    families={}
    for idx in ids:
        key=(int(data["group_seed"][idx]), int(data["role"][idx]),
             np.asarray(data["x"][idx],dtype=np.float32).tobytes())
        families.setdefault(key,[]).append(int(idx))
    out={}
    for arm,head in heads.items():
        if arm=="NO_CALIB":
            out[arm]={"threshold":0.,"n_families":len(families),"alpha":a.alpha}
            continue
        errors=[]
        with torch.no_grad():
            for inds in families.values():
                actions=data["candidate"][inds].astype(int).tolist()
                x=np.asarray(data["x"][inds[0]],dtype=np.float32)
                low=predict_low(encoder,head,x,int(data["role"][inds[0]]),
                                actions,torch,np,int(data["formal"][inds[0]]))
                residual=max(float(p-y) for p,y in zip(low,data["A"][inds]))
                errors.append(residual)
        n=len(errors)
        k=int(math.ceil((n+1)*(1-a.alpha)))-1
        threshold=float(sorted(errors)[k]) if 0<=k<n else float("inf")
        if not math.isfinite(threshold):
            raise RuntimeError("Insufficient independent calibration families")
        out[arm]={"threshold":threshold,"n_families":n,"alpha":a.alpha}
    store(target,out)
    return out



def calibration_diagnostics(a,seed,data,encoder,heads,np,torch):
    """冻结 HAPPO 的 CARE 式机制适配；不是原论文完整训练流程的复现。"""
    ids=np.where(data["split"]==1)[0]
    groups={}
    for i in ids:
        key=(int(data["group_seed"][i]),int(data["role"][i]),
             np.asarray(data["x"][i],dtype=np.float32).tobytes())
        groups.setdefault(key,[]).append(int(i))
    out={}
    for arm,head in heads.items():
        pred_A=[]; target_A=[]; pred_R=[]; target_R=[]
        rank_correct=0; ranks=0; regrets=[]
        with torch.no_grad():
            for inds in groups.values():
                xs=np.asarray(data["x"][inds[0]],dtype=np.float32)
                role=int(data["role"][inds[0]])
                acts=data["candidate"][inds].astype(int).tolist()
                dev=next(encoder.parameters()).device
                z=encoder(torch.as_tensor(np.repeat(xs[None],len(inds),axis=0),
                                          dtype=torch.float32,device=dev))
                q=head(z,torch.as_tensor(acts,dtype=torch.long,device=dev),
                       torch.full((len(inds),),role,dtype=torch.long,device=dev),
                       torch.full((len(inds),),int(data["formal"][inds[0]]),dtype=torch.long,device=dev))
                q=q.cpu().numpy()
                if head.variant=="BRANCH_UTILITIES":
                    pa=100*(q[:,1,1]-q[:,0,1])
                    pr=100*((q[:,1,1]-q[:,0,1])-(q[:,3,1]-q[:,2,1]))
                else:
                    pa=100*q[:,0,1]
                    pr=100*q[:,2,1] if head.variant in ("CARE_FULL","UNMATCHED") else None
                real=data["A"][inds]
                pred_A.extend(pa.tolist());target_A.extend(real.tolist())
                if pr is not None:
                    pred_R.extend(pr.tolist());target_R.extend(data["R"][inds].tolist())
                if len(inds)>1 and max(real)-min(real)>1e-8:
                    rank_correct+=int(np.argmax(pa)==np.argmax(real));ranks+=1
                best=float(max(0.,max(real)))
                choice=int(np.argmax(pa)) if max(pa)>0 else -1
                achieved=float(real[choice]) if choice>=0 else 0.
                regrets.append(max(0.,best-achieved))
        out[arm]=dict(n_candidates=len(pred_A),n_families=len(groups),
                      total_advantage_mae_min=float(np.mean(np.abs(np.asarray(pred_A)-target_A))),
                      response_mae_min=(float(np.mean(np.abs(np.asarray(pred_R)-target_R)))
                                         if pred_R else None),
                      alternative_rank_agreement=(rank_correct/ranks if ranks else None),
                      mean_oracle_regret_min=float(np.mean(regrets)) if regrets else None,
                      split="calibration_only_not_final_test")
    store(a.output/"models"/f"s{seed}"/f"CALIB_DIAGNOSTICS_{a.current_arm}.json",out)
    return out


def test_episode(cfg, rows, ctx, encoder,heads,risk,calib,a,seed,trialseed,np,torch):
    env=ctx["env_maker"](cfg,rows,"HAPPO")
    env.reset(seed=trialseed)
    stats={arm:dict(override=0,eligible=0,errors=0) for arm in ARMS}
    results={}
    for arm in (("HAPPO_BASE","HAPPO_LEGAL_ONLY",a.current_arm) if a.current_arm=="CARE_FULL" else (a.current_arm,)):
        env=ctx["env_maker"](cfg,rows,"HAPPO")
        env.reset(seed=trialseed)
        overrides=0
        eligible=0
        steps=0
        while not env.env.done:
            role=env.phase
            raw=env.care_obs
            nominal_action=nominal(ctx,role,raw,trialseed,env.env.now,np,torch)
            action=nominal_action
            options=valid_actions(env)
            if arm != "HAPPO_BASE" and role==1 and options and action not in options:
                probs=actor_prob(ctx,role,raw,np,torch)
                action=max(options,key=lambda i:probs[i])
                nominal_action=action
            if arm not in ("HAPPO_BASE","HAPPO_LEGAL_ONLY") and len(options)>=2:
                alternatives=[x for x in options if x!=nominal_action]
                if len(alternatives)>0:
                    eligible+=1
                    with torch.no_grad():
                        low=predict_low(encoder,heads[arm],normal_obs(ctx,raw,np),
                                        role,alternatives,torch,np,nominal_action)
                    threshold=calib[arm]["threshold"]
                    adjusted=low-threshold
                    index=int(np.argmax(adjusted))
                    if float(adjusted[index]) > a.margin:
                        import torch.nn.functional as F
                        x=torch.as_tensor(normal_obs(ctx,raw,np)[None],dtype=torch.float32,
                                          device=ctx["device"])
                        with torch.no_grad():
                            z=encoder(x)
                            proposed=torch.tensor([alternatives[index]],device=ctx["device"])
                            who=torch.tensor([role],device=ctx["device"])
                            risk_p=float(torch.sigmoid(risk(z,proposed,who)).item())
                        if risk_p <= a.max_adverse_probability:
                            action=alternatives[index]
                            overrides+=1
            if role == 0 and sum(q.state=="route" for q in env.env.passengers.values()) > 1:
                # 同一分钟逐个处理当前已揭示乘客；最后一位沿用原 P→A 阶段转换。
                station=STATIONS[int(action)]
                env.env._route(env.focal(),station)
                env.stat["route_"+station] += 1
                env.care_obs=env.observation()
            else:
                env.step(action)
            steps+=1
            if steps>2*env.env.cutoff+len(rows)+4:
                raise RuntimeError("Episode physical step loop nontermination")
        m=env.env.metrics()
        results[arm]=dict(metrics=m, control=dict(env.stat),override=overrides,
                          eligible=eligible, override_rate=overrides/max(eligible,1),
                          physical_steps=env.env.now)
    return results


def evaluate(a,seed,cfg,train_rows,ctx,encoder,heads,risk,calib,legacy,np,torch):
    dest=a.output/"evaluation"/a.current_arm/f"s{seed}.json"
    if dest.is_file():
        return load(dest)
    final=[]
    bank=load(Path(a.base_output)/"DEMAND_BANK.json")
    protocol=load(Path(a.base_output)/"PROTOCOL.json")
    from corridor_env import Request
    finals=[x for x in protocol["bank"] if x["role"]=="final"]
    if len(finals)!=4:
        raise ValueError("要求与基线相同的四条评估需求")
    for item in finals:
        trace_seed=item["name"]
        source=[Request(q["id"],q["time_min"],tuple(q["origin"]),tuple(q["destination"])) for q in bank[trace_seed]]
        if demand_hash(source)!=item["sha256"]:
            raise ValueError("评估需求指纹不符")
        for rate in a.rates:
            rows=compressed_trace(source,rate,np)
            for policy_seed in (123,124,125):
                scores=test_episode(cfg,rows,ctx,encoder,heads,risk,calib,a,seed,
                                    policy_seed,np,torch)
                for arm,result in scores.items():
                    final.append(dict(train_seed=seed,trace_seed=trace_seed,
                                      policy_seed=policy_seed,rate=rate,arm=arm,**result))
            print(f"seed={seed} pressure={rate:g} completed",flush=True)
    store(dest,final)
    return final


def summarize(a):
    rows=[]
    for p in sorted((a.output/"evaluation").glob("*/s*.json")):
        rows.extend(load(p))
    summary=[]
    import statistics
    for arm in ("HAPPO_BASE","HAPPO_LEGAL_ONLY",*ARMS):
        for rate in a.rates:
            these=[r for r in rows if r["arm"]==arm and abs(r["rate"]-rate)<1e-7]
            if not these:
                continue
            complete=[v["metrics"]["mean_att_min"] for v in these]
            cost=[v["metrics"]["fixed_cutoff_cost_min"] for v in these]
            comp=[v["metrics"]["completion"] for v in these]
            phases={k:statistics.mean(v["metrics"]["phase_min_per_generated"][k] for v in these)
                    for k in ("route","access","queue","pad_wait","takeoff","flight","landing","egress")}
            summary.append(dict(arm=arm,pressure=rate,n=len(these),
                         train_seeds=sorted(set(r["train_seed"] for r in these)),
                         mean_att_min=statistics.mean(complete) if all(x is not None for x in complete) else None,
                         censored=any(x is None for x in complete),
                         fixed_cutoff_cost_min=statistics.mean(cost),
                         completion=statistics.mean(comp),phases=phases,
                         override_rate=statistics.mean(r["override_rate"] for r in these)))
    store(a.output/"SUMMARY.json",summary)
    with (a.output/"SUMMARY.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=("arm","pressure","n","train_seeds","mean_att_min","censored",
                "fixed_cutoff_cost_min","completion","override_rate")
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in summary:w.writerow({k:row[k] for k in fields})
    return summary


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--train-trace",required=True,type=Path)
    p.add_argument("--harl-dir",required=True,type=Path)
    p.add_argument("--base-output",required=True,type=Path,
                   help="Original HAPPO E2 train output containing four frozen checkpoints")
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--phase",choices=("collect","train","eval","all"),default="all")
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--seeds",nargs="+",type=int,default=SEEDS)
    p.add_argument("--rates",nargs="+",type=float,default=RATES)
    p.add_argument("--snapshots",type=int,default=800,
                   help="number of candidate samples per train seed")
    p.add_argument("--calibration-snapshots",type=int,default=240)
    p.add_argument("--max-per-episode",type=int,default=36)
    p.add_argument("--max-collect-episodes",type=int,default=600)
    p.add_argument("--snapshot-probability",type=float,default=.13)
    p.add_argument("--branch-horizon",type=int,default=20,
                   help="counterfactual PHYSICAL minutes")
    p.add_argument("--encoder-epochs",type=int,default=25)
    p.add_argument("--head-epochs",type=int,default=90)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--alpha",type=float,default=.1,
                   help="family-wise conformal miscoverage")
    p.add_argument("--max-adverse-probability",type=float,default=.95)
    p.add_argument("--margin",type=float,default=0.,
                   help="min predicted improvement in undiscounted person-minutes")
    a=p.parse_args(argv)
    if not (0<a.alpha<1 and a.branch_horizon>0 and a.snapshots>0 and
            a.calibration_snapshots>0 and 0<a.snapshot_probability<=1):
        p.error("Invalid sampling/calibration parameters")
    if (not set(a.seeds)<=set(SEEDS) or len(set(a.seeds))!=len(a.seeds)
            or any(not math.isfinite(x) or x<1 or x>1.4 for x in a.rates)
            or len(set(a.rates))!=len(a.rates)):
        p.error("Only frozen 2101-2104 seeds and pressure <=1.4 are supported")
    return a


def main(argv=None):
    a=parse(argv)
    a.output=a.output.resolve()
    a.base_output=a.base_output.resolve()
    if a.output==a.base_output or a.base_output in a.output.parents:
        raise ValueError("CARE 输出不得写进原模型目录")
    import gymnasium as gym
    import numpy as np
    import torch
    import train_eval_corridor_uagmc_happo_mappo as base
    torch.set_num_threads(1)
    if a.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable, specify --device cpu")
    repo=Path(__file__).resolve().parent
    cfg=base.config(a.config,"E2")
    rows=base.trace(cfg,a.train_trace)
    protocol=source_check(a.base_output,cfg,rows,repo)
    for s in a.seeds: checkpoint(a.base_output,s)
    output_lock=base.lock_run(a.output)
    manifest=dict(schema=SCHEMA,source_environment_sha=protocol["environment_sha"],
                  source_dispatch_sha=protocol["dispatch_sha"],
                  train_trace_sha=demand_hash(rows),steps_per_HAPPO_seed=STEP_CONTRACT,
                  pretrained_HAPPO_seed_list=list(a.seeds),arms=list(ARMS),
                  config=asdict(cfg),
                  script_sha256=sha(__file__),routing="batch",
                  checkpoint_files={str(s):[sha(x) for x in checkpoint(a.base_output,s)[:2]] for s in a.seeds},
                  branch_horizon=a.branch_horizon, snapshots=a.snapshots,
                  calibration_snapshots=a.calibration_snapshots,
                  encoder_epochs=a.encoder_epochs,head_epochs=a.head_epochs,
                  alpha=a.alpha,margin=a.margin,max_adverse_probability=a.max_adverse_probability,rate_list=list(a.rates),
                  note="Frozen HAPPO CARE mechanism adaptation; counterfactual minutes separately counted")
    contract=a.output/"CONTRACT.json"
    if contract.is_file():
        if load(contract)!=json.loads(json.dumps(manifest)):
            raise ValueError("Output contract mismatch. Use new --output directory")
    else:
        store(contract,manifest)
    failures=0
    for arm,seed in ((arm,seed) for arm in ARMS for seed in a.seeds):
        a.current_arm=arm
        base.set_seed(seed,np,torch)
        ctx=load_model_context(a,seed,cfg,rows,base,np,torch,gym)
        try:
            if a.phase in ("collect","all"):
                collect(a,seed,cfg,rows,ctx,base,np,torch)
            if a.phase in ("train","all"):
                data_path=a.output/"data"/f"s{seed}.npz"
                if not data_path.is_file():
                    raise FileNotFoundError(f"Run --phase collect first: {data_path}")
                ds=np.load(data_path,allow_pickle=False)
                train_models(a,seed,ds,np,torch)
                encoder,heads,risk=scorer(a,seed,ctx,np,torch)
                family_calibration(a,seed,ds,encoder,heads,np,torch)
                calibration_diagnostics(a,seed,ds,encoder,heads,np,torch)
            if a.phase in ("eval","all"):
                dp=a.output/"data"/f"s{seed}.npz"
                if not dp.is_file():
                    raise FileNotFoundError(f"CARE data missing: {dp}")
                ds=np.load(dp,allow_pickle=False)
                encoder,heads,risk=scorer(a,seed,ctx,np,torch)
                calibration=family_calibration(a,seed,ds,encoder,heads,np,torch)
                evaluate(a,seed,cfg,rows,ctx,encoder,heads,risk,calibration,base,np,torch)
        except Exception:
            failures+=1
            store(a.output/"errors"/f"{arm}_s{seed}.json",dict(error=traceback.format_exc()))
            traceback.print_exc()
        finally:
            ctx["norm"].close()
    if a.phase in ("eval","all"):
        summarize(a)
    print("CARE6 completed for requested phase; original HAPPO weights unchanged.",flush=True)
    return int(failures>0)


if __name__=="__main__":
    raise SystemExit(main())
