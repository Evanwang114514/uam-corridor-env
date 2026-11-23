"""冻结 E2 MAPPO：直接收益、总收益与显式队友响应的三个 CARE 对照。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import copy
import hashlib
import heapq
import json
import math
from pathlib import Path
from statistics import mean, stdev
import traceback

from corridor_env import Request, validate_requests
import train_eval_corridor_uagmc_happo_mappo as base


ARMS = ("D_ONLY", "TOTAL_ONLY", "CARE_D_PLUS_R")
SEEDS = (2101, 2102, 2103, 2104)
RATES = (1.0, 1.1, 1.2, 1.3, 1.4)
POLICY_SEEDS = (123, 124, 125)
STEPS = 1_996_800
HORIZON = 20
ALPHA, MAX_RISK = .10, .95
TRAIN_FAMILIES, CAL_FAMILIES, VALID_PER_TRACE = 400, 120, 8
STATS = ("running_mean", "running_mean_sq", "debiasing_term")


def canonical(value):
    return json.loads(json.dumps(value))


def sha_rows(rows):
    return hashlib.sha256(json.dumps([asdict(q) for q in rows], sort_keys=True).encode()).hexdigest()


def matches(path, digest):
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return digest in {hashlib.sha256(x).hexdigest() for x in (raw, lf, lf.replace(b"\n", b"\r\n"))}


def inputs(a):
    old = base.read(a.base_output / "PROTOCOL.json")
    new = base.read(a.transfer_output / "PROTOCOL.json")
    if (old["schema"], old["dispatch_protocol"], old["steps_per_cell"]) != (
            "corridor_legacy_interface_v2", base.DISPATCH_PROTOCOL, STEPS):
        raise ValueError("原始 MAPPO 不是冻结的 E2 调度权合同")
    if new["schema"] != "e2_mappo_transfer40m_v1" or new["source_protocol"] != old:
        raise ValueError("补训批次没有引用同一份 MAPPO 训练合同")
    cfg = base.config(a.config, "E2")
    train = base.trace(cfg, a.train_trace)
    if old["configs"]["E2"] != canonical(asdict(cfg)) or old["train_trace_sha256"] != sha_rows(train):
        raise ValueError("环境或训练需求与原始 MAPPO 不一致")
    for key, name in (("entry_sha", Path(base.__file__).name), ("environment_sha", "corridor_env.py"),
                      ("dispatch_sha", "corridor_dispatch.py")):
        if not matches(Path(base.__file__).with_name(name), old[key]):
            raise ValueError(f"源代码与 MAPPO 合同不符：{name}")
    raw = base.read(a.base_output / "DEMAND_BANK.json")
    banks = defaultdict(dict)
    for item in old["bank"]:
        name = item["name"]
        rows = validate_requests([Request(x["id"], x["time_min"], tuple(x["origin"]),
                                          tuple(x["destination"])) for x in raw[name]])
        if sha_rows(rows) != item["sha256"] or not name.isascii() or not name.replace("_", "").isalnum():
            raise ValueError(f"评估轨迹不合法：{name}")
        banks[item["role"]][name] = rows
    if len(banks["validation"]) != 4 or len(banks["final"]) != 4:
        raise ValueError("需要四条冻结验证轨迹与四条冻结最终轨迹")
    return cfg, train, banks, old, new


def checkpoints(a, args, vendor, new):
    frozen = {}
    if new["args"] != canonical(args) or new["harl"] != vendor:
        raise ValueError("当前 HARL 与补训合同不同")
    for seed in SEEDS:
        tag = f"E2__MAPPO__s{seed}"
        cell = (a.base_output if seed < 2103 else a.transfer_output) / "cells" / tag
        job = base.read(cell / "JOB.json")
        if (job["method"], job["seed"], job["layer"], job["harl_args"], job["harl_files"]) != (
                "MAPPO", seed, "E2", canonical(args), vendor):
            raise ValueError(f"非匹配 MAPPO 存档：{cell}")
        ck = base.read(cell / "checkpoints" / f"{STEPS:09d}.json")
        if ck["real_steps"] != STEPS or any(not ck[k] or Path(ck[k]).name != ck[k] for k in ("model", "norm")):
            raise ValueError(f"非法最终存档指针：{cell}")
        model, norm = base.verified(cell, ck)
        pressure = a.transfer_output / "pressure" / tag
        fingerprint = {k: ck[k] for k in ("real_steps", "model_sha", "norm_sha")}
        if base.read(pressure / "CHECKPOINT.json") != fingerprint:
            raise ValueError(f"现有 MAPPO 回放与冻结权重不同：{tag}")
        done = base.read(pressure / "DONE.json")
        if len(done) != len(RATES) or {x["rate"] for x in done} != set(RATES) or any(
                x["method"] != "MAPPO" or x["train_seed"] != seed or x["n"] != 12 for x in done):
            raise ValueError(f"缺少可复用的五档 MAPPO 回放：{tag}")
        frozen[seed] = (model, norm, pressure, fingerprint)
    return frozen


class Frozen:
    def __init__(self, cfg, train, paths, args, gym, np, torch):
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        self.np, self.torch = np, torch
        self.maker = base.build_gym(gym, np)
        self.vec = DummyVecEnv([lambda: self.maker(cfg, train, "MAPPO")])
        self.norm = None
        try:
            self.norm = VecNormalize.load(str(paths[1]), self.vec)
            self.norm.training = self.norm.norm_reward = False
            self.actors, self.critic, self.vn = base.build_pair(
                "MAPPO", args, self.norm.observation_space, self.norm.action_space, torch.device("cpu"))
            state = torch.load(paths[0], map_location="cpu", weights_only=False)
            if state["real_steps"] != STEPS or set(state["value_statistics"]) != set(STATS):
                raise ValueError("最终权重缺少完整 ValueNorm 统计")
            with torch.no_grad():
                for actor, weights in zip(self.actors, state["actors"], strict=True):
                    actor.actor.load_state_dict(weights)
                    actor.prep_rollout()
                self.critic.critic.load_state_dict(state["critic"])
                self.critic.prep_rollout()
                for k in STATS:
                    dst, src = getattr(self.vn, k), state["value_statistics"][k]
                    if dst.shape != src.shape or not torch.isfinite(src).all():
                        raise ValueError(f"非法价值归一化统计：{k}")
                    dst.copy_(src)
            self.z = np.zeros((1, args["recurrent_n"], args["hidden_sizes"][-1]), np.float32)
            self.mask = np.ones((1, 1), np.float32)
        except BaseException:
            self.close()
            raise

    def close(self):
        (self.norm if self.norm is not None else self.vec).close()

    def normalize(self, x):
        return self.norm.normalize_obs(x[None])

    def action(self, role, x, u=None):
        with self.torch.no_grad():
            actor = self.actors[role]
            if u is None:
                # 最终回放沿用已有 MAPPO 的原生采样器；推断小头不消耗此随机流。
                return int(actor.get_actions(self.normalize(x), self.z, self.mask)[0].item())
            placeholder = self.torch.zeros((1, 1), dtype=self.torch.long)
            probs = actor.evaluate_actions(self.normalize(x), self.z, placeholder,
                                           self.mask, None, None)[2].probs[0].detach().cpu().numpy()
            return min(2, int(self.np.searchsorted(self.np.cumsum(probs), u, side="right")))

    def value(self, x):
        with self.torch.no_grad():
            val, _ = self.critic.get_values(self.normalize(x), self.z, self.mask)
            out = self.vn.denormalize(val)
            if self.torch.is_tensor(out):
                out = out.detach().cpu().numpy()
            return float(self.np.asarray(out).item())


class Episode:
    def __init__(self, ctx, cfg, rows):
        self.env = ctx.maker(cfg, rows, "MAPPO")
        self.x = self.env.reset(seed=123)[0]
        self.slot = 0

    def key(self):
        env = self.env
        p = env.focal() if env.phase == 0 else None
        return (env.env.now, env.phase, p.req.id if p else None)

    def options(self):
        env = self.env
        if env.phase == 0:
            return (0, 1, 2) if env.focal() is not None else ()
        targets = env.env.targets()
        return tuple(i for i, s in enumerate(base.STATIONS) if s in targets)

    def effective(self, act, options):
        # 只识别原 MAPPO 动作和哪个合法候选物理等价；真正执行仍交给原调机接口。
        if self.env.phase == 0 or not options or act in options:
            return act
        e = self.env.env
        remaining = {s: len(e.queue[s]) for s in e.dep}
        natural = max(e.dep, key=lambda s: e._lq_key(s, remaining))
        index = base.STATIONS.index(natural)
        return index if index in options else options[0]

    def step(self, action):
        env = self.env
        p = env.focal() if env.phase == 0 else None
        if p is not None and sum(q.state == "route" for q in env.env.passengers.values()) > 1:
            station = base.STATIONS[action]
            env.env._route(p, station)
            env.stat["route_" + station] += 1
            self.x = env.observation()
            self.slot += 1
            return 0., 0
        self.x, reward, _, _, info = env.step(action)
        inc = info["physical_increment"]
        if inc:
            self.slot = 0
        return float(reward), inc

    def close(self):
        self.env.close()


def compress(rows, rate):
    t0 = min(q.time_min for q in rows)
    return validate_requests([replace(q, time_min=t0 + (q.time_min - t0) / rate) for q in rows])


class Demand:
    def __init__(self, rows):
        self.rows = tuple(rows)
        self.bins = Counter(int(q.time_min // 30) for q in rows)
        self.end = max(q.time_min for q in rows)

    def sample(self, e, seed, rate, np):
        # 网络从不接触未揭示的真实需求；两支使用同一份从训练经验采样的外生需求。
        rng = np.random.default_rng(seed)
        now = e.now * e.cfg.tick_min
        t0 = min(q.time_min for q in self.rows)
        future = []
        for dt in range(1, HORIZON + 1):
            t = now + dt
            original = t0 + (t - t0) * rate
            if t >= e.cfg.cutoff_min or original > self.end:
                break
            for j in range(int(rng.poisson(self.bins[int(original // 30)] * rate / 30))):
                shape = self.rows[int(rng.integers(len(self.rows)))]
                future.append(Request(f"__care_{seed}_{dt}_{j}", t, shape.origin, shape.destination))
        return future


def known_only(ep):
    e = ep.env.env
    original, rows = e._events, ep.env.rows
    try:
        e._events = [item for item in original if item[2].__name__ != "_reveal"]
        heapq.heapify(e._events)
        ep.env.rows = ()
        result = copy.deepcopy(ep)
    finally:
        e._events, ep.env.rows = original, rows
    return result


def draw(seed, ep, np):
    return float(np.random.default_rng(np.random.SeedSequence(
        [seed, ep.env.env.now, ep.env.phase, ep.slot])).random())


def rollout(snapshot, ctx, future, seed, action, focal_role, replay=None):
    ep = copy.deepcopy(snapshot)
    e = ep.env.env
    # 哨兵只维持固定观察窗口，不加入乘客集合，不产生虚构人时。
    e.request_count = len(e.passengers) + len(future) + 1
    for q in future:
        e._push(e._ticks(q.time_min), e._reveal, q)
    clock = 0
    score, discount = 0., 1.
    peak = max(map(len, e.queue.values()), default=0)
    recorded, replay_miss = {}, 0
    next_x = None
    first = True
    try:
        while clock < HORIZON and not e.done:
            options = ep.options()
            key = ep.key()
            peer = ep.env.phase != focal_role
            if first:
                a = action
                first = False
            elif peer and replay is not None and key in replay and replay[key] in options:
                a = replay[key]
            else:
                if peer and replay is not None and options:
                    replay_miss += 1
                a = ctx.action(ep.env.phase, ep.x, draw(seed, ep, ctx.np))
            if peer and options:
                recorded[key] = ep.effective(a, options)
            reward, inc = ep.step(a)
            peak = max(peak, max(map(len, e.queue.values()), default=0))
            if inc:
                score += discount * reward
                discount *= .99
                clock += inc
                if next_x is None:
                    next_x = ctx.normalize(ep.x)[0]
        backlog = len(e.passengers) - e.finished
        if not e.done and (backlog or e._events):
            score += discount * ctx.value(ep.x)
        if next_x is None or not math.isfinite(score):
            raise ValueError("反事实窗口未产生合法的物理转移或效用")
        return score, recorded, next_x, peak, backlog if e.done else 0, replay_miss
    finally:
        ep.close()


def family(ep, ctx, demand, rng, rate, scenarios):
    options, role = ep.options(), ep.env.phase
    formal = ctx.action(role, ep.x, float(rng.random()))
    effective = ep.effective(formal, options)
    alt = [i for i in options if i != effective]
    snap = known_only(ep)
    out = [dict(candidate=i, A=0., D=0., R=0., risk=0., replay_miss=0) for i in alt]
    future_seeds = rng.integers(1, 2**31 - 1, size=scenarios)
    for fs in future_seeds:
        fs = int(fs)
        future = demand.sample(snap.env.env, fs, rate, ctx.np)
        b, peer, nx, q0, u0, _ = rollout(snap, ctx, future, fs, formal, role)
        bp, _, _, _, _, _ = rollout(snap, ctx, future, fs, formal, role, peer)
        if abs(b - bp) > 1e-6:
            raise RuntimeError("同一原动作在重决策和合法重放分支的效用不一致")
        for row in out:
            act = row["candidate"]
            rr, _, _, qp, up, _ = rollout(snap, ctx, future, fs, act, role)
            rp, _, _, _, _, miss = rollout(snap, ctx, future, fs, act, role, peer)
            a, d = rr - b, rp - bp
            row["A"] += a / scenarios
            row["D"] += d / scenarios
            row["R"] += (a - d) / scenarios
            row["risk"] += float(qp >= q0 + 5 or up > u0) / scenarios
            row["replay_miss"] += miss
    return dict(x=ctx.normalize(ep.x)[0].tolist(), next_x=nx.tolist(), role=role,
                formal=effective, tick=ep.env.env.now, candidates=out)


def stop(out):
    if (out / "STOP_REQUESTED").exists():
        raise InterruptedError("已请求暂停；保留已完成数据、权重和回合")


def collect_episode(out, ctx, cfg, rows, demand, seed, rate, limit, scenarios):
    rng = ctx.np.random.default_rng(seed)
    ep = Episode(ctx, cfg, rows)
    result = []
    try:
        while not ep.env.env.done and len(result) < limit:
            stop(out)
            if len(ep.options()) >= 2 and rng.random() < .08:
                result.append(family(ep, ctx, demand, rng, rate, scenarios))
            ep.step(ctx.action(ep.env.phase, ep.x, draw(seed, ep, ctx.np)))
    finally:
        ep.close()
    return result


def data_bank(out, ctx, cfg, train, validation, seed, split):
    # 训练、校准按整条需求分隔。验证使用冻结验证库；最终轨迹不进入此处。
    path = out / "data" / f"s{seed}" / split
    demand = Demand(train)
    rows = []
    if split == "validation":
        for name, reqs in validation.items():
            for rate in RATES:
                f = path / f"{name}_{rate:g}.json"
                if not f.exists():
                    key = 30_000_000 + seed * 1000 + len(rows)
                    records = collect_episode(out, ctx, cfg, compress(reqs, rate), demand,
                                              key, rate, VALID_PER_TRACE, 3)
                    if len(records) != VALID_PER_TRACE:
                        raise ValueError(f"验证轨迹采样不足：{name}/{rate}")
                    base.save(f, [dict(x, rate=rate, episode=name) for x in records])
                rows.extend(base.read(f))
        return rows
    goal = TRAIN_FAMILIES if split == "train" else CAL_FAMILIES
    for ep in range(150):
        if len(rows) >= goal:
            return rows
        f = path / f"{ep:03d}.json"
        if not f.exists():
            key = (10_000_000 if split == "train" else 20_000_000) + seed * 1000 + ep
            reqs = base.derived_trace(train, cfg, key)
            records = collect_episode(out, ctx, cfg, reqs, demand, key, 1., min(12, goal - len(rows)), 2)
            base.save(f, [dict(x, rate=1., episode=ep) for x in records])
        rows.extend(base.read(f))
    if len(rows) >= goal:
        return rows
    raise RuntimeError(f"样本不足：{split} {len(rows)}/{goal}")


def networks(width, torch):
    nn = torch.nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(28, 64, batch_first=True)
            self.body = nn.Sequential(nn.Linear(width - 168, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
            self.fuse = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
            self.predict = nn.Linear(64, 36)

        def forward(self, x):
            _, h = self.gru(x[:, :168].reshape(-1, 6, 28))
            return self.fuse(torch.cat((h[-1], self.body(x[:, 168:])), dim=1))

    def head(n):
        return nn.Sequential(nn.Linear(73, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n))

    return Encoder(), head


def features(z, role, candidate, formal, torch):
    onehot = torch.nn.functional.one_hot
    return torch.cat((z, onehot(role, 3).float(), onehot(candidate, 3).float(),
                      onehot(formal, 3).float()), dim=1)


def save_model(path, model, torch):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    torch.save(model.state_dict(), pending)
    pending.replace(path)


def fit(out, seed, arm, train, torch):
    root = out / "models" / f"s{seed}"
    x = torch.tensor([r["x"] for r in train], dtype=torch.float32)
    nx = torch.tensor([r["next_x"] for r in train], dtype=torch.float32)
    torch.manual_seed(seed + 2900)
    enc, make_head = networks(x.shape[1], torch)

    def learn(name, net, epochs, n, loss, random_key):
        path = root / (name + ".pt")
        if path.exists():
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        else:
            opt = torch.optim.Adam(net.parameters(), lr=3e-4)
            g = torch.Generator().manual_seed(random_key)
            for _ in range(epochs):
                stop(out)
                for ids in torch.randperm(n, generator=g).split(128):
                    opt.zero_grad(set_to_none=True)
                    value = loss(net, ids)
                    if not torch.isfinite(value):
                        raise ValueError(f"{name} 出现非有限损失")
                    value.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                    opt.step()
            save_model(path, net, torch)
        return net.eval()

    target_obs = torch.cat((nx[:, 140:168], nx[:, -8:]), dim=1)
    enc = learn("encoder", enc, 25, len(x), lambda m, i: torch.nn.functional.smooth_l1_loss(
        m.predict(m(x[i])), target_obs[i]), seed + 10)
    flat = [(i, r, y) for i, r in enumerate(train) for y in r["candidates"]]
    with torch.no_grad():
        z = enc(x)[torch.tensor([i for i, _, _ in flat])]
        f = features(z, torch.tensor([r["role"] for _, r, _ in flat]),
                     torch.tensor([y["candidate"] for _, _, y in flat]),
                     torch.tensor([r["formal"] for _, r, _ in flat]), torch)
    risk_target = torch.tensor([y["risk"] for _, _, y in flat], dtype=torch.float32)
    torch.manual_seed(seed + 3000)
    risk = learn("risk_relative", make_head(1), 90, len(flat),
                 lambda m, i: torch.nn.functional.binary_cross_entropy_with_logits(
                     m(f[i]).flatten(), risk_target[i]), seed + 20)
    keys = {"D_ONLY": ("D",), "TOTAL_ONLY": ("A",), "CARE_D_PLUS_R": ("D", "R")}[arm]
    label = torch.tensor([[y[k] / 100. for k in keys] for _, _, y in flat], dtype=torch.float32)
    torch.manual_seed(seed + 3100)
    model = learn(arm, make_head(len(keys)), 90, len(flat),
                  lambda m, i: torch.nn.functional.smooth_l1_loss(m(f[i]), label[i]), seed + 30)
    return enc, model, risk


def predict(team, normalized, role, formal, options, torch, *, direct_only=False):
    enc, model, risk = team
    with torch.no_grad():
        x = torch.as_tensor(normalized, dtype=torch.float32).reshape(1, -1)
        z = enc(x).expand(len(options), -1)
        f = features(z, torch.full((len(options),), role), torch.tensor(options),
                     torch.full((len(options),), formal), torch)
        # 完整臂直接用两个点预测之和排序，在合成值上一次性校准，不相加分位数。
        value = model(f)
        return (100 * (value[:, 0] if direct_only else value.sum(1))).tolist(), risk(f).sigmoid().flatten().tolist()


def prediction(row, team, torch):
    return predict(team, row["x"], row["role"], row["formal"],
                   [y["candidate"] for y in row["candidates"]], torch)


def calibration(data, team, arm, torch):
    errors = []
    for row in data:
        scores, _ = prediction(row, team, torch)
        field = "D" if arm == "D_ONLY" else "A"
        errors.append(max(s - y[field] for s, y in zip(scores, row["candidates"], strict=True)))
    idx = math.ceil((len(errors) + 1) * (1 - ALPHA)) - 1
    if idx >= len(errors):
        raise ValueError("独立校准轨迹样本不足")
    # 不把同回合候选当作独立，亦不声称该经验阈值保证负载变化后的覆盖率。
    return max(0., sorted(errors)[idx])


def choose(scores, risks, threshold):
    passed = [i for i, (score, risk) in enumerate(zip(scores, risks, strict=True))
              if score > threshold and risk <= MAX_RISK]
    return max(passed, key=lambda i: scores[i]) if passed else None


def audit_validation(rows, team, arm, threshold, torch, direct=None):
    result = []
    for row in rows:
        candidates = row["candidates"]
        scores, risks = prediction(row, team, torch)
        target = [x["A"] for x in candidates]
        picked = choose(scores, risks, threshold)
        best = max(range(len(scores)), key=lambda i: scores[i])
        true_order = max(range(len(target)), key=lambda i: target[i])
        flip = same_head_flip = None
        if direct is not None:
            direct_scores, _ = prediction(row, direct, torch)
            di = max(range(len(direct_scores)), key=lambda i: direct_scores[i])
            flip = dict(changed=best != di, delta_total=target[best] - target[di])
            same, _ = predict(team, row["x"], row["role"], row["formal"],
                              [x["candidate"] for x in candidates], torch, direct_only=True)
            si = max(range(len(same)), key=lambda i: same[i])
            same_head_flip = dict(changed=best != si, delta_total=target[best] - target[si])
        result.append(dict(arm=arm, rate=row["rate"], episode=row["episode"], role=row["role"],
                           tick=row["tick"], oracle_gain=max(0., max(target)),
                           predicted_rank_gain=target[best], applied_gain=0. if picked is None else target[picked],
                           rank_correct=(best == true_order if max(target) - min(target) > 1e-8 else None),
                           ranking_shift_vs_direct=flip, ranking_shift_same_head=same_head_flip,
                           picked=picked, candidates=candidates,
                           scores=scores, risk_probability=risks, threshold=threshold))
    return result


def validation_summary(rows):
    # 让预测误差、门槛过严和响应排序变化能够分别诊断。
    result = []
    for rate in RATES:
        for role in (None, 0, 1):
            group = [x for x in rows if x["rate"] == rate and (role is None or x["role"] == role)]
            if not group:
                continue
            ranked = [x["rank_correct"] for x in group if x["rank_correct"] is not None]
            flips = [x["ranking_shift_same_head"] for x in group if x["ranking_shift_same_head"] is not None]
            changes = [x["delta_total"] for x in flips if x["changed"]]
            result.append(dict(rate=rate, role=role, n=len(group),
                               oracle_positive=sum(x["oracle_gain"] > 0 for x in group),
                               positive_raw=sum(max(x["scores"]) > 0 for x in group),
                               positive_calibrated=sum(max(x["scores"]) > x["threshold"] for x in group),
                               accepted=sum(x["picked"] is not None for x in group),
                               realized_positive=sum(x["applied_gain"] > 0 for x in group),
                               realized_negative=sum(x["applied_gain"] < 0 for x in group),
                               mean_oracle_gain=mean(x["oracle_gain"] for x in group),
                               mean_applied_gain=mean(x["applied_gain"] for x in group),
                               rank_accuracy=mean(ranked) if ranked else None,
                               rank_flips_from_R=len(changes) if flips else None,
                               mean_label_gain_on_flips=mean(changes) if changes else None))
    return result


def replay_final(out, ctx, cfg, rows, method, threshold, team, archived, policy_seed):
    counts = Counter()
    base.set_seed(policy_seed, ctx.np, ctx.torch)
    ep = Episode(ctx, cfg, rows)
    try:
        while not ep.env.env.done:
            stop(out)
            role = ep.env.phase
            nominal = ctx.action(role, ep.x)
            action = nominal
            options = ep.options()
            if len(options) > 1:
                effective = ep.effective(nominal, options)
                alt = [x for x in options if x != effective]
                scores, risk = predict(team, ctx.normalize(ep.x)[0], role, effective, alt, ctx.torch)
                picked = choose(scores, risk, threshold)
                for key in ("all", "passenger" if role == 0 else "aircraft"):
                    counts[key + "_opportunities"] += 1
                    counts[key + "_raw_positive"] += sum(x > 0 for x in scores)
                    counts[key + "_after_calibration"] += sum(x > threshold for x in scores)
                    counts[key + "_after_risk"] += sum(
                        x > threshold and y <= MAX_RISK for x, y in zip(scores, risk, strict=True))
                    counts[key + "_overrides"] += picked is not None
                    counts[key + "_used_lower_rank"] += picked is not None and scores[picked] < max(scores)
                if picked is not None:
                    action = alt[picked]
            ep.step(action)
        metrics = ep.env.env.metrics()
        if not counts["all_overrides"] and metrics != archived["metrics"]:
            raise RuntimeError("零改写时未能逐回合复现原 MAPPO；请勿把不同评估协议混作配对")
        return dict(method=method, metrics=metrics, archived_MAPPO=archived["metrics"],
                    control=dict(ep.env.stat), gate=dict(counts))
    finally:
        ep.close()


def evaluate(a, ctx, cfg, final, seed, arm, threshold, team, frozen):
    for name, rows in final.items():
        for rate in RATES:
            dense = compress(rows, rate)
            for policy_seed in POLICY_SEEDS:
                stop(a.output)
                tag = f"{name}_{policy_seed}.json"
                dst = a.output / "evaluation" / arm / f"s{seed}" / f"{rate:.1f}" / tag
                if dst.exists():
                    continue
                archived = base.read(frozen[2] / f"{rate:.1f}" / tag)
                if (archived["trace"], archived["rate"], archived["policy_seed"]) != (name, rate, policy_seed):
                    raise ValueError("冻结 MAPPO 回放轨迹或随机种子不一致")
                result = replay_final(a.output, ctx, cfg, dense, arm, threshold, team, archived, policy_seed)
                base.save(dst, dict(seed=seed, rate=rate, trace=name, policy_seed=policy_seed, **result))
    done = a.output / "evaluation" / arm / f"s{seed}" / "DONE.json"
    base.save(done, dict(seed=seed, arm=arm, episodes=len(final) * len(RATES) * len(POLICY_SEEDS)))


def summarize(out):
    groups = defaultdict(list)
    for p in sorted((out / "evaluation").glob("*/*/DONE.json")):
        for f in sorted(p.parent.glob("*/*.json")):
            row = base.read(f)
            groups[row["method"], row["rate"], row["seed"]].append(row)
    def avg(values):
        return mean(values) if values and None not in values else None

    def mean_metric(rows, side, key):
        return avg([r[side][key] for r in rows])

    all_rows = []
    for arm in ARMS:
        for rate in RATES:
            seeds = []
            for seed in SEEDS:
                rows = groups.get((arm, rate, seed), [])
                if len(rows) != 12:
                    continue
                m = {}
                for k in ("mean_att_min", "completion", "fixed_cutoff_cost_min"):
                    v, b = (mean_metric(rows, side, k) for side in ("metrics", "archived_MAPPO"))
                    m[k] = v
                    m["MAPPO_" + k] = b
                    m["delta_" + k] = v - b if None not in (v, b) else None
                def awt(side):
                    seg = [r[side]["phase_min_per_generated"] for r in rows]
                    return avg([v["queue"] + v["pad_wait"] if None not in
                                (v["queue"], v["pad_wait"]) else None for v in seg])
                w, wb = awt("metrics"), awt("archived_MAPPO")
                gates = Counter()
                for r in rows:
                    gates.update(r["gate"])
                seeds.append(dict(seed=seed, n=len(rows), **m, awt=w, MAPPO_awt=wb,
                                  delta_awt=w - wb if None not in (w, wb) else None,
                                  gates=dict(gates), override_rate=gates["all_overrides"] /
                                  max(1, gates["all_opportunities"])))
            vals = [x["mean_att_min"] for x in seeds]
            all_rows.append(dict(method=arm, rate=rate, n_seeds=len(seeds), complete=len(seeds) == 4,
                                 mean_att_min=avg(vals), sd_att_min=stdev(vals) if len(vals) > 1 and
                                 None not in vals else None,
                                 **{k: avg([s[k] for s in seeds]) for k in
                                    ("MAPPO_mean_att_min", "delta_mean_att_min", "completion",
                                     "delta_fixed_cutoff_cost_min", "delta_awt", "override_rate")},
                                 seeds=seeds))
    base.save(out / "SUMMARY.json", all_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("config", "train-trace", "harl-dir", "base-output", "transfer-output", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    a.output = a.output.resolve()
    for src in (a.base_output.resolve(), a.transfer_output.resolve()):
        if src == a.output or src in a.output.parents or a.output in src.parents:
            p.error("新结果必须与冻结权重目录隔离")

    import gymnasium as gym
    import numpy as np
    import torch

    torch.set_num_threads(1)
    cfg, train, banks, old, new = inputs(a)
    args, _, vendor = base.harl_vendor(a.harl_dir.resolve(), np, torch)
    frozen = checkpoints(a, args, vendor, new)
    contract = canonical(dict(schema="e2_mappo_care_d_plus_r_v1", arms=ARMS, seeds=SEEDS, rates=RATES,
                              optimizer_steps_MAPPO=0, reference=old, reference_batch_sha=base.digest(
                                  a.transfer_output / "PROTOCOL.json"),
                              source_sha=base.digest(Path(__file__)), harl=vendor, policy_seeds=POLICY_SEEDS,
                              checkpoint={s: frozen[s][3] for s in SEEDS}, horizon=HORIZON, alpha=ALPHA,
                              max_risk=MAX_RISK, train_families=TRAIN_FAMILIES, cal_families=CAL_FAMILIES,
                              validation_per_trace_rate=VALID_PER_TRACE, device="cpu",
                              torch=str(torch.__version__), gym=gym.__version__, numpy=np.__version__,
                              routing="原 MAPPO 批量回放；上线与反事实同一物理动作执行器",
                              labels="配对仿真；按原动作计算新增风险；训练校准仅 1.0 倍"))
    failures = 0
    print("MAPPO 权重冻结；3 个 CARE 臂 × 4 种子，复用旧 MAPPO 压力回放。", flush=True)
    with base.lock_run(a.output):
        manifest = a.output / "CONTRACT.json"
        if manifest.exists() and base.read(manifest) != contract:
            raise ValueError("续跑合同不同；请保持原配置或使用新目录")
        base.save(manifest, contract)
        for arm in ARMS:
            for seed in SEEDS:
                ctx = None
                try:
                    stop(a.output)
                    done = a.output / "evaluation" / arm / f"s{seed}" / "DONE.json"
                    if done.exists():
                        continue
                    print(f"{arm} seed {seed}: 开始或继续", flush=True)
                    ctx = Frozen(cfg, train, frozen[seed], args, gym, np, torch)
                    data = {split: data_bank(a.output, ctx, cfg, train, banks["validation"], seed, split)
                            for split in ("train", "calibration", "validation")}
                    team = fit(a.output, seed, arm, data["train"], torch)
                    threshold = calibration(data["calibration"], team, arm, torch)
                    direct = fit(a.output, seed, "D_ONLY", data["train"], torch) if arm == "CARE_D_PLUS_R" else None
                    validation = audit_validation(data["validation"], team, arm, threshold, torch, direct)
                    base.save(a.output / "diagnostics" / arm / f"s{seed}.json",
                              dict(calibration_threshold=threshold,
                                   by_rate_role=validation_summary(validation), validation=validation))
                    evaluate(a, ctx, cfg, banks["final"], seed, arm, threshold, team, frozen[seed])
                    summarize(a.output)
                    print(f"{arm} seed {seed}: 五档回放完成", flush=True)
                except (KeyboardInterrupt, InterruptedError):
                    return 130
                except Exception:
                    failures += 1
                    msg = traceback.format_exc()
                    base.save(a.output / "errors" / arm / f"s{seed}.json", dict(error=msg))
                    print(msg, flush=True)
                finally:
                    if ctx is not None:
                        ctx.close()
        summarize(a.output)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
