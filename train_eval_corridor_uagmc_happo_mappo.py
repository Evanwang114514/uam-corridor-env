"""K3 走廊三基线：保留旧策略和更新器，在新物理中从零训练及回放。"""
from __future__ import annotations

import argparse
from corridor_dispatch import DISPATCH_PROTOCOL, PriorityCorridor
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
METHODS = ("UAGMC", "UAGMC_AIR", "HAPPO", "MAPPO")
PPO_METHODS = {"UAGMC", "UAGMC_AIR"}
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


def build_gym(gym, np):
    class LegacyPhase(gym.Env):
        """旧观测排列，新的物理状态；一个物理分钟只产生一次人时奖励。"""

        def __init__(self, cfg, rows, method):
            self.c = cfg
            self.rows = rows
            self.single = method == "UAGMC"
            self.uagmc_observation = method in PPO_METHODS
            self.env = PriorityCorridor(cfg, policy_returns=not self.single)
            self.action_space = gym.spaces.Discrete(3)
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, (6, 28) if self.uagmc_observation else (PACK + 8,), dtype=np.float32)

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
            frame, extras = self.snapshot(focal, not self.uagmc_observation)
            history = (self.history[1:] + [frame]) if post else self.history
            if self.uagmc_observation:
                return np.asarray(history, dtype=np.float32)
            # 延迟门未启用；保持旧版 8 个尾字段，不塞进新的资源或未来需求。
            tail = [int(post), *(int(post and self.route_index == i) for i in range(3)), 0., 0., 0., 0.]
            obs = np.asarray([x for f in history for x in f]+extras+tail, dtype=np.float32)
            if obs.shape != (PACK+8,) or not np.isfinite(obs).all():
                raise RuntimeError("联合观测维数或数值不符")
            return obs

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
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
            self.env.choose_returns(STATIONS[action], self.stat)
            return self.advance()

    class UagmcAir(gym.Wrapper):
        """保留原 LSTM-PPO；同一策略的两个离散输出按同一物理调机协议执行。"""

        def __init__(self, cfg, rows):
            super().__init__(LegacyPhase(cfg, rows, "UAGMC_AIR"))
            self.action_space = gym.spaces.MultiDiscrete([len(STATIONS), len(STATIONS)])

        @property
        def stat(self):
            return self.env.stat

        @property
        def corridor(self):
            return self.env.env

        def step(self, action):
            if not self.action_space.contains(action):
                raise ValueError("两个 PPO 输出均须为 V0/V1/V3 的索引")
            _, reward, done, cutoff, info = self.env.step(int(action[0]))
            if reward or done or cutoff or info["physical_increment"]:
                raise RuntimeError("乘客阶段不得计步、终止或产生奖励")
            return self.env.step(int(action[1]))

    def make(cfg, rows, method):
        if method == "UAGMC_AIR":
            return UagmcAir(cfg, rows)
        return LegacyPhase(cfg, rows, method)

    return make


def lstm_extractor():
    from torch import nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class TemporalLSTMExtractor(BaseFeaturesExtractor):
        """旧 UAGMC：逐帧线性编码、128 单层 LSTM、末帧读出。"""
        def __init__(self, observation_space, features_dim=128):
            super().__init__(observation_space, features_dim)
            self.frame_encoder = nn.Linear(observation_space.shape[1], features_dim)
            self.lstm = nn.LSTM(features_dim, 128, num_layers=1, batch_first=True)
            self.output_proj = nn.Linear(128, features_dim)

        def forward(self, obs):
            x, _ = self.lstm(self.frame_encoder(obs))
            return self.output_proj(x[:, -1])

    return TemporalLSTMExtractor


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


def build_pair(method, args, space, action_space, device):
    from harl.algorithms.actors.happo import HAPPO
    from harl.algorithms.actors.mappo import MAPPO
    from harl.algorithms.critics.v_critic import VCritic
    from harl.common.valuenorm import ValueNorm
    cls = HAPPO if method == "HAPPO" else MAPPO
    actors = [cls(args, space, action_space, device=device) for _ in range(2)]
    return actors, VCritic(args, space, device=device), ValueNorm(1, device=device)


def save_generation(folder, real, total, marks, norm, write_model, ext, write_rng=None):
    current = read(folder/"RESUME.json") if (folder/"RESUME.json").exists() else {}
    # 预注册存档各自保留；普通恢复点在两个槽之间轮换，不覆盖当前指针所指的上一代。
    stem = f"step_{real:09d}" if real in marks else (
        "rolling_b" if str(current.get("model", "")).startswith("rolling_a.") else "rolling_a")
    model = folder/(stem + ext)
    normal = folder/(stem + ".norm.pkl")
    model_tmp = folder/(stem + ".pending" + ext)
    normal_tmp = folder/(stem + ".norm.pending.pkl")
    write_model(model_tmp)
    norm.save(str(normal_tmp))
    model_tmp.replace(model)
    normal_tmp.replace(normal)
    meta = dict(real_steps=real, model=model.name, norm=normal.name,
                model_sha=digest(model), norm_sha=digest(normal))
    if write_rng:
        rng = folder/(stem + ".rng.pt")
        rng_tmp = folder/(stem + ".rng.pending.pt")
        write_rng(rng_tmp)
        rng_tmp.replace(rng)
        meta.update(rng=rng.name, rng_sha=digest(rng))
    if real in marks:
        save(folder/"checkpoints"/f"{real:09d}.json", meta)
    save(folder/"RESUME.json", meta)
    print(f"{folder.name}：{real:,} / {total:,} 真实步", flush=True)


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


def train_cell(folder, a, method, seed, cfg, rows, marks, env_class, args, author_update, gym, np, torch):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    set_seed(seed, np, torch)
    meta = read(folder/"RESUME.json") if (folder/"RESUME.json").is_file() else None
    start = meta["real_steps"] if meta else 0
    if start == a.steps:
        return
    if start > a.steps or start % ROLLOUT:
        raise RuntimeError("恢复位置不符合真实 rollout 边界")
    vector = SubprocVecEnv([lambda: env_class(cfg, rows, method) for _ in range(LANES)], start_method="spawn")
    single = method in PPO_METHODS
    norm = None
    try:
        if meta:
            model_path, norm_path = verified(folder, meta)
            norm = VecNormalize.load(str(norm_path), vector)
        else:
            norm = VecNormalize(vector, norm_obs=True, norm_reward=single, gamma=.99)
        norm.training = True
        if single:
            model = (PPO.load(model_path, env=norm, device=a.device) if meta else PPO(
                "MlpPolicy", norm, learning_rate=lambda x: 5e-4*x, n_steps=LENGTH, batch_size=512,
                n_epochs=10, gamma=.99, gae_lambda=.95, clip_range=.25, ent_coef=.01, vf_coef=.5,
                max_grad_norm=.5, seed=seed, device=a.device,
                policy_kwargs=dict(features_extractor_class=lstm_extractor(),
                                   net_arch=dict(pi=[256, 256], vf=[256, 256]))))
            if meta:
                rng_path = folder/meta["rng"]
                if digest(rng_path) != meta["rng_sha"]:
                    raise ValueError("PPO 随机状态存档校验失败")
                rng_restore(torch.load(rng_path, map_location="cpu", weights_only=False), a.device, np, torch)

            def checkpoint(n):
                save_generation(folder, n, a.steps, marks, norm, lambda p: model.save(str(p)), ".zip",
                                lambda p: torch.save(rng_snapshot(np, torch), p))

            def capture():
                n = model.num_timesteps
                if n > start and n != capture.last and (n in marks or n % (ROLLOUT*a.save_every) == 0):
                    checkpoint(n)
                    capture.last = n

            capture.last = start

            class Boundaries(BaseCallback):
                def _on_rollout_start(self):
                    capture()
                    if (a.output/"STOP_REQUESTED").exists():
                        if model.num_timesteps > start and model.num_timesteps != capture.last:
                            checkpoint(model.num_timesteps)
                        raise InterruptedError("到达更新边界，按暂停标记退出")

                def _on_step(self):
                    return True

            if model.num_timesteps != start:
                raise RuntimeError("PPO 模型步数与恢复指针不一致")
            model.learn(total_timesteps=a.steps-start, reset_num_timesteps=False, callback=Boundaries())
            capture()
            return

        from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
        from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
        device = torch.device(a.device)
        space, act = norm.observation_space, gym.spaces.Discrete(3)
        actors, critic, vn = build_pair(method, args, space, act, device)
        if meta:
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            if state["real_steps"] != start:
                raise RuntimeError("联合模型步数与恢复指针不一致")
            for actor, weights, optim in zip(actors, state["actors"], state["optimizers"]):
                actor.actor.load_state_dict(weights)
                actor.actor_optimizer.load_state_dict(optim)
            critic.critic.load_state_dict(state["critic"])
            critic.critic_optimizer.load_state_dict(state["critic_optimizer"])
            vn.load_state_dict(state["value_norm"])
            for k, v in state["value_statistics"].items():
                getattr(vn, k).copy_(v)
            rng_restore(state["rng"], device, np, torch)
        buffers = [OnPolicyActorBuffer(args, space, act) for _ in actors]
        cb = OnPolicyCriticBufferEP(args, space)
        runner = SimpleNamespace(algo_args={"train": args}, actor=actors, critic=critic,
                                 actor_buffer=buffers, critic_buffer=cb, value_normalizer=vn,
                                 state_type="EP", fixed_order=False, num_agents=2, action_aggregation="prod")
        obs = norm.reset()
        zeros = np.zeros((LANES, 1, 128), np.float32)
        ones = np.ones((LANES, 1), np.float32)
        masks = ones.copy()
        real = start
        while real < a.steps:
            if (a.output/"STOP_REQUESTED").exists():
                raise InterruptedError("到达更新边界，按暂停标记退出")
            for actor, buff in zip(actors, buffers):
                actor.prep_rollout()
                actor.lr_decay(real, a.steps)
                buff.step, buff.factor = 0, None
            cb.step = 0
            critic.prep_rollout()
            critic.lr_decay(real, a.steps)
            for k in range(LENGTH):
                with torch.no_grad():
                    values, _ = critic.get_values(obs, zeros, masks)
                    ap, lp, _ = actors[0].get_actions(obs, zeros, masks)
                buffers[0].obs[k], buffers[0].masks[k], cb.share_obs[k] = obs, masks, obs
                mid, reward, done, infos = norm.step(ap.detach().cpu().numpy().ravel())
                if any(done) or np.any(reward) or any(i["physical_increment"] != 0 for i in infos):
                    raise RuntimeError("乘客准备阶段不可计步或产生奖励")
                with torch.no_grad():
                    aa, la, _ = actors[1].get_actions(mid, zeros, masks)
                buffers[1].obs[k], buffers[1].masks[k] = mid, masks
                nxt, rewards, dones, infos = norm.step(aa.detach().cpu().numpy().ravel())
                if any(i["physical_increment"] != 1 for i in infos):
                    raise RuntimeError("飞机阶段每环境只计一个真实物理分钟")
                next_masks = (~dones).astype(np.float32)[:, None]
                bad = np.asarray([[0. if i.get("TimeLimit.truncated") else 1.] for i in infos], np.float32)
                for b, action, logp in zip(buffers, (ap, aa), (lp, la)):
                    b.insert(nxt, zeros, action.detach().cpu().numpy(), logp.detach().cpu().numpy(), next_masks, ones)
                cb.insert(nxt, zeros, values.detach().cpu().numpy(), rewards[:, None], next_masks, bad)
                obs, masks = nxt, next_masks
                real += LANES
            with torch.no_grad():
                next_value, _ = critic.get_values(obs, zeros, masks)
            cb.compute_returns(next_value.detach().cpu().numpy(), vn)
            for actor in actors:
                actor.prep_training()
            critic.prep_training()
            if method == "HAPPO":
                author_update(runner)
            else:
                advantages = cb.returns[:-1]-vn.denormalize(cb.value_preds[:-1])
                for actor, b in zip(actors, buffers):
                    actor.train(b, advantages.copy(), "EP")
                critic.train(cb, vn)
            if real in marks or real % (ROLLOUT*a.save_every) == 0 or (a.output/"STOP_REQUESTED").exists():
                state = dict(real_steps=real, actors=[x.actor.state_dict() for x in actors],
                             optimizers=[x.actor_optimizer.state_dict() for x in actors],
                             critic=critic.critic.state_dict(), critic_optimizer=critic.critic_optimizer.state_dict(),
                             value_norm=vn.state_dict(),
                             value_statistics={k: getattr(vn, k) for k in
                                               ("running_mean", "running_mean_sq", "debiasing_term")},
                             rng=rng_snapshot(np, torch))
                save_generation(folder, real, a.steps, marks, norm, lambda p: torch.save(state, p), ".pt")
            for b in buffers:
                b.after_update()
            cb.after_update()
    finally:
        (norm if norm is not None else vector).close()


def eval_cell(folder, a, method, cfg, bank, marks, env_class, args, np, torch):
    from stable_baselines3 import PPO
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
        vec = DummyVecEnv([lambda: env_class(cfg, bank[0][2], method)])
        norm = None
        try:
            norm = VecNormalize.load(str(norm_path), vec)
            norm.training = False
            norm.norm_reward = False
            if method in PPO_METHODS:
                model = PPO.load(model_path, device=a.device)
            else:
                device = torch.device(a.device)
                pair, _, _ = build_pair(method, args, norm.observation_space, norm.action_space, device)
                state = torch.load(model_path, map_location="cpu", weights_only=False)
                for actor, weights in zip(pair, state["actors"]):
                    actor.actor.load_state_dict(weights)
                    actor.prep_rollout()
                zeros, ones = np.zeros((1, 1, 128), np.float32), np.ones((1, 1), np.float32)
            results = []
            for name, role, rows in bank:
                if role == "final" and mark != a.steps:
                    continue
                for seed in ((123, 124, 125) if role == "final" else (123, 124)):
                    if (a.output/"STOP_REQUESTED").exists():
                        raise InterruptedError("评估暂停；再次运行从未完成的存档继续")
                    set_seed(seed, np, torch)
                    env = env_class(cfg, rows, method)
                    x, _ = env.reset(seed=123)
                    corridor = env.corridor if method == "UAGMC_AIR" else env.env
                    while not corridor.done:
                        obs = norm.normalize_obs(x[None])
                        with torch.no_grad():
                            if method in PPO_METHODS:
                                raw = model.predict(obs, deterministic=False)[0][0]
                                action = raw if method == "UAGMC_AIR" else int(raw)
                            else:
                                z, _, _ = pair[env.phase].get_actions(obs, zeros, ones)
                                action = int(z.item())
                        x, _, _, _, _ = env.step(action)
                    metrics = corridor.metrics()
                    results.append(dict(trace=name, role=role, policy_seed=seed, step=mark,
                                        metrics=metrics, control=dict(env.stat)))
            save(result_path, results)
            final.extend(results)
            print(f"{folder.name}：已评估 {mark:,} 步存档，{len(results)} 个回合", flush=True)
        finally:
            (norm if norm is not None else vec).close()
    if final:
        save(folder/"EVALUATION.json", dict(rows=final, weights_unchanged=True))
    return final


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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path, help="E3 的 11/11/11/24 基准 manifest 或配置")
    p.add_argument("--train-trace", required=True, type=Path)
    p.add_argument("--validation-traces", type=Path, nargs="+", help="自备验证 CSV；不传则按固定种子独立生成")
    p.add_argument("--final-traces", type=Path, nargs="+", help="自备最终 CSV；只评最终存档")
    p.add_argument("--harl-dir", type=Path, help="既有 HARL 作者源码目录；只跑 UAGMC 可省略")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--phase", choices=("train", "eval", "all"), default="all")
    p.add_argument("--methods", choices=METHODS, nargs="+", default=("UAGMC", "HAPPO", "MAPPO"))
    p.add_argument("--layers", choices=LAYERS, nargs="+", default=tuple(LAYERS))
    p.add_argument("--seeds", type=int, nargs="+", default=(2101, 2102, 2103, 2104))
    p.add_argument("--steps", type=int, default=1_996_800, help="每格真实步数，必须是 10240 的整数倍")
    p.add_argument("--save-every", type=int, default=5, help="每隔多少轮更新保存非评估恢复点")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a = p.parse_args(argv)
    if a.steps <= 0 or a.steps % ROLLOUT or a.save_every <= 0:
        p.error("训练长度须为正且对齐 10,240 步 rollout；保存间隔必须为正")
    if set(a.methods)-PPO_METHODS and not a.harl_dir:
        p.error("HAPPO/MAPPO 需要 --harl-dir")

    import gymnasium as gym
    import numpy as np
    import torch
    import stable_baselines3 as sb3

    torch.set_num_threads(1)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境无可用 CUDA；如需 CPU，请显式传 --device cpu")
    env_class = build_gym(gym, np)
    joint_args, update, vendor_files = harl_vendor(a.harl_dir.resolve(), np, torch) if a.harl_dir else (None, None, None)
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
    for i, seed in enumerate(dict.fromkeys(a.seeds)):
        methods = list(dict.fromkeys(a.methods))
        offset = i % len(methods)
        for layer in dict.fromkeys(a.layers):
            tasks.extend((layer, m, seed) for m in methods[offset:]+methods[:offset])
    contract = dict(schema="corridor_legacy_interface_v2", dispatch_protocol=DISPATCH_PROTOCOL, environment_sha=digest(Path(__file__).with_name("corridor_env.py")),
                    dispatch_sha=digest(Path(__file__).with_name("corridor_dispatch.py")),
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
            meta = dict(layer=layer, method=method, seed=seed, harl_args=joint_args if method not in PPO_METHODS else None,
                        harl_files=vendor_files if method not in PPO_METHODS else None,
                        harl_pin=HARL_COMMIT if method not in PPO_METHODS else None,
                        uagmc_air=("factorized_single_ppo" if method == "UAGMC_AIR" else None))
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
                real = read(folder/"RESUME.json")["real_steps"] if (folder/"RESUME.json").exists() else 0
                score = [x["metrics"] for x in evaluated if x["role"] == "validation" and x["step"] >= .8*a.steps]
                final = [x["metrics"] for x in evaluated if x["role"] == "final"]
                def aggregate(rows):
                    def mean(k):
                        v = [x[k] for x in rows]
                        return sum(v)/len(v) if v and all(x is not None for x in v) else None
                    return dict(n=len(rows), completion=mean("completion"), att_min=mean("mean_att_min"),
                                cutoff_cost_min=mean("fixed_cutoff_cost_min"))
                save(folder/"STATUS.json", dict(real_steps=real, trained=real == a.steps,
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
        save(a.output/"SUMMARY.json", dict(cells=rows, failures_this_call=failures))
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
