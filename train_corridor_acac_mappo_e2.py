"""E2 的 ACAC 决策事件适配：原物理不变，四方法、每方法四种子。"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import train_eval_corridor_uagmc_happo_mappo as base
from corridor_dispatch import PriorityCorridor
from corridor_env import Request, validate_requests

STEPS = 1_996_800
SEEDS = (2101, 2102, 2103, 2104)
MARKS = (204800, 501760, 1003520, 1597440, 1699840, 1802240, 1904640, STEPS)
RATES = (1.0, 1.1, 1.2, 1.3, 1.4)
METHODS = {
    "ACAC_FULL": (True, True, True),
    "ACAC_NO_TE": (False, True, True),
    "ACAC_NO_SA": (True, False, True),
    "ACAC_MICRO_GAE": (True, True, False),
}
VALUE_KEYS = ("running_mean", "running_mean_sq", "debiasing_term")
METRICS = ("mean_att_min", "completion", "fixed_cutoff_cost_min", "awt_min")


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


def expected_native(bank):
    return {(step, name, role, seed) for step in MARKS for name, role, _ in bank
            if role == "validation" or step == STEPS
            for seed in ((123, 124, 125) if role == "final" else (123, 124))}


def native_key(r):
    return r["step"], r["trace"], r["role"], r["policy_seed"]


def references(cli, args, vendor, bank):
    refs, provenance = {}, {}
    expected = expected_native(bank)
    transfer_contract = base.read(cli.transfer_output / "PROTOCOL.json")
    if transfer_contract["source_protocol"] != base.read(cli.base_output / "PROTOCOL.json"):
        raise ValueError("补训 MAPPO 所引用的物理和需求合同与首两种子不一致")
    for seed in SEEDS:
        tag = f"E2__MAPPO__s{seed}"
        cell = (cli.base_output if seed < 2103 else cli.transfer_output) / "cells" / tag
        job = base.read(cell / "JOB.json")
        if (job.get("method"), job.get("seed"), job.get("harl_args"), job.get("harl_files")) != (
                "MAPPO", seed, canonical(args), vendor):
            raise ValueError(f"MAPPO 对照种子或训练器指纹不符：{cell}")
        if not base.read(cell / "STATUS.json")["evaluated_through_final"]:
            raise ValueError(f"MAPPO 对照未完成：{cell}")
        path = cell / "EVALUATION.json"
        native = base.read(path)["rows"]
        if len(native) != len(expected) or {native_key(r) for r in native} != expected:
            raise ValueError(f"普通 MAPPO 原生评估键缺失或重复：{cell}")
        refs[seed] = (cell, native)
        ck = cell / "checkpoints" / f"{STEPS:09d}.json"
        base.verified(cell, base.read(ck))
        provenance[str(seed)] = dict(cell=str(cell.resolve()), evaluation_sha=base.digest(path),
                                     final_pointer_sha=base.digest(ck))
    return refs, provenance


def observed_env(gym, np, cfg):
    make = base.build_gym(gym, np)
    parent = type(make(cfg, (), "MAPPO"))

    class ObservedPhysics(PriorityCorridor):
        # 只计数真实转移，不扫描未来乘客、不改变预约或调度。
        def reset(self, requests=None):
            self.audit = Counter()
            return super().reset(requests)

        def _push(self, tick, fn, *args):
            if fn.__name__ == "_turned":
                self.audit["turn_started"] += 1
            elif fn.__name__ == "_charged":
                self.audit["charge_started"] += 1
            return super()._push(tick, fn, *args)

        def _dispatch(self, a, dest, riders=()):
            outcome = super()._dispatch(a, dest, riders)
            if outcome == "reserved":
                self.audit["service_reserved" if riders else "empty_reserved"] += 1
            return outcome

        def _launch(self, a, *args):
            self.audit["service_launched" if a.riders else "empty_launched"] += 1
            return super()._launch(a, *args)

        def _land(self, a):
            self.audit["service_landed" if a.riders else "empty_landed"] += 1
            super()._land(a)
            self.audit["ready_after_landing"] += a.state == "ready"

        def _turned(self, a):
            self.audit["turn_completed"] += 1
            super()._turned(a)
            self.audit["ready_after_turn"] += a.state == "ready"

        def _charged(self, a, target):
            self.audit["charge_completed"] += 1
            super()._charged(a, target)
            self.audit["ready_after_charge"] += a.state == "ready"

    class Observed(parent):
        def __init__(self, cfg, rows, method):
            super().__init__(cfg, rows, method)
            self.env = ObservedPhysics(cfg, policy_returns=True)

        def eligible(self):
            return self.focal() is not None if self.phase == 0 else bool(self.env.targets())

        def reset(self, **kwargs):
            x, info = super().reset(**kwargs)
            self.last_audit = Counter()
            return x, dict(info, eligible=self.eligible())

        def step(self, action):
            role = self.phase
            before = (self.stat["air_dispatched"], self.stat["air_charge_only"])
            x, r, done, truncated, info = super().step(action)
            if role == 1:
                events = self.env.audit - self.last_audit
                self.last_audit = self.env.audit.copy()
                info = dict(info, dispatched=self.stat["air_dispatched"] - before[0],
                            charge_only=self.stat["air_charge_only"] - before[1], lifecycle=dict(events),
                            busy=sum(a.state != "ready" for a in self.env.planes.values()))
            legal = (() if done or truncated or self.phase != 1 else self.env.targets())
            eligible = not (done or truncated) and (bool(legal) if self.phase == 1 else self.focal() is not None)
            return x, r, done, truncated, dict(info, eligible=eligible, air_targets=len(legal))

    return Observed, make


def model_types(torch):
    from torch import nn
    from torch.nn import functional as F
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    width = 64

    def linear(inp, out, activation="leaky_relu"):
        layer = nn.Linear(inp, out)
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain(activation))
        nn.init.zeros_(layer.bias)
        return layer

    class History(nn.Module):
        # ACAC：双 MLP 后接单层 GRU；只编码本角色有效决策，其他分钟保持上一状态。
        def __init__(self, dim, time_embedding):
            super().__init__()
            self.te = time_embedding
            self.body = nn.Sequential(linear(dim + 4 * time_embedding, width), nn.LeakyReLU(),
                                      linear(width, width), nn.LeakyReLU())
            self.gru = nn.GRU(width, width, batch_first=True)
            self.register_buffer("freq", torch.exp(-math.log(10000.) * torch.arange(0, 4, 2) / 4))

        def features(self, x, clock):
            if self.te:
                p = clock.unsqueeze(-1) * self.freq
                x = torch.cat((x, torch.stack((p.sin(), p.cos()), dim=-1).flatten(-2)), dim=-1)
            return self.body(x)

        def one(self, x, clock, active, h):
            result = self.gru(self.features(x, clock)[:, None], h[None])[0][:, 0]
            return torch.where(active[:, None], result, h)

        def sequence(self, x, clock, active, h, positions, lengths_cpu):
            # 每段不跨 episode；压紧事件序列训练 GRU，再把状态映回分钟轴。
            b, t, _ = x.shape
            count = active.long().cumsum(1)
            if not len(positions):
                return h[:, None].expand(b, t, width)
            i, j = positions.unbind(-1)
            packed_x = x.new_zeros((b, t, width))
            packed_x[i, count[i, j] - 1] = self.features(x[i, j], clock[i, j])
            packed = pack_padded_sequence(packed_x, lengths_cpu, batch_first=True, enforce_sorted=False)
            encoded = self.gru(packed, h[None])[0]
            encoded = pad_packed_sequence(encoded, batch_first=True, total_length=t)[0]
            memory = torch.cat((h[:, None], encoded), 1)
            return memory.gather(1, count[..., None].expand(b, t, width))

    class Actor(nn.Module):
        def __init__(self, dim, te):
            super().__init__()
            self.history = History(dim, te)
            self.head = nn.Sequential(linear(width, width), nn.LeakyReLU(), linear(width, 3, "linear"))

    class Critic(nn.Module):
        def __init__(self, dim, te, attention):
            super().__init__()
            self.history = nn.ModuleList(History(dim, te) for _ in range(2))
            self.attention = attention
            if attention:
                # 角色 token 上的一层非因果 GPT 风格融合：位置编码、前置 LN、残差和 MLP。
                self.position = nn.Parameter(torch.empty(2, width))
                nn.init.normal_(self.position, std=.02)
                self.attn = nn.MultiheadAttention(width, 2, batch_first=True, dropout=0.)
                self.ln1, self.ln2, self.lnf = (nn.LayerNorm(width) for _ in range(3))
                self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"),
                                        nn.Linear(4 * width, width))
                self.value = linear(2 * width, 1, "linear")
            else:
                self.value = nn.Sequential(linear(2 * width, width), nn.LeakyReLU(),
                                           linear(width, 1, "linear"))

        def forward(self, p, a, diagnostics=False):
            x = torch.stack((p, a), dim=-2).reshape(-1, 2, width)
            weights = None
            if self.attention:
                x = x + self.position
                n = self.ln1(x)
                block, weights = self.attn(n, n, n, need_weights=diagnostics, average_attn_weights=False)
                x = x + block
                x = self.lnf(x + self.ff(self.ln2(x)))
            return self.value(x.flatten(1)).reshape(p.shape[:-1]), weights

    return Actor, Critic, width


def advantages(np, reward, values, done, truncated, decisions, gamma, lam, macro):
    """每列独立。宏观边界取合法决策并集，同一分钟的双角色只算一个。"""
    length, lanes = reward.shape
    result = np.zeros_like(reward, dtype=np.float64)
    keep = np.ones_like(done, dtype=bool) if not macro else decisions.copy()
    for lane in range(lanes):
        lo = 0
        for hi in [*(np.flatnonzero(done[:, lane]) + 1), length]:
            if hi <= lo:
                continue
            starts = list(np.flatnonzero(keep[lo:hi, lane]) + lo)
            carry = 0.
            right = hi
            for left in reversed(starts):
                # HARL 的 proper-time-limit 在最后一步令优势为零；保留之前累计的奖励。
                endpoint = right
                if right == hi and truncated[hi - 1, lane]:
                    endpoint = hi - 1
                if right == hi and done[hi - 1, lane] and not truncated[hi - 1, lane]:
                    bootstrap = 0.
                else:
                    bootstrap = values[endpoint, lane]
                duration = endpoint - left
                subtotal = np.dot(gamma ** np.arange(duration), reward[left:endpoint, lane])
                discount = gamma ** duration
                carry = subtotal + discount * bootstrap - values[left, lane] + discount * lam * carry
                result[left, lane] = carry
                right = left
            lo = hi
    return result.astype(np.float32), keep


def segments(np, done, block):
    length, lanes = done.shape
    chunks = []
    for lane in range(lanes):
        left = 0
        for end in [*(np.flatnonzero(done[:, lane]) + 1), length]:
            for start in range(left, end, block):
                chunks.append((lane, start, min(start + block, end)))
            left = end
    return chunks


def update(torch, np, actors, critic, opts, vc_loss, vn, data, args, macro, block):
    device = next(critic.parameters()).device
    values = vn.denormalize(data["value"].reshape(-1, 1)).reshape(data["value"].shape)
    micro, _ = advantages(np, data["reward"], values, data["done"], data["truncated"],
                          data["valid"].any(2), args["gamma"], args["gae_lambda"], False)
    macro_adv, macro_keep = advantages(np, data["reward"], values, data["done"], data["truncated"],
                                        data["valid"].any(2), args["gamma"], args["gae_lambda"], True)
    adv, train_value = (macro_adv, macro_keep) if macro else (micro, np.ones_like(data["done"]))
    ret = adv + values[:-1]
    normalized = np.zeros_like(data["logp"])
    for role in range(2):
        hit = data["valid"][:, :, role]
        a = adv[hit]
        if len(a):
            normalized[:, :, role][hit] = (a - a.mean()) / (a.std() + 1e-5)
    chunks = segments(np, data["done"], block)
    totals, counts = Counter(), Counter()

    def batches(times, minibatches):
        for _ in range(times):
            for ids in np.array_split(np.random.permutation(len(chunks)), minibatches):
                if not len(ids):
                    continue
                lane, start, end = np.asarray([chunks[i] for i in ids]).T
                t = start[:, None] + np.arange(block)
                exists = t < end[:, None]
                t = np.minimum(t, end[:, None] - 1)
                yield lane, start, t, exists

    def tensor(x):
        return torch.as_tensor(x, device=device)

    def roll(net, role, lane, start, t, exists, key):
        n = lane[:, None]
        hit = data["valid"][t, n, role] & exists
        return net.sequence(tensor(data["obs"][t, n, role]), tensor(data["clock"][t, n]),
                            tensor(hit), tensor(data[key][start, lane, role]),
                            tensor(np.argwhere(hit)), torch.as_tensor(hit.sum(1).clip(1)))

    for lane, start, t, exists in batches(args["ppo_epoch"], args["actor_num_mini_batch"]):
        for role in range(2):
            hit = exists & data["valid"][t, lane[:, None], role]
            if not hit.any():
                continue
            history = roll(actors[role].history, role, lane, start, t, exists, "actor_h")
            dist = torch.distributions.Categorical(logits=actors[role].head(history[tensor(hit)]))
            i, j = t[hit], np.broadcast_to(lane[:, None], t.shape)[hit]
            lp = dist.log_prob(tensor(data["action"][i, j, role]))
            ratio = (lp - tensor(data["logp"][i, j, role])).exp()
            goal = tensor(normalized[i, j, role])
            clip = args["clip_param"]
            loss = -torch.minimum(ratio * goal, ratio.clamp(1-clip, 1+clip) * goal).mean()
            objective = loss - args["entropy_coef"] * dist.entropy().mean()
            optimize(torch, opts[role], actors[role], objective, args)
            totals[f"actor_{role}"] += loss.item()
            counts[f"actor_{role}"] += 1
    for lane, start, t, exists in batches(args["critic_epoch"], args["critic_num_mini_batch"]):
        hit = exists & train_value[t, lane[:, None]]
        if not hit.any():
            continue
        p, a = (roll(critic.history[role], role, lane, start, t, exists, "critic_h") for role in range(2))
        # 航空角色的当分钟观测在乘客动作之后产生；不能把它泄漏进乘客动作前的 V。
        a = torch.cat((tensor(data["critic_h"][start, lane, 1])[:, None], a[:, :-1]), dim=1)
        prediction, weights = critic(p[tensor(hit)], a[tensor(hit)], diagnostics=not counts["attention"])
        i, j = t[hit], np.broadcast_to(lane[:, None], t.shape)[hit]
        loss = vc_loss.cal_value_loss(prediction[:, None], tensor(data["value"][i, j, None]),
                                      tensor(ret[i, j, None]), vn)
        optimize(torch, opts[2], critic, args["value_loss_coef"] * loss, args)
        totals["critic"] += loss.item()
        counts["critic"] += 1
        if weights is not None and not counts["attention"]:
            totals["attention_cross_role"] = weights[..., 0, 1].mean().item()
            counts["attention"] = 1
    hit = data["valid"].any(2)
    return dict(loss={k: v / max(1, counts[k]) for k, v in totals.items() if k != "attention_cross_role"},
                attention_cross_role=totals.get("attention_cross_role"),
                minibatches=dict(counts),
                mean_abs_macro_vs_micro=float(np.abs(macro_adv[hit] - micro[hit]).mean()) if hit.any() else None)


def optimize(torch, opt, net, loss, args):
    if not torch.isfinite(loss):
        raise FloatingPointError("更新损失不是有限数")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if args["use_max_grad_norm"]:
        torch.nn.utils.clip_grad_norm_(net.parameters(), args["max_grad_norm"], error_if_nonfinite=True)
    opt.step()


def checked_pointer(folder, meta, step):
    # 仅接受本格下的两份存档；避免恢复指针被误指到别的实验。
    if meta["real_steps"] != step or any(Path(meta[k]).name != meta[k] for k in ("model", "norm")):
        raise ValueError("恢复步数或存档文件名不合法")
    return base.verified(folder, meta)


def close_vector(vector, failed):
    try:
        vector.close()
    except (EOFError, BrokenPipeError, ConnectionResetError):
        if not failed:
            raise
        worker_vec = getattr(vector, "venv", vector)
        for worker in worker_vec.processes:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=2)


def train(folder, cli, contract, method, seed, cfg, rows, env_type, args, gym, np, torch):
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from harl.algorithms.critics.v_critic import VCritic
    from harl.common.valuenorm import ValueNorm

    te, attention, macro = METHODS[method]
    lane, length = base.LANES, base.LENGTH
    pointer = folder / "RESUME.json"
    meta = base.read(pointer) if pointer.is_file() else None
    start = meta["real_steps"] if meta else 0
    if start % base.ROLLOUT or not 0 <= start <= STEPS:
        raise ValueError("恢复点不在正式 rollout 边界")
    if meta:
        model_path, norm_path = checked_pointer(folder, meta, start)
    if start == STEPS:
        return
    base.set_seed(seed, np, torch)
    device = torch.device(cli.device)
    vector = SubprocVecEnv([lambda: env_type(cfg, rows, "MAPPO") for _ in range(lane)], start_method="spawn")
    norm = None
    failed = True
    try:
        norm = VecNormalize.load(str(norm_path), vector) if meta else VecNormalize(
            vector, norm_obs=True, norm_reward=False, gamma=args["gamma"])
        if not norm.norm_obs or norm.norm_reward:
            raise ValueError("归一化开关与 MAPPO 不一致")
        norm.training = True
        dim = norm.observation_space.shape[0]
        Actor, Critic, width = model_types(torch)
        actors = [Actor(dim, te).to(device) for _ in range(2)]
        critic = Critic(dim, te, attention).to(device)
        vn = ValueNorm(1, device=device)
        vc_loss = VCritic(args, norm.observation_space, device=device)
        opts = [torch.optim.Adam(net.parameters(), lr=args["lr" if i < 2 else "critic_lr"],
                                 eps=args["opti_eps"], weight_decay=args["weight_decay"])
                for i, net in enumerate([*actors, critic])]
        aggregate = Counter()
        gap_counts = [Counter(), Counter(), Counter()]
        busy_counts = Counter()
        if meta:
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            if (state["real_steps"], state["job"]) != (start, contract):
                raise ValueError("存档步数或 cell 合同不符")
            for net, opt, weight, optimizer in zip([*actors, critic], opts, state["weights"],
                                                    state["optimizers"], strict=True):
                net.load_state_dict(weight)
                opt.load_state_dict(optimizer)
            for k in VALUE_KEYS:
                getattr(vn, k).copy_(state["value_statistics"][k])
            aggregate.update(state["audit"])
            gap_counts = [Counter(x) for x in state["gaps"]]
            busy_counts.update(state["busy"])
            base.rng_restore(state["rng"], device, np, torch)
        obs = norm.reset()
        p_ok = np.array([x["eligible"] for x in vector.reset_infos], dtype=bool)
        clock = np.ones(lane, dtype=np.int32)
        actor_h = [torch.zeros((lane, width), device=device) for _ in range(2)]
        critic_h = [torch.zeros_like(actor_h[0]) for _ in range(2)]
        last = np.full((lane, 2), -1, dtype=np.int32)
        last_joint = np.full(lane, -1, dtype=np.int32)
        history_length = np.zeros((lane, 2), dtype=np.int32)
        real = start
        began = time.monotonic()
        block = int(args["data_chunk_length"])

        def audit_report():
            return dict(counters=dict(aggregate), legal_gap_minutes={
                            name:dict(counts) for name, counts in zip(("passenger", "aircraft", "joint"), gap_counts)},
                        busy_aircraft_at_minute_boundary=dict(busy_counts), real_steps=real,
                        no_op_history_update=False)

        def live(phase):
            base.save(folder / "LIVE.json", dict(real_steps=real, phase=phase,
                real_fps=(real-start)/max(1, time.monotonic()-began)))

        def record():
            state = dict(job=contract, real_steps=real, weights=[n.state_dict() for n in [*actors, critic]],
                         optimizers=[o.state_dict() for o in opts],
                         value_statistics={k: getattr(vn, k).detach().cpu() for k in VALUE_KEYS},
                         rng=base.rng_snapshot(np, torch), audit=dict(aggregate), gaps=[dict(x) for x in gap_counts],
                         busy=dict(busy_counts))
            base.save_generation(folder, real, STEPS, MARKS, norm, lambda p: torch.save(state, p), ".pt")
            base.save(folder / "ASYNC_AUDIT.json", audit_report())

        def t(array):
            return torch.as_tensor(array, device=device)

        @torch.no_grad()
        def sample(role, obs, valid):
            ah = actors[role].history.one(t(obs), t(clock), t(valid), actor_h[role])
            ch = critic.history[role].one(t(obs), t(clock), t(valid), critic_h[role])
            actor_h[role], critic_h[role] = ah, ch
            dist = torch.distributions.Categorical(logits=actors[role].head(ah))
            action = dist.sample()
            # 无机会调用仍走物理接口，不训练该样本；原环境规定这一动作无效。
            return action.cpu().numpy(), dist.log_prob(action).cpu().numpy()

        while real < STEPS:
            if (cli.output / "STOP_REQUESTED").exists() or shutil.disk_usage(cli.output).free < 5 * 2**30:
                raise InterruptedError("暂停标记或磁盘可用低于 5 GiB；保留上一个完整 rollout")
            for i, opt in enumerate(opts):
                opt.param_groups[0]["lr"] = args["lr" if i < 2 else "critic_lr"] * (1.-real/STEPS)
            shape = (length, lane)
            data = dict(obs=np.empty((*shape, 2, dim), np.float32),
                        actor_h=np.empty((*shape, 2, width), np.float32),
                        critic_h=np.empty((*shape, 2, width), np.float32),
                        valid=np.empty((*shape, 2), bool), clock=np.empty(shape, np.int32),
                        value=np.empty((length+1, lane), np.float32),
                        reward=np.empty(shape, np.float32), done=np.empty(shape, bool),
                        truncated=np.empty(shape, bool), logp=np.empty((*shape, 2), np.float32),
                        action=np.empty((*shape, 2), np.int64))
            for k in range(length):
                data["obs"][k, :, 0] = obs
                data["clock"][k] = clock
                data["actor_h"][k] = torch.stack(actor_h, 1).cpu().numpy()
                data["critic_h"][k] = torch.stack(critic_h, 1).cpu().numpy()
                p, pl = sample(0, obs, p_ok)
                with torch.no_grad():
                    data["value"][k] = critic(critic_h[0], critic_h[1])[0].cpu().numpy()
                mid, r0, d0, info0 = norm.step(p)
                if np.any(r0) or np.any(d0) or any(i["physical_increment"] != 0 for i in info0):
                    raise RuntimeError("乘客阶段不能产生奖励或推进物理时钟")
                air_ok = np.array([x["eligible"] for x in info0], dtype=bool)
                av, al = sample(1, mid, air_ok)
                nxt, r, done, info1 = norm.step(av)
                if any(i["physical_increment"] != 1 for i in info1):
                    raise RuntimeError("每对决策只允许推进一个原生真实步")
                data["obs"][k, :, 1] = mid
                data["valid"][k] = np.stack((p_ok, air_ok), axis=1)
                data["action"][k] = np.stack((p, av), axis=1)
                data["logp"][k] = np.stack((pl, al), axis=1)
                data["reward"][k], data["done"][k] = r, done
                data["truncated"][k] = [x.get("TimeLimit.truncated", False) for x in info1]
                for role, valid in enumerate((p_ok, air_ok)):
                    stem = ("passenger", "aircraft")[role]
                    aggregate[stem+"_calls"] += lane
                    aggregate[stem+"_legal"] += int(valid.sum())
                    aggregate[stem+"_no_opportunity"] += int((~valid).sum())
                    eligible = valid & (last[:, role] >= 0)
                    gap_counts[role].update(map(int, clock[eligible]-last[eligible, role]))
                    last[valid, role] = clock[valid]
                    history_length[:, role] += valid
                    aggregate[stem+"_history_max"] = max(aggregate[stem+"_history_max"],
                                                           int(history_length[:, role].max()))
                aggregate["route_executed"] += int(p_ok.sum())
                aggregate["dispatch_reserved"] += sum(x["dispatched"] for x in info1)
                aggregate["charge_only"] += sum(x["charge_only"] for x in info1)
                for info in info1:
                    aggregate.update(info["lifecycle"])
                    busy_counts[info["busy"]] += 1
                aggregate["joint_valid_minutes"] += int((p_ok | air_ok).sum())
                aggregate["same_minute_valid_both"] += int((p_ok & air_ok).sum())
                aggregate["aircraft_multiple_targets"] += sum(x["air_targets"] >= 2 for x in info0)
                joint = p_ok | air_ok
                repeated = joint & (last_joint >= 0)
                gap_counts[2].update(map(int, clock[repeated]-last_joint[repeated]))
                last_joint[joint] = clock[joint]
                mask = t((~done)[:, None])
                actor_h = [h * mask for h in actor_h]
                critic_h = [h * mask for h in critic_h]
                last[done] = -1
                last_joint[done] = -1
                history_length[done] = 0
                clock = np.where(done, 1, clock+1)
                p_ok = np.array([x["eligible"] if not d else vector.reset_infos[j]["eligible"]
                                 for j, (x, d) in enumerate(zip(info1, done, strict=True))], dtype=bool)
                obs = nxt
                real += lane
                if (k+1) % 256 == 0:
                    live("collect_real")
            with torch.no_grad():
                p = critic.history[0].one(t(obs), t(clock), t(p_ok), critic_h[0])
                data["value"][-1] = critic(p, critic_h[1])[0].cpu().numpy()
            live("update")
            metrics = update(torch, np, actors, critic, opts, vc_loss, vn, data, args, macro, block)
            record()
            base.save(folder / "updates" / f"{real:09d}.json", dict(real_steps=real,
                real_fps=(real-start)/max(1, time.monotonic()-began), **metrics))
            live("trained_rollout")
        failed = False
    finally:
        close_vector(norm if norm is not None else vector, failed)


def scalar_metrics(metrics):
    phase = metrics["phase_min_per_generated"]
    parts = (phase["queue"], phase["pad_wait"])
    return dict(mean_att_min=metrics["mean_att_min"], completion=metrics["completion"],
                fixed_cutoff_cost_min=metrics["fixed_cutoff_cost_min"],
                awt_min=sum(parts) if all(x is not None for x in parts) and metrics["all_requests_finished"] else None)


def pair_rows(rows, ref, key):
    refmap = {key(r): r for r in ref}
    if len(refmap) != len(ref) or len({key(r) for r in rows}) != len(rows) or {key(r) for r in rows} != set(refmap):
        raise ValueError("MAPPO 配对键不一致，拒绝产出不可比均值")
    pairs = []
    for r in rows:
        test, control = scalar_metrics(r["metrics"]), scalar_metrics(refmap[key(r)]["metrics"])
        pairs.append(dict(key=key(r), model=test, mappo=control,
                          delta={k: test[k]-control[k] if None not in (test[k], control[k]) else None
                                 for k in METRICS}))
    return pairs


def summarize(rows):
    result = {}
    for key in METRICS:
        vals = [scalar_metrics(row["metrics"])[key] for row in rows]
        result[key] = sum(vals)/len(vals) if vals and None not in vals else None
    return dict(n=len(rows), **result)


def evaluate(folder, cli, job, method, cfg, bank, env_type, baseline_make, reference, args, np, torch):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    Actor, _, width = model_types(torch)
    te = METHODS[method][0]
    cell, reference_native = reference

    def load(network_folder, mark, own):
        pointer = base.read(network_folder / "checkpoints" / f"{mark:09d}.json")
        model, normal = checked_pointer(network_folder, pointer, mark)
        state = torch.load(model, map_location="cpu", weights_only=False)
        if state["real_steps"] != mark or (own and state["job"] != job):
            raise ValueError("评估权重不是本格所需的正式存档")
        ctor = env_type if own else baseline_make
        vector = DummyVecEnv([lambda: ctor(cfg, bank[0][2], "MAPPO")])
        norm = VecNormalize.load(str(normal), vector)
        norm.training = norm.norm_reward = False
        if own:
            actors = [Actor(norm.observation_space.shape[0], te).eval() for _ in range(2)]
            for a, w in zip(actors, state["weights"][:2], strict=True):
                a.load_state_dict(w)
        else:
            actors, _, _ = base.build_pair("MAPPO", args, norm.observation_space, norm.action_space, torch.device("cpu"))
            for a, w in zip(actors, state["actors"], strict=True):
                a.actor.load_state_dict(w)
                a.prep_rollout()
        return norm, actors, ctor, pointer

    @torch.no_grad()
    def episode(bundle, rows, seed, pressure, own):
        norm, actors, ctor, _ = bundle
        base.set_seed(seed, np, torch)
        env = ctor(cfg, rows, "MAPPO")
        hidden = [torch.zeros((1, width)) for _ in range(2)]
        z = np.zeros((1, args["recurrent_n"], args["hidden_sizes"][-1]), np.float32)
        m = np.ones((1, 1), np.float32)

        def act(x):
            role = env.phase
            obs = norm.normalize_obs(x[None])
            if own:
                valid = torch.tensor([env.eligible()])
                minute = (env.env.now // env.env._ticks(cfg.decision_min)) + 1
                hidden[role] = actors[role].history.one(torch.from_numpy(obs), torch.tensor([minute]),
                                                       valid, hidden[role])
                return int(torch.distributions.Categorical(logits=actors[role].head(hidden[role])).sample().item())
            return int(actors[role].get_actions(obs, z, m)[0].item())

        try:
            x, _ = env.reset(seed=123)
            while not env.env.done:
                if pressure and env.phase == 0:
                    pending = sum(p.state == "route" for p in env.env.passengers.values())
                    for _ in range(max(0, pending-1)):
                        station = base.STATIONS[act(x)]
                        env.env._route(env.focal(), station)
                        env.stat["route_"+station] += 1
                        x = env.observation()
                x = env.step(act(x))[0]
            return dict(metrics=env.env.metrics(), control=dict(env.stat))
        finally:
            env.close()

    def cached(path, signature, make):
        if (cli.output / "STOP_REQUESTED").exists():
            raise InterruptedError("评估暂停，已经完成的回合保留")
        if path.exists():
            row = base.read(path)
            if row["signature"] != signature:
                raise ValueError(f"评估缓存合同不一致：{path}")
            return row["result"]
        row = make()
        base.save(path, dict(signature=signature, result=row))
        return row

    native, pressure_all = [], []
    for mark in MARKS:
        model = load(folder, mark, True)
        try:
            weights = [model[3]["model_sha"], model[3]["norm_sha"]]
            for index, (name, role, rows) in enumerate(bank):
                if role == "final" and mark != STEPS:
                    continue
                for seed in ((123, 124, 125) if role == "final" else (123, 124)):
                    native.append(cached(folder / "eval" / f"{mark}_{index}_{seed}.json", weights,
                        lambda: dict(trace=name, role=role, step=mark, policy_seed=seed,
                                     **episode(model, rows, seed, False, True))))
            if mark != STEPS:
                continue
            control = load(cell, STEPS, False)
            try:
                for rate in RATES:
                    rows_model, rows_control = [], []
                    for index, (name, role, rows) in enumerate(bank):
                        if role != "final":
                            continue
                        first = min(r.time_min for r in rows)
                        dense = validate_requests([replace(r, time_min=first+(r.time_min-first)/rate) for r in rows])
                        for seed in (123, 124, 125):
                            leaf = f"{rate:.1f}_{index}_{seed}.json"
                            meta = dict(trace=name, policy_seed=seed, rate=rate)
                            rows_model.append(cached(folder / "pressure" / leaf, weights,
                                lambda: dict(meta, **episode(model, dense, seed, True, True))))
                            ref_weights = [control[3]["model_sha"], control[3]["norm_sha"]]
                            rows_control.append(cached(cli.output / "reference_pressure" / cell.name / leaf,
                                ref_weights, lambda: dict(meta, **episode(control, dense, seed, True, False))))
                    pressure_all.append(dict(rate=rate, model=summarize(rows_model), mappo=summarize(rows_control),
                        pairs=pair_rows(rows_model, rows_control, lambda r: (r["rate"], r["trace"], r["policy_seed"]))))
                    print(f"{folder.name}：压力 {rate:.1f}× 评估完成", flush=True)
            finally:
                control[0].close()
        finally:
            model[0].close()
    if {native_key(r) for r in native} != expected_native(bank) or len(pressure_all) != len(RATES):
        raise ValueError("评估合同未完成")
    final = [row for row in native if row["role"] == "final"]
    base.save(folder / "EVALUATION.json", dict(rows=native, weights_unchanged=True, pressure=pressure_all,
             paired_native=pair_rows(native, reference_native, native_key)))
    return dict(evaluated_through_final=True, real_steps=STEPS,
                final=summarize(final), rates=[{k:v for k,v in row.items() if k!="pairs"} for row in pressure_all])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "train-trace", "harl-dir", "base-output", "transfer-output", "output"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    cli = p.parse_args(argv)
    cli.output = cli.output.resolve()
    for source in (cli.base_output.resolve(), cli.transfer_output.resolve()):
        if source == cli.output or source in cli.output.parents or cli.output in source.parents:
            p.error("不得在基线目录中写入或覆盖结果")
    with base.lock_run(cli.output):
        try:
            return run(cli)
        except Exception:
            base.save(cli.output / "ERROR.json", dict(error=traceback.format_exc()))
            raise


def run(cli):
    import gymnasium as gym
    import numpy as np
    import stable_baselines3 as sb3
    import torch

    torch.set_num_threads(1)
    if cli.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，无可见 CUDA 设备")
    versions = dict(torch=str(torch.__version__), numpy=np.__version__, sb3=sb3.__version__, gymnasium=gym.__version__)
    cfg, rows, bank, old = bank_for(cli, versions)
    args, _, vendor = base.harl_vendor(cli.harl_dir.resolve(), np, torch)
    refs, ref_index = references(cli, args, vendor, bank)
    if not all(args[k] for k in ("use_policy_active_masks", "use_proper_time_limits", "use_gae", "use_valuenorm")):
        raise ValueError("HARL 策略掩码/时间截断/归一化合同不合")
    if (base.LANES, base.LENGTH, args["gamma"], args["gae_lambda"], args["actor_num_mini_batch"],
            args["critic_num_mini_batch"]) != (5, 2048, .99, .95, 20, 20):
        raise ValueError("HARL MAPPO rollout 或更新参数不一致")
    manifest = canonical(dict(schema="acac_e2_legal_decision_event_mappo_v1", config_sha=base.digest(cli.config),
        train_source_sha=base.digest(cli.train_trace), source_sha=base.digest(Path(__file__)), base_protocol=old,
        harl_commit=base.HARL_COMMIT, harl_files=vendor, harl_args=args, refs=ref_index, methods=METHODS,
        seeds=SEEDS, marks=MARKS, rates=RATES, total=16*STEPS, device=cli.device,
        adaptation="ACAC 决策事件适配。不是原论文持久宏动作环境；物理和分钟调机不变。",
        critic="角色独立历史；非因果角色注意力；航空历史在乘客动作前只用上一物理步及更早的信息",
        clock="同一分钟两个角色机会合并；累计区间奖励；gamma^tau；lambda 每宏事件一次；rollout 末截断引导",
        optimizer="HARL MAPPO clip/熵/线性 LR；GRU 需短序列 TBPTT；HARL critic loss + ValueNorm",
        resume="恢复网络/优化器/ValueNorm/ObsNorm/RNG；环境、时钟、RNN 隐状态从新 episode 开始，不是逐比特续跑"))
    path = cli.output / "PROTOCOL.json"
    if path.exists() and base.read(path) != manifest:
        raise ValueError("已有实验合同不同：禁止混用恢复点")
    base.save(path, manifest)
    base.save(cli.output / "DEMAND_BANK.json", dict(train=[asdict(r) for r in rows],
              **{name:[asdict(r) for r in trace] for name, _, trace in bank}))
    observed, baseline_make = observed_env(gym, np, cfg)
    all_results = {}
    manifest_sha = base.digest(path)
    for method in METHODS:
        results = {}
        for seed in SEEDS:
            folder = cli.output / "cells" / f"{method}__s{seed}"
            job = dict(protocol_sha=manifest_sha, seed=seed, method=method, layer="E2", steps=STEPS)
            try:
                jp = folder / "JOB.json"
                if jp.exists() and base.read(jp) != job:
                    raise ValueError("已有训练格身份不一致")
                base.save(jp, job)
                base.save(folder / "CONFIG.json", dict(job, te=METHODS[method][0],
                          attention=METHODS[method][1], macro_gae=METHODS[method][2], harl_args=args))
                status = folder / "STATUS.json"
                if (cli.output / "STOP_REQUESTED").exists():
                    raise InterruptedError("已暂停")
                if status.exists():
                    done = base.read(status)
                    if done["job"] != job or not done["evaluated_through_final"] or done["real_steps"] != STEPS:
                        raise ValueError("完成标记与本格合同不符")
                    for step in MARKS:
                        checked_pointer(folder, base.read(folder / "checkpoints" / f"{step:09d}.json"), step)
                    if done["evaluation_sha"] != base.digest(folder / "EVALUATION.json"):
                        raise ValueError("已完成评估的 SHA 不符")
                else:
                    print(f"{folder.name}：开始或恢复", flush=True)
                    train(folder, cli, job, method, seed, cfg, rows, observed, args, gym, np, torch)
                    done = evaluate(folder, cli, job, method, cfg, bank, observed, baseline_make, refs[seed], args, np, torch)
                    done.update(job=job, evaluation_sha=base.digest(folder / "EVALUATION.json"))
                    base.save(status, done)
                results[str(seed)] = done
            except (KeyboardInterrupt, InterruptedError) as exc:
                print(f"{folder.name}：暂停 {exc}", flush=True)
                return 130
            except Exception:
                base.save(folder / "ERROR.json", dict(job=job, error=traceback.format_exc()))
                raise
        if set(results) != {str(s) for s in SEEDS}:
            raise RuntimeError("四种子未全齐，不推进下一方法")
        base.save(cli.output / "methods" / method / "METHOD_SUMMARY.json", results)
        all_results[method] = results
    base.save(cli.output / "SUMMARY.json", dict(cells=16, real_steps=16*STEPS, methods=all_results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
