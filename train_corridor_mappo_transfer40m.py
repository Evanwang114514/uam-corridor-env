"""E2：补齐 HAPPO/MAPPO 两种子，再将四种 Lookahead 方法迁到 MAPPO。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
from statistics import mean
from types import SimpleNamespace
import traceback

import train_eval_corridor_uagmc_happo_mappo as base
import train_corridor_lookahead_core6 as core
import train_corridor_lookahead_extensions8 as ext
from corridor_env import Request, validate_requests


STEPS = 1_996_800
SEEDS = (2101, 2102, 2103, 2104)
RATES = (1.0, 1.1, 1.2, 1.3, 1.4)
POLICY_SEEDS = (123, 124, 125)
BASELINES = ("HAPPO", "MAPPO")
TRANSFERS = ("C1_TRUNCATED", "A1_FORMAL_BEHAVIOR", "P4_HYBRID", "F_FULL")
MARKS = [204800, 501760, 1003520, 1597440, 1699840, 1802240, 1904640, STEPS]
VALUE_KEYS = ("running_mean", "running_mean_sq", "debiasing_term")


def canonical(value):
    return json.loads(json.dumps(value))


def trace_sha(rows):
    return hashlib.sha256(json.dumps([asdict(q) for q in rows], sort_keys=True).encode()).hexdigest()


def source_match(path, expected):
    # 只容许无语义变化的 LF/CRLF 转换，不绕过源代码校验。
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return expected in {hashlib.sha256(x).hexdigest() for x in (raw, lf, lf.replace(b"\n", b"\r\n"))}


def read_bank(root, cli, versions):
    protocol = base.read(root / "PROTOCOL.json")
    if (protocol["schema"], protocol["dispatch_protocol"], protocol["steps_per_cell"],
            protocol["rollout"], protocol["dev_marks"]) != (
            "corridor_legacy_interface_v2", base.DISPATCH_PROTOCOL, STEPS, base.ROLLOUT, MARKS):
        raise ValueError("旧基线并非本轮要求的调度权 v2 与冻结训练预算")
    names = {"entry_sha": Path(base.__file__), "environment_sha": Path(base.__file__).with_name("corridor_env.py"),
             "dispatch_sha": Path(base.__file__).with_name("corridor_dispatch.py")}
    for key, path in names.items():
        if not source_match(path, protocol[key]):
            raise ValueError(f"旧基线源码指纹不一致：{path.name}")
    for key, value in versions.items():
        if str(protocol[key]) != str(value):
            raise ValueError(f"训练依赖版本不一致：{key}，旧={protocol[key]}，当前={value}")
    cfg = base.config(cli.config, "E2")
    train = base.trace(base.config(cli.config, "E1"), cli.train_trace)
    if canonical(asdict(cfg)) != protocol["configs"]["E2"] or trace_sha(train) != protocol["train_trace_sha256"]:
        raise ValueError("E2 物理配置或 1.0 倍训练需求与旧基线不一致")
    raw = base.read(root / "DEMAND_BANK.json")
    bank = []
    for item in protocol["bank"]:
        name = item["name"]
        rows = tuple(Request(r["id"], r["time_min"], tuple(r["origin"]), tuple(r["destination"]))
                     for r in raw[name])
        if trace_sha(rows) != item["sha256"]:
            raise ValueError(f"评估需求指纹错误：{name}")
        bank.append((name, item["role"], rows))
    if [sum(role == x for _, role, _ in bank) for x in ("validation", "final")] != [4, 4]:
        raise ValueError("需要原基线的四条验证和四条最终需求")
    return cfg, train, bank, protocol


def make_pair(method, args, space, action_space, device):
    actors, critic, vn = base.build_pair(method, args, space, action_space, device)
    # HARL 某些设备迁移会让统计张量脱离 state_dict；注册同一张量，不改数值或公式。
    for key in VALUE_KEYS:
        if key not in vn.state_dict():
            tensor = getattr(vn, key)
            delattr(vn, key)
            vn.register_buffer(key, tensor)
    return actors, critic, vn


def independent_update(runner):
    # 与原走廊 MAPPO 一致：各角色使用同一份集中式优势，不传递前序策略比率。
    cb, vn = runner.critic_buffer, runner.value_normalizer
    adv = cb.returns[:-1] - vn.denormalize(cb.value_preds[:-1])
    report = []
    for actor, buf in zip(runner.actor, runner.actor_buffer, strict=True):
        buf.factor = None
        report.append(actor.train(buf, adv.copy(), "EP"))
    return report, runner.critic.train(cb, vn)


def transfer_engine(arm):
    plugin = ext if arm in ("C1_TRUNCATED", "P4_HYBRID") else None
    flags = (plugin.ARMS if plugin else core.ARMS)[arm]
    hooks = (plugin.models, plugin.fit_extension, plugin.imagine_extension, plugin) if plugin else ()
    ns = core.run({arm: flags}, __file__, None, *hooks, assemble_only=True)
    ns["build_pair"] = lambda args, space, action, dev: make_pair("MAPPO", args, space, action, dev)

    # 沿用既有 PPO clip、正式/行为概率校正和 V-trace；只移除 HAPPO 的跨角色前序乘积。
    # 联合行为比率仍只用于原 V-trace；真实 actor 损失使用本角色的行为修正。
    core.patch(core.legacy.offpolicy_update, ns, [
        ("    prefix=np.zeros_like(flat)\n", ""),
        ("pi0-mu+prefix", "pi0-mu"),
        ("""        with torch.no_grad():
            new=[]
            for j in range(0,n,2048):
                z=slice(j,j+2048)
                lp,_,_=actor.evaluate_actions(obs[z],zs[z],act[z],masks[z],None,None)
                new.append(lp.cpu().numpy())
        prefix+=np.concatenate(new)-mu
""", ""),
    ])
    return ns, plugin


def baseline_engine():
    # 不改历史文件；新批次只修复 ValueNorm 落盘并避免异常清理掩盖原始报错。
    ns = dict(vars(base), build_pair=make_pair)

    def close(vector):
        import sys
        if sys.exc_info()[0] is None:
            return vector.close()
        vec = getattr(vector, "venv", vector)
        for worker in vec.processes:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=2)
        for remote in vec.remotes:
            remote.close()
        vec.closed = True

    ns["close_owned"] = close
    return core.patch(base.train_cell, ns, [
        ("        (norm if norm is not None else vector).close()",
         "        close_owned(norm if norm is not None else vector)"),
        ('        real = start\n        while real < a.steps:', '''        real = start
        if meta:
            resume = state["continuation"]
            raw = np.stack([vector.env_method("restore_run", payload, indices=j)[0]
                            for j, payload in enumerate(resume["envs"])])
            norm.old_obs, norm.returns = raw, resume["returns"]
            obs, masks = resume["obs"], resume["masks"]
            rng_restore(state["rng"], device, np, torch)
            del state, resume
        while real < a.steps:'''),
        ('                             rng=rng_snapshot(np, torch))\n                save_generation(folder, real, a.steps, marks, norm, lambda p: torch.save(state, p), ".pt")',
         '''                             continuation=dict(envs=vector.env_method("snapshot_run"), obs=obs,
                                               masks=masks, returns=norm.returns.copy()),
                             rng=rng_snapshot(np, torch))
                save_generation(folder, real, a.steps, marks, norm, lambda p: torch.save(state, p), ".pt")
                print(f"{folder.name}：已保存 {real:,} / {a.steps:,} 真实步", flush=True)'''),
    ])


def resumable_base(factory, cfg):
    parent = type(factory(cfg, (), "HAPPO"))

    class ResumePhase(parent):
        def snapshot_run(self):
            # 仅用于真实训练恢复，从不向 Actor、世界模型或反事实分支暴露。
            return pickle.dumps((self.env, self.history, self.phase, self.route_index, self.stat), protocol=5)

        def restore_run(self, data):
            self.env, self.history, self.phase, self.route_index, self.stat = pickle.loads(data)
            return self.observation()

    return ResumePhase


def summary(rows):
    metrics = [r["metrics"] for r in rows]
    def avg(vals):
        return mean(vals) if vals and all(v is not None for v in vals) else None

    result = {k: avg([m[k] for m in metrics])
              for k in ("mean_att_min", "fixed_cutoff_cost_min", "completion")}
    phases = ("route", "access", "queue", "pad_wait", "takeoff", "flight", "landing", "egress")
    for k in phases:
        result[k + "_min"] = avg([m["phase_min_per_generated"][k] for m in metrics])
    queue, pad = result["queue_min"], result["pad_wait_min"]
    result["awt_min"] = queue + pad if None not in (queue, pad) else None
    return dict(n=len(rows), **result)


def run_pressure(cell, destination, cli, cfg, method, backbone, bank, ctor, args, np, torch):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    ck = base.read(cell / "checkpoints" / f"{STEPS:09d}.json")
    model_path, norm_path = base.verified(cell, ck)
    signature = {k: ck[k] for k in ("real_steps", "model_sha", "norm_sha")}
    if signature["real_steps"] != STEPS:
        raise ValueError("压力回放只允许正式最终权重")
    path = destination / "CHECKPOINT.json"
    if path.exists() and base.read(path) != signature:
        raise ValueError("压力回放的原始权重已变更")
    base.save(path, signature)
    vector = DummyVecEnv([lambda: ctor(cfg, bank[0][2], backbone)])
    norm = None
    try:
        norm = VecNormalize.load(str(norm_path), vector)
        norm.training = norm.norm_reward = False
        actors, _, _ = make_pair(backbone, args, norm.observation_space, norm.action_space, torch.device("cpu"))
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        if state["real_steps"] != STEPS:
            raise ValueError("模型内部步数不是正式最终步数")
        for actor, weights in zip(actors, state["actors"], strict=True):
            actor.actor.load_state_dict(weights)
            actor.prep_rollout()
        del state
        z = np.zeros((1, args["recurrent_n"], args["hidden_sizes"][-1]), np.float32)
        mask = np.ones((1, 1), np.float32)

        @torch.no_grad()
        def action(role, obs):
            return int(actors[role].get_actions(norm.normalize_obs(obs[None]), z, mask)[0].item())

        result = []
        for rate in RATES:
            rates = []
            for name, _, rows in bank:
                t0 = min(q.time_min for q in rows)
                dense = validate_requests([replace(q, time_min=t0 + (q.time_min - t0) / rate) for q in rows])
                for policy_seed in POLICY_SEEDS:
                    if (cli.output / "STOP_REQUESTED").exists():
                        raise InterruptedError("压力回放已在回合边界暂停")
                    file = destination / f"{rate:.1f}" / f"{name}_{policy_seed}.json"
                    if file.exists():
                        rates.append(base.read(file))
                        continue
                    base.set_seed(policy_seed, np, torch)
                    env = ctor(cfg, dense, backbone)
                    try:
                        x, _ = env.reset(seed=123)
                        while not env.env.done:
                            if env.phase == 0:
                                pending = sum(q.state == "route" for q in env.env.passengers.values())
                                # 已揭示乘客逐一选站，最后一位走原接口；飞机仍每分钟只决策一次。
                                for _ in range(max(0, pending - 1)):
                                    station = base.STATIONS[action(0, x)]
                                    env.env._route(env.focal(), station)
                                    env.stat["route_" + station] += 1
                                    x = env.observation()
                            x = env.step(action(env.phase, x))[0]
                        row = dict(trace=name, policy_seed=policy_seed, rate=rate,
                                   metrics=env.env.metrics(), control=dict(env.stat))
                        base.save(file, row)
                        rates.append(row)
                    finally:
                        env.close()
            result.append(dict(method=method, rate=rate, **summary(rates)))
            print(f"{cell.name} 压力 {rate:.1f}× 完成 {len(rates)} 回合", flush=True)
        return result
    finally:
        (norm if norm is not None else vector).close()


def write_progress(root):
    rows = []
    for f in sorted(root.glob("pressure/*/DONE.json")):
        rows.extend(base.read(f))
    if rows:
        path = root / "PRESSURE_SUMMARY.csv"
        temp = path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)
        lookup = {(r["method"], r["train_seed"], r["rate"]): r for r in rows}
        pairs = []
        for method in ("HAPPO", *("MAPPO_" + arm for arm in TRANSFERS)):
            for rate in RATES:
                matched = [(s, lookup[(method, s, rate)], lookup[("MAPPO", s, rate)]) for s in SEEDS
                           if (method, s, rate) in lookup and ("MAPPO", s, rate) in lookup]
                def delta(m, b, field):
                    return m[field] - b[field] if None not in (m[field], b[field]) else None
                details = [dict(seed=s, **{field: delta(m, b, field) for field in
                          ("mean_att_min", "fixed_cutoff_cost_min", "completion", "awt_min")})
                           for s, m, b in matched]
                avg = {}
                for field in ("mean_att_min", "fixed_cutoff_cost_min", "completion", "awt_min"):
                    vals = [r[field] for r in details]
                    avg[field] = mean(vals) if vals and None not in vals else None
                pairs.append(dict(method=method, baseline="MAPPO", rate=rate, n_pairs=len(details),
                                  four_seeds_ready=len(details) == len(SEEDS), mean_delta=avg, seeds=details))
        base.save(root / "PAIRED_VS_MAPPO.json", pairs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "train-trace", "harl-dir", "base-output", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--phase", choices=("all", "train", "eval"), default="all")
    cli = p.parse_args()
    cli.output = cli.output.resolve()
    cli.base_output = cli.base_output.resolve()
    if (cli.output == cli.base_output or cli.output in cli.base_output.parents
            or cli.base_output in cli.output.parents):
        p.error("新输出必须与旧只读基线目录独立")
    cli.steps, cli.save_every = STEPS, 5

    import gymnasium as gym
    import numpy as np
    import stable_baselines3 as sb3
    import torch

    torch.set_num_threads(1)
    if cli.device == "cuda" and not torch.cuda.is_available():
        p.error("指定了 CUDA 但当前环境不可用")
    versions = dict(torch=str(torch.__version__), sb3=sb3.__version__, gymnasium=gym.__version__, numpy=np.__version__)
    cfg, train, bank, old = read_bank(cli.base_output, cli, versions)
    args, happo_update, vendor = base.harl_vendor(cli.harl_dir.resolve(), np, torch)
    for method in BASELINES:
        for seed in SEEDS[:2]:
            job = base.read(cli.base_output / "cells" / f"E2__{method}__s{seed}" / "JOB.json")
            if (job.get("harl_args"), job.get("harl_files"), job.get("method"), job.get("seed")) != (
                    canonical(args), vendor, method, seed):
                raise ValueError(f"旧基线优化器或 HARL 来源不一致：{method}/{seed}")
    root = Path(__file__).resolve().parent
    dependencies = [Path(__file__).name, Path(base.__file__).name, Path(core.__file__).name,
                    Path(ext.__file__).name, Path(core.legacy.__file__).name, "corridor_env.py", "corridor_dispatch.py"]
    plan = [(method, seed, True) for method in BASELINES for seed in SEEDS[2:]]
    plan += [(method, seed, False) for method in BASELINES for seed in SEEDS[:2]]
    plan += [(arm, seed, True) for arm in TRANSFERS for seed in SEEDS]
    contract = canonical(dict(schema="e2_mappo_transfer40m_v1", source_protocol=old,
                              source_root=str(cli.base_output), args=args, harl=vendor, versions=versions,
                              sources={n: base.digest(root / n) for n in dependencies},
                              training_device=cli.device, pressure_device="cpu", plan=plan,
                              real_steps=sum(new for _, _, new in plan) * STEPS, rates=RATES,
                              replay="已揭示乘客批量选站，飞机每分钟一次", policy_seeds=POLICY_SEEDS))
    factory = resumable_base(base.build_gym(gym, np), cfg)
    final_bank = [row for row in bank if row[1] == "final"]
    trainer = baseline_engine()
    engines = {arm: transfer_engine(arm) for arm in TRANSFERS}
    failures = 0
    with base.lock_run(cli.output):
        file = cli.output / "PROTOCOL.json"
        if file.exists() and base.read(file) != contract:
            p.error("当前批次合同与已有输出不一致，请恢复原配置或另用输出目录")
        base.save(file, contract)
        base.save(cli.output / "DEMAND_BANK.json", dict(train=[asdict(q) for q in train],
                  **{name: [asdict(q) for q in rows] for name, _, rows in bank}))
        print(f"新增 20 格，{contract['real_steps']:,} 真实步；另复用 4 格旧基线", flush=True)
        for method, seed, new in plan:
            name = method if method in BASELINES else "MAPPO_" + method
            tag = f"E2__{name}__s{seed}"
            if (cli.output / "STOP_REQUESTED").exists():
                return 130
            cell = (cli.output if new else cli.base_output) / "cells" / tag
            dest = cli.output / "pressure" / tag
            try:
                print(f"{tag}：{'新增训练格' if new else '复用冻结基线'}", flush=True)
                if method in BASELINES:
                    backbone, ctor = method, factory
                    train_fn, eval_fn, update = trainer, base.eval_cell, happo_update
                else:
                    backbone = "MAPPO"
                    engine, plugin = engines[method]
                    ctor = core.execution_class(gym, np, method, plugin)
                    train_fn, eval_fn, update = engine["train_cell"], engine["eval_cell"], independent_update
                if new:
                    if cli.phase != "eval":
                        base.save(cell / "JOB.json", dict(method=name, mechanism=method, backbone=backbone,
                                  seed=seed, layer="E2", harl_args=args, harl_files=vendor))
                        train_fn(cell, cli, method, seed, cfg, train, MARKS, ctor, args, update, gym, np, torch)
                    if int(base.read(cell / "RESUME.json")["real_steps"]) != STEPS:
                        raise RuntimeError("本格没有完整正式权重")
                    if cli.phase == "train":
                        continue
                    eval_cli = SimpleNamespace(output=cli.output, device="cpu", steps=STEPS)
                    native = eval_fn(cell, eval_cli, method, cfg, bank, MARKS, ctor, args, np, torch)
                    final = [r for r in native if r["role"] == "final" and r["step"] == STEPS]
                    if len(final) != len(final_bank) * len(POLICY_SEEDS):
                        raise RuntimeError("常压最终评估回合不齐")
                    base.save(cell / "STATUS.json", dict(real_steps=STEPS, trained=True,
                              evaluated_through_final=True, final=summary(final),
                              late_validation=summary([r for r in native if r["role"] == "validation"
                                                       and r["step"] >= .8 * STEPS])))
                elif cli.phase == "train":
                    continue
                elif not base.read(cell / "STATUS.json")["evaluated_through_final"]:
                    raise RuntimeError("旧基线缺少最终评估")
                results = run_pressure(cell, dest, cli, cfg, name, backbone, final_bank, ctor, args, np, torch)
                base.save(dest / "DONE.json", [dict(train_seed=seed, **r) for r in results])
                write_progress(cli.output)
            except (KeyboardInterrupt, InterruptedError):
                print("已暂停，保留完整存档；重跑原命令继续。", flush=True)
                return 130
            except Exception:
                failures += 1
                detail = traceback.format_exc()
                base.save(cli.output / "errors" / f"{tag}.json", dict(error=detail))
                print(f"{tag} 失败，继续下一格：\n{detail}", flush=True)
        write_progress(cli.output)
        base.save(cli.output / "SUMMARY.json", dict(training_cells=20, replay_cells=24,
                  failures_this_call=failures, pressure_cells_written=len(list(cli.output.glob("pressure/*/DONE.json")))))
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
