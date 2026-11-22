"""冻结 E2 HAPPO/MAPPO 权重，配对回放到达率压力，不训练。"""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys
import traceback

import train_eval_corridor_uagmc_happo_mappo as base
from corridor_env import Request, validate_requests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for path in ("source", "config", "harl-dir", "output"):
        p.add_argument("--" + path, type=Path, required=True)
    p.add_argument("--rates", type=float, nargs="+", default=[1.1, 1.2, 1.4, 1.6, 1.8])
    p.add_argument("--methods", nargs="+", choices=("HAPPO", "MAPPO"), default=["HAPPO", "MAPPO"])
    p.add_argument("--seeds", type=int, nargs="+", default=[2101, 2102])
    a = p.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in a.rates) or len(set(a.rates)) != len(a.rates):
        p.error("倍率须为不重复的有限正数")
    if len(set(a.methods)) != len(a.methods) or len(set(a.seeds)) != len(a.seeds):
        p.error("方法和种子不能重复")
    protocol = base.read(a.source / "PROTOCOL.json")
    code = Path(base.__file__).resolve().parent
    for key, name in (("entry_sha", Path(base.__file__).name),
                      ("environment_sha", "corridor_env.py"), ("dispatch_sha", "corridor_dispatch.py")):
        if base.digest(code / name) != protocol[key]:
            p.error(f"源存档与源码不匹配：{name}")
    if protocol["dispatch_protocol"] != base.DISPATCH_PROTOCOL:
        p.error("调机协议不匹配")
    cfg = base.config(a.config, "E2")
    if json.loads(json.dumps(asdict(cfg))) != protocol["configs"]["E2"]:
        p.error("E2 物理参数与源存档不匹配")
    mark = protocol["steps_per_cell"]
    raw = base.read(a.source / "DEMAND_BANK.json")
    bank = {}
    for item in protocol["bank"]:
        if item["role"] != "final":
            continue
        name = item["name"]
        rows = [Request(r["id"], r["time_min"], tuple(r["origin"]), tuple(r["destination"])) for r in raw[name]]
        if hashlib.sha256(json.dumps([asdict(q) for q in rows], sort_keys=True).encode()).hexdigest() != item["sha256"]:
            p.error(f"需求存档校验失败：{name}")
        t0 = min(q.time_min for q in rows)
        dense = {v: validate_requests([replace(q, time_min=t0 + (q.time_min - t0) / v) for q in rows]) for v in a.rates}
        if not 1 < len(rows) <= 300 or any(max(q.time_min for q in d) >= cfg.cutoff_min for d in dense.values()):
            p.error(f"需求数量或时间范围不适合压力回放：{name}")
        bank[name] = dense
    if not bank:
        p.error("缺少冻结的最终评估轨迹")

    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    torch.set_num_threads(1)
    device = torch.device("cpu")
    env_class = base.build_gym(gym, np)
    args, _, vendor = base.harl_vendor(a.harl_dir.resolve(), np, torch)
    # 每格独立落盘；重跑只跳过相同回放合同下已完整写出的格。
    arms = tuple(f"{v:g}x_batch" for v in a.rates)
    manifest = dict(source=str(a.source.resolve()), protocol=protocol, arms=arms,
                    rates=a.rates, methods=a.methods, seeds=a.seeds,
                    evaluator_sha256=base.digest(Path(__file__)), device="cpu", policy_seeds=[123, 124, 125],
                    versions=dict(torch=str(torch.__version__), numpy=np.__version__, gymnasium=gym.__version__))
    a.output.mkdir(parents=True, exist_ok=True)
    lock = base.lock_run(a.output)
    try:
        path = a.output / "MANIFEST.json"
        if path.exists() and base.read(path) != json.loads(json.dumps(manifest)):
            p.error("输出目录已有不同的回放合同，请另选目录")
        base.save(path, manifest)
        base.save(a.output / "PAIRED_DEMAND.json", {
            name: {str(rate): [asdict(q) for q in rows] for rate, rows in traces.items()} for name, traces in bank.items()
        })
        all_rows, failures = [], 0
        for method in a.methods:
            for seed in a.seeds:
                cell = a.source / "cells" / f"E2__{method}__s{seed}"
                out = a.output / "cells" / cell.name
                try:
                    if not base.read(cell / "STATUS.json")["evaluated_through_final"]:
                        raise ValueError("原始格没有最终评估")
                    job = base.read(cell / "JOB.json")
                    if job["harl_files"] != vendor or job["harl_args"] != args:
                        raise ValueError("HARL 源码或参数与训练存档不匹配")
                    ck = base.read(cell / "checkpoints" / f"{mark:09d}.json")
                    model, norm_path = base.verified(cell, ck)
                    signature = dict(model=ck["model_sha"], norm=ck["norm_sha"])
                    path = out / "REPLAY.json"
                    if path.exists():
                        saved = base.read(path)
                        if saved["checkpoint"] != signature:
                            raise ValueError("源存档已改变，请换输出目录")
                        all_rows.extend(saved["rows"])
                        continue
                    vec = DummyVecEnv([lambda: env_class(cfg, next(iter(bank.values()))[a.rates[0]], method)])
                    norm = None
                    try:
                        norm = VecNormalize.load(str(norm_path), vec)
                        norm.training = norm.norm_reward = False
                        actors, _, _ = base.build_pair(method, args, norm.observation_space, norm.action_space, device)
                        state = torch.load(model, map_location="cpu", weights_only=False)
                        for actor, weights in zip(actors, state["actors"], strict=True):
                            actor.actor.load_state_dict(weights)
                            actor.prep_rollout()
                        zeros = np.zeros((1, 1, 128), np.float32)
                        ones = np.ones((1, 1), np.float32)

                        def act(role, obs):
                            with torch.no_grad():
                                action, _, _ = actors[role].get_actions(norm.normalize_obs(obs[None]), zeros, ones)
                            return int(action.item())

                        results = []
                        for rate, arm in zip(a.rates, arms):
                            for name, traces in bank.items():
                                for policy_seed in (123, 124, 125):
                                    base.set_seed(policy_seed, np, torch)
                                    env = env_class(cfg, traces[rate], method)
                                    x, _ = env.reset(seed=123)
                                    corridor = env.env
                                    peak = 0
                                    while not corridor.done:
                                        if env.phase == 0:
                                            pending = sum(q.state == "route" for q in corridor.passengers.values())
                                            peak = max(peak, pending)
                                            if arm.endswith("_batch"):
                                                # 只处理当前已揭示的乘客；不推进时钟、不增加飞机决策。
                                                # 最后一位仍走原接口，再进入原调机阶段。
                                                for _ in range(max(0, pending - 1)):
                                                    station = base.STATIONS[act(0, x)]
                                                    corridor._route(env.focal(), station)
                                                    env.stat["route_" + station] += 1
                                                    x = env.observation()
                                        x, _, _, _, _ = env.step(act(env.phase, x))
                                    results.append(dict(method=method, train_seed=seed, trace=name, arm=arm,
                                                        policy_seed=policy_seed, max_pending=peak,
                                                        metrics=corridor.metrics(), control=dict(env.stat)))
                            print(f"{cell.name} {arm} 回放完成", flush=True)
                    finally:
                        (norm if norm is not None else vec).close()
                    base.save(out / "REPLAY.json", dict(checkpoint=signature, rows=results))
                    all_rows.extend(results)
                except Exception:
                    failures += 1
                    base.save(out / "REPLAY_ERROR.json", dict(traceback=traceback.format_exc()))
                    traceback.print_exc()

        # ATT 缺省表示存在删失，不能用已完成者平均替代；配对原始明细保留于逐格 JSON。
        phase = ("route", "access", "queue", "pad_wait", "takeoff", "flight", "landing", "egress")
        metrics = ("completion", "mean_att_min", "fixed_cutoff_cost_min")
        fields = ["method", "train_seed", "arm", "episodes", *metrics, *phase, "awt_queue_plus_pad", "max_pending"]
        with (a.output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for method, seed in sorted({(r["method"], r["train_seed"]) for r in all_rows}):
                for arm in arms:
                    rows = [r for r in all_rows if (r["method"], r["train_seed"], r["arm"]) == (method, seed, arm)]
                    if not rows:
                        continue
                    ms = [r["metrics"] for r in rows]
                    vals = {k: mean(m[k] for m in ms) if all(m[k] is not None for m in ms) else None for k in metrics}
                    parts = {k: mean(m["phase_min_per_generated"][k] for m in ms) for k in phase}
                    writer.writerow(dict(method=method, train_seed=seed, arm=arm, episodes=len(rows), **vals, **parts,
                                         awt_queue_plus_pad=parts["queue"] + parts["pad_wait"],
                                         max_pending=max(r["max_pending"] for r in rows)))
        return int(bool(failures))
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
