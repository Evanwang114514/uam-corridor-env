#!/usr/bin/env python3
"""走廊 E2 冻结模型压力回放：54 个已清点存档、可选方法与倍率、逐回合恢复。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any

VERSION = "corridor_pressure_eval_v2_batch"


def matches_source_hash(path: Path, expected: str) -> bool:
    """只允许 LF/CRLF 差异；保留编码、空格、注释和全部代码内容。"""
    if not path.is_file():
        return False
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return expected.lower() in {
        sha256(raw).hexdigest(), sha256(lf).hexdigest(),
        sha256(lf.replace(b"\n", b"\r\n")).hexdigest(),
    }
# 固定本轮已经清点的 54 个存档，训练继续产生的新存档不会悄悄加入。
SNAPSHOT = {
    **{k: (2101,2102,2103,2104) for k in (
        "HAPPO_ROLE_CRITICS","MAPPO_ROLE_CRITICS","PR2_RESPONSE",
        "LA_REAL_UPDATE","LA_EVENT_REPLAY","LAPG_ADAPTED","LA_SELECTIVE","LA_DISPATCH_EQUIVALENT")},
    **{k: (2101,2102,2103) for k in ("F_FULL","A1_FORMAL_BEHAVIOR","A2_GAE","A4_MEAN_DYNAMICS")},
    "A3_FROZEN_WM": (2101,2102), "A5_EVENT_REWARD": (2102,),
    "P1_PETS": (2101,), "P2_GRU": (2101,2102), "P4_HYBRID": (2101,2102),
    "C1_TRUNCATED": (2101,2102),
}


def batch_environment(original):
    """同一分钟逐个选站；飞机仍只在原调机阶段决策一次。"""
    class Batch(original):
        def step(self, action):
            return super().step(action)

        def pressure_step(self, actors, norm, z, mask, torch):
            x = self.observation()
            if self.phase == 0:
                pending = sum(p.state == "route" for p in self.env.passengers.values())
                for _ in range(max(0, pending - 1)):
                    with torch.no_grad():
                        a = int(actors[0].get_actions(norm.normalize_obs(x[None]), z, mask)[0].item())
                    station = ("V0","V1","V3")[a]
                    self.env._route(self.focal(), station)
                    self.stat["route_" + station] += 1
                    x = self.observation()
            return x
    return Batch
E2 = "E2"
CORE_SCHEMA = "corridor_lookahead_phase_e2_v2"
OLD_SCHEMA = "corridor_dispatch_v2_10_v1"
DEFAULT_MARK = 1_996_800
REPO_REQUIRED = (
    "corridor_env.py",
    "corridor_dispatch.py",
    "train_eval_corridor_uagmc_happo_mappo.py",
    "train_eval_corridor_lookahead_l0l5.py",
    "train_corridor_dispatch_v2_10.py",
    "train_corridor_lookahead_core6.py",
    "train_corridor_lookahead_extensions8.py",
)
CELL = re.compile(r"^E2__(.+)__s(\d+)$")


def jread(p: Path) -> Any:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def jwrite(p: Path, value: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False, default=str)
    tmp.replace(p)


def checksum(p: Path) -> str:
    h = sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rate_tag(v: float) -> str:
    return "r" + (format(v, ".8f").rstrip("0").rstrip(".")).replace(".", "p")


def clean_rates(raw: list[float]) -> list[float]:
    out = []
    for v in raw:
        if not math.isfinite(v) or v <= 0:
            raise ValueError("倍率必须为有限正数；1.1 表示到达强度增加 10%")
        if not any(abs(v - a) < 1e-10 for a in out):
            out.append(v)
    if len({rate_tag(v) for v in out}) != len(out):
        raise ValueError("倍率在目录标签精度内重复")
    return out


def discover(sources: list[Path], scan_roots: list[Path], output: Path) -> list[Path]:
    collected = []
    for p in sources:
        p = p.expanduser().resolve()
        if (p / "PROTOCOL.json").is_file():
            collected.append(p)
        else:
            raise FileNotFoundError(f"不是原始训练输出根目录，缺 PROTOCOL.json: {p}")
    for root in scan_roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"搜索根目录不存在: {root}")
        for file in root.rglob("PROTOCOL.json"):
            collected.append(file.parent.resolve())
    result = []
    for p in collected:
        if p == output or output in p.parents or p in output.parents:
            if p == output or p in output.parents:
                raise ValueError(f"独立结果目录不能放在原训练目录内部：{output} / {p}")
            continue
        if p not in result:
            try:
                manifest = jread(p / "PROTOCOL.json")
            except (OSError, ValueError):
                continue
            if manifest.get("schema") in (OLD_SCHEMA, CORE_SCHEMA):
                result.append(p)
            elif p in [s.expanduser().resolve() for s in sources]:
                raise ValueError(f"不支持的训练输出合同 {p}: {manifest.get('schema')}")
    return result


@dataclass(frozen=True)
class Group:
    source: Path
    batch_id: str
    protocol: dict
    bank: dict
    category: str
    step: int


@dataclass(frozen=True)
class Job:
    group: Group
    folder: Path
    method: str
    seed: int
    checkpoint: Path


def identify_type(manifest: dict, methods: list[str], root: Path) -> str:
    if manifest.get("schema") == OLD_SCHEMA:
        return "old"
    if manifest.get("schema") != CORE_SCHEMA:
        raise ValueError(f"{root}: 不支持的 protocol schema")
    import train_corridor_lookahead_core6 as core
    import train_corridor_lookahead_extensions8 as ext
    selected = set(manifest.get("selected_arms", methods))
    if selected and selected <= set(core.ARMS) and not selected <= set(ext.ARMS):
        return "core"
    if selected and selected <= set(ext.ARMS) and not selected <= set(core.ARMS):
        return "extension"
    used = set(methods)
    if used and used <= set(core.ARMS):
        return "core"
    if used and used <= set(ext.ARMS):
        return "extension"
    raise ValueError(f"{root}: 方法归属不明确 {selected or used}")


def collect_groups(paths: list[Path], methods: set[str] | None,
                   seeds: set[int] | None) -> tuple[list[Group], list[Job], list[dict]]:
    groups: list[Group] = []
    jobs: list[Job] = []
    skipped = []
    for index, root in enumerate(paths, 1):
        proto = jread(root / "PROTOCOL.json")
        bank_path = root / "DEMAND_BANK.json"
        if not bank_path.is_file():
            skipped.append(dict(source=str(root), reason="缺 DEMAND_BANK.json"))
            continue
        bank = jread(bank_path)
        finals = [key for key in bank if key.startswith("final_")]
        if not finals:
            skipped.append(dict(source=str(root), reason="原需求银行缺 final_*"))
            continue
        entries = sorted((x for x in (root / "cells").glob("E2__*__s*") if x.is_dir()), key=lambda p: p.name)
        parsed = [(p, CELL.fullmatch(p.name)) for p in entries]
        methods_in_source = [m.group(1) for _, m in parsed if m]
        if not methods_in_source:
            skipped.append(dict(source=str(root), reason="无 E2 cells"))
            continue
        category = identify_type(proto, methods_in_source, root)
        step = int(proto.get("steps", proto.get("steps_per_cell", DEFAULT_MARK)))
        if step != DEFAULT_MARK:
            raise ValueError(f"{root}: 不是本轮冻结训练步数 1,996,800: {step}")
        label = f"batch{index:02d}_{root.name}"
        group = Group(root, label, proto, bank, category, step)
        groups.append(group)
        for folder, m in parsed:
            if not m:
                continue
            name, seed = m.group(1), int(m.group(2))
            if seed not in SNAPSHOT.get(name, ()):
                continue
            if methods and name not in methods:
                continue
            if seeds and seed not in seeds:
                continue
            ck = folder / "checkpoints" / f"{step:09d}.json"
            status = folder / "STATUS.json"
            if not status.is_file() or jread(status).get("evaluated_through_final") is not True:
                skipped.append(dict(cell=str(folder), reason="最终评估尚未完成"))
                continue
            if not ck.is_file():
                skipped.append(dict(cell=str(folder), reason="尚无完成训练的最终 checkpoint"))
                continue
            pointer = jread(ck)
            if any(not isinstance(pointer.get(k), str) or Path(pointer[k]).name != pointer[k] for k in ("model","norm")):
                raise ValueError(f"存档路径异常：{ck}")
            model, norm = folder / pointer.get("model", ""), folder / pointer.get("norm", "")
            if not model.is_file() or not norm.is_file():
                skipped.append(dict(cell=str(folder), reason="最终权重或 VecNormalize 文件缺失"))
                continue
            jobs.append(Job(group, folder, name, seed, ck))
    chosen = {}
    for job in jobs:
        key = (job.method, job.seed)
        if key in chosen:
            left = jread(chosen[key].checkpoint)
            right = jread(job.checkpoint)
            if (left["model_sha"],left["norm_sha"]) != (right["model_sha"],right["norm_sha"]):
                raise ValueError(f"同一方法种子有不同完整模型，请用 --source 明确来源：{key}")
            skipped.append(dict(cell=str(job.folder),reason="相同权重重复来源，已去重"))
        else:
            chosen[key] = job
    return groups, list(chosen.values()), skipped


def cfg_from_dict(blob: dict):
    from corridor_env import Config, Region, Station
    raw = dict(blob)
    raw["stations"] = {
        k: Station(**{**value, "xy": tuple(value["xy"])})
        for k, value in raw["stations"].items()
    }
    for k in ("origins", "destinations"):
        raw[k] = tuple(Region(**{**value, "box": tuple(value["box"])}) for value in raw[k])
    return Config(**raw).validate()


def base_config(group: Group):
    p = group.protocol
    value = p.get("config") if group.category == "old" else p.get("configs", {}).get(E2)
    if value is None:
        raise ValueError(f"源数据合同缺 E2 物理配置: {group.source}")
    cfg = cfg_from_dict(value)
    if cfg.tick_min != 0.1 or cfg.decision_min != 1 or cfg.fleet != 48:
        raise ValueError(f"非预期 E2 tick、决策步长或机队: {group.source}")
    return cfg


def requests_from_rows(rows: list[dict], multiplier: float):
    from corridor_env import Request, validate_requests
    if not rows:
        raise ValueError("最终需求库为空")
    initial = min(float(r["time_min"]) for r in rows)
    result = [
        Request(
            id=str(r["id"]),
            time_min=initial + (float(r["time_min"]) - initial) / multiplier,
            origin=tuple(r["origin"]),
            destination=tuple(r["destination"]),
        )
        for r in rows
    ]
    return validate_requests(result)


def bank_for_rate(group: Group, r: float, cfg) -> tuple[list[tuple], str]:
    from corridor_env import validate_requests
    bank = []
    for name in sorted(k for k in group.bank if k.startswith("final_")):
        rows = requests_from_rows(group.bank[name], r)
        if len(rows) > 300:
            raise ValueError(f"{group.source}/{name}: 原观察最多容纳 300 乘客")
        if max(x.time_min for x in rows) >= cfg.cutoff_min:
            raise ValueError(f"{group.source}/{name}: 压缩后出现截止外请求")
        bank.append((name, "final", validate_requests(rows)))
    signature = sha256(json.dumps(
        {name: [asdict(row) for row in requests] for name, _, requests in bank},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return bank, signature


def hlink_or_copy(src: Path, dest: Path) -> None:
    if dest.exists():
        if checksum(src) != checksum(dest):
            raise ValueError(f"已有压力评估文件与原权重不一致：{dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)
    if checksum(dest) != checksum(src):
        raise IOError(f"复制或硬链接权重校验失败：{src}")


def prepare_stage(job: Job, out: Path, rate: float, bank_hash: str) -> Path:
    stage = out / rate_tag(rate) / job.group.batch_id / "cells" / job.folder.name
    ck_data = jread(job.checkpoint)
    model_src = job.folder / ck_data["model"]
    norm_src = job.folder / ck_data["norm"]
    for path, key in ((model_src, "model_sha"), (norm_src, "norm_sha")):
        if checksum(path) != ck_data[key]:
            raise ValueError(f"原存档文件哈希校验失败，拒绝评估：{path}")
    assert not stage.resolve().is_relative_to(job.group.source.resolve()), "禁止写入原训练输出目录"
    stage.mkdir(parents=True, exist_ok=True)
    record = {
        "version": VERSION,
        "source": str(job.folder.resolve()),
        "checkpoint_sha256": checksum(job.checkpoint),
        "model_sha256": ck_data["model_sha"],
        "norm_sha256": ck_data["norm_sha"],
        "rate": rate,
        "rate_tag": rate_tag(rate),
        "demand_sha256": bank_hash,
        "eval_seed_triplet": [123, 124, 125],
        "real_training_steps_added": 0,
        "routing": "batch",
        "evaluator_sha256": checksum(Path(__file__)),
        "runtime_code": {n: checksum(Path(__file__).with_name(n)) for n in REPO_REQUIRED},
    }
    marker = stage / "STRESS_CONTRACT.json"
    if marker.is_file() and jread(marker) != record:
        raise ValueError(f"此目录已有不同的压力评估合同，禁止覆盖：{stage}")
    hlink_or_copy(model_src, stage / ck_data["model"])
    hlink_or_copy(norm_src, stage / ck_data["norm"])
    hlink_or_copy(job.checkpoint, stage / "checkpoints" / job.checkpoint.name)
    if not marker.exists():
        jwrite(marker, record)
    return stage


def verify_code_provenance(group: Group) -> list[str]:
    messages = []
    proto = group.protocol
    expected = (
        proto.get("files", {}) if group.category == "old" else {
            "corridor_env.py": proto.get("environment_sha"),
            "corridor_dispatch.py": proto.get("dispatch_sha"),
        }
    )
    files = ("corridor_env.py", "corridor_dispatch.py", "train_corridor_dispatch_v2_10.py",
             "train_eval_corridor_uagmc_happo_mappo.py") if group.category == "old" else (
             "corridor_env.py", "corridor_dispatch.py")
    for filename in files:
        ref = expected.get(filename)
        actual = Path(__file__).resolve().with_name(filename)
        if filename == "train_corridor_dispatch_v2_10.py":
            # 原始和修复批入口不同；其依赖物理和共同 Actor 入口仍严格校验。
            continue
        if ref and not matches_source_hash(actual, ref):
            raise ValueError(f"{filename} 与来源 SHA256 不符，拒绝混用版本")
    if group.category != "old":
        for filename, key in (("train_eval_corridor_lookahead_l0l5.py","legacy_sha"),
                              ("train_corridor_lookahead_core6.py","core_sha")):
            ref = proto.get(key)
            if ref and not matches_source_hash(Path(__file__).with_name(filename), ref):
                raise ValueError(f"{filename} 与原训练实现不一致")
    return messages


def load_adapter(group: Group, device: str, harl_dir: Path):
    import gymnasium as gym
    import numpy as np
    import torch
    import train_eval_corridor_uagmc_happo_mappo as common
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    torch.set_num_threads(1)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 不可用")
    args = group.protocol.get("harl_args") if group.category == "old" else group.protocol.get("optimizer_contract")
    if not isinstance(args, dict):
        raise ValueError("合同缺 HARL 参数")
    _, _, vendor = common.harl_vendor(harl_dir.resolve(), np, torch)
    expected = group.protocol.get("harl") if group.category == "old" else None
    if expected is not None and vendor != expected:
        raise ValueError("HARL 作者实现与存档不一致")
    if group.category == "old":
        env_cls = batch_environment(common.build_gym(gym, np))
    else:
        import train_corridor_lookahead_core6 as core
        if group.category == "extension":
            import train_corridor_lookahead_extensions8 as plugin
        else:
            plugin = None

    def evaluate(stage, job, bank, cfg, outroot):
        job_meta = jread(job.folder/"JOB.json")
        recorded_vendor = job_meta.get("harl_files")
        if recorded_vendor is not None and recorded_vendor != vendor:
            raise ValueError("本格 HARL 文件指纹与当前作者实现不一致")
        if group.category == "old":
            backbone = "MAPPO" if job.method == "MAPPO_ROLE_CRITICS" else "HAPPO"
            ctor = env_cls
        else:
            backbone = "HAPPO"
            ctor = batch_environment(core.execution_class(gym, np, job.method, plugin))
        model_path, norm_path = common.verified(stage, jread(stage/"checkpoints"/f"{group.step:09d}.json"))
        vector = DummyVecEnv([lambda: ctor(cfg, bank[0][2], backbone)])
        norm = None
        try:
            norm = VecNormalize.load(str(norm_path), vector)
            norm.training = norm.norm_reward = False
            if group.category == "old":
                actors, _, _ = common.build_pair(backbone,args,norm.observation_space,norm.action_space,torch.device(device))
            else:
                actors, _, _ = core.legacy.build_pair(args,norm.observation_space,norm.action_space,torch.device(device))
            state = torch.load(model_path,map_location="cpu",weights_only=False)
            if int(state["real_steps"]) != group.step or len(state["actors"]) != 2:
                raise ValueError("模型步数或角色数量不符")
            if group.category == "old" and state.get("arm") != job.method:
                raise ValueError("模型组别不符")
            for actor, weights in zip(actors,state["actors"],strict=True):
                actor.actor.load_state_dict(weights,strict=True)
                actor.prep_rollout()
            z=np.zeros((1,args["recurrent_n"],args["hidden_sizes"][-1]),np.float32)
            mask=np.ones((1,1),np.float32)
            result=[]
            for name, role, rows in bank:
                for seed in (123,124,125):
                    destination=stage/"episodes"/f"{name}_{seed}.json"
                    if destination.exists():
                        result.append(jread(destination))
                        continue
                    if (outroot/"STOP_REQUESTED").exists():
                        raise KeyboardInterrupt("已暂停，原命令可继续")
                    common.set_seed(seed,np,torch)
                    env=ctor(cfg,rows,backbone)
                    x,_=env.reset(seed=123)
                    try:
                        while not env.env.done:
                            x=env.pressure_step(actors,norm,z,mask,torch)
                            with torch.no_grad():
                                action=int(actors[env.phase].get_actions(norm.normalize_obs(x[None]),z,mask)[0].item())
                            x=env.step(action)[0]
                        row=dict(trace=name,role=role,policy_seed=seed,step=group.step,
                                 metrics=env.env.metrics(),control=dict(env.stat))
                        jwrite(destination,row)
                        result.append(row)
                    finally:
                        env.close()
            return result
        finally:
            (norm if norm is not None else vector).close()
    return evaluate


def get_metrics(rows: list[dict], bank_size: int) -> tuple[list[dict], dict]:
    episodes = []
    for row in rows:
        if row.get("role") != "final":
            continue
        m = row["metrics"]
        p = m.get("phase_min_per_generated") or {}
        ctrl = row.get("control") or {}
        names = ("route", "access", "queue", "pad_wait", "takeoff", "flight", "landing", "egress")
        out = {
            "trace": row.get("trace"),
            "policy_seed": row.get("policy_seed"),
            "step": row.get("step"),
            "att_min": m.get("mean_att_min"),
            "cutoff_cost_min": m.get("fixed_cutoff_cost_min"),
            "completion": m.get("completion"),
            "generated": m.get("generated"),
            "finished": m.get("finished"),
            "air_decisions": ctrl.get("air_decisions"),
            "air_no_opportunity": ctrl.get("air_no_opportunity"),
            "air_multi_target": ctrl.get("air_multi_target"),
            "air_dispatched": ctrl.get("air_dispatched"),
        }
        out.update({f"phase_{name}_min": p.get(name) for name in names})
        if p.get("queue") is not None and p.get("pad_wait") is not None:
            out["air_wait_min"] = p["queue"] + p["pad_wait"]
        else:
            out["air_wait_min"] = None
        episodes.append(out)
    expected = bank_size * 3
    if len(episodes) != expected:
        raise ValueError(f"原生评估器返回 {len(episodes)} 个 Final 回合，预期 {expected}")
    keys = ("att_min", "cutoff_cost_min", "completion", "phase_route_min", "phase_access_min",
            "phase_queue_min", "phase_pad_wait_min", "phase_takeoff_min",
            "phase_flight_min", "phase_landing_min", "phase_egress_min", "air_wait_min")
    def avg(field):
        values = [v[field] for v in episodes]
        return sum(values) / len(values) if values and all(isinstance(x, (int, float)) and math.isfinite(x) for x in values) else None
    stats = {k: avg(k) for k in keys}
    stats.update(n_episodes=len(episodes), all_completed=all(r["completion"] == 1 for r in episodes),
                 full_att_defined=all(r["att_min"] is not None for r in episodes))
    return episodes, stats


def original_baseline(job: Job):
    eval_path = job.folder / "eval" / f"{job.group.step:09d}.json"
    if not eval_path.is_file():
        return None
    rows = jread(eval_path)
    matched = {}
    for row in rows:
        if row.get("role") == "final" and row.get("step") == job.group.step:
            key = (str(row.get("trace")), int(row.get("policy_seed")))
            matched[key] = row["metrics"].get("mean_att_min")
    return matched or None


def baseline_check(episodes: list[dict], original: dict, tolerance: float = 1e-6) -> dict:
    matches = []
    for row in episodes:
        key = (str(row["trace"]), int(row["policy_seed"]))
        if key not in original:
            continue
        expected, observed = original[key], row["att_min"]
        if expected is None or observed is None:
            continue
        matches.append(abs(observed - expected))
    return dict(paired=len(matches), max_abs_difference=max(matches) if matches else None,
                equal=bool(matches) and max(matches) <= tolerance if matches else None)


def aggregate_methods(per_seed: list[dict], paired: list[dict]) -> list[dict]:
    """先按独立训练种子聚合，评估回合不充当训练样本。"""
    grouped: dict[tuple, list[dict]] = {}
    for row in per_seed:
        grouped.setdefault((row["batch"], row["method"], row["rate"]), []).append(row)
    differences: dict[tuple, list[dict]] = {}
    for row in paired:
        differences.setdefault((row["batch"], row["method"], row["rate"]), []).append(row)
    fields = ("att_min", "cutoff_cost_min", "completion", "phase_route_min",
              "phase_queue_min", "phase_pad_wait_min", "air_wait_min")
    rows = []
    for (batch, method, rate), records in sorted(grouped.items()):
        item = dict(batch=batch, method=method, rate=rate,
                    train_seeds=len(records), seed_ids=",".join(str(r["train_seed"]) for r in records),
                    total_final_episodes=sum(r["n_episodes"] for r in records),
                    fully_completed_seeds=sum(bool(r["all_completed"]) for r in records))
        for field in fields:
            numbers = [float(r[field]) for r in records if r.get(field) is not None]
            item["mean_" + field] = sum(numbers)/len(numbers) if len(numbers)==len(records) else None
            mean = item["mean_" + field]
            item["sd_" + field] = (
                math.sqrt(sum((v-mean)**2 for v in numbers)/(len(numbers)-1))
                if mean is not None and len(numbers)>1 else None)
        prs = differences.get((batch, method, rate), [])
        item["matched_baseline_seeds"] = len(prs)
        for field in fields:
            vals = [r.get("delta_" + field) for r in prs]
            item["mean_delta_" + field] = (
                sum(vals)/len(vals) if vals and all(v is not None for v in vals) else None)
        rows.append(item)
    return rows


def to_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for item in rows for k in item))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: "" if v is None else v for k, v in row.items()})
    tmp.replace(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", default=[], type=Path,
                   help="原始训练输出根目录（含 PROTOCOL.json、DEMAND_BANK.json、cells），可重复")
    p.add_argument("--scan-root", action="append", default=[], type=Path,
                   help="自动搜索根目录之下的原始输出，可重复；只识别 A/B/core6/extensions8 合同")
    p.add_argument("--rates", nargs="+", type=float, default=[1.0, 1.1, 1.2],
                   help="需求到达强度倍率，例如 1 1.1 1.2；只压缩到达时刻，不增加乘客数量")
    p.add_argument("--output", required=False, type=Path, default=Path("pressure_evaluation"),
                   help="独立评估输出根目录，不能位于任何原始训练输出中")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                   help="仅推理，不训练；默认 cpu")
    p.add_argument("--harl-dir", type=Path, required=True, help="原训练使用的 HARL 根目录")
    p.add_argument("--reference", type=Path, required=True, help="刚才普通 HAPPO/MAPPO 的 baselines 输出目录")
    p.add_argument("--methods", nargs="*", default=None,
                   help="仅评估这些方法 ID；默认全部完成 checkpoint")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="仅评估指定训练种子，如 2101 2102；默认全部完成 checkpoint")
    p.add_argument("--dry-run", action="store_true", help="只读列出将运行的模型及回合数")
    p.add_argument("--fail-fast", action="store_true", help="遇到首个评估错误立即退出")
    p.add_argument("--strict-baseline", action="store_true",
                   help="若 1.0x 与原评估结果有差异，立即报错（默认仅警告）")
    p.add_argument("--max-cells", type=int, default=None, help="仅用于小规模工程 smoke-test")
    cli = p.parse_args(argv)

    if not cli.source and not cli.scan_root:
        p.error("请至少传入一个 --source 或 --scan-root")
    if cli.max_cells is not None and cli.max_cells <= 0:
        p.error("--max-cells 必须大于 0")
    try:
        rates = clean_rates(cli.rates)
        out = cli.output.expanduser().resolve()
        paths = discover(cli.source, cli.scan_root, out)
        if not paths:
            raise ValueError("未发现可识别的旧 A/B 或新 core6/extensions8 原始输出")
        for root in paths:
            if root == out or root in out.parents:
                raise ValueError("输出根目录必须独立于原训练目录")
        groups, jobs, skipped = collect_groups(paths,
            set(cli.methods) if cli.methods else None,
            set(cli.seeds) if cli.seeds else None)
        if cli.max_cells:
            jobs = jobs[:cli.max_cells]
        if not jobs:
            raise ValueError("未找到带最终权重和归一化文件的 E2 已完成训练格")
        print(f"原实验目录 {len(groups)} 个；可评估训练格 {len(jobs)}；压力 {rates}", flush=True)
        for grp in groups:
            total = sum(j.group.source == grp.source for j in jobs)
            print(f"  {grp.batch_id} [{grp.category}]：{total} 个最终 checkpoint", flush=True)
        print(f"预计 Final 回合：{len(jobs)} × {len(rates)} × 4 条需求 × 3 次随机采样 = {len(jobs)*len(rates)*12}", flush=True)
        for row in skipped:
            print(f"  [SKIP] {row.get('cell',row.get('source'))}: {row['reason']}", flush=True)
        if cli.dry_run:
            for j in jobs:
                print(f"  {j.group.batch_id} / {j.method} / s{j.seed} / {j.checkpoint.name}")
            return 0
        absent = [x for x in REPO_REQUIRED if not Path(__file__).resolve().with_name(x).is_file()]
        if absent:
            raise RuntimeError(f"本脚本须放在 GitHub 仓库根目录，缺依赖文件: {absent}")

        reference_bank = jread(cli.reference/"DEMAND_BANK.json")
        reference_cfg = jread(cli.reference/"PROTOCOL.json")["configs"]["E2"]
        # 所有比较使用同一份四条需求；发现来源银行不同就停止，不悄悄换需求。
        reference_final = {k:v for k,v in reference_bank.items() if k.startswith("final_")}
        if len(reference_final) != 4:
            raise ValueError("参照银行必须有四条最终需求")
        for group in groups:
            finals = {k:v for k,v in group.bank.items() if k.startswith("final_")}
            if finals != reference_final or asdict(base_config(group)) != asdict(cfg_from_dict(reference_cfg)):
                raise ValueError(f"来源与普通基线的需求或物理不一致：{group.source}")
        provenance = {str(g.source): verify_code_provenance(g) for g in groups}
        for source, warnings in provenance.items():
            for warning in warnings:
                print(f"[PROVENANCE WARNING] {source}: {warning}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        import train_eval_corridor_uagmc_happo_mappo as common
        output_lock = common.lock_run(out)
        jwrite(out / "STRESS_MANIFEST.json", {
            "version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
            "sources": [str(p) for p in paths], "rates": rates, "device": cli.device,
            "jobs": len(jobs), "pressure_interpretation": "compress arrival times only",
            "original_policy_sampling_seeds": [123, 124, 125],
            "physical_state_unchanged": True, "original_checkpoints_read_only": True,
            "code_hash_warnings": provenance,
            "routing": "batch：同刻逐个选站，飞机每分钟仍只决策一次",
        })

        adapters: dict[str, Any] = {}
        contexts: dict[str, tuple[Any, Any, str]] = {}
        summary_rows = []
        episode_rows = []
        errors = []
        start = time.monotonic()
        done = 0
        for rate in rates:
            for job in jobs:
                group = job.group
                key = str(group.source)
                label = f"[{done+1}/{len(jobs)*len(rates)}] {rate_tag(rate)} {group.batch_id}/{job.method}/s{job.seed}"
                try:
                    if key not in contexts:
                        cfg = base_config(group)
                        contexts[key] = (cfg, group.bank, group.category)
                    cfg = contexts[key][0]
                    bank, bank_hash = bank_for_rate(group, rate, cfg)
                    if len(bank) != 4:
                        raise ValueError(f"本轮要求 4 条最终需求，实际 {len(bank)}")
                    stage = prepare_stage(job, out, rate, bank_hash)
                    evaluation_root = out / rate_tag(rate) / group.batch_id
                    if key not in adapters:
                        adapters[key] = load_adapter(group, cli.device, cli.harl_dir)
                    print(f"{label} ...", flush=True)
                    raw = adapters[key](stage, job, bank, cfg, evaluation_root)
                    episodes, metrics = get_metrics(raw, len(bank))
                    repeated = None
                    if math.isclose(rate, 1.0, rel_tol=0, abs_tol=1e-10):
                        old = original_baseline(job)
                        if old:
                            repeated = baseline_check(episodes, old)
                            if not repeated["equal"]:
                                message = f"{label}: 1.0× 评估复算与旧评估不完全一致 {repeated}"
                                if cli.strict_baseline:
                                    raise ValueError(message)
                                print("[BASELINE WARNING] " + message, flush=True)
                    prefix = dict(batch=group.batch_id, category=group.category,
                                  method=job.method, train_seed=job.seed,
                                  rate=rate, multiplier=rate_tag(rate),
                                  checkpoint_step=group.step)
                    row = {**prefix, **metrics, "baseline_replay": repeated}
                    summary_rows.append(row)
                    episode_rows.extend({**prefix, **e} for e in episodes)
                    jwrite(stage / "STRESS_RESULT.json", row)
                    print(f"  done: ATT={metrics['att_min']} route={metrics['phase_route_min']} "
                          f"queue={metrics['phase_queue_min']} cutoff={metrics['cutoff_cost_min']} "
                          f"complete={metrics['all_completed']}", flush=True)
                except Exception:
                    error = dict(batch=group.batch_id, method=job.method, train_seed=job.seed,
                                 rate=rate, detail=traceback.format_exc())
                    errors.append(error)
                    print(f"[FAILED] {label}\n{error['detail']}", file=sys.stderr, flush=True)
                    if cli.fail_fast:
                        break
                finally:
                    done += 1
                    to_csv(out / "SUMMARY.csv", summary_rows)
                    to_csv(out / "EPISODES.csv", episode_rows)
                    jwrite(out / "ERRORS.json", errors)
                    print(f"Progress {done}/{len(jobs)*len(rates)}, elapsed {time.monotonic()-start:.0f}s", flush=True)
            if errors and cli.fail_fast:
                break

        indexed = {(r["batch"], r["method"], r["train_seed"], r["rate"]): r for r in summary_rows}
        paired = []
        for rec in summary_rows:
            if math.isclose(rec["rate"], 1.0, abs_tol=1e-10):
                continue
            base = indexed.get((rec["batch"], rec["method"], rec["train_seed"], 1.0))
            if not base:
                continue
            delta = {}
            for field in ("att_min", "cutoff_cost_min", "completion", "phase_route_min",
                          "phase_queue_min", "phase_pad_wait_min", "air_wait_min"):
                x, y = rec.get(field), base.get(field)
                delta["delta_" + field] = x-y if x is not None and y is not None else None
            paired.append(dict(batch=rec["batch"], method=rec["method"], seed=rec["train_seed"],
                               rate=rec["rate"], **delta))
        to_csv(out / "PAIRED_DELTAS.csv", paired)
        method_rows = aggregate_methods(summary_rows, paired)
        to_csv(out / "METHOD_AGGREGATE.csv", method_rows)
        jwrite(out / "RESULTS.json", dict(version=VERSION, summary=summary_rows,
                                         method_aggregate=method_rows,
                                         paired_deltas=paired, errors=errors, skipped=skipped))
        print(f"\nCompleted {len(summary_rows)}/{len(jobs)*len(rates)} evaluations; "
              f"{len(errors)} failures; 0 training steps; outputs: {out}", flush=True)
        print("重点查看 METHOD_AGGREGATE.csv、SUMMARY.csv、PAIRED_DELTAS.csv，确认 ATT 与 route/queue 分解。", flush=True)
        return 1 if errors else 0
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr, flush=True)
        if cli.fail_fast:
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
