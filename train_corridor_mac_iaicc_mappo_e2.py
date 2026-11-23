"""走廊 E2 Mac-IAICC-MAPPO 入口：当前冻结接口缺宏动作契约，正式训练拒绝启动。"""
from __future__ import annotations

import argparse
from pathlib import Path

import corridor_dispatch as dispatch
import corridor_env as physics
import train_eval_corridor_uagmc_happo_mappo as base

SEEDS = (2101, 2102, 2103, 2104)
STEPS = 1_996_800
METHODS = {
    "MAC_IAICC_FULL": ("individual_centralized", "role"),
    "NAIVE_MAC_IACC": ("shared", "joint"),
    "SHARED_CRITIC_ROLE_INTERVAL": ("shared_role_conditioned", "role"),
    "INDIVIDUAL_CRITIC_JOINT_INTERVAL": ("individual_centralized", "joint"),
}


def blocked_contract(args):
    cfg = base.config(args.config, "E2")
    return dict(
        status="BLOCKED_MACRO_CONTRACT",
        trained_cells=0,
        real_training_steps=0,
        planned_steps=len(METHODS)*len(SEEDS)*STEPS,
        methods=METHODS,
        seeds=SEEDS,
        dispatch_protocol=dispatch.DISPATCH_PROTOCOL,
        config_sha=base.digest(args.config),
        train_trace_sha=base.digest(args.train_trace),
        code_sha={Path(m.__file__).name: base.digest(Path(m.__file__))
                  for m in (base, dispatch, physics)},
        decision_min=cfg.decision_min,
        required_harl_commit=base.HARL_COMMIT,
        source_reference="https://github.com/yuchen-x/MacroMARL",
        adaptation="获批的后续版本必须保留各角色宏动作起止、团队奖励累计与 bootstrap，Actor 才可换 MAPPO PPO-Clip。",
        observed_contract=[
            dict(source="train_eval_corridor_uagmc_happo_mappo.py::build_gym/LegacyPhase.step",
                 fact="乘客选站后同分钟立即进入飞机优先级阶段；下一分钟又可发出新指令。"),
            dict(source="corridor_dispatch.py::PriorityCorridor.choose_returns",
                 fact="一次指令可预约多个并行返程，也可能仅充电；不记录独占角色动作或其完成事件。"),
            dict(source="corridor_env.py::CorridorEnv._dispatch/_land/_turned/_charged",
                 fact="资源有单机预约、着陆、周转和充电状态；单机 READY 不等于整个飞机角色结束。"),
            dict(source="corridor_env.py::CorridorEnv._route",
                 fact="乘客 Actor 是轮流分配不同旅客的调度器，而非持续执行同一个人的接驳任务。"),
        ],
        blockers=[
            "同角色的指令在已有任务未结束时可以重叠，不满足每角色同时执行一个持久宏动作的假设。",
            "没有可在线获得的角色级 start/done、联合下一边界，无法构造配对奖励累计和 bootstrap。",
            "用下一次合法决策、第一架飞机抵达或每分钟统一切分都会改变信用分配含义，不能冒充 Mac-IAICC。",
        ],
        required_interface=[
            "明确宏动作持有者：按旅客/飞机拆分，或批准角色忙时锁定新命令的控制合同。",
            "稳定任务 ID、开始和结束物理 tick、所有属于指令的并行任务、截尾/取消。",
            "为角色与联合时钟提供终点可见状态和未折扣团队奖励片段；不读取未揭示需求。",
            "如改变角色控制合同，需在同一接口另训匹配的普通 MAPPO，不能沿用现有基线为严格配对。",
        ],
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "train-trace", "harl-dir", "output"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = p.parse_args(argv)
    args.output = args.output.resolve()
    if args.output.name != "MAC_IAICC_E2_MAPPO_4METHOD":
        p.error("Mac-IAICC 使用独立目录 MAC_IAICC_E2_MAPPO_4METHOD")
    with base.lock_run(args.output):
        # 禁止用阻塞入口覆盖已有的任何正式运行或存档。
        if (args.output / "cells").exists() or (args.output / "PROTOCOL.json").exists():
            raise RuntimeError("目录有正式训练文件，拒绝覆盖")
        report = blocked_contract(args)
        base.save(args.output / "ASYNC_AUDIT.json", report)
        base.save(args.output / "SUMMARY.json", report)
        print("BLOCKED_MACRO_CONTRACT：现有 E2 不能忠实定义角色宏动作；未启动训练。", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
