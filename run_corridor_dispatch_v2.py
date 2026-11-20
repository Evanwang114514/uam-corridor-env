"""每个进程跑一个环境的六臂正式矩阵，只从本批存档恢复。"""

import argparse
from pathlib import Path

import train_eval_corridor_lookahead_l0l5 as lookahead
import train_eval_corridor_uagmc_happo_mappo as baselines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", choices=("E1", "E2"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-trace", type=Path, required=True)
    parser.add_argument("--harl-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="批次根目录；环境与算法分目录")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    root = args.output.resolve() / args.layer
    common = ["--config", str(args.config.resolve()), "--train-trace", str(args.train_trace.resolve()),
              "--harl-dir", str(args.harl_dir.resolve()), "--device", args.device,
              "--layers", args.layer, "--seeds", "2101", "2102", "--steps", "1996800"]
    methods = ("UAGMC_AIR", "HAPPO", "MAPPO")
    arms = ("L1_FULL", "L4_DISCARD_SHADOW", "L5_NO_VTRACE")
    print(f"{args.layer}：六种方法×两个种子，23,961,600 真实步。输出：{root}", flush=True)
    failures = []
    for entry, group, flag, names in ((baselines, "baselines", "--methods", methods),
                                      (lookahead, "lookahead", "--arms", arms)):
        output = root / group
        if (root / "STOP_REQUESTED").exists() or (output / "STOP_REQUESTED").exists():
            print(f"批次暂停：{root}", flush=True)
            return 130
        code = entry.main([*common, "--output", str(output), flag, *names])
        summary = output / "SUMMARY.json"
        rows = baselines.read(summary)["cells"] if summary.is_file() else []
        if code or sum(bool(r.get("evaluated_through_final")) for r in rows) != 6:
            failures.append(group)
        if (output / "STOP_REQUESTED").exists():
            return 130
    if failures:
        print(f"有未完成分组：{', '.join(failures)}；重跑原命令从本批存档继续。", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
