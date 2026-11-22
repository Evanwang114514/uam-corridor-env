"""完整控制权 UAGMC_AIR 双种子与八条非学习基线的同需求压力回放。"""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import traceback

import train_eval_corridor_uagmc_happo_mappo as b
from corridor_env import Config, Region, Station, Request, validate_requests, baseline_episode, cav_direct_episode

RULES = ("SPF_SRC","STTF","QTTI","READY","BOTH_V0","BOTH_V1","BOTH_V3","CAV_DIRECT")


def source_matches(path, expected):
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return expected in {hashlib.sha256(x).hexdigest() for x in (raw,lf,lf.replace(b"\n",b"\r\n"))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="E2/baselines 原存档目录")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rates", nargs="+", type=float, default=[1.1,1.2,1.4,1.6,1.8])
    p.add_argument("--seeds", nargs="+", type=int, default=[2101,2102])
    p.add_argument("--models", nargs="+", choices=("UAGMC_AIR",), default=["UAGMC_AIR"])
    p.add_argument("--rules", nargs="*", choices=RULES, default=list(RULES))
    a = p.parse_args()
    if any(not math.isfinite(r) or r <= 0 for r in a.rates) or len(set(a.rates)) != len(a.rates):
        p.error("倍率须为不重复的有限正数")
    if len(set(a.seeds)) != len(a.seeds):
        p.error("种子不能重复")
    a.source, a.output = a.source.resolve(), a.output.resolve()
    if a.output == a.source or a.source in a.output.parents:
        p.error("输出不得写入原训练目录")
    proto = b.read(a.source/"PROTOCOL.json")
    if proto["dispatch_protocol"] != b.DISPATCH_PROTOCOL:
        p.error("不是修复调度权后的训练协议")
    for name,key in (("corridor_env.py","environment_sha"),("corridor_dispatch.py","dispatch_sha"),
                     (Path(b.__file__).name,"entry_sha")):
        if not source_matches(Path(__file__).with_name(name), proto[key]):
            p.error(f"源码与模型合同不一致：{name}")
    raw = dict(proto["configs"]["E2"])
    raw["stations"] = {k:Station(**{**v,"xy":tuple(v["xy"])}) for k,v in raw["stations"].items()}
    for k in ("origins","destinations"):
        raw[k] = tuple(Region(**{**v,"box":tuple(v["box"])}) for v in raw[k])
    cfg = Config(**raw).validate()
    raw_bank = b.read(a.source/"DEMAND_BANK.json")
    bank = {}
    for meta in proto["bank"]:
        if meta["role"] != "final":
            continue
        name = meta["name"]
        rows = [Request(q["id"],q["time_min"],tuple(q["origin"]),tuple(q["destination"])) for q in raw_bank[name]]
        h = hashlib.sha256(json.dumps([asdict(q) for q in rows],sort_keys=True).encode()).hexdigest()
        if h != meta["sha256"]:
            p.error(f"需求校验失败：{name}")
        t0 = min(q.time_min for q in rows)
        bank[name] = {r:validate_requests([replace(q,time_min=t0+(q.time_min-t0)/r) for q in rows]) for r in a.rates}
        if len(rows)>300 or any(max(q.time_min for q in rs)>=cfg.cutoff_min for rs in bank[name].values()):
            p.error("需求数量或到达时刻越界")
    if len(bank)!=4:
        p.error("要求四条原最终评估需求")
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    torch.set_num_threads(1)
    # 使用原双角色环境的真实类，而不是将工厂函数作为父类。
    factory = b.build_gym(gym,np)
    env_cls = type(factory(cfg,(), "HAPPO"))
    a.output.mkdir(parents=True,exist_ok=True)
    lock = b.lock_run(a.output)
    try:
        mark = proto["steps_per_cell"]
        checkpoints = {}
        for seed in a.seeds:
            folder = a.source/"cells"/f"E2__UAGMC_AIR__s{seed}"
            if not b.read(folder/"STATUS.json")["evaluated_through_final"]:
                p.error(f"尚未完成：{folder}")
            ck = b.read(folder/"checkpoints"/f"{mark:09d}.json")
            if any(Path(ck[k]).name != ck[k] for k in ("model","norm")):
                p.error("存档路径异常")
            model,norm = b.verified(folder,ck)
            checkpoints[seed] = (model,norm,ck)
        contract = dict(source=str(a.source),protocol=proto,rates=a.rates,seeds=a.seeds,
                        rules=a.rules,models=a.models,routing="batch",device="cpu",
                        evaluator_sha256=b.digest(Path(__file__)),
                        checkpoints={s:ck for s,(_,_,ck) in checkpoints.items()},
                        numpy=np.__version__,torch=str(torch.__version__),gym=gym.__version__)
        contract = json.loads(json.dumps(contract))
        target = a.output/"MANIFEST.json"
        if target.exists() and b.read(target)!=contract:
            p.error("已有不同回放合同，请换输出目录")
        b.save(target,contract)
        b.save(a.output/"DEMAND_BANK.json",{n:{str(r):[asdict(q) for q in rows] for r,rows in v.items()} for n,v in bank.items()})
        all_rows,errors = [],[]

        def episode(path,meta,run):
            if (a.output/"STOP_REQUESTED").exists():
                raise KeyboardInterrupt("暂停；原命令继续")
            try:
                row = b.read(path) if path.exists() else {**meta,**run()}
                if not path.exists():
                    b.save(path,row)
                all_rows.append(row)
            except Exception:
                error = dict(**meta,error=traceback.format_exc())
                errors.append(error)
                b.save(path.with_suffix(".ERROR.json"),error)
                print(error["error"],flush=True)

        for seed,(model_path,norm_path,_) in checkpoints.items():
            vec = DummyVecEnv([lambda:factory(cfg,next(iter(bank.values()))[a.rates[0]],"UAGMC_AIR")])
            norm = None
            try:
                norm = VecNormalize.load(str(norm_path),vec)
                norm.training = norm.norm_reward = False
                # 沿用原 LSTM extractor 和双离散单策略，不拆成两个 MARL Actor。
                model = PPO.load(model_path,device="cpu")
                for r in a.rates:
                    for name,traces in bank.items():
                        for policy_seed in (123,124,125):
                            meta = dict(method="UAGMC_AIR",train_seed=seed,rate=r,trace=name,policy_seed=policy_seed)
                            def run():
                                b.set_seed(policy_seed,np,torch)
                                env = env_cls(cfg,traces[r],"UAGMC_AIR")
                                x,_ = env.reset(seed=123)
                                try:
                                    while not env.env.done:
                                        pending = sum(q.state=="route" for q in env.env.passengers.values())
                                        for _ in range(max(0,pending-1)):
                                            action = model.predict(norm.normalize_obs(x[None]),deterministic=False)[0][0]
                                            station = b.STATIONS[int(action[0])]
                                            env.env._route(env.focal(),station)
                                            env.stat["route_"+station] += 1
                                            x = env.observation()
                                        action = model.predict(norm.normalize_obs(x[None]),deterministic=False)[0][0]
                                        env.step(int(action[0]))
                                        x = env.step(int(action[1]))[0]
                                    return dict(metrics=env.env.metrics(),control=dict(env.stat))
                                finally:
                                    env.close()
                            episode(a.output/"episodes"/f"UAGMC_AIR_s{seed}_r{r}_{name}_p{policy_seed}.json",meta,run)
                    print(f"UAGMC_AIR s{seed} {r}x 已处理",flush=True)
            finally:
                (norm if norm is not None else vec).close()
        for r in a.rates:
            for rule in a.rules:
                for name,traces in bank.items():
                    meta = dict(method=rule,train_seed=None,rate=r,trace=name,policy_seed=None)
                    def run_rule():
                        forced = rule.removeprefix("BOTH_") if rule.startswith("BOTH_") else None
                        settings = replace(cfg,initial_fleet={forced:cfg.fleet}) if forced else cfg
                        return cav_direct_episode(settings,traces[r]) if rule=="CAV_DIRECT" else baseline_episode(
                            settings,traces[r],"FORCED" if forced else rule,
                            dict.fromkeys(b.STATIONS,3),dict.fromkeys(b.STATIONS,30),forced)
                    # 确定性规则每条需求只运行一次，不伪造评估种子。
                    episode(a.output/"episodes"/f"{rule}_r{r}_{name}.json",meta,run_rule)
                print(f"{rule} {r}x 已处理",flush=True)
        summaries=[]
        for method in ("UAGMC_AIR",*a.rules):
            for r in a.rates:
                rows=[x for x in all_rows if x["method"]==method and x["rate"]==r]
                units=[]
                for seed in (a.seeds if method=="UAGMC_AIR" else [None]):
                    ms=[x["metrics"] for x in rows if x["train_seed"]==seed]
                    expected=12 if method=="UAGMC_AIR" else 4
                    if len(ms)!=expected:
                        continue
                    def avg(key):
                        vals=[m.get(key) for m in ms]
                        return mean(vals) if all(v is not None for v in vals) else None
                    units.append(dict(att=avg("mean_att_min"),completion=avg("completion"),
                                      cost=avg("fixed_cutoff_cost_min")))
                ats=[x["att"] for x in units]
                valid=(len(units)==(len(a.seeds) if method=="UAGMC_AIR" else 1)
                       and all(v is not None for v in ats))
                summaries.append(dict(method=method,rate=r,units=len(units),episodes=len(rows),
                                      expected_episodes=12*len(a.seeds) if method=="UAGMC_AIR" else 4,
                                      att_min=mean(ats) if valid else None,
                                      att_sd=stdev(ats) if valid and len(ats)>1 else None,
                                      completion=mean(x["completion"] for x in units) if units else None,
                                      cutoff_cost_min=mean(x["cost"] for x in units) if units else None))
        b.save(a.output/"RESULTS.json",dict(rows=all_rows,summary=summaries,errors=errors))
        with (a.output/"SUMMARY.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
        print(f"结束：成功 {len(all_rows)} 回合，失败 {len(errors)}；未新增训练步数",flush=True)
        return int(bool(errors))
    finally:
        lock.close()


if __name__=="__main__":
    raise SystemExit(main())
