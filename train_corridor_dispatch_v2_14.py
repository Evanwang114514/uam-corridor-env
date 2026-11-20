"""走廊 v2 十四组训练矩阵。未实现的组禁止进入正式训练。"""

import argparse
import hashlib
import json
import math
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path


SEEDS = (2101, 2102, 2103, 2104)
STEPS = 1_996_800
SLOTS = 7


@dataclass(frozen=True)
class Arm:
    name: str
    group: int
    backbone: str
    entry: str | None
    mechanism: str
    prerequisite: str = ""


# 两个进程使用同一个物理层；E1/E2 双层不是本轮 112M 合同。
LANE = {
    "A": (
        Arm("HAPPO_ROLE_CRITICS", 1, "HAPPO", "role_critics", "各角色独立价值网络、归一化和优势；保留 HAPPO 顺序比率"),
        Arm("HAPPO_TIGER", 2, "HAPPO", None, "旧 K3 TIGER 增量迁移", "核对旧 CF+TIGER 的实现与介入点，隔离 CF"),
        Arm("PR2_RESPONSE", 3, "HAPPO", None, "下一合法事件的候选动作条件队友响应", "因果分支回放与直接调用队友 Actor 的对照"),
        Arm("LA_REAL_UPDATE", 3, "HAPPO", None, "从真实更新估计队友策略变化，不拟合通用世界模型", "训练时钟锚点缓存和独立策略预测验收"),
        Arm("LA_EVENT_REPLAY", 3, "HAPPO", None, "从已揭示状态分支至下一合法事件，只更新影子策略", "可验证的状态克隆与未揭示需求隔离"),
        Arm("ACAC_ADAPTED", 4, "HAPPO", None, "角色历史编码、注意力 Critic 和异步 GAE", "真正的事件边界和角色轨迹；保留对齐版对照"),
        Arm("EVENT_ALIGNED_HAPPO", 4, "HAPPO", None, "READY 等合法事件触发；按角色记录实际时间区间", "READY 接管及无机会不采样、不计梯度"),
    ),
    "B": (
        Arm("MAPPO_ROLE_CRITICS", 1, "MAPPO", "role_critics", "各角色独立价值网络、归一化和优势；保留 MAPPO 并行更新"),
        Arm("MAPPO_TIGER", 2, "MAPPO", None, "与 HAPPO 共用相同 TIGER 模块", "核对旧 CF+TIGER 的实现与介入点，隔离 CF"),
        Arm("LAPG_ADAPTED", 3, "HAPPO", None, "学习感知辅助梯度，不替换真实 HAPPO", "先验证小型双角色任务的梯度估计"),
        Arm("LA_SELECTIVE", 3, "HAPPO", None, "只在多目标且预测有证据的状态采用前瞻", "合法性和可靠性门控；保存实际混合行为概率"),
        Arm("LA_DISPATCH_EQUIVALENT", 3, "HAPPO", None, "物理等价的调机动作合并计算 Critic 重要性权重", "验证派机、预约、排队、充电等完整后继等价；不替换 Actor 比率"),
        Arm("MAC_IAICC_ADAPTED", 4, "HAPPO", None, "角色各自宏动作区间的集中式价值估计", "可追溯的角色事件区间、终止和价值目标"),
        Arm("EVENT_ALIGNED_TEAM_COMMIT", 4, "HAPPO", None, "事件对齐加队友正在执行的承诺和跨角色交互", "事件底座及不泄露未来请求的承诺张量"),
    ),
}


def jobs(lane, names=None):
    arms = LANE[lane]
    if names is not None:
        chosen = set(names)
        unknown = chosen - {arm.name for arm in arms}
        if unknown:
            raise ValueError(f"不属于队列 {lane}：{sorted(unknown)}")
        arms = tuple(arm for arm in arms if arm.name in chosen)
    return [(arm, seed) for seed in SEEDS for arm in arms]


def roles_train(folder, a, backbone, seed, cfg, rows, marks, env_cls, args, gym, np, torch):
    import train_eval_corridor_uagmc_happo_mappo as b
    from harl.algorithms.critics.v_critic import VCritic
    from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
    from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
    from harl.common.valuenorm import ValueNorm
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    b.set_seed(seed, np, torch)
    meta = b.read(folder/"RESUME.json") if (folder/"RESUME.json").is_file() else None
    start = meta["real_steps"] if meta else 0
    if not 0 <= start <= STEPS or start % b.ROLLOUT:
        raise ValueError("恢复点不在真实 rollout 边界")
    if start == STEPS:
        b.verified(folder, meta)
        return
    vector = SubprocVecEnv([lambda: env_cls(cfg, rows, backbone) for _ in range(b.LANES)], start_method="spawn")
    norm = None
    try:
        if meta:
            model_path, norm_path = b.verified(folder, meta)
            norm = VecNormalize.load(str(norm_path), vector)
        else:
            norm = VecNormalize(vector, norm_obs=True, norm_reward=False, gamma=.99)
        norm.training = True
        device = torch.device(a.device)
        space, act = norm.observation_space, gym.spaces.Discrete(3)
        actors, v0, n0 = b.build_pair(backbone, args, space, act, device)
        critics = [v0, VCritic(args, space, device=device)]
        norms = [n0, ValueNorm(1, device=device)]
        buffers = [OnPolicyActorBuffer(args, space, act) for _ in actors]
        value_buffers = [OnPolicyCriticBufferEP(args, space) for _ in critics]
        vstats = ("running_mean", "running_mean_sq", "debiasing_term")
        if meta:
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            if state["real_steps"] != start or state["schema"] != "role_critics_v1":
                raise ValueError("存档模式或真实步数不匹配")
            for role in range(2):
                actors[role].actor.load_state_dict(state["actors"][role])
                actors[role].actor_optimizer.load_state_dict(state["optimizers"][role])
                critics[role].critic.load_state_dict(state["critics"][role])
                critics[role].critic_optimizer.load_state_dict(state["critic_optimizers"][role])
                norms[role].load_state_dict(state["value_norms"][role])
                for key in vstats:
                    getattr(norms[role], key).copy_(state["value_statistics"][role][key])
            b.rng_restore(state["rng"], device, np, torch)

        def checkpoint(real):
            state = dict(schema="role_critics_v1", real_steps=real,
                         actors=[x.actor.state_dict() for x in actors],
                         optimizers=[x.actor_optimizer.state_dict() for x in actors],
                         critics=[x.critic.state_dict() for x in critics],
                         critic_optimizers=[x.critic_optimizer.state_dict() for x in critics],
                         value_norms=[x.state_dict() for x in norms],
                         value_statistics=[{k: getattr(x, k) for k in vstats} for x in norms],
                         rng=b.rng_snapshot(np, torch))
            b.save_generation(folder, real, STEPS, marks, norm, lambda path: torch.save(state, path), ".pt")

        # 与 v2 恢复协议相同：更新边界恢复优化器与 RNG，重建环境回合。
        obs = norm.reset()
        zeros = np.zeros((b.LANES, 1, 128), np.float32)
        ones = np.ones((b.LANES, 1), np.float32)
        masks = ones.copy()
        real = start
        while real < STEPS:
            if (a.output/"STOP_REQUESTED").exists():
                raise InterruptedError("用户暂停")
            for actor, buf in zip(actors, buffers):
                actor.prep_rollout()
                actor.lr_decay(real, STEPS)
                buf.step, buf.factor = 0, None
            for critic, buf in zip(critics, value_buffers):
                critic.prep_rollout()
                critic.lr_decay(real, STEPS)
                buf.step = 0
            for k in range(b.LENGTH):
                # 两个价值网络读同一决策前状态。只拆价值估计，不给 Critic 额外信息或改时钟。
                with torch.no_grad():
                    values = [v.get_values(obs, zeros, masks)[0] for v in critics]
                    ap, lp, _ = actors[0].get_actions(obs, zeros, masks)
                buffers[0].obs[k], buffers[0].masks[k] = obs, masks
                mid, reward, done, infos = norm.step(ap.cpu().numpy().ravel())
                if any(done) or np.any(reward) or any(i["physical_increment"] != 0 for i in infos):
                    raise RuntimeError("乘客动作不能推进物理时钟")
                with torch.no_grad():
                    aa, la, _ = actors[1].get_actions(mid, zeros, masks)
                buffers[1].obs[k], buffers[1].masks[k] = mid, masks
                nxt, rewards, dones, infos = norm.step(aa.cpu().numpy().ravel())
                if any(i["physical_increment"] != 1 for i in infos):
                    raise RuntimeError("调机阶段每个环境只计一个真实步")
                next_masks = (~dones).astype(np.float32)[:, None]
                bad = np.asarray([[0. if i.get("TimeLimit.truncated") else 1.] for i in infos], np.float32)
                for buf, action, logp in zip(buffers, (ap, aa), (lp, la)):
                    buf.insert(nxt, zeros, action.cpu().numpy(), logp.cpu().numpy(), next_masks, ones)
                for buf, val in zip(value_buffers, values):
                    buf.share_obs[k] = obs
                    buf.insert(nxt, zeros, val.cpu().numpy(), rewards[:, None], next_masks, bad)
                obs, masks = nxt, next_masks
                real += b.LANES

            for critic, buf, vn in zip(critics, value_buffers, norms):
                with torch.no_grad():
                    next_value = critic.get_values(obs, zeros, masks)[0]
                buf.compute_returns(next_value.cpu().numpy(), vn)
                critic.prep_training()
            for actor in actors:
                actor.prep_training()
            for_adv = [buf.returns[:-1] - vn.denormalize(buf.value_preds[:-1])
                       for buf, vn in zip(value_buffers, norms)]
            if backbone == "HAPPO":
                factor = np.ones((b.LENGTH, b.LANES, 1), np.float32)
                for i in torch.randperm(2).tolist():
                    actor, buf = actors[i], buffers[i]
                    buf.update_factor(factor)
                    def flat(x):
                        return x.reshape(-1, *x.shape[2:])
                    available = None if buf.available_actions is None else flat(buf.available_actions[:-1])
                    inputs = (flat(buf.obs[:-1]), flat(buf.rnn_states[0:1]), flat(buf.actions),
                              flat(buf.masks[:-1]), available, flat(buf.active_masks[:-1]))
                    with torch.no_grad():
                        prior = actor.evaluate_actions(*inputs)[0]
                    actor.train(buf, for_adv[i].copy(), "EP")
                    with torch.no_grad():
                        posterior = actor.evaluate_actions(*inputs)[0]
                        factor = factor * torch.exp(posterior-prior).prod(-1).cpu().numpy().reshape(factor.shape)
            else:
                for actor, buf, adv in zip(actors, buffers, for_adv):
                    actor.train(buf, adv.copy(), "EP")
            for critic, buf, vn in zip(critics, value_buffers, norms):
                critic.train(buf, vn)
            if real in marks or real % (b.ROLLOUT*5) == 0 or (a.output/"STOP_REQUESTED").exists():
                checkpoint(real)
                print(f"{folder.name}：{real:,} / {STEPS:,} 真实步", flush=True)
            for buf in (*buffers, *value_buffers):
                buf.after_update()
    finally:
        (norm if norm is not None else vector).close()


def execute(a, selected):
    import gymnasium as gym
    import numpy as np
    import stable_baselines3 as sb3
    import torch
    import train_eval_corridor_uagmc_happo_mappo as b

    torch.set_num_threads(1)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；不会自动改成 CPU")
    env_cls = b.build_gym(gym, np)
    args, _, vendor = b.harl_vendor(a.harl_dir.resolve(), np, torch)
    cfg = b.config(a.config, a.layer)
    rows = b.trace(cfg, a.train_trace)
    bank = [(f"{role}_{seed}", role, b.derived_trace(rows, cfg, seed))
            for role, start in (("validation", 4301), ("final", 8301))
            for seed in range(start, start+4)]
    marks = sorted({math.ceil(STEPS*f/b.ROLLOUT)*b.ROLLOUT for f in (.1, .25, .5, .8, .85, .9, .95, 1.)})
    serialized = {"train": [asdict(q) for q in rows],
                  **{n: [asdict(q) for q in trace] for n, _, trace in bank}}
    signatures = {tuple(sorted((q.time_min, q.origin, q.destination) for q in t))
                  for t in [rows, *(t for _, _, t in bank)]}
    if len(signatures) != len(serialized):
        raise ValueError("训练、验证、最终轨迹有重复")
    entry = Path(__file__)
    contract = dict(schema="corridor_dispatch_v2_14_v1", layer=a.layer, lane=a.lane, config=asdict(cfg),
                    cells=[dict(arm=asdict(arm), seed=seed) for arm, seed in selected],
                    steps=STEPS, marks=marks, harl=vendor, harl_args=args,
                    device=a.device, versions=[torch.__version__, sb3.__version__, np.__version__, gym.__version__],
                    dispatch=b.DISPATCH_PROTOCOL, demand_sha=hashlib.sha256(json.dumps(serialized, sort_keys=True).encode()).hexdigest(),
                    files={p: b.digest(entry.with_name(p)) for p in (
                        entry.name, "corridor_env.py", "corridor_dispatch.py", "train_eval_corridor_uagmc_happo_mappo.py")})
    contract = json.loads(json.dumps(contract, sort_keys=True))
    failures = 0
    interrupted = False
    with b.lock_run(a.output):
        manifest = a.output/"PROTOCOL.json"
        if manifest.exists() and b.read(manifest) != contract:
            raise ValueError("恢复合同不一致；请勿混用物理层、方法或代码版本")
        b.save(manifest, contract)
        b.save(a.output/"DEMAND_BANK.json", serialized)
        print(f"{a.layer}/{a.lane}：{len(selected)} 格，{STEPS*len(selected):,} 真实步", flush=True)
        for arm, seed in selected:
            folder = a.output/"cells"/f"{a.layer}__{arm.name}__s{seed}"
            try:
                if (a.output/"STOP_REQUESTED").exists():
                    raise InterruptedError("用户暂停")
                b.save(folder/"JOB.json", dict(arm=asdict(arm), seed=seed, layer=a.layer))
                roles_train(folder, a, arm.backbone, seed, cfg, rows, marks, env_cls, args, gym, np, torch)
                results = b.eval_cell(folder, a, arm.backbone, cfg, bank, marks, env_cls, args, np, torch)
                final = [x for x in results if x["role"] == "final" and x["step"] == STEPS]
                if len(final) != 12:
                    raise RuntimeError(f"最终评估不完整：{len(final)}/12")
                b.save(folder/"STATUS.json", dict(state="COMPLETE", real_steps=STEPS, evaluated_through_final=True))
            except (InterruptedError, KeyboardInterrupt):
                interrupted = True
                print("已停止；保留更新边界的存档，重跑原命令可续训。", flush=True)
                break
            except Exception:
                failures += 1
                b.save(folder/"STATUS.json", dict(state="FAILED", evaluated_through_final=False,
                                                   error=traceback.format_exc()))
                print(f"{folder.name} 失败，继续下一格：\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        states = [{"cell": p.parent.name, **b.read(p)} for p in sorted(a.output.glob("cells/*/STATUS.json"))]
        b.save(a.output/"SUMMARY.json", dict(cells=states, planned=len(selected), failures=failures, paused=interrupted))
    return 1 if failures else 130 if interrupted else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="只输出矩阵登记，不加载训练依赖，不写结果")
    parser.add_argument("--lane", choices=tuple(LANE))
    parser.add_argument("--layer", choices=("E1", "E2"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-trace", type=Path)
    parser.add_argument("--harl-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--only", nargs="+", help="显式试跑已接线的组；不是正式完整矩阵")
    a = parser.parse_args(argv)
    a.steps = STEPS
    assert all(len(arms) == SLOTS for arms in LANE.values())
    assert len({x.name for arms in LANE.values() for x in arms}) == 2*SLOTS
    if a.plan:
        print(json.dumps(dict(seeds=SEEDS, steps_per_cell=STEPS, steps_per_lane=SLOTS*len(SEEDS)*STEPS,
                              lanes={k: [asdict(x) for x in LANE[k]] for k in LANE if not a.lane or k == a.lane}),
                         ensure_ascii=False, indent=2))
        return 0
    if not all((a.lane, a.layer, a.config, a.train_trace, a.harl_dir, a.output)):
        parser.error("训练需指定队列、物理层、配置、需求、HARL 与输出目录")
    try:
        selected = jobs(a.lane, a.only)
    except ValueError as e:
        parser.error(str(e))
    blocked = {arm.name: arm.prerequisite for arm, _ in selected if arm.entry != "role_critics"}
    if blocked:
        parser.error("正式队列尚未接齐；未启动任何格：\n" +
                     "\n".join(f"  {name}: {why}" for name, why in blocked.items()))
    return execute(a, selected)


if __name__ == "__main__":
    raise SystemExit(main())
