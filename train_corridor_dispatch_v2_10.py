"""走廊 v2 十组修复矩阵；不改变分钟调度协议，不包含异步组。"""

import argparse
import copy
import heapq
import hashlib
import json
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path


SEEDS = (2101, 2102, 2103, 2104)
STEPS = 1_996_800
SLOTS = 5
BRANCH_HORIZON = 12
SHADOW_MIX = .25
BRANCHES_PER_ROLLOUT = 4


@dataclass(frozen=True)
class Arm:
    name: str
    group: int
    backbone: str
    entry: str | None
    mechanism: str


# 两个进程使用同一个物理层，不能将 E1、E2 混当成两个算法队列。
LANE = {
    "A": (
        Arm("HAPPO_ROLE_CRITICS", 1, "HAPPO", "role_critics", "各角色独立价值网络、归一化和优势；保留 HAPPO 顺序比率"),
        Arm("HAPPO_TIGER", 2, "HAPPO", "tiger", "旧 K3 的时间编码和跨角色注意力，移植到 Actor"),
        Arm("PR2_RESPONSE", 3, "HAPPO", "pr2", "预测候选选站后首个有效调机机会的条件响应"),
        Arm("LA_REAL_UPDATE", 3, "HAPPO", "evolution", "用相邻真实策略更新的差分构造影子策略"),
        Arm("LA_EVENT_REPLAY", 3, "HAPPO", "event", "真实分支回放，按队友实际策略边缘化后的价值更新乘客影子策略"),
    ),
    "B": (
        Arm("MAPPO_ROLE_CRITICS", 1, "MAPPO", "role_critics", "各角色独立价值网络、归一化和优势；保留 MAPPO 并行更新"),
        Arm("MAPPO_TIGER", 2, "MAPPO", "tiger", "与 HAPPO 共用同一时间交互模块，不叠加 CF"),
        Arm("LAPG_ADAPTED", 3, "HAPPO", "lapg", "配对回放队友更新前后的收益，对真实 HAPPO 加有界学习感知梯度"),
        Arm("LA_SELECTIVE", 3, "HAPPO", "selective", "有真实调机分歧且外推通过上一轮前瞻验证时启用"),
        Arm("LA_DISPATCH_EQUIVALENT", 3, "HAPPO", "equivalent", "与真实更新外推同底座，只在价值校正合并物理等价动作概率"),
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


def committed_copy(env):
    """只复制已揭示状态；分支中的完成条件只计算已生成乘客。"""
    e = env.env
    events, rows = e._events, env.rows
    try:
        e._events = [ev for ev in events if ev[2].__name__ != "_reveal"]
        heapq.heapify(e._events)
        env.rows = ()
        out = copy.deepcopy(env)
    finally:
        e._events, env.rows = events, rows
    out.env.request_count = len(out.env.passengers)
    return out


def physical_dispatch_key(e):
    # 调机统计不是物理后继。飞机、电量、队列、资源、预约和事件一并比较。
    return (e.now, e.done, e.finished, e.area_ticks, repr(e.planes), repr(e.passengers),
            repr(e.local), repr(e.pa), repr(e.queue), repr(e.turnq), repr(e.chargeq),
            repr(e.turn_active), repr(e.charge_active),
            tuple((s, repr(vars(pad))) for s, pad in e.pads.items()),
            tuple((t, fn.__name__, repr(args)) for t, _, fn, args in sorted(e._events)))


def experimental_env(env_cls, cfg, backbone):
    import train_eval_corridor_uagmc_happo_mappo as b
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from cloudpickle import dumps
    # 旧入口返回工厂而非类；实例化不重置环境、不启动模拟。
    base = type(env_cls(cfg, (), backbone))

    class Instrumented(base):
        def __init__(self, cfg, rows, backbone, hook):
            super().__init__(cfg, rows, backbone)
            self.hook = hook

        def branch_at_pre(self):
            if self.phase != 0 or self.env.done or self.focal() is None:
                return None
            return dumps(committed_copy(self))

        def step(self, action):
            p = self.phase
            x, r, end, cut, info = super().step(action)
            if p == 0 and self.hook in ("selective", "equivalent", "lapg"):
                legal = self.env.targets()
                info["air_targets"] = len(legal)
                if self.hook == "equivalent":
                    if len(legal) <= 1:
                        # 空机会或唯一候选的所有请求走同一物理派机路径。
                        info["dispatch_classes"] = (0, 0, 0)
                    else:
                        keys = []
                        for station in b.STATIONS:
                            shadow = committed_copy(self)
                            shadow.env.choose_returns(station, shadow.stat)
                            keys.append(physical_dispatch_key(shadow.env))
                        info["dispatch_classes"] = tuple(keys.index(key) for key in keys)
            return x, r, end, cut, info
    return Instrumented, SubprocVecEnv


def action_dist(actor, x, z, m, torch):
    # 只查询 HARL 分布，不额外消耗行为采样随机数。
    action = torch.zeros((len(x), 1), dtype=torch.long, device=actor.tpdv["device"])
    return actor.evaluate_actions(x, z, action, m, None, None)[2]


def class_mass(probs, classes, actions, np):
    taken = np.take_along_axis(classes, actions, axis=-1)
    return (probs * (classes == taken)).sum(axis=-1, keepdims=True)


def corrected_targets(reward, values, discount, bootstrap, continues, ratios, np, lam):
    # 价值使用行为校正；PPO 的策略权重在 Actor 更新里单独计算。
    weights = np.minimum(ratios, 1.0)
    delta = weights * (reward + discount * bootstrap - values)
    acc = np.zeros_like(values[0])
    corrected = np.empty_like(values)
    for k in range(len(values)-1, -1, -1):
        acc = delta[k] + discount[k] * lam * continues[k] * weights[k] * acc
        corrected[k] = values[k] + acc
    nxt = bootstrap.copy()
    nxt[:-1] = np.where(continues[:-1], corrected[1:], bootstrap[:-1])
    return corrected, reward + discount * nxt - values


def train_actors(actors, buffers, advantages, backbone, torch, np, anchor=None):
    if backbone == "MAPPO":
        for actor, buf, adv in zip(actors, buffers, advantages):
            actor.train(buf, adv.copy(), "EP")
        return
    prefix = np.zeros_like(advantages[0])
    for i in torch.randperm(2).tolist():
        actor, buf = actors[i], buffers[i]
        def flat(x):
            return x.reshape(-1, *x.shape[2:])
        obs, zs, act, masks = map(flat, (buf.obs[:-1], buf.rnn_states[:-1], buf.actions, buf.masks[:-1]))
        beh = flat(buf.action_log_probs)
        if anchor is None:
            buf.update_factor(np.exp(prefix))
            actor.train(buf, advantages[i].copy(), "EP")
        else:
            # 裁剪新策略相对正式锚点；重要性权重使用实际混合行为。
            old = anchor[i].reshape(-1, 1)
            weight = np.exp(np.clip(old-beh+prefix.reshape(-1, 1), -10, 10))
            adv = flat((advantages[i]-advantages[i].mean())/(advantages[i].std()+1e-5))
            n = len(obs)
            width, rem = divmod(n, actor.actor_num_mini_batch)
            if rem:
                raise ValueError("真实批次不能整除作者 minibatch 数")
            def tensor(value):
                return torch.as_tensor(value, **actor.tpdv)
            for _ in range(actor.ppo_epoch):
                for ids in torch.randperm(n).numpy().reshape(-1, width):
                    logp, ent, _ = actor.evaluate_actions(obs[ids], zs[ids], act[ids], masks[ids], None, None)
                    ratio = (logp-tensor(old[ids])).clamp(-20, 20).exp()
                    v = tensor(adv[ids])
                    clipped = ratio.clamp(1-actor.clip_param, 1+actor.clip_param)
                    loss = -(tensor(weight[ids])*torch.minimum(ratio*v, clipped*v)).mean()-actor.entropy_coef*ent
                    actor.actor_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.actor.parameters(), actor.max_grad_norm, error_if_nonfinite=True)
                    actor.actor_optimizer.step()
        with torch.no_grad():
            new = []
            for lo in range(0, len(obs), 1024):
                k = slice(lo, lo+1024)
                new.append(actor.evaluate_actions(obs[k], zs[k], act[k], masks[k], None, None)[0].cpu().numpy())
            prefix += (np.concatenate(new)-beh).reshape(prefix.shape)


def tiger_pair(pair, width, lr, opt_eps, decay, torch, device):
    # 复用 K3 TIGER 增量的角色嵌入、可学习余弦频率和跨角色注意力。
    # 走廊只有合法可见的六帧历史；过去的真实动作或未来请求不额外输入。
    nn = torch.nn
    class TemporalInteraction(nn.Module):
        def __init__(self, base, role):
            super().__init__()
            self.base, self.role = base, role
            w = 64
            self.project = nn.ModuleList(nn.Sequential(nn.Linear(28, w), nn.SiLU(), nn.LayerNorm(w))
                                         for _ in range(2))
            self.roles = nn.Embedding(2, w)
            self.frequency = nn.Parameter(10.**(-torch.linspace(0, 9, w)))
            self.phase = nn.Parameter(torch.zeros(w))
            self.attn = nn.MultiheadAttention(w, 4, batch_first=True, dropout=0.)
            self.out = nn.Linear(w, width)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            self.register_buffer("lag", torch.arange(5, -1, -1, dtype=torch.float32))

        def forward(self, x):
            frame = x[:, :168].reshape(-1, 6, 28)
            u = torch.stack([p(frame) for p in self.project], dim=1)
            position = torch.cos(self.lag[:, None]*self.frequency+self.phase)
            tokens = (u+self.roles.weight[None, :, None]+position[None, None]).flatten(1, 2)
            q = u[:, self.role, -1:]+self.roles.weight[self.role]
            v = self.attn(q, tokens, tokens, need_weights=False)[0][:, 0]
            return self.base(x)+self.out(v)
    for role, actor in enumerate(pair):
        actor.actor.base = TemporalInteraction(actor.actor.base, role).to(device)
        actor.actor_optimizer = torch.optim.Adam(actor.actor.parameters(), lr=lr, eps=opt_eps, weight_decay=decay)


def actor_extension(actors, arm, args, torch, device):
    if arm.entry == "tiger":
        tiger_pair(actors, args["hidden_sizes"][-1], args["lr"], args["opti_eps"],
                   args["weight_decay"], torch, device)


def categorical_kl(q, p, np):
    return float(np.mean(np.sum(q*(np.log(np.maximum(q, 1e-8))-np.log(np.maximum(p, 1e-8))), axis=-1)))


def forecast_actor(current, previous, torch):
    shadow = copy.deepcopy(current)
    # 只外推最近一次真实参数变化，不假称是完整未来优化器更新。
    with torch.no_grad():
        for name, value in shadow.actor.named_parameters():
            old = previous[name].to(value.device)
            delta = value-old
            cap = .025*value.norm().clamp_min(1e-3)
            value.add_(delta * (cap/delta.norm().clamp_min(1e-8)).clamp(max=1.))
    shadow.prep_rollout()
    return shadow


class BranchProbe:
    def __init__(self, norm, critic, vn, z, ones, gamma, np, torch):
        self.norm, self.critic, self.vn = norm, critic, vn
        self.z, self.ones = z[:1], ones[:1]
        self.gamma, self.np, self.torch = gamma, np, torch

    def dist(self, actor, x):
        with self.torch.no_grad():
            return action_dist(actor, self.norm.normalize_obs(x[None]), self.z, self.ones,
                               self.torch).probs[0].cpu().numpy()

    def run(self, env, actors, draws, horizon, first=None, watch=False, override=None):
        # 同一候选族用相同随机数；分支使用推演的动作而非未来真实动作。
        x = env.observation()
        gain, power, elapsed = 0., 1., 0
        while not env.env.done and elapsed < horizon:
            role = env.phase
            if watch and role == 1 and len(env.env.targets()) >= 2:
                return gain, power, elapsed, (env, x)
            if first is not None:
                action, first = first, None
            else:
                use = (role == 0 and env.focal() is not None) or (role == 1 and len(env.env.targets()) >= 2)
                if use:
                    p = self.dist(actors[role], x)
                    action = min(2, int(self.np.searchsorted(self.np.cumsum(p), draws[elapsed, role], side="right")))
                else:
                    # 无有效选择时，各请求在既有分钟协议下物理等价。
                    action = 0
            x, reward, _, _, info = env.step(action)
            if info["physical_increment"]:
                gain += power*reward
                power *= self.gamma
                elapsed += 1
        if not env.env.done:
            with self.torch.no_grad():
                value = self.critic.get_values(self.norm.normalize_obs(x[None]), self.z, self.ones)[0]
                # Critic 用于相同终点的近似价值；真实奖励只来自分支物理模拟。
                gain += power*float(self.vn.denormalize(value.cpu().numpy())[0, 0])
        return gain, power, elapsed, None

    def family(self, env, actors, draws):
        qs, probs, found, delays = [], [], [], []
        for action in range(3):
            trial = copy.deepcopy(env)
            gain, discount, minutes, hit = self.run(trial, actors, draws, BRANCH_HORIZON, action, watch=True)
            if hit is None:
                qs.append([gain]*3)
                probs.append([1/3]*3)
                found.append(False)
                delays.append(BRANCH_HORIZON)
            else:
                future, x = hit
                probs.append(self.dist(actors[1], x))
                found.append(True)
                delays.append(minutes)
                qs.append([gain + discount*self.run(copy.deepcopy(future), actors, draws[minutes:],
                           BRANCH_HORIZON-minutes, first=a)[0] for a in range(3)])
        return self.np.asarray(qs), self.np.asarray(probs), self.np.asarray(found), delays


def response_net(dim, torch, device):
    nn = torch.nn
    return nn.Sequential(nn.Linear(dim+3, 64), nn.SiLU(), nn.Linear(64, 3)).to(device)


def replay_shadow(engine, blobs, actors, resp, resp_opt, norm, rng, np, torch):
    from cloudpickle import loads
    obs, values, features, targets, chance, nll = [], [], [], [], [], []
    for blob in blobs:
        env = loads(blob)
        raw = env.observation()
        q, p, seen, delay = engine.family(env, actors, rng.random((BRANCH_HORIZON, 2)))
        x = norm.normalize_obs(raw[None])[0]
        choice = p
        if resp is not None:
            feat = np.concatenate((np.repeat(x[None], 3, axis=0), np.eye(3, dtype=np.float32)), axis=1)
            ft = torch.as_tensor(feat, device=actors[0].tpdv["device"])
            with torch.no_grad():
                estimated = resp(ft).softmax(-1).cpu().numpy()
            if seen.any():
                # 同源 Actor 概率是干净标签；只在发生合法响应时加入其损失。
                nll.append(categorical_kl(p[seen], estimated[seen], np))
                features.extend(feat[seen])
                targets.extend(p[seen])
            choice = np.where(seen[:, None], .5*p+.5*estimated, p)
        obs.append(x)
        values.append(np.sum(q*choice, axis=1))
        chance.append(dict(found=seen.tolist(), delay=delay))
    if features:
        device = actors[0].tpdv["device"]
        x = torch.as_tensor(np.asarray(features), device=device)
        y = torch.as_tensor(np.asarray(targets), dtype=torch.float32, device=device)
        loss = -(y*resp(x).log_softmax(-1)).sum(-1).mean()
        resp_opt.zero_grad(set_to_none=True)
        loss.backward()
        resp_opt.step()
    result = dict(families=len(obs), opportunities=chance, response_kl=nll)
    if not obs:
        return None, result
    shadow = copy.deepcopy(actors[0])
    shadow.prep_training()
    # 只把相对候选收益教给影子，真实 Actor 不直接接受反事实奖励。
    x = np.asarray(obs, dtype=np.float32)
    q = np.asarray(values, dtype=np.float32)
    q = (q-q.mean(axis=-1, keepdims=True))/np.maximum(1., q.std(axis=-1, keepdims=True))
    logp = action_dist(shadow, x, np.repeat(engine.z, len(x), axis=0),
                       np.repeat(engine.ones, len(x), axis=0), torch).logits
    target = (logp.detach()+.5*torch.as_tensor(q, **shadow.tpdv)).softmax(-1)
    loss = -(target*logp).sum(-1).mean()
    shadow.actor_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(shadow.actor.parameters(), .5, error_if_nonfinite=True)
    shadow.actor_optimizer.step()
    shadow.prep_rollout()
    result.update(shadow_loss=float(loss.detach()), rel_value_span=float(np.ptp(values, axis=-1).mean()))
    return shadow, result


def learning_aware_step(actors, before, engine, blobs, buffers, active_flags, rng, np, torch):
    """有界跨段得分函数项；不是完整 COALA-PG 的无偏元训练合同。"""
    from cloudpickle import loads
    report = []
    for role, actor in enumerate(actors):
        teammate_replaced = list(before)
        teammate_replaced[1-role] = actors[1-role]
        gains = []
        for blob in blobs:
            env = loads(blob)
            draws = rng.random((BRANCH_HORIZON, 2))
            anchor_gain = engine.run(copy.deepcopy(env), before, draws, BRANCH_HORIZON)[0]
            updated_gain = engine.run(env, teammate_replaced, draws, BRANCH_HORIZON)[0]
            gains.append(updated_gain-anchor_gain)
        buf = buffers[role]
        # 探针来自第 0 条轨迹；得分来自其余轨迹，不用同一条真实轨迹给自己打分。
        eligible = active_flags[role].copy()
        eligible[:, 0] = False
        ids = np.flatnonzero(eligible)
        if not gains or not len(ids):
            report.append(dict(probes=len(gains), update=False))
            continue
        ids = rng.choice(ids, min(256, len(ids)), replace=False)
        n = buf.actions.shape[0]*buf.actions.shape[1]
        lp = before[role].evaluate_actions(buf.obs[:-1].reshape(n, -1)[ids],
             buf.rnn_states[:-1].reshape(n, *buf.rnn_states.shape[2:])[ids],
             buf.actions.reshape(n, 1)[ids], buf.masks[:-1].reshape(n, 1)[ids], None, None)[0]
        scale = .025*np.clip(np.mean(gains)/25., -1., 1.)
        gradients = torch.autograd.grad(-float(scale)*lp.mean(), tuple(before[role].actor.parameters()))
        if scale:
            actor.actor_optimizer.zero_grad(set_to_none=True)
            for p, grad in zip(actor.actor.parameters(), gradients):
                p.grad = grad.detach()
            torch.nn.utils.clip_grad_norm_(actor.actor.parameters(), .5, error_if_nonfinite=True)
            actor.actor_optimizer.step()
        report.append(dict(probes=len(gains), gains=gains, score_weight=float(scale), update=bool(scale)))
    return report


def roles_train(folder, a, arm, seed, cfg, rows, marks, env_cls, args, gym, np, torch):
    import train_eval_corridor_uagmc_happo_mappo as b
    from harl.algorithms.critics.v_critic import VCritic
    from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
    from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
    from harl.common.valuenorm import ValueNorm
    from stable_baselines3.common.vec_env import VecNormalize

    backbone, hook = arm.backbone, arm.entry
    offpolicy = hook in ("evolution", "selective", "equivalent", "event", "pr2")
    extrapolate = hook in ("evolution", "selective", "equivalent")
    replay = hook in ("event", "pr2", "lapg")
    b.set_seed(seed, np, torch)
    meta = b.read(folder/"RESUME.json") if (folder/"RESUME.json").is_file() else None
    start = meta["real_steps"] if meta else 0
    if not 0 <= start <= STEPS or start % b.ROLLOUT:
        raise ValueError("恢复点不在真实 rollout 边界")
    if start == STEPS:
        b.verified(folder, meta)
        return
    wrapped, Vec = experimental_env(env_cls, cfg, backbone)
    vector = Vec([lambda: wrapped(cfg, rows, backbone, hook) for _ in range(b.LANES)], start_method="spawn")
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
        actor_extension(actors, arm, args, torch, device)
        critics, norms = [v0], [n0]
        if hook == "role_critics":
            critics.append(VCritic(args, space, device=device))
            norms.append(ValueNorm(1, device=device))
        buffers = [OnPolicyActorBuffer(args, space, act) for _ in actors]
        value_buffers = [OnPolicyCriticBufferEP(args, space) for _ in critics]
        vstats = ("running_mean", "running_mean_sq", "debiasing_term")
        rng = np.random.default_rng(seed+72611)
        prev = [None, None]
        probe_obs = [None, None]
        pending = [None, None]
        reliable = [False, False]
        blobs = []
        resp = response_net(space.shape[0], torch, device) if hook == "pr2" else None
        resp_opt = torch.optim.Adam(resp.parameters(), lr=3e-4) if resp is not None else None
        if meta:
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            if state["real_steps"] != start or state["schema"] != "dispatch10_v1" or state["arm"] != arm.name:
                raise ValueError("存档模式或真实步数不匹配")
            for role in range(2):
                actors[role].actor.load_state_dict(state["actors"][role])
                actors[role].actor_optimizer.load_state_dict(state["optimizers"][role])
            for i, (critic, vn) in enumerate(zip(critics, norms)):
                critic.critic.load_state_dict(state["critics"][i])
                critic.critic_optimizer.load_state_dict(state["critic_optimizers"][i])
                vn.load_state_dict(state["value_norms"][i])
                for key in vstats:
                    getattr(vn, key).copy_(state["value_statistics"][i][key])
            prev, probe_obs, pending, reliable, blobs, rng_state = state["extras"]
            rng.bit_generator.state = rng_state
            if resp is not None:
                resp.load_state_dict(state["response"])
                resp_opt.load_state_dict(state["response_opt"])
            b.rng_restore(state["rng"], device, np, torch)

        def checkpoint(real):
            state = dict(schema="dispatch10_v1", real_steps=real, arm=arm.name,
                         actors=[x.actor.state_dict() for x in actors],
                         optimizers=[x.actor_optimizer.state_dict() for x in actors],
                         critics=[x.critic.state_dict() for x in critics],
                         critic_optimizers=[x.critic_optimizer.state_dict() for x in critics],
                         value_norms=[x.state_dict() for x in norms],
                         value_statistics=[{k: getattr(x, k) for k in vstats} for x in norms],
                         extras=(prev, probe_obs, pending, reliable, blobs, rng.bit_generator.state),
                         response=resp.state_dict() if resp is not None else None,
                         response_opt=resp_opt.state_dict() if resp_opt is not None else None,
                         rng=b.rng_snapshot(np, torch))
            b.save_generation(folder, real, STEPS, marks, norm, lambda path: torch.save(state, path), ".pt")

        # 与 v2 恢复协议相同：更新边界恢复优化器与 RNG，重建环境回合。
        obs = norm.reset()
        zeros = np.zeros((b.LANES, args["recurrent_n"], args["hidden_sizes"][-1]), np.float32)
        ones = np.ones((b.LANES, 1), np.float32)
        masks = ones.copy()
        real = start
        last_saved = start
        begun = time.monotonic()
        while real < STEPS:
            if (a.output/"STOP_REQUESTED").exists():
                if real > last_saved:
                    checkpoint(real)
                raise InterruptedError("用户暂停")
            for actor, buf in zip(actors, buffers):
                actor.prep_rollout()
                actor.lr_decay(real, STEPS)
                buf.step, buf.factor = 0, None
            for critic, buf in zip(critics, value_buffers):
                critic.prep_rollout()
                critic.lr_decay(real, STEPS)
                buf.step = 0
            before = copy.deepcopy(actors) if hook == "lapg" and real else None
            engine = BranchProbe(norm, v0, n0, zeros, ones, args["gamma"], np, torch) if replay else None
            shadow = [None, None]
            diag = {}
            if extrapolate and prev[0] is not None:
                shadow = [forecast_actor(actor, state, torch) for actor, state in zip(actors, prev)]
            if hook in ("event", "pr2") and blobs:
                shadow[0], diag["event_replay"] = replay_shadow(engine, blobs, actors, resp, resp_opt,
                                                               norm, rng, np, torch)
            running_off = any(s is not None for s in shadow)
            fresh_blobs = []
            sample_times = set(rng.choice(b.LENGTH, 4*BRANCHES_PER_ROLLOUT, replace=False)) if replay else ()
            ts = (b.LENGTH, b.LANES, 1)
            anchor = np.zeros((2, *ts), np.float32)
            active_flags = np.zeros((2, b.LENGTH, b.LANES), bool)
            mix_count = [0, 0]
            terminal = np.full(ts, np.nan, np.float32)
            critic_ratios = np.ones(ts, np.float32)

            def act_at(i, x, opportunity, classes=None):
                actor = actors[i]
                if not running_off:
                    act, lp, _ = actor.get_actions(x, zeros, masks)
                    return act, lp, lp.cpu().numpy(), np.ones((b.LANES, 1), np.float32)
                formal = action_dist(actor, x, zeros, masks, torch)
                if shadow[i] is None or (hook == "selective" and not reliable[i]):
                    behavior = formal
                else:
                    future = action_dist(shadow[i], x, zeros, masks, torch)
                    mix = SHADOW_MIX*torch.as_tensor(opportunity[:, None], dtype=torch.float32, device=device)
                    mix_count[i] += int(opportunity.sum())
                    behavior = torch.distributions.Categorical(probs=(1-mix)*formal.probs+mix*future.probs)
                act = behavior.sample()
                pi, mu = formal.log_prob(act)[:, None], behavior.log_prob(act)[:, None]
                ratio = (pi-mu).exp().cpu().numpy()
                if classes is not None:
                    # 只用等价类质量修正 Critic；正式 Actor 始终用原始动作概率。
                    index = act.cpu().numpy()[:, None]
                    p = class_mass(formal.probs.cpu().numpy(), classes, index, np)
                    q = class_mass(behavior.probs.cpu().numpy(), classes, index, np)
                    ratio = p/np.maximum(q, 1e-8)
                return act[:, None], mu, pi.cpu().numpy(), ratio

            for k in range(b.LENGTH):
                original = norm.get_original_obs()
                focal = original[:, -9] > .5
                active_flags[0, k] = focal
                if k in sample_times and len(fresh_blobs) < BRANCHES_PER_ROLLOUT:
                    possible = [0] if hook == "lapg" and focal[0] else list(np.flatnonzero(focal)) if hook != "lapg" else []
                    if possible:
                        blob = vector.env_method("branch_at_pre", indices=int(rng.choice(possible)))[0]
                        if blob is not None:
                            fresh_blobs.append(blob)
                # 两个价值网络读同一决策前状态。只拆价值估计，不给 Critic 额外信息或改时钟。
                with torch.no_grad():
                    values = [v.get_values(obs, zeros, masks)[0] for v in critics]
                    # 选择性版本只改变 Aircraft 行为；这时已经知道候选调机目标数。
                    ap, lp, anchor[0, k], pratio = act_at(0, obs, focal & (hook != "selective"))
                buffers[0].obs[k], buffers[0].masks[k] = obs, masks
                mid, reward, done, infos = norm.step(ap.cpu().numpy().ravel())
                if any(done) or np.any(reward) or any(i["physical_increment"] != 0 for i in infos):
                    raise RuntimeError("乘客动作不能推进物理时钟")
                air = np.asarray([i.get("air_targets", 2) >= 2 for i in infos])
                active_flags[1, k] = air
                classes = np.asarray([i["dispatch_classes"] for i in infos]) if hook == "equivalent" else None
                with torch.no_grad():
                    aa, la, anchor[1, k], aratio = act_at(1, mid, air if hook == "selective" else np.ones_like(air), classes)
                critic_ratios[k] = pratio*aratio
                buffers[1].obs[k], buffers[1].masks[k] = mid, masks
                nxt, rewards, dones, infos = norm.step(aa.cpu().numpy().ravel())
                if any(i["physical_increment"] != 1 for i in infos):
                    raise RuntimeError("调机阶段每个环境只计一个真实步")
                next_masks = (~dones).astype(np.float32)[:, None]
                bad = np.asarray([[0. if i.get("TimeLimit.truncated") else 1.] for i in infos], np.float32)
                if running_off:
                    for i, info in enumerate(infos):
                        if dones[i] and not bad[i, 0]:
                            with torch.no_grad():
                                term = v0.get_values(info["terminal_observation"][None], zeros[:1], ones[:1])[0]
                            terminal[k, i] = n0.denormalize(term.cpu().numpy())[0]
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
                if running_off:
                    buf.value_preds[-1] = next_value.cpu().numpy()
                    vals = vn.denormalize(buf.value_preds)
                    cont = buf.masks[1:].astype(bool)
                    discount = args["gamma"]*(cont | ~buf.bad_masks[1:].astype(bool))
                    bootstrap = np.where(np.isfinite(terminal), terminal, vals[1:])
                    buf.returns[:-1], offadv = corrected_targets(buf.rewards, vals[:-1], discount, bootstrap,
                                                                 cont, critic_ratios, np, args["gae_lambda"])
                else:
                    buf.compute_returns(next_value.cpu().numpy(), vn)
                critic.prep_training()
            for actor in actors:
                actor.prep_training()
            for_adv = ([offadv]*2 if running_off else
                       [buf.returns[:-1] - vn.denormalize(buf.value_preds[:-1])
                        for buf, vn in zip(value_buffers, norms)])
            if len(for_adv) == 1:
                for_adv *= 2
            if extrapolate:
                def prob(model, x):
                    with torch.no_grad():
                        return action_dist(model, x, np.repeat(zeros[:1], len(x), axis=0),
                                           np.repeat(ones[:1], len(x), axis=0), torch).probs.cpu().numpy()
                prev = [{key: p.detach().cpu().clone() for key, p in actor.actor.named_parameters()} for actor in actors]
                expect = []
                for i, actor in enumerate(actors):
                    x = buffers[i].obs[:-1].reshape(-1, space.shape[0])
                    ids = np.linspace(0, len(x)-1, 64, dtype=int)
                    probe_obs[i] = x[ids].copy()
                    expect.append((prob(actor, probe_obs[i]), prob(shadow[i], probe_obs[i]))
                                  if shadow[i] is not None else None)
                pending = expect
            train_actors(actors, buffers, for_adv, backbone, torch, np, anchor if running_off else None)
            if extrapolate:
                probe_results = []
                for i, pair in enumerate(pending):
                    if pair is None:
                        probe_results.append(None)
                        continue
                    q = prob(actors[i], probe_obs[i])
                    flat_err, pred_err = (categorical_kl(q, p, np) for p in pair)
                    reliable[i] = bool(pred_err+1e-6 < flat_err)
                    probe_results.append(dict(anchor_kl=flat_err, forecast_kl=pred_err, enabled=reliable[i]))
                diag["forecast"] = probe_results
            if before is not None and fresh_blobs:
                diag["learning_aware"] = learning_aware_step(actors, before, engine, fresh_blobs, buffers,
                                                            active_flags, rng, np, torch)
            for critic, buf, vn in zip(critics, value_buffers, norms):
                critic.train(buf, vn)
            blobs = fresh_blobs
            diag.update(mixed=mix_count, physical_steps=real, real_fps=(real-start)/max(1., time.monotonic()-begun),
                        critic_rho_mean=float(critic_ratios.mean()),
                        critic_rho_ess=float(critic_ratios.mean()**2/np.square(critic_ratios).mean()))
            b.save(folder/"updates"/f"{real:09d}.json", diag)
            b.save(folder/"LIVE.json", diag)
            if real in marks or real % (b.ROLLOUT*5) == 0 or (a.output/"STOP_REQUESTED").exists():
                checkpoint(real)
                last_saved = real
            for buf in (*buffers, *value_buffers):
                buf.after_update()
    finally:
        (norm if norm is not None else vector).close()


def eval_arm(folder, a, arm, cfg, bank, marks, env_cls, args, np, torch):
    import train_eval_corridor_uagmc_happo_mappo as b
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    results = []
    device = torch.device(a.device)
    z = np.zeros((1, args["recurrent_n"], args["hidden_sizes"][-1]), np.float32)
    mask = np.ones((1, 1), np.float32)
    for mark in marks:
        path = folder/"eval"/f"{mark:09d}.json"
        if path.exists():
            results.extend(b.read(path))
            continue
        model_path, norm_path = b.verified(folder, b.read(folder/"checkpoints"/f"{mark:09d}.json"))
        vec = DummyVecEnv([lambda: env_cls(cfg, bank[0][2], arm.backbone)])
        norm = None
        try:
            norm = VecNormalize.load(str(norm_path), vec)
            norm.training = norm.norm_reward = False
            actors, _, _ = b.build_pair(arm.backbone, args, norm.observation_space, norm.action_space, device)
            actor_extension(actors, arm, args, torch, device)
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            if state["arm"] != arm.name or state["real_steps"] != mark:
                raise ValueError("评估存档组别或真实步数不符")
            for actor, weights in zip(actors, state["actors"]):
                actor.actor.load_state_dict(weights)
                actor.prep_rollout()
            part = []
            for name, role, rows in bank:
                if role == "final" and mark != STEPS:
                    continue
                for seed in ((123, 124, 125) if role == "final" else (123, 124)):
                    if (a.output/"STOP_REQUESTED").exists():
                        raise InterruptedError("用户暂停评估")
                    b.set_seed(seed, np, torch)
                    env = env_cls(cfg, rows, arm.backbone)
                    x, _ = env.reset(seed=123)
                    with torch.no_grad():
                        while not env.env.done:
                            # 所有组只评正式 Actor，不允许在验证库运行在线前瞻或更新。
                            action = actors[env.phase].get_actions(norm.normalize_obs(x[None]), z, mask)[0].item()
                            x = env.step(int(action))[0]
                    part.append(dict(trace=name, role=role, policy_seed=seed, step=mark,
                                     metrics=env.env.metrics(), control=dict(env.stat)))
                    env.close()
            b.save(path, part)
            results.extend(part)
            print(f"{folder.name}：已评估 {mark:,} 步存档，{len(part)} 个回合", flush=True)
        finally:
            (norm if norm is not None else vec).close()
    b.save(folder/"EVALUATION.json", dict(rows=results, weights_unchanged=True))
    return results


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
    contract = dict(schema="corridor_dispatch_v2_10_v1", layer=a.layer, lane=a.lane, config=asdict(cfg),
                    cells=[dict(arm=asdict(arm), seed=seed) for arm, seed in selected],
                    steps=STEPS, marks=marks, harl=vendor, harl_args=args,
                    origins=dict(tiger="旧 K3 的 TimeEncode/attention，不复现原 TIGER 的价值分解",
                                 pr2="条件响应思想适配，非原 PR2 优化器",
                                 lapg="有界跨 rollout 辅助项，非 COALA-PG 无偏元学习"),
                    lookahead=dict(horizon=BRANCH_HORIZON, mix=SHADOW_MIX, samples=BRANCHES_PER_ROLLOUT,
                                   branch_demand="只复制已揭示需求，未知未来不揭示"),
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
                roles_train(folder, a, arm, seed, cfg, rows, marks, env_cls, args, gym, np, torch)
                results = eval_arm(folder, a, arm, cfg, bank, marks, env_cls, args, np, torch)
                final = [x for x in results if x["role"] == "final" and x["step"] == STEPS]
                if len(final) != 12:
                    raise RuntimeError(f"最终评估不完整：{len(final)}/12")
                late = [x for x in results if x["role"] == "validation" and x["step"] >= .8*STEPS]
                def aggregate(items):
                    def mean(key):
                        vals = [item["metrics"][key] for item in items]
                        return sum(vals)/len(vals) if vals and all(v is not None for v in vals) else None
                    return dict(n=len(items), att_min=mean("mean_att_min"), completion=mean("completion"),
                                cutoff_cost_min=mean("fixed_cutoff_cost_min"))
                b.save(folder/"STATUS.json", dict(state="COMPLETE", real_steps=STEPS, evaluated_through_final=True,
                       final=aggregate(final), late_validation=aggregate(late)))
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
    parser.add_argument("--only", nargs="+", help="仅执行指定方法；不能与完整矩阵混用输出目录")
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
    return execute(a, selected)


if __name__ == "__main__":
    raise SystemExit(main())
