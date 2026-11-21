"""走廊 E2：Lookahead 的完整基准和五项单因素消融。"""
from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path

import train_eval_corridor_lookahead_l0l5 as legacy

ARMS = {
    "F_FULL": (1, True, True),
    "A1_FORMAL_BEHAVIOR": (1, False, True),
    "A2_GAE": (1, True, True),
    "A3_FROZEN_WM": (1, True, True),
    "A4_MEAN_DYNAMICS": (1, True, True),
    "A5_EVENT_REWARD": (1, True, True),
}


def execution_class(gym, np, collect_counts):
    base = legacy.build_gym(gym, np)

    class Instrumented(base):
        def advance(self, routes=None):
            if not collect_counts:
                return super().advance(routes)
            # 以 drain 后的真实人数为积分区间左端；不读未来请求。
            e = self.env
            start = e.now
            ticks = e._ticks(self.c.decision_min)
            segments = {}
            drain = e._drain

            def observe():
                drain()
                segments[e.now] = len(e.passengers) - e.finished

            e._drain = observe
            try:
                obs, reward, done, cutoff, info = super().advance(routes)
            finally:
                del e._drain
            counts = np.zeros(ticks, dtype=np.float32)
            active = 0
            for k in range(e.now - start):
                active = segments.get(start + k, active)
                counts[k] = active
            if abs(reward + counts.sum() * self.c.tick_min) > 1e-5:
                raise RuntimeError("真实门到门人时积分不守恒")
            info["active_path"] = counts
            return obs, reward, done, cutoff, info

    return Instrumented


def phase_model(dim, torch, kind):
    nn = torch.nn
    core = dim - 8
    events = kind == "A5_EVENT_REWARD"
    size = core if events else core + 1

    def mlp(i, o):
        return nn.Sequential(nn.Linear(i, 192), nn.SiLU(), nn.Linear(192, 192),
                             nn.SiLU(), nn.Linear(192, o))

    class PhaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.post = mlp(dim + 3, 2 * core + 3)
            self.advance = mlp(dim + 3, 2 * size + (10 if events else 0))

        def post_parameters(self, x, p):
            return self.post(torch.cat((x, nn.functional.one_hot(p, 3).float()), -1))

        def next_parameters(self, mid, p, a):
            # 选站的效果已包含在 mid，调机模型不重新跳过选站阶段。
            return self.advance(torch.cat((mid, nn.functional.one_hot(a, 3).float()), -1))

    return PhaseModel()


def fit(model, opt, ring, norm, torch, np, rng, device, kind, updates):
    data = {k: np.concatenate([b[k] for b in ring]) for k in ring[0]}
    ids = np.arange(len(data["r"]))
    train, val = ids[ids % 8 != 0], ids[ids % 8 == 0][:1024]
    events = kind == "A5_EVENT_REWARD"
    core = norm.observation_space.shape[0] - 8

    def tx(value, dtype=torch.float32):
        return torch.as_tensor(value, device=device, dtype=dtype)

    def loss(ix):
        x, mid, nxt = [tx(norm.normalize_obs(data[k][ix])) for k in ("x", "post", "next")]
        p, a, executed = [tx(data[k][ix], torch.long) for k in ("p", "a", "executed_p")]
        if hasattr(model, "set_history"):
            model.set_history(tx(data["actions_history"][ix], torch.long))
        post = model.post_parameters(x, p)
        pred = model.next_parameters(mid, p, a)
        u, lu = legacy.gaussian(post[:, :-3], torch)
        v, lv = legacy.gaussian(pred[:, :-10] if events else pred, torch)
        mask = tx(~data["done"][ix])
        rew = tx(data["r"][ix])

        def nll(y, mean, log_sd):
            return .5 * ((y - mean) * (-log_sd).exp()).square() + log_sd

        ep = nll(mid[:, :core], u, lu).mean()
        en = (nll(nxt[:, :core], v[:, :core], lv[:, :core]).mean(-1) * mask).sum() / mask.sum().clamp_min(1)
        ec = torch.nn.functional.cross_entropy(post[:, -3:], executed)
        if events:
            # 一个物理步十个 0.1 分钟子区间，人数单位为 100 人。
            counts = torch.nn.functional.softplus(pred[:, -10:])
            er = (counts - tx(data["active_path"][ix]) / 100).square().mean()
            estimate = -10 * counts.sum(-1)
        else:
            er = nll(rew / 100, v[:, -1], lv[:, -1]).mean()
            estimate = v[:, -1] * 100
        next_mse = ((nxt[:, :core] - v[:, :core]).square().mean(-1) * mask).sum() / mask.sum().clamp_min(1)
        return ep + en + ec + er, ((rew - estimate).abs().mean(),
                                  (mid[:, :core] - u).square().mean().sqrt(), next_mse.sqrt())

    model.train()
    for _ in range(updates):
        batch = rng.choice(train, 256, replace=len(train) < 256)
        objective, _ = loss(batch)
        opt.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        opt.step()
    model.eval()
    with torch.no_grad():
        objective, (mae, post_rmse, next_rmse) = loss(val)
    report = dict(loss=float(objective), reward_mae=float(mae), n=len(val), optimizer_steps=updates,
                  post_rmse=float(post_rmse), next_rmse=float(next_rmse))
    model.validation_errors = (report["post_rmse"], report["next_rmse"])
    return data, report


def patch(fn, ns, substitutions):
    source = inspect.getsource(fn)
    for before, after in substitutions:
        if source.count(before) != 1:
            raise RuntimeError(f"旧入口版本不匹配：{fn.__name__}: {before!r}")
        source = source.replace(before, after)
    exec(compile(source, inspect.getsourcefile(fn), "exec"), ns)
    return ns[fn.__name__]


def run(arms, script, argv=None, model_factory=None, fit_factory=None, imagine_factory=None):
    # 原工程只读；沿用原 HARL 更新、随机数、评估银行和双滚动存档实现。
    ns = vars(legacy).copy()
    scope = dict(kind=None)
    script = Path(script).resolve()
    ns.update(ARMS=arms, __file__=str(script))

    def wm(dim, torch):
        if model_factory:
            model = model_factory(dim, torch, scope["kind"])
            if model is not None:
                return model
        return phase_model(dim, torch, scope["kind"])

    def fitter(model, opt, ring, norm, torch, np, rng, device, fits):
        kind = scope["kind"]
        iterations = 0 if kind == "A3_FROZEN_WM" and fits else 48
        if fit_factory:
            custom = fit_factory(model, opt, ring, norm, torch, np, rng, device, kind, iterations)
            if custom is not None:
                return custom
        return fit(model, opt, ring, norm, torch, np, rng, device, kind, iterations)

    def draw(param, torch):
        mean, log_std = legacy.gaussian(param, torch)
        return mean if scope["kind"] == "A4_MEAN_DYNAMICS" else mean + torch.randn_like(mean) * log_std.exp()

    def shifted_frames(x, norm, torch):
        # 历史槽使用各自的归一化统计，不能直接移动已归一化的值。
        stats = norm.obs_rms
        mean = torch.as_tensor(stats.mean[:168], dtype=x.dtype, device=x.device)
        std = torch.as_tensor(stats.var[:168] + norm.epsilon, dtype=x.dtype, device=x.device).sqrt()
        raw = x[:, 28:168] * std[28:168] + mean[28:168]
        return ((raw - mean[:140]) / std[:140]).clamp(-norm.clip_obs, norm.clip_obs)

    def outcome(model, mid, pi, a, device, torch):
        param = model.next_parameters(torch.as_tensor(mid, dtype=torch.float32, device=device), pi, a.long().flatten())
        if scope["kind"] == "A5_EVENT_REWARD":
            counts = torch.nn.functional.softplus(param[:, -10:])
            # 预测事件区间人数；奖励算式固定，不学习标量奖励。
            return torch.cat((draw(param[:, :-10], torch), -.1 * counts.sum(-1, keepdim=True)), -1)
        return draw(param, torch)

    def trace_returns(reward, value, bootstrap, cont, discount, gain, enabled=True):
        if scope["kind"] != "A2_GAE":
            return legacy.vtrace(reward, value, bootstrap, cont, discount, gain, enabled)
        import numpy as np
        ratio = np.exp(np.clip(gain, -40, 40))
        adv = np.empty_like(value)
        carry = np.zeros_like(value[0])
        for i in range(len(value) - 1, -1, -1):
            carry = reward[i] + discount[i] * bootstrap[i] - value[i] + .95 * discount[i] * cont[i] * carry
            adv[i] = carry
        return value + adv, adv, ratio

    ns.update(wm_model=wm, fit_world=fitter, sample_corrected=draw, phase_outcome=outcome,
              shifted_frames=shifted_frames,
              vtrace=trace_returns, incident_kind=lambda: scope["kind"])
    imaginer = patch(legacy.imagine, ns, [
        ("return mean+torch.randn_like(mean)*log_std.exp()", "return sample_corrected(param,torch)"),
        ("obs=norm.normalize_obs(dataset['x'][rng.choice(starts,SHADOW_N,replace=True)])",
         "indices=rng.choice(starts,SHADOW_N,replace=True)\n        obs=norm.normalize_obs(dataset['x'][indices])\n        if hasattr(model,'set_history'):\n            model.set_history(torch.as_tensor(dataset['actions_history'][indices],device=device))"),
        ("out=sample(model.next_parameters(x,pi,a.long().flatten()))",
         "out=phase_outcome(model,mid,pi,a,device,torch)"),
        ("y=sample(pp[:,:-3]).clamp(-10.,10.)",
         "y=sample(pp[:,:-3]).clamp(-10.,10.)\n                y[:,:140]=shifted_frames(x,norm,torch)"),
        ("nxt=np.concatenate((array(out[:,:core].clamp(-10.,10.)),",
         "out[:,:140]=y[:,:140]\n                nxt=np.concatenate((array(out[:,:core].clamp(-10.,10.)),"),
        ("                obs=nxt", "                if hasattr(model,'push_history'):\n                    model.push_history(executed,a.long().flatten())\n                obs=nxt"),
    ])
    if imagine_factory:
        def imagine(*args):
            return imagine_factory(imaginer, scope["kind"], *args)
        ns["imagine"] = imagine
    trainer = patch(legacy.train_cell, ns, [
        ('        initial, since = real, time.monotonic()', '        initial, since = real, time.monotonic()\n        action_history = np.full((n,6,2),3,np.int64)'),
        ("fit_world(dynamics, opt, ring, norm, torch, np, rng, device)",
         "fit_world(dynamics, opt, ring, norm, torch, np, rng, device, fits)"),
        ("fits += 48", "fits += report['optimizer_steps']"),
        ('            for k in range(t):', '            if incident_kind() == "A5_EVENT_REWARD":\n                block["active_path"] = []\n            if incident_kind() == "P2_GRU":\n                block["actions_history"] = []\n            for k in range(t):'),
        ("                        block[key].append(value.copy())",
         '                        block[key].append(value.copy())\n                    if "active_path" in block:\n                        block["active_path"].append(np.stack([infos[j]["active_path"] for j in ix]))\n                    if "actions_history" in block:\n                        block["actions_history"].append(action_history[ix].copy())'),
        ('                age = np.where(done,0,age+1)',
         '                if incident_kind() == "P2_GRU":\n                    action_history[:, :-1] = action_history[:, 1:].copy()\n                    action_history[:, -1, 0] = executed_batch\n                    action_history[:, -1, 1] = array(a).ravel()\n                    action_history[done] = 3\n                age = np.where(done,0,age+1)'),
        ('                nxt, r, done, infos = norm.step(array(a).ravel())',
         '                if incident_kind() == "P2_GRU":\n                    executed_batch = norm.get_original_obs()[:, -7:-4].argmax(1).copy()\n                nxt, r, done, infos = norm.step(array(a).ravel())'),
    ])

    def train_cell(folder, cli, arm, seed, cfg, rows, marks, env_class, args, author, gym, np, torch):
        scope["kind"] = arm
        if arm == "A5_EVENT_REWARD" and (cfg.tick_min, cfg.decision_min) != (.1, 1.):
            raise ValueError("人时消融要求 0.1 分钟子步及 1 分钟物理步")
        env_class = execution_class(gym, np, arm == "A5_EVENT_REWARD")
        return trainer(folder, cli, arm, seed, cfg, rows, marks, env_class, args, author, gym, np, torch)

    ns.update(train_cell=train_cell, core_sha=legacy.digest(Path(__file__)),
              legacy_sha=legacy.digest(Path(legacy.__file__)))
    entry = patch(legacy.main, ns, [
        ("a = p.parse_args(argv)", 'a = p.parse_args(argv)\n    if a.layers != ["E2"] or a.steps != 1_996_800 or tuple(a.seeds) != (2101,2102,2103,2104):\n        p.error("正式合同固定 E2、四个配对种子及 1,996,800 真实步")'),
        ("default=tuple(LAYERS)", 'default=["E2"]'),
        ('schema="corridor_lookahead_happo_l0l5_v2"', 'schema="corridor_lookahead_phase_e2_v1", core_sha=core_sha, legacy_sha=legacy_sha'),
    ])
    return entry(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(run(ARMS, __file__))
