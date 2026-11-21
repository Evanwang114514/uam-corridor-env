"""走廊 E2 八组扩展登记；尚未达到定义的组禁止启动。"""
from __future__ import annotations

import copy
import sys

import train_corridor_lookahead_core6 as core

ARMS = {
    "P1_PETS": (1, True, True),
    "P2_GRU": (1, True, True),
    "P3_EVENT_GNN": (1, True, True),
    "P4_HYBRID": (1, True, True),
    "T1_TIGER": (1, True, True),
    "I1_LTL": (1, True, True),
    "I2_CLL": (1, True, True),
    "C1_TRUNCATED": (2, True, True),
}
PENDING = {
    "P3_EVENT_GNN": "需要实体级资源图及事件类型和时间监督",
    "P4_HYBRID": "需要不携带未揭示需求的仿真快照与需求残差采样器",
    "T1_TIGER": "需要有类型的时间交互图同时进入动力学与影子策略更新",
    "I1_LTL": "需要下一次合法队友响应的目标编码及候选条件策略",
    "I2_CLL": "需要可配对的仿真反事实分支及隐式响应差分监督",
}


def models(dim, torch, kind):
    nn = torch.nn
    if kind == "P1_PETS":
        class Ensemble(nn.Module):
            def __init__(self):
                super().__init__()
                self.members = nn.ModuleList([core.phase_model(dim, torch, kind) for _ in range(5)])
                self.indices = None

            def select(self, fun, *args):
                if self.indices is None:
                    raise RuntimeError("集成模型必须已为每条虚拟轨迹指定成员")
                # 同一虚拟轨迹在所有时刻及两个阶段保持同一组选中的模型。
                values = torch.stack([getattr(m, fun)(*args) for m in self.members], dim=1)
                return values[torch.arange(len(values), device=values.device), self.indices]

            def post_parameters(self, x, p):
                return self.select("post_parameters", x, p)

            def next_parameters(self, mid, p, a):
                return self.select("next_parameters", mid, p, a)

        return Ensemble()
    if kind == "P2_GRU":
        class HistoryModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.history = nn.GRU(38, 96, batch_first=True)
                self.post = core.phase_model(dim + 96, torch, kind).post
                self.advance = core.phase_model(dim + 96, torch, kind).advance
                self.post[-1] = nn.Linear(192, 2 * (dim - 8) + 3)
                self.advance[-1] = nn.Linear(192, 2 * (dim - 7))

            def set_history(self, history):
                self.actions = history

            def push_history(self, p, a):
                self.actions = torch.cat((self.actions[:, 1:], torch.stack((p, a), -1)[:, None]), 1)

            def features(self, x, a, phase):
                frames = x[:, :168].reshape(-1, 6, 28)
                lag = torch.arange(-5, 1, device=x.device, dtype=x.dtype)[None, :, None].expand(len(x), -1, -1)
                phase = torch.full_like(lag, phase)
                actions = nn.functional.one_hot(self.actions.long(), 4).reshape(len(x), 6, 8).float()
                _, h = self.history(torch.cat((frames, lag, phase, actions), -1))
                return torch.cat((x, h[-1], nn.functional.one_hot(a, 3).float()), -1)

            def post_parameters(self, x, p):
                return self.post(self.features(x, p, 0))

            def next_parameters(self, mid, p, a):
                return self.advance(self.features(mid, a, 1))

        return HistoryModel()
    return None


def fit_extension(model, opt, ring, norm, torch, np, rng, device, kind, iterations):
    if kind != "P1_PETS":
        return None
    data = {k: np.concatenate([b[k] for b in ring]) for k in ring[0]}
    train = np.flatnonzero(np.arange(len(data["r"])) % 8 != 0)
    valid = np.flatnonzero(np.arange(len(data["r"])) % 8 == 0)
    reports = []
    for member in model.members:
        boot = rng.choice(train, len(train), replace=True)
        which = np.empty(len(data["r"]), dtype=np.int64)
        which[np.arange(len(which)) % 8 != 0] = boot
        which[np.arange(len(which)) % 8 == 0] = valid
        _, report = core.fit(member, opt, [{k: v[which] for k, v in data.items()}],
                             norm, torch, np, rng, device, kind, iterations)
        reports.append(report)
    return data, dict(optimizer_steps=iterations * len(reports), members=reports)


class MixtureBehavior:
    """返回真实混合行为分布的对数概率，而不是抽中的成员或单步策略的概率。"""

    def __init__(self, first, second, alpha, torch):
        self.first, self.second, self.alpha, self.torch = first, second, alpha, torch

    def prep_rollout(self):
        self.first.prep_rollout()
        self.second.prep_rollout()

    def distribution(self, obs, z, masks):
        actions = self.torch.zeros((len(obs), 1), dtype=self.torch.long, device=next(self.first.actor.parameters()).device)
        d1 = self.first.evaluate_actions(obs, z, actions, masks, None, None)[2]
        d2 = self.second.evaluate_actions(obs, z, actions, masks, None, None)[2]
        return self.torch.distributions.Categorical(probs=(1 - self.alpha) * d1.probs + self.alpha * d2.probs)

    def get_actions(self, obs, z, masks):
        dist = self.distribution(obs, z, masks)
        a = dist.sample()
        return a[:, None], dist.log_prob(a)[:, None], z

    def evaluate_actions(self, obs, z, actions, masks, *unused):
        dist = self.distribution(obs, z, masks)
        actions = self.torch.as_tensor(actions, device=dist.logits.device).long().reshape(-1)
        return dist.log_prob(actions)[:, None], dist.entropy().mean(), dist


def imagine_extension(imaginer, kind, team, args, space, action_space, model, norm, dataset,
                      depth, author, Buffer, CB, np, torch, rng, device, cutoff):
    params = (args, space, action_space, model, norm, dataset)
    end = (author, Buffer, CB, np, torch, rng, device, cutoff)
    if kind == "P1_PETS":
        model.indices = torch.as_tensor(rng.integers(0, len(model.members), size=core.legacy.SHADOW_N), device=device)
        try:
            return imaginer(team, *params, depth, *end)
        finally:
            model.indices = None
    if kind != "C1_TRUNCATED":
        return imaginer(team, *params, depth, *end)
    report = imaginer(team, *params, 1, *end)
    first = copy.deepcopy(team[0])
    report.extend(imaginer(team, *params, 1, *end))
    # 冻结阈值，不以 ATT 选择；不确定性代理为真实留出集对应阶段预测误差。
    role_limits = ((.015, .45), (.025, .55))
    check = np.flatnonzero((np.arange(len(dataset["r"])) % 8 == 0) & (~dataset["done"]))[:256]
    if not len(check):
        raise RuntimeError("截断需要非终止的留出状态")
    zero = np.zeros((len(check), args["recurrent_n"], args["hidden_sizes"][-1]), np.float32)
    masks = np.ones((len(check), 1), np.float32)
    a = np.zeros((len(check), 1), np.int64)
    for j, (key, limits) in enumerate(zip(("x", "post"), role_limits)):
        obs = norm.normalize_obs(dataset[key][check])
        with torch.no_grad():
            first[j].prep_rollout()
            team[0][j].prep_rollout()
            dist1 = first[j].evaluate_actions(obs, zero, a, masks, None, None)[2]
            dist2 = team[0][j].evaluate_actions(obs, zero, a, masks, None, None)[2]
            kl = max(0., float(torch.distributions.kl_divergence(dist2, dist1).mean()))
        u = model.validation_errors[j]
        alpha = 0. if u > limits[1] else min(1., limits[0] / (kl + 1e-8))
        team[0][j] = MixtureBehavior(first[j], team[0][j], alpha, torch)
        report[-1].setdefault("truncation", []).append(dict(role=key, kl=kl, heldout_rmse=u, alpha=alpha))
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" not in argv and "--help" not in argv:
        requested = list(ARMS)
        if "--arms" in argv:
            start = argv.index("--arms") + 1
            requested = []
            for token in argv[start:]:
                if token.startswith("--"):
                    break
                requested.append(token)
        blocked = [f"{name}：{PENDING[name]}" for name in requested if name in PENDING]
        if blocked:
            raise SystemExit("尚不具备八组正式训练条件，未启动：\n" + "\n".join(blocked))
    return core.run(ARMS, __file__, argv, models, fit_extension, imagine_extension)


if __name__ == "__main__":
    raise SystemExit(main())
