"""走廊 E2：Lookahead 的完整基准和五项单因素消融。"""
from __future__ import annotations

import inspect
import linecache
import pickle
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


def execution_class(gym, np, kind, plugin=None):
    base = legacy.build_gym(gym, np)

    class Instrumented(base):
        def reset(self, **kwargs):
            self.history_records = []
            return super().reset(**kwargs)

        def observation(self, post=False, focal=None):
            raw = super().observation(post, focal)
            return plugin.observation(self, raw, post, np, kind) if plugin else raw

        def __init__(self, cfg, rows, method):
            super().__init__(cfg, rows, method)
            extra = plugin.extra_width(kind) if plugin else 0
            if extra:
                self.observation_space = gym.spaces.Box(-np.inf, np.inf, (legacy.PACK + 8 + extra,), dtype=np.float32)

        def step(self, action):
            role = self.phase
            legal = (self.focal() is not None if role == 0 else bool(self.env.targets())) if kind in ("T1_TIGER", "I1_LTL", "I2_CLL") else False
            if plugin:
                plugin.record(self, role, legal, int(action), kind)
            old_now = self.env.now
            before = len(self.env.passengers)
            value, reward, done, cutoff, info = super().step(action)
            if plugin and kind in ("P4_HYBRID", "I2_CLL") and role == 1:
                info["reveals"] = tuple((self.env._ticks(p.req.time_min) - old_now, p.req.origin, p.req.destination)
                                        for p in list(self.env.passengers.values())[before:])
            if kind == "I1_LTL":
                info["legal_opportunity"] = legal
            return value, reward, done, cutoff, info

        def branch_state(self):
            return plugin.branch_state(self)

        def resume_state(self):
            # 真实续训专用：物理状态含真实未揭示事件，绝不写入模型回放。
            return pickle.dumps((self.env, self.history, self.history_records, self.phase, self.route_index,
                                 self.stat), protocol=5)

        def restore_state(self, payload):
            self.env, self.history, self.history_records, self.phase, self.route_index, self.stat = pickle.loads(payload)
            return self.observation()

        def advance(self, routes=None):
            if kind != "A5_EVENT_REWARD":
                return super().advance(routes)
            # 以 drain 后的真实人数为积分区间左端；不读未来请求。
            e = self.env
            start = e.now
            start_area = e.area_ticks
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
            person_ticks = 0
            for k in range(e.now - start):
                active = segments.get(start + k, active)
                counts[k] = active
                person_ticks += int(active)
            actual_ticks = e.area_ticks - start_area
            if person_ticks != actual_ticks:
                raise RuntimeError(f"真实门到门人时积分不守恒：重建={person_ticks}，环境={actual_ticks}，起止tick={start},{e.now}")
            if abs(float(reward) + person_ticks * self.c.tick_min) > 1e-8:
                raise RuntimeError(f"奖励与整数人时不符：reward={reward}，person_ticks={person_ticks}")
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
        pred = model.next_parameters(mid, executed, a)
        u, lu = legacy.gaussian(post[:, :-3], torch)
        v, lv = legacy.gaussian(pred[:, :-10] if events else pred, torch)
        mask = tx(~data["done"][ix])
        rew = tx(data["r"][ix])

        def nll(y, mean, log_sd):
            return .5 * ((y - mean) * (-log_sd).exp()).square() + log_sd

        # 由已知历史平移决定的字段不进入高斯拟合目标。
        ep = nll(mid[:, 140:core], u[:, 140:], lu[:, 140:]).mean()
        en = (nll(nxt[:, 140:core], v[:, 140:core], lv[:, 140:core]).mean(-1) * mask).sum() / mask.sum().clamp_min(1)
        ec = torch.nn.functional.cross_entropy(post[:, -3:], executed)
        if events:
            # 一个物理步十个 0.1 分钟子区间，人数单位为 100 人。
            counts = torch.nn.functional.softplus(pred[:, -10:])
            er = (counts - tx(data["active_path"][ix]) / 100).square().mean()
            estimate = -10 * counts.sum(-1)
        else:
            er = nll(rew / 100, v[:, -1], lv[:, -1]).mean()
            estimate = v[:, -1] * 100
        next_mse = ((nxt[:, 140:core] - v[:, 140:core]).square().mean(-1) * mask).sum() / mask.sum().clamp_min(1)
        auxiliary = model.auxiliary(data, ix, tx, mask) if hasattr(model, "auxiliary") else 0.
        return ep + en + ec + er + auxiliary, ((rew - estimate).abs().mean(),
                                  (mid[:, 140:core] - u[:, 140:]).square().mean().sqrt(), next_mse.sqrt())

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
    target = f"<{Path(ns['__file__']).name}:{fn.__name__}>"
    linecache.cache[target] = (len(source), None, source.splitlines(True), target)
    exec(compile(source, target, "exec"), ns)
    return ns[fn.__name__]


def object_vector(items):
    import numpy as np
    result = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        result[i] = item
    return result


def corrected_trace(reward, value, bootstrap, cont, discount, log_ratio, *, gae=False, enabled=True):
    """物理分钟的价值校正；给 HAPPO 的优势不重复包含 actor 重要性权重。"""
    import numpy as np
    if not all(np.isfinite(t).all() for t in (reward,value,bootstrap,discount,log_ratio)):
        raise ValueError("价值目标收到非有限输入")
    ratio=np.exp(np.clip(log_ratio,-40,40))
    rho=np.minimum(1.,ratio) if enabled and not gae else np.ones_like(ratio)
    c=np.minimum(1.,ratio) if enabled and not gae else np.ones_like(ratio)
    if gae:
        c*=.95
    delta=reward+discount*bootstrap-value
    acc=np.zeros_like(value[0])
    correction=np.empty_like(value)
    for k in range(len(value)-1,-1,-1):
        acc=rho[k]*delta[k]+discount[k]*cont[k]*c[k]*acc
        correction[k]=acc
    returns=value+correction
    if gae:
        return returns,correction,ratio
    nxt=bootstrap.copy()
    nxt[:-1]=np.where(cont[:-1],returns[1:],bootstrap[:-1])
    return returns,reward+discount*nxt-value,ratio


def run(arms, script, argv=None, model_factory=None, fit_factory=None, imagine_factory=None, plugin=None, *, assemble_only=False):
    # 原工程只读；沿用原 HARL 更新、随机数、评估银行和双滚动存档实现。
    ns = vars(legacy).copy()
    scope = dict(kind=None)
    script = Path(script).resolve()
    ns.update(ARMS=arms, __file__=str(script))

    def safe_close(vector):
        # 异常清理只终止此向量环境自己创建的子进程，绝不扫描或终止他人进程。
        import sys
        if sys.exc_info()[0] is None:
            return vector.close()
        vec = getattr(vector, "venv", vector)
        for process in getattr(vec, "processes", ()):
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
        for remote in getattr(vec, "remotes", ()):
            remote.close()
        vec.closed = True

    def approve_repair(path, canonical):
        # 只接受本次已审计旧版本的两项代码指纹迁移，其他合同必须逐项相同。
        old = legacy.read(path)
        known = {"d3b7d0253f96a1ac87bd17418331e342daf5e142032b745dd5abcb7fe9a98027",
                 "83dfdb0239410e809532769644dacb19135bcceef547142134d4d6ea2e3541b8"}
        changed = {k for k in set(old) | set(canonical) if old.get(k) != canonical.get(k)}
        if not changed <= {"core_sha", "entry_sha"} or old.get("core_sha") not in known:
            raise ValueError(f"修复续训不能更改其他合同：{sorted(changed)}")
        audit = path.with_name("PERSON_TIME_REPAIR.json")
        if not audit.exists():
            legacy.save(audit, dict(original_protocol=old, repaired_protocol=canonical,
                                   reason="整数人时校验替代float32校验；异常退出回收自有仿真进程"))

    ns.update(safe_close=safe_close, approve_repair=approve_repair)

    def build_pair(args, space, action_space, device):
        if plugin:
            return plugin.build_pair(args, space, action_space, device, scope["kind"])
        return legacy.build_pair(args, space, action_space, device)

    def make_environment(gym, np):
        return execution_class(gym, np, scope["kind"], plugin)

    def eval_cell(folder, cli, kind, cfg, bank, marks, env_class, args, np, torch):
        scope["kind"] = kind
        eval_fn = patch(legacy.eval_cell, ns, [])
        ns["eval_cell"] = eval_cell
        import gymnasium as gym
        return eval_fn(folder, cli, kind, cfg, bank, marks, make_environment(gym, np), args, np, torch)

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
        return corrected_trace(reward,value,bootstrap,cont,discount,gain,
                               gae=scope["kind"]=="A2_GAE",enabled=enabled)

    ns.update(wm_model=wm, fit_world=fitter, sample_corrected=draw, phase_outcome=outcome, build_pair=build_pair,
              shifted_frames=shifted_frames,
              vtrace=trace_returns, incident_kind=lambda: scope["kind"], plugin=plugin, object_vector=object_vector)
    patch(legacy.checkpoint, ns, [
        ("extra, norm, torch):", "extra, norm, torch, more):"),
        ('    save(folder/"RESUME.json", meta)', '    meta.update(more)\n    save(folder/"RESUME.json", meta)'),
    ])
    imaginer = patch(legacy.imagine, ns, [
        ("return mean+torch.randn_like(mean)*log_std.exp()", "return sample_corrected(param,torch)"),
        ("obs=norm.normalize_obs(dataset['x'][rng.choice(starts,SHADOW_N,replace=True)])",
         "indices=rng.choice(starts,SHADOW_N,replace=True)\n        obs=norm.normalize_obs(dataset['x'][indices])\n        if hasattr(model,'set_history'):\n            model.set_history(torch.as_tensor(dataset['actions_history'][indices],device=device))"),
        ("out=sample(model.next_parameters(x,pi,a.long().flatten()))",
         "out=phase_outcome(model,mid,executed,a,device,torch)"),
        ("y=sample(pp[:,:-3]).clamp(-10.,10.)",
         "y=sample(pp[:,:-3]).clamp(-10.,10.)\n                y[:,:140]=shifted_frames(x,norm,torch)"),
        ("nxt=np.concatenate((array(out[:,:core].clamp(-10.,10.)),",
         "out[:,:140]=y[:,:140]\n                nxt=np.concatenate((array(out[:,:core].clamp(-10.,10.)),"),
        ("                obs=nxt", "                if hasattr(model,'push_history'):\n                    model.push_history(executed,a.long().flatten())\n                obs=nxt"),
        ("                mid=np.concatenate", "                if hasattr(model,'advance_history'):\n                    model.advance_history(x,y,pi,norm,0)\n                mid=np.concatenate"),
        ("                out[:,:140]=y[:,:140]", "                out[:,:140]=y[:,:140]\n                if hasattr(model,'advance_history'):\n                    model.advance_history(torch.as_tensor(mid,device=device),out,a.long().flatten(),norm,1)"),
    ])
    if imagine_factory:
        def imagine(*args):
            return imagine_factory(imaginer, scope["kind"], *args)
        ns["imagine"] = imagine
    trainer = patch(legacy.train_cell, ns, [
        ("        (norm if norm is not None else vec).close()",
         "        safe_close(norm if norm is not None else vec)"),
        ("        depth, use_shadow, use_vtrace = ARMS[arm]",
         '''        depth,use_shadow,use_vtrace = ARMS[arm]
        replay_needed=bool(depth or arm=="I2_CLL")'''),
        ("        opt = torch.optim.Adam(dynamics.parameters(), lr=3e-4) if depth else None",
         '''        parameters=list(dynamics.parameters()) if dynamics is not None else []
        opt=torch.optim.Adam(parameters,lr=3e-4) if parameters else None'''),
        ('''            if depth:
                path = folder/ptr["world"]''',
         '''            if replay_needed:
                path = folder/ptr["world"]'''),
        ('''                dynamics.load_state_dict(extra["wm"])
                opt.load_state_dict(extra["opt"])''',
         '''                if dynamics is not None:
                    dynamics.load_state_dict(extra["wm"])
                if opt is not None:
                    opt.load_state_dict(extra["opt"])'''),
        ('''            extra = (dict(wm=dynamics.state_dict(), opt=opt.state_dict(), ring=ring,
                          rng=rng.bit_generator.state) if depth and real < cli.steps else None)''',
         '''            extra=(dict(wm=dynamics.state_dict() if dynamics is not None else None,
                        opt=opt.state_dict() if opt is not None else None,ring=ring,
                        rng=rng.bit_generator.state) if replay_needed and real<cli.steps else None)'''),
        ('''            if depth:
                for ix in sorted''','''            if replay_needed:
                for ix in sorted'''),
        ('''            if depth:
                ring =''','''            if replay_needed:
                ring ='''),
        ('        initial, since = real, time.monotonic()', '        initial, since = real, time.monotonic()\n        action_history = np.full((n,6,2),3,np.int64)'),
        ('        obs, age = norm.reset(), np.zeros(n, np.int32)', '''        obs, age = norm.reset() if not ptr else None, np.zeros(n,np.int32)
        if ptr:
            if digest(folder/ptr["continuation"]) != ptr["continuation_sha"]:
                raise ValueError("真实续训状态校验失败")
            continuation = torch.load(folder/ptr["continuation"],map_location="cpu",weights_only=False)
            if continuation["real"] != real:
                raise ValueError("真实续训步数校验失败")
            raw_obs=np.stack([vec.env_method("restore_state",s,indices=j)[0] for j,s in enumerate(continuation["env"])])
            norm.old_obs=raw_obs
            norm.returns=continuation["returns"]
            obs, masks, age = continuation["obs"], continuation["masks"], continuation["age"]
            rng.bit_generator.state=continuation["generator"]
            rng_restore(continuation["global_rng"],cli.device,np,torch)'''),
        ('        def capture():', '''        if ptr:
            action_history=continuation["action_history"]
            del continuation
        implicit=incident_kind() in ("I1_LTL","I2_CLL")
        def capture():'''),
        ('            result = checkpoint(folder, real, marks, ptr, state, extra, norm, torch)',
         '''            tag = "rolling_b" if (ptr or {}).get("tag") == "rolling_a" else "rolling_a"
            more={}
            if real < cli.steps:
                path=folder/(tag+".physical.pt")
                continuation=dict(real=real,env=vec.env_method("resume_state"),obs=obs,masks=masks,age=age,
                                  returns=norm.returns,action_history=action_history,global_rng=state["rng"],
                                  generator=rng.bit_generator.state)
                atomic_write(path,lambda p:torch.save(continuation,p))
                more=dict(continuation=path.name,continuation_sha=digest(path))
            result = checkpoint(folder, real, marks, ptr, state, extra, norm, torch, more)'''),
        ("fit_world(dynamics, opt, ring, norm, torch, np, rng, device)",
         "fit_world(dynamics, opt, ring, norm, torch, np, rng, device, fits)"),
        ("fits += 48", "fits += report['optimizer_steps']"),
        ("active = bool(depth and updates >= WARM)", "active = bool(depth and not implicit and updates >= WARM)"),
        ('            for k in range(t):', '''            if incident_kind() == "A5_EVENT_REWARD":
                block["active_path"]=[]
            if incident_kind() == "P2_GRU":
                block["actions_history"]=[]
            if incident_kind() == "P4_HYBRID":
                block["snapshots"]=[]
            if incident_kind() in ("P4_HYBRID","I2_CLL"):
                block["reveals"]=[]
            if implicit:
                legal=np.zeros((2,t,n),bool)
                branches=[[],[]]
            if incident_kind() in ("P4_HYBRID","I2_CLL"):
                selected=set(rng.choice(sorted(k*n+j for k,js in positions.items() for j in js),128 if not implicit else 24,replace=False))
            for k in range(t):'''),
        ('                    x = norm.get_original_obs()[ix].copy()', '''                    x = norm.get_original_obs()[ix].copy()
                    if incident_kind() == "P4_HYBRID":
                        sample_snapshots=object_vector([vec.env_method("branch_state",indices=int(j))[0] if k*n+j in selected else None for j in ix])
                    if incident_kind() == "I2_CLL":
                        for j in ix:
                            if k*n+j in selected:
                                s=vec.env_method("branch_state",indices=int(j))[0]
                                if s is not None:
                                    branches[0].append((obs[j].copy(),s))'''),
        ('                if np.any(reward) or np.any(done)', '''                if implicit and incident_kind() == "I1_LTL":
                    legal[0,k]=[i["legal_opportunity"] for i in infos]
                if incident_kind() == "I2_CLL" and ix is not None:
                    for j in ix:
                        if k*n+j in selected:
                            s=vec.env_method("branch_state",indices=int(j))[0]
                            if s is not None:
                                branches[1].append((mid[j].copy(),s))
                if np.any(reward) or np.any(done)'''),
        ("                        block[key].append(value.copy())",
         '''                        block[key].append(value.copy())
                    if "active_path" in block:
                        block["active_path"].append(np.stack([infos[j]["active_path"] for j in ix]))
                    if "actions_history" in block:
                        block["actions_history"].append(action_history[ix].copy())
                    if "snapshots" in block:
                        block["snapshots"].append(sample_snapshots)
                    if "reveals" in block:
                        block["reveals"].append(object_vector([infos[j]["reveals"] for j in ix]))
                if implicit and incident_kind() == "I1_LTL":
                    legal[1,k]=[i["legal_opportunity"] for i in infos]'''),
        ('                age = np.where(done,0,age+1)',
         '                if incident_kind() == "P2_GRU":\n                    action_history[:, :-1] = action_history[:, 1:].copy()\n                    action_history[:, -1, 0] = executed_batch\n                    action_history[:, -1, 1] = array(a).ravel()\n                    action_history[done] = 3\n                age = np.where(done,0,age+1)'),
        ('                nxt, r, done, infos = norm.step(array(a).ravel())',
         '                if incident_kind() == "P2_GRU":\n                    executed_batch = norm.get_original_obs()[:, -7:-4].argmax(1).copy()\n                nxt, r, done, infos = norm.step(array(a).ravel())'),
        ('            if active:\n                old_logits, shadow_logits', '''            if implicit:
                auxiliary=plugin.implicit_targets(incident_kind(),actors,bs,legal,branches,ring,norm,cfg,rng,np,torch)
                plugin.heldout_report(auxiliary,audit)
            if active:
                old_logits, shadow_logits'''),
        ('            updates += 1', '''            if implicit:
                fits += plugin.fit_implicit(actors,auxiliary,audit,rng,np,torch)
            updates += 1'''),
    ])

    def train_cell(folder, cli, arm, seed, cfg, rows, marks, env_class, args, author, gym, np, torch):
        scope["kind"] = arm
        if (cfg.tick_min, cfg.decision_min) != (.1, 1.):
            raise ValueError("正式走廊合同要求 0.1 分钟子步及 1 分钟物理步")
        env_class = make_environment(gym, np)
        return trainer(folder, cli, arm, seed, cfg, rows, marks, env_class, args, author, gym, np, torch)

    ns.update(train_cell=train_cell, eval_cell=eval_cell, core_sha=legacy.digest(Path(__file__)),
              legacy_sha=legacy.digest(Path(legacy.__file__)), plugin=plugin)
    patch(legacy.paired_summary, ns, [
        ('[(name, "L0_HAPPO") for name in ARMS if name != "L0_HAPPO"] + [\n                ("L1_FULL","L5_NO_VTRACE"), ("L5_NO_VTRACE","L4_DISCARD_SHADOW")]', '[(name,"F_FULL") for name in ARMS if name!="F_FULL"]'),
    ])
    entry = patch(legacy.main, ns, [
        ('            raise ValueError("当前配置、版本或需求与既有输出不一致，请使用新输出路径")',
         '            approve_repair(path, canonical)'),
        ("a = p.parse_args(argv)", 'a = p.parse_args(argv)\n    if a.layers != ["E2"] or a.steps != 1_996_800 or tuple(a.seeds) != (2101,2102,2103,2104):\n        p.error("正式合同固定 E2、四个配对种子及 1,996,800 真实步")'),
        ("default=tuple(LAYERS)", 'default=["E2"]'),
        ('schema="corridor_lookahead_happo_l0l5_v2"', 'schema="corridor_lookahead_phase_e2_v2", core_sha=core_sha, legacy_sha=legacy_sha'),
        ('steps_per_cell=a.steps, rollout=ROLLOUT,', 'steps_per_cell=a.steps, rollout=ROLLOUT, selected_arms=list(a.arms),\n                    optimizer_contract=joint_args, method_boundary=(plugin.METHOD_BOUNDARY if plugin else "走廊 P→A 分阶段预测映射；旧训练预算和 HARL 底座"),'),
    ])
    if assemble_only:
        return ns
    return entry(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(run(ARMS, __file__))
