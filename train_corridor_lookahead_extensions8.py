"""走廊 E2 八组扩展：预测器、时序互动、隐式响应与自适应截断。"""
from __future__ import annotations

import copy
import heapq
import pickle
import sys

import train_corridor_lookahead_core6 as core
from corridor_env import Request

ARMS = {
    "P1_PETS": (1, True, True),
    "P2_GRU": (1, True, True),
    "P3_EVENT_GNN": (1, True, True),
    "P4_HYBRID": (1, True, True),
    "T1_TIGER": (1, True, True),
    "I1_LTL": (0, False, False),
    "I2_CLL": (0, False, False),
    "C1_TRUNCATED": (2, True, True),
}
METHOD_BOUNDARY = {
    "P1_PETS":"五个 bootstrap 概率模型和 TS∞；不使用 PETS 的 CEM/MPC",
    "P2_GRU":"滚动六帧，实际执行选站与调机请求历史、角色相位和相对时间",
    "P3_EVENT_GNN":"物理实体和事件图只给世界模型；预测下一已承诺事件类与相对时间",
    "P4_HYBRID":"精确走廊物理；只从已观察训练回放拟合未知需求；奖励精确积分",
    "T1_TIGER":"有向角色历史与时间注意力接入模型及策略；保留 HARL，不迁移原值分解底座",
    "I1_LTL":"候选动作条件隐式响应，训练标签为真实下次合法决策，不用影子行为",
    "I2_CLL":"合法快照的三候选配对反事实，相同随机数，参考动作 V0，不用影子行为",
    "C1_TRUNCATED":"按角色分开计算 KL 与留出误差，两步混合，真实采样权重用混合分布",
}
BASE = core.legacy.PACK
SITE = ("V0", "V1", "V3", "V2")
EVENTS = ("_arrive_queue", "_launch", "_air_phase", "_land", "_turned", "_charged", "_finish")
FRAME, WINDOW = 15, 6
TIGER_WIDTH = FRAME * WINDOW
PLANE_STATES = ("ready", "reserved", "takeoff", "flight", "landing", "turn_queue", "turnaround", "charge_queue", "charging")
GRAPH_WIDTH = 16
GRAPH_COUNTS = (4, 4, 4, 4, 48, 64)
GRAPH_N = sum(GRAPH_COUNTS)
GRAPH_EXTRA = GRAPH_N * GRAPH_WIDTH + 2


def extra_width(kind):
    return TIGER_WIDTH if kind == "T1_TIGER" else GRAPH_EXTRA if kind == "P3_EVENT_GNN" else 0


def record(env, role, legal, action, kind):
    if kind != "T1_TIGER":
        return
    out = role_record(env.env, role, legal, action)
    env.history_records = (env.history_records + [out])[-WINDOW:]


def role_record(e, role, legal, action):
    queues = [len(e.queue[s]) for s in SITE[:3]]
    ready = [sum(p.station == s and p.state == "ready" for p in e.planes.values()) for s in SITE]
    return (1., float(role), float(legal), *(float(legal and action == i) for i in range(3)),
            e.now * e.cfg.tick_min / 600, *(q/300 for q in queues), *(v/48 for v in ready),
            len(e.passengers)/300)


def _pending_events(e):
    # 只读取已承诺事件，排序后保留最临近的 64 条；记录溢出量。
    return sorted(v for v in e._events if v[2].__name__ in EVENTS)


def graph(e, np):
    nodes = np.zeros((GRAPH_N, GRAPH_WIDTH), np.float32)
    pending = _pending_events(e)

    def put(i, station, features):
        nodes[i, 0] = 1.
        if station in SITE:
            nodes[i, 1+SITE.index(station)] = 1.
        nodes[i, 5:5+len(features)] = features

    for j, s in enumerate(SITE):
        st = e.cfg.stations[s]
        put(j, s, [st.xy[0]/60, st.xy[1]/60, e.local[s]/48, e.pa[s]/48,
                   e.turn_active[s]/48, len(e.turnq[s])/48, float(s in e.arr)])
        access = sum(p.dep == s and p.state == "access" for p in e.passengers.values())
        put(4+j, s, [len(e.queue.get(s, ()))/300, access/300])
        pad = e.pads.get(s)
        delays = [] if pad is None else [max(0,max((b+pad.gap for _,b in lane),default=e.now)-e.now)
                                         *e.cfg.tick_min/60 for lane in pad.lanes]
        put(8+j, s, [st.pads/8, sum(v > 0 for v in delays)/8, min(delays, default=0),
                     max(delays, default=0), st.takeoff_min, st.landing_min])
        put(12+j, s, [e.charge_active[s]/48, len(e.chargeq[s])/48,
                      float(e.cfg.finite_charging), st.chargers/48, st.charge_soc_per_min])
    if len(e.planes) != 48:
        raise ValueError("事件图的 48 架飞机槽位必须匹配冻结 E2")
    for k, p in enumerate(sorted(e.planes.values(), key=lambda a: a.id)):
        status = PLANE_STATES.index(p.state)+1 if p.state in PLANE_STATES else 0
        put(16+k, p.station, [p.soc, status/len(PLANE_STATES), len(p.riders)/4,
                              *(float(p.target == s) for s in SITE), p.id/48])
    start = 16 + 48
    for k, (when, _, fn, args) in enumerate(pending[:64]):
        subject = args[0]
        kind = EVENTS.index(fn.__name__)
        site = getattr(subject, "station", None) or getattr(subject, "dep", None) or getattr(subject, "arr", None)
        air = hasattr(subject, "soc")
        put(start+k, site, [(when-e.now)*e.cfg.tick_min/60, float(air),
                            subject.id/48 if air else 0.,
                            *(float(j == kind) for j in range(len(EVENTS)))])
    label = (EVENTS.index(pending[0][2].__name__)+1, (pending[0][0]-e.now)*e.cfg.tick_min/60) if pending else (0,0)
    nodes[0,-1] = max(0,len(pending)-64)/300
    return np.concatenate((nodes.ravel(), np.asarray(label,np.float32)))


def observation(env, raw, post, np, kind):
    if kind == "P3_EVENT_GNN":
        extra = graph(env.env, np)
    elif kind == "T1_TIGER":
        extra = np.zeros((WINDOW,FRAME),np.float32)
        n = len(env.history_records)
        if n:
            extra[-n:] = env.history_records
        extra = extra.ravel()
    else:
        return raw
    # 原尾部的相位和实际选站投影位置不变。
    return np.concatenate((raw[:-8],extra,raw[-8:]))


def branch_state(env):
    e = env.env
    if e.done:
        return None
    events,total = e._events,e.request_count
    try:
        e._events = [v for v in events if v[2].__name__ != "_reveal"]
        heapq.heapify(e._events)
        e.request_count=len(e.passengers)+1
        payload = pickle.dumps((e,env.history,env.history_records,env.phase,env.route_index), protocol=5)
    finally:
        e._events,e.request_count=events,total
    return payload


def restore_branch(payload, constructor):
    from collections import Counter
    e, history, records, role, route = pickle.loads(payload)
    if any(v[2].__name__ == "_reveal" for v in e._events):
        raise ValueError("虚拟快照不允许携带真实未揭示请求")
    # 哨兵仅用来禁止在预测区间提前停止，不生成乘客，也不写入人时。
    e.request_count = len(e.passengers)+1
    obj = constructor(e.cfg, (), "HAPPO")
    obj.env,obj.history,obj.history_records,obj.phase,obj.route_index,obj.stat = e,history,records,role,route,Counter()
    return obj


def old_obs(x, torch):
    return torch.cat((x[..., :BASE],x[..., -8:]),-1)


def temporal_encoder(torch, width=64):
    nn = torch.nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Sequential(nn.Linear(FRAME,width),nn.SiLU(),nn.LayerNorm(width))
            self.query = nn.Linear(168,width)
            self.role = nn.Embedding(2,width)
            self.edge = nn.Embedding(3,width)
            self.freq = nn.Parameter(2.**(-torch.linspace(-2,4,width)))
            self.shift = nn.Parameter(torch.zeros(width))
            self.attention = nn.MultiheadAttention(width,4,batch_first=True)
            self.static_edge=nn.Linear(2*width,1)

        def forward(self,x,role):
            h=x[:,BASE:BASE+TIGER_WIDTH].reshape(-1,WINDOW,FRAME)
            exists = h[:,:,0] > 0
            prior_role = (h[:,:,1]>0).long()
            q=self.query(x[:,:168])+self.role.weight[role]
            # 分别编码自历史、跨角色历史与最近同分钟交互。
            ordinal = torch.arange(WINDOW,device=x.device)[None,:]
            back=WINDOW-1-ordinal
            # 决策事件索引不同于物流分钟；P→A 的本分钟静态边时差为零。
            lag=torch.div(back+(1 if role==1 else 2),2,rounding_mode="floor").to(x.dtype)
            edge=torch.where(prior_role==role,0,1)
            if role == 1:
                edge=torch.where((ordinal==WINDOW-1)&(prior_role==0),2,edge)
            key=self.project(h)+self.role(prior_role)+self.edge(edge)
            key=key+torch.cos(lag[...,None]*self.freq+self.shift)
            # K=1：唯一另一角色已知才有邻接；最近邻实例确定其跨角色历史权重。
            known=exists & (prior_role!=role)
            last=torch.where(known,ordinal,-1).amax(1)
            other=key[torch.arange(len(x),device=x.device),last.clamp_min(0)]
            gate=self.static_edge(torch.cat((q,other),-1)).sigmoid()
            key=torch.where(known[...,None], key*gate[:,None], key)
            # 当前节点作始终存在的自环；历史边仅连接已记录节点。
            kv=torch.cat((q[:,None],key),1)
            pad=torch.cat((torch.zeros((len(x),1),device=x.device,dtype=torch.bool),~exists),1)
            return self.attention(q[:,None],kv,kv,key_padding_mask=pad,need_weights=False)[0][:,0]

    return Encoder()


def graph_model(dim,torch):
    nn=torch.nn
    width=48
    c=dim-8
    kinds=torch.repeat_interleave(torch.arange(len(GRAPH_COUNTS)),torch.tensor(GRAPH_COUNTS))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("kinds",kinds)
            self.node=nn.Linear(GRAPH_WIDTH,width)
            self.kind=nn.Embedding(6,width)
            self.message=nn.ModuleList(nn.Linear(width,width) for _ in range(6))
            self.relay=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.LayerNorm(width))
            self.broadcast=nn.Sequential(nn.Linear(2*width,width),nn.SiLU())
            context=4*width+width+BASE+8+3
            self.post=nn.Sequential(nn.Linear(context,192),nn.SiLU(),nn.Linear(192,2*c+3))
            self.advance=nn.Sequential(nn.Linear(context,192),nn.SiLU(),nn.Linear(192,2*(c+1)))
            self.events=nn.Linear(context,len(EVENTS)+2)

        def features(self,x,a):
            raw=x[:,BASE:BASE+GRAPH_N*GRAPH_WIDTH].reshape(-1,GRAPH_N,GRAPH_WIDTH)
            present=(raw[:,:,0]>0).float()
            # 固定实体永远存在；可变事件的存在位由模型预测。
            present[:,:64]=1.
            memberships=(raw[:,:,1:5]>0).float()*present[...,None]
            for j in range(4):
                memberships[:,j,j]=1.
                for o in (4,8,12):
                    memberships[:,o+j,j]=1.
            # 飞机既连接当前机场，也连接已承诺的目标机场。
            memberships[:,16:64]=torch.maximum(memberships[:,16:64],(raw[:,16:64,8:12]>0).float())
            h=torch.nn.functional.silu(self.node(raw)+self.kind(self.kinds)[None])
            msgs=torch.cat([layer(chunk) for layer,chunk in zip(self.message,h.split(GRAPH_COUNTS,1))],1)
            station=memberships.transpose(1,2).bmm(msgs)/memberships.sum(1).clamp_min(1.)[...,None]
            relay=self.relay(torch.cat((h[:,:4],station),-1))
            context=self.broadcast(torch.cat((h,memberships.bmm(relay)),-1))
            pool=(context*present[...,None]).sum(1)/present.sum(1).clamp_min(1.)[:,None]
            return torch.cat((old_obs(x,torch),relay.flatten(1),pool,nn.functional.one_hot(a,3).float()),-1)

        def post_parameters(self,x,p):
            return self.post(self.features(x,p))

        def next_parameters(self,mid,p,a):
            context=self.features(mid,a)
            self.event_prediction=self.events(context)
            return self.advance(context)

        def auxiliary(self,data,inds,tx,mask):
            target=tx(data["next"][inds,BASE+GRAPH_N*GRAPH_WIDTH:BASE+GRAPH_N*GRAPH_WIDTH+2])
            typ=target[:,0].long()
            pred=self.event_prediction
            ce=torch.nn.functional.cross_entropy(pred[:,:-1],typ,reduction="none")
            time=(pred[:,-1]-target[:,1]).square()*(typ!=0)
            return ((ce+time)*mask).sum()/mask.sum().clamp_min(1.)

    return Model()


def tiger_model(dim,torch):
    nn=torch.nn
    c=dim-8

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal=temporal_encoder(torch)
            dw=BASE+8+64+3
            self.post=nn.Sequential(nn.Linear(dw,192),nn.SiLU(),nn.Linear(192,2*c+3))
            self.advance=nn.Sequential(nn.Linear(dw,192),nn.SiLU(),nn.Linear(192,2*(c+1)))

        def features(self,x,a,role):
            return torch.cat((old_obs(x,torch),self.temporal(x,role),nn.functional.one_hot(a,3).float()),-1)

        def post_parameters(self,x,p):
            return self.post(self.features(x,p,0))

        def next_parameters(self,mid,p,a):
            return self.advance(self.features(mid,a,1))

        def advance_history(self,source,out,action,norm,role):
            # 由虚拟轨迹自身追加角色节点，不能让高斯头改写已经发生的互动。
            stats=norm.obs_rms
            loc=torch.as_tensor(stats.mean,dtype=source.dtype,device=source.device)
            scale=torch.as_tensor(stats.var+norm.epsilon,dtype=source.dtype,device=source.device).sqrt()
            x=source*scale+loc
            records=x[:,BASE:BASE+TIGER_WIDTH].reshape(-1,WINDOW,FRAME)
            cur=torch.zeros_like(records[:,0])
            cur[:,0]=1
            cur[:,1]=role
            res=x[:,172:210]
            queue=res[:,[0,10,20]]
            ready=res[:,[2,12,22,30]]
            legal=(x[:,BASE-1]>.5) if role==0 else ((queue>.001).any(-1)&(ready[:,3]>.001))
            cur[:,2]=legal.float()
            cur[:,3:6]=nn.functional.one_hot(action,3).float()*legal[:,None]
            cur[:,6]=res[:,35]
            cur[:,7:10]=queue.clamp(0,1)
            cur[:,10:14]=(ready*20/48).clamp(0,1)
            cur[:,14]=res[:,33].clamp(0,1)
            rolled=torch.cat((records[:,1:],cur[:,None]),1).flatten(1)
            k=slice(BASE,BASE+TIGER_WIDTH)
            out[:,k]=((rolled-loc[k])/scale[k]).clamp(-norm.clip_obs,norm.clip_obs)

    return Model()


def models(dim, torch, kind):
    nn = torch.nn
    if kind=="P4_HYBRID":
        # 经验需求分布由存档中的真实回放重建；不建无用的标量奖励头。
        return nn.Module()
    if kind == "P3_EVENT_GNN":
        return graph_model(dim,torch)
    if kind == "T1_TIGER":
        return tiger_model(dim,torch)
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

            def features(self, x, a, phase, p=None):
                frames = x[:, :168].reshape(-1, 6, 28)
                lag = torch.arange(-5, 1, device=x.device, dtype=x.dtype)[None, :, None].expand(len(x), -1, -1)
                history = self.actions.long()
                if phase:
                    last=torch.stack((p,torch.full_like(p,3)),-1)[:,None]
                    history=torch.cat((history[:,1:],last),1)
                actions = nn.functional.one_hot(history, 4).reshape(len(x), 6, 8).float()
                _, h = self.history(torch.cat((frames, lag, torch.full_like(lag,phase), actions), -1))
                return torch.cat((x, h[-1], nn.functional.one_hot(a, 3).float()), -1)

            def post_parameters(self, x, p):
                return self.post(self.features(x, p, 0))

            def next_parameters(self, mid, p, a):
                return self.advance(self.features(mid, a, 1, p))

        return HistoryModel()
    return None


def build_pair(args, space, action_space, device, kind):
    if kind not in ("P3_EVENT_GNN","T1_TIGER","I1_LTL","I2_CLL"):
        return core.legacy.build_pair(args,space,action_space,device)
    import gymnasium as gym
    import numpy as np
    import torch
    from harl.models.base.distributions import FixedCategorical

    nn=torch.nn
    legacy_space=gym.spaces.Box(-np.inf,np.inf,(BASE+8,),dtype=np.float32)
    actors,critic,vn=core.legacy.build_pair(args,legacy_space,action_space,device)
    if kind in ("P3_EVENT_GNN","T1_TIGER"):
        class PolicyFeatures(nn.Module):
            def __init__(self,base,role=None):
                super().__init__()
                self.base,self.role=base,role
                if role is not None:
                    self.history=temporal_encoder(torch)
                    self.residual=nn.Linear(64,args["hidden_sizes"][-1])
                    nn.init.normal_(self.residual.weight,std=.005)
                    nn.init.zeros_(self.residual.bias)

            def forward(self,x):
                out=self.base(old_obs(x,torch))
                if self.role is None:
                    return out
                h=(self.history(x,0)+self.history(x,1))*.5 if self.role==2 else self.history(x,self.role)
                return out+self.residual(h)

        critic.critic.base=PolicyFeatures(critic.critic.base,2 if kind=="T1_TIGER" else None).to(device)
        if kind=="T1_TIGER":
            critic.critic_optimizer=torch.optim.Adam(critic.critic.parameters(),lr=args["critic_lr"],
                                                       eps=args["opti_eps"],weight_decay=args["weight_decay"])
        for j,actor in enumerate(actors):
            actor.actor.base=PolicyFeatures(actor.actor.base,j if kind=="T1_TIGER" else None).to(device)
            if kind=="T1_TIGER":
                actor.actor_optimizer=torch.optim.Adam(actor.actor.parameters(),lr=args["lr"],
                                                       eps=args["opti_eps"],weight_decay=args["weight_decay"])
        return actors,critic,vn

    class LatentPolicy(nn.Module):
        def __init__(self,base,role):
            super().__init__()
            self.base=base
            self.role=role
            # 固定随机正交投影只压缩真实响应状态；动作、机会和时延有独立语义槽。
            gen=torch.Generator(device="cpu").manual_seed(43871+role)
            weight=torch.randn((188,20),generator=gen)
            self.register_buffer("response_projection",torch.linalg.qr(weight,mode="reduced").Q)
            self.predictor=nn.Sequential(nn.Linear(BASE+8,128),nn.SiLU(),nn.Linear(128,3*25))
            self.latent_score=nn.Sequential(nn.Linear(25,32),nn.SiLU(),nn.Linear(32,1))
            self.context_score=nn.Linear(args["hidden_sizes"][-1],25)
            nn.init.zeros_(self.latent_score[-1].weight)
            nn.init.zeros_(self.latent_score[-1].bias)

        def convert(self,x):
            return torch.as_tensor(x,**self.base.tpdv)

        def pred(self,x):
            raw=self.predictor(self.convert(x)).reshape(-1,3,25)
            if kind=="I2_CLL":
                raw=raw-raw[:,:1]
            return raw

        def semantic(self,peer_obs,peer_act,delay):
            x=self.convert(peer_obs)
            features=torch.cat((x[:,:168],x[:,172:192]),-1)
            y=torch.tanh(features@self.response_projection)
            a=nn.functional.one_hot(torch.as_tensor(peer_act,device=x.device).long().flatten(),3).float()
            t=self.convert(delay).reshape(-1,1).clamp(0,20)/20
            return torch.cat((y,a,torch.ones_like(t),t),-1)

        def dist(self,obs,available=None):
            x=self.convert(obs)
            features=self.base.base(x)
            z=self.pred(x).detach()
            w=self.context_score(features)[:,None].tanh()
            add=self.latent_score(z).squeeze(-1)+(w*z).sum(-1)*.2
            logits=self.base.act.action_out.linear(features)+add
            if available is not None:
                logits=logits.masked_fill(self.convert(available)==0,-1e10)
            return FixedCategorical(logits=logits)

        def forward(self,obs,rnn_states,masks,available_actions=None,deterministic=False):
            d=self.dist(obs,available_actions)
            a=d.mode() if deterministic else d.sample()
            return a,d.log_probs(a),self.convert(rnn_states)

        def evaluate_actions(self,obs,rnn_states,action,masks,available_actions=None,active_masks=None):
            d=self.dist(obs,available_actions)
            ent=d.entropy()
            if active_masks is None or not self.base.use_policy_active_masks:
                entropy=ent.mean()
            else:
                mask=self.convert(active_masks).reshape(-1)
                entropy=(ent*mask).sum()/mask.sum().clamp_min(1)
            return d.log_probs(self.convert(action).long().reshape(-1,1)),entropy,d

    for j,actor in enumerate(actors):
        actor.actor=LatentPolicy(actor.actor,j).to(device)
        # 预测器只收监督梯度，不让 PPO 间接改变其训练目标。
        actor.actor_optimizer=torch.optim.Adam((p for name,p in actor.actor.named_parameters() if not name.startswith("predictor.")),
                                               lr=args["lr"],eps=args["opti_eps"],weight_decay=args["weight_decay"])
    return actors,critic,vn


def fit_extension(model, opt, ring, norm, torch, np, rng, device, kind, iterations):
    if kind == "P4_HYBRID":
        data = {k: np.concatenate([b[k] for b in ring]) for k in ring[0]}
        model.demand = LearnedDemand(ring,np)
        return data,dict(optimizer_steps=0,demand_fit_rows=model.demand.n,
                         demand_holdout_mae=model.demand.mae)
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


class LearnedDemand:
    """一分钟到达数的分段经验分布；标记为同一批次内 OD 与相对到达时刻。"""

    def __init__(self,ring,np):
        self.bins={}
        holdout=[]
        self.n=0
        row=0
        for block in ring:
            for age,reveals,done in zip(block["age"],block["reveals"],block["done"]):
                row+=1
                if done:
                    continue
                key=int(age)//30
                if row%8:
                    self.bins.setdefault(key,[]).append(reveals)
                    self.n+=1
                else:
                    holdout.append((key,len(reveals)))
        means={k:sum(map(len,v))/len(v) for k,v in self.bins.items()}
        errors=[abs(y-means[key]) for key,y in holdout if key in means]
        self.mae=float(np.mean(errors)) if errors else None

    def schedule(self,e,rng):
        choices=self.bins.get(int(e.now*e.cfg.tick_min)//30)
        if not choices:
            return
        for j,(offset,origin,destination) in enumerate(choices[int(rng.integers(len(choices)))]):
            tick=e.now+int(offset)
            if not e.now < tick <= e.cutoff:
                continue
            key=f"_synthetic_{e.now}_{j}"
            e._push(tick,e._reveal,Request(key,tick*e.cfg.tick_min,origin,destination))
            e.request_count+=1


def sample_peer(actor,raw,norm,np,torch,quantile):
    x=norm.normalize_obs(raw[None])
    n=actor.actor
    if hasattr(n,"dist"):
        prob=n.dist(x).probs[0]
    else:
        z=np.zeros((1,1,128),np.float32)
        ones=np.ones((1,1),np.float32)
        prob=actor.evaluate_actions(x,z,np.zeros((1,1),np.int64),ones,None,None)[2].probs[0]
    cdf=prob.detach().cpu().numpy().cumsum()
    action=min(2,int(np.searchsorted(cdf,quantile,side="right")))
    return x, action


def factual_pairs(buffers,legal,role,np):
    """匹配严格后继的合法队友响应，不把 reset 或 rollout 末端当作无机会。"""
    t,n=legal.shape[1:]
    out=[]
    for k in range(n):
        next_peer=None
        end=t-1
        for minute in range(t-1,-1,-1):
            if not buffers[0].masks[minute+1,k,0]:
                next_peer=None
                end=minute
            for current in (1,0):
                if not legal[current,minute,k]:
                    continue
                if current!=role:
                    next_peer=(minute,buffers[current].obs[minute,k],
                               int(buffers[current].actions[minute,k,0]))
                elif next_peer is not None and next_peer[0]-minute<=20:
                    out.append((buffers[role].obs[minute,k],int(buffers[role].actions[minute,k,0]),
                                next_peer[1],next_peer[2],next_peer[0]-minute,True))
                elif end-minute>20:
                    out.append((buffers[role].obs[minute,k],int(buffers[role].actions[minute,k,0]),
                                buffers[role].obs[minute,k],0,0,False))
    return out


def _branch_label(role, candidate, payload, constructor, actors, norm, demand, seed, np, torch):
    fork=restore_branch(payload,constructor)
    e=fork.env
    start=e.now
    raw=fork.observation(post=(role==1))
    for step in range(41):
        who=fork.phase
        if e.done or (e.now-start)*e.cfg.tick_min>20:
            break
        # 按绝对物理时点和角色配对随机数，不随分支循环次数漂移。
        rng=np.random.default_rng(np.random.SeedSequence([seed,e.now,who]))
        legal=fork.focal() is not None if who==0 else bool(e.targets())
        x,a=sample_peer(actors[who],raw,norm,np,torch,float(rng.random()))
        if step and who!=role and legal:
            return actors[role].actor.semantic(x,[a],[(e.now-start)*e.cfg.tick_min])[0]
        if not step:
            a=candidate
        if who==1:
            demand.schedule(e,rng)
        raw,_,_,_,_=fork.step(a)
    return torch.zeros(25,device=actors[role].tpdv["device"])


def implicit_targets(kind,actors,buffers,legal,branches,ring,norm,cfg,rng,np,torch):
    data=[]
    if kind=="I2_CLL":
        import gymnasium as gym
        ctor=core.execution_class(gym,np,kind,sys.modules[__name__])
        demand=LearnedDemand(ring[:-1],np)
        if not demand.n:
            return [None,None]
    with torch.no_grad():
        for role in range(2):
            if kind=="I1_LTL":
                items=factual_pairs(buffers,legal,role,np)
                if not items:
                    data.append(None)
                    continue
                x,a,state,action,delay,present=zip(*items)
                y=actors[role].actor.semantic(np.stack(state),action,delay)
                y=y*torch.as_tensor(present,device=y.device)[:,None]
                data.append((np.stack(x),np.asarray(a),y))
            else:
                xs,ys=[],[]
                for x,payload in branches[role]:
                    tmp=restore_branch(payload,ctor)
                    if not (tmp.focal() is not None if role==0 else tmp.env.targets()):
                        continue
                    seed=int(rng.integers(1,2**31))
                    y=torch.stack([_branch_label(role,i,payload,ctor,actors,norm,demand,seed,np,torch)
                                   for i in range(3)])
                    xs.append(x)
                    ys.append(y-y[:1])
                data.append((np.stack(xs),None,torch.stack(ys)) if xs else None)
    return data


def heldout_report(data,audit):
    audit["implicit_rows"]=[0 if d is None else len(d[0]) for d in data]


def fit_implicit(actors,data,audit,rng,np,torch):
    rows=[]
    steps=0
    for actor,pack in zip(actors,data):
        if pack is None or len(pack[0])<2:
            rows.append(None)
            continue
        x,a,y=pack
        model=actor.actor
        ids=np.arange(len(x))
        train=ids[ids%8!=0]
        valid=ids[ids%8==0]

        def loss(ix):
            z=model.pred(x[ix])
            ti=torch.as_tensor(ix,device=z.device)
            if a is not None:
                z=z[torch.arange(len(ix),device=z.device),torch.as_tensor(a[ix],device=z.device)]
            return torch.nn.functional.smooth_l1_loss(z,y[ti])

        # 每步无动量的监督更新，因此断点不存在丢失辅助优化器状态的问题。
        opt=torch.optim.SGD(model.predictor.parameters(),lr=.02)
        for _ in range(4):
            ix=rng.choice(train,min(len(train),256),replace=False)
            opt.zero_grad(set_to_none=True)
            loss(ix).backward()
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            steps+=1
        with torch.no_grad():
            rows.append(dict(holdout_loss=float(loss(valid)),target_rms=float(y.square().mean().sqrt()),
                             validation_rows=len(valid)))
    audit["implicit_supervision"]=rows
    return steps


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


def imagine_hybrid(team,args,space,action_space,model,norm,dataset,author,Buffer,CB,np,torch,rng,cutoff):
    import gymnasium as gym
    legacy=core.legacy
    actors,critic,vn=team
    n,h=legacy.SHADOW_N,legacy.SHADOW_H
    shape=dict(args,episode_length=h,n_rollout_threads=n)
    for actor in actors:
        actor.actor_num_mini_batch=10
        actor.entropy_coef=.001
    critic.critic_num_mini_batch=10
    bs=[Buffer(shape,space,action_space) for _ in actors]
    cb=CB(shape,space)
    runner=legacy.runner_for(shape,actors,critic,bs,cb,vn)
    z=np.zeros((n,args["recurrent_n"],args["hidden_sizes"][-1]),np.float32)
    ones=np.ones((n,1),np.float32)
    starts=[i for i,s in enumerate(dataset["snapshots"]) if s is not None and i%8 and not dataset["done"][i]]
    # 截止附近不作为起点，防止不足 20 分钟却统计为完整虚拟预算。
    ctor=core.execution_class(gym,np,"P4_HYBRID",sys.modules[__name__])
    admissible=[i for i in starts if dataset["age"][i]+h<cutoff]
    if not admissible:
        raise RuntimeError("已承诺世界模型没有可用的 20 分钟起点")
    indices=rng.choice(admissible,n,replace=True)
    forks=[restore_branch(dataset["snapshots"][i],ctor) for i in indices]
    for fork in forks:
        e=fork.env
        if e.now+h*e._ticks(e.cfg.decision_min)>=e.cutoff or fork.phase!=0:
            raise RuntimeError("虚拟起点的物理时钟或决策阶段不一致")
    obs=norm.normalize_obs(np.stack([e.observation() for e in forks]))
    for a in actors:
        a.prep_rollout()
    critic.prep_rollout()
    rewards=[]
    def arr(x):
        return x.detach().cpu().numpy()

    with torch.no_grad():
        for j in range(h):
            values,_=critic.get_values(obs,z,ones)
            p,lp,_=actors[0].get_actions(obs,z,ones)
            mid=norm.normalize_obs(np.stack([e.step(int(i))[0] for e,i in zip(forks,arr(p).reshape(-1))]))
            a,la,_=actors[1].get_actions(mid,z,ones)
            results=[]
            for e,i in zip(forks,arr(a).reshape(-1)):
                model.demand.schedule(e.env,rng)
                results.append(e.step(int(i)))
            if any(done or cut for _,_,done,cut,_ in results):
                raise RuntimeError("虚拟回合在批准的 20 分钟内提前终止")
            nxt=norm.normalize_obs(np.stack([x[0] for x in results]))
            r=np.asarray([x[1] for x in results],np.float32)[:,None]
            rewards.append(float(r.mean()))
            bs[0].obs[j],bs[1].obs[j],cb.share_obs[j]=obs,mid,obs
            for b,k,log in zip(bs,(p,a),(lp,la)):
                b.insert(nxt,z,arr(k).reshape(n,1),arr(log).reshape(n,1),ones,ones)
            cb.insert(nxt,z,arr(values),r,ones,ones)
            obs=nxt
        vals,_=critic.get_values(obs,z,ones)
        cb.compute_returns(arr(vals),vn)
    for actor in actors:
        actor.prep_training()
    critic.prep_training()
    am,cm=author(runner)
    return [dict(mean_imagined_reward=float(np.mean(rewards)),actor=am,critic=cm,
                 simulator_physical_steps=n*h,observed_only_starts=len(admissible))]


def imagine_extension(imaginer, kind, team, args, space, action_space, model, norm, dataset,
                      depth, author, Buffer, CB, np, torch, rng, device, cutoff):
    params = (args, space, action_space, model, norm, dataset)
    end = (author, Buffer, CB, np, torch, rng, device, cutoff)
    if kind == "P4_HYBRID":
        return imagine_hybrid(team,*params,author,Buffer,CB,np,torch,rng,cutoff)
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
    # 子进程通过固定模块名导入观察插件，包括 Windows spawn。
    import train_corridor_lookahead_extensions8 as implementation
    return core.run(ARMS,__file__,argv,implementation.models,implementation.fit_extension,
                    implementation.imagine_extension,implementation)


if __name__ == "__main__":
    raise SystemExit(main())
