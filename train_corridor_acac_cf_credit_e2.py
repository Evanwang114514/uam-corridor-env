"""E2 分钟批次补充：B3 配对反事实动作 Q 与 B4 配对差分信用。"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import copy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
import traceback

import train_corridor_acac_batch_e2 as batch


base, old = batch.base, batch.old
METHODS = {"B3_COMA": "order", "B4_DIFFERENCE": "credit"}
PARENT = {"order": (16, 2*len(batch.SEEDS)*batch.STEPS),
          "credit": (12, 3*len(batch.SEEDS)*batch.STEPS)}
WARMUP_ROLLOUTS, LABEL_WINDOW = 4, 8


def check_copy(original, cloned):
    """物理事件回调和事件对象必须指向自己的克隆世界。"""
    a,b = original.env,cloned.env
    if (a is b or (a.now,a.area_ticks,original.remaining)!=
            (b.now,b.area_ticks,cloned.remaining)):
        raise AssertionError("反事实快照改变了物理起点")
    for l,r in zip(a._events,b._events,strict=True):
        if (l[:2]!=r[:2] or l[2].__name__!=r[2].__name__
                or getattr(r[2],"__self__",None) is not b):
            raise AssertionError("克隆事件回调没有绑定到分支")
        for x,y in zip(l[3],r[3],strict=True):
            if hasattr(x,"req") and (x is y or b.passengers[y.req.id] is not y):
                raise AssertionError("分支事件仍引用主轨迹乘客")
            if type(x).__name__=="Aircraft" and (x is y or b.planes[y.id] is not y):
                raise AssertionError("分支事件仍引用主轨迹飞机")


@dataclass
class Start:
    lane: object
    hidden: object
    pid: str
    q: int
    event: object


@dataclass
class Branch:
    root: int
    action: int
    lane: object
    h: object
    steps: int = 0
    gain: float = 0.
    terminal: bool = False
    truncated: bool = False


def draw_keyed(torch,np,actors,norm,offers,role,seed):
    if not offers:
        return []
    ln = [p.lane for p,_,_ in offers]
    obs = torch.as_tensor(norm.normalize_obs(np.stack([x for _,_,x in offers])))
    clock = torch.tensor([x.clock for x in ln],dtype=torch.float32)
    h = torch.from_numpy(np.stack([p.h[role] for p,_,_ in offers]))
    model = actors[role]
    h2 = model.history.one(obs,clock,torch.ones(len(ln),dtype=torch.bool),h)
    probs = model.head(h2).softmax(-1).cumsum(-1).cpu().numpy()
    result = []
    for i,(part,pid,_) in enumerate(offers):
        lane = part.lane
        # 同一真实决策键在各分支共享随机数，早终止/无机会不会移位其他键。
        u = batch.uniform("cf_follow",seed,lane.index,lane.episode,lane.now,role,pid)
        result.append(min(2,int(probs[i].searchsorted(u,side="right"))))
        part.h[role] = h2[i].cpu().numpy()
    return result


def branch_rollout(torch,np,actors,norm,starts,method,seed,gamma):
    pieces=[]
    for i,start in enumerate(starts):
        event=start.event
        actions=range(len(base.STATIONS)) if method=="B3_COMA" else sorted({event.action,event.cf_ref})
        for action in actions:
            lane=copy.deepcopy(start.lane)
            check_copy(start.lane,lane)
            part=Branch(i,action,lane,start.hidden.copy())
            lane.commit(start.pid,action)
            pieces.append(part)
    active=list(pieces)
    while active:
        while True:
            ps=[(p,p.lane.remaining[0],p.lane.observation(p.lane.remaining[0])) for p in active if p.lane.remaining]
            if not ps:
                break
            for (p,pid,_),action in zip(ps,draw_keyed(torch,np,actors,norm,ps,0,seed),strict=True):
                p.lane.commit(pid,action)
        air=[(p,None,p.lane.mid) for p in active if p.lane.view.eligible()]
        by_obj={id(p):a for (p,_,_),a in zip(air,draw_keyed(torch,np,actors,norm,air,1,seed),strict=True)}
        more=[]
        for p in active:
            reward,done,trunc=p.lane.advance(by_obj.get(id(p),0))
            p.gain += gamma**p.steps*reward
            p.steps += 1
            p.terminal,p.truncated=done,trunc
            if not (done or trunc):
                p.lane.begin_tick()
                if not p.lane.pending:
                    p.lane.commit(None,0)
                more.append(p)
        active=more
    by_root={i:{} for i in range(len(starts))}
    for p in pieces:
        metrics=p.lane.env.metrics()
        by_root[p.root][p.action]=dict(gain=p.gain,steps=p.steps,terminal=p.terminal,
            truncated=p.truncated,completion=metrics["completion"],att=metrics["mean_att_min"])
        p.lane.view.close()
    return by_root


class Collector(batch.Collector):
    def collect_cf(self,length,real):
        self.snapshots={}
        self.index=-1
        self.gates=[int(batch.uniform("cf_pick",self.seed,real,ln.index)*max(1,length-700))
                    for ln in self.lanes]
        self.real=real
        return self.collect(length,None)

    def tick_values(self,lanes):
        self.index+=1
        return super().tick_values(lanes)

    def draw(self,role,offers):
        events=super().draw(role,offers)
        if role!=0:
            return events
        multi=[(offer,e) for offer,e in zip(offers,events,strict=True) if len(offer[0].pending)>1]
        if not multi:
            return events
        t,np=self.t,self.np
        logits=self.actors[0].head(t(np.stack([e.ah_after for _,e in multi])))
        probs=logits.softmax(-1).cpu().numpy()
        for j,((ln,pid,_),e) in enumerate(multi):
            # 特征只用焦点乘客的决策前观测与已处理历史，不取未揭示的需求。
            e.cf_features=np.concatenate((e.obs,e.ah_after,
                         np.asarray([e.q,len(ln.pending)],np.float32)))
            e.cf_probs=probs[j].copy()
            # 参照在采样动作之外用独立键预定义；不能看到实际动作或分支结局后再挑。
            e.cf_ref=min(2,int(3*batch.uniform("cf_reference",self.seed,ln.index,ln.episode,ln.now,pid)))
            k=min(len(ln.pending)-1,int(batch.uniform("cf_atom",self.seed,self.real,ln.index,ln.episode,ln.now)*len(ln.pending)))
            if ln.index not in self.snapshots and self.index>=self.gates[ln.index] and e.q==k:
                cloned=copy.deepcopy(ln)
                check_copy(ln,cloned)
                self.snapshots[ln.index]=Start(cloned,self.ah[ln.index].cpu().numpy().copy(),pid,k,e)
        return events


def q_head(torch,dim):
    nn=torch.nn
    net=nn.Sequential(nn.Linear(dim+3,64),nn.LeakyReLU(),
                      nn.Linear(64,64),nn.LeakyReLU(),nn.Linear(64,3))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


def feature(np,event,method):
    ref=np.zeros(3,np.float32)
    if method=="B4_DIFFERENCE":
        ref[event.cf_ref]=1.
    return np.concatenate((event.cf_features,ref))


def apply_credit(torch,np,net,t,ticks,method,scale):
    data=[e for lane in ticks for m in lane if len(m.passengers)>1 for e in m.passengers]
    if not data:
        raise RuntimeError("没有用于反事实信用的多乘客分钟")
    diff=[]
    for i in range(0,len(data),512):
        evs=data[i:i+512]
        vals=net(t(np.stack([feature(np,e,method) for e in evs]))).cpu().numpy()*scale
        for e,q in zip(evs,vals,strict=True):
            baseline=float(np.dot(e.cf_probs,q)) if method=="B3_COMA" else float(q[e.cf_ref])
            advantage=float(q[e.action]-baseline)
            diff.append(abs(advantage-e.adv))
            e.adv=advantage
    return dict(cf_changed_atom_count=len(data),cf_mean_abs_delta_vs_shared=float(np.mean(diff)))


def labels(np,starts,results,method,center,scale):
    recorded, samples = [], []
    degenerate=0
    for i,start in enumerate(starts):
        branch=results[i]
        event=start.event
        if method=="B3_COMA":
            # 三个合法站都有真实分支结果，训练绝对 Q；策略加权中心化只在分配信用时使用。
            y=[(branch[a]["gain"]-center)/scale for a in range(3)]
            mask=[True]*3
        else:
            # 差分基于同一行为快照的事前固定参照动作，剩余两角色都重新响应。
            delta=float(branch[event.action]["gain"]-branch[event.cf_ref]["gain"])
            y=[0.]*3
            y[event.action]=delta/scale
            mask=[a in (event.action,event.cf_ref) for a in range(3)]
            degenerate+=abs(delta)<1e-8
        samples.append((feature(np,event,method),y,mask,event.cf_ref))
        recorded.append(dict(lane=start.lane.index,tick=start.lane.now,minute=start.lane.clock,
            episode=start.lane.episode,q=start.q,pid=start.pid,action=event.action,reference=event.cf_ref,
            policy_prob=event.cf_probs.tolist(),branches=branch,target=y,label_mask=mask))
    return samples,recorded,degenerate


def learn_q(torch,np,t,net,opt,samples,method,args,seed,step):
    rng=np.random.default_rng(np.random.SeedSequence([seed,step,0x4346]))
    items=list(samples)
    x0=np.stack([r[0] for r in items])
    y0=np.asarray([r[1] for r in items],np.float32)
    mask0=np.asarray([r[2] for r in items],bool)
    ref0=np.asarray([r[3] for r in items],np.int64)
    meters=batch.Metrics()
    for ids in batch.batches(np,len(items),args["critic_epoch"], min(4,len(items)),rng):
        raw=net(t(x0[ids]))
        mask=t(mask0[ids])
        y=t(y0[ids])[mask]
        if method=="B4_DIFFERENCE":
            raw=raw-raw.gather(1,t(ref0[ids])[:,None])
        q=raw[mask]
        loss=torch.nn.functional.smooth_l1_loss(q,y)
        batch.old.optimize(torch,opt,net,args["value_loss_coef"]*loss,args)
        with torch.no_grad():
            meters.add(cf_loss=loss.item(),cf_explained_variance=batch.ev(torch,q,y).item())
    return meters.report()


def train(cli,cell,job,cfg,rows,env_type,source,args,np,torch):
    import cloudpickle
    from harl.algorithms.critics.v_critic import VCritic

    method,seed=job["method"],job["seed"]
    device=torch.device(cli.device)
    base.set_seed(seed,np,torch)
    actors,critic,vn,norm,dim,_,loaded=batch.load_source(
        cli,source["folder"],torch,device,env_type,cfg,rows,source)
    actors[1].requires_grad_(False)
    dim_cf=dim+64+2
    q=q_head(torch,dim_cf).to(device)
    modules={"passenger":actors[0],"aircraft":actors[1],"critic":critic,"cf_q":q}
    lr={"passenger":args["lr"],"critic":args["critic_lr"],"cf_q":args["critic_lr"]}
    opts={k:torch.optim.Adam(modules[k].parameters(),lr=v,eps=args["opti_eps"],
                            weight_decay=args["weight_decay"]) for k,v in lr.items()}
    source_center=float(vn.denormalize(np.zeros((1,1),np.float32))[0,0])
    source_scale=max(1.,float(vn.denormalize(np.ones((1,1),np.float32))[0,0]-source_center))
    if not math.isfinite(source_center+source_scale):
        raise RuntimeError("源模型价值尺度无效")
    air_hash=loaded["aircraft"]["sha"]
    batch.contract(cell/"LOAD_AUDIT.json",dict(source=source["pointer"],strict_layers=loaded,
        frozen_aircraft=air_hash,obsnorm_frozen=True,
        update={k:[n for n,_ in modules[k].named_parameters()] for k in lr},
        init_q_sha=batch.tensor_hash(torch,q),target_mean=source_center,target_scale=source_scale))
    vc=VCritic(args,norm.observation_space,device=device)
    coll=Collector(env_type,cfg,rows,actors,critic,norm,torch,np,device,method,seed)
    Actor,_,_=old.model_types(torch)
    branch_actors=[Actor(dim,True).eval() for _ in range(2)]
    frozen_norm=batch.FrozenNormFile(source["folder"]/source["pointer"]["norm"])
    samples=deque(maxlen=LABEL_WINDOW*base.LANES)
    real,counts=0,Counter()
    base.set_seed(seed,np,torch)
    resume=cell/"RESUME.json"
    if resume.is_file():
        meta=base.read(resume)
        real=int(meta["real_steps"])
        if not 0<real<=batch.STEPS or real%base.ROLLOUT:
            raise ValueError("反事实续训必须从完整真实 rollout 边界恢复")
        mp,_=old.checked_pointer(cell,meta,real)
        if meta["norm_sha"]!=source["pointer"]["norm_sha"]:
            raise ValueError("续训 ObsNorm 与原冻结尺度不一致")
        state=torch.load(mp,map_location="cpu",weights_only=False)
        if (state["job"],state["real_steps"])!=(job,real):
            raise ValueError("反事实续训格身份错误")
        for k,net in modules.items():
            net.load_state_dict(state["models"][k],strict=True)
        for k,opt in opts.items():
            opt.load_state_dict(state["optimizers"][k])
        for k in batch.VALUE_KEYS:
            getattr(vn,k).copy_(state["values"][k])
        coll.lanes=cloudpickle.loads(state["lanes"])
        coll.ah,coll.ch=state["ah"].to(device),state["ch"].to(device)
        samples.extend(state["samples"])
        counts.update(state["counts"])
        base.rng_restore(state["rng"],device,np,torch)
        print(f"{cell.name}：从 {real:,} 步续跑",flush=True)
    start_time,start=time.monotonic(),real
    try:
        while real<batch.STEPS:
            batch.guard(cli)
            for k,opt in opts.items():
                opt.param_groups[0]["lr"]=lr[k]*(1-real/batch.STEPS)
            with torch.no_grad():
                ticks,bootstrap=coll.collect_cf(base.LENGTH,real)
                batch.tick_advantages(np,ticks,bootstrap,vn,args)
                snapshots=[coll.snapshots[i] for i in sorted(coll.snapshots)]
                if len(snapshots)!=base.LANES:
                    raise RuntimeError(f"每道应至少有一个有效多乘客采样点：{len(snapshots)}/{base.LANES}")
                for cpu,current in zip(branch_actors,actors,strict=True):
                    cpu.load_state_dict(current.state_dict(),strict=True)
                branch=branch_rollout(torch,np,branch_actors,norm,snapshots,method,seed,args["gamma"])
                new,trace,degenerate=labels(np,snapshots,branch,method,source_center,source_scale)
                use_cf=(real//base.ROLLOUT>=WARMUP_ROLLOUTS and
                        (method=="B3_COMA" or counts["informative_cf_states"]>=base.LANES))
                # PPO 只读上轮已拟合的头。未采样原子动作不接受任何伪分支标签。
                cf=apply_credit(torch,np,q,coll.t,ticks,method,source_scale) if use_cf else {}
            samples.extend(new)
            stats=learn_q(torch,np,coll.t,q,opts["cf_q"],samples,method,args,seed,real)
            stats.update(cf)
            stats.update(batch.credit_update(torch,np,coll.t,ticks,actors[0],critic,opts["passenger"],
                                             opts["critic"],vn,vc,args,real==start))
            flat=[m for lane in ticks for m in lane]
            parts=[p for ar in branch.values() for p in ar.values()]
            physical=len(flat)
            if physical!=base.ROLLOUT:
                raise AssertionError("分支不能混入真实步计数")
            current=dict(physical_minutes=physical,multi_passenger_minutes=sum(len(m.passengers)>1 for m in flat),
                         three_plus_passenger_minutes=sum(len(m.passengers)>2 for m in flat),
                         sampled_cf_states=len(snapshots),informative_cf_states=len(snapshots)-degenerate,
                         cf_branch_count=len(parts),cf_branch_minutes=sum(p["steps"] for p in parts),
                         cf_completed=sum(p["terminal"] for p in parts),cf_cutoff=sum(p["truncated"] for p in parts),
                         cf_activated_rollouts=int(use_cf))
            counts.update(current)
            real+=physical
            is_mark=real in batch.MARKS
            if is_mark or real==start+base.ROLLOUT:
                if batch.tensor_hash(torch,actors[1])!=air_hash:
                    raise AssertionError("冻结的 Aircraft Actor 被改动")
            if (cli.retain_rollouts=="all" or cli.retain_rollouts=="marks" and is_mark
                    or cli.retain_rollouts=="final" and real==batch.STEPS):
                batch.retain_data(cell,real,ticks,bootstrap)
            base.save(cell/"cf_labels"/f"{real:09d}.json",dict(step=real,method=method,rows=trace))
            state=dict(job=job,real_steps=real,models={k:n.state_dict() for k,n in modules.items()},
                       optimizers={k:o.state_dict() for k,o in opts.items()},values=batch.stats_state(vn),
                       lanes=cloudpickle.dumps(coll.lanes),ah=coll.ah.detach().cpu(),ch=coll.ch.detach().cpu(),
                       samples=list(samples),counts=dict(counts),rng=base.rng_snapshot(np,torch))
            base.save_generation(cell,real,batch.STEPS,batch.MARKS,frozen_norm,lambda p:torch.save(state,p),".pt")
            row=dict(real_steps=real,real_fps=(real-start)/max(1.,time.monotonic()-start_time),
                     **current,**stats)
            base.save(cell/"updates"/f"{real:09d}.json",row)
            base.save(cell/"LIVE.json",dict(row,cumulative=dict(counts)))
        if not counts["cf_activated_rollouts"]:
            raise RuntimeError("反事实信用一次都没有进入 PPO；拒绝标记为已完成")
        base.save(cell/"TRAINED.json",dict(real_steps=real,counts=dict(counts),frozen_aircraft=air_hash,
                  final_sha={k:batch.tensor_hash(torch,m) for k,m in modules.items()}))
    finally:
        for ln in coll.lanes:
            ln.view.close()


def verified_parent(root, method):
    name=METHODS[method]
    p=root/name/"PROTOCOL.json"
    contract=base.read(p)
    sha=base.digest(p)
    if (contract["schema"]!="corridor_acac_minute_batch_v1" or contract["line"]!=name
            or not old.same_source(Path(batch.__file__),contract["code_sha"])):
        raise ValueError("不要覆盖正在运行的旧脚本；补充入口所用批次源码与上游不符")
    done=root/name/"RUN_COMPLETE.json"
    if done.is_file():
        c=base.read(done)
        if ((c["line"],c["cells"],c["real_steps"],c["failures"])!=(name,*PARENT[name],0)
                or c["protocol_sha"]!=sha or c["rates"]!=list(batch.RATES)):
            raise ValueError("上游的完成标记与原定合同不匹配")
    else:
        # 若原队列确实跑到末尾，允许个别独立格失败，保留失败清单；中途被杀不能冒充完成。
        summary=base.read(root/name/"SUMMARY.json")
        methods=batch.ORDER[1:] if name=="order" else batch.CREDIT
        expected={(m,s) for m in methods for s in batch.SEEDS}
        ok={(m,int(s)) for m,rows in summary["methods"].items() for s,d in rows.items()
            if d.get("job",{}).get("protocol_sha")==sha}
        failed={(r["method"],r["seed"]) for r in summary["failures"]}
        if not failed or (ok|failed)!=expected or ok&failed:
            raise ValueError("上游未遍历完整矩阵；请续跑上游，不自动启动后置方法")
        print(f"上游矩阵已遍历完整；保留 {len(failed)} 个失败格，继续独立追加实验",flush=True)
    return contract,sha


def run(cli):
    import gymnasium as gym
    import numpy as np
    import stable_baselines3 as sb3
    import torch

    torch.set_num_threads(1)
    if cli.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 不可用，不自动换设备")
    parent,parent_sha=verified_parent(cli.batch_output,cli.method)
    versions=dict(torch=str(torch.__version__),numpy=np.__version__,sb3=sb3.__version__,gymnasium=gym.__version__)
    cfg,original,bank,base_protocol=old.bank_for(cli,versions)
    args,_,vendor=base.harl_vendor(cli.harl_dir,np,torch)
    pt=cli.acac_source/"PROTOCOL.json"
    src=base.read(pt)
    if (src["schema"]!="acac_e2_legal_decision_event_mappo_v1" or src["base_protocol"]!=base_protocol
            or src["harl_args"]!=old.canonical(args) or src["harl_files"]!=vendor
            or not old.same_source(Path(old.__file__),src["source_sha"])):
        raise ValueError("原 ACAC 权重的环境或 PPO 合同不同")
    if (parent["base_protocol_sha"]!=base.digest(cli.base_output/"PROTOCOL.json")
            or parent["source_protocol_sha"]!=base.digest(pt) or parent["ppo"]!=old.canonical(args)
            or parent["train_source"]!=old.demand_sha(original)
            or tuple(parent["marks"])!=batch.MARKS or tuple(parent["seeds"])!=batch.SEEDS
            or parent["steps"]!=batch.STEPS):
        raise ValueError("与上游对照的源码、训练步数或数据口径不一致")
    cli.burst_pairs,cli.burst_triples=parent["burst_pairs"],parent["burst_triples"]
    rows=batch.burst_trace(original,cli.burst_pairs,cli.burst_triples)
    if parent["train_burst"]!=old.demand_sha(rows):
        raise ValueError("反事实线没有复用上游相同的 T1 训练需求")
    sources=batch.source_cells(cli,pt)
    signature=batch.contract(cli.output/"PROTOCOL.json",dict(schema="acac_batch_cf_credit_e2_v1",
        method=cli.method,source_sha=base.digest(Path(__file__)),parent_sha=parent_sha,
        parent_line=METHODS[cli.method],base_protocol_sha=parent["base_protocol_sha"],
        weights={s:entry["pointer"] for s,entry in sources.items()},
        train_burst=parent["train_burst"],burst_pairs=cli.burst_pairs,burst_triples=cli.burst_triples,
        eval_contract=parent["eval_contract"],ppo=args,device=cli.device,
        seeds=batch.SEEDS,marks=batch.MARKS,rates=batch.RATES,steps=batch.STEPS,
        observe="只用决策前可见观测、已执行历史与当前已揭示批次长度",
        branch="真实环境深拷贝；固定前缀；固定随机键；两角色后续响应；不截短终止",
        target="B3 三动作 G 与策略加权基线；B4 实际动作 G 减事前独立随机键所选合法参照动作 G",
        training="先有真实标签再使用上轮头；仅多乘客分钟用反事实信用；Aircraft/ObsNorm 冻结",
        retain_rollouts=cli.retain_rollouts,warmup_rollouts=WARMUP_ROLLOUTS,label_window=LABEL_WINDOW,
        cf_samples_per_rollout=base.LANES))
    batch.contract(cli.output/"DEMAND_BANK.json",dict(train=[asdict(r) for r in rows],
             **{name:[asdict(r) for r in ys] for name,_,ys in bank}))
    env_type,_=old.observed_env(gym,np,cfg)
    batch.alignment(cli,cfg,bank,env_type,sources,np,torch)
    # 已有标准五档及突发 A0 都直接复用；只有 B0 完整时才报告配对差。
    upstream=cli.batch_output/METHODS[cli.method]
    baseline=base.read(upstream/"REUSED_BASELINES.json")
    by_seed={}
    errors=[]

    def report():
        parent_report=cli.batch_output/"credit"/"SUMMARY.json"
        current=base.read(parent_report).get("methods",{}).get("B0_SHARED",{}) if parent_report.is_file() else {}
        matched={}
        for s in map(str,batch.SEEDS):
            b0=current.get(s)
            if b0 is not None and (b0.get("real_steps")==batch.STEPS and len(b0.get("rates",[]))==len(batch.RATES)):
                matched[s]=b0
        factory=argparse.Namespace(line="credit",output=cli.output)
        result={cli.method:by_seed}
        if matched:
            result={"B0_SHARED":matched,**result}
        batch.write_report(factory,baseline,result,errors)

    try:
        for seed in batch.SEEDS:
            cell=cli.output/"cells"/f"{cli.method}__s{seed}"
            job=dict(protocol_sha=signature,seed=seed,method=cli.method,steps=batch.STEPS)
            status=cell/"STATUS.json"
            try:
                batch.guard(cli)
                batch.contract(cell/"JOB.json",job)
                if status.is_file():
                    row=base.read(status)
                    if row["job"]!=job or row["evaluation_sha"]!=base.digest(cell/"EVALUATION.json"):
                        raise ValueError("完成格与评估记录不一致")
                else:
                    print(f"{cell.name}：开始或继续",flush=True)
                    train(cli,cell,job,cfg,rows,env_type,sources[seed],args,np,torch)
                    # B2 是批次入口里已有的“原始乘客顺序、信用头不参与推理”评估路径。
                    row=batch.eval_cell(cli,cell,job,"B2_PREFIX_Q",cfg,bank,env_type,sources[seed],np,torch)
                    row["counterfactual"]=base.read(cell/"TRAINED.json")["counts"]
                    base.save(status,row)
                by_seed[str(seed)]=row
                print(f"{cell.name}：完成",flush=True)
            except (KeyboardInterrupt,InterruptedError):
                raise
            except Exception:
                item=dict(method=cli.method,seed=seed,error=traceback.format_exc())
                errors.append(item)
                base.save(cell/"ERROR.json",item)
                print(item["error"],flush=True)
            report()
    finally:
        report()
    if not errors and len(by_seed)==len(batch.SEEDS):
        base.save(cli.output/"RUN_COMPLETE.json",dict(method=cli.method,protocol_sha=signature,failures=0,
            cells=len(by_seed),real_steps=sum(r["real_steps"] for r in by_seed.values()),
            branch_minutes=sum(r["counterfactual"]["cf_branch_minutes"] for r in by_seed.values())))
    return int(bool(errors))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method",choices=tuple(METHODS),required=True)
    for name in ("config","train-trace","harl-dir","base-output","acac-source","batch-output","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--retain-rollouts",choices=("final","marks","all"),default="final")
    p.add_argument("--allow-trusted-checkpoints",action="store_true")
    cli=p.parse_args(argv)
    if not cli.allow_trusted_checkpoints:
        p.error("仅允许本账号产生的可信权重，请显式指定 --allow-trusted-checkpoints")
    for k in ("config","train_trace","harl_dir","base_output","acac_source","batch_output","output"):
        setattr(cli,k,getattr(cli,k).resolve())
    cli.output=cli.output/cli.method
    for r in (cli.base_output,cli.acac_source,cli.batch_output):
        if cli.output==r or r in cli.output.parents or cli.output in r.parents:
            p.error("补充实验必须使用独立输出目录")
    with base.lock_run(cli.output):
        try:
            verified_parent(cli.batch_output,cli.method)
            return run(cli)
        except (KeyboardInterrupt,InterruptedError) as exc:
            print(f"暂停：{exc}",flush=True)
            return 130
        except Exception:
            base.save(cli.output/"RUN_ERROR.json",dict(error=traceback.format_exc()))
            raise


if __name__=="__main__":
    raise SystemExit(main())
