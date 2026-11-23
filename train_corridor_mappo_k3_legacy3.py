"""旧 K3 Lookahead 三臂直接移植至走廊 E2 + MAPPO。"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import shutil
import time
import traceback
import types
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import train_eval_corridor_uagmc_happo_mappo as base
import train_eval_corridor_lookahead_l0l5 as legacy
from corridor_env import Request, validate_requests
from train_eval_corridor_lookahead_l0l5 import (
    LANES, LENGTH, ROLLOUT, digest, read, save, verified, plain, set_seed, rng_snapshot, rng_restore
)

STEPS = 1_996_800
SEEDS = (2101, 2102, 2103, 2104)
RATES = (1.0, 1.1, 1.2, 1.3, 1.4)
POLICY_SEEDS = (123, 124, 125)
MARKS = [204800, 501760, 1003520, 1597440, 1699840, 1802240, 1904640, STEPS]
VALUE_KEYS = ("running_mean", "running_mean_sq", "debiasing_term")
SHADOW_H, SHADOW_N, WARM, RING, SAMPLES = 20, 128, 2, 8, 2048
ARMS = {"L1_FULL": (1, True, True), "L4_DISCARD_SHADOW": (1, False, True),
        "L5_NO_VTRACE": (1, True, False)}
K3_SOURCE = "uam_lookahead_happo_k3_24run_48m_v1/ENTRYPOINT.py"
K3_MAPPO_SOURCE = "train_uam_lookahead_mappo_k3_dual_96m_ONEFILE.py"


# 从最初 K3 HAPPO 归档逐字搬运的机制代码。
def vtrace(reward,value,bootstrap,continues,discount,logp,enabled=True):
    """价值校正用联合比率；返回不含角色比率的 Q−V，供独立 PPO 权重使用。"""
    import numpy as np
    if not all(np.isfinite(x).all() for x in (reward,value,bootstrap,logp)):
        raise ValueError('V-trace 输入包含非有限数值')
    ratio=np.exp(np.clip(logp,-40,40))
    c=np.minimum(1.,ratio) if enabled else np.ones_like(ratio)
    corrected=np.empty_like(value)
    delta=reward+discount*bootstrap-value
    carry=np.zeros_like(value[0])
    for i in range(len(value)-1,-1,-1):
        carry=c[i]*delta[i]+discount[i]*continues[i]*c[i]*carry
        corrected[i]=value[i]+carry
    # 切开 episode 后的 reset 样本；最后一个 rollout 步使用真实 next value。
    nxt=bootstrap.copy()
    nxt[:-1]=np.where(continues[:-1],corrected[1:],bootstrap[:-1])
    advantage=reward+discount*nxt-value
    if not np.isfinite(corrected).all() or not np.isfinite(advantage).all():
        raise ValueError('V-trace 目标非有限')
    return corrected,advantage,ratio


def wm_model(dim,torch):
    from torch import nn
    class Dynamics(nn.Module):
        """共享当前状态代理；单独预测乘客选站后观测与整分钟转移、奖励。"""
        def __init__(self):
            super().__init__()
            def net(ni,no):
                return nn.Sequential(nn.Linear(ni,192),nn.SiLU(),nn.Linear(192,192),
                                     nn.SiLU(),nn.Linear(192,no))
            core=dim-8
            self.post=net(dim+3,2*core+3)
            self.next=net(dim+6,2*(core+1))

        def post_parameters(self,x,p):
            return self.post(torch.cat((x,nn.functional.one_hot(p,3).float()),-1))

        def next_parameters(self,x,p,a):
            return self.next(torch.cat((x,nn.functional.one_hot(p,3).float(),
                                        nn.functional.one_hot(a,3).float()),-1))
    return Dynamics()


def gaussian(params,torch):
    mean,log_std=params.chunk(2,dim=-1)
    return mean,log_std.clamp(-5,1)


def fit_world(model,opt,ring,norm,torch,np,rng,device):
    data={k:np.concatenate([block[k] for block in ring])
          for k in ('x','post','next','p','executed_p','a','r','done','age')}
    # 不把自动 reset 引入学习目标；留出真实样本由固定下标切出。
    ids=np.arange(len(data['r']))
    train=ids[ids%8!=0]; valid=ids[ids%8==0]
    if not len(train) or not len(valid):
        raise ValueError('真实回放池不足')
    pack=norm.observation_space.shape[0]-8
    reward_unit=100.
    def tx(x,dtype=torch.float32):
        return torch.as_tensor(x,dtype=dtype,device=device)
    def err(index):
        x=tx(norm.normalize_obs(data['x'][index]))
        before=tx(norm.normalize_obs(data['post'][index])[:,:pack])
        after=tx(norm.normalize_obs(data['next'][index])[:,:pack])
        p,a=[tx(data[k][index],torch.long) for k in ('p','a')]
        reward=tx(data['r'][index]/reward_unit).reshape(-1)
        mask=tx(~data['done'][index]).reshape(-1)
        post_parameters=model.post_parameters(x,p)
        u,lu=gaussian(post_parameters[:,:-3],torch)
        logits=post_parameters[:,-3:]
        v,lv=gaussian(model.next_parameters(x,p,a),torch)
        def nll(y,z,l):
            return .5*((y-z)*torch.exp(-l)).square()+l
        e_post=nll(before,u,lu).mean()
        e_next=(nll(after,v[:,:pack],lv[:,:pack]).mean(-1)*mask).sum()/mask.sum().clamp_min(1.)
        e_rew=nll(reward,v[:,-1],lv[:,-1]).mean()
        applied=tx(data['executed_p'][index],torch.long)
        e_applied=torch.nn.functional.cross_entropy(logits,applied)
        return e_post+e_next+e_rew+e_applied,(before,u,after,v,mask,reward,x,logits,applied)
    model.train()
    for _ in range(48):
        ix=rng.choice(train,256,replace=len(train)<256)
        loss,_=err(ix)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        opt.step()
    model.eval()
    with torch.no_grad():
        loss,(post,post_hat,nxt,y,mask,r,x,logits,applied)=err(valid[:1024])
        report=dict(nll=float(loss),post_rmse=float((post-post_hat).square().mean().sqrt()),
                    next_rmse=float((((nxt-y[:,:pack]).square().mean(-1)*mask).sum()
                                    /mask.sum().clamp_min(1)).sqrt()),
                    persistence_rmse=float((((nxt-x[:,:pack]).square().mean(-1)*mask).sum()
                                    /mask.sum().clamp_min(1)).sqrt()),
                    executed_passenger_accuracy=float((logits.argmax(-1)==applied).float().mean()),
                    reward_mae=float((r-y[:,-1]).abs().mean()*reward_unit),n=len(valid[:1024]))
    return data,report


def shadow_team(real,vc,vn):
    # deepcopy 整个训练对象以保持参数与各自优化器引用；任何虚拟更新都不写回。
    return copy.deepcopy((real,vc,vn))


def value_probe(vn):
    return {k:getattr(vn,k).detach().cpu() for k in ('running_mean','running_mean_sq','debiasing_term')}


def runner_for(args,actors,critic,an,cb,vn):
    return types.SimpleNamespace(algo_args={'train':args},actor=actors,critic=critic,actor_buffer=an,
                                 critic_buffer=cb,value_normalizer=vn,state_type='EP',
                                 fixed_order=False,num_agents=2,action_aggregation='prod')


def policy_logits(actor,obs,zeros,masks,actions):
    """通过作者接口拿分布；不重写 actor 网络。"""
    _,ent,dist=actor.evaluate_actions(obs,zeros,actions,masks,None,None)
    if not hasattr(dist,'logits'):
        raise ValueError('本轮必须是 K3 单离散 Categorical')
    return dist.logits,ent


def probe_logits(actors,bs,torch,ix):
    out=[]
    with torch.no_grad():
        for a,b in zip(actors,bs):
            n=b.actions.shape[0]*b.actions.shape[1]
            z=b.rnn_states[:-1].reshape(n,*b.rnn_states.shape[2:])
            logits,_=policy_logits(a,b.obs[:-1].reshape(n,-1)[ix],z[ix],
                                   b.masks[:-1].reshape(n,1)[ix],b.actions.reshape(n,1)[ix])
            out.append(logits.detach().cpu().numpy())
    return out


# 只换底座：直接搬运旧 K3 MAPPO 实现；虚拟轨迹其余部分与 HAPPO 原版一致。
def offpolicy_update(actors,critic,cb,vn,buffs,adv,anchor,np,torch):
    """保留 MAPPO 的独立裁剪更新；影子采样只添加本角色行为校正。"""
    advantage=(adv-adv.mean())/(adv.std()+1e-5)
    flat=advantage.reshape(-1,1)
    report=[]
    for i in range(len(actors)):
        actor,b=actors[i],buffs[i]
        n=flat.shape[0]
        obs=b.obs[:-1].reshape(n,-1)
        zs=b.rnn_states[:-1].reshape(n,*b.rnn_states.shape[2:])
        act=b.actions.reshape(n,1).astype(np.int64)
        masks=b.masks[:-1].reshape(n,1)
        mu=b.action_log_probs.reshape(n,1)
        pi0=anchor[i].reshape(n,1)
        sums=[]
        # 本角色使用 π_old/μ；联合比率仅用于 critic 的 V-trace，不重复乘入 actor。
        w=np.exp(np.clip(pi0-mu,-10,10)).astype(np.float32)
        width=n//actor.actor_num_mini_batch
        if n%actor.actor_num_mini_batch:
            raise ValueError('正式 minibatch 不整除')
        for _ in range(actor.ppo_epoch):
            for ix in torch.randperm(n).numpy().reshape(-1,width):
                raw,ent,_=actor.evaluate_actions(obs[ix],zs[ix],act[ix],masks[ix],None,None)
                target=(torch.as_tensor(flat[ix],**actor.tpdv))
                ratio=(raw-torch.as_tensor(pi0[ix],**actor.tpdv)).clamp(-20,20).exp()
                weight=torch.as_tensor(w[ix],**actor.tpdv)
                term=torch.minimum(ratio*target,ratio.clamp(1-actor.clip_param,1+actor.clip_param)*target)
                loss=-(weight*term).mean()
                actor.actor_optimizer.zero_grad(set_to_none=True)
                (loss-actor.entropy_coef*ent).backward()
                grad=torch.nn.utils.clip_grad_norm_(actor.actor.parameters(),actor.max_grad_norm,
                                                    error_if_nonfinite=True)
                actor.actor_optimizer.step()
                sums.append([loss.item(),ent.item(),grad.item()])
        report.append(dict(role=i,losses=np.mean(sums,axis=0),mean_weight=float(w.mean()),
                           max_weight=float(w.max())))
    cm=critic.train(cb,vn)
    return report,cm


def mappo_update(actors,critic,bs,cb,vn):
    # 和 88M MAPPO_MATCHED 一样：作者 MAPPO actor 各自更新，共用中心化价值估计。
    advantage=cb.returns[:-1]-vn.denormalize(cb.value_preds[:-1])
    if any(b.factor is not None for b in bs):
        raise RuntimeError('MAPPO 不应有 HAPPO 前序概率因子')
    return [a.train(b,advantage.copy(),'EP') for a,b in zip(actors,bs)],critic.train(cb,vn)


def imagine(team,args,space,action_space,model,norm,dataset,depth,author,Buffer,CB,np,torch,rng,device):
    actors,critic,vn=team
    shape=dict(args,episode_length=SHADOW_H,n_rollout_threads=SHADOW_N)
    for a in actors:
        a.actor_num_mini_batch=10
        a.entropy_coef=.001
    critic.critic_num_mini_batch=10
    bs=[Buffer(shape,space,action_space) for _ in actors]; cb=CB(shape,space)
    zero=np.zeros((SHADOW_N,args['recurrent_n'],args['hidden_sizes'][-1]),np.float32)
    ones=np.ones((SHADOW_N,1),np.float32)
    d=space.shape[0]; core=d-8
    starts=np.arange(len(dataset['r']))
    starts=starts[(starts%8!=0)&(dataset['age']<480)&(~dataset['done'])]
    if not len(starts):
        raise RuntimeError('没有合法的虚拟起点')
    def array(t): return t.detach().cpu().numpy()
    def sample(param):
        mean,log_std=gaussian(param,torch)
        return mean+torch.randn_like(mean)*log_std.exp()
    raw_tail=np.zeros((4,d),np.float32)
    raw_tail[1:,-8]=1.
    raw_tail[1:,-7:-4]=np.eye(3)
    tails=norm.normalize_obs(raw_tail)[:,-8:]
    reports=[]
    for _ in range(depth):
        obs=norm.normalize_obs(dataset['x'][rng.choice(starts,SHADOW_N,replace=True)])
        for a in actors: a.prep_rollout()
        critic.prep_rollout()
        reward_avg=[]
        with torch.no_grad():
            for k in range(SHADOW_H):
                values,_=critic.get_values(obs,zero,ones)
                p,lp,_=actors[0].get_actions(obs,zero,ones)
                x=torch.as_tensor(obs,device=device,dtype=torch.float32); pi=p.long().flatten()
                pp=model.post_parameters(x,pi)
                y=sample(pp[:,:-3]).clamp(-10.,10.)
                # 无待选乘客时真实环境把请求动作映射为 0；在影子内用学习到的执行类别。
                executed=torch.distributions.Categorical(logits=pp[:,-3:]).sample()
                mid=np.concatenate((array(y),tails[1+array(executed)]),axis=-1)
                a,la,_=actors[1].get_actions(mid,zero,ones)
                out=sample(model.next_parameters(x,pi,a.long().flatten()))
                nxt=np.concatenate((array(out[:,:core].clamp(-10.,10.)),np.repeat(tails[:1],SHADOW_N,axis=0)),axis=-1)
                r=array((out[:,-1:]*100.).clamp(-300.,0.))
                reward_avg.append(float(r.mean()))
                bs[0].obs[k]=obs; bs[1].obs[k]=mid; cb.share_obs[k]=obs
                for b,act,logp in zip(bs,(p,a),(lp,la)):
                    b.insert(nxt,zero,array(act),array(logp),ones,ones)
                cb.insert(nxt,zero,array(values),r,ones,ones)
                obs=nxt
            v,_=critic.get_values(obs,zero,ones)
            cb.compute_returns(array(v),vn)
        for a in actors: a.prep_training()
        critic.prep_training()
        am,cm=mappo_update(actors,critic,bs,cb,vn)
        reports.append(dict(mean_imagined_reward=float(np.mean(reward_avg)),actor=am,critic=cm))
    return reports


def independent_update(runner):
    return mappo_update(runner.actor, runner.critic, runner.actor_buffer,
                        runner.critic_buffer, runner.value_normalizer)


def close_training_vector(vector):
    # 只在 worker 已退出、正常关闭失败时收尾；不要覆盖真实异常。
    try:
        vector.close()
    except (EOFError, BrokenPipeError, ConnectionResetError):
        vec = getattr(vector, "venv", vector)
        for remote in vec.remotes:
            remote.close()
        for worker in vec.processes:
            worker.join(timeout=1)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)


# E2 的真实轨迹接口与存档管理；真实/影子更新走上面的历史机制。
def atomic_write(path, writer):
    tmp = path.with_suffix(path.suffix + ".pending")
    writer(tmp)
    tmp.replace(path)


def checkpoint(folder, real, marks, previous, state, extra, norm, torch):
    # 评估点只存正式团队和归一化；大回放池仅存于训练恢复槽。
    tag = "rolling_b" if (previous or {}).get("tag") == "rolling_a" else "rolling_a"
    stem = f"step_{real:09d}" if real in marks else tag
    model, npath = folder/(stem+".pt"), folder/(stem+".norm.pkl")
    atomic_write(model, lambda p: torch.save(state, p))
    atomic_write(npath, lambda p: norm.save(str(p)))
    meta = dict(real_steps=real, model=model.name, norm=npath.name, model_sha=digest(model),
                norm_sha=digest(npath), tag=tag, imagined_steps=state["imagined_steps"],
                model_updates=state["model_updates"])
    if real in marks:
        save(folder/"checkpoints"/f"{real:09d}.json", meta)
    if extra is not None:
        path = folder/(tag+".world.pt")
        atomic_write(path, lambda p: torch.save(extra, p))
        meta.update(world=path.name, world_sha=digest(path))
    save(folder/"RESUME.json", meta)
    return meta

def train_cell(folder, cli, arm, seed, cfg, rows, marks, env_class, args, author, gym, np, torch):
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer as Buffer
    from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP as CB

    set_seed(seed, np, torch)
    ptr = read(folder/"RESUME.json") if (folder/"RESUME.json").is_file() else None
    real = ptr["real_steps"] if ptr else 0
    if real == cli.steps:
        return
    if real > cli.steps or real % ROLLOUT:
        raise ValueError("恢复点不在本合同的 rollout 边界")
    vec = SubprocVecEnv([lambda: env_class(cfg, rows, "MAPPO") for _ in range(LANES)], start_method="spawn")
    norm = None
    try:
        if ptr:
            model_file, norm_file = verified(folder, ptr)
            norm = VecNormalize.load(str(norm_file), vec)
        else:
            norm = VecNormalize(vec, norm_obs=True, norm_reward=False, gamma=.99)
        norm.training = True
        space, action_space = norm.observation_space, gym.spaces.Discrete(3)
        device = torch.device(cli.device)
        actors, critic, vn = build_pair(args, space, action_space, device)
        depth, use_shadow, use_vtrace = ARMS[arm]
        rng = np.random.default_rng(61811+seed)
        rs = rng_snapshot(np, torch)
        dynamics = wm_model(space.shape[0], torch).to(device) if depth else None
        opt = torch.optim.Adam(dynamics.parameters(), lr=3e-4) if depth else None
        rng_restore(rs, cli.device, np, torch)
        ring, updates, imag, fits = [], 0, 0, 0
        if ptr:
            state = torch.load(model_file, map_location="cpu", weights_only=False)
            if state["real_steps"] != real:
                raise ValueError("模型训练步数与指针不一致")
            for actor, w, o in zip(actors, state["actors"], state["optimizers"]):
                actor.actor.load_state_dict(w)
                actor.actor_optimizer.load_state_dict(o)
            critic.critic.load_state_dict(state["critic"])
            critic.critic_optimizer.load_state_dict(state["critic_optimizer"])
            vn.load_state_dict(state["value_norm"])
            for k, v in state["value_statistics"].items():
                getattr(vn, k).copy_(v)
            updates, imag, fits = state["updates"], state["imagined_steps"], state["model_updates"]
            if depth:
                path = folder/ptr["world"]
                if digest(path) != ptr["world_sha"]:
                    raise ValueError("世界模型与回放池存档校验失败")
                extra = torch.load(path, map_location="cpu", weights_only=False)
                dynamics.load_state_dict(extra["wm"])
                opt.load_state_dict(extra["opt"])
                ring = extra["ring"]
                rng.bit_generator.state = extra["rng"]
                del extra
            rng_restore(state["rng"], cli.device, np, torch)
            del state

        n, t = LANES, LENGTH
        bs = [Buffer(args, space, action_space) for _ in actors]
        cb = CB(args, space)
        runner = runner_for(args, actors, critic, bs, cb, vn)
        z = np.zeros((n, args["recurrent_n"], args["hidden_sizes"][-1]), np.float32)
        ones = np.ones((n, 1), np.float32)
        masks = ones.copy()
        obs, age = norm.reset(), np.zeros(n, np.int32)
        initial, since = real, time.monotonic()
        stop = cli.output/"STOP_REQUESTED"

        def array(x):
            return x.detach().cpu().numpy()

        def live(phase):
            save(folder/"LIVE.json", dict(phase=phase, real_steps=real, updates=updates,
                 imagined_steps=imag, model_updates=fits, real_fps=(real-initial)/max(1., time.monotonic()-since)))

        def capture():
            state = dict(real_steps=real, updates=updates, imagined_steps=imag, model_updates=fits,
                         actors=[a.actor.state_dict() for a in actors],
                         optimizers=[a.actor_optimizer.state_dict() for a in actors],
                         critic=critic.critic.state_dict(), critic_optimizer=critic.critic_optimizer.state_dict(),
                         value_norm=vn.state_dict(), value_statistics=value_probe(vn), rng=rng_snapshot(np, torch))
            extra = (dict(wm=dynamics.state_dict(), opt=opt.state_dict(), ring=ring,
                          rng=rng.bit_generator.state) if depth and real < cli.steps else None)
            result = checkpoint(folder, real, marks, ptr, state, extra, norm, torch)
            if real == cli.steps:
                # 正式最终存档不依赖大回放池；只清理本格两个已失效的恢复缓存。
                for name in ("rolling_a.world.pt", "rolling_b.world.pt"):
                    (folder/name).unlink(missing_ok=True)
            print(f"{folder.name}：真实 {real:,} / {cli.steps:,}，虚拟 {imag:,}，拟合更新 {fits:,}", flush=True)
            return result

        while real < cli.steps:
            if stop.exists() or shutil.disk_usage(cli.output).free < cli.min_free_gib * 2**30:
                if real > (ptr["real_steps"] if ptr else 0):
                    ptr = capture()
                raise InterruptedError("收到暂停或磁盘低于余量要求，已在更新边界暂停")
            for actor in actors:
                actor.prep_rollout()
                actor.lr_decay(real, cli.steps)
            critic.prep_rollout()
            critic.lr_decay(real, cli.steps)
            active = bool(depth and updates >= WARM)
            shadow, audit = None, {}
            if active:
                live("fit_and_imagine")
                rs = rng_snapshot(np, torch)
                dataset, report = fit_world(dynamics, opt, ring, norm, torch, np, rng, device)
                shadow = shadow_team(actors, critic, vn)
                audit.update(model=report, shadow=imagine(shadow, args, space, action_space, dynamics, norm,
                    dataset, depth, author, Buffer, CB, np, torch, rng, device))
                del dataset
                rng_restore(rs, cli.device, np, torch)
                imag += depth*SHADOW_H*SHADOW_N
                fits += 48
            behavior = shadow[0] if active and use_shadow else actors
            for actor in behavior:
                actor.prep_rollout()
            for b in bs:
                b.step, b.factor = 0, None
            cb.step = 0
            anchor = np.zeros((2,t,n,1), np.float32)
            ends = np.full((t,n,1), np.nan, np.float32)
            positions = {}
            if depth:
                for ix in sorted(rng.choice(t*n,SAMPLES,replace=False)):
                    positions.setdefault(ix//n, []).append(ix%n)
            block = {k: [] for k in ("x","post","next","p","executed_p","a","r","done","age")}
            for k in range(t):
                ix = positions.get(k)
                if ix is not None:
                    x = norm.get_original_obs()[ix].copy()
                with torch.no_grad():
                    vals, _ = critic.get_values(obs, z, masks)
                    p, lp, _ = behavior[0].get_actions(obs, z, masks)
                    anchor[0,k] = array(actors[0].evaluate_actions(obs,z,array(p),masks)[0] if use_shadow and active else lp)
                mid, reward, done, infos = norm.step(array(p).ravel())
                if np.any(reward) or np.any(done) or any(i["physical_increment"] for i in infos):
                    raise RuntimeError("选站准备不能推进物理分钟或产生奖励")
                if ix is not None:
                    post = norm.get_original_obs()[ix].copy()
                with torch.no_grad():
                    a, la, _ = behavior[1].get_actions(mid, z, masks)
                    anchor[1,k] = array(actors[1].evaluate_actions(mid,z,array(a),masks)[0] if use_shadow and active else la)
                nxt, r, done, infos = norm.step(array(a).ravel())
                if any(info["physical_increment"] != 1 for info in infos):
                    raise RuntimeError("真实步计数不符")
                nm = (~done).astype(np.float32)[:,None]
                bad = np.asarray([[0. if i.get("TimeLimit.truncated") else 1.] for i in infos], np.float32)
                if active:
                    for j,i in enumerate(infos):
                        if done[j] and not bad[j,0]:
                            with torch.no_grad():
                                end, _ = critic.get_values(np.asarray(i["terminal_observation"])[None],z[:1],ones[:1])
                            ends[k,j] = vn.denormalize(array(end))[0]
                bs[0].obs[k], bs[1].obs[k], cb.share_obs[k] = obs,mid,obs
                bs[0].masks[k], bs[1].masks[k] = masks,masks
                for b,act,log in zip(bs,(p,a),(lp,la)):
                    b.insert(nxt,z,array(act),array(log),nm,ones)
                cb.insert(nxt,z,array(vals),r[:,None],nm,bad)
                if ix is not None:
                    ps = np.argmax(post[:,-7:-4],axis=1)
                    for key,value in zip(block,(x,post,norm.get_original_obs()[ix],array(p).ravel()[ix],
                                                 ps,array(a).ravel()[ix],r[ix],done[ix],age[ix])):
                        block[key].append(value.copy())
                age = np.where(done,0,age+1)
                obs,masks = nxt,nm
                real += n
                if (k+1)%256 == 0:
                    live("collect_real")
            with torch.no_grad():
                v_next,_ = critic.get_values(obs,z,masks)
            if depth:
                ring = (ring+[{k:np.concatenate(v) for k,v in block.items()}])[-RING:]
            ix = np.linspace(0,n*t-1,256,dtype=int)
            if active:
                old_logits, shadow_logits = probe_logits(actors,bs,torch,ix), probe_logits(shadow[0],bs,torch,ix)
            for actor in actors:
                actor.prep_training()
            critic.prep_training()
            if active:
                cb.value_preds[-1] = array(v_next)
                vals = vn.denormalize(cb.value_preds)
                cont = cb.masks[1:].astype(bool)
                discount = args["gamma"]*(cont|(~cb.bad_masks[1:].astype(bool)))
                bootstrap = np.where(np.isfinite(ends),ends,vals[1:])
                gain = (anchor-np.stack([b.action_log_probs for b in bs])).sum(axis=0)
                cb.returns[:-1],adv,ratio = vtrace(cb.rewards,vals[:-1],bootstrap,cont,discount,gain,enabled=use_vtrace)
                am,cm = offpolicy_update(actors,critic,cb,vn,bs,adv,anchor,np,torch)
                new = probe_logits(actors,bs,torch,ix)
                def kl(p,q):
                    return float(np.mean(np.sum(np.exp(p)*(p-q),axis=-1)))
                audit.update(actor_order=[0, 1],kl_new_vs_old=[kl(a,b) for a,b in zip(new,old_logits)],
                             kl_new_vs_shadow=[kl(a,b) for a,b in zip(new,shadow_logits)],
                             joint_rho_mean=float(ratio.mean()),joint_rho_above_one=float((ratio>1).mean()),
                             joint_rho_relative_ess=float(ratio.mean()**2/np.square(ratio.astype(np.float64)).mean()))
            else:
                cb.compute_returns(array(v_next),vn)
                am,cm = author(runner)
            updates += 1
            audit.update(real_steps=real,updates=updates,imagined_steps=imag,model_updates=fits,actor=am,critic=cm)
            save(folder/"updates"/f"{real:09d}.json",plain(audit))
            live("trained_rollout")
            if real in marks or updates%cli.save_every == 0 or stop.exists():
                ptr = capture()
            for b in bs:
                b.after_update()
            cb.after_update()
    finally:
        close_training_vector(norm if norm is not None else vec)

def eval_cell(folder, a, method, cfg, bank, marks, env_class, args, np, torch):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    final = []
    for mark in marks:
        ck = folder/"checkpoints"/f"{mark:09d}.json"
        result_path = folder/"eval"/f"{mark:09d}.json"
        if not ck.is_file():
            continue
        if result_path.is_file():
            final.extend(read(result_path))
            continue
        model_path, norm_path = verified(folder, read(ck))
        vec = DummyVecEnv([lambda: env_class(cfg, bank[0][2], "MAPPO")])
        norm = None
        try:
            norm = VecNormalize.load(str(norm_path), vec)
            norm.training = False
            norm.norm_reward = False
            device = torch.device(a.device)
            pair, _, _ = build_pair(args, norm.observation_space, norm.action_space, device)
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            if state["real_steps"] != mark:
                raise ValueError("评估权重与里程碑不符")
            for actor, weights in zip(pair, state["actors"]):
                actor.actor.load_state_dict(weights)
                actor.prep_rollout()
            del state
            zeros, ones = np.zeros((1, 1, 128), np.float32), np.ones((1, 1), np.float32)
            results = []
            for name, role, rows in bank:
                if role == "final" and mark != a.steps:
                    continue
                for seed in ((123, 124, 125) if role == "final" else (123, 124)):
                    if (a.output/"STOP_REQUESTED").exists():
                        raise InterruptedError("评估暂停；再次运行从未完成的存档继续")
                    set_seed(seed, np, torch)
                    env = env_class(cfg, rows, "MAPPO")
                    x, _ = env.reset(seed=123)
                    while not env.env.done:
                        obs = norm.normalize_obs(x[None])
                        with torch.no_grad():
                            z, _, _ = pair[env.phase].get_actions(obs, zeros, ones)
                            action = int(z.item())
                        x, _, _, _, _ = env.step(action)
                    metrics = env.env.metrics()
                    results.append(dict(trace=name, role=role, policy_seed=seed, step=mark,
                                        metrics=metrics, control=dict(env.stat), method=method))
            save(result_path, results)
            final.extend(results)
            print(f"{folder.name}：已评估 {mark:,} 步存档，{len(results)} 个回合", flush=True)
        finally:
            (norm if norm is not None else vec).close()
    if final:
        save(folder/"EVALUATION.json", dict(rows=final, weights_unchanged=True))
    return final

def canonical(value):
    return json.loads(json.dumps(value))


def trace_sha(rows):
    return hashlib.sha256(json.dumps([asdict(q) for q in rows], sort_keys=True).encode()).hexdigest()


def source_match(path, expected):
    # 只容许无语义变化的 LF/CRLF 转换，不绕过源代码校验。
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return expected in {hashlib.sha256(x).hexdigest() for x in (raw, lf, lf.replace(b"\n", b"\r\n"))}


def read_bank(root, cli, versions):
    protocol = base.read(root / "PROTOCOL.json")
    if (protocol["schema"], protocol["dispatch_protocol"], protocol["steps_per_cell"],
            protocol["rollout"], protocol["dev_marks"]) != (
            "corridor_legacy_interface_v2", base.DISPATCH_PROTOCOL, STEPS, base.ROLLOUT, MARKS):
        raise ValueError("旧基线并非本轮要求的调度权 v2 与冻结训练预算")
    names = {"entry_sha": Path(base.__file__), "environment_sha": Path(base.__file__).with_name("corridor_env.py"),
             "dispatch_sha": Path(base.__file__).with_name("corridor_dispatch.py")}
    for key, path in names.items():
        if not source_match(path, protocol[key]):
            raise ValueError(f"旧基线源码指纹不一致：{path.name}")
    for key, value in versions.items():
        if str(protocol[key]) != str(value):
            raise ValueError(f"训练依赖版本不一致：{key}，旧={protocol[key]}，当前={value}")
    cfg = base.config(cli.config, "E2")
    train = base.trace(base.config(cli.config, "E1"), cli.train_trace)
    if canonical(asdict(cfg)) != protocol["configs"]["E2"] or trace_sha(train) != protocol["train_trace_sha256"]:
        raise ValueError("E2 物理配置或 1.0 倍训练需求与旧基线不一致")
    raw = base.read(root / "DEMAND_BANK.json")
    bank = []
    for item in protocol["bank"]:
        name = item["name"]
        rows = tuple(Request(r["id"], r["time_min"], tuple(r["origin"]), tuple(r["destination"]))
                     for r in raw[name])
        if trace_sha(rows) != item["sha256"]:
            raise ValueError(f"评估需求指纹错误：{name}")
        bank.append((name, item["role"], rows))
    if [sum(role == x for _, role, _ in bank) for x in ("validation", "final")] != [4, 4]:
        raise ValueError("需要原基线的四条验证和四条最终需求")
    return cfg, train, bank, protocol


def make_pair(method, args, space, action_space, device):
    actors, critic, vn = base.build_pair(method, args, space, action_space, device)
    # HARL 某些设备迁移会让统计张量脱离 state_dict；注册同一张量，不改数值或公式。
    for key in VALUE_KEYS:
        if key not in vn.state_dict():
            tensor = getattr(vn, key)
            delattr(vn, key)
            vn.register_buffer(key, tensor)
    return actors, critic, vn


def summary(rows):
    metrics = [r["metrics"] for r in rows]
    def avg(vals):
        return mean(vals) if vals and all(v is not None for v in vals) else None

    result = {k: avg([m[k] for m in metrics])
              for k in ("mean_att_min", "fixed_cutoff_cost_min", "completion")}
    phases = ("route", "access", "queue", "pad_wait", "takeoff", "flight", "landing", "egress")
    for k in phases:
        result[k + "_min"] = avg([m["phase_min_per_generated"][k] for m in metrics])
    queue, pad = result["queue_min"], result["pad_wait_min"]
    result["awt_min"] = queue + pad if None not in (queue, pad) else None
    return dict(n=len(rows), **result)


def run_pressure(cell, destination, cli, cfg, method, backbone, bank, ctor, args, np, torch):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    ck = base.read(cell / "checkpoints" / f"{STEPS:09d}.json")
    model_path, norm_path = base.verified(cell, ck)
    signature = {k: ck[k] for k in ("real_steps", "model_sha", "norm_sha")}
    if signature["real_steps"] != STEPS:
        raise ValueError("压力回放只允许正式最终权重")
    path = destination / "CHECKPOINT.json"
    if path.exists() and base.read(path) != signature:
        raise ValueError("压力回放的原始权重已变更")
    base.save(path, signature)
    vector = DummyVecEnv([lambda: ctor(cfg, bank[0][2], backbone)])
    norm = None
    try:
        norm = VecNormalize.load(str(norm_path), vector)
        norm.training = norm.norm_reward = False
        actors, _, _ = make_pair(backbone, args, norm.observation_space, norm.action_space, torch.device("cpu"))
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        if state["real_steps"] != STEPS:
            raise ValueError("模型内部步数不是正式最终步数")
        for actor, weights in zip(actors, state["actors"], strict=True):
            actor.actor.load_state_dict(weights)
            actor.prep_rollout()
        del state
        z = np.zeros((1, args["recurrent_n"], args["hidden_sizes"][-1]), np.float32)
        mask = np.ones((1, 1), np.float32)

        @torch.no_grad()
        def action(role, obs):
            return int(actors[role].get_actions(norm.normalize_obs(obs[None]), z, mask)[0].item())

        result = []
        for rate in RATES:
            rates = []
            for name, _, rows in bank:
                t0 = min(q.time_min for q in rows)
                dense = validate_requests([replace(q, time_min=t0 + (q.time_min - t0) / rate) for q in rows])
                for policy_seed in POLICY_SEEDS:
                    if (cli.output / "STOP_REQUESTED").exists():
                        raise InterruptedError("压力回放已在回合边界暂停")
                    file = destination / f"{rate:.1f}" / f"{name}_{policy_seed}.json"
                    if file.exists():
                        rates.append(base.read(file))
                        continue
                    base.set_seed(policy_seed, np, torch)
                    env = ctor(cfg, dense, backbone)
                    try:
                        x, _ = env.reset(seed=123)
                        while not env.env.done:
                            if env.phase == 0:
                                pending = sum(q.state == "route" for q in env.env.passengers.values())
                                # 已揭示乘客逐一选站，最后一位走原接口；飞机仍每分钟只决策一次。
                                for _ in range(max(0, pending - 1)):
                                    station = base.STATIONS[action(0, x)]
                                    env.env._route(env.focal(), station)
                                    env.stat["route_" + station] += 1
                                    x = env.observation()
                            x = env.step(action(env.phase, x))[0]
                        row = dict(trace=name, policy_seed=policy_seed, rate=rate,
                                   metrics=env.env.metrics(), control=dict(env.stat))
                        base.save(file, row)
                        rates.append(row)
                    finally:
                        env.close()
            result.append(dict(method=method, rate=rate, **summary(rates)))
            print(f"{cell.name} 压力 {rate:.1f}× 完成 {len(rates)} 回合", flush=True)
        return result
    finally:
        (norm if norm is not None else vector).close()


def build_pair(args, space, action, device):
    return make_pair("MAPPO", args, space, action, device)


# 最初 K3 CONFIG.json 的完整 HARL 参数；不同的配置明确拒绝，不静默覆盖。
K3_ARGS = json.loads(r'''{
  "hidden_sizes":[128,128],"activation_func":"relu","use_feature_normalization":true,
  "initialization_method":"orthogonal_","gain":0.01,"use_naive_recurrent_policy":false,
  "use_recurrent_policy":false,"recurrent_n":1,"data_chunk_length":10,"lr":0.0005,
  "critic_lr":0.0005,"opti_eps":0.00001,"weight_decay":0,"std_x_coef":1,"std_y_coef":0.5,
  "ppo_epoch":5,"critic_epoch":5,"use_clipped_value_loss":true,"clip_param":0.2,
  "actor_num_mini_batch":20,"critic_num_mini_batch":20,"entropy_coef":0.01,
  "value_loss_coef":1,"use_max_grad_norm":true,"max_grad_norm":10.0,"use_gae":true,
  "gamma":0.99,"gae_lambda":0.95,"use_huber_loss":true,"use_policy_active_masks":true,
  "huber_delta":10.0,"action_aggregation":"prod","share_param":false,"fixed_order":false,
  "n_rollout_threads":5,"num_env_steps":10000000,"episode_length":2048,"log_interval":5,
  "eval_interval":25,"use_valuenorm":true,"use_linear_lr_decay":false,
  "use_proper_time_limits":true,"model_dir":null
}''')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "train-trace", "harl-dir", "base-output", "transfer-output", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--phase", choices=("all", "train", "eval"), default="all")
    parser.add_argument("--min-free-gib", type=int, default=5)
    cli = parser.parse_args(argv)
    cli.steps, cli.save_every = STEPS, 1
    cli.output = cli.output.resolve()
    for ref in (cli.base_output.resolve(), cli.transfer_output.resolve()):
        if cli.output == ref or cli.output in ref.parents or ref in cli.output.parents:
            parser.error("输出目录与历史基线目录必须独立")
    if cli.min_free_gib < 1:
        parser.error("磁盘保护至少保留 1 GiB")

    import gymnasium as gym
    import numpy as np
    import stable_baselines3 as sb3
    import torch

    torch.set_num_threads(1)
    if cli.device == "cuda" and not torch.cuda.is_available():
        parser.error("请求 CUDA，但当前 PyTorch 无法使用 CUDA")
    versions = dict(torch=str(torch.__version__), sb3=sb3.__version__,
                    gymnasium=gym.__version__, numpy=np.__version__)
    cfg, rows, bank, base_protocol = read_bank(cli.base_output, cli, versions)
    args, _, vendor = legacy.harl_vendor(cli.harl_dir.resolve(), np, torch)
    if canonical(args) != K3_ARGS:
        diff = {k: [K3_ARGS.get(k), args.get(k)] for k in K3_ARGS.keys() | args.keys()
                if K3_ARGS.get(k) != args.get(k) or (k in K3_ARGS) != (k in args)}
        raise ValueError(f"原始 K3 参数不一致（原版、当前）：{diff}")
    final_bank = [entry for entry in bank if entry[1] == "final"]
    ctor = legacy.build_gym(gym, np)
    reference_ctor = base.build_gym(gym, np)
    tasks = [(arm, seed) for arm in ARMS for seed in SEEDS]
    path = Path(__file__).resolve()
    contract = canonical(dict(schema="corridor_e2_original_k3_lookahead_mappo_v1",
        original_source=K3_SOURCE, historical_mappo_source=K3_MAPPO_SOURCE,
        code_sha={q.name: digest(q) for q in (path, Path(base.__file__), Path(legacy.__file__),
            Path(base.__file__).with_name("corridor_env.py"), Path(base.__file__).with_name("corridor_dispatch.py"))},
        base_protocol=base_protocol, args=args, harl=vendor, runtime=versions, arms=ARMS, plan=tasks,
        steps_per_cell=STEPS, total_real_steps=STEPS*len(tasks), rates=RATES, device=cli.device,
        linear_decay_explicit=True, shadow_start_max_age_exclusive=480,
        checkpoint_every_rollout=True, resume="恢复模型、优化器、归一化、回放池、随机状态；环境开新回合",
        train_rate=1.0, pressure_interface="已揭示乘客批量选站；飞机每分钟一次；冻结正式 actor"))
    failures = 0
    with legacy.lock_run(cli.output):
        manifest = cli.output / "PROTOCOL.json"
        if manifest.exists() and read(manifest) != contract:
            parser.error("输出目录合同不同，不能混写")
        save(manifest, contract)
        save(cli.output / "DEMAND_BANK.json", dict(train=[asdict(r) for r in rows],
             **{name: [asdict(r) for r in trace] for name, _, trace in bank}))
        print(f"3 臂 × 4 种子，{STEPS*len(tasks):,} 真实步；先跑满一个方法的四个种子。", flush=True)
        all_tasks = ([("MAPPO", s) for s in SEEDS] if cli.phase == "eval" else []) + tasks
        if cli.phase == "all":
            all_tasks += [("MAPPO", s) for s in SEEDS]
        for arm, seed in all_tasks:
            tag = f"E2__{('MAPPO_' if arm != 'MAPPO' else '') + arm}__s{seed}"
            err = cli.output / "errors" / (tag + ".json")
            if (cli.output / "STOP_REQUESTED").exists():
                return 130
            try:
                if arm == "MAPPO":
                    root = cli.base_output if seed in SEEDS[:2] else cli.transfer_output
                    cell = root / "cells" / tag
                    job = base.read(cell / "JOB.json")
                    if (job.get("method"), job.get("seed"), job.get("harl_files"), job.get("harl_args")) != (
                            "MAPPO", seed, vendor, canonical(args)):
                        raise ValueError("历史 MAPPO 来源、种子或优化器合同不匹配")
                    if not base.read(cell / "STATUS.json")["evaluated_through_final"]:
                        raise ValueError("历史 MAPPO 尚未完成最终评估；不替换成别的种子")
                else:
                    cell = cli.output / "cells" / tag
                    if cli.phase != "eval":
                        save(cell / "JOB.json", dict(method="MAPPO_" + arm, mechanism=arm,
                             backbone="MAPPO", layer="E2", seed=seed, harl_args=args, harl_files=vendor))
                        train_cell(cell, cli, arm, seed, cfg, rows, MARKS, ctor, args,
                                   independent_update, gym, np, torch)
                    ptr = read(cell / "RESUME.json")
                    if (ptr["real_steps"], ptr["imagined_steps"], ptr["model_updates"]) != (STEPS, 494080, 9264):
                        raise ValueError("正式训练、虚拟步或预测器更新总量不齐")
                    if cli.phase == "train":
                        continue
                    native = eval_cell(cell, SimpleNamespace(device="cpu", steps=STEPS, output=cli.output),
                                       arm, cfg, bank, MARKS, ctor, args, np, torch)
                    if len(native) != 76:
                        raise ValueError(f"常压存档评估数不符：{len(native)}")
                    save(cell / "STATUS.json", dict(real_steps=STEPS, trained=True, evaluated_through_final=True,
                         final=summary([r for r in native if r["role"] == "final"]),
                         late_validation=summary([r for r in native if r["role"] == "validation" and r["step"] >= .8*STEPS])))
                dest = cli.output / "pressure" / tag
                results = run_pressure(cell, dest, cli, cfg, arm, "MAPPO", final_bank,
                                       reference_ctor if arm == "MAPPO" else ctor, args, np, torch)
                save(dest / "DONE.json", [dict(train_seed=seed, **row) for row in results])
            except (KeyboardInterrupt, InterruptedError):
                print("已暂停；沿用原命令可从已落盘进度继续。", flush=True)
                return 130
            except Exception:
                failures += 1
                error = traceback.format_exc()
                save(err, dict(error=error))
                print(f"{tag} 失败，继续下一格：\n{error}", flush=True)
        pressure = [row for done in sorted(cli.output.glob("pressure/*/DONE.json")) for row in read(done)]
        save(cli.output / "SUMMARY.json", dict(failures_this_call=failures, planned_train=12,
             pressure_cells=len(pressure)//len(RATES), planned_pressure=16, results=pressure))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
