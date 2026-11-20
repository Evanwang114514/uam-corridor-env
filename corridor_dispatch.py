"""旧实验的分钟边界空返协议，接入走廊物理执行器。"""

from corridor_env import CorridorEnv

DISPATCH_PROTOCOL = "old_minute_pre_dispatch_v2"


class PriorityCorridor(CorridorEnv):
    def __init__(self, cfg, *, policy_returns=True):
        self.policy_returns = policy_returns
        super().__init__(cfg)

    def _reposition_auto(self):
        # 学习臂只在显式调机点派机，分钟内事件只更新飞机就绪状态。
        if not self.policy_returns:
            return super()._reposition_auto()
        return None

    def _launch(self, a, *args):
        # 预约和实际起飞分别计数，避免把截止前尚未起飞的排班算成真实空返。
        dest = a.target if self.policy_returns and a.station in self.arr and not a.riders else None
        super()._launch(a, *args)
        if dest is not None:
            self.totals["policy_empty_launch_" + dest] += 1

    def _return_planes(self):
        # 旧调机批次按飞机 ID 字符串顺序遍历，不由策略点名飞机。
        return sorted(
            (p for p in self.planes.values() if p.station in self.arr and p.state == "ready" and not p.riders),
            key=lambda p: str(p.id),
        )

    def _can_return(self, p, dest):
        # 新走廊的停车与航程约束不能被优先级绕过；电量暂不足仍可尝试充电。
        return (self._eligible(p, dest) and self._park_available(dest)
                and (not self.cfg.energy or self._soc_cost(p.station, dest) + self.cfg.reserve_soc <= 1 + 1e-9))

    def targets(self):
        planes = self._return_planes()
        # 旧版按非空需求队列选目标，不再用 READY+在途量剔除队列。
        return tuple(s for s in self.dep if self.queue[s] and any(self._can_return(p, s) for p in planes))

    def _lq_key(self, station, remaining):
        ready = sum(p.state == "ready" and not p.riders and p.station == station for p in self.planes.values())
        return remaining[station], -ready, -self.dep.index(station)

    def choose_returns(self, requested, stat):
        if not self.policy_returns or self.now % self._ticks(self.cfg.decision_min):
            raise RuntimeError("调机动作必须在旧式分钟调机点执行")
        legal = self.targets()
        stat["air_decisions"] += 1
        stat["air_no_opportunity"] += not legal
        stat["air_multi_target"] += len(legal) >= 2
        stat["requested_" + requested] += 1
        if not legal:
            return
        remaining = {s: len(self.queue[s]) for s in self.dep}
        # 合法的策略选站优先于 LQ；仅对非法请求做旧式投影。
        natural = max(self.dep, key=lambda s: self._lq_key(s, remaining))
        priority = requested if requested in legal else natural if natural in legal else legal[0]
        stat["air_priority_projection"] += priority != requested
        stat["applied_" + priority] += 1
        for plane in self._return_planes():
            possible = [s for s in self.dep if remaining[s] and self._can_return(plane, s)]
            if not possible:
                continue
            dest = priority if priority in possible else max(possible, key=lambda s: self._lq_key(s, remaining))
            result = self._dispatch(plane, dest)
            if result == "charging":
                stat["air_charge_only"] += 1
            if result != "reserved":
                continue
            # 仅扣本批已承诺的运力，不修改真实乘客排队状态。
            remaining[dest] = max(0, remaining[dest] - self.cfg.seats)
            stat["air_dispatched"] += 1
            stat["executed_" + dest] += 1
            stat["air_requested_executed"] += dest == requested
            self.totals["priority_match" if dest == priority else "priority_fallback"] += 1
