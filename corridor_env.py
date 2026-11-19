"""独立的 UAM 走廊环境。仅依赖 Python 标准库，不包含训练器。"""

from __future__ import annotations

import argparse
from bisect import insort
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import csv
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path
import random


@dataclass(frozen=True)
class Station:
    role: str
    xy: tuple[float, float]
    pads: int = 1
    takeoff_min: float = 0.5
    landing_min: float = 0.6
    separation_min: float = 0.0
    stands: int = 3
    turnaround_min: float = 3.0
    chargers: int = 8
    charge_soc_per_min: float = 0.04
    parking: int = 24


@dataclass(frozen=True)
class Region:
    # 顺序为 x 最小、x 最大、y 最小、y 最大；坐标单位为公里。
    box: tuple[float, float, float, float]
    weight: float = 1.0
    distribution: str = "uniform"


def default_stations():
    return {
        "V0": Station("departure", (0, -8)),
        "V1": Station("departure", (-3, 0)),
        "V3": Station("departure", (0, 8)),
        "V2": Station("arrival", (50, 0)),
    }


@dataclass(frozen=True)
class Config:
    stations: dict[str, Station] = field(default_factory=default_stations)
    origins: tuple[Region, ...] = (Region((-5, 4, -12, 12)),)
    destinations: tuple[Region, ...] = (Region((47, 53, -4, 4)),)
    tick_min: float = 0.1
    decision_min: float = 1.0
    cutoff_min: float = 600.0
    access_kmh: float = 40.0
    egress_kmh: float = 40.0
    air_kmh: float = 200.0
    cav_kmh: float = 40.0
    # 新地图直接用公里；导入旧坐标时，站点和 OD 共用同一个换算。
    km_per_unit: float = 1.0
    egress: bool = True
    fleet: int = 48
    initial_fleet: dict[str, int] | None = None
    seats: int = 1
    boarding: str = "immediate"
    tlof: bool = False
    operation_time: bool = True
    turnaround: bool = False
    finite_stands: bool = False
    finite_charging: bool = False
    parking: bool = False
    energy: bool = True
    initial_soc: float = 1.0
    reserve_soc: float = 0.2
    target_soc: float = 0.8
    fixed_flight_soc: float = 0.03
    soc_per_km: float = 0.005
    reposition: str = "lq"
    lateral: bool = False
    fixed_returns: dict[str, str] = field(default_factory=dict)
    passengers: int = 300
    arrival_min: float = 1.0
    arrival_mode: str = "uniform"
    time_scale: float = 1.0
    seed: int = 123
    demand_csv: str | None = None

    def departures(self):
        return tuple(sorted(k for k, v in self.stations.items() if v.role == "departure"))

    def arrivals(self):
        return tuple(sorted(k for k, v in self.stations.items() if v.role == "arrival"))

    def allocation(self):
        if self.initial_fleet is not None:
            return dict(sorted(self.initial_fleet.items()))
        dep = self.departures()
        q, rem = divmod(self.fleet, len(dep))
        return {v: q + (i < rem) for i, v in enumerate(dep)}

    def validate(self):
        def positive(*values):
            return all(math.isfinite(x) and x > 0 for x in values)

        def nonnegative(*values):
            return all(math.isfinite(x) and x >= 0 for x in values)

        if not self.departures() or not self.arrivals():
            raise ValueError("地图至少需要一个出发站和一个到达站")
        if not positive(self.tick_min, self.decision_min, self.cutoff_min, self.km_per_unit,
                        self.access_kmh, self.egress_kmh, self.air_kmh, self.cav_kmh,
                        self.arrival_min, self.time_scale):
            raise ValueError("时钟、速度、比例尺和到达时间参数必须为有限正数")
        if self.decision_min < self.tick_min:
            raise ValueError("决策间隔不可小于物理时间步长")
        if not all(type(x) is int and x > 0 for x in (self.fleet, self.seats, self.passengers)):
            raise ValueError("机队、座位数和乘客数必须为正整数")
        if self.boarding not in {"immediate", "full"} or self.reposition not in {"off", "fixed", "lq", "manual"}:
            raise ValueError("载客模式或调机模式非法")
        if self.arrival_mode not in {"uniform", "varying", "poisson"}:
            raise ValueError("到达模式必须是 uniform、varying 或 poisson")
        if not (0 <= self.reserve_soc < self.target_soc <= 1 and 0 <= self.initial_soc <= 1
                and nonnegative(self.fixed_flight_soc, self.soc_per_km)):
            raise ValueError("电量、能耗或充电目标非法")
        if self.finite_charging and not self.energy:
            raise ValueError("启用有限充电位须同时启用电量模型")
        if self.tlof and not self.operation_time:
            raise ValueError("共享 TLOF 需要使用实际起降占用时间")
        if self.finite_stands and not self.turnaround:
            raise ValueError("有限周转位须同时启用周转")
        for k, v in self.stations.items():
            if not k or v.role not in {"departure", "arrival"} or len(v.xy) != 2 or not all(map(math.isfinite, v.xy)):
                raise ValueError(f"机场位置或角色非法：{k}")
            if not all(type(x) is int and x >= 0 for x in (v.pads, v.stands, v.chargers, v.parking)):
                raise ValueError(f"机场容量必须是非负整数：{k}")
            if not nonnegative(v.takeoff_min, v.landing_min, v.separation_min, v.turnaround_min):
                raise ValueError(f"机场时长非法：{k}")
            if self.tlof and (not v.pads or not positive(v.takeoff_min, v.landing_min)):
                raise ValueError(f"TLOF 开启时须有起降位与正的起降占用时间：{k}")
            if self.finite_stands and v.turnaround_min and not v.stands:
                raise ValueError(f"周转开启时须有周转位：{k}")
            if self.energy and not positive(v.charge_soc_per_min):
                raise ValueError(f"充电速率须为正：{k}")
            if (self.finite_charging and not v.chargers) or (self.parking and not v.parking):
                raise ValueError(f"开启的充电或停车限制须有正容量：{k}")
        alloc = self.allocation()
        if not set(alloc) <= self.stations.keys() or any(type(n) is not int or n < 0 for n in alloc.values()) or sum(alloc.values()) != self.fleet:
            raise ValueError("初始机队分配必须使用合法机场、非负数量，且合计等于机队规模")
        if self.parking and any(n > self.stations[k].parking for k, n in alloc.items()):
            raise ValueError("初始机队分配超过停车位容量；不会自动放宽")
        if any(k not in self.arrivals() or v not in self.departures() for k, v in self.fixed_returns.items()):
            raise ValueError("固定回流映射必须是 到达站:出发站")
        for zones in (self.origins, self.destinations):
            if not zones or not positive(sum(z.weight for z in zones)):
                raise ValueError("必须提供有效的起点和终点刷新区")
            for z in zones:
                x0, x1, y0, y1 = z.box
                if not all(map(math.isfinite, z.box)) or x0 > x1 or y0 > y1 or not positive(z.weight) or z.distribution not in {"uniform", "centered"}:
                    raise ValueError(f"刷新区参数非法：{z}")
        return self


@dataclass(frozen=True)
class Request:
    id: str
    time_min: float
    origin: tuple[float, float]
    destination: tuple[float, float]


def make_requests(cfg: Config):
    # 到达节奏和空间各用独立随机流，只扫到达时间时不会改 OD。
    space = random.Random(cfg.seed)
    clock = random.Random(cfg.seed ^ 0x5A19)

    def point(zones):
        z = space.choices(zones, weights=[v.weight for v in zones])[0]
        x0, x1, y0, y1 = z.box
        draw = space.uniform if z.distribution == "uniform" else space.triangular
        return draw(x0, x1), draw(y0, y1)

    if cfg.demand_csv is not None:
        with Path(cfg.demand_csv).open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        time_keys = {"time_min", "time"}
        if not rows or not time_keys.intersection(rows[0]):
            raise ValueError("需求 CSV 须含 time_min（或 time，单位分钟）及 OD 列")
        tkey = "time_min" if "time_min" in rows[0] else "time"
        result = [
            Request(r["id"], float(r[tkey]) * cfg.time_scale,
                    (float(r["origin_x"]), float(r["origin_y"])),
                    (float(r["dest_x"]), float(r["dest_y"]))) for r in rows
        ]
        return validate_requests(result)
    result, minute = [], 0.0
    for i in range(cfg.passengers):
        result.append(Request(str(i), minute * cfg.time_scale, point(cfg.origins), point(cfg.destinations)))
        if cfg.arrival_mode == "poisson":
            minute += clock.expovariate(1 / cfg.arrival_min)
        elif cfg.arrival_mode == "varying":
            # 一轮低谷和一轮高峰；平均间隔与均匀版本近似一致。
            minute += cfg.arrival_min * (1 + 0.55 * math.sin(2 * math.pi * (i + 0.5) / cfg.passengers))
        else:
            minute += cfg.arrival_min
    return result


def validate_requests(requests):
    if not requests or len({r.id for r in requests}) != len(requests):
        raise ValueError("需求不能为空且乘客 ID 不可重复")
    for r in requests:
        if not math.isfinite(r.time_min) or r.time_min < 0 or any(
            len(x) != 2 or not all(map(math.isfinite, x)) for x in (r.origin, r.destination)
        ):
            raise ValueError(f"乘客坐标或揭示时间非法：{r.id}")
    return sorted(requests, key=lambda r: r.time_min)


class PadCalendar:
    """共享起降位的预约逻辑，沿用旧五机场版的双端同步预约思想。"""

    def __init__(self, capacity, gap):
        self.lanes = [[] for _ in range(capacity)]
        self.gap = gap

    def earliest(self, start, duration):
        available = []
        for i, lane in enumerate(self.lanes):
            t = start
            for a, b in lane:
                if t + duration + self.gap <= a:
                    break
                if t < b + self.gap:
                    t = b + self.gap
            available.append((t, i))
        return min(available)

    def book(self, tick, duration, lane):
        insort(self.lanes[lane], (tick, tick + duration))


@dataclass
class Passenger:
    req: Request
    state: str = "route"
    since: int = 0
    dep: str | None = None
    arr: str | None = None
    phases: Counter = field(default_factory=Counter)


@dataclass
class Aircraft:
    id: int
    station: str | None
    soc: float
    state: str = "ready"
    target: str | None = None
    riders: tuple[str, ...] = ()


class CorridorEnv:
    """调用 reset / step；观察只包含已揭示需求和已承诺的飞机活动。"""

    def __init__(self, config: Config | None = None):
        self.cfg = (config or Config()).validate()
        self.dep = self.cfg.departures()
        self.arr = self.cfg.arrivals()

    def _ticks(self, minutes):
        return max(0, math.ceil(minutes / self.cfg.tick_min - 1e-9))

    def dist_km(self, a, b):
        return math.dist(a, b) * self.cfg.km_per_unit

    def _drive(self, a, b, kmh):
        return self._ticks(60 * self.dist_km(a, b) / kmh)

    def _air(self, a, b):
        return max(1, self._ticks(60 * self.dist_km(self.cfg.stations[a].xy, self.cfg.stations[b].xy) / self.cfg.air_kmh))

    def _destination(self, point):
        return min(self.arr, key=lambda s: (self.dist_km(point, self.cfg.stations[s].xy), s))

    def reset(self, requests=None):
        rows = make_requests(self.cfg) if requests is None else validate_requests(list(requests))
        self.request_count = len(rows)
        self.now = 0
        self._seq = itertools.count()
        self._events = []
        self.passengers = {}
        self.finished = 0
        self.area_ticks = 0
        self.totals = Counter()
        self.pa = Counter()
        self.local = Counter(self.cfg.allocation())
        self.queue = {s: deque() for s in self.dep}
        self.turnq = {s: deque() for s in self.cfg.stations}
        self.chargeq = {s: deque() for s in self.cfg.stations}
        self.turn_active = Counter()
        self.charge_active = Counter()
        self.pads = {
            s: PadCalendar(v.pads, self._ticks(v.separation_min))
            for s, v in self.cfg.stations.items()
        } if self.cfg.tlof else {}
        self.planes = {
            i: Aircraft(i, s, self.cfg.initial_soc)
            for i, s in enumerate(s for s, n in self.local.items() for _ in range(n))
        }
        for q in rows:
            self._push(self._ticks(q.time_min), self._reveal, q)
        self.cutoff = self._ticks(self.cfg.cutoff_min)
        self.done = False
        self._drain()
        return self.observation()

    def _push(self, tick, fn, *args):
        heapq.heappush(self._events, (tick, next(self._seq), fn, args))

    def _reveal(self, r):
        self.passengers[r.id] = Passenger(r, since=self.now, arr=self._destination(r.destination))

    def _phase(self, p, state):
        p.phases[p.state] += self.now - p.since
        p.since = self.now
        p.state = state

    def _route(self, p, station):
        p.dep = station
        self._phase(p, "access")
        xy = self.cfg.stations[station].xy
        self._push(self.now + self._drive(p.req.origin, xy, self.cfg.access_kmh), self._arrive_queue, p)

    def _arrive_queue(self, p):
        self._phase(p, "queue")
        self.queue[p.dep].append(p.req.id)

    def _finish(self, p):
        self._phase(p, "finished")
        self.finished += 1

    def _park_available(self, s):
        return not self.cfg.parking or self.local[s] + self.pa[s] < self.cfg.stations[s].parking

    def _soc_cost(self, a, b):
        if not self.cfg.energy:
            return 0.0
        return self.cfg.fixed_flight_soc + self.cfg.soc_per_km * self.dist_km(self.cfg.stations[a].xy, self.cfg.stations[b].xy)

    def _charge(self, a, target):
        a.state = "charge_queue"
        self.chargeq[a.station].append((a, target, self.now))
        self._fill_charge(a.station)

    def _fill_charge(self, s):
        v = self.cfg.stations[s]
        while self.chargeq[s] and (not self.cfg.finite_charging or self.charge_active[s] < v.chargers):
            a, target, since = self.chargeq[s].popleft()
            a.state = "charging"
            self.charge_active[s] += 1
            self.totals["charge_wait_ticks"] += self.now - since
            dt = self._ticks(max(0.0, target - a.soc) / v.charge_soc_per_min)
            self._push(self.now + dt, self._charged, a, target)

    def _charged(self, a, target):
        a.soc = target
        a.state = "ready"
        self.charge_active[a.station] -= 1
        self._fill_charge(a.station)

    def _after_turn(self, a):
        if self.cfg.energy and a.soc + 1e-9 < self.cfg.target_soc:
            self._charge(a, self.cfg.target_soc)
        else:
            a.state = "ready"

    def _fill_turn(self, s):
        v = self.cfg.stations[s]
        while self.turnq[s] and (not self.cfg.finite_stands or self.turn_active[s] < v.stands):
            a, since = self.turnq[s].popleft()
            a.state = "turnaround"
            self.turn_active[s] += 1
            self.totals["turn_wait_ticks"] += self.now - since
            self._push(self.now + self._ticks(v.turnaround_min), self._turned, a)

    def _turned(self, a):
        self.turn_active[a.station] -= 1
        self._after_turn(a)
        self._fill_turn(a.station)

    def _flight_slot(self, origin, dest, cruise):
        a, b = (self._ticks(self.cfg.stations[origin].takeoff_min),
                self._ticks(self.cfg.stations[dest].landing_min)) if self.cfg.operation_time else (0, 0)
        if not self.cfg.tlof:
            return self.now, a, b
        candidate = self.now
        # 两端冲突时只延后起飞；不让飞机无预约升空。
        while True:
            start, lane_a = self.pads[origin].earliest(candidate, a)
            needed = start + a + cruise
            landing, lane_b = self.pads[dest].earliest(needed, b)
            if landing == needed:
                self.pads[origin].book(start, a, lane_a)
                self.pads[dest].book(landing, b, lane_b)
                return start, a, b
            candidate = landing - a - cruise

    def _dispatch(self, a, dest, riders=()):
        src = a.station
        if not self._park_available(dest):
            return "parking_full"
        cost = self._soc_cost(src, dest)
        need = cost + self.cfg.reserve_soc if self.cfg.energy else 0.0
        if need > 1 + 1e-9:
            return "energy_infeasible"
        if a.soc + 1e-9 < need:
            self._charge(a, min(1.0, max(need, self.cfg.target_soc)))
            return "charging"
        cruise = self._air(src, dest)
        start, takeoff, landing = self._flight_slot(src, dest, cruise)
        # 目的地停车位自承诺时保守预留，飞机落地后转换为实际占用。
        self.pa[dest] += 1
        a.state, a.target, a.riders = "reserved", dest, tuple(riders)
        self.totals["pad_depart_wait_ticks"] += start - self.now
        for pid in riders:
            self._phase(self.passengers[pid], "pad_wait")
        self._push(start, self._launch, a, takeoff, cruise, landing, cost)
        return "reserved"

    def _launch(self, a, takeoff, cruise, landing, cost):
        a.state = "airborne"
        a.soc = max(0.0, a.soc - cost)
        self.local[a.station] -= 1
        a.station = None
        self.totals["service_flights" if a.riders else "empty_flights"] += 1
        for pid in a.riders:
            self._phase(self.passengers[pid], "takeoff" if takeoff else "flight")
        if takeoff:
            self._push(self.now + takeoff, self._air_phase, a, "flight")
        if landing:
            self._push(self.now + takeoff + cruise, self._air_phase, a, "landing")
        self._push(self.now + takeoff + cruise + landing, self._land, a)

    def _air_phase(self, a, state):
        for pid in a.riders:
            self._phase(self.passengers[pid], state)

    def _land(self, a):
        s = a.target
        self.pa[s] -= 1
        self.local[s] += 1
        a.station, a.target = s, None
        for pid in a.riders:
            p = self.passengers[pid]
            if self.cfg.egress:
                self._phase(p, "egress")
                d = self._drive(self.cfg.stations[s].xy, p.req.destination, self.cfg.egress_kmh)
                self._push(self.now + d, self._finish, p)
            else:
                self._finish(p)
        a.riders = ()
        if self.cfg.turnaround and self.cfg.stations[s].turnaround_min > 0:
            a.state = "turn_queue"
            self.turnq[s].append((a, self.now))
            self._fill_turn(s)
        else:
            self._after_turn(a)

    def _local_service(self):
        for a in self.planes.values():
            if a.state != "ready" or a.station not in self.dep or not self.queue[a.station]:
                continue
            queue = self.queue[a.station]
            # 同一载客航班只绑定相同目的机场的乘客，默认单客无凑批。
            first = self.passengers[queue[0]]
            ids = tuple(itertools.islice((i for i in queue if self.passengers[i].arr == first.arr), self.cfg.seats))
            if self.cfg.boarding == "full" and len(ids) < self.cfg.seats:
                continue
            status = self._dispatch(a, first.arr, ids)
            if status == "reserved":
                for pid in ids:
                    queue.remove(pid)

    def _eligible(self, a, dest):
        if a.state != "ready" or dest not in self.dep or a.station == dest:
            return False
        return a.station in self.arr or self.cfg.lateral and a.station in self.dep

    def _reposition_auto(self):
        if self.cfg.reposition not in {"fixed", "lq"}:
            return
        inbound = Counter(a.target for a in self.planes.values() if a.target in self.dep)
        ready = Counter(a.station for a in self.planes.values() if a.state == "ready")
        for a in self.planes.values():
            if a.state != "ready" or a.station in self.dep and self.queue[a.station]:
                continue
            candidates = (self.cfg.fixed_returns.get(a.station, self.dep[0]),) if self.cfg.reposition == "fixed" and a.station in self.arr else (
                self.dep if self.cfg.reposition == "lq" else ()
            )
            # 目的站满位时，仅看出发站 READY 会误判为运力已足；
            # 若没有已预约离站的飞机，允许回流释放停车位。
            can_release = (self.cfg.parking and a.station in self.arr and not self._park_available(a.station)
                           and not any(p.station == a.station and p.state == "reserved"
                                       for p in self.planes.values()))
            candidates = [s for s in candidates if self._eligible(a, s) and (
                len(self.queue[s]) > (inbound[s] + ready[s]) * self.cfg.seats
                or can_release and self.queue[s] and self.passengers[self.queue[s][0]].arr == a.station
            )]
            candidates.sort(key=lambda s: (-len(self.queue[s]), self._air(a.station, s), s))
            for dest in candidates:
                status = self._dispatch(a, dest)
                if status in {"reserved", "charging"}:
                    ready[a.station] -= 1
                    if status == "reserved":
                        inbound[dest] += 1
                    break

    def _drain(self):
        while True:
            while self._events and self._events[0][0] == self.now:
                _, _, fn, args = heapq.heappop(self._events)
                fn(*args)
            if self.finished == self.request_count or self.now >= self.cutoff:
                self.done = True
                return
            self._local_service()
            self._reposition_auto()
            if not self._events or self._events[0][0] != self.now:
                return

    def observation(self):
        access = Counter(p.dep for p in self.passengers.values() if p.state == "access")
        return {
            "time_min": self.now * self.cfg.tick_min,
            "routing": [
                {"id": p.req.id, "origin": p.req.origin, "destination": p.req.destination, "arrival_station": p.arr}
                for p in self.passengers.values() if p.state == "route"
            ],
            "queue": {s: len(q) for s, q in self.queue.items()},
            "committed_access": {s: access[s] for s in self.dep},
            "resources": {
                s: {"parked": self.local[s], "reserved_parking": self.pa[s],
                    "turnaround": self.turn_active[s], "turn_queue": len(self.turnq[s]),
                    "charging": self.charge_active[s], "charge_queue": len(self.chargeq[s])}
                for s in self.cfg.stations
            },
            "aircraft": {a.id: {"station": a.station, "state": a.state, "soc": a.soc, "target": a.target}
                         for a in self.planes.values()},
            "reposition_candidates": {
                a.id: [s for s in self.dep if self._eligible(a, s) and self._park_available(s)
                       and (not self.cfg.energy or self._soc_cost(a.station, s) + self.cfg.reserve_soc <= 1.0 + 1e-9)]
                for a in self.planes.values() if a.state == "ready"
            } if self.cfg.reposition == "manual" else {},
        }

    def step(self, routes=None, reposition=None):
        if self.done:
            raise RuntimeError("回合已结束，请先 reset")
        routes = routes or {}
        reposition = reposition or {}
        if reposition and self.cfg.reposition != "manual":
            raise ValueError("仅手动调机模式接收飞机动作")
        for pid, station in routes.items():
            if pid not in self.passengers or self.passengers[pid].state != "route" or station not in self.dep:
                raise ValueError(f"非法选站动作：{pid} → {station}")
        for ident, dest in reposition.items():
            if ident not in self.planes or not self._eligible(self.planes[ident], dest):
                raise ValueError(f"非法调机动作：{ident} → {dest}")
        for pid, station in routes.items():
            self._route(self.passengers[pid], station)
        applied = {ident: self._dispatch(self.planes[ident], dest) for ident, dest in reposition.items()}
        start_area = self.area_ticks
        until = min(self.cutoff, self.now + self._ticks(self.cfg.decision_min))
        self._drain()
        while not self.done and self.now < until:
            next_tick = min(until, self._events[0][0]) if self._events else until
            self.area_ticks += (next_tick - self.now) * (len(self.passengers) - self.finished)
            self.now = next_tick
            self._drain()
        complete = self.finished == self.request_count
        info = {"reposition_status": applied, "elapsed_end_min": self.now * self.cfg.tick_min}
        if self.done:
            info["metrics"] = self.metrics()
        return self.observation(), -(self.area_ticks - start_area) * self.cfg.tick_min, complete, self.done and not complete, info

    def metrics(self):
        totals = Counter()
        completed = []
        for p in self.passengers.values():
            totals.update(p.phases)
            if p.state == "finished":
                completed.append(sum(p.phases.values()))
            else:
                totals[p.state] += self.now - p.since
        if sum(totals.values()) != self.area_ticks:
            raise AssertionError("逐乘客分段与累计人时不守恒")
        n = len(self.passengers)
        full = self.finished == self.request_count
        minute = self.cfg.tick_min
        return {
            "generated": n, "planned": self.request_count, "finished": self.finished,
            "completion": self.finished / n if n else None,
            "all_requests_finished": full,
            "door_to_door": self.cfg.egress,
            "mean_att_min": sum(completed) * minute / n if full and self.cfg.egress else None,
            "mean_to_landing_min": sum(completed) * minute / n if full and not self.cfg.egress else None,
            "fixed_cutoff_cost_min": self.area_ticks * minute / n if n else None,
            "phase_min_per_generated": {k: totals[k] * minute / n if n else None
                                       for k in ("route", "access", "queue", "pad_wait", "takeoff", "flight", "landing", "egress")},
            "aircraft_states": dict(Counter(a.state for a in self.planes.values())),
            "aircraft_diagnostics": dict(self.totals),
        }


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description="UAM 走廊环境参数；不启动训练。")
    parser.add_argument("--config", type=Path, help="可选 JSON 参数文件；CMD 参数优先")
    parser.add_argument("--save-config", type=Path, help="另存实际生效的配置，不覆盖已有文件")
    parser.add_argument("--station", action="append", help="完整地图：可重复 ID:departure或arrival:x:y")
    parser.add_argument("--station-position", action="append", help="只改一站坐标：可重复 ID:x:y")
    parser.add_argument("--station-set", action="append", help="按站改参数：可重复 ID:字段:值")
    parser.add_argument("--origin-zone", action="append", help="可重复 xmin,xmax,ymin,ymax[,权重[,uniform或centered]]")
    parser.add_argument("--destination-zone", action="append", help="同起点区格式")
    parser.add_argument("--destination-shift-km", type=float, help="向东平移目的机场及终点区")
    parser.add_argument("--initial-fleet", help="例如 V0:16,V1:16,V3:16")
    parser.add_argument("--fixed-returns", help="固定回流映射，例如 V2:V1")
    parser.add_argument("--geometry", action="store_true", help="只算同 OD 的无排队交通量级，不运行环境")
    parser.add_argument("--baselines", action="store_true", help="运行四条在线规则、逐站客机集中与 CAV 直达，不训练")
    parser.add_argument("--baseline-output", type=Path, help="基线结果目录；同配置重跑会跳过已结算组")
    parser.add_argument("--rule-service-rates", help="规则估计服务率（人/分钟），例如 V0:3,V1:3；默认各站 3")
    parser.add_argument("--qtti-denominators", help="规则拥堵系数分母，例如 V0:30,V1:6；默认 V0 为 30、其余为 6")
    switches = ("egress", "tlof", "operation_time", "turnaround", "finite_stands",
                "finite_charging", "parking", "energy", "lateral")
    for flag in switches:
        parser.add_argument("--" + flag.replace("_", "-"), choices=("on", "off"))
    for key in ("tick_min", "decision_min", "cutoff_min", "access_kmh", "egress_kmh", "air_kmh", "cav_kmh",
                "km_per_unit", "initial_soc", "reserve_soc", "target_soc", "fixed_flight_soc", "soc_per_km",
                "arrival_min", "time_scale"):
        parser.add_argument("--" + key.replace("_", "-"), type=float)
    for key in ("fleet", "seats", "passengers", "seed"):
        parser.add_argument("--" + key, type=int)
    parser.add_argument("--demand-csv")
    parser.add_argument("--reposition", choices=("off", "fixed", "lq", "manual"))
    parser.add_argument("--boarding", choices=("immediate", "full"))
    parser.add_argument("--arrival-mode", choices=("uniform", "varying", "poisson"))
    args = parser.parse_args(argv)
    d = asdict(Config())
    def mappings(s, cast):
        result = {}
        for x in s.split(","):
            a, b = x.split(":", 1)
            if a in result:
                raise ValueError(f"机场配置重复：{a}")
            result[a] = cast(b)
        return result

    def region(s):
        parts = s.split(",")
        if not 4 <= len(parts) <= 6:
            raise ValueError(f"刷新区格式错误：{s}")
        return Region(tuple(map(float, parts[:4])), float(parts[4]) if len(parts) > 4 else 1.0,
                      parts[5] if len(parts) > 5 else "uniform")

    try:
        raw = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
        if not isinstance(raw, dict):
            raise ValueError("JSON 配置必须是对象")
        if set(raw) - set(d):
            raise ValueError("JSON 中存在未知配置字段：" + ", ".join(sorted(set(raw) - set(d))))
        d.update(raw)
        for k in d.keys() & vars(args).keys():
            v = getattr(args, k)
            if v is not None:
                d[k] = (v == "on") if k in switches else v
        if any(type(d[k]) is not bool for k in switches):
            raise ValueError("JSON 资源开关须使用 true 或 false")
        if args.station:
            d["stations"] = {}
            for spec in args.station:
                name, role, x, y = spec.split(":")
                if name in d["stations"]:
                    raise ValueError(f"机场 ID 重复：{name}")
                d["stations"][name] = asdict(Station(role, (float(x), float(y))))
        for spec in args.station_position or ():
            name, x, y = spec.split(":")
            d["stations"][name]["xy"] = (float(x), float(y))
        typed = {k: int if k in {"pads", "stands", "chargers", "parking"} else float
                 for k in Station.__dataclass_fields__ if k not in {"role", "xy"}}
        for spec in args.station_set or ():
            name, key, value = spec.split(":")
            targets = d["stations"].values() if name == "all" else (d["stations"][name],)
            for item in targets:
                item[key] = typed[key](value)
        if args.initial_fleet is not None:
            d["initial_fleet"] = mappings(args.initial_fleet, int)
        if args.fixed_returns is not None:
            d["fixed_returns"] = mappings(args.fixed_returns, str)
        d["origins"] = tuple(map(region, args.origin_zone)) if args.origin_zone else tuple(Region(**x) for x in d["origins"])
        d["destinations"] = tuple(map(region, args.destination_zone)) if args.destination_zone else tuple(Region(**x) for x in d["destinations"])
        if args.destination_shift_km is not None:
            if d["demand_csv"] is not None:
                raise ValueError("目的侧整体平移仅用于生成需求；固定 CSV 的 OD 不自动改写")
            if not math.isfinite(d["km_per_unit"]) or d["km_per_unit"] <= 0:
                raise ValueError("比例尺须为正才能平移")
            dx = args.destination_shift_km / d["km_per_unit"]
            for item in d["stations"].values():
                if item["role"] == "arrival":
                    x, y = item["xy"]
                    item["xy"] = (x + dx, y)
            d["destinations"] = tuple(Region((z.box[0] + dx, z.box[1] + dx, *z.box[2:]), z.weight, z.distribution)
                                      for z in d["destinations"])
        d["stations"] = {k: Station(**v) for k, v in d["stations"].items()}
        cfg = Config(**d).validate()
        def rule_values(raw, defaults):
            overrides = mappings(raw, float) if raw else {}
            if not set(overrides) <= set(defaults):
                raise ValueError("规则参数只接受当前地图的出发站")
            if any(not math.isfinite(v) or v <= 0 for v in overrides.values()):
                raise ValueError("规则估计参数必须为有限正数")
            return defaults | overrides
        if not args.baselines and (args.baseline_output is not None or args.rule_service_rates is not None
                                   or args.qtti_denominators is not None):
            raise ValueError("规则参数及结果目录须与 --baselines 一起使用")
        if args.baselines and cfg.reposition != "lq":
            raise ValueError("航空基线统一使用 --reposition lq")
        args.rule_service_rates = rule_values(args.rule_service_rates, dict.fromkeys(cfg.departures(), 3.0))
        args.qtti_denominators = rule_values(
            args.qtti_denominators, {s: 30.0 if s == "V0" else 6.0 for s in cfg.departures()})
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return cfg, args


def geometric_reference(cfg, requests):
    """同一 OD 的直达与无排队航线基准，不是排队仿真结果。"""
    env = CorridorEnv(cfg)
    totals = Counter()
    for r in requests:
        arr = env._destination(r.destination)
        egress = env._drive(cfg.stations[arr].xy, r.destination, cfg.egress_kmh) if cfg.egress else 0
        totals["cav"] += env._drive(r.origin, r.destination, cfg.cav_kmh)
        totals["uam_best"] += egress + min(
            env._drive(r.origin, cfg.stations[dep].xy, cfg.access_kmh) + env._air(dep, arr) + (
                env._ticks(cfg.stations[dep].takeoff_min) + env._ticks(cfg.stations[arr].landing_min)
                if cfg.operation_time else 0
            ) for dep in env.dep
        )
    return {f"{k}_mean_min": v * cfg.tick_min / len(requests) for k, v in totals.items()}


BASELINE_RULES = ("SPF_SRC", "STTF", "QTTI", "READY")
BASELINE_PHASES = ("route", "access", "queue", "pad_wait", "takeoff", "flight", "landing", "egress")
CAV_DIRECT = "CAV_DIRECT"


def baseline_route_actions(env, obs, rule, rates, denominators, forced=None):
    """按已揭示的决策快照打分，不查看尚未到达的需求。"""
    if forced is not None:
        return {p["id"]: forced for p in obs["routing"]}
    if not obs["routing"]:
        return {}
    cfg = env.cfg
    access_left = {s: [] for s in env.dep}
    supply = {s: [] for s in env.dep}
    if rule == "READY":
        for p in env.passengers.values():
            if p.state == "access":
                due = p.since + env._drive(p.req.origin, cfg.stations[p.dep].xy, cfg.access_kmh)
                access_left[p.dep].append(max(0, due - env.now) * cfg.tick_min)
        for a in obs["aircraft"].values():
            if a["state"] == "ready" and a["station"] in supply:
                supply[a["station"]].append((0.0, a["soc"]))
        # 只读已经排定的充电结束事件，不读取事件队列里的未揭示请求。
        for due, _, fn, payload in env._events:
            if fn == env._charged:
                a, target = payload
                if a.station in supply:
                    supply[a.station].append(((due - env.now) * cfg.tick_min, target))

    def score(p, station):
        dep = cfg.stations[station]
        access = env._drive(p["origin"], dep.xy, cfg.access_kmh) * cfg.tick_min
        queued = obs["queue"][station]
        wait = int(queued / rates[station])
        # 保留昨天的相对地面权重；分钟与距离改用新版统一尺度。
        if rule == "SPF_SRC":
            return 0.75 * access + wait
        if rule == "STTF":
            return access + wait
        if rule == "QTTI":
            return access + wait + access * obs["committed_access"][station] / denominators[station]

        arr_id = p["arrival_station"]
        arr = cfg.stations[arr_id]
        cruise = env._air(station, arr_id) * cfg.tick_min
        outbound = cruise + (dep.takeoff_min + arr.landing_min if cfg.operation_time else 0.0)
        inbound = cruise + (arr.takeoff_min + dep.landing_min if cfg.operation_time else 0.0)
        soc = env._soc_cost(station, arr_id)
        fallback = inbound + soc / arr.charge_soc_per_min if cfg.energy else inbound
        need = soc + cfg.reserve_soc if cfg.energy else 0.0
        if need > 1.0 + 1e-9:
            return math.inf
        ready = min((eta + (max(0.0, need - charge) / dep.charge_soc_per_min if cfg.energy else 0.0)
                     for eta, charge in supply[station]), default=fallback)
        cycle = outbound + inbound
        if cfg.energy:
            cycle += soc / arr.charge_soc_per_min + soc / dep.charge_soc_per_min
        if cfg.turnaround:
            cycle += arr.turnaround_min + dep.turnaround_min
        ahead = queued + sum(eta <= access for eta in access_left[station])
        # READY 是就绪与循环时间的近似，不把未来选择或完整资源队列当作已知。
        batches = ahead // cfg.seats
        wait = ready + batches * cycle / max(1, len(supply[station]))
        return max(access, wait) + outbound

    return {p["id"]: min(env.dep, key=lambda s: (score(p, s), s)) for p in obs["routing"]}


def baseline_episode(cfg, requests, rule, rates, denominators, forced=None):
    env = CorridorEnv(cfg)
    obs = env.reset(requests)
    choices = Counter()
    while not env.done:
        actions = baseline_route_actions(env, obs, rule, rates, denominators, forced)
        choices.update(actions.values())
        obs, _, _, _, _ = env.step(actions)
    return {"metrics": env.metrics(), "elapsed_min": env.now * cfg.tick_min,
            "passenger_choices": {s: choices[s] for s in env.dep}}


def cav_direct_episode(cfg, requests):
    """同一 OD 与截止口径；CAV 请求揭示后直达，假定无候车、拥堵与车队约束。"""
    env = CorridorEnv(cfg)
    cutoff = env._ticks(cfg.cutoff_min)
    generated = finished = area = last_finish = 0
    for r in requests:
        release = env._ticks(r.time_min)
        if release > cutoff:
            continue
        drive = env._drive(r.origin, r.destination, cfg.cav_kmh)
        end = release + drive
        generated += 1
        area += min(drive, cutoff - release)
        if end <= cutoff:
            finished += 1
            last_finish = max(last_finish, end)
    full = finished == len(requests)
    cost = area * cfg.tick_min / generated if generated else None
    return {
        "elapsed_min": (last_finish if full else cutoff) * cfg.tick_min,
        "passenger_choices": {},
        "metrics": {
            "generated": generated, "planned": len(requests), "finished": finished,
            "completion": finished / generated if generated else None,
            "all_requests_finished": full, "door_to_door": True,
            "mean_att_min": cost if full else None, "mean_to_landing_min": None,
            "fixed_cutoff_cost_min": cost,
            "phase_min_per_generated": {"cav_drive": cost},
            "aircraft_states": {}, "aircraft_diagnostics": {},
        },
    }


def write_baseline_json(path, payload):
    # 同目录先写临时文件，再提交完整结果，防止中断留下半份 JSON。
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def run_baselines(cfg, args, requests, manifest):
    out = args.baseline_output or (
        Path("workspace/artifacts/uam_corridor_baselines") / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    # 用 JSON 规范化元组和列表，保证 Windows/Linux 重读合同的含义一致。
    contract = json.loads(json.dumps({
        **manifest,
        "protocol": "corridor_baselines_with_cav_v2",
        "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rule_service_rates": args.rule_service_rates,
        "qtti_denominators": args.qtti_denominators,
        "aircraft_return": "lq",
        "cav_direct": {"speed_kmh": cfg.cav_kmh, "wait_min": 0, "distance": "euclidean_km"},
        "arms": [*BASELINE_RULES, *(f"BOTH_{s}" for s in cfg.departures()), CAV_DIRECT],
    }))
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("结果目录已有不同的配置、需求或代码版本；请换 --baseline-output")
    elif any(out.iterdir()):
        raise ValueError("结果目录非空且没有实验合同；请换 --baseline-output")
    else:
        write_baseline_json(manifest_path, contract)

    arms = [(r, None) for r in BASELINE_RULES] + [("FORCED", s) for s in cfg.departures()] + [(CAV_DIRECT, None)]
    rows = []
    for i, (rule, forced) in enumerate(arms, start=1):
        key = contract["arms"][i - 1]
        path = out / f"arm_{i:02d}.json"
        row = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if row.get("status") not in {"COMPLETE", "CENSORED", "INFEASIBLE"}:
            arm_cfg = replace(cfg, initial_fleet={forced: cfg.fleet}) if forced is not None else cfg
            row = {"group": key, "rule": rule, "forced_station": forced,
                   "initial_fleet": {} if rule == CAV_DIRECT else arm_cfg.allocation()}
            if forced is not None and cfg.parking and cfg.fleet > cfg.stations[forced].parking:
                row.update(status="INFEASIBLE", error=f"{forced} 只有 {cfg.stations[forced].parking} 个停车位，"
                           f"不能容纳 {cfg.fleet} 架初始飞机")
            else:
                try:
                    if rule == CAV_DIRECT:
                        result = cav_direct_episode(cfg, requests)
                    else:
                        result = baseline_episode(arm_cfg, requests, rule, args.rule_service_rates,
                                                  args.qtti_denominators, forced)
                    status = "COMPLETE" if result["metrics"]["all_requests_finished"] else "CENSORED"
                    row.update(result, status=status)
                except Exception as exc:
                    row.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
            write_baseline_json(path, row)
        rows.append(row)
        m = row.get("metrics", {})
        print(f"[{i}/{len(arms)}] {key} {row['status']} | "
              + (f"完成 {m['finished']}/{m['generated']} 已生成；"
                 f"门到门 ATT={m['mean_att_min']}；累计人时/人={m['fixed_cutoff_cost_min']}"
                 if m else row["error"]), flush=True)
    fields = ("group", "status", "forced_station", "planned", "generated", "finished", "completion",
              "all_requests_finished", "mean_att_min", "mean_to_landing_min", "mean_awt_min",
              "fixed_cutoff_cost_min", "service_flights", "empty_flights", "elapsed_min",
              "charge_wait_aircraft_min", "turn_wait_aircraft_min",
              *(f"{p}_min_per_generated" for p in BASELINE_PHASES), "cav_drive_min_per_generated", "error")
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            m = row.get("metrics", {})
            phases = m.get("phase_min_per_generated", {})
            diagnostics = m.get("aircraft_diagnostics", {})
            full = m.get("all_requests_finished", False)
            writer.writerow({**row, **m,
                             **{f"{p}_min_per_generated": v for p, v in phases.items()},
                             "mean_awt_min": sum(phases.get(p, 0.0) for p in ("route", "queue", "pad_wait")) if full else None,
                             "service_flights": diagnostics.get("service_flights", 0) if m else None,
                             "empty_flights": diagnostics.get("empty_flights", 0) if m else None,
                             "charge_wait_aircraft_min": diagnostics.get("charge_wait_ticks", 0) * cfg.tick_min if m else None,
                             "turn_wait_aircraft_min": diagnostics.get("turn_wait_ticks", 0) * cfg.tick_min if m else None})
    print(f"结果：{out.resolve()}", flush=True)
    return int(any(row["status"] == "ERROR" for row in rows))


def main(argv=None):
    cfg, args = parse_config(argv)
    canonical = json.dumps(asdict(cfg), sort_keys=True, ensure_ascii=False)
    output = {"config": asdict(cfg), "config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
              "resolved_initial_fleet": cfg.allocation()}
    requests = make_requests(cfg)
    payload = json.dumps([asdict(r) for r in requests], sort_keys=True, ensure_ascii=False)
    output["demand"] = {"count": len(requests), "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                        "last_request_min": max(r.time_min for r in requests)}
    if args.geometry:
        output["geometry_only"] = geometric_reference(cfg, requests)
    if args.save_config:
        with args.save_config.open("x", encoding="utf-8") as stream:
            json.dump(asdict(cfg), stream, ensure_ascii=False, indent=2)
    if args.baselines:
        return run_baselines(cfg, args, requests, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
