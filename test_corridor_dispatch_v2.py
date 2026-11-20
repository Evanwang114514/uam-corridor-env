"""只验证调机协议和接口，不跑训练。"""

from collections import Counter
from dataclasses import replace
import unittest

from corridor_dispatch import PriorityCorridor
from corridor_env import Config, Request


class DispatchTests(unittest.TestCase):
    def prepared(self, turnaround):
        cfg = replace(Config(), energy=False, operation_time=False, turnaround=turnaround)
        env = PriorityCorridor(cfg)
        rows = [Request("p1", 0, cfg.stations["V1"].xy, cfg.stations["V2"].xy),
                Request("p3", 0, cfg.stations["V3"].xy, cfg.stations["V2"].xy)]
        env.reset(rows)
        # 两个出发站排队，只有一架空机；在 0.7 分钟变可调。
        plane = env.planes[0]
        env.local[plane.station] -= 1
        env.local["V2"] += 1
        plane.station, plane.state = "V2", "charging"
        env.planes = {plane.id: plane}
        for p, s in zip(rows, ("V1", "V3")):
            env._route(env.passengers[p.id], s)
        env._push(7, env._charged, plane, 1.0)
        env.charge_active["V2"] = 1
        _, reward, done, cutoff, _ = env.step()
        self.assertEqual((env.now, reward, done, cutoff), (10, -2.0, False, False))
        self.assertEqual(env.targets(), ("V1", "V3"))
        self.assertEqual((plane.state, plane.target), ("ready", None))
        return env, plane

    def test_actual_target_follows_either_legal_choice(self):
        for turnaround in (False, True):
            for target in ("V1", "V3"):
                with self.subTest(turnaround=turnaround, target=target):
                    env, plane = self.prepared(turnaround)
                    stat = Counter()
                    env.choose_returns(target, stat)
                    self.assertEqual((plane.state, plane.target), ("reserved", target))
                    self.assertEqual((stat["air_multi_target"], stat["air_requested_executed"]), (1, 1))
                    env.step()
                    self.assertEqual(env.metrics()["aircraft_diagnostics"]["empty_flights"], 1)
                    self.assertEqual(plane.target, target)

    def test_illegal_choice_gets_legacy_projection(self):
        env, plane = self.prepared(False)
        stat = Counter()
        env.choose_returns("V0", stat)
        self.assertEqual(plane.target, "V1")
        self.assertEqual(stat["air_priority_projection"], 1)


if __name__ == "__main__":
    unittest.main()
