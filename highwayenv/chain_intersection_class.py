from __future__ import annotations

import random
from typing import Any

import numpy as np
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

from highwayenv.intersection_class import IntersectionEnv, MultiAgentIntersectionEnv
from src import project_globals


class ChainIntersectionEnv(IntersectionEnv):


    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            : 2,
            : 80,
            : [],
            : True,
        })
        return config

    @property
    def n_intersections(self) -> int:
        return int(self.config.get("n_intersections", 2))

    def _outer_exit_targets(self) -> set[str]:

        N = self.n_intersections
        targets: set[str] = set()
        for i in range(N):
            targets.add(f"I{i}_o0")
            targets.add(f"I{i}_o2")
        targets.add("I0_o1")
        targets.add(f"I{N - 1}_o3")
        return targets

    def _make_road(self) -> None:
        N = self.n_intersections
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5
        left_turn_radius = right_turn_radius + lane_width
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 100
        connector_length = self.config.get("connector_length", 80)

        spacing = 2 * outer_distance + connector_length
        centers = [np.array([i * spacing, 0.0]) for i in range(N)]

        net = RoadNetwork()
        n_lt, c_lt, s_lt = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        for i in range(N):
            prefix = f"I{i}_"
            center = centers[i]


            skip_corners: set[int] = set()
            if i > 0:
                skip_corners.add(1)
            if i < N - 1:
                skip_corners.add(3)

            for corner in range(4):
                angle = np.radians(90 * corner)
                is_horizontal = corner % 2
                priority = 3 if is_horizontal else 1
                rotation = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ])

                o = prefix + "o" + str(corner)
                ir = prefix + "ir" + str(corner)
                il_right = prefix + "il" + str((corner - 1) % 4)
                il_left = prefix + "il" + str((corner + 1) % 4)
                il_str = prefix + "il" + str((corner + 2) % 4)


                if corner not in skip_corners:
                    start = center + rotation @ np.array([lane_width / 2, access_length + outer_distance])
                    end = center + rotation @ np.array([lane_width / 2, outer_distance])
                    net.add_lane(o, ir, StraightLane(start, end, line_types=[s_lt, c_lt],
                                                     priority=priority, speed_limit=10))


                r_center = center + rotation @ np.array([outer_distance, outer_distance])
                net.add_lane(ir, il_right, CircularLane(
                    r_center, right_turn_radius,
                    angle + np.radians(180), angle + np.radians(270),
                    line_types=[n_lt, c_lt], priority=priority, speed_limit=10))


                l_center = center + rotation @ np.array(
                    [-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2])
                net.add_lane(ir, il_left, CircularLane(
                    l_center, left_turn_radius,
                    angle + np.radians(0), angle + np.radians(-90),
                    clockwise=False, line_types=[n_lt, n_lt], priority=priority - 1, speed_limit=10))


                start = center + rotation @ np.array([lane_width / 2, outer_distance])
                end = center + rotation @ np.array([lane_width / 2, -outer_distance])
                net.add_lane(ir, il_str, StraightLane(start, end, line_types=[s_lt, n_lt],
                                                      priority=priority, speed_limit=10))


                exit_corner = (corner - 1) % 4
                if exit_corner not in skip_corners:
                    il_exit = prefix + "il" + str(exit_corner)
                    o_exit = prefix + "o" + str(exit_corner)
                    start = center + rotation @ np.flip(
                        [lane_width / 2, access_length + outer_distance], axis=0)
                    end = center + rotation @ np.flip(
                        [lane_width / 2, outer_distance], axis=0)
                    net.add_lane(il_exit, o_exit,
                                 StraightLane(end, start, line_types=[n_lt, c_lt],
                                              priority=priority, speed_limit=10))


        for i in range(N - 1):
            pA = f"I{i}_"
            pB = f"I{i + 1}_"
            cA = centers[i]
            cB = centers[i + 1]


            a_il3_pos = cA + np.array([outer_distance, lane_width / 2])
            b_ir1_pos = cB + np.array([-outer_distance, lane_width / 2])
            net.add_lane(pA + "il3", pB + "ir1",
                         StraightLane(a_il3_pos, b_ir1_pos, line_types=[s_lt, c_lt],
                                      priority=2, speed_limit=10))


            b_il1_pos = cB + np.array([-outer_distance, -lane_width / 2])
            a_ir3_pos = cA + np.array([outer_distance, -lane_width / 2])
            net.add_lane(pB + "il1", pA + "ir3",
                         StraightLane(b_il1_pos, a_ir3_pos, line_types=[s_lt, c_lt],
                                      priority=2, speed_limit=10))

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        src, dst = vehicle.lane_index[0], vehicle.lane_index[1]
        if dst not in self._outer_exit_targets():
            return False
        if "_il" not in src:
            return False
        return vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance

    def _clear_vehicles(self) -> None:
        targets = self._outer_exit_targets()

        def is_leaving(vehicle):
            src, dst = vehicle.lane_index[0], vehicle.lane_index[1]
            if dst not in targets:
                return False
            if "_il" not in src:
                return False
            return (vehicle.lane.local_coordinates(vehicle.position)[0]
                    >= vehicle.lane.length - 4 * vehicle.LENGTH)

        self.road.vehicles = [
            v for v in self.road.vehicles
            if v in self.controlled_vehicles or not is_leaving(v)
        ]

    def _reset(self) -> None:
        project_globals.after_is_arrived_flags = [False] * len(self.controlled_vehicles)
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])
        if hasattr(self, "arrived_vehicles"):
            self.arrived_vehicles.clear()


        for v in self.controlled_vehicles:
            if v not in self.road.vehicles:
                self.road.vehicles.append(v)

        scenarios = self.config.get("chain_scenarios") or []
        if not scenarios:
            return

        ix = self.config.get("custom_regular_episode_index", None)
        if ix is not None:
            chosen = scenarios[int(ix) % len(scenarios)]
        else:
            chosen = random.choice(scenarios)

        BASE_LONG = 40
        for i, (lane_key, destination, off) in enumerate(chosen["agents"]):
            if i >= len(self.controlled_vehicles):
                break
            vehicle = self.controlled_vehicles[i]
            lane = self.road.network.get_lane(lane_key)
            vehicle.position = np.array(lane.position(BASE_LONG + off, 0))
            vehicle.lane_index = lane_key
            vehicle.target_lane_index = lane_key
            vehicle.heading = lane.heading_at(vehicle.position)
            if hasattr(vehicle, "plan_route_to"):
                vehicle.plan_route_to(destination)
            else:
                vehicle.route = [lane_key]


class MultiAgentChainIntersectionEnv(ChainIntersectionEnv, MultiAgentIntersectionEnv):

    pass
