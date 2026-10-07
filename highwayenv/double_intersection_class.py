from __future__ import annotations

import random

import numpy as np
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

from highwayenv.intersection_class import (
    IntersectionEnv,
    MultiAgentIntersectionEnv,
)
from src.experiment.scenarios import (
    double_intersection_base_scenarios,
    double_intersection_conflict_base_scenarios,
    DOUBLE_INTERSECTION_HELD_OUT_INDICES,
    DOUBLE_INTERSECTION_EXCLUDED_INDICES,
    DOUBLE_INTERSECTION_CONFLICT_HELD_OUT_INDICES,
)
from src import project_globals


class DoubleIntersectionEnv(IntersectionEnv):


    OUTER_EXIT_TARGETS = {
        , "A_o1", "A_o2",
        , "B_o2", "B_o3",
    }

    def _make_road(self) -> None:
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5
        left_turn_radius = right_turn_radius + lane_width
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 100

        connector_length = self.config.get("connector_length", 80)

        half_sep = outer_distance + connector_length / 2
        center_a = np.array([-half_sep, 0.0])
        center_b = np.array([+half_sep, 0.0])

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        def build_intersection(prefix, center, skip_exit=None, skip_approach=None):
            for corner in range(4):
                angle = np.radians(90 * corner)
                is_horizontal = corner % 2
                priority = 3 if is_horizontal else 1
                rotation = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)],
                ])

                o  = prefix + "o"  + str(corner)
                ir = prefix + "ir" + str(corner)
                il_right = prefix + "il" + str((corner - 1) % 4)
                il_left  = prefix + "il" + str((corner + 1) % 4)
                il_str   = prefix + "il" + str((corner + 2) % 4)

                if corner != skip_approach:
                    start = center + rotation @ np.array([lane_width / 2, access_length + outer_distance])
                    end   = center + rotation @ np.array([lane_width / 2, outer_distance])
                    net.add_lane(o, ir, StraightLane(start, end, line_types=[s, c],
                                                     priority=priority, speed_limit=10))

                r_center = center + rotation @ np.array([outer_distance, outer_distance])
                net.add_lane(ir, il_right, CircularLane(
                    r_center, right_turn_radius,
                    angle + np.radians(180), angle + np.radians(270),
                    line_types=[n, c], priority=priority, speed_limit=10))

                l_center = center + rotation @ np.array(
                    [-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2])
                net.add_lane(ir, il_left, CircularLane(
                    l_center, left_turn_radius,
                    angle + np.radians(0), angle + np.radians(-90),
                    clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))

                start = center + rotation @ np.array([lane_width / 2, outer_distance])
                end   = center + rotation @ np.array([lane_width / 2, -outer_distance])
                net.add_lane(ir, il_str, StraightLane(start, end, line_types=[s, n],
                                                      priority=priority, speed_limit=10))

                exit_corner = (corner - 1) % 4
                if exit_corner != skip_exit:
                    il_exit = prefix + "il" + str(exit_corner)
                    o_exit  = prefix + "o"  + str(exit_corner)
                    start = center + rotation @ np.flip(
                        [lane_width / 2, access_length + outer_distance], axis=0)
                    end   = center + rotation @ np.flip(
                        [lane_width / 2, outer_distance], axis=0)
                    net.add_lane(il_exit, o_exit,
                                 StraightLane(end, start, line_types=[n, c],
                                              priority=priority, speed_limit=10))

        build_intersection("A_", center_a, skip_exit=3, skip_approach=3)
        build_intersection("B_", center_b, skip_exit=1, skip_approach=1)


        a_il3_pos = center_a + np.array([outer_distance, lane_width / 2])

        angle_west = np.radians(90)
        rot_west = np.array([
            [np.cos(angle_west), -np.sin(angle_west)],
            [np.sin(angle_west),  np.cos(angle_west)],
        ])
        b_ir1_pos = center_b + rot_west @ np.array([lane_width / 2, outer_distance])

        net.add_lane("A_il3", "B_ir1",
                     StraightLane(a_il3_pos, b_ir1_pos, line_types=[s, c],
                                  priority=2, speed_limit=10))

        b_il1_pos = center_b + np.array([-outer_distance, -lane_width / 2])

        angle_east = np.radians(270)
        rot_east = np.array([
            [np.cos(angle_east), -np.sin(angle_east)],
            [np.sin(angle_east),  np.cos(angle_east)],
        ])
        a_ir3_pos = center_a + rot_east @ np.array([lane_width / 2, outer_distance])

        net.add_lane("B_il1", "A_ir3",
                     StraightLane(b_il1_pos, a_ir3_pos, line_types=[s, c],
                                  priority=2, speed_limit=10))

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def has_arrived(self, vehicle, exit_distance: float = 25) -> bool:
        src, dst = vehicle.lane_index[0], vehicle.lane_index[1]
        if dst not in self.OUTER_EXIT_TARGETS:
            return False
        if "_il" not in src:
            return False
        return vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance

    def _clear_vehicles(self) -> None:
        def is_leaving(vehicle):
            src, dst = vehicle.lane_index[0], vehicle.lane_index[1]
            if dst not in self.OUTER_EXIT_TARGETS:
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
        if hasattr(self, 'arrived_vehicles'):
            self.arrived_vehicles.clear()

        BASE_LONG = 40


        all_regular = list(double_intersection_base_scenarios)

        all_conflict = list(double_intersection_conflict_base_scenarios)

        if not all_regular:
            print("[DoubleIntersectionEnv._reset] WARNING: no scenarios loaded!")
            return

        use_held_out = self.config.get("use_held_out_scenarios", False)
        use_conflict_only = self.config.get("use_conflict_scenarios_only", False)
        conflict_ratio = self.config.get("conflict_ratio", 0.0)
        custom_regular = self.config.get("custom_regular_scenarios") or []
        custom_only = bool(self.config.get("custom_regular_only", False))

        _excluded_regular = DOUBLE_INTERSECTION_HELD_OUT_INDICES | DOUBLE_INTERSECTION_EXCLUDED_INDICES
        active_regular = [s for i, s in enumerate(all_regular) if i not in _excluded_regular]
        active_conflict = [s for i, s in enumerate(all_conflict)
                           if i not in DOUBLE_INTERSECTION_CONFLICT_HELD_OUT_INDICES]

        chosen_scenario = None
        force_idx = self.config.get("force_scenario_index", None)
        if force_idx is not None:
            chosen_scenario = all_regular[int(force_idx) % len(all_regular)]
        elif custom_regular and custom_only:
            ix_pick = self.config.get("custom_regular_episode_index", None)
            if ix_pick is not None:
                chosen_scenario = custom_regular[int(ix_pick) % len(custom_regular)]
            else:
                chosen_scenario = random.choice(custom_regular)
        elif use_held_out:
            held_regular = [all_regular[i] for i in sorted(DOUBLE_INTERSECTION_HELD_OUT_INDICES)
                            if i < len(all_regular)]
            held_conflict = [all_conflict[i] for i in sorted(DOUBLE_INTERSECTION_CONFLICT_HELD_OUT_INDICES)
                             if i < len(all_conflict)]
            held_out_pool = held_regular + held_conflict
            chosen_scenario = random.choice(held_out_pool) if held_out_pool else random.choice(all_regular)
        elif use_conflict_only:
            chosen_scenario = random.choice(active_conflict if active_conflict else all_conflict)
        elif conflict_ratio > 0.0 and active_conflict and random.random() < conflict_ratio:
            chosen_scenario = random.choice(active_conflict)
        else:
            chosen_scenario = random.choice(active_regular)
        assert chosen_scenario is not None

        all_scenarios = all_regular + all_conflict
        try:
            self.last_scenario_index = all_scenarios.index(chosen_scenario)
        except ValueError:
            self.last_scenario_index = -1

        for i, (lane_key, destination, off) in enumerate(chosen_scenario["agents"]):
            vehicle = self.controlled_vehicles[i]
            lane = self.road.network.get_lane(lane_key)
            vehicle.position = np.array(lane.position(BASE_LONG + off, 0))
            vehicle.lane_index = lane_key
            vehicle.target_lane_index = lane_key
            vehicle.heading = lane.heading_at(vehicle.position)
            if hasattr(vehicle, 'plan_route_to'):
                vehicle.plan_route_to(destination)
            else:
                vehicle.route = [lane_key]

        all_vehicles = self.road.vehicles
        controlled_count = len(self.controlled_vehicles)

        safe_static_scenario = []
        for lane_key, destination, off in chosen_scenario["static"]:
            lane = self.road.network.get_lane(lane_key)
            position = np.array(lane.position(BASE_LONG + off, 0))

            too_close = False
            for cv in self.controlled_vehicles:
                if np.linalg.norm(cv.position - position) < 10:
                    too_close = True
                    break
            for existing_key, _, existing_off in safe_static_scenario:
                existing_lane = self.road.network.get_lane(existing_key)
                if np.linalg.norm(np.array(existing_lane.position(BASE_LONG + existing_off, 0)) - position) < 10:
                    too_close = True
                    break

            if not too_close:
                safe_static_scenario.append((lane_key, destination, off))

        for i in range(controlled_count, min(len(all_vehicles), controlled_count + len(safe_static_scenario))):
            static_index = i - controlled_count
            if static_index >= len(safe_static_scenario):
                break
            lane_key, destination, off = safe_static_scenario[static_index]
            vehicle = all_vehicles[i]
            lane = self.road.network.get_lane(lane_key)
            vehicle.position = np.array(lane.position(BASE_LONG + off, 0))
            vehicle.lane_index = lane_key
            vehicle.target_lane_index = lane_key
            vehicle.heading = lane.heading_at(vehicle.position)
            if hasattr(vehicle, 'plan_route_to'):
                vehicle.plan_route_to(destination)
            else:
                vehicle.route = [lane_key]

        pass


class MultiAgentDoubleIntersectionEnv(DoubleIntersectionEnv, MultiAgentIntersectionEnv):

    pass
