from __future__ import annotations

import random

import numpy as np
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork

from highwayenv.intersection_class import (
    IntersectionEnv,
    MultiAgentIntersectionEnv,
    rotate_scenario_clockwise,
)
from src.experiment.scenarios import (
    roundabout_base_scenarios,
    roundabout_conflict_base_scenarios,
    ROUNDABOUT_HELD_OUT_INDICES,
    ROUNDABOUT_CONFLICT_HELD_OUT_INDICES,
)
from src import project_globals


class RoundaboutEnv(IntersectionEnv):


    def _make_road(self) -> None:
        lane_width = AbstractLane.DEFAULT_WIDTH
        radius = 40.0
        access_length = 100.0
        connector_length = 8.0
        speed_limit = 10

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        center = np.array([0.0, 0.0])

        for corner in range(4):
            rot_angle = np.radians(90 * corner)
            rotation = np.array([
                [np.cos(rot_angle), -np.sin(rot_angle)],
                [np.sin(rot_angle),  np.cos(rot_angle)],
            ])

            outward = rotation @ np.array([0.0, 1.0])
            tangent  = rotation @ np.array([1.0, 0.0])

            junction = center + outward * radius

            approach_shift = tangent * (lane_width / 2)
            exit_shift     = -tangent * (lane_width / 2)


            approach_outer = junction + outward * access_length + approach_shift
            approach_inner = junction + approach_shift
            net.add_lane(
                 + str(corner), "ir" + str(corner),
                StraightLane(approach_outer, approach_inner,
                             line_types=[s, c], priority=1, speed_limit=speed_limit),
            )


            prev_corner = (corner - 1) % 4
            ang_start = np.arctan2(outward[1], outward[0])
            ang_end   = ang_start - np.pi / 2

            net.add_lane(
                 + str(corner), "ir" + str(prev_corner),
                CircularLane(
                    center, radius,
                    ang_start, ang_end,
                    clockwise=False,
                    line_types=[n, c], priority=2, speed_limit=speed_limit,
                ),
            )


            exit_conn_start = junction + exit_shift
            exit_conn_end   = junction + outward * connector_length + exit_shift
            net.add_lane(
                 + str(corner), "il" + str(corner),
                StraightLane(exit_conn_start, exit_conn_end,
                             line_types=[n, n], priority=1, speed_limit=speed_limit),
            )


            exit_outer = exit_conn_end + outward * access_length
            net.add_lane(
                 + str(corner), "o" + str(corner),
                StraightLane(exit_conn_end, exit_outer,
                             line_types=[n, c], priority=0, speed_limit=speed_limit),
            )

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _reset(self) -> None:

        project_globals.after_is_arrived_flags = [False] * len(self.controlled_vehicles)

        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])
        if hasattr(self, 'arrived_vehicles'):
            self.arrived_vehicles.clear()

        BASE_LONG = 40

        def rotate_complete_scenario(scenario, rotation):
            rotated_agents = [rotate_scenario_clockwise([agent], rotation)[0] for agent in scenario["agents"]]
            rotated_static = [rotate_scenario_clockwise([static], rotation)[0] for static in scenario["static"]]
            return {"agents": rotated_agents, "static": rotated_static}


        all_regular = []
        for base_scenario in roundabout_base_scenarios:
            all_regular.append(base_scenario)
            for rotation in [1, 2, 3]:
                all_regular.append(rotate_complete_scenario(base_scenario, rotation))


        all_conflict = []
        for base_scenario in roundabout_conflict_base_scenarios:
            all_conflict.append(base_scenario)
            for rotation in [1, 2, 3]:
                all_conflict.append(rotate_complete_scenario(base_scenario, rotation))

        if not all_regular:
            print("[RoundaboutEnv._reset] WARNING: no scenarios loaded!")
            return

        use_held_out = self.config.get("use_held_out_scenarios", False)
        use_conflict_only = self.config.get("use_conflict_scenarios_only", False)
        conflict_ratio = self.config.get("conflict_ratio", 0.0)
        custom_regular = self.config.get("custom_regular_scenarios") or []
        custom_only = bool(self.config.get("custom_regular_only", False))

        active_regular = [s for i, s in enumerate(all_regular)
                          if i not in ROUNDABOUT_HELD_OUT_INDICES]
        active_conflict = [s for i, s in enumerate(all_conflict)
                           if i not in ROUNDABOUT_CONFLICT_HELD_OUT_INDICES]

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
            held_regular = [all_regular[i] for i in sorted(ROUNDABOUT_HELD_OUT_INDICES)
                            if i < len(all_regular)]
            held_conflict = [all_conflict[i] for i in sorted(ROUNDABOUT_CONFLICT_HELD_OUT_INDICES)
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


class MultiAgentRoundaboutEnv(RoundaboutEnv, MultiAgentIntersectionEnv):

    pass
