from __future__ import annotations

import random

import numpy as np
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import observation_factory
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

from src.experiment.experiment_config import Experiment
from src.experiment.scenarios import (
    base_complete_scenarios_6_cars,
    conflict_base_scenarios,
    EXCLUDED_SCENARIO_INDICES,
    HELD_OUT_SCENARIO_INDICES,
    CONFLICT_HELD_OUT_INDICES,
)
from highwayenv.CustomControlledVehicle import CustomControlledVehicle
from highwayenv.custom_action import action_factory
from src import project_globals


def rotate_lane_id(lane_id, rotation):

    if len(lane_id) < 2 or not lane_id[-1].isdigit():
        return lane_id

    prefix = lane_id[:-1]
    direction = int(lane_id[-1])

    new_direction = (direction + rotation) % 4
    return prefix + str(new_direction)


def rotate_scenario_clockwise(scenario, rotation):

    rotated_scenario = []
    for lane_key, destination, offset in scenario:

        if isinstance(lane_key, tuple):
            rotated_lane = tuple(
                rotate_lane_id(part, rotation) if isinstance(part, str) else part
                for part in lane_key
            )
        else:
            rotated_lane = rotate_lane_id(lane_key, rotation)


        rotated_dest = rotate_lane_id(destination, rotation)

        rotated_scenario.append((rotated_lane, rotated_dest, offset))

    return rotated_scenario


class IntersectionEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        return config

    def define_spaces(self) -> None:

        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()


    def _reward(self, action: int) -> float:

        active_vehicles = [
            vehicle for vehicle, flag in zip(self.controlled_vehicles, project_globals.after_is_arrived_flags)
            if not flag
        ]

        if not active_vehicles:
            return 0.0

        min_reward = min(self._agent_reward(vehicle) for vehicle in active_vehicles)

        if not getattr(self, '_reward_printed_this_step', False):
            self._reward_printed_this_step = True

        return min_reward


    def _rewards(self, action: int) -> dict[str, float]:

        agents_rewards = [
            self._agent_rewards(vehicle) for vehicle in self.controlled_vehicles
        ]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards)
                  / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }

    def _agent_reward(self, vehicle: Vehicle) -> float:

        if self.has_arrived(vehicle):
            return self.config["arrived_reward"]
        elif vehicle.crashed:
            return self.config["collision_reward"]
        elif vehicle.speed > Experiment.FIXED_THROTTLE:
            return self.config.get("high_speed_reward", Experiment.HIGH_SPEED_REWARD)
        else:
            return self.config.get("starvation_reward", Experiment.STARVATION_REWARD)

    def _agent_rewards(self, vehicle: Vehicle) -> dict[str, float]:

        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": 1,
            "arrived_reward": self.has_arrived(vehicle),
            "starvation_reward": True
        }

    def update_after_is_arrived_flags(self):
        for i, vehicle in enumerate(self.controlled_vehicles):
            if hasattr(vehicle, 'is_arrived') and vehicle.is_arrived:
                project_globals.after_is_arrived_flags[i] = True

    def _is_terminated(self) -> bool:

        self.update_after_is_arrived_flags()


        if not hasattr(self, 'arrived_vehicles'):
            self.arrived_vehicles = set()


        for vehicle in self.controlled_vehicles:
            if self.has_arrived(vehicle) and vehicle not in self.arrived_vehicles:

                vehicle.position = np.array(
                    [0.0 + len(self.arrived_vehicles) * 20.0, 1000 + len(self.arrived_vehicles) * 20.0],
                    dtype=np.float64)
                vehicle.target_speed = 0.0
                vehicle.MAX_SPEED = 0
                vehicle.is_arrived = True
                self.arrived_vehicles.add(vehicle)

        arrived_count = len(self.arrived_vehicles)


        if arrived_count == len(self.controlled_vehicles):
            return True

        return any(vehicle.crashed for vehicle in self.controlled_vehicles)

    def get_observation(self):

        self.update_after_is_arrived_flags()

        obs = []

        for vehicle in self.controlled_vehicles:
            if hasattr(vehicle, 'is_arrived') and vehicle.is_arrived:
                obs.append([0.0, 0.0, 0.0, 0.0])
            else:
                obs.append([
                    vehicle.position[0],
                    vehicle.position[1],
                    vehicle.velocity[0],
                    vehicle.velocity[1]
                ])

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:

        return vehicle.crashed or self.has_arrived(vehicle)

    def _is_truncated(self) -> bool:

        return self.time >= self.config["duration"]

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(
            self._agent_reward(vehicle) for vehicle in self.controlled_vehicles
        )
        info["agents_terminated"] = tuple(
            self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles
        )
        return info

    def _reset(self) -> None:


        project_globals.after_is_arrived_flags = [False] * len(self.controlled_vehicles)

        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])
        if hasattr(self, 'arrived_vehicles'):
            self.arrived_vehicles.clear()

        BASE_LONG = 40

        base_complete_scenarios = base_complete_scenarios_6_cars


        def rotate_complete_scenario(scenario, rotation):
            rotated_agents = [rotate_scenario_clockwise([agent], rotation)[0] for agent in scenario["agents"]]
            rotated_static = [rotate_scenario_clockwise([static], rotation)[0] for static in scenario["static"]]
            return {
                "agents": rotated_agents,
                "static": rotated_static
            }


        all_scenarios = []
        for base_scenario in base_complete_scenarios:
            all_scenarios.append(base_scenario)
            for rotation in [1, 2, 3]:
                rotated_scenario = rotate_complete_scenario(base_scenario, rotation)
                all_scenarios.append(rotated_scenario)


        all_conflict = []
        for base_scenario in conflict_base_scenarios:
            all_conflict.append(base_scenario)
            for rotation in [1, 2, 3]:
                all_conflict.append(rotate_complete_scenario(base_scenario, rotation))


        _training_excluded = EXCLUDED_SCENARIO_INDICES | HELD_OUT_SCENARIO_INDICES
        active_regular = [
            s for i, s in enumerate(all_scenarios)
            if i not in _training_excluded
        ]
        active_conflict = [
            s for i, s in enumerate(all_conflict)
            if i not in CONFLICT_HELD_OUT_INDICES
        ]


        force_idx = self.config.get("force_scenario_index", None)
        custom_regular = self.config.get("custom_regular_scenarios") or []
        custom_only = bool(self.config.get("custom_regular_only", False))
        use_held_out = self.config.get("use_held_out_scenarios", False)
        use_conflict_only = self.config.get("use_conflict_scenarios_only", False)
        conflict_ratio = self.config.get("conflict_ratio", 0.0)
        if force_idx is not None:
            chosen_scenario = all_scenarios[int(force_idx) % len(all_scenarios)]
        elif custom_regular and custom_only:
            ix_pick = self.config.get("custom_regular_episode_index", None)
            if ix_pick is not None:
                chosen_scenario = custom_regular[int(ix_pick) % len(custom_regular)]
            else:
                chosen_scenario = random.choice(custom_regular)
        elif use_held_out:
            held_regular = [all_scenarios[i] for i in sorted(HELD_OUT_SCENARIO_INDICES)]
            held_conflict = [all_conflict[i] for i in sorted(CONFLICT_HELD_OUT_INDICES)
                             if i < len(all_conflict)]
            held_out_pool = held_regular + held_conflict
            chosen_scenario = random.choice(held_out_pool)
        elif use_conflict_only:
            chosen_scenario = random.choice(active_conflict)
        elif conflict_ratio > 0.0 and active_conflict and random.random() < conflict_ratio:
            chosen_scenario = random.choice(active_conflict)
        else:
            chosen_scenario = random.choice(active_regular)


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
                vehicle.route = [lane_key, (destination, 'ir' + destination[1:], 0)]


        all_vehicles = self.road.vehicles
        controlled_count = len(self.controlled_vehicles)


        safe_static_scenario = []
        for i, (lane_key, destination, off) in enumerate(chosen_scenario["static"]):
            lane = self.road.network.get_lane(lane_key)
            position = np.array(lane.position(BASE_LONG + off, 0))


            too_close_to_agent = False
            for controlled_vehicle in self.controlled_vehicles:
                if np.linalg.norm(controlled_vehicle.position - position) < 10:
                    too_close_to_agent = True
                    break


            too_close_to_static = False
            for existing_lane_key, existing_dest, existing_off in safe_static_scenario:
                existing_lane = self.road.network.get_lane(existing_lane_key)
                existing_position = np.array(existing_lane.position(BASE_LONG + existing_off, 0))

                if np.linalg.norm(existing_position - position) < 10:
                    too_close_to_static = True
                    break


            if not too_close_to_agent and not too_close_to_static:
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
                vehicle.route = [lane_key, (destination, 'ir' + destination[1:], 0)]


        is_conflict = chosen_scenario in all_conflict
        if custom_regular and custom_only:
            scenario_index = -5
            base_scenario_num = -5
            rotation_num = 0
            self.last_scenario_pool = "regular"
        elif is_conflict:
            self.last_scenario_pool = "conflict"
            try:
                scenario_index = all_conflict.index(chosen_scenario)
            except ValueError:
                scenario_index = -1
            pool_size = len(active_conflict)
            tag = "CONFLICT"
        else:
            self.last_scenario_pool = "regular"
            try:
                scenario_index = all_scenarios.index(chosen_scenario)
            except ValueError:
                scenario_index = -1
            pool_size = len(active_regular)
            tag = "regular"

        base_scenario_num = scenario_index // 4 if scenario_index >= 0 else -1
        rotation_num = scenario_index % 4 if scenario_index >= 0 else 0
        rotation_names = ["Original", "90° CW", "180° CW", "270° CW"]


        self.last_scenario_index = int(scenario_index)
        self.last_base_scenario = int(base_scenario_num)
        self.last_rotation = int(rotation_num)
        self.last_static_placed = int(len(safe_static_scenario))
        self.last_static_total = int(len(chosen_scenario["static"]))

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._reward_printed_this_step = False
        obs, reward, terminated, truncated, info = super().step(action)
        self._clear_vehicles()
        return obs, reward, terminated, truncated, info

    def _make_road(self) -> None:

        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5
        left_turn_radius = right_turn_radius + lane_width
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )

            start = rotation @ np.array(
                [lane_width / 2, access_length + outer_distance]
            )
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane(
                "o" + str(corner),
                "ir" + str(corner),
                StraightLane(
                    start, end, line_types=[s, c], priority=priority, speed_limit=10
                ),
            )

            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner - 1) % 4),
                CircularLane(
                    r_center,
                    right_turn_radius,
                    angle + np.radians(180),
                    angle + np.radians(270),
                    line_types=[n, c],
                    priority=priority,
                    speed_limit=10,
                ),
            )

            l_center = rotation @ (
                np.array(
                    [
                        -left_turn_radius + lane_width / 2,
                        left_turn_radius - lane_width / 2,
                    ]
                )
            )
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 1) % 4),
                CircularLane(
                    l_center,
                    left_turn_radius,
                    angle + np.radians(0),
                    angle + np.radians(-90),
                    clockwise=False,
                    line_types=[n, n],
                    priority=priority - 1,
                    speed_limit=10,
                ),
            )

            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 2) % 4),
                StraightLane(
                    start, end, line_types=[s, n], priority=priority, speed_limit=10
                ),
            )

            start = rotation @ np.flip(
                [lane_width / 2, access_length + outer_distance], axis=0
            )
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane(
                "il" + str((corner - 1) % 4),
                "o" + str((corner - 1) % 4),
                StraightLane(
                    end, start, line_types=[n, c], priority=priority, speed_limit=10
                ),
            )

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:


        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3


        simulation_steps = 3
        for _ in range(simulation_steps):
            [
                (
                    self.road.act(),
                    self.road.step(1 / self.config["simulation_frequency"]),
                )
                for _ in range(self.config["simulation_frequency"])
            ]

        self.controlled_vehicles = []
        controlled_conf = self.config["controlled_cars"]
        other_conf = self.config["static_cars"]


        for other_car in controlled_conf.values():
            lane = self.road.network.get_lane(other_car["start_lane"])
            controlled_vehicle = CustomControlledVehicle(
                self.road,
                lane.position(other_car["init_location"]["longitudinal"], other_car["init_location"]["lateral"]),
                speed=other_car["speed"],
                heading=lane.heading_at(other_car["init_location"]["longitudinal"]),
            )
            controlled_vehicle.color = other_car["color"]
            controlled_vehicle.plan_route_to(other_car["destination"])
            self.road.vehicles.append(controlled_vehicle)
            self.controlled_vehicles.append(controlled_vehicle)


        for other_car in other_conf.values():
            lane = self.road.network.get_lane(other_car["start_lane"])
            other_vehicle = Vehicle(
                self.road,
                lane.position(other_car["init_location"]["longitudinal"], other_car["init_location"]["lateral"]),
                speed=other_car["speed"],
                heading=lane.heading_at(other_car["init_location"]["longitudinal"]),
            )
            self.road.vehicles.append(other_vehicle)

        for v in self.road.vehicles:
            if (
                    v is not controlled_vehicle
                    and np.linalg.norm(v.position - controlled_vehicle.position) < 20
            ):
                self.road.vehicles.remove(v)

    def _spawn_vehicle(
            self,
            longitudinal: float = 0,
            position_deviation: float = 1.0,
            speed_deviation: float = 1.0,
            spawn_probability: float = 0.6,
            go_straight: bool = False,
    ) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            ("o" + str(route[0]), "ir" + str(route[0]), 0),
            longitudinal=(
                    longitudinal + 5 + self.np_random.normal() * position_deviation
            ),
            speed=8 + self.np_random.normal() * speed_deviation,
        )
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = (
            lambda vehicle: "il" in vehicle.lane_index[0]
                            and "o" in vehicle.lane_index[1]
                            and vehicle.lane.local_coordinates(vehicle.position)[0]
                            >= vehicle.lane.length - 4 * vehicle.LENGTH
        )
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles
               or not (is_leaving(vehicle))
        ]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return (
                "il" in vehicle.lane_index[0]
                and "o" in vehicle.lane_index[1]
                and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
        )


class MultiAgentIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                        "lateral": False,
                        "longitudinal": True,
                    },
                },
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {"type": "Kinematics"},
                },
                "controlled_vehicles": 2,
            }
        )
        return config


class ContinuousIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    "features": [
                        "presence",
                        "x",
                        "y",
                        "vx",
                        "vy",
                        "long_off",
                        "lat_off",
                        "ang_off",
                    ],
                },
                "action": {
                    "type": "ContinuousAction",
                    "steering_range": [-np.pi / 3, np.pi / 3],
                    "longitudinal": True,
                    "lateral": True,
                    "dynamical": True,
                },
            }
        )

        return config
