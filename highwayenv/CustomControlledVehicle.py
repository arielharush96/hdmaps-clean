import copy
from typing import List, Optional, Tuple, Union

import numpy as np

from highway_env import utils
from highway_env.road.road import LaneIndex, Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


class CustomControlledVehicle(ControlledVehicle):


    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: LaneIndex = None,
        target_speed: float = None,
        route: Route = None,
        max_speed: float = 30,
    ):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.max_speed = max_speed

    def act(self, action: Union[dict, str] = None) -> None:

        self.follow_road()
        if action == "FASTER":
            self.target_speed = min(self.max_speed, self.target_speed + self.DELTA_SPEED)
        elif action == "SLOWER":
            self.target_speed = max(5, self.target_speed - self.DELTA_SPEED)
        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                self.position
            ):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                self.position
            ):
                self.target_lane_index = target_lane_index

        action = {
            : self.steering_control(self.target_lane_index),
            : self.speed_control(self.target_speed),
        }
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )

        super().act(action)
