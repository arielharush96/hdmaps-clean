from __future__ import annotations

import gymnasium as gym
import numpy as np

from src import project_globals
from hdmaps.config import full_env_config
from hdmaps.hierarchy.zones import intersection_spacing, zone_from_x
from hdmaps.protocol import LMS_PER_INTERSECTION, TARGET_SPEEDS
from highwayenv.utils import register_chain_intersection_env

register_chain_intersection_env()


class Corridor:
    def __init__(
        self,
        n_int: int,
        n_agents: int,
        *,
        connector_length: int = 150,
        duration: int = 200,
        target_speeds=TARGET_SPEEDS,
    ):
        self.n_int = int(n_int)
        self.n_agents = int(n_agents)
        self.n_lms = self.n_int * LMS_PER_INTERSECTION
        self.connector_length = int(connector_length)
        self.duration = int(duration)
        controlled = {}
        for i in range(self.n_agents):
            controlled[f"car{i + 1}"] = {
                : ("I0_o0", "I0_ir0", 0),
                : "I0_o2",
                : 5,
                : {"longitudinal": 40, "lateral": 0},
                : [0, 204, 0],
            }
        cfg = full_env_config(
            {
                : controlled,
                : {},
                : self.n_int,
                : self.connector_length,
                : [],
                : True,
                : self.duration,
                : 1,
                : 0,
            }
        )
        cfg["action"]["target_speeds"] = list(target_speeds)
        self.env = gym.make("RELchain-intersection-v0", render_mode=None, config=cfg)
        self.done = False
        self.crashed = False

    def _inner(self):
        env = self.env
        while hasattr(env, "env") and not hasattr(env, "controlled_vehicles"):
            env = env.env
        return env

    def spacing(self) -> float:
        return intersection_spacing(self.connector_length)

    def read_state(self) -> np.ndarray:
        inner = self._inner()
        out = []
        for v in inner.controlled_vehicles:
            if getattr(v, "is_arrived", False):
                out.append([0.0, 0.0, 0.0, 0.0])
                continue
            vel = v.velocity if hasattr(v, "velocity") else np.zeros(2)
            out.append([float(v.position[0]), float(v.position[1]), float(vel[0]), float(vel[1])])
        return np.asarray(out, dtype=np.float32)

    def positions(self) -> np.ndarray:
        inner = self._inner()
        return np.asarray(
            [[float(v.position[0]), float(v.position[1])] for v in inner.controlled_vehicles],
            dtype=np.float32,
        )

    def agent_zones(self) -> list[int]:
        inner = self._inner()
        spacing = self.spacing()
        return [zone_from_x(float(v.position[0]), self.n_int, spacing) for v in inner.controlled_vehicles]

    def lm_groups(self) -> list[list[int]]:
        zones = self.agent_zones()
        zone_agents: dict[int, list[int]] = {}
        for idx, z in enumerate(zones):
            zone_agents.setdefault(z, []).append(idx)
        groups: list[list[int]] = [[] for _ in range(self.n_lms)]
        for z, ags in zone_agents.items():
            base = z * LMS_PER_INTERSECTION
            for j, a in enumerate(ags):
                lm_idx = base + (j % LMS_PER_INTERSECTION)
                if lm_idx < self.n_lms:
                    groups[lm_idx].append(a)
        return groups

    def reset(self, scenario: dict) -> np.ndarray:
        inner = self._inner()
        cl = scenario.get("connector_length")
        if cl is not None:
            self.connector_length = int(cl)
            inner.config["connector_length"] = int(cl)
        inner.config["chain_scenarios"] = [scenario]
        inner.config["chain_scenarios_only"] = True
        project_globals.after_is_arrived_flags = [False] * self.n_agents
        self.env.reset()
        self.done = False
        self.crashed = False
        return self.read_state()

    def step(self, actions: list[int]) -> np.ndarray:
        if self.done:
            return self.read_state()
        project_globals.after_is_arrived_flags = project_globals.after_is_arrived_flags[: self.n_agents]
        _, _, done, truncated, _ = self.env.step(tuple(int(a) for a in actions))
        inner = self._inner()
        if any(getattr(v, "crashed", False) for v in inner.controlled_vehicles):
            self.crashed = True
        if done or truncated:
            self.done = True
        return self.read_state()

    def n_arrived(self) -> int:
        return int(np.sum(self.arrived_mask()))

    def arrived_mask(self) -> np.ndarray:
        return np.asarray(
            [bool(getattr(v, "is_arrived", False)) for v in self._inner().controlled_vehicles],
            dtype=bool,
        )

    def crashed_mask(self) -> np.ndarray:
        return np.asarray(
            [bool(getattr(v, "crashed", False)) for v in self._inner().controlled_vehicles],
            dtype=bool,
        )

    def speeds(self) -> np.ndarray:
        inner = self._inner()
        out = []
        for v in inner.controlled_vehicles:
            vel = v.velocity if hasattr(v, "velocity") else np.zeros(2)
            out.append(float(np.linalg.norm(vel)))
        return np.asarray(out, dtype=np.float32)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
