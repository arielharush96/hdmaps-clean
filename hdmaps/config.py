from __future__ import annotations

from hdmaps.protocol import (
    ARRIVED_REWARD,
    COLLISION_REWARD,
    HIGH_SPEED_REWARD,
    STARVATION_REWARD,
    TARGET_SPEEDS,
)


def full_env_config(updates: dict) -> dict:
    cfg = {
        : {
            : "Kinematics",
            : ["x", "y", "vx", "vy"],
            : {
                : [-100, 100],
                : [-100, 100],
                : [-20, 20],
                : [-20, 20],
            },
            : True,
            : False,
            : False,
        },
        : {
            : "CustomMultiAgentAction",
            : {"type": "CustomDiscreteAction"},
            : list(TARGET_SPEEDS),
        },
        : COLLISION_REWARD,
        : ARRIVED_REWARD,
        : False,
        : STARVATION_REWARD,
        : HIGH_SPEED_REWARD,
        : False,
        : 200,
        : 0,
        : 0,
        : 900,
        : 800,
        : [0.5, 0.6],
        : 3.9,
    }
    cfg.update(updates)
    return cfg
