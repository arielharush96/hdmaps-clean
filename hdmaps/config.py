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
        "observation": {
            "type": "Kinematics",
            "features": ["x", "y", "vx", "vy"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20],
            },
            "absolute": True,
            "flatten": False,
            "observe_intentions": False,
        },
        "action": {
            "type": "CustomMultiAgentAction",
            "action_config": {"type": "CustomDiscreteAction"},
            "target_speeds": list(TARGET_SPEEDS),
        },
        "collision_reward": COLLISION_REWARD,
        "arrived_reward": ARRIVED_REWARD,
        "normalize_reward": False,
        "starvation_reward": STARVATION_REWARD,
        "high_speed_reward": HIGH_SPEED_REWARD,
        "offroad_terminal": False,
        "duration": 200,
        "initial_vehicle_count": 0,
        "spawn_probability": 0,
        "screen_width": 900,
        "screen_height": 800,
        "centering_position": [0.5, 0.6],
        "scaling": 3.9,
    }
    cfg.update(updates)
    return cfg
