from __future__ import annotations

import numpy as np

from hdmaps.protocol import (
    STAG_CONNECTOR_MAX_M,
    STAG_CONNECTOR_MIN_M,
    STAG_ENTRY_SPACING_M,
    STAG_HORIZON_MAX,
    STAG_HORIZON_MIN,
)
from hdmaps.scenarios.regional import generate_scenario


def connector_for_scale(n_int: int, n_int_max: int = 16) -> int:
    if n_int_max <= 1:
        return STAG_CONNECTOR_MIN_M
    t = (n_int - 1) / (n_int_max - 1)
    return int(round(STAG_CONNECTOR_MIN_M + t * (STAG_CONNECTOR_MAX_M - STAG_CONNECTOR_MIN_M)))


def horizon_for_connector(connector_m: int) -> int:
    span = STAG_CONNECTOR_MAX_M - STAG_CONNECTOR_MIN_M
    t = 0.0 if span <= 0 else (connector_m - STAG_CONNECTOR_MIN_M) / span
    return int(round(STAG_HORIZON_MIN + t * (STAG_HORIZON_MAX - STAG_HORIZON_MIN)))


def generate_staggered_scenario(
    n_int: int,
    n_agents: int,
    rng: np.random.Generator,
    *,
    varied: bool = True,
    entry_spacing: float = STAG_ENTRY_SPACING_M,
) -> dict:
    scenario = generate_scenario(n_int, n_agents, rng, varied=varied)
    connector = connector_for_scale(n_int)
    agents = []
    for k, (lane, dest, off) in enumerate(scenario["agents"]):
        agents.append((lane, dest, float(off) - k * float(entry_spacing)))
    return {
        : agents,
        : [],
        : connector,
        : horizon_for_connector(connector),
        : True,
    }
