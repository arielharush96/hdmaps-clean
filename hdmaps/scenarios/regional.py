from __future__ import annotations

import numpy as np

from hdmaps.protocol import SIM_CONNECTOR_M, SIM_JITTER_M, SPAWN_MIN_SEP_M, SPAWN_RETRIES


def spawn_sep_ok(agents, min_sep: float = SPAWN_MIN_SEP_M) -> bool:
    by_lane: dict[tuple, list[float]] = {}
    for lane, _dest, off in agents:
        by_lane.setdefault(tuple(lane) if not isinstance(lane, str) else (lane,), []).append(float(off))
    for offs in by_lane.values():
        offs = sorted(offs)
        for a, b in zip(offs, offs[1:]):
            if abs(b - a) < float(min_sep):
                return False
    return True


def _retry(factory, retries: int = SPAWN_RETRIES):
    last = None
    for _ in range(max(1, int(retries))):
        last = factory()
        if spawn_sep_ok(last["agents"]):
            return last
    return last


def _approach(i: int, corner: int) -> tuple:
    return (f"I{i}_o{corner}", f"I{i}_ir{corner}", 0)


def generate_scenario(
    n_int: int,
    n_agents: int,
    rng: np.random.Generator,
    *,
    jitter: float = SIM_JITTER_M,
    hop_window: int | None = None,
    varied: bool = False,
) -> dict:
    far_east = f"I{n_int - 1}_o3"
    far_west = "I0_o1"
    if hop_window is None:
        hop_window = int(rng.choice([1, 2, 3])) if n_int > 2 else 1
    go_east = bool(rng.random() < 0.5)

    def _jit() -> float:
        return float(rng.uniform(-jitter, jitter))

    def _jit_through() -> float:
        return float(min(rng.uniform(-jitter, jitter), 25.0))

    agents: list[tuple] = []
    if not varied:
        for i in range(n_int):
            agents.append((_approach(i, 0), f"I{i}_o2", _jit()))
            agents.append((_approach(i, 2), f"I{i}_o0", _jit()))
            if n_int == 1:
                if go_east:
                    agents.append((_approach(i, 1), "I0_o3", _jit_through()))
                else:
                    agents.append((_approach(i, 3), "I0_o1", _jit_through()))
                continue
            if go_east:
                is_sink = i == n_int - 1
                if not is_sink:
                    j = min(i + hop_window, n_int - 1)
                    dest = far_east if j == n_int - 1 else f"I{j}_o2"
                    origin = _approach(i, 1) if i == 0 else _approach(i, 0)
                    agents.append((origin, dest, _jit_through() - 35.0))
                else:
                    agents.append((_approach(i, 2), f"I{i}_o0", _jit_through() - 35.0))
            else:
                is_sink = i == 0
                if not is_sink:
                    j = max(i - hop_window, 0)
                    dest = far_west if j == 0 else f"I{j}_o0"
                    origin = _approach(i, 3) if i == n_int - 1 else _approach(i, 2)
                    agents.append((origin, dest, _jit_through() - 35.0))
                else:
                    agents.append((_approach(i, 0), f"I{i}_o2", _jit_through() - 35.0))
        return {"agents": agents[:n_agents], "static": [], "connector_length": SIM_CONNECTOR_M}

    ep_go_east = go_east
    counts = [0] * n_int
    remaining = n_agents
    cap = 4 if n_int == 1 else 3
    for i in range(n_int):
        max_here = min(cap, remaining - (n_int - 1 - i))
        min_here = max(1, remaining - cap * (n_int - 1 - i))
        max_here = max(min_here, max_here)
        c = int(rng.integers(min_here, max_here + 1))
        counts[i] = c
        remaining -= c
    while remaining > 0:
        for i in range(n_int):
            if counts[i] < cap and remaining > 0:
                counts[i] += 1
                remaining -= 1
        if remaining > 0 and all(c >= cap for c in counts):
            break

    for i in range(n_int):
        k = counts[i]
        car1_south = rng.random() < 0.5
        if car1_south:
            agents.append((_approach(i, 0), f"I{i}_o2", _jit()))
        else:
            agents.append((_approach(i, 2), f"I{i}_o0", _jit()))
        if k >= 2:
            use_through = (rng.random() < 0.5) and (n_int > 1)
            if not use_through:
                if car1_south:
                    agents.append((_approach(i, 2), f"I{i}_o0", _jit()))
                else:
                    agents.append((_approach(i, 0), f"I{i}_o2", _jit()))
            else:
                car_hop = int(rng.integers(1, max(2, min(4, n_int))))
                if ep_go_east:
                    j = min(i + car_hop, n_int - 1)
                    dest = far_east if j == n_int - 1 else f"I{j}_o2"
                    origin = _approach(i, 1) if i == 0 else _approach(i, 0)
                    agents.append((origin, dest, _jit_through() - 35.0))
                else:
                    j = max(i - car_hop, 0)
                    dest = far_west if j == 0 else f"I{j}_o0"
                    origin = _approach(i, 3) if i == n_int - 1 else _approach(i, 2)
                    agents.append((origin, dest, _jit_through() - 35.0))
        if k >= 3:
            if car1_south:
                agents.append((_approach(i, 0), f"I{i}_o2", -60.0 + float(rng.uniform(-5, 5))))
            else:
                agents.append((_approach(i, 2), f"I{i}_o0", -60.0 + float(rng.uniform(-5, 5))))
        if k >= 4:
            if car1_south:
                agents.append((_approach(i, 2), f"I{i}_o0", -60.0 + float(rng.uniform(-5, 5))))
            else:
                agents.append((_approach(i, 0), f"I{i}_o2", -60.0 + float(rng.uniform(-5, 5))))
    return {"agents": agents[:n_agents], "static": [], "connector_length": SIM_CONNECTOR_M}


def generate_suite(n_int: int, n_agents: int, n_episodes: int, rng: np.random.Generator, *, varied: bool) -> list[dict]:
    return [
        _retry(lambda: generate_scenario(n_int, n_agents, rng, varied=varied))
        for _ in range(n_episodes)
    ]


def generate_single_intersection(n_agents: int, rng: np.random.Generator, *, mixed: bool = False, jitter: float = SIM_JITTER_M) -> dict:
    rot = int(rng.integers(0, 4))

    def _lane(corner: int) -> tuple:
        c = (int(corner) + rot) % 4
        return (f"I0_o{c}", f"I0_ir{c}", 0)

    def _dest(corner: int) -> str:
        c = (int(corner) + rot) % 4
        return f"I0_o{c}"

    def _jit() -> float:
        return float(rng.uniform(-jitter, jitter))

    n_agents = max(1, int(n_agents))
    agents: list[tuple] = []
    if mixed:
        corners = [0, 2, 1, 3]
        dests = [2, 3, 0, 1]
        for i in range(n_agents):
            off = _jit() if i < 3 else -60.0 + float(rng.uniform(-5, 5))
            agents.append((_lane(corners[i % 4]), _dest(dests[i % 4]), off))
    else:
        agents.append((_lane(0), _dest(2), _jit()))
        if n_agents >= 2:
            agents.append((_lane(2), _dest(0), _jit()))
        if n_agents >= 3:
            agents.append((_lane(1), _dest(3), min(_jit(), 25.0)))
        if n_agents >= 4:
            agents.append((_lane(0), _dest(2), -60.0 + float(rng.uniform(-5, 5))))
    return {"agents": agents[:n_agents], "static": [], "connector_length": SIM_CONNECTOR_M}
