from __future__ import annotations

import numpy as np

from hdmaps.hierarchy.forward import run_fanout_hierarchy
from hdmaps.hierarchy.packing import agent_obs
from hdmaps.policies.agent import agent_act
from hdmaps.protocol import (
    ARRIVED_REWARD,
    COLLISION_REWARD,
    STEP_REWARD,
    TRAIN_MAX_AGENTS_PER_MASTER,
)


def train_lm_groups(cell) -> list[list[int]]:
    zones = cell.agent_zones()
    zone_agents: dict[int, list[int]] = {}
    for idx, z in enumerate(zones):
        zone_agents.setdefault(int(z), []).append(idx)
    groups: list[list[int]] = []
    cap = TRAIN_MAX_AGENTS_PER_MASTER
    for z in sorted(zone_agents):
        ags = zone_agents[z]
        for i in range(0, len(ags), cap):
            chunk = ags[i : i + cap]
            if chunk:
                groups.append(chunk)
    return groups


def worker_reward(idx: int, arrived_now, arrived_prev, crashed) -> float:
    if bool(crashed[idx]):
        return float(COLLISION_REWARD)
    if bool(arrived_now[idx]) and not bool(arrived_prev[idx]):
        return float(ARRIVED_REWARD)
    if not bool(arrived_prev[idx]):
        return float(STEP_REWARD)
    return 0.0


def maximin(agent_ids, rewards: dict[int, float], arrived_prev) -> float:
    vals = [rewards[i] for i in agent_ids if i in rewards and not bool(arrived_prev[i])]
    return float(min(vals)) if vals else 0.0


def _split_mixed(groups: list[list[int]], mixed: bool) -> tuple[list[list[int]], list[int] | None]:
    if not mixed:
        return groups, None
    ids = [i for g in groups for i in g]
    if len(ids) < 3:
        return groups, None
    raw = [ids[-1]]
    raw_set = set(raw)
    live = [[i for i in g if i not in raw_set] for g in groups]
    live = [g for g in live if g]
    if not live:
        return groups, None
    return live + [raw], raw


def run_training_episode(
    cell,
    scenario,
    master,
    agent,
    *,
    max_steps: int,
    rng: np.random.Generator,
    mixed: bool = False,
):
    state = cell.reset(scenario)
    arrived_prev = cell.arrived_mask()
    steps = 0
    spacing = cell.spacing()
    prev_plans = None
    master_streams: dict[str, list[dict]] = {"lm": [], "im": [], "gm": []}
    worker_stream: list[dict] = []
    episode_reward = 0.0

    while steps < max_steps and not cell.done:
        steps += 1
        groups = train_lm_groups(cell)
        zones = cell.agent_zones()
        if not groups:
            groups = [[i for i in range(cell.n_agents)]]
        hier_groups, mixed_raw = _split_mixed(groups, mixed)
        lm_embs, prev_plans, records, _levels = run_fanout_hierarchy(
            master,
            hier_groups,
            state,
            zones,
            spacing,
            prev_plans,
            deterministic=False,
            permute=True,
            rng=rng,
            mixed_raw_group=mixed_raw,
        )
        obs_list = []
        order = []
        for lm_idx, grp in enumerate(hier_groups):
            for a_idx in grp:
                local = state[a_idx][:4].copy()
                local[0] -= zones[a_idx] * spacing
                obs_list.append(agent_obs(local, lm_embs[lm_idx]))
                order.append(a_idx)
        acts, w_val, w_lp = agent_act(agent, obs_list, deterministic=False)
        full = [0] * cell.n_agents
        for a_idx, act in zip(order, acts):
            full[a_idx] = int(act)
        state = cell.step(full)
        arrived_now = cell.arrived_mask()
        crashed = cell.crashed_mask()
        rewards = {i: worker_reward(i, arrived_now, arrived_prev, crashed) for i in range(cell.n_agents)}
        episode_reward += float(sum(rewards.values()))
        for rec in records:
            rec["reward"] = maximin(rec["agents"], rewards, arrived_prev)
            role = rec.get("role", "lm")
            master_streams.setdefault(role, []).append(
                {
                    : rec["obs"],
                    : rec["action"],
                    : rec["value"],
                    : rec["log_prob"],
                    : rec["reward"],
                }
            )
        for j, a_idx in enumerate(order):
            worker_stream.append(
                {
                    : obs_list[j],
                    : int(acts[j]),
                    : float(w_val[j]),
                    : float(w_lp[j]),
                    : rewards[a_idx],
                }
            )
        arrived_prev = arrived_now

    result = {
        : 100.0 * cell.n_arrived() / max(1, cell.n_agents),
        : int(cell.crashed),
        : int(cell.n_arrived()),
        : float(episode_reward),
        : steps,
        : len(groups),
        : cell.n_int,
        : cell.n_agents,
    }
    return result, master_streams, worker_stream
