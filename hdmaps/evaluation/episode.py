from __future__ import annotations

import numpy as np

from hdmaps.hierarchy.forward import run_fanout_hierarchy
from hdmaps.hierarchy.packing import agent_obs
from hdmaps.hierarchy.tree import hierarchy_label
from hdmaps.hierarchy.zones import in_radius
from hdmaps.policies.agent import agent_actions
from hdmaps.protocol import DETERMINISTIC_AGENT, DETERMINISTIC_MASTER


def _groups_from_cell(cell):
    zones = cell.agent_zones()
    by: dict[int, list[int]] = {}
    for i, z in enumerate(zones):
        by.setdefault(int(z), []).append(i)
    return [by[z] for z in sorted(by)], zones


def run_simultaneous_episode(cell, scenario, master, agent, *, max_steps: int):
    state = cell.reset(scenario)
    steps = 0
    spacing = cell.spacing()
    prev_plans = None
    groups: list[list[int]] = []
    while steps < max_steps and not cell.done:
        steps += 1
        groups, zones = _groups_from_cell(cell)
        lm_embs, prev_plans, _recs, _levels = run_fanout_hierarchy(
            master,
            groups,
            state,
            zones,
            spacing,
            prev_plans,
            deterministic=DETERMINISTIC_MASTER,
            permute=False,
        )
        obs_list = []
        order = []
        for lm_idx, grp in enumerate(groups):
            for a_idx in grp:
                local = state[a_idx][:4].copy()
                local[0] -= zones[a_idx] * spacing
                obs_list.append(agent_obs(local, lm_embs[lm_idx]))
                order.append(a_idx)
        acts = agent_actions(agent, obs_list, DETERMINISTIC_AGENT)
        full = [0] * cell.n_agents
        for a_idx, act in zip(order, acts):
            full[a_idx] = int(act)
        state = cell.step(full)
    n_lm = len([g for g in groups if g]) if groups else cell.n_int
    return {
        : 100.0 * cell.n_arrived() / max(1, cell.n_agents),
        : int(cell.crashed),
        : steps,
        : n_lm,
        : hierarchy_label(n_lm),
    }


def run_staggered_episode(cell, scenario, master, agent, *, max_steps: int, coord_radius: float = 80.0):
    state = cell.reset(scenario)
    steps = 0
    spacing = cell.spacing()
    prev_plans = None
    peak_lms = 0
    while steps < max_steps and not cell.done:
        steps += 1
        zones = cell.agent_zones()
        pos = cell.positions()
        inner = cell._inner()
        active = []
        for i, v in enumerate(inner.controlled_vehicles):
            if getattr(v, "is_arrived", False) or getattr(v, "crashed", False):
                continue
            center = (zones[i] * spacing, 0.0)
            if in_radius(pos[i], center, coord_radius):
                active.append(i)
        zone_agents: dict[int, list[int]] = {}
        for i in active:
            zone_agents.setdefault(zones[i], []).append(i)
        groups = [zone_agents[z] for z in sorted(zone_agents)]
        peak_lms = max(peak_lms, len(groups))
        lm_of: dict[int, np.ndarray] = {}
        if groups:
            lm_embs, prev_plans, _recs, _levels = run_fanout_hierarchy(
                master,
                groups,
                state,
                zones,
                spacing,
                prev_plans,
                deterministic=DETERMINISTIC_MASTER,
                permute=False,
            )
            keys = sorted(zone_agents)
            lm_of = {keys[i]: lm_embs[i] for i in range(len(groups))}
        else:
            prev_plans = None
        obs_list = []
        zero = np.zeros(4, dtype=np.float32)
        for i in range(cell.n_agents):
            local = state[i][:4].copy()
            local[0] -= zones[i] * spacing
            emb = lm_of.get(zones[i], zero) if i in active else zero
            obs_list.append(agent_obs(local, emb))
        acts = agent_actions(agent, obs_list, DETERMINISTIC_AGENT)
        state = cell.step([int(a) for a in acts])
    return {
        : 100.0 * cell.n_arrived() / max(1, cell.n_agents),
        : int(cell.crashed),
        : steps,
        : peak_lms,
        : cell.n_lms,
        : hierarchy_label(peak_lms),
    }
