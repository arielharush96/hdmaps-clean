from __future__ import annotations

import numpy as np

from hdmaps.hierarchy.packing import pack_children, pack_local, pack_mixed
from hdmaps.hierarchy.tree import build_fanout_tree
from hdmaps.protocol import EMBEDDING_DIM


def empty_plans(levels: list[list[dict]]) -> list[dict[int, np.ndarray]]:
    zero = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    return [{node["idx"]: zero.copy() for node in level} for level in levels]


def _parent_emb(plans, level: int, parent_idx) -> np.ndarray:
    zero = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    if parent_idx is None or level + 1 >= len(plans):
        return zero
    return np.asarray(plans[level + 1].get(int(parent_idx), zero), dtype=np.float32).reshape(-1)[:EMBEDDING_DIM]


def _agents_of_child(live, records, ell, child_idx):
    if ell == 1:
        return list(live[child_idx])
    for rec in records:
        if rec.get("tree_level") == ell - 1 and rec.get("tree_idx") == child_idx:
            return list(rec["agents"])
    return []


def run_fanout_hierarchy(
    master,
    groups: list[list[int]],
    state: np.ndarray,
    zones: list[int],
    spacing: float,
    prev_plans: list[dict[int, np.ndarray]] | None,
    *,
    deterministic: bool,
    permute: bool = False,
    rng=None,
    mixed_raw_group: list[int] | None = None,
):
    mixed = list(mixed_raw_group) if mixed_raw_group else None
    live = [g for g in groups if g and g != mixed]
    if not live and mixed:
        live = [mixed]
        mixed = None
    levels = build_fanout_tree(len(live))
    zero = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    if mixed and levels and len(levels) == 1:
        for node in levels[0]:
            node["parent"] = 0
        levels.append([{"idx": 0, "children": [n["idx"] for n in levels[0]], "parent": None}])
    if not levels:
        return [zero.copy() for _ in groups], [], [], levels
    if prev_plans is None or len(prev_plans) != len(levels):
        prev_plans = empty_plans(levels)
    new_plans = empty_plans(levels)
    records: list[dict] = []

    packed_lm = []
    for i, grp in enumerate(live):
        parent_e = _parent_emb(prev_plans, 0, levels[0][i]["parent"])
        shifted = state[grp].copy()
        shifted[:, 0] -= float(zones[grp[0]]) * float(spacing)
        packed_lm.append(pack_local(parent_e, shifted, center_x=0.0, permute=permute, rng=rng))
    acts, vals, lps = master.act(packed_lm, deterministic=deterministic)
    raw = np.asarray(master.last_raw, dtype=np.float32)
    lm_embs = []
    for i, grp in enumerate(live):
        emb = np.asarray(acts[i], dtype=np.float32).reshape(-1)[:EMBEDDING_DIM]
        lm_embs.append(emb)
        new_plans[0][levels[0][i]["idx"]] = emb
        records.append(
            {
                : "lm",
                : 0,
                : i,
                : np.asarray(packed_lm[i], dtype=np.float32),
                : np.asarray(raw[i], dtype=np.float32).reshape(-1)[:EMBEDDING_DIM],
                : emb,
                : float(vals[i]),
                : float(lps[i]),
                : list(grp),
            }
        )
    child_embs = lm_embs

    for ell in range(1, len(levels)):
        packed = []
        agent_sets = []
        for node in levels[ell]:
            parent_e = _parent_emb(prev_plans, ell, node["parent"])
            kids = [child_embs[c] for c in node["children"]]
            sub = []
            for c in node["children"]:
                sub.extend(_agents_of_child(live, records, ell, c))
            is_root = ell == len(levels) - 1 and node["idx"] == 0
            if mixed and is_root:
                raw = state[mixed].copy()
                raw[:, 0] -= float(zones[mixed[0]]) * float(spacing)
                packed.append(pack_mixed(parent_e, kids, raw, permute=permute, rng=rng))
                sub = list(sub) + list(mixed)
            else:
                packed.append(pack_children(kids, parent_e, permute=permute, rng=rng))
            agent_sets.append(sub)
        acts, vals, lps = master.act(packed, deterministic=deterministic)
        raw = np.asarray(master.last_raw, dtype=np.float32)
        nxt = []
        for j, node in enumerate(levels[ell]):
            emb = np.asarray(acts[j], dtype=np.float32).reshape(-1)[:EMBEDDING_DIM]
            nxt.append(emb)
            new_plans[ell][node["idx"]] = emb
            records.append(
                {
                    : "gm" if ell == len(levels) - 1 else "im",
                    : ell,
                    : node["idx"],
                    : np.asarray(packed[j], dtype=np.float32),
                    : np.asarray(raw[j], dtype=np.float32).reshape(-1)[:EMBEDDING_DIM],
                    : emb,
                    : float(vals[j]),
                    : float(lps[j]),
                    : list(agent_sets[j]),
                }
            )
        child_embs = nxt

    root_emb = child_embs[-1] if child_embs else zero
    aligned = []
    k = 0
    for g in groups:
        if not g:
            aligned.append(zero.copy())
        elif mixed and g == mixed:
            aligned.append(np.asarray(root_emb, dtype=np.float32).reshape(-1)[:EMBEDDING_DIM])
        else:
            aligned.append(lm_embs[k] if k < len(lm_embs) else zero.copy())
            k += 1
    return aligned, new_plans, records, levels
