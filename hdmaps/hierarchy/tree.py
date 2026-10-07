from __future__ import annotations

import math

from hdmaps.protocol import FANOUT


def hierarchy_counts(n_local_masters: int, fanout: int = FANOUT) -> tuple[int, int, int]:
    n_lm = int(n_local_masters)
    if n_lm <= 0:
        return 0, 0, 0
    if n_lm == 1:
        return 1, 0, 0
    n_im = 0 if n_lm <= fanout else int(math.ceil(n_lm / fanout))
    return n_lm, n_im, 1


def hierarchy_label(n_local_masters: int, fanout: int = FANOUT) -> str:
    n_lm, n_im, n_gm = hierarchy_counts(n_local_masters, fanout)
    return f"{n_lm} / {n_im} / {n_gm}"


def chunk_indices(n_items: int, fanout: int = FANOUT) -> list[list[int]]:
    if n_items <= 0:
        return []
    return [list(range(i, min(i + fanout, n_items))) for i in range(0, n_items, fanout)]


def build_fanout_tree(n_leaves: int, fanout: int = FANOUT) -> list[list[dict]]:
    n_leaves = int(n_leaves)
    if n_leaves <= 0:
        return []
    levels: list[list[dict]] = [[{"idx": i, "children": []} for i in range(n_leaves)]]
    while len(levels[-1]) > 1:
        prev = levels[-1]
        nxt = []
        for g in chunk_indices(len(prev), fanout):
            nxt.append({"idx": len(nxt), "children": list(g)})
        levels.append(nxt)
    parents = [{} for _ in levels]
    for ell in range(len(levels) - 1):
        for p_idx, node in enumerate(levels[ell + 1]):
            for c in node["children"]:
                parents[ell][c] = p_idx
    for ell, level in enumerate(levels):
        for node in level:
            node["parent"] = parents[ell].get(node["idx"])
    return levels
