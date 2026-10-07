from __future__ import annotations

import numpy as np

from hdmaps.protocol import (
    EMBEDDING_DIM,
    NUM_MASTER_SLOTS,
    SLOT_VEC_DIM,
    TYPE_MASTER,
    TYPE_UNUSED,
    TYPE_VEHICLE,
)


def pad_vec(vec, size: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    out = np.zeros(size, dtype=np.float32)
    out[: min(size, len(arr))] = arr[: min(size, len(arr))]
    return out


def slot(payload, type_bit: float, slot_vec_dim: int = SLOT_VEC_DIM) -> np.ndarray:
    return np.concatenate(
        [np.asarray([float(type_bit)], dtype=np.float32), pad_vec(payload, slot_vec_dim)]
    )


def unused_slot() -> np.ndarray:
    return np.zeros(SLOT_VEC_DIM + 1, dtype=np.float32)


def slot_mask(packed: np.ndarray) -> np.ndarray:
    slots = np.asarray(packed, dtype=np.float32).reshape(NUM_MASTER_SLOTS, SLOT_VEC_DIM + 1)
    return (np.max(np.abs(slots), axis=1) > 1e-8).astype(np.float32)


def pack_node(
    parent_emb,
    payloads,
    payload_bits,
    *,
    permute: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    slots = [slot(parent_emb, TYPE_MASTER)]
    for payload, bit in zip(list(payloads), list(payload_bits)):
        slots.append(slot(payload, bit))
    while len(slots) < NUM_MASTER_SLOTS:
        slots.append(unused_slot())
    slots = slots[:NUM_MASTER_SLOTS]
    if permute and len(slots) > 1:
        kids = slots[1:]
        order = rng.permutation(len(kids)) if rng is not None else np.random.permutation(len(kids))
        slots = [slots[0]] + [kids[int(i)] for i in order]
    return np.concatenate(slots).astype(np.float32)


def pack_local(parent_emb, group_states, *, center_x: float = 0.0, permute: bool = False, rng=None) -> np.ndarray:
    states = np.asarray(group_states, dtype=np.float32)
    if states.size == 0:
        payloads = []
    else:
        states = states.reshape(-1, 4)
        payloads = []
        for st in states:
            local = np.asarray(st[:4], dtype=np.float32).copy()
            local[0] -= float(center_x)
            payloads.append(local)
    bits = [TYPE_VEHICLE] * len(payloads)
    return pack_node(parent_emb, payloads, bits, permute=permute, rng=rng)


def pack_children(child_embs, parent_emb=None, *, permute: bool = False, rng=None) -> np.ndarray:
    kids = [np.asarray(e, dtype=np.float32).reshape(-1)[:EMBEDDING_DIM] for e in list(child_embs)[: NUM_MASTER_SLOTS - 1]]
    parent = np.zeros(EMBEDDING_DIM, dtype=np.float32) if parent_emb is None else parent_emb
    bits = [TYPE_MASTER] * len(kids)
    return pack_node(parent, kids, bits, permute=permute, rng=rng)


def pack_mixed(parent_emb, child_embs, vehicle_states, *, permute: bool = False, rng=None) -> np.ndarray:
    payloads = []
    bits = []
    for e in list(child_embs):
        payloads.append(np.asarray(e, dtype=np.float32).reshape(-1)[:EMBEDDING_DIM])
        bits.append(TYPE_MASTER)
    states = np.asarray(vehicle_states, dtype=np.float32)
    if states.size:
        for st in states.reshape(-1, 4):
            payloads.append(np.asarray(st[:4], dtype=np.float32))
            bits.append(TYPE_VEHICLE)
    capacity = NUM_MASTER_SLOTS - 1
    payloads = payloads[:capacity]
    bits = bits[:capacity]
    return pack_node(parent_emb, payloads, bits, permute=permute, rng=rng)


def agent_obs(local_state, embedding) -> np.ndarray:
    st = np.asarray(local_state, dtype=np.float32).reshape(-1)[:4]
    emb = pad_vec(embedding, EMBEDDING_DIM)
    return np.concatenate([st, emb]).astype(np.float32)
