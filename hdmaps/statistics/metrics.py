from __future__ import annotations

import math

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / den
    margin = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / den
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return 100.0 * lo, 100.0 * hi


def round_wilson(lo: float, hi: float) -> tuple[float, float]:
    return round(lo, 1), round(hi, 1)


def mean_sem(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0
    sem = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
    return mean, sem


def crash_rate(crashed: np.ndarray) -> tuple[float, int, int]:
    flags = np.asarray(crashed, dtype=int).reshape(-1)
    n = int(flags.size)
    k = int(np.sum(flags > 0))
    pct = 100.0 * k / n if n else 0.0
    return pct, k, n
