from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hdmaps.corridor import Corridor
from hdmaps.evaluation.episode import run_simultaneous_episode, run_staggered_episode
from hdmaps.hierarchy.tree import hierarchy_label
from hdmaps.paths import agent_checkpoint, master_checkpoint
from hdmaps.policies.master import load_models
from hdmaps.protocol import (
    SCALE_CONFIGS,
    SIM_DURATION,
    SIM_MAX_STEPS,
    SIM_UNIFORM_SEED,
    SIM_VARIED_SEED,
    STAG_COORD_RADIUS_M,
    STAG_EPISODES,
    STAG_SEED,
)
from hdmaps.scenarios.regional import generate_suite
from hdmaps.scenarios.staggered import generate_staggered_scenario, horizon_for_connector
from hdmaps.seed import seed_all
from hdmaps.statistics.metrics import crash_rate, mean_sem, round_wilson, wilson_interval


def _summarize(rows: list[dict], n_int: int, n_agents: int) -> dict:
    arrivals = np.array([r["arrival_pct"] for r in rows], dtype=float)
    crashed = np.array([r["crashed"] for r in rows], dtype=int)
    crash_pct, k, n = crash_rate(crashed)
    lo, hi = round_wilson(*wilson_interval(k, n))
    arr_mean, arr_sem = mean_sem(arrivals)
    return {
        : n_agents,
        : n_int,
        : hierarchy_label(n_int),
        : n,
        : crash_pct,
        : k,
        : lo,
        : hi,
        : arr_mean,
        : arr_sem,
        : rows,
    }


def evaluate_simultaneous(
    *,
    out_dir: Path | None = None,
    agent_path: Path | None = None,
    master_path: Path | None = None,
    configs: tuple[tuple[int, int], ...] | None = None,
    episodes_per_suite: int | None = None,
    suites: tuple[str, ...] = ("varied", "uniform"),
) -> list[dict]:
    seed_all(0)
    master, agent = load_models(
        str(agent_path or agent_checkpoint()),
        str(master_path or master_checkpoint()),
    )
    selected_configs = tuple(configs) if configs is not None else SCALE_CONFIGS
    n_ep = int(100 if episodes_per_suite is None else episodes_per_suite)
    results = []
    for n_int, n_agents in selected_configs:
        cell = Corridor(n_int, n_agents, connector_length=150, duration=SIM_DURATION)
        rows = []
        for suite, varied, seed in (
            ("varied", True, SIM_VARIED_SEED),
            ("uniform", False, SIM_UNIFORM_SEED),
        ):
            if suite not in suites:
                continue
            rng = np.random.default_rng(seed)
            for sc in generate_suite(n_int, n_agents, n_ep, rng, varied=varied):
                rows.append(run_simultaneous_episode(cell, sc, master, agent, max_steps=SIM_MAX_STEPS))
        cell.close()
        results.append(_summarize(rows, n_int, n_agents))
        print(

            % (n_agents, results[-1]["crash_pct"], results[-1]["arrival_pct"], results[-1]["n_episodes"]),
            flush=True,
        )
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            saved = [{k: v for k, v in row.items() if k != "episodes"} for row in results]
            (out_dir / "simultaneous_eval.json").write_text(json.dumps(saved, indent=2, default=float), encoding="utf-8")
    return results


def evaluate_staggered(
    *,
    out_dir: Path | None = None,
    agent_path: Path | None = None,
    master_path: Path | None = None,
) -> list[dict]:
    seed_all(0)
    master, agent = load_models(
        str(agent_path or agent_checkpoint()),
        str(master_path or master_checkpoint()),
    )
    n_ep = STAG_EPISODES
    results = []
    for n_int, n_agents in SCALE_CONFIGS:
        rng = np.random.default_rng(STAG_SEED)
        scenarios = [generate_staggered_scenario(n_int, n_agents, rng, varied=True) for _ in range(n_ep)]
        connector = int(scenarios[0]["connector_length"])
        duration = int(scenarios[0]["max_steps"])
        cell = Corridor(n_int, n_agents, connector_length=connector, duration=max(SIM_DURATION, duration))
        rows = []
        for sc in scenarios:
            horizon = int(sc.get("max_steps") or horizon_for_connector(connector))
            rows.append(
                run_staggered_episode(
                    cell, sc, master, agent, max_steps=horizon, coord_radius=STAG_COORD_RADIUS_M
                )
            )
        cell.close()
        results.append(_summarize(rows, n_int, n_agents))
        print(

            % (n_agents, results[-1]["crash_pct"], results[-1]["n_episodes"]),
            flush=True,
        )
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            saved = [{k: v for k, v in row.items() if k != "episodes"} for row in results]
            (out_dir / "staggered_eval.json").write_text(json.dumps(saved, indent=2, default=float), encoding="utf-8")
    return results
