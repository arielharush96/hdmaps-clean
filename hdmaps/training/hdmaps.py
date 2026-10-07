from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from hdmaps.corridor import Corridor
from hdmaps.paths import repo_root
from hdmaps.policies.master import load_models
from hdmaps.protocol import (
    SIM_CONNECTOR_M,
    SIM_DURATION,
    TRAIN_CHECKPOINT_EVERY,
    TRAIN_EPISODES,
    TRAIN_JOINT_EPISODES,
    TRAIN_MASTER_LR,
    TRAIN_MAX_AGENTS,
    TRAIN_MAX_STEPS,
    TRAIN_N_INT,
    TRAIN_ROLLOUT,
    TRAIN_SEED,
    TRAIN_STAGE1_EPISODES,
    TRAIN_STAGE1_LR,
    TRAIN_STAGE1_SEED,
)
from hdmaps.scenarios.regional import generate_scenario, generate_single_intersection, _retry
from hdmaps.seed import seed_all
from hdmaps.training.ppo import update_agent, update_master_roles
from hdmaps.training.rollout import run_training_episode


def sample_stage1(rng: np.random.Generator) -> dict:
    n_agents = int(rng.choice([1, 2, 3, 3, 4, 4]))
    mixed = bool(n_agents >= 3 and rng.random() < 0.5)
    sc = _retry(lambda: generate_single_intersection(n_agents, rng, mixed=mixed))
    sc["connector_length"] = SIM_CONNECTOR_M
    return {
        : 1,
        : n_agents,
        : SIM_CONNECTOR_M,
        : mixed,
        : sc,
    }


def sample_stage2(rng: np.random.Generator) -> dict:
    n_int = TRAIN_N_INT
    n_agents = int(rng.choice([4, 6, 6, 6]))
    connector = SIM_CONNECTOR_M
    sc = _retry(lambda: generate_scenario(n_int, n_agents, rng, varied=True))
    sc["connector_length"] = connector
    return {
        : n_int,
        : n_agents,
        : connector,
        : n_agents // n_int,
        : False,
        : sc,
    }


def _save_pair(master, agent, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    master.save(str(dest / "master.pth"))
    agent.save(str(dest / "agent.pth"))


def _write_summary(dest: Path, summary: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "train_log.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")


def _load_summary(dest: Path) -> dict:
    path = dest / "train_log.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_complete(dest: Path, n_episodes: int) -> bool:
    if not (dest / "master.pth").is_file() or not (dest / "agent.pth").is_file() or not (dest / "train_log.json").is_file():
        return False
    try:
        data = _load_summary(dest)
    except Exception:
        return False
    return int(data.get("episodes", 0)) >= int(n_episodes)


def _latest_checkpoint(dest: Path) -> tuple[int, Path] | None:
    found: list[tuple[int, Path]] = []
    for path in dest.glob("master_ep*.pth"):
        try:
            found.append((int(path.stem.split("ep")[-1]), path))
        except ValueError:
            continue
    if not found:
        return None
    return max(found, key=lambda item: item[0])


def _save_rng(rng: np.random.Generator, dest: Path, episode: int) -> None:
    (dest / ("rng_ep%04d.json" % episode)).write_text(
        json.dumps(rng.bit_generator.state),
        encoding="utf-8",
    )


def _load_rng(rng: np.random.Generator, dest: Path, episode: int) -> None:
    path = dest / ("rng_ep%04d.json" % episode)
    if path.is_file():
        rng.bit_generator.state = json.loads(path.read_text(encoding="utf-8"))


def _history_within_cap(dest: Path) -> bool:
    paths = [dest / "history.jsonl", dest / "history_incomplete.jsonl"]
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("n_int", 0)) > TRAIN_N_INT or int(row.get("n_agents", 0)) > TRAIN_MAX_AGENTS:
                return False
    return True


def archive_overscale_stage(dest: Path) -> Path | None:
    dest = Path(dest)
    if not dest.exists() or _history_within_cap(dest):
        return None
    bak = dest.with_name(dest.name + "_overscale")
    n = 1
    while bak.exists():
        n += 1
        bak = dest.with_name("%s_overscale_%d" % (dest.name, n))
    dest.rename(bak)
    print("[pipeline] archived overscale %s -> %s" % (dest.name, bak.name), flush=True)
    return bak


def _truncate_history(history_path: Path, n_keep: int) -> None:
    if n_keep <= 0 or not history_path.is_file():
        if history_path.is_file():
            history_path.unlink()
        return
    lines = [line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    history_path.write_text("\n".join(lines[:n_keep]) + ("\n" if lines[:n_keep] else ""), encoding="utf-8")


def _phase(episode_index: int) -> str:
    if episode_index < TRAIN_JOINT_EPISODES:
        return "joint"
    return "master" if ((episode_index - TRAIN_JOINT_EPISODES) % 2 == 0) else "worker"


def _flush_master(master, buffers: dict[str, list], min_len: int, force: bool) -> int:
    total = sum(len(v) for v in buffers.values())
    if total == 0:
        return 0
    if (not force) and total < min_len:
        return 0
    losses = update_master_roles(master, buffers)
    for key in list(buffers):
        buffers[key] = []
    return len(losses)


def _flush_worker(agent, buffer: list, min_len: int, force: bool) -> int:
    if not buffer:
        return 0
    if (not force) and len(buffer) < min_len:
        return 0
    info = update_agent(agent, buffer)
    buffer.clear()
    return 0 if info is None else 1


def train_loop(
    *,
    n_episodes: int,
    seed: int,
    lr: float,
    sample_fn,
    agent_path: Path | None,
    master_path: Path | None,
    dest: Path,
    smoke: bool = False,
    label: str = "train",
) -> dict:
    dest = Path(dest)
    if not smoke:
        archive_overscale_stage(dest)
    dest.mkdir(parents=True, exist_ok=True)
    history_path = dest / "history.jsonl"
    seed_all(seed)
    rng = np.random.default_rng(seed)
    start_ep = 0
    load_master = Path(master_path) if master_path else None
    load_agent = Path(agent_path) if agent_path else None
    resume = None if smoke else _latest_checkpoint(dest)
    if resume is not None and resume[0] < n_episodes:
        start_ep, ckpt = resume
        load_master = ckpt
        agent_ckpt = dest / ("agent_ep%04d.pth" % start_ep)
        if agent_ckpt.is_file():
            load_agent = agent_ckpt
        elif (dest / "agent.pth").is_file():
            load_agent = dest / "agent.pth"
        _load_rng(rng, dest, start_ep)
        _truncate_history(history_path, start_ep)
        print("[%s] resume from %s (episode %d)" % (label, ckpt.name, start_ep), flush=True)
    elif history_path.is_file() and resume is None:
        bak = dest / "history_incomplete.jsonl"
        history_path.replace(bak)
        print("[%s] archived incomplete history to %s" % (label, bak.name), flush=True)

    master, agent = load_models(
        str(load_agent) if load_agent else "",
        str(load_master) if load_master else "",
    )
    for param_group in master.model.policy.optimizer.param_groups:
        param_group["lr"] = lr
    for param_group in agent.policy.optimizer.param_groups:
        param_group["lr"] = lr
    cache: dict[tuple[int, int, int], Corridor] = {}
    t0 = time.time()
    master_buf: dict[str, list] = {"lm": [], "im": [], "gm": []}
    worker_buf: list = []
    max_steps = 8 if smoke else TRAIN_MAX_STEPS

    for ep in range(start_ep, n_episodes):
        if smoke:
            spec = {
                : 1,
                : 3,
                : 150,
                : False,
                : 3,
                : generate_single_intersection(3, rng, mixed=False),
            }
            spec["scenario"]["connector_length"] = 150
        else:
            spec = sample_fn(rng)
        n_int = int(spec["n_int"])
        n_agents = int(spec["n_agents"])
        connector = int(spec["connector"])
        mixed = bool(spec.get("mixed", False))
        if (not smoke) and (n_int > TRAIN_N_INT or n_agents > TRAIN_MAX_AGENTS):
            raise ValueError(

                % (TRAIN_N_INT, TRAIN_MAX_AGENTS, n_int, n_agents)
            )
        key = (n_int, n_agents, connector)
        if key not in cache:
            while len(cache) >= 4:
                cache.pop(next(iter(cache))).close()
            cache[key] = Corridor(n_int, n_agents, connector_length=connector, duration=SIM_DURATION)
        cell = cache[key]
        result, role_streams, worker_stream = run_training_episode(
            cell,
            spec["scenario"],
            master,
            agent,
            max_steps=max_steps,
            rng=rng,
            mixed=mixed,
        )
        for role, stream in role_streams.items():
            master_buf.setdefault(role, []).extend(stream)
        worker_buf.extend(worker_stream)
        phase = "joint" if smoke else _phase(ep)
        n_updates = 0
        if phase in ("joint", "master"):
            n_updates += _flush_master(master, master_buf, TRAIN_ROLLOUT, force=False)
        if phase in ("joint", "worker"):
            n_updates += _flush_worker(agent, worker_buf, TRAIN_ROLLOUT, force=False)
        row = {
            : ep + 1,
            : n_int,
            : n_agents,
            : connector,
            : result["steps"],
            : bool(result["crashed"]),
            : int(result["arrived"]),
            : float(result["reward"]),
            : n_updates,
            : phase,
            : mixed,
        }
        if "agents_per_int" in spec:
            row["agents_per_int"] = int(spec["agents_per_int"])
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        if (ep + 1) % 10 == 0 or ep == start_ep or (ep + 1) == n_episodes:
            elapsed = time.time() - t0
            print(

                % (
                    label,
                    ep + 1,
                    n_episodes,
                    n_int,
                    n_agents,
                    row["steps"],
                    row["arrived"],
                    row["crashed"],
                    phase,
                    row["n_updates"],
                    elapsed,
                ),
                flush=True,
            )
        if (ep + 1) % TRAIN_CHECKPOINT_EVERY == 0:
            master.save(str(dest / ("master_ep%04d.pth" % (ep + 1))))
            agent.save(str(dest / ("agent_ep%04d.pth" % (ep + 1))))
            _save_pair(master, agent, dest)
            _save_rng(rng, dest, ep + 1)
            print("[%s] checkpoint ep %d" % (label, ep + 1), flush=True)

    _flush_master(master, master_buf, 2, force=True)
    _flush_worker(agent, worker_buf, 2, force=True)
    for cell in cache.values():
        cell.close()

    _save_pair(master, agent, dest)
    summary = {
        : label,
        : n_episodes,
        : seed,
        : lr,
        : str(master_path) if master_path else None,
        : str(agent_path) if agent_path else None,
        : time.time() - t0,
        : start_ep,
        : TRAIN_JOINT_EPISODES,
        : max_steps,
        : None,
        : None,
    }
    rows = (
        [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if history_path.is_file()
        else []
    )
    if rows:
        summary["mean_arrival"] = float(np.mean([100.0 * r["arrived"] / max(1, r["n_agents"]) for r in rows]))
        summary["crash_rate"] = 100.0 * float(np.mean([1.0 if r["crashed"] else 0.0 for r in rows]))
        summary["max_n_int"] = int(max(r["n_int"] for r in rows))
        summary["max_n_agents"] = int(max(r["n_agents"] for r in rows))
    _write_summary(dest, summary)
    print(

        % (label, summary["mean_arrival"] or 0.0, summary["crash_rate"] or 0.0),
        flush=True,
    )
    return summary


def stage_seeds(seed: int | None) -> tuple[int, int]:
    if seed is None:
        return TRAIN_STAGE1_SEED, TRAIN_SEED
    offset = TRAIN_SEED - TRAIN_STAGE1_SEED
    return int(seed), int(seed) + offset


def train(
    *,
    smoke: bool = False,
    out_dir: Path | None = None,
    episodes: int | None = None,
    seed: int | None = None,
) -> dict:
    dest = Path(out_dir) if out_dir else (repo_root() / "outputs" / "train")
    n_episodes = 2 if smoke else int(episodes or TRAIN_EPISODES)
    return train_loop(
        n_episodes=n_episodes,
        seed=stage_seeds(seed)[1],
        lr=TRAIN_MASTER_LR,
        sample_fn=sample_stage2,
        agent_path=None,
        master_path=None,
        dest=dest,
        smoke=smoke,
        label="stage2",
    )


def train_pipeline(*, smoke: bool = False, out_dir: Path | None = None, seed: int | None = None) -> dict:
    dest = Path(out_dir) if out_dir else (repo_root() / "outputs" / "train")
    stage1_dir = dest / "stage1"
    stage2_dir = dest / "stage2"
    seed1, seed2 = stage_seeds(seed)
    n1 = 2 if smoke else TRAIN_STAGE1_EPISODES
    n2 = 2 if smoke else TRAIN_EPISODES
    if (not smoke) and _stage_complete(stage1_dir, n1):
        s1 = _load_summary(stage1_dir)
        print("[pipeline] skip stage1, using %s" % (stage1_dir / "master.pth"), flush=True)
    else:
        s1 = train_loop(
            n_episodes=n1,
            seed=seed1,
            lr=TRAIN_STAGE1_LR,
            sample_fn=sample_stage1,
            agent_path=None,
            master_path=None,
            dest=stage1_dir,
            smoke=smoke,
            label="stage1",
        )
    if not smoke:
        archive_overscale_stage(stage2_dir)
    if (not smoke) and _stage_complete(stage2_dir, n2):
        s2 = _load_summary(stage2_dir)
        print("[pipeline] skip stage2, using %s" % (stage2_dir / "master.pth"), flush=True)
    else:
        s2 = train_loop(
            n_episodes=n2,
            seed=seed2,
            lr=TRAIN_MASTER_LR,
            sample_fn=sample_stage2,
            agent_path=stage1_dir / "agent.pth",
            master_path=stage1_dir / "master.pth",
            dest=stage2_dir,
            smoke=smoke,
            label="stage2",
        )
    dest.mkdir(parents=True, exist_ok=True)
    if (stage2_dir / "master.pth").is_file():
        (dest / "master.pth").write_bytes((stage2_dir / "master.pth").read_bytes())
    if (stage2_dir / "agent.pth").is_file():
        (dest / "agent.pth").write_bytes((stage2_dir / "agent.pth").read_bytes())
    sidecar = stage2_dir / "master_custom_params.pt"
    if sidecar.is_file():
        (dest / "master_custom_params.pt").write_bytes(sidecar.read_bytes())
    summary = {"stage1": s1, "stage2": s2, "master": str(dest / "master.pth"), "agent": str(dest / "agent.pth")}
    _write_summary(dest, summary)
    return summary
