from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def agent_checkpoint() -> Path:
    return repo_root() / "models" / "agent" / "agent.pth"


def master_checkpoint() -> Path:
    return repo_root() / "models" / "master" / "master.pth"
