from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from hdmaps.protocol import EMBEDDING_DIM, MASTER_OBS_DIM, TRAIN_BATCH_SIZE, TRAIN_GAMMA, TRAIN_MASTER_LR, TRAIN_ROLLOUT

MASTER_NET_ARCH = dict(pi=[128, 256, 128], vf=[128, 256, 128])


class _MasterEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int, emb_dim: int, max_steps: int = 128):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(emb_dim,), dtype=np.float32)
        self._obs_dim = obs_dim
        self._emb_dim = emb_dim
        self._max_steps = max_steps
        self._rng = np.random.default_rng()
        self._state = np.zeros(obs_dim, dtype=np.float32)
        self._steps = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._state = self._rng.standard_normal(self._obs_dim).astype(np.float32)
        return self._state.copy(), {}

    def step(self, action):
        self._steps += 1
        return self._state.copy(), 0.0, False, self._steps >= self._max_steps, {}


def _hidden_sizes(net) -> list[int]:
    return [m.out_features for m in net if hasattr(m, "out_features")]


def is_paper_master(model: PPO) -> bool:
    try:
        obs_ok = tuple(model.observation_space.shape) == (MASTER_OBS_DIM,)
        act_ok = tuple(model.action_space.shape) == (EMBEDDING_DIM,)
        pi = _hidden_sizes(model.policy.mlp_extractor.policy_net)
        vf = _hidden_sizes(model.policy.mlp_extractor.value_net)
        return obs_ok and act_ok and pi == [128, 256, 128] and vf == [128, 256, 128]
    except Exception:
        return False


class MasterModel:
    def __init__(self, embedding_size=EMBEDDING_DIM, observation_dim=MASTER_OBS_DIM, **kwargs):
        self.embedding_size = int(embedding_size)
        self.observation_dim = int(observation_dim)
        self.last_raw = np.zeros((0, self.embedding_size), dtype=np.float32)
        env = _MasterEnv(self.observation_dim, self.embedding_size)
        self.model = PPO(
            ,
            env,
            learning_rate=float(kwargs.get("learning_rate", TRAIN_MASTER_LR)),
            n_steps=TRAIN_ROLLOUT,
            batch_size=TRAIN_BATCH_SIZE,
            gamma=TRAIN_GAMMA,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            policy_kwargs=dict(net_arch=MASTER_NET_ARCH, activation_fn=torch.nn.Tanh),
            verbose=0,
            device="cpu",
        )

    def load(self, path: str) -> bool:
        loaded = PPO.load(path, device="cpu")
        if not is_paper_master(loaded):
            return False
        self.model = loaded
        parent = os.path.dirname(os.path.abspath(path))
        sidecar = os.path.join(parent, "master_custom_params.pt")
        if os.path.exists(sidecar):
            params = torch.load(sidecar, map_location="cpu", weights_only=False)
            if "observation_dim" in params:
                self.observation_dim = int(params["observation_dim"])
            if "embedding_size" in params:
                self.embedding_size = int(params["embedding_size"])
        return True

    def save(self, path: str) -> None:
        self.model.save(path)
        sidecar = os.path.join(os.path.dirname(os.path.abspath(path)), "master_custom_params.pt")
        torch.save(
            {"observation_dim": self.observation_dim, "embedding_size": self.embedding_size},
            sidecar,
        )

    def embeddings(self, inputs: list[np.ndarray], deterministic: bool) -> list[np.ndarray]:
        actions, _, _ = self.act(inputs, deterministic)
        return [actions[i].reshape(-1)[: self.embedding_size] for i in range(actions.shape[0])]

    def act(self, inputs: list[np.ndarray], deterministic: bool):
        if not inputs:
            empty = np.zeros((0, self.embedding_size), dtype=np.float32)
            self.last_raw = empty
            return empty, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        obs_t = torch.as_tensor(np.asarray(inputs, dtype=np.float32), device=self.model.device)
        with torch.no_grad():
            raw, values, log_probs = self.model.policy.forward(obs_t, deterministic=deterministic)
        raw_np = raw.detach().cpu().numpy().astype(np.float32)
        self.last_raw = raw_np
        act = np.tanh(raw_np)
        val = values.detach().cpu().numpy().reshape(-1).astype(np.float32)
        lp = log_probs.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return act, val, lp


def load_models(agent_path: str | None, master_path: str | None):
    from hdmaps.policies.agent import load_agent

    master = MasterModel()
    if master_path and Path(master_path).is_file():
        master.load(str(master_path))
    agent = load_agent(agent_path or "")
    return master, agent
