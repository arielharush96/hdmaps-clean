from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from hdmaps.protocol import AGENT_OBS_DIM, TRAIN_AGENT_LR, TRAIN_BATCH_SIZE, TRAIN_GAMMA, TRAIN_ROLLOUT

WORKER_NET_ARCH = dict(pi=[64, 32, 16, 8], vf=[64, 32, 16, 8])


class _StaticEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, True, {}


def _hidden_sizes(net) -> list[int]:
    return [m.out_features for m in net if hasattr(m, "out_features")]


def is_paper_agent(model: PPO) -> bool:
    try:
        obs_ok = tuple(model.observation_space.shape) == (AGENT_OBS_DIM,)
        pi = _hidden_sizes(model.policy.mlp_extractor.policy_net)
        vf = _hidden_sizes(model.policy.mlp_extractor.value_net)
        return obs_ok and pi == [64, 32, 16, 8] and vf == [64, 32, 16, 8]
    except Exception:
        return False


def make_agent(learning_rate: float = TRAIN_AGENT_LR) -> PPO:
    env = _StaticEnv(AGENT_OBS_DIM)
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=float(learning_rate),
        n_steps=TRAIN_ROLLOUT,
        batch_size=TRAIN_BATCH_SIZE,
        gamma=TRAIN_GAMMA,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=WORKER_NET_ARCH, activation_fn=torch.nn.ReLU),
        verbose=0,
        device="cpu",
    )


def load_agent(path: str) -> PPO:
    env = _StaticEnv(AGENT_OBS_DIM)
    if path and Path(path).is_file():
        try:
            loaded = PPO.load(path, env=env, device="cpu")
            if is_paper_agent(loaded):
                return loaded
        except Exception:
            pass
    return make_agent()


def agent_act(agent_model: PPO, observations: list[np.ndarray], deterministic: bool):
    if not observations:
        empty = np.zeros(0, dtype=np.int64)
        return [], empty, empty
    obs_t = torch.as_tensor(np.asarray(observations, dtype=np.float32))
    with torch.no_grad():
        actions, values, log_probs = agent_model.policy.forward(obs_t, deterministic=deterministic)
    acts = actions.detach().cpu().numpy().reshape(-1).astype(int).tolist()
    val = values.detach().cpu().numpy().reshape(-1).astype(np.float32)
    lp = log_probs.detach().cpu().numpy().reshape(-1).astype(np.float32)
    return acts, val, lp


def agent_actions(agent_model: PPO, observations: list[np.ndarray], deterministic: bool) -> list[int]:
    acts, _, _ = agent_act(agent_model, observations, deterministic)
    return acts
