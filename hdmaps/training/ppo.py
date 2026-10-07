from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

from hdmaps.protocol import (
    AGENT_OBS_DIM,
    EMBEDDING_DIM,
    MASTER_OBS_DIM,
    TRAIN_BATCH_SIZE,
    TRAIN_CLIP,
    TRAIN_ENT_COEF,
    TRAIN_GAE_LAMBDA,
    TRAIN_GAMMA,
    TRAIN_PPO_EPOCHS,
    TRAIN_ROLLOUT,
    TRAIN_VF_COEF,
)


def _as_tensor(x, dtype=torch.float32):
    return x if torch.is_tensor(x) else torch.as_tensor(x, dtype=dtype)


def _flatten_trajs(stream: list[dict]):
    by: dict = {}
    order = []
    for tr in stream:
        tid = tr.get('tid')
        if tid is None:
            tid = ('row', len(order))
        if tid not in by:
            by[tid] = []
            order.append(tid)
        by[tid].append(tr)
    flat = []
    starts = []
    for tid in order:
        traj = by[tid]
        for i, tr in enumerate(traj):
            flat.append(tr)
            starts.append(1.0 if i == 0 else 0.0)
    return flat, starts


def update_continuous(policy, stream: list[dict], *, obs_dim: int = MASTER_OBS_DIM, act_dim: int = EMBEDDING_DIM):
    stream, starts = _flatten_trajs(stream)
    if len(stream) < 2:
        return None
    buf = RolloutBuffer(
        buffer_size=max(TRAIN_ROLLOUT, len(stream)),
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32),
        gamma=TRAIN_GAMMA,
        gae_lambda=TRAIN_GAE_LAMBDA,
        n_envs=1,
    )
    last_obs = None
    for i, tr in enumerate(stream):
        last_obs = tr['obs']
        buf.add(
            obs=np.asarray(tr['obs'], np.float32).reshape(1, -1),
            action=np.asarray(tr['action'], np.float32).reshape(1, -1),
            reward=np.array([tr['reward']], np.float32),
            episode_start=np.array([starts[i]], np.float32),
            value=torch.as_tensor([[tr['value']]], dtype=torch.float32),
            log_prob=torch.as_tensor([tr['log_prob']], dtype=torch.float32),
        )
        if buf.full:
            break
    if buf.pos == 0:
        return None
    with torch.no_grad():
        last_value = policy.predict_values(torch.as_tensor(np.asarray(last_obs).reshape(1, -1), dtype=torch.float32))
    buf.compute_returns_and_advantage(last_values=last_value, dones=np.array([1.0], np.float32))
    orig = buf.full
    buf.full = True
    try:
        data = buf._get_samples(np.arange(buf.pos))
    finally:
        buf.full = orig
    obs = _as_tensor(data.observations).reshape(data.observations.shape[0], -1)
    act = _as_tensor(data.actions).reshape(data.actions.shape[0], -1)
    old_lp = _as_tensor(data.old_log_prob).reshape(-1)
    n = int(obs.shape[0])
    returns = _as_tensor(data.returns).reshape(-1)[:n]
    adv = _as_tensor(data.advantages).reshape(-1)[:n]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    last_loss = None
    idx = np.arange(n)
    for _ in range(TRAIN_PPO_EPOCHS):
        np.random.shuffle(idx)
        for start in range(0, n, TRAIN_BATCH_SIZE):
            mb = idx[start : start + TRAIN_BATCH_SIZE]
            values, log_probs, entropy = policy.evaluate_actions(obs[mb], act[mb])
            log_probs = log_probs.reshape(-1)
            ratio = torch.exp(log_probs - old_lp[mb][: log_probs.shape[0]])
            a = adv[mb][: log_probs.shape[0]]
            surr1 = ratio * a
            surr2 = torch.clamp(ratio, 1.0 - TRAIN_CLIP, 1.0 + TRAIN_CLIP) * a
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((values.reshape(-1)[: a.shape[0]] - returns[mb][: a.shape[0]]) ** 2).mean()
            entropy_loss = -entropy.mean() if entropy is not None else obs.new_tensor(0.0)
            loss = policy_loss + TRAIN_VF_COEF * value_loss + TRAIN_ENT_COEF * entropy_loss
            policy.optimizer.zero_grad()
            if torch.isfinite(loss).all():
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                policy.optimizer.step()
            last_loss = {
                'policy_loss': float(policy_loss.detach()),
                'value_loss': float(value_loss.detach()),
                'total_loss': float(loss.detach()),
            }
    return last_loss


def update_discrete(policy, stream: list[dict]):
    stream, starts = _flatten_trajs(stream)
    if len(stream) < 2:
        return None
    buf = RolloutBuffer(
        buffer_size=max(TRAIN_ROLLOUT, len(stream)),
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(AGENT_OBS_DIM,), dtype=np.float32),
        action_space=spaces.Discrete(2),
        gamma=TRAIN_GAMMA,
        gae_lambda=TRAIN_GAE_LAMBDA,
        n_envs=1,
    )
    last_obs = None
    for i, tr in enumerate(stream):
        last_obs = tr['obs']
        buf.add(
            obs=np.asarray(tr['obs'], np.float32).reshape(1, -1),
            action=np.array([int(tr['action'])], np.float32),
            reward=np.array([tr['reward']], np.float32),
            episode_start=np.array([starts[i]], np.float32),
            value=torch.as_tensor([[tr['value']]], dtype=torch.float32),
            log_prob=torch.as_tensor([tr['log_prob']], dtype=torch.float32),
        )
        if buf.full:
            break
    if buf.pos == 0:
        return None
    with torch.no_grad():
        last_value = policy.predict_values(torch.as_tensor(np.asarray(last_obs).reshape(1, -1), dtype=torch.float32))
    buf.compute_returns_and_advantage(last_values=last_value, dones=np.array([1.0], np.float32))
    orig = buf.full
    buf.full = True
    try:
        data = buf._get_samples(np.arange(buf.pos))
    finally:
        buf.full = orig
    obs = _as_tensor(data.observations).reshape(data.observations.shape[0], -1)
    act = _as_tensor(data.actions).reshape(-1).long()
    old_lp = _as_tensor(data.old_log_prob).reshape(-1)
    n = int(obs.shape[0])
    returns = _as_tensor(data.returns).reshape(-1)[:n]
    adv = _as_tensor(data.advantages).reshape(-1)[:n]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    last_loss = None
    idx = np.arange(n)
    for _ in range(TRAIN_PPO_EPOCHS):
        np.random.shuffle(idx)
        for start in range(0, n, TRAIN_BATCH_SIZE):
            mb = idx[start : start + TRAIN_BATCH_SIZE]
            values, log_probs, entropy = policy.evaluate_actions(obs[mb], act[mb])
            log_probs = log_probs.reshape(-1)
            ratio = torch.exp(log_probs - old_lp[mb][: log_probs.shape[0]])
            a = adv[mb][: log_probs.shape[0]]
            surr1 = ratio * a
            surr2 = torch.clamp(ratio, 1.0 - TRAIN_CLIP, 1.0 + TRAIN_CLIP) * a
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((values.reshape(-1)[: a.shape[0]] - returns[mb][: a.shape[0]]) ** 2).mean()
            entropy_loss = -entropy.mean() if entropy is not None else obs.new_tensor(0.0)
            loss = policy_loss + TRAIN_VF_COEF * value_loss + TRAIN_ENT_COEF * entropy_loss
            policy.optimizer.zero_grad()
            if torch.isfinite(loss).all():
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                policy.optimizer.step()
            last_loss = {'total_loss': float(loss.detach())}
    return last_loss


def update_master(master, stream: list[dict]):
    return update_continuous(master.model.policy, stream)


def update_master_roles(master, streams_by_role: dict[str, list[dict]]):
    losses = {}
    for role in ('lm', 'im', 'gm'):
        info = update_master(master, list(streams_by_role.get(role) or []))
        if info is not None:
            losses[role] = info
    return losses


def update_agent(agent, stream: list[dict]):
    return update_discrete(agent.policy, stream)
