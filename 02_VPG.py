# ================================================================
# Vanilla Policy Gradient (VPG) — LunarLanderContinuous-v3
# Reuses DQN-style utilities: seeding, device, config dataclass,
# logging cadence, evaluation helper, optional plotting/saving.
# ================================================================

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# ------------------------------
# Reuse from DQN: global seeding
# ------------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
import random

random.seed(seed)

# ------------------------------
# Reuse from DQN: device picking
# ------------------------------
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)


# --------------------------------------
# Reuse DQN pattern: training config dataclass
# (fields tailored for VPG)
# --------------------------------------
@dataclass
class VPGTrainingConfig:
    seed: int = 42
    total_episodes: int = 500
    max_steps_per_episode: int = 1000
    gamma: float = 0.99
    lr_policy: float = 3e-4
    lr_value: float = 3e-4
    hidden_sizes: Tuple[int, int] = (256, 256)
    grad_clip_norm: Optional[float] = 10.0  # None to disable
    normalize_advantages: bool = True
    entropy_coef: float = 0.0  # set >0 to add entropy bonus
    eval_episodes: int = 5
    log_every: int = 50
    results_dir: str = "trained_networks_lunar_lander"


cfg = VPGTrainingConfig(seed=seed)

# -------------------------
# Environment (continuous)
# -------------------------
env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
# Deterministic-ish runs
try:
    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)
except TypeError:
    # For older gymnasium, reset signature may differ
    pass

# Obs/Action dims
state, info = env.reset()
n_obs = len(state)
assert hasattr(env.action_space, "shape") and len(env.action_space.shape) == 1, (
    "This script expects a 1D continuous action space."
)
n_actions = env.action_space.shape[0]

print(f"\nEnvironment: LunarLanderContinuous-v3")
print(f"State dimension: {n_obs}")
print(f"Action dimension: {n_actions} (Box[-1, 1])")
print("Initial state:", state)


# ---------------------------------
# Neural nets (VPG-specific parts)
# ---------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: Tuple[int, int], out_dim: int):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNetwork(nn.Module):
    """State-value function V(s)."""

    def __init__(self, n_obs: int, hidden_sizes: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.body = MLP(n_obs, hidden_sizes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return shape: (B,) for convenience
        return self.body(x).squeeze(-1)


class GaussianPolicy(nn.Module):
    """
    Tanh-squashed Gaussian policy π(a|s) for continuous actions in [-1, 1]^dim.
    Returns both mean and log_std; log_std is clipped for numerical stability.
    """

    def __init__(
        self,
        n_obs: int,
        n_actions: int,
        hidden_sizes: Tuple[int, int] = (256, 256),
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
    ):
        super().__init__()
        self.mu_net = MLP(n_obs, hidden_sizes, n_actions)
        self.log_std = nn.Parameter(torch.zeros(n_actions))
        self.log_std_bounds = log_std_bounds

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu_net(x)
        log_std = torch.clamp(
            self.log_std, self.log_std_bounds[0], self.log_std_bounds[1]
        )
        return mu, log_std

    @staticmethod
    def _tanh_squash(z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(z)

    def sample_action_and_log_prob(
        self, s_np: np.ndarray, device: torch.device
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Sample action a = tanh(z) with z ~ N(mu, std), and compute log π(a|s)
        including tanh correction.
        Returns (action_np, log_prob_torch).
        """
        s = torch.as_tensor(s_np, dtype=torch.float32, device=device).unsqueeze(
            0
        )  # (1, n_obs)
        mu, log_std = self.forward(s)  # (1, n_actions), (n_actions,)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.rsample()  # reparameterized sample (1, n_actions)
        a = self._tanh_squash(z)  # (1, n_actions)
        # Log prob with tanh correction
        log_prob = dist.log_prob(z).sum(dim=-1) - torch.log(1 - a.pow(2) + 1e-6).sum(
            dim=-1
        )  # (1,)
        return a.squeeze(0).detach().cpu().numpy(), log_prob.squeeze(
            0
        )  # np (n_actions,), torch scalar

    def deterministic_action(self, s_np: np.ndarray, device: torch.device) -> np.ndarray:
        """Mean action (tanh(mu)) for evaluation."""
        s = torch.as_tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)
        mu, _ = self.forward(s)
        a = torch.tanh(mu)
        return a.squeeze(0).detach().cpu().numpy()


# ------------------------------------------
# VPG utilities: trajectories & advantages
# ------------------------------------------
@torch.no_grad()
def collect_trajectory(
    env: gym.Env,
    policy: GaussianPolicy,
    value_net: ValueNetwork,
    device: torch.device,
    max_steps: int,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[torch.Tensor],
    List[float],
    List[bool],
    List[float],
]:
    """
    Roll out one episode with the current policy.
    Returns: states, actions, log_probs, rewards, dones, values
    """
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    dones: List[bool] = []
    values: List[float] = []

    obs, _ = env.reset()
    for _ in range(max_steps):
        # Important: do NOT detach log_prob (we want its graph) -> so we temporarily
        # disable torch.no_grad() for the sampling step:
        torch.set_grad_enabled(True)
        action_np, logp = policy.sample_action_and_log_prob(obs, device=device)
        torch.set_grad_enabled(False)

        # Baseline value (no grad here, we store float)
        v = value_net(
            torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        ).item()

        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = bool(terminated or truncated)

        states.append(obs)
        actions.append(action_np)
        log_probs.append(logp)  # keep as torch tensors for autograd
        rewards.append(float(reward))
        dones.append(done)
        values.append(v)

        obs = next_obs
        if done:
            break

    return states, actions, log_probs, rewards, dones, values


def compute_returns_and_advantages(
    rewards: List[float], values: List[float], dones: List[bool], gamma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo returns with value baseline advantage: A_t = G_t - V(s_t)
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.bool_)

    returns: List[float] = []
    G = 0.0
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            G = 0.0
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    returns_np = np.asarray(returns, dtype=np.float32)
    advantages_np = returns_np - values
    return returns_np, advantages_np


# ------------------------------------------
# Reuse from DQN: evaluation helper pattern
# (deterministic actions for eval rollouts)
# ------------------------------------------
@torch.no_grad()
def evaluate_policy_vpg(
    policy: GaussianPolicy,
    env_name: str,
    device: torch.device,
    n_episodes: int,
    seed: int,
    max_steps_per_episode: int,
) -> float:
    eval_env = gym.make(env_name, continuous=True, render_mode=None)
    try:
        eval_env.reset(seed=seed + 12345)  # slight offset
    except TypeError:
        pass
    total = 0.0
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        ep_r = 0.0
        for _ in range(max_steps_per_episode):
            action_np = policy.deterministic_action(obs, device)
            obs, r, terminated, truncated, _ = eval_env.step(action_np)
            ep_r += r
            if terminated or truncated:
                break
        total += ep_r
    eval_env.close()
    mean_r = total / n_episodes
    print(f"[Eval] mean return over {n_episodes} episodes: {mean_r:.2f}")
    return mean_r


# ------------------------------------------
# Optional plotting/saving (DQN-style hooks)
# ------------------------------------------
_have_plot_helper = False
try:
    from plot_rl_results import plot_training_statistics

    _have_plot_helper = True
except Exception:
    pass

_have_saver = False
try:
    from model_saver import save_model_pytorch, save_lunar_model_onnx

    _have_saver = True
except Exception:
    pass


def _local_plot_losses(
    policy_losses: List[float],
    value_losses: List[float],
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), dpi=110)
    ax[0].plot(policy_losses)
    ax[0].set_title("Policy Loss")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Loss")

    ax[1].plot(value_losses)
    ax[1].set_title("Value Loss")
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Loss")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path + "_losses.png", dpi=300)
        print(f"Saved losses plot to {save_path}_losses.png")
    plt.close(fig)


# ------------------------------------------
# Training loop (reusing DQN's cadence)
# ------------------------------------------
def train_vpg_continuous(
    env: gym.Env,
    cfg: VPGTrainingConfig,
    device: torch.device,
    n_obs: int,
    n_actions: int,
) -> Tuple[
    GaussianPolicy, ValueNetwork, List[float], List[int], List[float], List[float]
]:
    policy = GaussianPolicy(n_obs, n_actions, hidden_sizes=cfg.hidden_sizes).to(device)
    value_net = ValueNetwork(n_obs, hidden_sizes=cfg.hidden_sizes).to(device)

    policy_opt = optim.AdamW(policy.parameters(), lr=cfg.lr_policy, amsgrad=True)
    value_opt = optim.AdamW(value_net.parameters(), lr=cfg.lr_value, amsgrad=True)
    value_criterion = nn.MSELoss()

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    policy_losses: List[float] = []
    value_losses: List[float] = []

    for ep in range(cfg.total_episodes):
        states, actions, log_probs, rewards, dones, values = collect_trajectory(
            env=env,
            policy=policy,
            value_net=value_net,
            device=device,
            max_steps=cfg.max_steps_per_episode,
        )

        # Returns & advantages
        returns_np, adv_np = compute_returns_and_advantages(
            rewards, values, dones, gamma=cfg.gamma
        )

        # Convert to tensors
        log_probs_t = torch.stack(log_probs)  # (T,)
        returns_t = torch.as_tensor(returns_np, dtype=torch.float32, device=device)
        advantages_t = torch.as_tensor(adv_np, dtype=torch.float32, device=device)

        if cfg.normalize_advantages and advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (
                advantages_t.std(unbiased=False) + 1e-8
            )

        # === Policy update ===
        policy_opt.zero_grad(set_to_none=True)
        policy_loss = -(log_probs_t * advantages_t).mean()
        if cfg.entropy_coef != 0.0:
            # Entropy for tanh-Gaussian is tricky; a simple proxy is entropy of the unsquashed Normal
            # which you could add by recomputing Normal at states. Keeping 0.0 by default.
            pass
        policy_loss.backward()
        if cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=cfg.grad_clip_norm)
        policy_opt.step()

        # === Value update ===
        value_opt.zero_grad(set_to_none=True)
        states_t = torch.as_tensor(np.asarray(states, dtype=np.float32), device=device)
        value_preds = value_net(states_t)  # (T,)
        value_loss = value_criterion(
            value_preds, returns_t
        )  # MSE to Monte Carlo returns
        value_loss.backward()
        if cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=cfg.grad_clip_norm)
        value_opt.step()

        # Logging
        ep_return = float(np.sum(rewards))
        episode_rewards.append(ep_return)
        episode_lengths.append(len(rewards))
        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))

        if (ep + 1) % cfg.log_every == 0 or ep == 0:
            avg_r = (
                float(np.mean(episode_rewards[-25:]))
                if len(episode_rewards) >= 1
                else ep_return
            )
            print(
                f"[Episode {ep + 1:4d}] len={episode_lengths[-1]:4d}  "
                f"return={ep_return:8.2f}  avg25={avg_r:8.2f}  "
                f"π_loss={policy_losses[-1]:.3f}  V_loss={value_losses[-1]:.3f}"
            )

    return (
        policy,
        value_net,
        episode_rewards,
        episode_lengths,
        policy_losses,
        value_losses,
    )


# -----------------
# Train + Evaluate
# -----------------
policy, value_net, ep_rewards, ep_lengths, pi_losses, v_losses = train_vpg_continuous(
    env=env, cfg=cfg, device=device, n_obs=n_obs, n_actions=n_actions
)

# Separate eval env (train env can stay open/closed independently)
mean_eval_return = evaluate_policy_vpg(
    policy=policy,
    env_name="LunarLander-v3",
    device=device,
    n_episodes=cfg.eval_episodes,
    seed=cfg.seed,
    max_steps_per_episode=cfg.max_steps_per_episode,
)

# -----------------
# Visualize results
# -----------------
os.makedirs(cfg.results_dir, exist_ok=True)
title = "Vanilla Policy Gradient (LunarLanderContinuous-v3)"
save_prefix = os.path.join(cfg.results_dir, "vpg_lunar")

if _have_plot_helper:
    # Reuse the exact plotting helper used in DQN (rewards & episode lengths)
    plot_training_statistics(
        episode_rewards=ep_rewards,
        episode_lengths=ep_lengths,
        title=title,
        save_path=save_prefix,
        window=25,
    )
else:
    # Local minimal plots if helper isn't available
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots(figsize=(8, 4), dpi=110)
    ax1.plot(ep_rewards)
    ax1.set_title(title + " — Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    fig1.tight_layout()
    fig1.savefig(save_prefix + "_rewards.png", dpi=300)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=110)
    ax2.plot(ep_lengths)
    ax2.set_title(title + " — Episode Lengths")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Length")
    fig2.tight_layout()
    fig2.savefig(save_prefix + "_lengths.png", dpi=300)
    plt.close(fig2)

# Also store losses locally for convenience
_local_plot_losses(pi_losses, v_losses, save_path=save_prefix)

# -----------------
# Save models (DQN-style saver if present)
# -----------------
if _have_saver:
    save_model_pytorch(policy, os.path.join(cfg.results_dir, "vpg_policy"))
    save_model_pytorch(value_net, os.path.join(cfg.results_dir, "vpg_value"))
    # ONNX export (optional): only if your saver supports continuous policies
    try:
        save_lunar_model_onnx(policy, os.path.join(cfg.results_dir, "vpg_policy"))
        save_lunar_model_onnx(value_net, os.path.join(cfg.results_dir, "vpg_value"))
    except Exception as e:
        print(f"ONNX export skipped or failed: {e}")
else:
    torch.save(policy.state_dict(), os.path.join(cfg.results_dir, "vpg_policy.pt"))
    torch.save(value_net.state_dict(), os.path.join(cfg.results_dir, "vpg_value.pt"))
    print(f"Saved models to {cfg.results_dir}/[vpg_policy.pt, vpg_value.pt]")

# Cleanup
env.close()
