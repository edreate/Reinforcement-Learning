"""Proximal Policy Optimization (PPO) on Seddler, a hex-tile trading and building board game.

The environment is the one of ``src/deep-q-learning/seddler/dqn_seddler.py``: every game is
dealt at a table of 2, 3 or 4, shown in the 4-player layout (1002 numbers in, 366 actions
out), and the moves that are the only legal one are made for the agent. DQN does not get far
on it (see that file); PPO does. What PPO needs from the game:

1. **Action masking.** The policy's logits of the illegal actions are set to a large negative
   number before the softmax, so they are never drawn and carry no gradient. The mask of every
   step is kept in the rollout, since the update has to rebuild the same distribution.
2. **Many games at once.** A game is slow next to a network, so ``num_envs`` processes each
   play their own; the network picks all their moves in one batch.
3. **Input and reward scale.** Each feature is standardized with a running mean and spread
   (kept in the network, so the exported bot does the same), and rewards are scaled by
   ``reward_scale`` (the game pays them in game units: +-200 for the outcome, 20 a point).

The rest is PPO as usual: rollouts of ``num_steps`` decisions per environment, GAE(lambda)
advantages, the clipped surrogate loss, a value loss and an entropy bonus, several epochs of
minibatches per rollout, a learning rate that falls to zero.

Run from the repository root::

    uv run python src/proximal-policy-optimization/seddler/ppo_seddler.py
    uv run python src/proximal-policy-optimization/seddler/ppo_seddler.py --timesteps 20000 --eval-episodes 20
    uv run python src/proximal-policy-optimization/seddler/ppo_seddler.py --opponent heuristic

The trained policy is saved as PyTorch weights and as one Seddler bot per table size
(``ppo-2p``, ``ppo-3p``, ``ppo-4p``, each a ``card.json`` and a ``model.onnx``). Copy those
folders into Seddler's ``models/`` folder to play against them.
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray

SRC_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_PATH))

from utils.model_saver import save_model_pytorch  # noqa: E402
from utils.plot_rl_results import plot_training_statistics  # noqa: E402
from utils.seddler_table import (  # noqa: E402
    OPPONENT_BOTS,
    AnyTableSeddlerEnv,
    TableNetwork,
    action_mask,
)

REPO_ROOT = SRC_PATH.parent
OUTPUT_DIR = REPO_ROOT / "training_output_seddler" / "ppo_seddler"


@dataclass
class PPOTrainingConfig:
    seed: int = 42
    total_timesteps: int = 3_000_000  # decisions, over all environments (a game is ~70)
    num_envs: int = 8  # games played at once, a process each
    num_steps: int = 512  # decisions per environment in a rollout
    num_minibatches: int = 8
    update_epochs: int = 4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4  # falls linearly to 0 over the run
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, int] = (512, 512)
    reward_scale: float = 0.01  # game units (+-200 a game, 20 a point) to the network's
    eval_episodes: int = 300
    opponent_counts: tuple[int, ...] = (1, 2, 3)  # bots at the table, drawn per game
    opponent: str = "random"  # "random" or "heuristic"


# =============================================================================
# 1. Environments: one game per process
# =============================================================================


def env_worker(
    remote: Connection, opponent_counts: tuple[int, ...], opponent: str, seed: int
) -> None:
    """Play games in this process: step on request, and deal a new game when one ends.

    Each answer is ``(observation, legal-action mask, reward, done, episode)``. When the
    game ended on this step the observation and mask are the new game's first, and
    ``episode`` holds the finished game's reward, length, table size and whether it was won.
    """
    env = AnyTableSeddlerEnv(opponent_counts, OPPONENT_BOTS[opponent])
    n_actions = int(env.action_space.n)
    state, info = env.reset(seed=seed)
    episode_reward, episode_length = 0.0, 0
    remote.send((state, action_mask(info["valid_actions"], n_actions)))
    while True:
        action = remote.recv()
        if action is None:
            break
        state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        # A game cut off at its turn limit counts as an end too: it was not won.
        done = terminated or truncated
        episode = None
        if done:
            episode = {
                "reward": episode_reward,
                "length": episode_length,
                "players": info["num_players"],
                "won": info["winner"] == info["seat"],
            }
            state, info = env.reset()
            episode_reward, episode_length = 0.0, 0
        mask = action_mask(info["valid_actions"], n_actions)
        remote.send((state, mask, float(reward), done, episode))
    remote.close()


class ParallelEnvs:
    """``num_envs`` games, each in its own process, stepped together."""

    def __init__(self, training_config: PPOTrainingConfig) -> None:
        context = mp.get_context("spawn")
        self.remotes: list[Connection] = []
        self.processes = []
        for i in range(training_config.num_envs):
            remote, worker_remote = context.Pipe()
            process = context.Process(
                target=env_worker,
                args=(
                    worker_remote,
                    training_config.opponent_counts,
                    training_config.opponent,
                    training_config.seed + i,
                ),
                daemon=True,
            )
            process.start()
            worker_remote.close()
            self.remotes.append(remote)
            self.processes.append(process)

    def first(self) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        """The first observation and mask of every environment: (N, n_obs), (N, n_actions)."""
        states, masks = zip(*[remote.recv() for remote in self.remotes], strict=True)
        return np.stack(states), np.stack(masks)

    def step(
        self, actions: NDArray[np.int64]
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.float32],
        NDArray[np.bool_],
        list[dict[str, Any]],
    ]:
        """Play one action in each environment; returns the finished games' statistics too."""
        for remote, action in zip(self.remotes, actions, strict=True):
            remote.send(int(action))
        states, masks, rewards, dones, episodes = zip(
            *[remote.recv() for remote in self.remotes], strict=True
        )
        return (
            np.stack(states),
            np.stack(masks),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            [e for e in episodes if e is not None],
        )

    def close(self) -> None:
        for remote in self.remotes:
            remote.send(None)
        for process in self.processes:
            process.join(timeout=5)


# =============================================================================
# 2. Actor-Critic Network
# =============================================================================

MASKED_LOGIT = -1e9  # the logit of an illegal action: its probability is exactly 0
MIN_OBS_STD = 0.1  # a feature that is nearly always 0 must not be blown up
MAX_NORMALIZED_OBS = 5.0


def layer(n_in: int, n_out: int, gain: float = 2**0.5) -> nn.Linear:
    """A linear layer with orthogonal weights and zero bias, as PPO is usually set up."""
    linear = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(linear.weight, gain)
    nn.init.zeros_(linear.bias)
    return linear


class ActorCritic(nn.Module):
    """Two MLPs on the standardized state: the policy's logits and the state's value.

    ``forward`` returns the logits alone: that is what the exported Seddler bot runs (it
    plays the legal action with the highest score).
    """

    def __init__(
        self, n_obs: int, n_actions: int, hidden_sizes: tuple[int, int] = (512, 512)
    ) -> None:
        super().__init__()
        h1, h2 = hidden_sizes
        self.register_buffer("obs_mean", torch.zeros(n_obs))
        self.register_buffer("obs_var", torch.ones(n_obs))
        self.register_buffer("obs_count", torch.tensor(1e-4))
        self.actor = nn.Sequential(
            layer(n_obs, h1),
            nn.Tanh(),
            layer(h1, h2),
            nn.Tanh(),
            layer(h2, n_actions, gain=0.01),  # near-uniform policy at the start
        )
        self.critic = nn.Sequential(
            layer(n_obs, h1),
            nn.Tanh(),
            layer(h1, h2),
            nn.Tanh(),
            layer(h2, 1, gain=1.0),
        )

    @torch.no_grad()
    def update_normalization(self, states: torch.Tensor) -> None:
        """Fold a batch of states (B, n_obs) into the running mean and variance."""
        batch_mean, batch_var, batch_count = (
            states.mean(0),
            states.var(0),
            states.shape[0],
        )
        total = self.obs_count + batch_count
        delta = batch_mean - self.obs_mean
        self.obs_var.copy_(
            (
                self.obs_var * self.obs_count
                + batch_var * batch_count
                + delta**2 * self.obs_count * batch_count / total
            )
            / total
        )
        self.obs_mean.add_(delta * batch_count / total)
        self.obs_count.copy_(total)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        std = self.obs_var.sqrt().clamp(min=MIN_OBS_STD)
        return ((x - self.obs_mean) / std).clamp(-MAX_NORMALIZED_OBS, MAX_NORMALIZED_OBS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalize(x))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.normalize(x)).squeeze(-1)

    def action_and_value(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw legal actions (or take ``actions``): actions, log-probs, entropies, values.

        Shapes:
            states: (B, n_obs)   masks: (B, n_actions) bool   actions: (B,)
            returns: four tensors of shape (B,)
        """
        x = self.normalize(states)
        logits = self.actor(x).masked_fill(~masks, MASKED_LOGIT)
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, n_actions)
        probs = log_probs.exp()
        if actions is None:
            actions = torch.multinomial(probs, 1).squeeze(-1)
        log_prob = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs.masked_fill(~masks, 0.0)).sum(dim=-1)
        return actions, log_prob, entropy, self.critic(x).squeeze(-1)


# =============================================================================
# 3. The PPO Algorithm - Step by Step
# =============================================================================


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation, backwards through a rollout.

    ``dones[t]`` says the game ended on step ``t``: nothing after it counts for step ``t``.

    Shapes:
        rewards, values, dones: (T, N)    last_value: (N,) the value after the last step
        returns: advantages (T, N) and returns = advantages + values (T, N)
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(rewards.shape[0])):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
        advantages[t] = last_advantage
        next_value = values[t]
    return advantages, advantages + values


def ppo_update(
    *,
    model: ActorCritic,
    optimizer: optim.Optimizer,
    states: torch.Tensor,
    masks: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    training_config: PPOTrainingConfig,
) -> dict[str, float]:
    """Several epochs of minibatch steps on one rollout (all tensors flat: (T*N, ...)).

    Loss per minibatch:
        - policy: -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A), ratio = pi_new / pi_old
        - value:  0.5 * (V - return)^2
        - entropy bonus: keeps the policy from narrowing too early
    """
    batch_size = states.shape[0]
    minibatch_size = batch_size // training_config.num_minibatches
    stats: dict[str, float] = {}

    for _ in range(training_config.update_epochs):
        order = torch.randperm(batch_size, device=states.device)
        for start in range(0, batch_size, minibatch_size):
            i = order[start : start + minibatch_size]
            _, log_prob, entropy, value = model.action_and_value(
                states[i], masks[i], actions[i]
            )
            ratio = (log_prob - old_log_probs[i]).exp()

            advantage = advantages[i]
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            clipped = ratio.clamp(
                1 - training_config.clip_coef, 1 + training_config.clip_coef
            )
            policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
            value_loss = 0.5 * (value - returns[i]).pow(2).mean()
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + training_config.value_coef * value_loss
                - training_config.entropy_coef * entropy_mean
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
            optimizer.step()

            stats = {
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy_mean.item()),
            }
    return stats


def train_ppo(
    *,
    training_config: PPOTrainingConfig,
    device: torch.device,
    envs: ParallelEnvs,
    model: ActorCritic,
    optimizer: optim.Optimizer,
) -> tuple[ActorCritic, list[float], list[int]]:
    num_steps, num_envs = training_config.num_steps, training_config.num_envs
    n_obs, n_actions = model.obs_mean.shape[0], model.actor[-1].out_features
    rollout_size = num_steps * num_envs
    num_updates = max(training_config.total_timesteps // rollout_size, 1)

    # One rollout: (T, N, ...)
    states = torch.zeros((num_steps, num_envs, n_obs), device=device)
    masks = torch.zeros(
        (num_steps, num_envs, n_actions), dtype=torch.bool, device=device
    )
    actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    log_probs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    recent_wins: deque[bool] = deque(maxlen=500)
    recent_rewards: deque[float] = deque(maxlen=500)

    state_np, mask_np = envs.first()
    state = torch.as_tensor(state_np, device=device)
    mask = torch.as_tensor(mask_np, device=device)
    started = time.monotonic()

    for update in range(num_updates):
        # the learning rate falls linearly to 0
        optimizer.param_groups[0]["lr"] = training_config.learning_rate * (
            1.0 - update / num_updates
        )

        # --- Collect a rollout with the current policy ---
        for t in range(num_steps):
            states[t], masks[t] = state, mask
            with torch.no_grad():
                action, log_prob, _, value = model.action_and_value(state, mask)
            actions[t], log_probs[t], values[t] = action, log_prob, value

            state_np, mask_np, reward_np, done_np, finished = envs.step(
                action.cpu().numpy()
            )
            rewards[t] = torch.as_tensor(reward_np, device=device) * (
                training_config.reward_scale
            )
            dones[t] = torch.as_tensor(done_np, dtype=torch.float32, device=device)
            state = torch.as_tensor(state_np, device=device)
            mask = torch.as_tensor(mask_np, device=device)

            for episode in finished:
                episode_rewards.append(episode["reward"])
                episode_lengths.append(episode["length"])
                recent_rewards.append(episode["reward"])
                recent_wins.append(episode["won"])

        # --- Advantages and returns ---
        with torch.no_grad():
            last_value = model.value(state)
            advantages, returns = compute_gae(
                rewards,
                values,
                dones,
                last_value,
                training_config.gamma,
                training_config.gae_lambda,
            )

        # --- Update ---
        flat_states = states.reshape(rollout_size, n_obs)
        stats = ppo_update(
            model=model,
            optimizer=optimizer,
            states=flat_states,
            masks=masks.reshape(rollout_size, n_actions),
            actions=actions.reshape(rollout_size),
            old_log_probs=log_probs.reshape(rollout_size),
            advantages=advantages.reshape(rollout_size),
            returns=returns.reshape(rollout_size),
            training_config=training_config,
        )
        # the input scale follows the states seen (after the update, so that the rollout's
        # log-probs were computed with the scale the update used)
        model.update_normalization(flat_states)

        if (update + 1) % 10 == 0 or update == 0 or update == num_updates - 1:
            steps_done = (update + 1) * rollout_size
            minutes = (time.monotonic() - started) / 60
            print(
                f"[Update {update + 1:4d}/{num_updates}]  steps={steps_done:8d}  "
                f"games={len(episode_rewards):6d}  "
                f"last500: reward={np.mean(recent_rewards or [0.0]):7.1f}  "
                f"wins={np.mean(recent_wins or [False]):4.0%}  "
                f"entropy={stats['entropy']:.2f}  value_loss={stats['value_loss']:.3f}  "
                f"{minutes:5.1f} min",
                flush=True,
            )

    return model, episode_rewards, episode_lengths


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def evaluate_policy(
    model: ActorCritic,
    training_config: PPOTrainingConfig,
    env: AnyTableSeddlerEnv,
) -> float:
    """
    Run evaluation games taking the legal action with the highest logit, as the exported
    bot plays. Returns mean return; prints the win rate at each table size.
    """
    model = model.to("cpu").eval()
    total_reward = 0.0
    games: dict[int, list[bool]] = {}

    state, info = env.reset(seed=training_config.seed + 10_000)
    for episode in range(training_config.eval_episodes):
        if episode > 0:
            state, info = env.reset()
        episode_reward = 0.0
        while True:
            logits = model(torch.as_tensor(state).unsqueeze(0))[0]
            valid = info["valid_actions"]
            action = valid[int(torch.argmax(logits[valid]).item())]
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        total_reward += episode_reward
        games.setdefault(info["num_players"], []).append(info["winner"] == info["seat"])

    average_reward = total_reward / max(training_config.eval_episodes, 1)
    print(
        f"[Eval] mean reward over {training_config.eval_episodes} "
        f"episodes: {average_reward:.2f}"
    )
    for players, won in sorted(games.items()):
        print(
            f"[Eval] {players} players: won {sum(won)} of {len(won)} "
            f"({sum(won) / len(won):.0%}; chance alone: {1 / players:.0%})"
        )
    return average_reward


# =============================================================================
# Save as Seddler bots
# =============================================================================


def save_seddler_bots(
    model: ActorCritic,
    env: AnyTableSeddlerEnv,
    training_config: PPOTrainingConfig,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Write ``ppo-<n>p/`` (card.json and model.onnx) for each table size trained on."""
    model = model.to("cpu").eval()
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M")
    folder = output_dir / f"seddler_bots_{timestamp}"
    for players in sorted(env.envs):
        table_model = TableNetwork(model, env, players).eval()
        n_features = len(env.feature_index[players])
        bot_dir = folder / f"ppo-{players}p"
        bot_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(
                table_model,
                (torch.zeros(1, n_features),),
                (bot_dir / "model.onnx").as_posix(),
                input_names=["obs"],
                output_names=["logits"],
                dynamic_shapes=({0: torch.export.Dim("batch")},),
                dynamo=True,
                external_data=False,
                verbose=False,
            )
        card = {
            "name": "PPO",
            "description": (
                f"PPO, {training_config.total_timesteps} decisions against "
                f"{training_config.opponent} bots at 2 to 4 player tables."
            ),
            "codec": "catanatron-v0",
            "num_players": players,
            "weights": "model.onnx",
            "sample": False,
            "trained_with": (
                "Reinforcement-Learning "
                "src/proximal-policy-optimization/seddler/ppo_seddler.py"
            ),
        }
        (bot_dir / "card.json").write_text(json.dumps(card, indent=2) + "\n")
    print(f"✅ Seddler bots written to {folder.resolve()}")
    return folder


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timesteps", type=int, help="training decisions (default 3M)")
    parser.add_argument(
        "--eval-episodes", type=int, help="evaluation games (default 300)"
    )
    parser.add_argument("--num-envs", type=int, help="games played at once (default 8)")
    parser.add_argument(
        "--opponent", choices=sorted(OPPONENT_BOTS), help="default random"
    )
    parser.add_argument("--init", type=Path, help="weights (.pth) to start from")
    parser.add_argument("--output-dir", type=Path, help=f"default {OUTPUT_DIR}")
    args = parser.parse_args()

    training_config = PPOTrainingConfig()
    if args.timesteps is not None:
        training_config.total_timesteps = args.timesteps
    if args.eval_episodes is not None:
        training_config.eval_episodes = args.eval_episodes
    if args.num_envs is not None:
        training_config.num_envs = args.num_envs
    if args.opponent is not None:
        training_config.opponent = args.opponent

    seed = training_config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - as in the notebooks
    random.seed(seed)

    # The networks are small and the batches of moves smaller: the CPU is as fast as a GPU
    # here, and the game processes take most of the time anyway.
    device = torch.device("cpu")
    print("Using device: ", device)

    eval_env = AnyTableSeddlerEnv(
        training_config.opponent_counts, OPPONENT_BOTS[training_config.opponent]
    )
    n_observations = eval_env.observation_space.shape[0]
    n_actions = int(eval_env.action_space.n)
    print(f"Number of possible actions: {n_actions}")
    print(f"Number of state observations: {n_observations}")

    model = ActorCritic(n_observations, n_actions, training_config.hidden_sizes).to(
        device
    )
    if args.init is not None:
        model.load_state_dict(torch.load(args.init, map_location=device))
    optimizer = optim.Adam(
        model.parameters(), lr=training_config.learning_rate, eps=1e-5
    )

    envs = ParallelEnvs(training_config)
    try:
        model, episode_rewards, episode_lengths = train_ppo(
            training_config=training_config,
            device=device,
            envs=envs,
            model=model,
            optimizer=optimizer,
        )
    finally:
        envs.close()

    # Evaluation
    evaluate_policy(model, training_config, eval_env)

    # Visualize Training Results
    output_dir = args.output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if episode_rewards:
        plot_training_statistics(
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            title="PPO on Seddler",
            save_path=str(output_dir / "ppo_seddler_training_results"),
            window=500,
            show=False,
        )

    # Save Network to Disk
    save_model_pytorch(model, output_dir / "trained_ppo")
    save_seddler_bots(model, eval_env, training_config, output_dir)


if __name__ == "__main__":
    main()
