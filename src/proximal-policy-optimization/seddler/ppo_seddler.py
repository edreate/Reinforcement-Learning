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

Set the run up in ``PPOTrainingConfig`` (``seddler_config.py``; see "How to train PPO" in
the README next to this file), then run from the repository root::

    uv run python src/proximal-policy-optimization/seddler/ppo_seddler.py

Each run folder gets ``trained_ppo.pth`` (a state_dict), ``train_config.json`` (the config,
generation, best evaluation and the model spec that ``model.load_model`` rebuilds the network
from), ``rewards.json`` (the training reward of every update and every evaluation) and the
Seddler bot ``ppo-gen<N>/`` (``bot.json`` and ``ppo-gen<N>-2p.onnx`` / ``-3p`` /
``-4p``). Copy that folder into Seddler's ``models/`` folder to play against it.

To start from supervised pretraining (``pretrain.py``), set ``init_weights`` to its
``trained_ppo.pth``.
The network (``model.py``) and its size (``ModelConfig``) are the same in both.
"""

import multiprocessing as mp
import sys
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common import (
    evaluate_policy,
    parent_generation,
    pick_device,
    save_seddler_bot,
    set_seed,
    write_json,
)
from model import ActorCritic, model_spec, num_parameters
from numpy.typing import NDArray
from seddler_config import SRC_PATH, ModelConfig, PPOTrainingConfig

sys.path.insert(0, str(SRC_PATH))

from utils.plot_rl_results import plot_training_statistics
from utils.seddler_table import (
    OPPONENT_BOTS,
    AnyTableSeddlerEnv,
    action_mask,
)

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
# 2. The PPO Algorithm - Step by Step
#    (the network, ActorCritic, is in model.py)
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
    eval_env: AnyTableSeddlerEnv,
    run_dir: Path,
) -> tuple[ActorCritic, list[float], list[int]]:
    """Train; every ``eval_every`` updates evaluate, and save the weights when they beat
    the best evaluation so far (the first one is of the starting weights). Every update's
    training reward and every evaluation go to ``rewards.json``; the run's config and best
    evaluation to ``train_config.json``. The best weights are loaded back into ``model``
    at the end.
    """
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

    weights_path = run_dir / "trained_ppo.pth"
    metadata: dict[str, Any] = {
        "kind": "ppo",
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generation": training_config.generation + 1,
        "bot": f"ppo-gen{training_config.generation + 1}",
        "num_updates": num_updates,
        "model": model_spec(model),
        "config": asdict(training_config),
        "best_eval_reward": None,
        "best_update": None,
    }
    rewards_log: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}

    def evaluate_and_save(update: int) -> None:
        eval_reward, eval_win_rate, win_rate_by_players = evaluate_policy(
            model, eval_env, training_config.eval_episodes, training_config.seed + 10_000
        )
        rewards_log["eval"].append(
            {
                "update": update,
                "steps": update * rollout_size,
                "reward": eval_reward,
                "win_rate": eval_win_rate,
                "win_rate_by_players": win_rate_by_players,
            }
        )
        write_json(run_dir / "rewards.json", rewards_log)
        best = metadata["best_eval_reward"]
        if best is None or eval_reward > best:
            metadata["best_eval_reward"], metadata["best_update"] = eval_reward, update
            torch.save(model.state_dict(), weights_path)
            print(f"[Eval] new best {eval_reward:.2f}, saved {weights_path.name}")
        write_json(run_dir / "train_config.json", metadata, indent=2)

    evaluate_and_save(0)  # the starting weights: the bar to beat

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
        update_rewards: list[float] = []  # the games that ended during this rollout
        update_wins: list[bool] = []
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
                update_rewards.append(episode["reward"])
                update_wins.append(episode["won"])

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

        steps_done = (update + 1) * rollout_size
        minutes = (time.monotonic() - started) / 60
        rewards_log["train"].append(
            {
                "update": update + 1,
                "steps": steps_done,
                "games": len(episode_rewards),
                "update_games": len(update_rewards),
                "update_reward": float(np.mean(update_rewards))
                if update_rewards
                else None,
                "update_win_rate": float(np.mean(update_wins)) if update_wins else None,
                "last500_reward": float(np.mean(recent_rewards or [0.0])),
                "last500_win_rate": float(np.mean(recent_wins or [False])),
                **stats,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "minutes": minutes,
            }
        )
        write_json(run_dir / "rewards.json", rewards_log)

        if (update + 1) % 10 == 0 or update == 0 or update == num_updates - 1:
            print(
                f"[Update {update + 1:4d}/{num_updates}]  steps={steps_done:8d}  "
                f"games={len(episode_rewards):6d}  "
                f"last500: reward={np.mean(recent_rewards or [0.0]):7.1f}  "
                f"wins={np.mean(recent_wins or [False]):4.0%}  "
                f"entropy={stats['entropy']:.2f}  value_loss={stats['value_loss']:.3f}  "
                f"{minutes:5.1f} min",
                flush=True,
            )

        if (update + 1) % training_config.eval_every == 0 or update == num_updates - 1:
            evaluate_and_save(update + 1)

    print(
        f"Best evaluation: {metadata['best_eval_reward']:.2f} "
        f"at update {metadata['best_update']}"
    )
    metadata["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(run_dir / "train_config.json", metadata, indent=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model, episode_rewards, episode_lengths


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    training_config = PPOTrainingConfig()
    if training_config.init_weights is not None:
        # continue the generation count of the run the weights come from
        generation = parent_generation(training_config.init_weights)
        if generation is not None:
            training_config.generation = generation
    print(
        f"Generation: starting from gen{training_config.generation}, "
        f"training gen{training_config.generation + 1}"
    )

    set_seed(training_config.seed)

    # The game processes take most of the time, so the device matters little. Measured
    # on an M-series Mac: picking moves for 8 games is faster on the CPU (0.6 ms vs
    # 0.9 ms on MPS, which pays a copy each way), a minibatch update is faster on the
    # GPU (2 ms vs 10 ms).
    device = pick_device(training_config.device)
    print("Using device: ", device)
    rollout_size = training_config.num_envs * training_config.num_steps
    print(
        f"Rollout: {training_config.num_envs} games"
        f" x {training_config.num_steps} decisions = {rollout_size} per update,"
        f" {max(training_config.total_timesteps // rollout_size, 1)} updates"
    )

    eval_env = AnyTableSeddlerEnv(
        training_config.opponent_counts, OPPONENT_BOTS[training_config.opponent]
    )
    n_observations = eval_env.observation_space.shape[0]
    n_actions = int(eval_env.action_space.n)
    print(f"Number of possible actions: {n_actions}")
    print(f"Number of state observations: {n_observations}")

    model_config = ModelConfig()
    model = ActorCritic(n_observations, n_actions, model_config).to(device)
    print(
        f"Model: {model_config.hidden_sizes} hidden, {num_parameters(model):,} parameters"
    )
    if training_config.init_weights is not None:
        print(f"Starting from {training_config.init_weights}")
        model.load_state_dict(
            torch.load(training_config.init_weights, map_location=device)
        )
    optimizer = optim.Adam(
        model.parameters(), lr=training_config.learning_rate, eps=1e-5
    )

    # Everything of this run goes in one folder: weights, train_config.json, rewards.json,
    # plot, bot
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M")
    generation = training_config.generation + 1
    run_dir = training_config.output_dir / f"run_gen{generation}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {run_dir}")

    envs = ParallelEnvs(training_config)
    try:
        model, episode_rewards, episode_lengths = train_ppo(
            training_config=training_config,
            device=device,
            envs=envs,
            model=model,
            optimizer=optimizer,
            eval_env=eval_env,
            run_dir=run_dir,
        )
    finally:
        envs.close()

    # Visualize Training Results
    if episode_rewards:
        plot_training_statistics(
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            title="PPO on Seddler",
            save_path=str(run_dir / "ppo_seddler_training_results"),
            window=500,
            show=False,
        )

    # The best weights (already saved as trained_ppo.pth) as a Seddler bot
    decisions = f"{training_config.total_timesteps / 1e6:g}M"
    start = (
        f", from {Path(training_config.init_weights).parent.name}"
        if training_config.init_weights
        else ""
    )
    save_seddler_bot(
        model,
        eval_env,
        run_dir / f"ppo-gen{generation}",
        name=f"PPO gen{generation}",
        description=(
            f"PPO, {decisions} decisions against {training_config.opponent} bots{start}."
        ),
    )


if __name__ == "__main__":
    main()
