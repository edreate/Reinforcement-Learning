"""Deep Q-Learning on Seddler, a hex-tile trading and building board game.

The DQN of ``src/deep-q-learning/01_deep_q_learning.ipynb`` with the environment
swapped: LunarLander becomes ``seddler.env.SeddlerEnv``, at a table of any size
(``src/utils/seddler_table.py``: one network, 1002 numbers in and 366 Q-values out, plays
2, 3 and 4 players). What changes because of the game:

1. **Action masking.** In most states only a few of the 366 actions are legal.
   Exploring draws from the legal ones, the greedy action and the TD target's max look
   only at legal Q-values, and each transition keeps the next state's legal actions.
2. **Keeping the Q-values from running away.** The notebook's DQN, unchanged, diverges
   here: after 1500 games it expected returns near 480 where a won game pays about 4,
   and played worse than random moves. Most (state, action) pairs are never tried, their
   Q-values are noise, and the target's max over the legal ones picks the noise up and
   feeds it back. Three things stop that: n-step returns (``n_step`` real rewards before
   the target leans on the network), Double DQN, and LayerNorm in the network. The
   training log prints ``Q``, the mean value the network expects, to watch for it.
3. **Input and reward scale.** Features are standardized with the warm-up games' mean and
   spread, and rewards are scaled by ``reward_scale`` before they reach the network (the
   game pays them in game units: +-200 for the outcome, 20 a victory point); the printed
   and plotted episode rewards are unscaled.

**How far it gets.** DQN learns here, but slowly and not well: after 4000 games against
random bots it won 44% / 38% / 23% of 150 evaluation games at 2 / 3 / 4 players (uniform
random moves win about 7%, an equal player 50% / 33% / 25%). PPO does much better on this
game: see ``src/proximal-policy-optimization/seddler/ppo_seddler.py``.

Run from the repository root::

    uv run python src/deep-q-learning/seddler/dqn_seddler.py
    uv run python src/deep-q-learning/seddler/dqn_seddler.py --episodes 200 --eval-episodes 20

The trained network is saved as PyTorch weights and as one Seddler bot per table size
(``dqn-2p``, ``dqn-3p``, ``dqn-4p``, each a ``card.json`` and a ``model.onnx``). Copy
those folders into Seddler's ``models/`` folder to play against them.
"""

import argparse
import json
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
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
OUTPUT_DIR = REPO_ROOT / "training_output_seddler" / "deep_q_network_seddler"

# =============================================================================
# 1. Gymnasium Environment Setup
# =============================================================================

# AnyTableSeddlerEnv (src/utils/seddler_table.py): a game against 1 to 3 bots in the
# 4-player layout, which makes the moves that are the only legal one itself.

# =============================================================================
# 2. Deep Q Network Components for Training
# =============================================================================

# --- 2.1 Replay Buffer ---


@dataclass
class Transition:
    """A single experience step in the environment."""

    state: NDArray[np.float32]
    action: int
    next_state: NDArray[np.float32]
    reward: float
    done: bool
    #: The legal actions in ``next_state`` (all False when it is final).
    next_mask: NDArray[np.bool_]
    #: What ``next_state``'s value is worth from ``state``: gamma ** (steps between them).
    discount: float


class ReplayBuffer:
    """Fixed-size buffer to store and sample transitions for experience replay.

    The transitions live in preallocated arrays (a ring: the oldest is overwritten), so a
    batch is one fancy-index per field instead of a stack of Python objects.
    """

    def __init__(self, capacity: int, n_obs: int, n_actions: int) -> None:
        """Initialize the buffer with a maximum capacity."""
        self.capacity = capacity
        self.states = np.zeros((capacity, n_obs), dtype=np.float32)
        self.next_states = np.zeros((capacity, n_obs), dtype=np.float32)
        self.next_masks = np.zeros((capacity, n_actions), dtype=np.bool_)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, transition: Transition) -> None:
        """Store a new transition in the buffer."""
        i = self.position
        self.states[i] = transition.state
        self.next_states[i] = transition.next_state
        self.next_masks[i] = transition.next_mask
        self.actions[i] = transition.action
        self.rewards[i] = transition.reward
        self.dones[i] = transition.done
        self.discounts[i] = transition.discount
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[NDArray[Any], ...]:
        """Randomly sample a batch: states, actions, rewards, next states, masks, dones, discounts."""
        i = np.random.randint(0, self.size, size=batch_size)  # noqa: NPY002
        return (
            self.states[i],
            self.actions[i],
            self.rewards[i],
            self.next_states[i],
            self.next_masks[i],
            self.dones[i],
            self.discounts[i],
        )

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self.size


# --- 2.2 Neural Network: Online and Target Q-Network ---


MIN_OBS_STD = 0.1


class QNetwork(nn.Module):
    """
    Simple MLP that maps a state vector to Q-values for each discrete action.

    The game's features are counts on different scales (19 cards in the bank, a 0/1 flag
    for a road), so the input is standardized first: ``(x - obs_mean) / obs_std``. The two
    are buffers, set once from the warm-up games (:meth:`set_normalization`) and saved and
    exported with the weights.

    Args:
        n_obs: state dimension
        n_actions: number of discrete actions
        hidden_sizes: tuple with hidden layer sizes
    """

    def __init__(
        self, n_obs: int, n_actions: int, hidden_sizes: tuple[int, int] = (512, 512)
    ) -> None:
        super().__init__()
        h1, h2 = hidden_sizes
        self.register_buffer("obs_mean", torch.zeros(n_obs))
        self.register_buffer("obs_std", torch.ones(n_obs))
        self.net = nn.Sequential(
            nn.Linear(n_obs, h1),
            nn.LayerNorm(h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.ReLU(),
            nn.Linear(h2, n_actions),
        )

    def set_normalization(self, states: NDArray[np.float32]) -> None:
        """Take the mean and spread of each feature from ``states`` (N, n_obs)."""
        mean = torch.as_tensor(states.mean(axis=0), dtype=torch.float32)
        std = torch.as_tensor(states.std(axis=0), dtype=torch.float32)
        self.obs_mean.copy_(mean)
        # A feature that is nearly always 0 in the warm-up (a city on one given node) has a
        # tiny spread; dividing by it would turn the 1 it takes later into hundreds.
        self.obs_std.copy_(std.clamp(min=MIN_OBS_STD))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.obs_mean) / self.obs_std)


@torch.no_grad()
def soft_update(target_net: nn.Module, online_net: nn.Module, tau: float) -> None:
    """Polyak averaging."""
    for p_t, p in zip(target_net.parameters(), online_net.parameters(), strict=False):
        p_t.data.lerp_(p.data, tau)


# --- 2.3 Exploration: Epsilon Greedy Policy ---


class EpsilonScheduler:
    """Exponential decay schedule."""

    def __init__(self, eps_start: float, eps_end: float, decay: int) -> None:
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay = decay
        self.time_step = 0  # step counter

    def step(self) -> float:
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -self.time_step / self.decay
        )
        self.time_step += 1
        return eps


def greedy_action(
    q_net: nn.Module,
    state_np: NDArray[np.float32],
    valid_actions: list[int],
    device: torch.device,
) -> int:
    """The legal action with the highest Q-value."""
    state_t = torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(
        0
    )  # (1, n_obs)
    with torch.inference_mode():
        q_values = q_net(state_t)[0]  # (n_actions,)
        valid = torch.as_tensor(valid_actions, dtype=torch.long, device=device)
        return valid_actions[int(torch.argmax(q_values[valid]).item())]


def select_action_epsilon_greedy(
    q_net: nn.Module,
    state_np: NDArray[np.float32],
    valid_actions: list[int],
    eps: float,
    device: torch.device,
) -> int:
    """Choose a legal action using epsilon-greedy.

    Args:
        q_net:   policy network
        state_np: 1D state array (n_obs,)
        valid_actions: the legal actions now
        eps: exploration rate
        device: torch device for model forward
    Returns:
        action (int)
    """

    # Exploration: Take a random legal action
    if random.random() < eps:
        return random.choice(valid_actions)

    # Exploitation: Choose the best legal action
    return greedy_action(q_net, state_np, valid_actions, device)


# --- 2.4 Huber Loss ---

criterion = nn.SmoothL1Loss()


# =============================================================================
# 3. The DQN Algorithm - Step by Step
# =============================================================================


@dataclass
class DQNTrainingConfig:
    seed: int = 42
    total_episodes: int = 4000  # one episode is one whole game
    replay_buffer_capacity: int = 200_000
    batch_size: int = 256
    gamma: float = 0.99
    learning_rate: float = 3e-4
    tau: float = 5e-3
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay: int = 40_000  # in decisions (a game is 50 to 100 of them)
    grad_clip_norm: float | None = 10.0  # set None to disable
    train_every: int = 2  # decisions per gradient step
    n_step: int = 5  # rewards summed before the target bootstraps from the network
    double_dqn: bool = (
        True  # the online network picks the next action, the target rates it
    )
    warmup_steps: int = (
        5_000  # random decisions before learning: they set the input scale
    )
    eval_episodes: int = 300
    max_steps_per_episode: int = 2000  # a game is cut off at its own turn limit first
    reward_scale: float = 0.01  # game units (+-200 a game, 20 a point) to the network's
    opponent_counts: tuple[int, ...] = (1, 2, 3)  # bots at the table, drawn per game
    opponent: str = "random"  # "random" or "planner"


def compute_td_targets(
    online_q_net: nn.Module,
    target_q_net: nn.Module,
    next_states: torch.Tensor,
    next_masks: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    discounts: torch.Tensor,
    *,
    double_dqn: bool = True,
) -> torch.Tensor:
    """Compute one-step TD targets (no gradients through either network).

    The max over the next state's Q-values takes only its legal actions. With
    ``double_dqn`` the online network picks that action and the target network rates it
    (Double DQN): a plain max over noisy estimates is too high, and with a different
    handful of legal actions in every state that bias does not cancel out.

    Shapes:
        next_states: (B, n_obs)
        next_masks:  (B, n_actions) bool, all False where the state is final
        rewards:     (B,)
        dones:       (B,) in {0,1}
        discounts:   (B,) gamma ** n, n the steps from the state to its next state
        returns:     (B,)
    """
    with torch.no_grad():
        target_next_q = target_q_net(next_states)  # (B, n_actions)
        if double_dqn:
            online_next_q = online_q_net(next_states).masked_fill(
                ~next_masks, -torch.inf
            )
            best_next = online_next_q.argmax(dim=1, keepdim=True)  # (B, 1)
            next_state_values = target_next_q.gather(1, best_next).squeeze(1)
        else:
            next_state_values = (
                target_next_q.masked_fill(~next_masks, -torch.inf).max(dim=1).values
            )
        # a final state has no legal action and no value
        next_state_values = torch.where(
            dones.bool(), torch.zeros_like(next_state_values), next_state_values
        )
        return rewards + discounts * next_state_values  # (B,)


def dqn_loss(
    online_q_net: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute Smooth L1 (Huber) loss on TD errors.

    Shapes:
        states:   (B, n_obs)
        actions:  (B, 1) int64
        targets:  (B,)
        returns:  scalar loss
    """
    q_values = online_q_net(states)  # (B, n_actions)
    q_sa = q_values.gather(1, actions).squeeze(1)  # (B,)
    return criterion(q_sa, targets)


def sample_replay_buffer_for_training(
    replay_buffer: ReplayBuffer, training_config: DQNTrainingConfig, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """
    Sample a batch from `replay_buffer` and convert to tensors for DQN training.

    Returns:
        states_t       : (B, n_obs) float32
        actions_t      : (B, 1) long
        rewards_t      : (B,) float32
        next_states_t  : (B, n_obs) float32
        next_masks_t   : (B, n_actions) bool, the legal actions there
        dones_t        : (B,) float32  (1.0 if done else 0.0)
        discounts_t    : (B,) float32
    """
    states, actions, rewards, next_states, next_masks, dones, discounts = (
        replay_buffer.sample(training_config.batch_size)
    )
    return (
        torch.as_tensor(states, device=device),
        torch.as_tensor(actions, device=device).unsqueeze(1),
        torch.as_tensor(rewards, device=device),
        torch.as_tensor(next_states, device=device),
        torch.as_tensor(next_masks, device=device),
        torch.as_tensor(dones, device=device),
        torch.as_tensor(discounts, device=device),
    )


def optimize_dqn_step(
    *,
    replay_buffer: ReplayBuffer,
    online_q: nn.Module,
    target_q: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    training_config: DQNTrainingConfig,
    device: torch.device,
) -> float | None:
    """
    One gradient step on a minibatch from replay.

    Pipeline:
        - sample (B transitions) -> tensors on device
        - compute targets y_t using target network (legal next actions only)
        - compute loss Huber(Q_theta - y)
        - backprop, (optionally) clip grads, optimizer step
        - soft-update target by tau

    Returns:
        loss value (float) or None if not enough data.
    """
    if len(replay_buffer) < training_config.batch_size:
        return None

    states, actions, rewards, next_states, next_masks, dones, discounts = (
        sample_replay_buffer_for_training(
            replay_buffer=replay_buffer, training_config=training_config, device=device
        )
    )

    targets = compute_td_targets(
        online_q_net=online_q,
        target_q_net=target_q,
        next_states=next_states,
        next_masks=next_masks,
        rewards=rewards,
        dones=dones,
        discounts=discounts,
        double_dqn=training_config.double_dqn,
    )

    loss = dqn_loss(
        online_q_net=online_q,
        states=states,
        actions=actions,
        targets=targets,
        criterion=criterion,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    if training_config.grad_clip_norm is not None:
        nn.utils.clip_grad_norm_(
            online_q.parameters(), max_norm=training_config.grad_clip_norm
        )

    optimizer.step()
    soft_update(target_q, online_q, training_config.tau)

    return float(loss.item())


@torch.no_grad()
def mean_state_value(
    replay_buffer: ReplayBuffer, q_net: nn.Module, device: torch.device, n: int = 2048
) -> float:
    """The mean of max-legal-Q over stored states: what the network expects a game to pay.

    In the network's units (rewards times ``reward_scale``) a won game pays about 4, so a
    value far above that says the Q-values are running away from the returns.
    """
    _, _, _, next_states, next_masks, dones, _ = replay_buffer.sample(n)
    keep = dones == 0
    q = q_net(torch.as_tensor(next_states[keep], device=device))
    q = q.masked_fill(~torch.as_tensor(next_masks[keep], device=device), -torch.inf)
    return float(q.max(dim=1).values.mean().item())


def train_dqn(
    *,
    training_config: DQNTrainingConfig,
    device: torch.device,
    env: AnyTableSeddlerEnv,
    online_q: QNetwork,
    target_q: QNetwork,
    replay_buffer: ReplayBuffer,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    eps_scheduler: EpsilonScheduler,
) -> tuple[nn.Module, list[float], list[int]]:
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    wins: list[bool] = []
    n_actions = int(env.action_space.n)
    no_legal_actions = np.zeros(n_actions, dtype=np.bool_)

    total_steps = 0
    eps = 1.0
    started = time.monotonic()
    state, info = env.reset(seed=training_config.seed)

    for episode in range(training_config.total_episodes):
        if episode > 0:
            state, info = env.reset()
        episode_reward = 0.0
        pending: deque[tuple[NDArray[np.float32], int, float]] = deque()

        for steps_in_episode in range(training_config.max_steps_per_episode):  # noqa: B007
            warming_up = total_steps < training_config.warmup_steps
            if warming_up:
                action = random.choice(info["valid_actions"])
            else:
                eps = eps_scheduler.step()
                action = select_action_epsilon_greedy(
                    q_net=online_q,
                    state_np=state,
                    valid_actions=info["valid_actions"],
                    eps=eps,
                    device=device,
                )

            next_state, reward, terminated, truncated, info = env.step(action)
            # A game cut off at its turn limit counts as an end too: it was not won.
            done = bool(terminated or truncated)

            # n-step returns: a transition is stored once the n rewards after it are known
            # (or the game is over), with their discounted sum as its reward.
            pending.append(
                (state, int(action), float(reward) * training_config.reward_scale)
            )
            next_mask = (
                no_legal_actions
                if done
                else action_mask(info["valid_actions"], n_actions)
            )
            while pending and (done or len(pending) == training_config.n_step):
                first_state, first_action, _ = pending[0]
                n_step_reward = sum(
                    training_config.gamma**k * r for k, (_, _, r) in enumerate(pending)
                )
                replay_buffer.push(
                    Transition(
                        state=first_state,
                        action=first_action,
                        reward=n_step_reward,
                        next_state=next_state,
                        next_mask=next_mask,
                        done=done,
                        discount=training_config.gamma ** len(pending),
                    )
                )
                pending.popleft()

            episode_reward += reward
            state = next_state
            total_steps += 1

            if total_steps == training_config.warmup_steps:
                # the warm-up's states give each input feature its mean and spread
                online_q.set_normalization(replay_buffer.states[: len(replay_buffer)])
                target_q.load_state_dict(online_q.state_dict())
            if not warming_up and total_steps % training_config.train_every == 0:
                optimize_dqn_step(
                    replay_buffer=replay_buffer,
                    online_q=online_q,
                    target_q=target_q,
                    optimizer=optimizer,
                    criterion=criterion,
                    training_config=training_config,
                    device=device,
                )

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps_in_episode + 1)
        wins.append(info["winner"] == info["seat"])

        if (episode + 1) % 100 == 0 or episode == 0:
            avg_r = np.mean(episode_rewards[-100:])
            win_rate = np.mean(wins[-100:])
            avg_len = np.mean(episode_lengths[-100:])
            minutes = (time.monotonic() - started) / 60
            mean_q = mean_state_value(replay_buffer, online_q, device)
            print(
                f"[Episode {episode + 1:5d}]  steps={total_steps:7d}  "
                f"avg100: reward={avg_r:7.1f}  len={avg_len:5.1f}  wins={win_rate:4.0%}  "
                f"eps={eps:.3f}  Q={mean_q:6.2f}  {minutes:5.1f} min",
                flush=True,
            )

    return online_q, episode_rewards, episode_lengths


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def evaluate_policy(
    q_net: nn.Module,
    training_config: DQNTrainingConfig,
    device: torch.device,
    env: AnyTableSeddlerEnv,
) -> float:
    """
    Run evaluation episodes with greedy legal actions (eps=0).
    Returns mean return; prints the win rate at each table size.
    """
    total_reward = 0.0
    games: dict[int, list[bool]] = {}

    for _ in range(training_config.eval_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        for _ in range(training_config.max_steps_per_episode):
            action = greedy_action(q_net, state, info["valid_actions"], device)
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
    q_net: nn.Module,
    env: AnyTableSeddlerEnv,
    training_config: DQNTrainingConfig,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Write ``dqn-<n>p/`` (card.json and model.onnx) for each table size trained on."""
    q_net = q_net.to("cpu").eval()
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M")
    folder = output_dir / f"seddler_bots_{timestamp}"
    for players in sorted(env.envs):
        model = TableNetwork(q_net, env, players).eval()
        n_features = len(env.feature_index[players])
        bot_dir = folder / f"dqn-{players}p"
        bot_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(
                model,
                (torch.zeros(1, n_features),),
                (bot_dir / "model.onnx").as_posix(),
                input_names=["obs"],
                output_names=["q_values"],
                dynamic_shapes=({0: torch.export.Dim("batch")},),
                dynamo=True,
                external_data=False,
            )
        card = {
            "name": "DQN",
            "description": (
                f"Deep Q-Network, {training_config.total_episodes} games against "
                f"{training_config.opponent} bots at 2 to 4 player tables."
            ),
            "codec": "catanatron-v0",
            "num_players": players,
            "weights": "model.onnx",
            "sample": False,
            "trained_with": "Reinforcement-Learning src/deep-q-learning/seddler/dqn_seddler.py",
        }
        (bot_dir / "card.json").write_text(json.dumps(card, indent=2) + "\n")
    print(f"✅ Seddler bots written to {folder.resolve()}")
    return folder


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, help="training games (default 4000)")
    parser.add_argument(
        "--eval-episodes", type=int, help="evaluation games (default 300)"
    )
    parser.add_argument(
        "--opponent", choices=sorted(OPPONENT_BOTS), help="default random"
    )
    parser.add_argument("--output-dir", type=Path, help=f"default {OUTPUT_DIR}")
    args = parser.parse_args()

    training_config = DQNTrainingConfig()
    if args.opponent is not None:
        training_config.opponent = args.opponent
    if args.episodes is not None:
        training_config.total_episodes = args.episodes
    if args.eval_episodes is not None:
        training_config.eval_episodes = args.eval_episodes

    seed = training_config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - as in the notebook
    random.seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device: ", device)

    env = AnyTableSeddlerEnv(
        training_config.opponent_counts, OPPONENT_BOTS[training_config.opponent]
    )
    n_observations = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)
    print(f"Number of possible actions: {n_actions}")
    print(f"Number of state observations: {n_observations}")

    # 3.1 Initializations
    online_q = QNetwork(n_observations, n_actions).to(device)
    target_q = QNetwork(n_observations, n_actions).to(device)
    # before starting to train start with exact same parameters/weights
    target_q.load_state_dict(online_q.state_dict())

    replay_buffer = ReplayBuffer(
        training_config.replay_buffer_capacity, n_observations, n_actions
    )
    optimizer = optim.AdamW(
        online_q.parameters(), lr=training_config.learning_rate, amsgrad=True
    )
    eps_scheduler = EpsilonScheduler(
        training_config.eps_start, training_config.eps_end, training_config.eps_decay
    )

    # Training
    online_q, episode_rewards, episode_lengths = train_dqn(
        training_config=training_config,
        device=device,
        env=env,
        online_q=online_q,
        target_q=target_q,
        replay_buffer=replay_buffer,
        criterion=criterion,
        optimizer=optimizer,
        eps_scheduler=eps_scheduler,
    )

    # Evaluation
    target_q.to(device)
    evaluate_policy(target_q, training_config, device, env)

    # Visualize Training Results
    output_dir = args.output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_training_statistics(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        title="Deep Q-Learning on Seddler",
        save_path=str(output_dir / "deep_q_learning_seddler_training_results"),
        window=100,
        show=False,
    )

    # Save Network to Disk
    # We will save target_q network as that is what we were training indirectly
    save_model_pytorch(target_q, output_dir / "trained_dqn")
    save_seddler_bots(target_q, env, training_config, output_dir)


if __name__ == "__main__":
    main()
