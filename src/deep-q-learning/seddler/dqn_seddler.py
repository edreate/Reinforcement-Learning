"""Deep Q-Learning on Seddler, a hex-tile trading and building board game.

The DQN of ``src/deep-q-learning/01_deep_q_learning.ipynb`` with the environment
swapped: LunarLander becomes ``seddler.env.SeddlerEnv``. Two things change because of
the game:

1. **A table of any size.** Each episode deals a game against 1, 2 or 3 bots, drawn at
   random. The tables' observations (614, 808 or 1002 numbers) and actions (328, 347 or
   366) differ, but every feature and action of a smaller table has the same name in
   the 4-player layout. So :class:`AnyTableSeddlerEnv` places each table's vector in
   the 4-player one (an empty seat's features stay 0) and numbers its actions as the
   4-player table does. One network, 1002 in and 366 Q-values out, plays every size.
2. **Action masking.** In most states only a few of the 366 actions are legal.
   Exploring draws from the legal ones, the greedy action and the TD target's max look
   only at legal Q-values, and each transition keeps the next state's legal actions.

The rest (replay buffer, network, soft target update, epsilon schedule, Huber loss,
AdamW) is the notebook's. Rewards are scaled by ``reward_scale`` before they reach the
network (the game pays them in game units: +-200 for the outcome, 20 a victory point);
the printed and plotted episode rewards are unscaled.

Run from the repository root::

    uv run python src/deep-q-learning/seddler/dqn_seddler.py
    uv run python src/deep-q-learning/seddler/dqn_seddler.py --episodes 2 --eval-episodes 2

The trained network is saved as PyTorch weights and as one Seddler bot per table size
(``dqn-2p``, ``dqn-3p``, ``dqn-4p``, each a ``card.json`` and a ``model.onnx``). Copy
those folders into Seddler's ``models/`` folder to play against them.
"""

import argparse
import json
import math
import random
import sys
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from seddler.codecs.catanatron import actions as catanatron_actions
from seddler.codecs.catanatron import observation as catanatron_observation
from seddler.env import BotMaker, SeddlerEnv
from seddler.policies import HeuristicBot, RandomBot

SRC_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_PATH))

from utils.model_saver import save_model_pytorch  # noqa: E402
from utils.plot_rl_results import plot_training_statistics  # noqa: E402

REPO_ROOT = SRC_PATH.parent
OUTPUT_DIR = REPO_ROOT / "training_output_seddler" / "deep_q_network_seddler"

# =============================================================================
# 1. Gymnasium Environment Setup
# =============================================================================

MAX_PLAYERS = 4
OPPONENT_BOTS: dict[str, BotMaker] = {"random": RandomBot, "heuristic": HeuristicBot}


class AnyTableSeddlerEnv(gym.Env[NDArray[np.float32], np.int64]):
    """A Seddler game against 1 to 3 bots, drawn each episode, in the 4-player layout.

    ``info["valid_actions"]`` holds 4-player action numbers, the ones ``step`` takes, and
    ``info["num_players"]`` the table size of the episode.
    """

    def __init__(
        self, opponent_counts: Iterable[int] = (1, 2, 3), opponent: BotMaker = RandomBot
    ) -> None:
        self.opponent_counts = tuple(opponent_counts)
        all_features = catanatron_observation.features(MAX_PLAYERS)
        all_actions = catanatron_actions.action_space(MAX_PLAYERS)
        feature_number = {name: i for i, name in enumerate(all_features)}
        action_number = {action: i for i, action in enumerate(all_actions)}

        self.envs: dict[int, SeddlerEnv] = {}
        # a table's feature/action number -> the 4-player one, and actions back again
        self.feature_index: dict[int, NDArray[np.int64]] = {}
        self.action_index: dict[int, NDArray[np.int64]] = {}
        self.table_action: dict[int, NDArray[np.int64]] = {}
        for count in self.opponent_counts:
            players = count + 1
            self.envs[players] = SeddlerEnv([opponent] * count)
            self.feature_index[players] = np.array(
                [feature_number[f] for f in catanatron_observation.features(players)]
            )
            to_all = np.array(
                [action_number[a] for a in catanatron_actions.action_space(players)]
            )
            self.action_index[players] = to_all
            back = np.full(len(all_actions), -1, dtype=np.int64)
            back[to_all] = np.arange(len(to_all))
            self.table_action[players] = back

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (len(all_features),), np.float32
        )
        self.action_space = gym.spaces.Discrete(len(all_actions))
        self.players = MAX_PLAYERS

    @property
    def env(self) -> SeddlerEnv:
        """The table of the current episode."""
        return self.envs[self.players]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.players = 1 + int(self.np_random.choice(self.opponent_counts))
        state, info = self.env.reset(seed=int(self.np_random.integers(2**31)))
        return self._observation(state), self._info(info)

    def step(
        self, action: int | np.int64
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        table_action = int(self.table_action[self.players][int(action)])
        if table_action < 0:  # not at this table, e.g. robbing seat 3 of 2
            table_action = self.env.num_actions  # never valid: earns the invalid reward
        state, reward, terminated, truncated, info = self.env.step(table_action)
        return self._observation(state), reward, terminated, truncated, self._info(info)

    def _observation(self, state: NDArray[np.float32]) -> NDArray[np.float32]:
        full = np.zeros(self.observation_space.shape, dtype=np.float32)
        full[self.feature_index[self.players]] = state
        return full

    def _info(self, info: dict[str, Any]) -> dict[str, Any]:
        valid = self.action_index[self.players][info["valid_actions"]]
        return {
            **info,
            "valid_actions": [int(a) for a in valid],
            "num_players": self.players,
        }


def action_mask(valid_actions: list[int], n_actions: int) -> NDArray[np.bool_]:
    """True for the legal actions."""
    mask = np.zeros(n_actions, dtype=np.bool_)
    mask[valid_actions] = True
    return mask


# =============================================================================
# 2. Deep Q Network Components for Training
# =============================================================================

# --- 2.1 Replay Buffer ---


@dataclass
class Transition:
    """A single experience step in the environment."""

    state: NDArray[np.float32]
    action: int
    next_state: NDArray[np.float32] | None
    reward: float
    done: bool
    #: The legal actions in ``next_state`` (None when it is final).
    next_mask: NDArray[np.bool_] | None = None


class ReplayBuffer:
    """Fixed-size buffer to store and sample transitions for experience replay."""

    def __init__(self, capacity: int) -> None:
        """Initialize the buffer with a maximum capacity."""
        self.memory: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        """Store a new transition in the buffer."""
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """Randomly sample a batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return len(self.memory)


# --- 2.2 Neural Network: Online and Target Q-Network ---


class QNetwork(nn.Module):
    """
    Simple MLP that maps a state vector to Q-values for each discrete action.

    Args:
        n_obs: state dimension
        n_actions: number of discrete actions
        hidden_sizes: tuple with hidden layer sizes
    """

    def __init__(
        self, n_obs: int, n_actions: int, hidden_sizes: tuple[int, int] = (1024, 1024)
    ) -> None:
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(n_obs, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
    total_episodes: int = 500  # one episode is one whole game
    replay_buffer_capacity: int = 100_000
    batch_size: int = 128
    gamma: float = 0.99
    learning_rate: float = 1e-4
    tau: float = 5e-3
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: int = 10000
    grad_clip_norm: float | None = 10.0  # set None to disable
    eval_episodes: int = 20
    max_steps_per_episode: int = 2000  # a game is cut off at its own turn limit first
    reward_scale: float = 0.01  # game units (+-200 a game, 20 a point) to the network's
    opponent_counts: tuple[int, ...] = (1, 2, 3)  # bots at the table, drawn per game
    opponent: str = "random"  # "random" or "heuristic"


def compute_td_targets(
    target_q_net: nn.Module,
    non_final_next_states: torch.Tensor,
    non_final_next_masks: torch.Tensor,
    non_final_mask: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute one-step TD targets (no gradients through target network).

    The max over the next state's Q-values takes only its legal actions.

    Shapes:
        rewards:   (B,)
        dones:     (B,) in {0,1}
        returns:   (B,)
    """
    batch_size = rewards.shape[0]
    next_state_values = torch.zeros(batch_size, device=rewards.device)

    if non_final_next_states.numel() > 0:
        with torch.inference_mode():
            next_q = target_q_net(non_final_next_states)  # (num_non_final, n_actions)
            next_q = next_q.masked_fill(~non_final_next_masks, -torch.inf)
            max_next_q = next_q.max(dim=1).values  # (num_non_final,)
        next_state_values[non_final_mask] = max_next_q

    targets = rewards + gamma * (1.0 - dones) * next_state_values
    return targets  # (B,)


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
        states_t                 : (B, *state_shape) float32
        actions_t                : (B, 1) long
        rewards_t                : (B,) float32
        non_final_next_states_t  : (N, *state_shape) float32  [N <= B]
        non_final_next_masks_t   : (N, n_actions) bool, the legal actions there
        non_final_mask           : (B,) bool
        dones_t                  : (B,) float32  (1.0 if done else 0.0)
    """
    batch: Iterable[Transition] = replay_buffer.sample(training_config.batch_size)

    states, actions, next_states, next_masks, rewards, dones = zip(
        *[
            (t.state, t.action, t.next_state, t.next_mask, t.reward, t.done)
            for t in batch
        ],
        strict=False,
    )

    # Core tensors
    states_t = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device)

    # Mask and pack only non-final next states (and their legal actions)
    non_final_mask = torch.as_tensor(
        [ns is not None for ns in next_states], dtype=torch.bool, device=device
    )

    if non_final_mask.any().item():
        non_final_next_states_t = torch.as_tensor(
            np.stack([ns for ns in next_states if ns is not None]),
            dtype=torch.float32,
            device=device,
        )
        non_final_next_masks_t = torch.as_tensor(
            np.stack([m for m in next_masks if m is not None]),
            dtype=torch.bool,
            device=device,
        )
    else:
        # Safe placeholder (will not be indexed when mask is all False)
        state_dim = states_t.shape[1:]
        non_final_next_states_t = torch.empty(
            (0, *state_dim), dtype=torch.float32, device=device
        )
        non_final_next_masks_t = torch.empty((0, 0), dtype=torch.bool, device=device)

    return (
        states_t,
        actions_t,
        rewards_t,
        non_final_next_states_t,
        non_final_next_masks_t,
        non_final_mask,
        dones_t,
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

    (
        states,
        actions,
        rewards,
        non_final_next_states,
        non_final_next_masks,
        non_final_mask,
        dones,
    ) = sample_replay_buffer_for_training(
        replay_buffer=replay_buffer, training_config=training_config, device=device
    )

    targets = compute_td_targets(
        target_q_net=target_q,
        non_final_next_states=non_final_next_states,
        non_final_next_masks=non_final_next_masks,
        non_final_mask=non_final_mask,
        rewards=rewards,
        dones=dones,
        gamma=training_config.gamma,
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


def train_dqn(
    *,
    training_config: DQNTrainingConfig,
    device: torch.device,
    env: AnyTableSeddlerEnv,
    online_q: nn.Module,
    target_q: nn.Module,
    replay_buffer: ReplayBuffer,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    eps_scheduler: EpsilonScheduler,
) -> tuple[nn.Module, list[float], list[int]]:
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    wins: list[bool] = []
    n_actions = int(env.action_space.n)

    total_steps = 0
    state, info = env.reset(seed=training_config.seed)

    for episode in range(training_config.total_episodes):
        if episode > 0:
            state, info = env.reset()
        episode_reward = 0.0

        for steps_in_episode in range(training_config.max_steps_per_episode):  # noqa: B007
            eps = eps_scheduler.step()
            action = select_action_epsilon_greedy(
                q_net=online_q,
                state_np=state,
                valid_actions=info["valid_actions"],
                eps=eps,
                device=device,
            )

            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            transition = Transition(
                state=state,
                action=int(action),
                reward=float(reward) * training_config.reward_scale,
                next_state=next_state if not done else None,
                next_mask=(
                    action_mask(info["valid_actions"], n_actions) if not done else None
                ),
                done=done,
            )

            # Store transition in replay buffer
            replay_buffer.push(transition=transition)

            optimize_dqn_step(
                replay_buffer=replay_buffer,
                online_q=online_q,
                target_q=target_q,
                optimizer=optimizer,
                criterion=criterion,
                training_config=training_config,
                device=device,
            )

            episode_reward += reward
            state = next_state
            total_steps += 1

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps_in_episode + 1)
        wins.append(info["winner"] == info["seat"])

        if (episode + 1) % 50 == 0 or episode == 0:
            avg_r = np.mean(episode_rewards[-25:])
            win_rate = np.mean(wins[-25:])
            print(
                f"[Episode {episode + 1:4d}]  players={info['num_players']}  "
                f"len={steps_in_episode + 1:4d}  reward={episode_reward:8.2f}  "
                f"avg25={avg_r:8.2f}  wins25={win_rate:4.0%}  eps={eps:.3f}"
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
    Run a few evaluation episodes with greedy legal actions (eps=0).
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

    average_reward = total_reward / training_config.eval_episodes
    print(
        f"[Eval] mean reward over {training_config.eval_episodes} "
        f"episodes: {average_reward:.2f}"
    )
    for players, won in sorted(games.items()):
        print(
            f"[Eval] {players} players: won {sum(won)} of {len(won)} "
            f"(chance alone: {1 / players:.0%})"
        )
    return average_reward


# =============================================================================
# Save as Seddler bots
# =============================================================================


class TableQNetwork(nn.Module):
    """The 4-player network seen as one table size's: its features in, its actions out.

    Seddler runs a bot's ONNX model on the table's own observation and plays the legal
    action with the highest score, so the layout mapping goes into the exported graph.
    """

    def __init__(self, q_net: nn.Module, env: AnyTableSeddlerEnv, players: int) -> None:
        super().__init__()
        self.q_net = q_net
        features = env.feature_index[players]
        place = torch.zeros(len(features), env.observation_space.shape[0])
        place[torch.arange(len(features)), torch.as_tensor(features)] = 1.0
        self.register_buffer("place", place)  # (table features, 4-player features)
        self.register_buffer("actions", torch.as_tensor(env.action_index[players]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_net(x @ self.place).index_select(1, self.actions)


def save_seddler_bots(
    q_net: nn.Module, env: AnyTableSeddlerEnv, training_config: DQNTrainingConfig
) -> Path:
    """Write ``dqn-<n>p/`` (card.json and model.onnx) for each table size trained on."""
    q_net = q_net.to("cpu").eval()
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M")
    folder = OUTPUT_DIR / f"seddler_bots_{timestamp}"
    for players in sorted(env.envs):
        model = TableQNetwork(q_net, env, players).eval()
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
    parser.add_argument("--episodes", type=int, help="training games (default 500)")
    parser.add_argument("--eval-episodes", type=int, help="evaluation games (default 20)")
    args = parser.parse_args()

    training_config = DQNTrainingConfig()
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

    replay_buffer = ReplayBuffer(training_config.replay_buffer_capacity)
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_statistics(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        title="Deep Q-Learning on Seddler",
        save_path=str(OUTPUT_DIR / "deep_q_learning_seddler_training_results"),
        window=25,
        show=False,
    )

    # Save Network to Disk
    # We will save target_q network as that is what we were training indirectly
    save_model_pytorch(target_q, OUTPUT_DIR / "trained_dqn")
    save_seddler_bots(target_q, env, training_config)


if __name__ == "__main__":
    main()
