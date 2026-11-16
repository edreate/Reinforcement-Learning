"""Seddler at a table of any size, for one network: the environment and the export wrapper.

Shared by the Seddler trainers (``src/deep-q-learning/seddler``,
``src/proximal-policy-optimization/seddler``).

Each episode deals a game against 1, 2 or 3 bots, drawn at random. The tables' observations
(614, 808 or 1002 numbers) and actions (328, 347 or 366) differ, but every feature and action
of a smaller table has the same name in the 4-player layout. So :class:`AnyTableSeddlerEnv`
places each table's vector in the 4-player one (an empty seat's features stay 0) and numbers
its actions as the 4-player table does. One network, 1002 in and 366 scores out, plays every
size; :class:`TableNetwork` turns it back into one table size's for the exported bot.

About three steps in four have a single legal action (roll the dice, discard, ...). The
environment makes those itself (``skip_forced_moves``), so the agent only sees decisions and a
game is ~70 steps instead of ~250.
"""

from collections.abc import Iterable
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from seddler.codecs.catanatron import actions as catanatron_actions
from seddler.codecs.catanatron import observation as catanatron_observation
from seddler.env import BotMaker, SeddlerEnv
from seddler.policies import HeuristicBot, RandomBot

MAX_PLAYERS = 4
OPPONENT_BOTS: dict[str, BotMaker] = {"random": RandomBot, "heuristic": HeuristicBot}


class AnyTableSeddlerEnv(gym.Env[NDArray[np.float32], np.int64]):
    """A Seddler game against 1 to 3 bots, drawn each episode, in the 4-player layout.

    ``info["valid_actions"]`` holds 4-player action numbers, the ones ``step`` takes, and
    ``info["num_players"]`` the table size of the episode.
    """

    def __init__(
        self,
        opponent_counts: Iterable[int] = (1, 2, 3),
        opponent: BotMaker = RandomBot,
        *,
        skip_forced_moves: bool = True,
    ) -> None:
        self.opponent_counts = tuple(opponent_counts)
        self.skip_forced_moves = skip_forced_moves
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
        state, _, _, _, info = self._skip_forced(state, 0.0, False, False, info)
        return self._observation(state), self._info(info)

    def step(
        self, action: int | np.int64
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        table_action = int(self.table_action[self.players][int(action)])
        if table_action < 0:  # not at this table, e.g. robbing seat 3 of 2
            table_action = self.env.num_actions  # never valid: earns the invalid reward
        state, reward, terminated, truncated, info = self._skip_forced(
            *self.env.step(table_action)
        )
        return self._observation(state), reward, terminated, truncated, self._info(info)

    def _skip_forced(
        self,
        state: NDArray[np.float32],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Make the moves that are the only legal one, adding their rewards to reward."""
        while (
            self.skip_forced_moves
            and not (terminated or truncated)
            and len(info["valid_actions"]) == 1
        ):
            state, more, terminated, truncated, info = self.env.step(
                info["valid_actions"][0]
            )
            reward += more
        return state, reward, terminated, truncated, info

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


class TableNetwork(nn.Module):
    """A 4-player network seen as one table size's: its features in, its actions out.

    Seddler runs a bot's ONNX model on the table's own observation and plays the legal
    action with the highest score, so the layout mapping goes into the exported graph.
    """

    def __init__(self, net: nn.Module, env: AnyTableSeddlerEnv, players: int) -> None:
        super().__init__()
        self.net = net
        features = env.feature_index[players]
        place = torch.zeros(len(features), env.observation_space.shape[0])
        place[torch.arange(len(features)), torch.as_tensor(features)] = 1.0
        self.register_buffer("place", place)  # (table features, 4-player features)
        self.register_buffer("actions", torch.as_tensor(env.action_index[players]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x @ self.place).index_select(1, self.actions)
