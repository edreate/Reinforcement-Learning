"""Load games played in Seddler for supervised pretraining.

The data is made in the Seddler project, not here. What this loader reads: a folder of
``.npz`` shards, each a run of whole games at ONE table size, in the ``catanatron-v0``
codec as Seddler's ``SeddlerEnv`` gives it, one row per move of the recorded seat:

    obs     (N, 614 | 808 | 1002)  float   the observation    (env.step / env.reset)
    mask    (N, 328 | 347 | 366)   bool    the legal actions  (env.action_masks())
    action  (N,)                   int     the move played, a codec action number
    reward  (N,)                   float   the reward of that move (env.step)
    done    (N,)                   bool    True on the last move of each game

The table size is read from the width of ``obs``. Here each shard is

1. given the discounted return after each move (for the critic), from ``reward``/``done``;
2. placed in the 4-player layout PPO trains in (1002 in, 366 out; ``seddler_table.py``);
3. stripped of the moves with a single legal action, which PPO's environment makes itself.

Each sample is then (obs float16, mask bool, action int64, return float32). The validation
set is whole shards, so no game is split between training and validation.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from seddler_config import SRC_PATH
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset

sys.path.insert(0, str(SRC_PATH))

from utils.seddler_table import AnyTableSeddlerEnv


def discounted_returns(
    rewards: NDArray[np.float32], dones: NDArray[np.bool_], gamma: float
) -> NDArray[np.float32]:
    """The return from each move to the end of its game: G_t = r_t + gamma * G_{t+1}."""
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        if dones[t]:  # the last move of a game: nothing after it counts
            running = 0.0
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def to_four_player_layout(
    obs: NDArray, mask: NDArray, action: NDArray, layout: AnyTableSeddlerEnv
) -> tuple[NDArray[np.float16], NDArray[np.bool_], NDArray[np.int64]]:
    """One table size's rows in the 4-player layout (an empty seat's features stay 0)."""
    players = {len(f): p for p, f in layout.feature_index.items()}[obs.shape[1]]
    features, actions = layout.feature_index[players], layout.action_index[players]
    full_obs = np.zeros((len(obs), layout.observation_space.shape[0]), dtype=np.float16)
    full_obs[:, features] = obs
    full_mask = np.zeros((len(obs), int(layout.action_space.n)), dtype=np.bool_)
    full_mask[:, actions] = mask
    return full_obs, full_mask, actions[action.astype(np.int64)]


def load_shards(paths: list[Path], gamma: float, reward_scale: float) -> TensorDataset:
    """All decisions of ``paths`` as one dataset of (obs, mask, action, return)."""
    layout = AnyTableSeddlerEnv()  # the 4-player layout and each table's place in it
    columns: list[list[NDArray]] = [[], [], [], []]
    for path in paths:
        with np.load(path) as shard:
            returns = discounted_returns(
                shard["reward"] * reward_scale, shard["done"], gamma
            )
            obs, mask, action = to_four_player_layout(
                shard["obs"], shard["mask"], shard["action"], layout
            )
        decision = mask.sum(axis=1) > 1  # not a forced move
        for column, values in zip(columns, (obs, mask, action, returns), strict=True):
            column.append(values[decision])
    return TensorDataset(*(torch.from_numpy(np.concatenate(c)) for c in columns))


def make_loader(dataset: TensorDataset, batch_size: int, *, shuffle: bool) -> DataLoader:
    """Batches of ``batch_size`` taken by indexing the tensors once per batch (much faster
    than the default of one sample at a time, stacked afterwards)."""
    sampler = RandomSampler(dataset) if shuffle else range(len(dataset))
    batches = BatchSampler(sampler, batch_size, drop_last=False)
    return DataLoader(dataset, sampler=batches, batch_size=None)


def make_dataloaders(
    data_dir: Path,
    batch_size: int,
    val_fraction: float,
    gamma: float,
    reward_scale: float,
) -> tuple[DataLoader, DataLoader]:
    """Train and validation loaders over the shards of ``data_dir``."""
    paths = sorted(data_dir.glob("*.npz"))
    if len(paths) < 2:
        raise FileNotFoundError(
            f"need at least 2 .npz shards in {data_dir}, found {len(paths)} "
            "(the data is made in the Seddler project: see the README)"
        )
    num_val = min(max(1, round(len(paths) * val_fraction)), len(paths) - 1)
    train = load_shards(paths[:-num_val], gamma, reward_scale)
    val = load_shards(paths[-num_val:], gamma, reward_scale)
    print(
        f"Data: {len(paths)} shards in {data_dir}: "
        f"{len(train):,} training and {len(val):,} validation decisions"
    )
    return (
        make_loader(train, batch_size, shuffle=True),
        make_loader(val, batch_size, shuffle=False),
    )
