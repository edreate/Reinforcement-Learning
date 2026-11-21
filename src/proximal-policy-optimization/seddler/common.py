"""What pretraining and PPO share: seeding, the device, JSON files, evaluation games and
the export of a network as a Seddler bot."""

import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from model import ActorCritic
from seddler_config import SRC_PATH

sys.path.insert(0, str(SRC_PATH))

from utils.seddler_table import AnyTableSeddlerEnv, TableNetwork


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - as in the notebooks
    random.seed(seed)


def pick_device(name: str) -> torch.device:
    """``"auto"``: CUDA, then MPS, then the CPU. Anything else is used as given."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_json(path: Path, data: Any, indent: int | None = None) -> None:
    """Write ``data`` as JSON, whole or not at all (a stopped run leaves a valid file)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent, default=str) + "\n")
    tmp.replace(path)


def parent_generation(weights: str | Path) -> int | None:
    """The generation of the run that wrote ``weights`` (a run's ``trained_ppo.pth``), read
    from the ``train_config.json`` next to it; None when there is none."""
    path = Path(weights).parent / "train_config.json"
    if not path.exists():
        return None
    generation = json.loads(path.read_text()).get("generation")
    return None if generation is None else int(generation)


@torch.no_grad()
def evaluate_policy(
    model: ActorCritic, env: AnyTableSeddlerEnv, episodes: int, seed: int
) -> tuple[float, float, dict[str, float]]:
    """
    Play ``episodes`` games taking the legal action with the highest logit, as the exported
    bot plays. Returns mean return, win rate, and win rate per table size ({"2": ...}).
    The same seed every time, so evaluations during a run see the same games.
    """
    model = (
        copy.deepcopy(model).to("cpu").eval()
    )  # a copy: training goes on on the device
    total_reward = 0.0
    games: dict[int, list[bool]] = {}

    state, info = env.reset(seed=seed)
    for episode in range(episodes):
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

    average_reward = total_reward / max(episodes, 1)
    print(f"[Eval] mean reward over {episodes} episodes: {average_reward:.2f}")
    for players, won in sorted(games.items()):
        print(
            f"[Eval] {players} players: won {sum(won)} of {len(won)} "
            f"({sum(won) / len(won):.0%}; chance alone: {1 / players:.0%})"
        )
    all_games = [w for won in games.values() for w in won]
    by_players = {str(p): sum(won) / len(won) for p, won in sorted(games.items())}
    return average_reward, sum(all_games) / max(len(all_games), 1), by_players


def save_seddler_bot(
    model: ActorCritic,
    env: AnyTableSeddlerEnv,
    bot_dir: Path,
    name: str,
    description: str,
) -> Path:
    """Write a Seddler bot folder: ``bot.json`` and one ``<folder>-<n>p.onnx`` per table
    size trained on. Copy the folder into Seddler's ``models/`` to play it."""
    model = model.to("cpu").eval()
    bot_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, str] = {}
    for players in sorted(env.envs):
        table_model = TableNetwork(model, env, players).eval()
        n_features = len(env.feature_index[players])
        onnx_name = f"{bot_dir.name}-{players}p.onnx"
        models[str(players)] = onnx_name
        with torch.no_grad():
            torch.onnx.export(
                table_model,
                (torch.zeros(1, n_features),),
                (bot_dir / onnx_name).as_posix(),
                input_names=["obs"],
                output_names=["logits"],
                dynamic_shapes=({0: torch.export.Dim("batch")},),
                dynamo=True,
                external_data=False,
                verbose=False,
            )
    bot = {
        "name": name,
        "description": description,
        "codec": "catanatron-v0",
        "models": models,
    }
    write_json(bot_dir / "bot.json", bot, indent=2)
    print(f"✅ Seddler bot written to {bot_dir.resolve()}")
    return bot_dir
