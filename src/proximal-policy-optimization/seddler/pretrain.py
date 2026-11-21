"""Supervised pretraining (behaviour cloning) on games played in Seddler.

The network is PPO's ``ActorCritic`` (``model.py``, sized by ``ModelConfig``), trained on:

- **actor**: cross-entropy between its logits, illegal actions masked out, and the move
  played. It learns to play as the recorded player did.
- **critic**: squared error to the discounted return that followed the move, with PPO's
  gamma and reward scale. PPO then starts with a value function that fits the policy, and
  its first updates do not undo the pretraining with noisy advantages.

The input standardization (mean and spread kept in the network) is fitted on the training
data first. The data comes from the Seddler project in the format ``dataset.py`` describes.
Every setting is in ``PretrainConfig`` (``seddler_config.py``): set it there, then run
from the repository root::

    uv run python src/proximal-policy-optimization/seddler/pretrain.py

Each run gets its own folder, ``training_output_seddler/pretrain/run_gen<N>_<date>_<time>/``
(``<N>`` is the generation it trains), with the files PPO reads: ``trained_ppo.pth`` (the
weights with the lowest validation loss so far) and ``train_config.json`` (the config,
generation, best epoch, evaluation and model spec), plus ``history.json`` (every epoch) and
the Seddler bot ``bc-gen<N>/``.

**Generations**, as in PPO: set ``init_weights`` to a previous run's ``trained_ppo.pth``,
pretrain or PPO, to continue from it; its generation is read from the
``train_config.json`` next to it and this run trains the next one. Epoch 0 is the
validation of the starting weights, so the run only saves weights that beat them. To
start PPO from a run, set ``PPOTrainingConfig.init_weights`` to its ``trained_ppo.pth``.
"""

import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 - the usual name
from common import (
    evaluate_policy,
    parent_generation,
    pick_device,
    save_seddler_bot,
    set_seed,
    write_json,
)
from dataset import make_dataloaders
from model import ActorCritic, model_spec, num_parameters
from seddler_config import SRC_PATH, ModelConfig, PretrainConfig
from torch.utils.data import DataLoader

sys.path.insert(0, str(SRC_PATH))

from utils.seddler_table import OPPONENT_BOTS, AnyTableSeddlerEnv

NORMALIZATION_CHUNK = 50_000  # decisions folded into the input statistics at a time


def batch_loss(
    model: ActorCritic,
    batch: list[torch.Tensor],
    device: torch.device,
    value_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The loss of one batch, and its parts for the log."""
    obs, mask, action, ret = (t.to(device, non_blocking=True) for t in batch)
    logits, value = model.masked_logits_and_value(obs.float(), mask)
    policy_loss = F.cross_entropy(logits, action)
    value_loss = 0.5 * (value - ret).pow(2).mean()
    loss = policy_loss + value_coef * value_loss
    stats = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "accuracy": (logits.argmax(-1) == action).float().mean().item(),
    }
    return loss, stats


@torch.no_grad()
def validate(
    model: ActorCritic, loader: DataLoader, device: torch.device, value_coef: float
) -> dict[str, float]:
    """Mean loss parts over the validation set (weighted by batch size)."""
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        loss, stats = batch_loss(model, batch, device, value_coef)
        n = len(batch[0])
        for key, value in {"loss": loss.item(), **stats}.items():
            totals[key] = totals.get(key, 0.0) + value * n
        count += n
    model.train()
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def fit_normalization(
    model: ActorCritic, loader: DataLoader, device: torch.device
) -> None:
    """Set the input mean and spread from the training data, a chunk at a time."""
    obs = loader.dataset.tensors[0]
    for start in range(0, len(obs), NORMALIZATION_CHUNK):
        model.update_normalization(
            obs[start : start + NORMALIZATION_CHUNK].to(device).float()
        )


def main() -> None:
    config = PretrainConfig()
    if config.init_weights is not None:
        # continue the generation count of the run the weights come from
        start_generation = parent_generation(config.init_weights)
        if start_generation is not None:
            config = replace(config, generation=start_generation)
    generation = config.generation + 1
    print(f"Generation: starting from gen{config.generation}, training gen{generation}")
    model_config = ModelConfig()

    set_seed(config.seed)
    device = pick_device(config.device)
    print("Using device:", device)

    train_loader, val_loader = make_dataloaders(
        config.data_dir,
        config.batch_size,
        config.val_fraction,
        config.gamma,
        config.reward_scale,
    )
    env = AnyTableSeddlerEnv(config.opponent_counts, OPPONENT_BOTS[config.opponent])
    n_obs, n_actions = env.observation_space.shape[0], int(env.action_space.n)
    model = ActorCritic(n_obs, n_actions, model_config).to(device)
    print(
        f"Model: hidden {model_config.hidden_sizes}, {num_parameters(model):,} parameters"
    )
    if config.init_weights is not None:
        # The weights were trained on their input scale: keep it, do not refit it
        print(f"Starting from {config.init_weights}")
        model.load_state_dict(torch.load(config.init_weights, map_location=device))
    else:
        fit_normalization(model, train_loader, device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = config.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(  # cosine decay to 0
        optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * step / total_steps))
    )

    # Everything of this run goes in one folder: weights, train_config.json,
    # history.json, bot
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M")
    run_dir = config.output_dir / f"run_gen{generation}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "trained_ppo.pth"
    print(f"Run folder: {run_dir}")

    bot = f"bc-gen{generation}"
    metadata: dict[str, Any] = {
        "kind": "pretrain",
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generation": generation,
        "bot": bot,
        "model": model_spec(model),
        "config": asdict(config),
        "model_config": asdict(model_config),
        "best_val_loss": None,
        "best_epoch": None,
    }
    history: list[dict[str, Any]] = []
    started = time.monotonic()

    def record(epoch: int, train: dict[str, float] | None) -> None:
        """Validate, log the epoch, and save the weights when they beat the best so far."""
        val = validate(model, val_loader, device, config.value_coef)
        minutes = (time.monotonic() - started) / 60
        history.append(
            {
                "epoch": epoch,
                "train_loss": train["loss"] if train else None,
                "train_accuracy": train["accuracy"] if train else None,
                **{f"val_{k}": v for k, v in val.items()},
                "minutes": minutes,
            }
        )
        write_json(run_dir / "history.json", history, indent=2)
        trained = (
            f"train: loss={train['loss']:.3f} acc={train['accuracy']:.1%}  "
            if train
            else "starting weights  "
        )
        print(
            f"[Epoch {epoch:3d}/{config.epochs}]  {trained}val: loss={val['loss']:.3f} "
            f"acc={val['accuracy']:.1%} value_loss={val['value_loss']:.3f}  "
            f"{minutes:5.1f} min",
            flush=True,
        )
        best = metadata["best_val_loss"]
        if best is None or val["loss"] < best:
            metadata["best_val_loss"], metadata["best_epoch"] = val["loss"], epoch
            torch.save(model.state_dict(), weights_path)
            print(f"[Val] new best {val['loss']:.3f}, saved {weights_path.name}")
        write_json(run_dir / "train_config.json", metadata, indent=2)

    record(0, None)  # the starting weights: the bar to beat
    for epoch in range(1, config.epochs + 1):
        train_totals = {"loss": 0.0, "accuracy": 0.0}
        for batch in train_loader:
            loss, stats = batch_loss(model, batch, device, config.value_coef)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_totals["loss"] += loss.item()
            train_totals["accuracy"] += stats["accuracy"]
        record(epoch, {k: v / len(train_loader) for k, v in train_totals.items()})

    print(
        f"Best validation loss: {metadata['best_val_loss']:.3f} "
        f"at epoch {metadata['best_epoch']}"
    )
    # The best weights: evaluate them, export them as a Seddler bot
    model.load_state_dict(torch.load(weights_path, map_location=device))
    eval_reward, eval_win_rate, win_rate_by_players = evaluate_policy(
        model, env, config.eval_episodes, config.seed + 10_000
    )
    metadata["eval"] = {
        "opponent": config.opponent,
        "episodes": config.eval_episodes,
        "reward": eval_reward,
        "win_rate": eval_win_rate,
        "win_rate_by_players": win_rate_by_players,
    }
    metadata["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(run_dir / "train_config.json", metadata, indent=2)

    start = (
        f", from {Path(config.init_weights).parent.name}" if config.init_weights else ""
    )
    save_seddler_bot(
        model,
        env,
        run_dir / bot,
        name=f"BC gen{generation}",
        description=(
            f"Behaviour cloning of the Seddler games in {config.data_dir.name}{start}."
        ),
    )


if __name__ == "__main__":
    main()
