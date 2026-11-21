"""Play many games in Seddler and write them as ``.npz`` shards for pretraining.

Each shard is ``GAMES_PER_SHARD`` whole games at one table size, one row per move of the
teacher (see ``dataset.py`` for the format). Shards cycle through 2, 3 and 4 players.
"""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
from numpy.typing import NDArray
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from seddler.env import SeddlerEnv
from seddler.policies import Bot, PlannerBot

type BotMaker = Callable[[random.Random], Bot]

OUT_DIR: Final[Path] = Path("training_output_seddler/pre-training-data")
NUM_SHARDS: Final[int] = 300  # cycles through 2, 3, 4 players
GAMES_PER_SHARD: Final[int] = 300
TABLE_SIZES: Final[tuple[int, ...]] = (2, 3, 4)
TEACHER: Final[BotMaker] = PlannerBot  # whose moves the network learns
OPPONENT: Final[BotMaker] = PlannerBot  # who plays the other seats


class ShardArrays(TypedDict):
    """One shard as written to disk; N is the number of the teacher's moves."""

    obs: NDArray[np.float32]  # (N, observation size)
    mask: NDArray[np.bool_]  # (N, number of actions)
    action: NDArray[np.int64]  # (N,)
    reward: NDArray[np.float32]  # (N,)
    done: NDArray[np.bool_]  # (N,)


@dataclass(frozen=True, slots=True)
class GameResult:
    """What one finished game added to its shard."""

    moves: int
    fallbacks: (
        int  # teacher moves the codec had no number for: a random legal one instead
    )


@dataclass(slots=True)
class TableStats:
    """Running totals for one table size, for the summary at the end."""

    shards: int = 0
    games: int = 0
    moves: int = 0
    fallbacks: int = 0
    bytes_written: int = 0


@dataclass(slots=True)
class Rows:
    """The columns of a shard while it is being played."""

    obs: list[NDArray[np.float32]] = field(default_factory=list[NDArray[np.float32]])
    mask: list[NDArray[np.bool_]] = field(default_factory=list[NDArray[np.bool_]])
    action: list[int] = field(default_factory=list[int])
    reward: list[float] = field(default_factory=list[float])
    done: list[bool] = field(default_factory=list[bool])

    def to_arrays(self) -> ShardArrays:
        return ShardArrays(
            obs=np.asarray(self.obs, dtype=np.float32),
            mask=np.asarray(self.mask, dtype=np.bool_),
            action=np.asarray(self.action, dtype=np.int64),
            reward=np.asarray(self.reward, dtype=np.float32),
            done=np.asarray(self.done, dtype=np.bool_),
        )


def play_game(env: SeddlerEnv, teacher: Bot, seed: int, rows: Rows) -> GameResult:
    """One whole game, appending a row to ``rows`` for every move of the teacher."""
    obs, info = env.reset(seed=seed)
    valid: list[int] = info["valid_actions"]
    moves = 0
    fallbacks = 0
    done = False
    while not done:
        move = teacher.choose(env.game, env.seat)
        encoded: int | None = env.codec.encode(
            env.game, move
        )  # the move as a codec number
        action: int
        if encoded is None or encoded not in valid:  # no number for it: rare (~2%)
            action = random.choice(valid)
            fallbacks += 1
        else:
            action = encoded
        rows.obs.append(obs)
        rows.mask.append(env.action_masks())
        rows.action.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        valid = info["valid_actions"]
        done = terminated or truncated
        rows.reward.append(reward)
        rows.done.append(done)
        moves += 1
    return GameResult(moves=moves, fallbacks=fallbacks)


def play_shard(
    players: int, seed: int, on_game: Callable[[GameResult], None]
) -> tuple[ShardArrays, int]:
    """GAMES_PER_SHARD whole games at one table size. Calls ``on_game`` after each game.
    Returns the shard's arrays and how many teacher moves fell back to random ones."""
    env = SeddlerEnv([OPPONENT] * (players - 1))
    teacher: Bot = TEACHER(random.Random(seed))
    rows = Rows()
    fallbacks = 0
    for game in range(GAMES_PER_SHARD):
        result = play_game(env, teacher, seed + game, rows)
        fallbacks += result.fallbacks
        on_game(result)
    return rows.to_arrays(), fallbacks


def save_shard(path: Path, arrays: ShardArrays) -> None:
    np.savez_compressed(
        path,
        obs=arrays["obs"],
        mask=arrays["mask"],
        action=arrays["action"],
        reward=arrays["reward"],
        done=arrays["done"],
    )


def bot_name(maker: BotMaker) -> str:
    return maker.__name__ if isinstance(maker, type) else repr(maker)


class CollectionProgress:
    """Three bars: shards written, games played in all, games played in this shard."""

    def __init__(self, console: Console, total_games: int) -> None:
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description:<22}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.fields[moves]:>11,} moves"),
            TimeElapsedColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(),
            console=console,
        )
        self.shards: TaskID = self.progress.add_task("Shards", total=NUM_SHARDS, moves=0)
        self.games: TaskID = self.progress.add_task("Games", total=total_games, moves=0)
        self.shard: TaskID = self.progress.add_task(
            "Shard", total=GAMES_PER_SHARD, moves=0
        )
        self.total_moves = 0
        self.shard_moves = 0

    def __enter__(self) -> "CollectionProgress":
        self.progress.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.progress.stop()

    def start_shard(self, label: str) -> None:
        self.shard_moves = 0
        self.progress.reset(
            self.shard, total=GAMES_PER_SHARD, description=label, moves=0
        )

    def on_game(self, result: GameResult) -> None:
        self.shard_moves += result.moves
        self.total_moves += result.moves
        self.progress.update(self.shard, advance=1, moves=self.shard_moves)
        self.progress.update(self.games, advance=1, moves=self.total_moves)

    def finish_shard(self, message: str) -> None:
        self.progress.update(self.shards, advance=1, moves=self.total_moves)
        self.progress.console.log(message)


def summary_table(stats: dict[int, TableStats], seconds: float) -> Table:
    table = Table(title=f"Collected in {seconds / 60:.1f} min → {OUT_DIR}")
    for column in (
        "players",
        "shards",
        "games",
        "moves",
        "moves/game",
        "fallback",
        "size",
    ):
        table.add_column(column, justify="right")
    total = TableStats()
    for players, s in sorted(stats.items()):
        total.shards += s.shards
        total.games += s.games
        total.moves += s.moves
        total.fallbacks += s.fallbacks
        total.bytes_written += s.bytes_written
        table.add_row(str(players), *_stat_cells(s))
    table.add_section()
    table.add_row("all", *_stat_cells(total), style="bold")
    return table


def _stat_cells(s: TableStats) -> tuple[str, str, str, str, str, str]:
    return (
        f"{s.shards:,}",
        f"{s.games:,}",
        f"{s.moves:,}",
        f"{s.moves / max(s.games, 1):.1f}",
        f"{s.fallbacks / max(s.moves, 1):.2%}",
        f"{s.bytes_written / 2**20:,.1f} MB",
    )


def main() -> None:
    console = Console()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_games = NUM_SHARDS * GAMES_PER_SHARD
    console.print(
        f"[bold]Collecting {total_games:,} games[/] in {NUM_SHARDS} shards of "
        f"{GAMES_PER_SHARD} (tables of {', '.join(map(str, TABLE_SIZES))}), "
        f"teacher {bot_name(TEACHER)} vs {bot_name(OPPONENT)} → {OUT_DIR}"
    )

    stats: dict[int, TableStats] = {p: TableStats() for p in TABLE_SIZES}
    started = time.perf_counter()
    with CollectionProgress(console, total_games) as bars:
        for shard in range(NUM_SHARDS):
            # Interleave the table sizes: the last shards by name become validation
            players = TABLE_SIZES[shard % len(TABLE_SIZES)]
            path = OUT_DIR / f"shard_{shard:05d}.npz"
            bars.start_shard(f"{path.name} ({players}p)")
            arrays, fallbacks = play_shard(
                players, seed=shard * GAMES_PER_SHARD, on_game=bars.on_game
            )
            save_shard(path, arrays)

            moves = bars.shard_moves
            table = stats[players]
            table.shards += 1
            table.games += GAMES_PER_SHARD
            table.moves += moves
            table.fallbacks += fallbacks
            table.bytes_written += path.stat().st_size
            bars.finish_shard(
                f"[green]✓[/] {path.name}  {players}p  {GAMES_PER_SHARD} games  "
                f"{moves:,} moves ({moves / GAMES_PER_SHARD:.1f}/game)  "
                f"fallback {fallbacks / max(moves, 1):.1%}  "
                f"{path.stat().st_size / 2**20:.1f} MB"
            )

    console.print(summary_table(stats, time.perf_counter() - started))


if __name__ == "__main__":
    main()
