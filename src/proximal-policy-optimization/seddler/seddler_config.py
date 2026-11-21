"""Every setting of the Seddler training, one dataclass per step.

    0. (Seddler project)  play many games and write them as shards (see the README)
    1. pretrain.py        PretrainConfig      supervised learning on those games
    2. ppo_seddler.py     PPOTrainingConfig   PPO, starting from the pretrained weights

Steps 1 and 2 build the same network from ``ModelConfig``, so the pretrained weights load
straight into PPO.
"""

from dataclasses import dataclass
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[2]
REPO_ROOT = SRC_PATH.parent
OUTPUT_ROOT = REPO_ROOT / "training_output_seddler"


@dataclass
class ModelConfig:
    """The actor-critic network, shared by pretraining and PPO.

    Each of actor and critic is an MLP 1002 -> hidden_sizes -> (366 logits | 1 value).
    (512, 512) is about 1.7M parameters in all; the scripts print the exact count.
    """

    hidden_sizes: tuple[int, int] = (512, 512)


@dataclass
class PPOTrainingConfig:
    seed: int = 42
    total_timesteps: int = 1_000_000  # decisions, over all environments (a game is ~70)
    # num_envs * num_steps is the rollout, one update's worth of decisions. Keep it near
    # 4096: a larger one means fewer updates for the same total_timesteps.
    num_envs: int = 16  # games played at once, a process each (about 175 MB each)
    num_steps: int = 128  # decisions per environment in a rollout
    num_minibatches: int = 8
    update_epochs: int = 4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4  # falls linearly to 0 over the run
    max_grad_norm: float = 0.5
    reward_scale: float = 0.01  # game units (+-200 a game, 20 a point) to the network's
    eval_episodes: int = 300
    eval_every: int = 500  # updates between evaluations; the best one is kept
    # A run's trained_ppo.pth to continue from (a pretrain run's too); None = scratch
    init_weights: str | None = None
    # Generation of the starting weights: 0 = scratch. When init_weights has a
    # train_config.json next to it (any pretrain or PPO run), its generation is read from
    # there. The run trains generation + 1 and writes its bot as ppo-gen<generation + 1>/.
    generation: int = 0
    opponent_counts: tuple[int, ...] = (1, 2, 3)  # bots at the table, drawn per game
    opponent: str = "planner"  # "random" or "planner"
    device: str = "auto"  # "auto" (cuda, then mps, then cpu), "cuda", "mps" or "cpu"
    output_dir: Path = OUTPUT_ROOT / "ppo_seddler"


@dataclass
class PretrainConfig:
    """Supervised learning on games played in Seddler: the actor learns the moves played
    (cross-entropy over the legal actions), the critic the discounted return that followed
    (so PPO starts with a value function that fits the policy)."""

    data_dir: Path = OUTPUT_ROOT / "data"  # the shards Seddler wrote (see dataset.py)
    val_fraction: float = 0.05  # share of the shards (whole games) kept for validation
    epochs: int = 10
    batch_size: int = 1024
    learning_rate: float = 1e-3  # cosine decay to 0 over the run
    weight_decay: float = 1e-4
    value_coef: float = 0.5
    # A run's trained_ppo.pth to continue from (an earlier pretrain or PPO run's); None =
    # scratch. Its input standardization is kept, not refitted on the data.
    init_weights: str | None = None
    # Generation of the starting weights, as in PPOTrainingConfig: 0 = scratch, read from
    # the train_config.json next to init_weights when there is one. The run trains
    # generation + 1 and writes its bot as bc-gen<generation + 1>/.
    generation: int = 0
    # The critic's targets must be the returns PPO will compute: same gamma, same scale.
    gamma: float = PPOTrainingConfig.gamma
    reward_scale: float = PPOTrainingConfig.reward_scale
    # Greedy evaluation games after training, against these bots
    eval_episodes: int = 300
    opponent: str = "planner"
    opponent_counts: tuple[int, ...] = (1, 2, 3)
    seed: int = 42
    device: str = "auto"
    output_dir: Path = OUTPUT_ROOT / "pretrain"
