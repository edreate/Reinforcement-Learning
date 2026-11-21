# Seddler: from DQN to PPO

What was tried to make an agent learn Seddler, in order, and what each step did.

- **Opponents:** Seddler's `RandomBot` in every run (random legal moves, weighted towards
  building so that games end). Tables of 2, 3 and 4 players, drawn per game.
- **Win rates** are greedy evaluation games at 2 / 3 / 4 players. An equal player wins
  50% / 33% / 25%. Uniform random moves win about 7% overall.
- **Machine:** M2 Pro, 12 cores.
- **Code:** DQN in `src/deep-q-learning/seddler/dqn_seddler.py`, PPO in `ppo_seddler.py`
  here, the environment both use in `src/utils/seddler_table.py`.

## Pretraining, then PPO

PPO from scratch against the heuristic bots does not learn, so the network first learns
from games played in Seddler (supervised, behaviour cloning), then PPO fine-tunes it.

| Step | Where | Settings |
|---|---|---|
| 0. Play many games with `SeddlerEnv`, write them as shards | a collection script (below) | its constants |
| 1. Supervised pretraining | `pretrain.py` | `PretrainConfig` |
| 2. PPO from the pretrained weights | `ppo_seddler.py` | `PPOTrainingConfig` |

All settings are dataclasses in `seddler_config.py`. `ModelConfig` sets the network size
(`hidden_sizes`) for both steps, and both build `ActorCritic` from `model.py`, so the
pretrained weights load straight into PPO.

| File | What it holds |
|---|---|
| `seddler_config.py` | `ModelConfig`, `PretrainConfig`, `PPOTrainingConfig` |
| `model.py` | `ActorCritic` (actor and critic MLPs, input standardization), `load_model` |
| `dataset.py` | Reads the shards, the DataLoaders for pretraining |
| `pretrain.py` | Supervised training: actor on the moves played, critic on the returns |
| `ppo_seddler.py` | PPO: parallel games, rollouts, GAE, the clipped update |
| `common.py` | Evaluation games, export as a Seddler bot, seeding, device |

### 0. Collect games with `SeddlerEnv`

The data is games played with Seddler's own environment, `seddler.env.SeddlerEnv`, not
the `AnyTableSeddlerEnv` wrapper PPO trains in: one row per move of one seat (the
*teacher*, whose moves the network learns), at the table's own size and numbering.
`dataset.py` does the conversion to PPO's layout when it loads them.

**How a game is recorded.** `SeddlerEnv([Bot] * opponents)` deals a game and plays the
other seats with those bots. At each of the teacher's moves:

1. `obs` is the observation `reset` / `step` returned, `env.action_masks()` the legal moves.
2. The teacher picks a move. A Seddler bot answers with a game move, not a number:
   `teacher.choose(env.game, env.seat)`, then `env.codec.encode(env.game, move)` gives its
   number. A few of the teacher's moves (about 2% of the old `HeuristicBot`'s) have no number
   in the codec (or not a legal one); a random legal number is played instead.
3. `env.step(action)` plays it and the opponents' replies, and returns the reward (game
   units: +200 win, -200 loss, +20 per own point, ...) and whether the game ended.

Every move is recorded, the forced ones (a single legal action) too: `SeddlerEnv` does
not skip them, and their rewards belong in the returns. The loader drops them afterwards.

**The script.** Save it as, say, `collect_games.py` and run it from the repository root
(`uv run python collect_games.py`); it writes to `training_output_seddler/data/`, the
default `PretrainConfig.data_dir`:

```python
"""Play games in Seddler and write them as .npz shards for pretrain.py."""

import random
from pathlib import Path

import numpy as np
from seddler.env import SeddlerEnv
from seddler.policies import PlannerBot

OUT_DIR = Path("training_output_seddler/data")  # PretrainConfig.data_dir
NUM_SHARDS = 300  # cycles through 2, 3, 4 players
GAMES_PER_SHARD = 100
TEACHER = PlannerBot  # whose moves the network learns
OPPONENT = PlannerBot  # who plays the other seats


def play_shard(players: int, seed: int) -> dict[str, np.ndarray]:
    """GAMES_PER_SHARD whole games at one table size, one row per move of the teacher."""
    env = SeddlerEnv([OPPONENT] * (players - 1))
    teacher = TEACHER(random.Random(seed))
    rows: dict[str, list] = {k: [] for k in ("obs", "mask", "action", "reward", "done")}
    for game in range(GAMES_PER_SHARD):
        obs, info = env.reset(seed=seed + game)
        done = False
        while not done:
            move = teacher.choose(env.game, env.seat)
            action = env.codec.encode(env.game, move)  # the move as a codec number
            if action not in info["valid_actions"]:  # no number for it: rare (~2%)
                action = random.choice(info["valid_actions"])
            rows["obs"].append(obs)
            rows["mask"].append(env.action_masks())
            rows["action"].append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rows["reward"].append(reward)
            rows["done"].append(done)
    return {
        "obs": np.asarray(rows["obs"], dtype=np.float32),
        "mask": np.asarray(rows["mask"], dtype=np.bool_),
        "action": np.asarray(rows["action"], dtype=np.int64),
        "reward": np.asarray(rows["reward"], dtype=np.float32),
        "done": np.asarray(rows["done"], dtype=np.bool_),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for shard in range(NUM_SHARDS):
        # Interleave the table sizes: the last shards by name become the validation set
        players = (2, 3, 4)[shard % 3]
        arrays = play_shard(players, seed=shard * GAMES_PER_SHARD)
        path = OUT_DIR / f"shard_{shard:05d}.npz"
        np.savez_compressed(path, **arrays)
        print(f"{path}: {players} players, {len(arrays['action']):,} moves", flush=True)


if __name__ == "__main__":
    main()
```

To learn from someone other than `PlannerBot`, change `TEACHER` (any Seddler bot, e.g.
`RandomBot`) or replace the two lines that pick `action` (a person, an exported
model through `CodecBot`). `OPPONENT` sets who the teacher plays against.

**Cost, measured on the M2 Pro** (one process, `HeuristicBot` against `HeuristicBot`, before seddler replaced both
heuristic bots with `PlannerBot`):
about 0.16 s and 5 KB on disk per game. The 30,000 games above take about 80 minutes and
150 MB; to go faster, run several copies at once, each on its own range of shard numbers
(and so its own seeds). A game is about 90 / 80 / 65 recorded moves at 2 / 3 / 4 players,
of which about 50 / 44 / 33 are decisions.

#### The shard format

A shard is one `.npz` file (`np.savez_compressed`) with five arrays of the same length
`N`, one row per move of the teacher, in the order played:

| Array | Shape | dtype | What it is |
|---|---|---|---|
| `obs` | (N, 614 / 808 / 1002) for 2 / 3 / 4 players | float32 | The observation before the move (`reset` / the previous `step`) |
| `mask` | (N, 328 / 347 / 366) | bool | `env.action_masks()`: the legal moves |
| `action` | (N,) | int64 | The move played, a number in **this table's** codec numbering (not the 4-player one) |
| `reward` | (N,) | float32 | What `env.step(action)` returned, in game units (unscaled) |
| `done` | (N,) | bool | `terminated or truncated`: True on the last move of each game |

The rules the loader relies on:

- **One table size per shard.** The size is read from the width of `obs`.
- **Whole games, back to back.** A game's rows are contiguous and its last row has
  `done = True`, so the last row of a shard is always `done = True`. A game cut off at
  the turn limit (`truncated`) ends the same way.
- **Forced moves may stay in** (rows whose `mask` has one `True`).
- **Any file names**, as long as they end in `.npz`: shards are read in sorted name order,
  and the last `val_fraction` of them (at least one) are the validation set. Interleave
  table sizes in the names, as the script does, or validation sees only one size.

Check a shard (here a 2-player shard of 10 games, from the quick test below):

```python
>>> z = np.load("training_output_seddler/data/shard_00000.npz")
>>> {k: (z[k].shape, z[k].dtype) for k in z.files}
{'obs': ((968, 614), dtype('float32')), 'mask': ((968, 328), dtype('bool')),
 'action': ((968,), dtype('int64')), 'reward': ((968,), dtype('float32')),
 'done': ((968,), dtype('bool'))}
>>> z["done"].sum()  # games in the shard
np.int64(10)
```

#### What `dataset.py` does with them

For each shard, in `load_shards`:

1. **Returns.** The discounted return after each move,
   `G_t = reward_t * reward_scale + gamma * G_{t+1}`, restarting at every `done`. `gamma`
   and `reward_scale` are PPO's (0.995 and 0.01), so the critic learns the values PPO
   will compute. Forced moves are still in at this point, so their rewards count.
2. **4-player layout.** Each observation is placed in the 1002-wide vector (an empty
   seat's features stay 0), each mask in the 366-wide one, and each action renumbered
   to its 4-player number, using `AnyTableSeddlerEnv`'s `feature_index` / `action_index`.
3. **Decisions only.** Rows with a single legal action are dropped.
4. **Samples** are `(obs float16, mask bool, action int64, return float32)`, all shards
   concatenated into one `TensorDataset` in memory: about 2.4 KB per decision, so 10,000
   games (about 440k decisions) are about 1 GB.

`make_dataloaders` splits the shards into training and validation (whole shards, so no
game is split) and batches them by indexing the tensors once per batch; the training
loader is shuffled. It stops with an error when the folder has fewer than two shards.

### 1. Pretrain

Set `PretrainConfig` in `seddler_config.py`, then run from the repository root:

```
uv run python src/proximal-policy-optimization/seddler/pretrain.py
```

**What a run does, in order**

1. **Generation.** With `init_weights` set, the generation is read from the
   `train_config.json` next to those weights; the run trains the next one (see below).
2. **Seed and device** (`seed`, `device = "auto"`: CUDA, then MPS, then CPU).
3. **Data.** All shards of `data_dir` are loaded as above. It prints the counts:
   `Data: 300 shards in ...: 1,234,567 training and 65,000 validation decisions`.
4. **Network.** `ActorCritic` from `ModelConfig` (1002 in, 366 logits and 1 value out).
   From scratch, its input standardization (mean and spread kept in the network) is
   fitted on the training observations, 50,000 at a time. From `init_weights`, those
   weights are loaded and their standardization is kept.
5. **Optimizer.** AdamW (`learning_rate`, `weight_decay`), the learning rate falling to
   0 on a cosine over all `epochs`.
6. **Epoch 0** validates the starting weights and saves them as the best so far: the bar
   every later epoch has to beat.
7. **Each epoch** goes once over the training set. The loss of a batch is
   - *actor*: cross-entropy between the logits, illegal actions masked to -1e9, and
     the move played;
   - *critic*: `0.5 * (value - return)^2`, weighted by `value_coef`.

   Then it validates, appends the epoch to `history.json`, saves `trained_ppo.pth` when
   the validation loss is the lowest so far, and rewrites `train_config.json`.
8. **At the end** the best weights are loaded back, play `eval_episodes` greedy games
   against `opponent` at `opponent_counts` tables (the same evaluation PPO runs), the
   result goes into `train_config.json`, and the network is exported as the Seddler bot
   `bc-gen<N>/`.

Pretraining the critic matters: PPO started from a trained actor and an untrained critic
gets noisy advantages, and its first updates undo the actor.

**The log**, one line per epoch (accuracy: how often the legal action with the highest
logit is the move played):

```
Generation: starting from gen0, training gen1
Using device: mps
Data: 300 shards in .../training_output_seddler/data: ... training and ... validation decisions
Model: hidden (512, 512), 1,740,655 parameters
Run folder: training_output_seddler/pretrain/run_gen1_<date>_<time>
[Epoch   0/10]  starting weights  val: loss=... acc=... value_loss=...    0.0 min
[Val] new best ..., saved trained_ppo.pth
[Epoch   1/10]  train: loss=... acc=...  val: loss=... acc=... value_loss=...    ... min
...
Best validation loss: ... at epoch ...
[Eval] mean reward over 300 episodes: ...
[Eval] 2 players: won ... of ... (...; chance alone: 50%)
✅ Seddler bot written to training_output_seddler/pretrain/run_gen1_<date>_<time>/bc-gen1
```

**What it writes**, in `training_output_seddler/pretrain/run_gen<N>_<date>_<time>/`
(`<N>` is the generation it trains):

| File | What it holds |
|---|---|
| `trained_ppo.pth` | The weights of the epoch with the lowest validation loss so far, as a `state_dict` |
| `train_config.json` | `kind: "pretrain"`, the generation and bot name, the model spec, `PretrainConfig` and `ModelConfig`, the best epoch and validation loss, and (at the end) the evaluation. Rewritten after every epoch |
| `history.json` | One entry per epoch (epoch 0 is the starting weights): training loss and accuracy, validation loss, policy loss, value loss and accuracy, minutes |
| `bc-gen<N>/` | The pretrained network as a Seddler bot: `bot.json` and `bc-gen<N>-2p.onnx` / `-3p` / `-4p` |

**What to set** (`PretrainConfig`)

| Field | What it does | Default |
|---|---|---|
| `data_dir` | The shards folder | `training_output_seddler/data` |
| `val_fraction` | Share of the shards kept for validation | 0.05 |
| `epochs` | Passes over the training set | 10 |
| `batch_size` | Decisions per gradient step | 1024 |
| `learning_rate`, `weight_decay` | AdamW; the rate falls to 0 on a cosine | 1e-3, 1e-4 |
| `value_coef` | Weight of the critic's loss | 0.5 |
| `init_weights` | A run's `trained_ppo.pth` to continue from; `None` = scratch | `None` |
| `gamma`, `reward_scale` | For the critic's returns. Leave them equal to PPO's | 0.995, 0.01 |
| `eval_episodes`, `opponent`, `opponent_counts` | The evaluation at the end | 300, `"planner"`, (1, 2, 3) |
| `seed`, `device` | | 42, `"auto"` |

`ModelConfig.hidden_sizes` sets the network for both pretraining and PPO; keep it the same
for both, or the weights will not load.

**A quick test** before a long collection: set `NUM_SHARDS = 6`, `GAMES_PER_SHARD = 10` in
the script (about 10 s), then `epochs = 2`, `eval_episodes = 4` and `val_fraction = 0.34`
in `PretrainConfig` (with 6 shards, 0.05 would still keep 1). On so little data the
validation loss may not beat epoch 0's; the run then keeps the starting weights, as it
should.

**Generations**, as in PPO: to continue from an earlier run (pretrain or PPO), set
`init_weights` in `PretrainConfig` to its weights:

```python
init_weights: str | None = (
    "training_output_seddler/pretrain/run_gen1_<date>_<time>/trained_ppo.pth"
)
```

The generation is read from the `train_config.json` next to the weights and the run
trains the next one (`bc-gen2/` here). Epoch 0 validates the starting weights, so the run
only saves weights that beat them. Their input standardization is kept, not refitted on
the new data. The optimizer and learning-rate schedule start fresh: lower `learning_rate`
when only a little more training is wanted.

### 2. PPO from the pretrained weights

Set `init_weights` in `PPOTrainingConfig` to the pretrain run's weights
(`training_output_seddler/pretrain/run_gen1_<date>_<time>/trained_ppo.pth`), then run
`ppo_seddler.py` as below. The run trains the next generation (gen2
from a gen1 pretrain) and only saves weights that beat the pretrained ones in evaluation.
For fine-tuning, a lower `learning_rate` (1e-4) and `entropy_coef` (0.001) keep PPO from
moving far from the pretrained policy in its first updates.

## How to train PPO

All PPO settings live in the `PPOTrainingConfig` dataclass in `seddler_config.py`. Edit
it, then run from the repository root:

```
uv run python src/proximal-policy-optimization/seddler/ppo_seddler.py
```

On a Mac, prefix `caffeinate -i` so it does not sleep during a long run. Each run gets
its own folder, `training_output_seddler/ppo_seddler/run_gen<N>_<date>_<time>/` (`<N>` is
the generation it trains):

| File | What it holds |
|---|---|
| `trained_ppo.pth` | The best weights, as a `state_dict` |
| `train_config.json` | The config, the generation, the best evaluation, and under `"model"` everything needed to rebuild the network (`load_model(run_dir)` in `model.py` does it) |
| `rewards.json` | `"train"`: one entry per update (reward and win rate of the games that ended in it and of the last 500, losses, entropy, learning rate). `"eval"`: one entry per evaluation (reward, win rate, win rate per table size). Rewritten after every update, so it can be plotted while the run goes |
| `ppo-gen<N>/` | The Seddler bot: `bot.json` and `ppo-gen<N>-2p.onnx` / `-3p` / `-4p`. Copy the folder into Seddler's `models/` |
| `ppo_seddler_training_results.png` | The training plot |

Every `eval_every` updates (and at the start and end) the policy plays `eval_episodes`
games on the same seed; `trained_ppo.pth` is overwritten only when the mean reward beats
the best so far, and the bot is exported from that best.

**Generations:** set `init_weights` to the previous run's `trained_ppo.pth` to continue
from it (`None` trains from scratch). The first evaluation is then of those weights, so
the new run only saves weights that beat them. `generation` in the config is the starting
weights' generation (0 = scratch); the run trains `generation + 1` and names its bot
`ppo-gen<generation + 1>`. When `init_weights` sits next to a `train_config.json`, the
generation is read from there, so chained runs count up on their own. Pretraining and PPO
share the count, so the bot names show the lineage: `bc-gen1` → `ppo-gen2` → `ppo-gen3`.

**What to set**

| Field | What it does | Set to |
|---|---|---|
| `total_timesteps` | Decisions to train for, over all games (a game is about 70) | 24M for a long run; 3M takes ~30 min at 8 games |
| `num_envs` | Games played at once, one process each (about 175 MB of RAM each) | The machine's core count or a bit more: 32 on the M2 Pro or the 32-thread i9 |
| `num_steps` | Decisions each game plays before an update | `4096 / num_envs`: 128 for 32 games, 512 for 8 |
| `device` | Where the network runs | `"auto"`: CUDA, then MPS, then CPU |
| `opponent` | Bots at the table | `"random"`, or `"planner"` once the policy beats random bots |
| `eval_episodes` | Greedy games played after training to report win rates | 300; 20 for a quick test |

**The rule to keep:** `num_envs * num_steps` is one update's batch. Keep it near 4096.
Raising `num_envs` without lowering `num_steps` makes each update bigger and the number of
updates smaller for the same `total_timesteps`, so the policy learns less per decision.

**Ready-made settings**

| Run | `total_timesteps` | `num_envs` | `num_steps` | Time on the M2 Pro |
|---|---|---|---|---|
| Smoke test (also set `eval_episodes = 20`) | 20_000 | 8 | 512 | ~1 min |
| The 3M run reported below | 3_000_000 | 8 | 512 | ~30 min |
| Long run (current values) | 24_000_000 | 32 | 128 | ~2.5 h (estimate) |

**How many games at once (measured on the M2 Pro, 12 cores, games only)**

| `num_envs` | Decisions/s | RAM of the game processes |
|---|---|---|
| 8 | 1,835 | 1.9 GB |
| 16 | 2,566 | 3.3 GB |
| 24 | 2,952 | 4.5 GB |
| 32 | 3,154 | 5.6 GB |

Gains flatten past the number of performance cores. On a machine with more cores, more
games pay off further.

**Device:** the games take most of the time, so the device matters little. On the M2
Pro, picking moves is faster on the CPU (0.6 ms vs 0.9 ms on MPS) and an update minibatch
is faster on MPS (2 ms vs 10 ms); over a run they come out about even. CUDA, which
copies to and from the GPU more cheaply, should gain a little more.

There are no command-line flags: every setting, `init_weights` and `opponent` included,
is set in `PPOTrainingConfig`.

## Result

| Algorithm | Training | Wins at 2 / 3 / 4 players | Evaluation games |
|---|---|---|---|
| DQN, as first committed | 500 games, ~19 min | did not learn (flat reward curve) | - |
| DQN, stabilised (current defaults) | 4000 games, ~28 min | 44% / 38% / 23% | 150 |
| PPO, 3M run | 3M decisions (~50,000 games), ~30 min | 95% / 86% / 79% | 300 |

PPO trained only against random bots does not beat the heuristic bots: a 1.5M-decision
policy won 1 of 90 games against `HeuristicBot` (since replaced by `PlannerBot`).

## The environment, measured first

| Fact | Value | Why it matters |
|---|---|---|
| Observation / actions (4-player layout) | 1002 numbers / 366 actions | Large, mostly 0/1 input; most actions illegal in any state |
| Steps with a single legal action | 62-76% | Roll, discard and the like are not decisions |
| Legal actions per step, mean | 3.5-4.9 | The max in the TD target is over a handful of actions |
| Game length, uniform random agent | 140-250 steps | About 60-70 decisions once forced moves are skipped |
| Games cut off at the 200-roll limit, uniform random agent | 72-78% | A random agent rarely finishes: winning needs deliberate building |
| Environment speed, one process | 1600-3000 steps/s vs random bots, 400-1400 vs heuristic | The game, not the network, limits training speed |
| Largest input value | 18 (cards in the bank) | Features are on different scales |
| Reward (game units) | +200 win, -200 loss, +20 per own point, -2 per opponent point, -1 per card discarded | Scaled by 0.01 for the network: a won game pays about 4 |

## What was learned

1. **Measure the game before tuning the algorithm.** Three steps in four are forced
   moves. Making them inside the environment shortens the horizon about four times and
   stops the replay buffer filling with non-decisions.
2. **The original DQN failed because its Q-values diverged, not because it was
   under-trained.** After 1500 games the network predicted a return near 480 for the
   first state of a game where the real discounted return was between -1 and 3. The
   greedy policy built on those values mostly ended its turn or traded with the bank
   (mean 3.7 points per game) and won 1 game in 40.
3. **Why it diverges here.** Most (state, action) pairs are never tried, so their
   Q-values are noise. The TD target takes a max over the legal ones, picks the noise
   up, and with gamma 0.99 feeds it back amplified. Double DQN alone did not stop this:
   with a soft target update the two networks are nearly the same.
4. **Compare predicted values with real returns.** Printing the mean max-legal Q beside
   the win rate (the `Q=` column in the DQN log) showed the divergence at once; the
   reward curve alone only showed "not learning".
5. **Input standardisation needs a floor.** A feature that is nearly always 0 in the
   warm-up games (a city on one given node) has a tiny spread; dividing by it turns the
   1 it takes later into hundreds.
6. **Slower updates only slow the divergence.** LayerNorm, a slower target network,
   fewer gradient steps and a lower gamma each reduced the overestimate, and none
   produced learning on their own.
7. **n-step returns are what stabilised DQN.** With 5 to 10 real rewards before the
   bootstrap, predicted values stayed near the real returns (about 1).
8. **Stable is not the same as good.** The best DQN was the diagnostic variant with
   inputs divided by 1000, which is nearly blind to the board. It learns a fixed
   preference over actions (build before trading or ending the turn), which is enough
   to beat random bots 88% / 69% / 56% but is not play. Every state-aware DQN variant
   did worse than it within the same budget.
9. **PPO fits this game better.** It needs no max over untried actions, so masking is
   enough to handle the action space, and it parallelises across processes. It passed
   the best principled DQN within about 5 minutes.
10. **A policy that beats random bots has not learned the game.** It loses almost every
    game to the heuristic bots.

## Steps, in order

| # | Change | Setup | Effect |
|---|---|---|---|
| 0 | Original DQN | 500 games, 1024x1024 MLP, raw inputs, batch 128, one gradient step per step | Flat reward curve, no learning |
| 1 | Skip forced moves; Double DQN; array replay buffer; input standardisation; longer exploration | 512x512, batch 256, buffer 200k, eps 1.0 to 0.02 over ~40k decisions, 5000 random warm-up decisions | 3-4% wins by game 1500, below random (7%), falling as exploration dropped |
| 2 | Floor of 0.1 on the input spread | as 1 | No change: 1 win in 90 evaluation games. Predicted Q about 480 vs real returns under 4 |
| 3 | Inputs divided by 1000 (diagnostic) | as 1, no other change | Q about 2, matching returns. 78% / 66% / 39% after 1500 games; 88% / 69% / 56% after 4000 |
| 4 | LayerNorm after each hidden layer | spread floor 0.1 and 1 | Q grew more slowly (about 10 at 40k steps) but still too high; 5-13% wins |
| 5a | Gradient step every 4 decisions | with LayerNorm | Q 5.5 at 900 games; 3-11% wins |
| 5b | Slower target network, tau 5e-4 | with LayerNorm | Q 2.8 at 700 games; about 5% wins |
| 5c | gamma 0.95 | with LayerNorm | Q 2.6, bounded; about 6% wins |
| 6a | n-step returns, n = 5 | with LayerNorm | Q about 1.6; 29% / 5% / 23% at 900 games |
| 6b | n-step returns, n = 10 | with LayerNorm | Q about 0.9; 14% / 26% / 15% at 900 games |
| 6c | No LayerNorm, spread floor 10, n = 1 (control) | - | Q about 85: diverges again. n-step is the part that matters |
| 7a | 4000 games, n = 10 | LayerNorm, lr 1e-4 | 36% / 21% / 10%, 43 min |
| 7b | 4000 games, n = 5, lr 3e-4, gradient step every 2 decisions | LayerNorm | 44% / 38% / 23%, 28 min. Kept as the DQN default |
| 7c | 4000 games, n = 5, no LayerNorm, spread floor 30 | partly state-blind | 64% / 61% / 31%: better only because it sees less of the board |
| 8 | PPO | 1.5M decisions, 10 processes | 93% / 83% / 64% in 13 min |
| 9 | PPO, 3M run | 3M decisions, 8 processes | 95% / 86% / 79% in 30 min |

Evaluation sizes: 90 games for steps 2-3 (1500 games), 60 for step 6, 150 for step 7
and the 4000-game run of step 3, 300 for steps 8-9. The small ones are noisy; read
them as "learning or not", not as exact rates.

## The DQN that is in the repo

| Part | Setting |
|---|---|
| Network | 1002 -> 512 -> 512 -> 366, LayerNorm and ReLU after each hidden layer |
| Input | Standardised with the mean and spread of the 5000 warm-up decisions, spread floor 0.1 |
| Target | Double DQN, 5-step returns, gamma 0.99, soft update tau 5e-3 |
| Masking | Random and greedy actions and the target's argmax all take legal actions only |
| Replay | 200,000 transitions in preallocated arrays, batch 256, one gradient step per 2 decisions |
| Optimiser | AdamW, lr 3e-4, Huber loss, gradient norm clipped at 10 |
| Exploration | eps 1.0 to 0.02, exponential, time constant 40,000 decisions |
| A cut-off game | Treated as an end: it was not won |

## The PPO that is in the repo

| Part | Setting |
|---|---|
| Networks | Separate actor and critic, each 1002 -> 512 -> 512, tanh, orthogonal init; the actor's last layer starts near zero so the policy starts near uniform over legal actions |
| Masking | Logits of illegal actions set to -1e9 before the softmax; the mask of every step is stored in the rollout and reused in the update |
| Input | Running mean and variance kept in the network (so the exported bot has them), spread floor 0.1, clipped to +-5 |
| Rollout | 32 game processes x 128 decisions = 4096 per update (the 3M run: 8 x 512) |
| Update | 4 epochs, 8 minibatches of 512, clip 0.2, value coefficient 0.5, entropy coefficient 0.01, gradient norm clipped at 0.5 |
| Advantages | GAE, gamma 0.995, lambda 0.95, normalised per minibatch |
| Optimiser | Adam, lr 3e-4 falling linearly to 0 |
| Device | `device = "auto"` (CUDA, then MPS, then CPU); the game processes take most of the time, so it matters little. The 3M run used the CPU |
| Speed | About 1700 decisions/s with 8 games (the 3M run) |

How the 3M run went (training games, sampled actions, last 500 games):

| Decisions | Minutes | Wins | Policy entropy |
|---|---|---|---|
| 4,096 | 0 | 4% | 1.97 |
| 450,000 | 4.7 | 54% | 1.29 |
| 940,000 | 9.6 | 71% | 0.95 |
| 1,430,000 | 14.5 | 76% | 0.85 |
| 1,930,000 | 19.4 | 79% | 0.86 |
| 2,910,000 | 29.0 | 81% | 0.79 |

Most of the gain comes in the first 1M decisions; the last 2M add about 10 points.
Greedy evaluation is higher than these sampled training rates.

## Not done

- Training against the planner bots (`opponent = "planner"`, optionally with
  `init_weights` set to start from the random-trained policy).
- Self-play.
- Checking the exported bots inside Seddler itself (they load in onnxruntime with the
  right shapes).
- Re-running the DQN default after the final cleanup, beyond a 150-game smoke test.
- Updating `dqn_seddler.ipynb`, which still has the original DQN.
