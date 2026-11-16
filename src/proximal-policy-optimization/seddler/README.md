# Seddler: from DQN to PPO

What was tried to make an agent learn Seddler, in order, and what each step did.

- **Opponents:** Seddler's `RandomBot` in every run (random legal moves, weighted towards
  building so that games end). Tables of 2, 3 and 4 players, drawn per game.
- **Win rates** are greedy evaluation games at 2 / 3 / 4 players. An equal player wins
  50% / 33% / 25%. Uniform random moves win about 7% overall.
- **Machine:** M2 Pro, 12 cores.
- **Code:** DQN in `src/deep-q-learning/seddler/dqn_seddler.py`, PPO in `ppo_seddler.py`
  here, the environment both use in `src/utils/seddler_table.py`.

## Result

| Algorithm | Training | Wins at 2 / 3 / 4 players | Evaluation games |
|---|---|---|---|
| DQN, as first committed | 500 games, ~19 min | did not learn (flat reward curve) | - |
| DQN, stabilised (current defaults) | 4000 games, ~28 min | 44% / 38% / 23% | 150 |
| PPO (current defaults) | 3M decisions (~50,000 games), ~30 min | 95% / 86% / 79% | 300 |

PPO trained only against random bots does not beat the heuristic bots: a 1.5M-decision
policy won 1 of 90 games against `HeuristicBot`.

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
| 9 | PPO, default run | 3M decisions, 8 processes | 95% / 86% / 79% in 30 min |

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
| Rollout | 8 game processes x 512 decisions = 4096 per update |
| Update | 4 epochs, 8 minibatches of 512, clip 0.2, value coefficient 0.5, entropy coefficient 0.01, gradient norm clipped at 0.5 |
| Advantages | GAE, gamma 0.995, lambda 0.95, normalised per minibatch |
| Optimiser | Adam, lr 3e-4 falling linearly to 0 |
| Device | CPU: the networks are small and the game processes take most of the time |
| Speed | About 1700 decisions/s |

How the default run went (training games, sampled actions, last 500 games):

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

- Training against the heuristic bots (`--opponent heuristic`, optionally with
  `--init <weights.pth>` to start from the random-trained policy). `HeuristicBotV2` is
  not wired into the scripts.
- Self-play.
- Checking the exported bots inside Seddler itself (they load in onnxruntime with the
  right shapes).
- Re-running the DQN default after the final cleanup, beyond a 150-game smoke test.
- Updating `dqn_seddler.ipynb`, which still has the original DQN.
