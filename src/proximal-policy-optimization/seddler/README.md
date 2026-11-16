# Seddler: from DQN to PPO

What was tried to make an agent learn Seddler, in order, and what each step did.
All runs are against Seddler's `RandomBot` (random legal moves, weighted towards
building) at tables of 2, 3 and 4 players, on an M2 Pro.

Win rates are greedy evaluation games at 2 / 3 / 4 players. An equal player wins
50% / 33% / 25%. Uniform random moves win about 7% overall.

## Result

| Algorithm | Training | Wins at 2 / 3 / 4 players |
|---|---|---|
| DQN, as first committed | 500 games | did not learn (flat reward curve) |
| DQN, stabilised (current defaults) | 4000 games, ~28 min | 44% / 38% / 23% |
| PPO (current defaults) | 3M decisions, ~30 min | 95% / 86% / 79% |

PPO trained only against random bots does not beat the heuristic bots: a 1.5M-decision
policy won 1 of 90 games against `HeuristicBot`.

## What was learned

1. **Most steps are not decisions.** About 74% of steps have one legal action (roll,
   discard). Making those inside the environment cuts a game from ~250 steps to ~60-70.
2. **The original DQN failed because its Q-values diverged.** After 1500 games it
   predicted returns near 480 where a won game pays about 4 (in scaled units), and played
   worse than random. Most (state, action) pairs are never tried, their Q-values are
   noise, and the target's max over legal actions feeds that noise back.
3. **Input scaling can hurt.** Standardising with a tiny spread turns a rarely-set 0/1
   feature into a value in the hundreds. The spread needs a floor.
4. **Slower updates only slow the divergence.** A slower target network, fewer gradient
   steps, a lower gamma and LayerNorm each reduced the overestimate; none produced
   learning on their own.
5. **n-step returns are what stabilised DQN.** With 5 to 10 real rewards before the
   bootstrap, the predicted values stayed near the real returns.
6. **Stable DQN still learns poorly.** The best DQN variant was one nearly blind to the
   board (inputs divided by 1000): it learns fixed action preferences, which beats
   random bots but is not play. State-aware variants did worse than it.
7. **PPO learns quickly here.** With action masking and parallel games it passed every
   DQN variant within about 5 minutes.

## Steps, in order

| # | Change | Effect |
|---|---|---|
| 0 | Original DQN: 500 games, 1024x1024 MLP, raw inputs | Flat reward curve, no learning |
| 1 | Skip forced moves; Double DQN; array replay buffer; input standardisation; 3000 games | 3-4% wins by game 1500, below random (7%) |
| 2 | Floor of 0.1 on the input spread | No change: 1 win in 90 evaluation games. Predicted Q about 480 vs real returns under 4 |
| 3 | Inputs divided by 1000 (diagnostic: network nearly blind to the state) | Q-values sane (~2); 78% / 66% / 39% after 1500 games, 88% / 69% / 56% after 4000 |
| 4 | LayerNorm in the network | Q grew more slowly (about 10 at 40k steps) but still too high; wins 5-13% |
| 5 | One at a time: train every 4 steps, tau 5e-4, gamma 0.95 | Q lower (5.5, 2.8, 2.6); no learning by 700-900 games |
| 6 | n-step returns (n = 5 or 10) with LayerNorm | Q about 1-1.6, matching real returns; first learning: ~20% wins at 900 games |
| 7 | Longer runs of the stable variants, 4000 games | n=10: 36% / 21% / 10%. n=5, lr 3e-4, train every 2 steps: 44% / 38% / 23% (kept as default) |
| 8 | PPO: masked policy, 8-10 game processes, GAE, running input normalisation | 1.5M decisions, 13 min: 93% / 83% / 64% |
| 9 | PPO default run, 3M decisions | 95% / 86% / 79% |

Steps 1-7 are in `src/deep-q-learning/seddler/dqn_seddler.py`, steps 8-9 in
`ppo_seddler.py` here. The environment both use is `src/utils/seddler_table.py`.

## Not done

- Training against the heuristic bots (`--opponent heuristic`, optionally with
  `--init <weights.pth>`).
- Self-play.
- Checking the exported bots inside Seddler itself (they load in onnxruntime with the
  right shapes).
