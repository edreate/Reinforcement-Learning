# config

The parameters you are most likely to change, in one place. Every file is read and checked when
the program starts (`src/seddler/config.py`): a missing key, an unknown one or a value of the
wrong type stops it with the file and key named.

| File | What it sets |
|---|---|
| `game.toml` | The default game for the environment, self-play and training |
| `room.toml` | A live room's settings: the lobby's default and menu choices for each |
| `server.toml` | The web server's port and folders |
| `training.toml` | The observation encoder and self-play runs |
| `recording.toml` | What a recording logs beside each move: how many turns the other players remember a player's cards for |
| `reward.toml` | The environment's reward: the weight of each term (win, loss, points gained, ...) |

The values here are defaults. Code can still choose its own, for example
`GameConfig(num_players=4)`. Each value lives only here: the browser loads the lobby's menus
and the limits it shows from `GET /api/config`, and `make`, the dev server and the Pi scripts
read the port from `server.toml`. Environment variables (`SEDDLER_RECORDINGS` and the like) win over
`server.toml`. To use another folder of these files, set `SEDDLER_CONFIG=/path/to/folder`.

Not here on purpose:
- **The rules** (costs, deck, bank, board) are in `src/seddler/engine/rules.py`. Changing one
  changes how recorded games replay, so it goes with a `RULES_VERSION` bump.
- **Bot tuning** (the heuristic bots' weights) stays in each bot's module. It defines that bot's
  version.
- **The client's timings** are in `web/src/config.ts`.
