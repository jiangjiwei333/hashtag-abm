# Agent-based simulation (Model 0 and Model 1)

Event-level C++ simulation of hashtag posting used for Fig. 4, Fig. 6, Fig. 7 and
Fig. 9(a) of the paper. This is the code that produced the published results.

Both models share one engine. At every event an agent either creates a new hashtag
(probability `p_new`) or adopts one of the hashtags it has been exposed to through the
user relationship network within the preceding `m` hours. Model 0 weights the candidates
by exposure count alone (`LIFE_TIME_TYPE=const_dk`, `d_k` fixed to 1); Model 1 multiplies
the exposure weight by a per-hashtag effective contagiousness `d_k(r)` that follows an
Ornstein--Uhlenbeck process updated once per 10-minute window
(`LIFE_TIME_TYPE=ou`). One model day is a fixed number of events,
`one_day = 24 x one_hour = 24 x 93,626 = 2,247,024`; no intra-day activity cycle is
imposed, matching the Methods section "Model time scale".

## Layout

```
modules/
  simulation.cpp          entry point: parses the 19 positional arguments into SimulationConfig
  config/config.h         SimulationConfig
  initialize/initialize.h agent/hashtag initialisation, network loading
  agent/agent.h           per-agent state; is_new_user() classifies new users (see below)
  hashtag/hashtag.h       per-hashtag counters and the per-day output writers
  models/models.h         the event loop and the adoption rules of Model 0 / Model 1
  others/others.h         edge-list reader and vector writer
simulation.sh             one run: builds the output paths, calls the binary, post-processes
parallel_sim.sh           the 12 runs behind the paper, in three passes
process_result/           txt -> npy/pickle conversion and the automatic check plots
real_data/                network input (not redistributed; see real_data/README.md)
```

## Build and run

```sh
# compiles automatically, then runs; PARALLEL_JOBS and SEED can be overridden
./parallel_sim.sh A      # Model 0: main run (m=24) + memory sweep     8 runs
./parallel_sim.sh B      # Model 0: network randomisations             3 runs
./parallel_sim.sh C      # Model 1: main run                           1 run
./parallel_sim.sh all    # all twelve
```

A single run, if you prefer:

```sh
MODEL=0 ./simulation.sh const_dk 24 1  0     0 0.1 real 1     # Model 0
MODEL=1 ./simulation.sh ou       84 13 0.007 0 0.1 real 1     # Model 1
```

with argument order `LIFE_TIME_TYPE MEMORY D0 THETA MU SIGMA NETWORK SEED`. `MODEL` only
tags the output folder; the C++ stores it in `config.model` but never reads it back.
Manual compilation is `g++ -std=c++17 -O3 -mtune=native -march=native modules/simulation.cpp -o modules/simulation`.

All results in the paper use `SEED=1`, 397,369 agents and 30 model days.

## Which run produces which figure

| run | figure panels |
|---|---|
| A, `m=24` | Fig. 4 (a) (b) (c) (d) |
| A, all eight memory windows | Fig. 4 (f) |
| B, three randomised networks | Fig. 4 (e) |
| C | Fig. 6 (a)--(f), Fig. 7 (a) (b), Fig. 9 (a) |

Fig. 2 and Fig. 5 come from the empirical data only and do not use this code; see
`../analysis/`.

## Output

Per run, under `simulation_results/model<MODEL>/`:

| path | content |
|---|---|
| `results/<name>.txt` (`.pkl` after post-processing) | daily usage counts `x_k(t)`, one line per saved day, `tag,count,tag,count,...` |
| `new_users/<name>.txt` | per-day count of users posting `k` who did not post it on the previous day |
| `xt/<name>.txt` | 10-minute-resolution counts |
| `diffu_value/sample<name>.txt` | sampled `d_k(r)` trajectories (Fig. 9a) |
| `figs/`, `error/` | automatic check plots and calibration error lines, not used in the paper |

Only the last `N_SAVE_FROM_LAST_STEP` days are written; the new-user file is row-aligned
with `results/`, so column `j` of both refers to the same model day.

**New-user definition** (called "new adopters" in the paper)**.** `Agent::is_new_user(tag, day)` keeps `used_tags` as a sorted
vector of `(tag, last_day_used)` and returns true when the agent posts the hashtag today
but did not post it on the previous day. A second post of the same hashtag on the same day
returns false, so each user is counted at most once per hashtag per day. This
definition is used because its one-day horizon is identical for the data and the model; a
"first use ever" definition would be left-censored over different horizons (7 days of data
versus 30 model days). The empirical counterpart is `analysis/new_adopters.py --posts ...`.

**Sampled trajectories.** Hashtags with `tag % 10000 == 0` have their `d_k(r)` recorded at
every 10-minute update, giving roughly 150 trajectories per run for Fig. 9(a). The
selection is deterministic on purpose: drawing it at random would consume random numbers
in `add_hashtag` and shift the hashtag-side random stream. For Model 0 the file contains
one value per sampled hashtag, since `d_k` never changes.

## Note on the random seed

The original version of `modules/simulation.cpp` passed an uninitialised `int seed;` to
`initialize_hashtags(config, seed + 1)`, which is undefined behaviour. It is fixed here to
`config.seed + 1`. Results are statistically equivalent to earlier runs but not
bit-identical, because the hashtag-side random stream differs.
