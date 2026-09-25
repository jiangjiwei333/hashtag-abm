# Simulation input

Everything the run needs is resolved relative to the working directory, i.e. relative to
the folder containing `simulation.sh`.

## 1. User relationship network — required by the C++ binary

The paths are hard coded in `modules/initialize/initialize.h`, function `initialize_network`.

| `NETWORK` argument | file | used for |
|---|---|---|
| `real` | `retweet_network/network_empirical.txt` | Model 0 and Model 1 main runs, memory sweep |
| `real_to_random` | `retweet_network/network_shuffled_random.txt` | Fig. 4(e), `random` |
| `real_keep_in_degree` | `retweet_network/network_shuffled_keep_in_degree.txt` | Fig. 4(e), `keep in` |
| `real_keep_out_degree` | `retweet_network/network_shuffled_keep_out_degree.txt` | Fig. 4(e), `keep out` |

**Format.** One directed link per line, two whitespace-separated 0-based integer user ids

```
i j
```

read as "user `i` retweeted user `j`", i.e. `i` is exposed to `j`'s posts (a directed edge `i -> j` in the Methods). The empirical
network has 397,369 nodes and 12,125,312 links, and is the largest connected component of
the retweet network described in the Methods section of the paper.

The randomised variants are shuffled copies of the empirical network (complete rewiring preserving
the numbers of nodes and links; in-degree-preserving and out-degree-preserving shuffles, see the
legend of Fig. 4(e)); they are provided together with the empirical network.

## 2. Empirical daily counts — required only by the check plots

`real_data/hashtag_counts_daily.pkl` — the empirical daily hashtag usage-count table
(hashtag x date), read by `process_result/plot_results.py` so that the automatic check
plots can overlay the simulation on the data.

This file is **not** needed to produce the results used in the paper.
`plot_results.py` is the last step of `simulation.sh`, after the simulation itself and
after `to_npy.py`, `save_high_resolution_xt.py` and `get_hashtagdf.py` have written
`results/`, `new_users/` and `xt/`. If the table is missing, that last
step fails and no check plots or `error/` line are written, but every output the paper
figures are drawn from is already on disk.

## Availability

These files are not redistributed here: they derive from the restricted tweet corpus.
See `../../data/README.md`.
