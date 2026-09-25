# Data

The raw tweets cannot be redistributed under the X (formerly Twitter) Developer Agreement and
Policy and are **not** part of this repository. Every figure and number in the paper can be
reproduced from the processed (aggregated and anonymized) tables below, which are available from
the corresponding author, Misako Takayasu (takayasu@comp.isct.ac.jp), upon reasonable request.
Place them at the paths given here (the defaults in the notebook's configuration cell).

## Empirical tables

| file | place in | content | used by |
|---|---|---|---|
| `hashtag_counts_daily.pkl` (25 MB) | `simulation/real_data/` | daily usage count `x_k(t)`: index = hashtag (370,473), columns = 7 days, 11–17 March 2011; NaN = not used that day | Fig. 1, 2, 4, 6, 7; check plots |
| `hashtag_counts_hourly.pkl` (481 MB) | `simulation/real_data/` | hourly counts `x_k(h)`, 168 columns | Fig. 1 |
| `hashtag_counts_10min_lcc.pkl` (1.3 GB) | `simulation/real_data/` | 10-minute counts of posts by users in the network's largest connected component (LCC) | Fig. 5 |
| `retweet_network/network_empirical.txt` (144 MB) | `simulation/real_data/retweet_network/` | user relationship network, one directed link `i j` per line (whitespace separated, 0-based ids; `i` follows `j`); 397,369 users, 12,125,312 links | simulation, Fig. 8a |
| `retweet_network/network_shuffled_random.txt`, `network_shuffled_keep_in_degree.txt`, `network_shuffled_keep_out_degree.txt` (≈150 MB each) | same | randomised networks for Fig. 4(e): complete rewiring, in-degree-preserving and out-degree-preserving shuffles | simulation |

User ids in the network files are anonymised consecutive integers with no link to the original accounts.

## Derived tables

| file | place in | content | produced by |
|---|---|---|---|
| `new_adopters_daily.pkl` (26 MB) | `results/` | per hashtag and day, the number of users posting it who did not post it the day before (new adopters; "new users" in the code) | `analysis/new_adopters.py --posts <raw posts>` |
| `model0_expected_counts_10min.pkl` (1.0 GB) | `results/` | Model-0 expected count `xhat_k(r)` per hashtag and 10-minute window, exposures from the preceding 24 h | `analysis/recompute_expected_counts.py --m-hours 24` |

Both need the raw post table to regenerate; they are deposited so that Fig. 2(e), 4(c), 6(e) and Fig. 5 can be drawn without it.

## Simulation output

The twelve runs behind Fig. 4, 6, 7 and 9(a) are not deposited (≈4.7 GB); `simulation/parallel_sim.sh` with `SEED=1` regenerates them in about an hour on a multi-core machine (see `simulation/README.md`).

## Formats

All `.pkl` files are pandas DataFrames written with pandas 2.x (`pd.read_pickle`). Because the
tables carry Tokyo-local timestamps, `pytz` must be installed to unpickle them (see
`requirements.txt`). Column conventions: rows are hashtags, columns are time (days, hours or
10-minute windows), values are counts.
