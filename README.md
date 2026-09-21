# Agent-based model with decaying attention reproducing hashtag scaling laws

Code for *Agent-based model with decaying attention reproducing hashtag scaling laws: random multiplicative
growth as a bridge* (Jiang, Yamada, Takayasu & Takayasu, 2026; arXiv:XXXX.XXXXX).

* Code: this repository (archived release: Zenodo DOI to be assigned)
* Data: the processed tables are archived separately on Zenodo (DOI to be assigned); see `data/README.md`
  for the file list and where to place them. The raw tweets are not redistributed.

The repository contains

* the C++ agent-based simulation of hashtag posting (**Model 0**: users create new hashtags with probability
  $p$ or adopt hashtags in proportion to their exposure in followed users' recent posts; **Model 1**: Model 0
  with a decaying *effective contagiousness* $d_k(r)$ that weights the exposures and follows a mean-reverting
  Ornstein–Uhlenbeck process),
* the Python analyses of the empirical regularities (usage distribution, growth-rate statistics, next-day
  new-adopter and deactivation scaling), the estimation of $d_k(r)$ from data, the calibration errors and the
  random multiplicative process, and
* the notebook that draws the figure panels of the paper.

## Layout

```
simulation/            C++ agent-based model -- see simulation/README.md
  modules/             simulation.cpp and the headers it includes
  simulation.sh        one run
  parallel_sim.sh      the twelve runs behind the paper (passes A, B, C)
  process_result/      simulation output -> npy/pickle tables, automatic check plots
  real_data/           network input (not redistributed; see simulation/real_data/README.md)
analysis/
  result_analysis.py              statistics and plotting routines behind the figure panels
                                  (usage distribution, growth rates, deactivation, d_k(r) drift and potential, autocorrelation)
  new_adopters.py                 next-day new adopters n_k^new(t+1) vs x_k(t)  (Fig. 2e, 4c, 6e)
  recompute_expected_counts.py    Model-0 expected counts xhat_k(r), trailing m-hour exposure window
  check_expected_counts.py        loaders and consistency checks for the expected-count tables
  fig5_contagiousness.py          d_k(r) = x_k(r)/xhat_k(r) and the Fig. 5 panels
  calibration_errors.py           error components and the grid summary (Fig. 9b)
  random_multiplicative_process.py  macroscopic RMP driven by the simulated growth-rate pool (Fig. 7)
  postprocess.py                  raw simulation output -> daily-count tables
figures/
  paper_figures.ipynb  figure panels; all paths in one configuration cell at the top
  conceptual/          TikZ source of the schematic (Fig. 3)
data/                  processed empirical tables (deposited separately; see data/README.md)
```

## Requirements

* C++17 compiler (tested with g++ 11; on Apple clang the build script drops `-march=native`)
* Python ≥ 3.9 with the packages in `requirements.txt`

```bash
conda create -n hashtag-abm python=3.12
conda activate hashtag-abm
pip install -r requirements.txt
```

The post-processing invoked by `simulation/simulation.sh` uses whatever `python` is on the
path; point it at the environment explicitly with `PYTHON=/abs/path/to/python`.

All comparison statistics of the simulations use the **final 7 model days** of each 30-day run
(`N_DAYS_CMP = 7` in the notebook), matching the 7-day empirical window and the Methods section.

## Reproduction

1. **Data.** Download the processed tables from the Zenodo deposit and place them as listed in
   `data/README.md` (empirical tables and network files under `simulation/real_data/`, derived tables
   under `results/`).

2. **Run the simulations.** Twelve runs, all with `SEED=1`:

   ```bash
   cd simulation
   ./parallel_sim.sh A     # Model 0: main run (m=24) and the memory-window sweep   8 runs
   ./parallel_sim.sh B     # Model 0: three randomised networks                     3 runs
   ./parallel_sim.sh C     # Model 1: calibrated parameters                         1 run
   ```

   `parallel_sim.sh` compiles first and then runs the passes `PARALLEL_JOBS` at a time
   (default 65). One model day is 24 × 93,626 = 2,247,024 posting events; a 30-day run of
   the full system (397,369 users) keeps every user's posting history in memory and was
   executed on a multi-core server. `simulation.sh` post-processes each run into
   `.npy`/`.pkl` tables automatically. Details, output format and the run-to-panel map are
   in `simulation/README.md`.

3. **Effective contagiousness from data** (Fig. 5; needs the restricted post table):

   ```bash
   python analysis/recompute_expected_counts.py --posts <posts.pkl> --network <lcc_network.pkl> \
          --m-hours 24 --begin "2011-03-12 00:00" --end "2011-03-18 00:00" --out results/model0_expected_counts_10min.pkl
   python analysis/fig5_contagiousness.py --counts <hashtag_counts_10min_lcc.pkl> \
          --xhat m24=results/model0_expected_counts_10min.pkl --tag save_ibaraki --out-dir results/fig5
   ```

   Exposures are the posts of followed users within the preceding $m = 24$ hours, using past
   information only; the analysis starts on 12 March so that every window has a complete history.

4. **Empirical new-user tables** (Fig. 2e, and the empirical series in Fig. 4c / 6e):

   ```bash
   python analysis/new_adopters.py --posts <posts.pkl> --save-empirical results/new_adopters_daily.pkl
   ```

   A new adopter (called a "new user" in the code identifiers, e.g. `is_new_user`, `new_users/`) is a user who posts hashtag $k$ on day $t+1$ but not on day $t$;
   this one-day horizon is identical for the data and the model.

5. **Figures.** Run `figures/paper_figures.ipynb`. Single-panel PDFs are written per panel; the
   multi-panel figures of the paper are assembled from them in a vector editor.

## Model parameters (Table 1)

| symbol | meaning | `simulation.sh` argument | value |
|---|---|---|---|
| $N$ | number of users | `N_AGENTS` (in the script) | 397,369 |
| – | events per model hour / day | `ONE_HOUR` | 93,626 / 2,247,024 |
| $p$ | probability of creating a new hashtag | `P_NEW` | 0.0218 (empirical) |
| $m$ | memory window (h) | 2nd argument | 84 (Model 1); 24 for the Model 0 runs of Fig. 4 |
| $\theta$ | decay rate of $d_k$ | 4th argument | 0.007 |
| $\mu$ | long-run level of $d_k$ | 5th argument | 0 |
| $\sigma$ | noise amplitude of $d_k$ | 6th argument | 0.1 (fixes the scale) |
| $d_0$ | initial $d_k \sim U(0, d_0)$ | 3rd argument | 13 |

Model 0 is the same binary with `LIFE_TIME_TYPE=const_dk` and $d_k \equiv 1$, i.e. adoption weighted by
exposure counts alone.

## Figure ↔ code

| figure | data / run | code |
|---|---|---|
| Fig. 1 | daily count series | notebook |
| Fig. 2 | empirical daily counts | notebook; `result_analysis.py`, `new_adopters.py` (panel e) |
| Fig. 3 | – | `figures/conceptual/fig3_model0_schematic.tex` |
| Fig. 4 | Model 0 runs (passes A and B) | notebook; `new_adopters.py` (panel c) |
| Fig. 5 | 10-minute counts and expected counts | `recompute_expected_counts.py`, `fig5_contagiousness.py` |
| Fig. 6 | Model 1 run (pass C) | notebook; `new_adopters.py` (panel e) |
| Fig. 7 | Model 1 run | `random_multiplicative_process.py`; notebook |
| Fig. 8 | network and hourly counts | notebook |
| Fig. 9 | Model 1 run (`diffu_value/`), calibration grid | notebook; `calibration_errors.py` |
| Table 1 | calibration grid | `calibration_errors.py` |

## Notes on reproducibility

* The simulation code here is the code that produced the published results, not a reimplementation.
  One change was made relative to the version used during the research: the original
  `modules/simulation.cpp` passed an uninitialised `int seed;` to `initialize_hashtags(config, seed + 1)`,
  which is undefined behaviour; it now passes `config.seed + 1`. Runs are therefore statistically
  equivalent to, but not bit-identical with, results produced before that fix.
* No intra-day activity cycle is imposed. One model day is a fixed number of events, matching the
  Methods section "Model time scale"; the 10-minute windows that set the update interval of $d_k(r)$
  are one sixth of a model hour each.
* The $\theta \times d_0$ calibration grid behind Fig. 9(b) and Table 1 was run before the seed fix.

## License

MIT (see `LICENSE`).
