"""new_adopters.py -- next-day new adopters n_k^new(t+1) versus current usage x_k(t)  (Fig. 2e / 4c / 6e).

A new adopter of hashtag k on day t+1 ("new user" in the code identifiers) is a user who posts k on
day t+1 but did not post it on day t. The one-day horizon is the same for the data and for the model,
which is why this definition is used rather than "first use ever" (that would be left-censored over
different horizons: 7 days of data versus 30 model days).

empirical_new_adopters() : from the raw post table (restricted data)
simulated_new_adopters() : from new_users/<run>.txt written by the simulation
adopter_pairs()          : aligns the two tables into (x_k(t), n_k^new(t+1)) pairs
plot_adopter_scaling()   : scatter, medians and quartiles in logarithmic bins, power-law fit

usage:
  # 1. empirical table (once; restricted data)
  python new_adopters.py --posts /path/to/all_hashtag_retweet_info_with_main_tag_processed.pkl \
                         --save-empirical results/new_adopters_daily.pkl
  # 2. figure: empirical (blue) and simulation (orange)
  python new_adopters.py --emp-counts /path/to/hashtag_counts_daily.pkl --emp-new results/new_adopters_daily.pkl \
                         --sim-counts /path/to/results/<run>.pkl --sim-new /path/to/new_users/<run>.txt \
                         --out results/new_adopters.pdf
"""
import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_expected_counts import to_naive_tokyo  # noqa: E402
from postprocess import daily_counts  # noqa: E402


# ------------------------------------------------------------------------------------------------
def empirical_new_adopters(posts):
    """posts: DataFrame with u_id, tokyo_time, hashtags (list or str). Returns DataFrame index hashtag,
    columns = dates (Timestamp, Tokyo local), value = number of users who posted the hashtag on that day
    but not on the previous day. The first date column has no history and should not be used as n^new
    (adopter_pairs never does)."""
    d = posts[['u_id', 'tokyo_time', 'hashtags']]
    if d['hashtags'].map(lambda v: isinstance(v, (list, tuple, set))).any():
        d = d.explode('hashtags')
    d = d.dropna(subset=['hashtags'])
    date = to_naive_tokyo(d['tokyo_time']).normalize()
    d = pd.DataFrame({'u_id': d['u_id'].to_numpy(), 'hashtags': d['hashtags'].to_numpy(), 'date': date})
    trip = d.drop_duplicates(['u_id', 'hashtags', 'date'])                 # (user, hashtag, day) triples
    prev = trip.assign(date=trip['date'] + pd.Timedelta(days=1))           # the same triples shifted by one day
    m = trip.merge(prev, on=['u_id', 'hashtags', 'date'], how='left', indicator=True)
    new = m[m['_merge'] == 'left_only']                                    # no post of the hashtag the day before
    table = new.groupby(['hashtags', 'date']).size().unstack(fill_value=0)
    return table.sort_index(axis=1)


def simulated_new_adopters(path):
    """new_users/<run>.txt -> DataFrame index hashtag id, columns = saved-day index (0 = first saved line)."""
    cols = {}
    with open(path) as f:
        for day, line in enumerate(f):
            v = np.fromstring(line.strip(), dtype=np.int64, sep=',')
            cols[day] = pd.Series(v[1::2], index=v[0::2]) if v.size else pd.Series(dtype=np.int64)
    return pd.DataFrame(cols).fillna(0).astype(np.int64)


def adopter_pairs(counts, new_users):
    """Positional alignment: column j of both tables is the same day. Returns x = x_k(t), y = n_k^new(t+1)."""
    n = min(counts.shape[1], new_users.shape[1])
    counts = counts.iloc[:, :n]
    new_users = new_users.iloc[:, :n]
    tags = counts.index.union(new_users.index)
    c = counts.reindex(tags).fillna(0).to_numpy(dtype=float)
    u = new_users.reindex(tags).fillna(0).to_numpy(dtype=float)
    x = c[:, :-1].ravel()
    y = u[:, 1:].ravel()
    m = x > 0
    return x[m], y[m]


def plot_adopter_scaling(x, y, label=None, c='C0', marker='o', scatter=True, fit=True, n_bins=15, y_min=None,
                         fit_xmin=None, fit_xmax=None, min_count=2, alpha_digits=1, xlim=None, ylim=None,
                         alpha_in_label=True):
    """Same presentation as the existing Fig. 2(e): rasterised scatter, binned medians with [Q1, Q3] bars,
    power-law fit of the binned medians. Returns the fitted exponent.

    y_min=1 drops (x, y) pairs with y < 1 before binning (convention of the original figure).
    fit_xmin / fit_xmax restrict the FIT to the scaling regime; min_count is the minimum number of pairs a bin
    needs to be shown and used in the fit. With alpha_in_label=True the exponent is appended to the series
    legend entry ("real ($\\alpha = 0.98$)") and the fitted line itself carries no legend entry; otherwise the
    line is labelled with the exponent as before. xlim/ylim, if given, are applied to the axes."""
    if y_min is not None:
        keep = y >= y_min
        x, y = x[keep], y[keep]

    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins)
    idx = np.digitize(x, edges)
    rows = []
    for b in np.unique(idx):
        sel = idx == b
        if sel.sum() < min_count:
            continue
        rows.append((np.median(x[sel]), *np.percentile(y[sel], [25, 50, 75]), int(sel.sum())))
    binned = pd.DataFrame(rows, columns=['x', 'q1', 'q2', 'q3', 'n'])

    slope, intercept = np.nan, None
    if fit:
        pos = binned.q2 > 0
        if fit_xmin is not None:
            pos &= binned.x >= fit_xmin
        if fit_xmax is not None:
            pos &= binned.x <= fit_xmax
        slope, intercept, *_ = stats.linregress(np.log(binned.x[pos]), np.log(binned.q2[pos]))

    series_label = label
    if fit and alpha_in_label and label is not None and not np.isnan(slope):
        series_label = label + r' ($\alpha = %.*f$)' % (alpha_digits, slope)

    if scatter:
        plt.scatter(x, y, alpha=0.1, c=c, s=20, rasterized=True)
    plt.errorbar(binned.x, binned.q2, yerr=[binned.q2 - binned.q1, binned.q3 - binned.q2],
                 fmt=marker, capsize=5, markersize=10, ecolor='k', markeredgecolor='k', color=c,
                 capthick=2, linewidth=2, alpha=0.8, label=series_label)
    if fit and intercept is not None:
        xs = np.logspace(np.log10(binned.x[pos].min()), np.log10(binned.x[pos].max()), 20)
        line_label = None if (alpha_in_label and label is not None) else r'$\alpha = %.*f$' % (alpha_digits, slope)
        plt.plot(xs, np.exp(intercept) * xs ** slope, '--', c='k', linewidth=2, alpha=0.8, label=line_label)

    plt.xscale('log'); plt.yscale('log')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('$x(t)$'); plt.ylabel('$n^{new}(t+1)$')
    plt.gca().xaxis.set_minor_locator(LogLocator(subs='all', numticks=100))
    plt.legend(fontsize=20, loc='upper left', framealpha=0)
    plt.tight_layout()
    return slope


def main():
    matplotlib.use('Agg')   # headless when run as a script; leaves the notebook backend alone on import
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--posts', help='raw post table (restricted) -> empirical new-adopter table')
    ap.add_argument('--min-y', type=float, default=None, help='drop pairs with n_new < MIN_Y (e.g. 1) before binning')
    ap.add_argument('--fit-xmin', type=float, default=None, help='fit only bins with x >= FIT_XMIN (scaling regime)')
    ap.add_argument('--min-count', type=int, default=2, help='minimum pairs per bin')
    ap.add_argument('--save-empirical', default='results/new_adopters_daily.pkl')
    ap.add_argument('--emp-counts', help='empirical daily-count table (hashtag_counts_daily.pkl)')
    ap.add_argument('--emp-new', help='empirical new-adopter table (output of --posts)')
    ap.add_argument('--sim-counts', help='simulation posted-record file (results/<run>.txt) or daily_counts.pkl')
    ap.add_argument('--sim-new', help='simulation new_users/<run>.txt')
    ap.add_argument('--out', default='results/new_adopters.pdf')
    args = ap.parse_args()

    if args.posts:
        print('computing empirical new adopters ...', flush=True)
        table = empirical_new_adopters(pd.read_pickle(args.posts))
        os.makedirs(os.path.dirname(args.save_empirical) or '.', exist_ok=True)
        table.to_pickle(args.save_empirical)
        print(f'saved {args.save_empirical}: {table.shape[0]:,} hashtags x {table.shape[1]} days '
              f'(columns {table.columns[0].date()} .. {table.columns[-1].date()}; drop the first day when analysing)')

    plt.figure(figsize=(8, 6))
    slopes = {}
    if args.emp_counts and args.emp_new:
        counts = pd.read_pickle(args.emp_counts)
        new = pd.read_pickle(args.emp_new)
        new = new.iloc[:, :counts.shape[1]]                 # both start on the first day of the data
        x, y = adopter_pairs(counts, new)
        slopes['empirical'] = plot_adopter_scaling(x, y, label='real', c='C0', marker='o', fit=True, y_min=args.min_y, fit_xmin=args.fit_xmin, min_count=args.min_count)
    if args.sim_counts and args.sim_new:
        sc = pd.read_pickle(args.sim_counts) if args.sim_counts.endswith('.pkl') else daily_counts(args.sim_counts)
        sn = simulated_new_adopters(args.sim_new)
        keep = [j for j in range(min(sc.shape[1], sn.shape[1])) if sc.iloc[:, j].notna().any()]   # skip empty record lines
        x, y = adopter_pairs(sc.iloc[:, keep], sn.iloc[:, keep])
        slopes['simulation'] = plot_adopter_scaling(x, y, label='simulation', c='C1', marker='s', fit=False, y_min=args.min_y, min_count=args.min_count)
    if slopes:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        plt.savefig(args.out, bbox_inches='tight', pad_inches=0.05, dpi=300)
        print('fitted exponents:', {k: round(v, 3) for k, v in slopes.items() if not np.isnan(v)}, '->', args.out)


if __name__ == '__main__':
    main()
