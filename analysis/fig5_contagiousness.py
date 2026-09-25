"""fig5_contagiousness.py -- effective contagiousness d_k(r) = x_k(r) / xhat_k(r) and the Fig. 5 panels,
for one or several expected-count tables side by side (e.g. the original table and the recomputed ones).

For every variant it writes to <out-dir>/<label>/:
    series_<tag>.pdf        Fig. 5(a): x_k(r), xhat_k(r) and d_k(r) for one hashtag
    drift_<tag>.pdf         Fig. 5(b): d_k(r+1) - d_k(r) versus d_k(r) for that hashtag, binned medians + regression
    drift_all.pdf           Fig. 5(c): the same pooled over all reliable hashtags
    potential_all.pdf       Fig. 5(d): potential U(d) from the pooled drift
    d_table.pkl             the d_k(r) table (index hashtag, columns 10-minute windows; NaN = unreliable window)
and prints a summary line (number of reliable hashtags, regression slope, intercept, zero crossing).
The regression slope of the pooled drift is the empirical counterpart of -theta per 10-minute window.

Selection rule (as in the paper): windows with xhat_k(r) >= --min-expected are reliable; hashtags with at least
--min-windows reliable windows are kept. The first day of the count table is skipped in the drift analysis.

usage:
    python fig5_contagiousness.py \
        --counts /path/to/hashtag_counts_10min_lcc.pkl \
        --xhat m24=results/model0_expected_counts_10min.pkl \
        --tag save_ibaraki --out-dir results/fig5
"""
import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from result_analysis import plot_potential, potention_analysis_paper, visulize_diffusion_series_paper  # noqa: E402


def contagiousness_table(counts, xhat, min_expected=30, min_windows=2):
    """d_k(r) on the column grid of `counts`; xhat is reindexed to it (windows not computed -> unreliable)."""
    xhat = xhat.reindex(index=xhat.index, columns=counts.columns)
    reliable = (xhat >= min_expected).sum(axis=1)
    tags = reliable[reliable >= min_windows].index
    tags = tags[tags.isin(counts.index)]
    x = counts.loc[tags].fillna(0)
    xh = xhat.loc[tags]
    d = x / xh
    d = d.replace([np.inf, -np.inf], np.nan)
    d[~(xh >= min_expected)] = np.nan
    return d, xh


def run_variant(label, counts, xhat, tag, out_dir, min_expected, min_windows, nbins_tag, min_sample_tag,
                nbins_all, min_sample_all):
    os.makedirs(out_dir, exist_ok=True)
    d, xh = contagiousness_table(counts, xhat, min_expected, min_windows)
    d.to_pickle(os.path.join(out_dir, 'd_table.pkl'))
    skip = 24 * 6                                  # skip the first day of the count table (as in the paper analysis)
    summary = {'variant': label, 'n_reliable_hashtags': len(d),
               'n_reliable_windows': int((~d.isna()).sum().sum())}

    if tag in d.index:
        visulize_diffusion_series_paper(counts.fillna(0), xh.fillna(0), d, name=tag)
        plt.savefig(os.path.join(out_dir, f'series_{tag}.pdf'), bbox_inches='tight', pad_inches=0.05, dpi=200)
        plt.close('all')
        _, _, slope, intercept, x0 = potention_analysis_paper(
            d.iloc[:, skip:], tag_name=tag, onesample=True, nbins=nbins_tag, min_sample=min_sample_tag,
            plot_scatter=True, regression=True, show_mu=False, return_=True)
        plt.savefig(os.path.join(out_dir, f'drift_{tag}.pdf'), bbox_inches='tight', pad_inches=0.05, dpi=200)
        plt.close('all')
        summary.update({f'{tag}_slope': slope, f'{tag}_intercept': intercept, f'{tag}_zero_crossing': x0,
                        f'{tag}_reliable_windows': int((~d.loc[tag].isna()).sum())})
    else:
        print(f'[{label}] hashtag {tag!r} is not among the reliable hashtags of this variant')

    result_bined, result, slope, intercept, x0 = potention_analysis_paper(
        d.iloc[:, skip:], top_k=len(d), nbins=nbins_all, min_sample=min_sample_all, onesample=False, tag_name='',
        plot_scatter=False, regression=True, show_mu=False, return_=True)
    plt.legend(fontsize=20, loc='lower left', framealpha=0)
    plt.savefig(os.path.join(out_dir, 'drift_all.pdf'), bbox_inches='tight', pad_inches=0.05, dpi=200)
    plt.close('all')
    plot_potential(result, result_bined)
    plt.savefig(os.path.join(out_dir, 'potential_all.pdf'), bbox_inches='tight', pad_inches=0.05, dpi=200)
    plt.close('all')
    summary.update({'all_slope': slope, 'all_intercept': intercept, 'all_zero_crossing': x0,
                    'all_n_pairs': len(result)})
    return summary


def main():
    matplotlib.use('Agg')   # headless when run as a script; leaves the notebook backend alone on import
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--counts', required=True, help='10-minute counts of LCC users: hashtag_counts_10min_lcc.pkl')
    ap.add_argument('--xhat', nargs='+', required=True, help='label=path pairs of expected-count tables')
    ap.add_argument('--tag', default='save_ibaraki')
    ap.add_argument('--out-dir', default='results/fig5')
    ap.add_argument('--min-expected', type=float, default=30)
    ap.add_argument('--min-windows', type=int, default=2)
    ap.add_argument('--nbins-tag', type=int, default=50)
    ap.add_argument('--min-sample-tag', type=int, default=3)
    ap.add_argument('--nbins-all', type=int, default=50)
    ap.add_argument('--min-sample-all', type=int, default=10)
    args = ap.parse_args()

    counts = pd.read_pickle(args.counts)
    counts.columns = pd.DatetimeIndex(counts.columns)
    rows = []
    for item in args.xhat:
        label, path = item.split('=', 1)
        xhat = pd.read_pickle(path)
        xhat.columns = pd.DatetimeIndex(xhat.columns)
        if (xhat.columns.tz is None) != (counts.columns.tz is None):      # harmonise time-zone convention
            xhat.columns = (xhat.columns.tz_localize('Asia/Tokyo') if xhat.columns.tz is None
                            else xhat.columns.tz_convert('Asia/Tokyo').tz_localize(None))
        common = xhat.columns.intersection(counts.columns)
        print(f'[{label}] {xhat.shape[0]:,} hashtags x {xhat.shape[1]} windows; {len(common)} windows shared with the count table')
        rows.append(run_variant(label, counts, xhat, args.tag, os.path.join(args.out_dir, label),
                                args.min_expected, args.min_windows, args.nbins_tag, args.min_sample_tag,
                                args.nbins_all, args.min_sample_all))
    summary = pd.DataFrame(rows).set_index('variant')
    pd.set_option('display.width', 200)
    print('\n' + summary.T.to_string())
    summary.to_csv(os.path.join(args.out_dir, 'summary.csv'))


if __name__ == '__main__':
    main()
