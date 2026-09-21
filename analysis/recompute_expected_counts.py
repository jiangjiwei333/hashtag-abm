"""recompute_expected_counts.py -- Model-0 expected counts xhat_k(r) for every 10-minute window,
using the exposure rule of the paper: exposures are the posts of followed users within the past m hours
(0 <= t_now - t <= m h, no look-ahead).

Output has the same layout as the original tag_likely_df_hour_*.pkl table (index = hashtag, columns = window
start time, value = xhat_k(r), zeros filled) so that it can be used as a drop-in replacement in the Fig. 5
analysis. A companion CSV lists, per window, the number of LCC posts, the number with a non-empty exposure
set (= column sum) and the number whose own hashtag was among the exposed ones.

Only posts by users of the LCC network are used (posting users and exposure sources), as in the original
preprocessing. Windows are computed from --begin (inclusive) to --end (exclusive); --begin should be at least
m hours after the start of the data so that every window has a complete history.

usage:
    python recompute_expected_counts.py \
        --posts   /path/to/all_hashtag_retweet_info_with_main_tag_processed.pkl \
        --network /path/to/lcc_network_ver2.pkl \
        --like    /path/to/tag_likely_df_hour_2011-03-12~2011-03-18_date_hour_10min.pkl \
        --m-hours 24 --begin "2011-03-12 00:00" --end "2011-03-18 00:00" \
        --n-jobs 40 --out /path/to/model0_expected_counts_10min.pkl
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_expected_counts import load, to_naive_tokyo  # noqa: E402

# module-level state shared with forked workers (Linux fork: no pickling of the large index)
_T = _U = _C = None      # post times (s), user ids, hashtag codes, sorted by time
_BY_USER = _FOLLOWING = None
_SPAN = 0
_NTAGS = 0


def _window(t0):
    """xhat for the window [t0, t0 + 600 s): returns sparse (codes, values) and post statistics."""
    lo = np.searchsorted(_T, t0, side='left')
    hi = np.searchsorted(_T, t0 + 600, side='left')
    acc = np.zeros(_NTAGS)
    n_posts = hi - lo
    n_exposed = 0
    n_own_exposed = 0
    for i in range(lo, hi):
        u, t, own = _U[i], _T[i], _C[i]
        pieces = []
        for v in _FOLLOWING.get(u, ()):
            entry = _BY_USER.get(v)
            if entry is None:
                continue
            times, codes = entry
            a = np.searchsorted(times, t - _SPAN, side='left')
            b = np.searchsorted(times, t, side='right')
            if b > a:
                pieces.append(codes[a:b])
        if not pieces:
            continue
        exposed = np.concatenate(pieces)
        uniq, cnt = np.unique(exposed, return_counts=True)
        np.add.at(acc, uniq, cnt / cnt.sum())          # Pi^(0) ~ exposure count, normalised per post
        n_exposed += 1
        if (uniq == own).any():
            n_own_exposed += 1
    codes = np.flatnonzero(acc)
    return t0, codes, acc[codes], n_posts, n_exposed, n_own_exposed


def main():
    global _T, _U, _C, _BY_USER, _FOLLOWING, _SPAN, _NTAGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--posts', required=True)
    ap.add_argument('--network', required=True)
    ap.add_argument('--m-hours', type=float, required=True)
    ap.add_argument('--begin', required=True, help='first window start, Tokyo local, e.g. "2011-03-12 00:00"')
    ap.add_argument('--end', required=True, help='end (exclusive), e.g. "2011-03-18 00:00"')
    ap.add_argument('--like', help='existing table whose column convention (time zone) should be copied')
    ap.add_argument('--n-jobs', type=int, default=8)
    ap.add_argument('--out', required=True, help='output pickle (wide table); a *_window_stats.csv is written next to it')
    args = ap.parse_args()

    data, tags, by_user, following = load(args.posts, args.network)
    _T = data['t'].to_numpy()
    _U = data['u_id'].to_numpy()
    _C = data['code'].to_numpy()
    _BY_USER, _FOLLOWING = by_user, following
    _SPAN = int(args.m_hours * 3600)
    _NTAGS = len(tags)

    t_begin = np.int64(np.datetime64(pd.Timestamp(args.begin), 's').astype(np.int64))
    t_end = np.int64(np.datetime64(pd.Timestamp(args.end), 's').astype(np.int64))
    starts = list(range(int(t_begin), int(t_end), 600))
    data_start = pd.Timestamp(np.datetime64(int(_T.min()), 's'))
    if pd.Timestamp(args.begin) - data_start < pd.Timedelta(hours=args.m_hours):
        print(f'WARNING: --begin is less than {args.m_hours:g} h after the first post ({data_start}); '
              f'early windows have truncated histories', flush=True)
    print(f'{len(starts)} windows, m = {args.m_hours:g} h, {args.n_jobs} processes ...', flush=True)

    tic = time.time()
    ctx = mp.get_context('fork')
    results = []
    with ctx.Pool(processes=args.n_jobs) as pool:
        for k, res in enumerate(pool.imap_unordered(_window, starts, chunksize=4), 1):
            results.append(res)
            if k % 50 == 0 or k == len(starts):
                print(f'\r  {k}/{len(starts)} windows, {(time.time() - tic) / 60:.1f} min', end='', flush=True)
    print()
    results.sort(key=lambda r: r[0])

    # assemble the wide table (rows: hashtags with any exposure, columns: windows)
    used = np.zeros(_NTAGS, dtype=bool)
    for _, codes, _, _, _, _ in results:
        used[codes] = True
    row_of = -np.ones(_NTAGS, dtype=np.int64)
    row_of[np.flatnonzero(used)] = np.arange(used.sum())
    mat = np.zeros((int(used.sum()), len(results)), dtype=np.float64)
    for j, (_, codes, vals, _, _, _) in enumerate(results):
        mat[row_of[codes], j] = vals
    col_index = pd.DatetimeIndex([pd.Timestamp(np.datetime64(r[0], 's')) for r in results])
    tz_aware = True
    if args.like:
        ref = pd.read_pickle(args.like)
        tz_aware = pd.DatetimeIndex(ref.columns).tz is not None
    if tz_aware:
        col_index = col_index.tz_localize('Asia/Tokyo')
    table = pd.DataFrame(mat, index=pd.Index(tags[np.flatnonzero(used)], name='hashtags'), columns=col_index)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]      # same ordering as the original table
    table.to_pickle(args.out)

    stats = pd.DataFrame({'window': col_index,
                          'n_posts': [r[3] for r in results],
                          'n_with_exposure': [r[4] for r in results],
                          'n_own_tag_exposed': [r[5] for r in results]})
    stats_path = os.path.splitext(args.out)[0] + '_window_stats.csv'
    stats.to_csv(stats_path, index=False)
    print(f'saved {args.out}: {table.shape[0]:,} hashtags x {table.shape[1]} windows; stats -> {stats_path}')
    print(f'posts with exposure: {stats.n_with_exposure.sum() / stats.n_posts.sum():.3%}; '
          f'own hashtag among exposed: {stats.n_own_tag_exposed.sum() / stats.n_posts.sum():.3%}')


if __name__ == '__main__':
    main()
