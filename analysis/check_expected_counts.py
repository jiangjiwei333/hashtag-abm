"""check_expected_counts.py -- which exposure window produced the stored expected counts xhat_k(r)?

Recomputes the Model-0 expected counts for a few 10-minute windows under several hypotheses and compares
them tag by tag with the stored table (tag_likely_df_hour_..._date_hour_10min.pkl):

    trailing m   exposures = posts of followed users with  0 <= t_now - t <= m hours   (paper definition)
    legacy   m   exposures = posts of followed users with  t <= t_now + m hours        (condition of get_tag_likelihood.py:
                 no lower bound, i.e. the whole history plus m hours of "future")

The computation is deterministic, so exactly one hypothesis should reproduce the stored values.
Only posts by users of the LCC network are used, as in the original preprocessing.

Sanity check printed first: sum_k xhat_k(r) equals the number of posts in the window whose exposure set is
non-empty, so the stored column sum must not exceed the number of LCC posts in the window (otherwise the time
zone / window alignment is off).

usage (run from anywhere; nothing is written to disk):
    python check_expected_counts.py \
        --posts   /path/to/all_hashtag_retweet_info_with_main_tag_processed.pkl \
        --network /path/to/lcc_network_ver2.pkl \
        --stored  /path/to/tag_likely_df_hour_2011-03-12~2011-03-18_date_hour_10min.pkl \
        --windows "2011-03-14 12:00" "2011-03-14 18:00" \
        --m-values 1 24 48 84
"""
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm


def to_naive_tokyo(ts):
    """tz-aware -> Asia/Tokyo local, tz dropped; naive values are assumed to be Tokyo local already."""
    ts = pd.DatetimeIndex(ts)
    if ts.tz is not None:
        ts = ts.tz_convert('Asia/Tokyo').tz_localize(None)
    return ts


def load(posts_path, network_path):
    print('loading posts ...', flush=True)
    data = pd.read_pickle(posts_path)
    data = data.explode('hashtags')[['u_id', 'tokyo_time', 'hashtags']].dropna(subset=['hashtags'])
    net = pd.read_pickle(network_path)
    lcc_users = set(net['target'].unique()) | set(net['source'].unique())
    data = data[data['u_id'].isin(lcc_users)].sort_values('tokyo_time').reset_index(drop=True)
    data['t'] = to_naive_tokyo(data['tokyo_time']).values.astype('datetime64[s]').astype(np.int64)
    codes, tags = pd.factorize(data['hashtags'])
    data['code'] = codes
    print(f'{len(data):,} (post, hashtag) rows by {data.u_id.nunique():,} LCC users; {len(tags):,} hashtags', flush=True)

    print('indexing posts by user ...', flush=True)
    by_user = {}
    for u, g in data.groupby('u_id', sort=False):
        by_user[u] = (g['t'].to_numpy(), g['code'].to_numpy())
    following = net.groupby('source')['target'].apply(lambda s: list(set(s))).to_dict()
    return data, tags, by_user, following


def expected_counts(window_posts, by_user, following, n_tags, mode, m_hours):
    """xhat over all tags for the posts of one window; returns (vector, number of posts with exposure)."""
    acc = np.zeros(n_tags)
    n_exposed = 0
    span = int(m_hours * 3600)
    for u, t in zip(window_posts['u_id'].to_numpy(), window_posts['t'].to_numpy()):
        pieces = []
        for v in following.get(u, ()):
            entry = by_user.get(v)
            if entry is None:
                continue
            times, codes = entry
            if mode == 'trailing':
                lo = np.searchsorted(times, t - span, side='left')
                hi = np.searchsorted(times, t, side='right')
            else:  # legacy: no lower bound
                lo = 0
                hi = np.searchsorted(times, t + span, side='right')
            if hi > lo:
                pieces.append(codes[lo:hi])
        if not pieces:
            continue
        exposed = np.concatenate(pieces)
        uniq, cnt = np.unique(exposed, return_counts=True)
        np.add.at(acc, uniq, cnt / cnt.sum())     # Pi^(0) ~ exposure count, normalised per post
        n_exposed += 1
    return acc, n_exposed


def compare(stored_col, tags, computed):
    """stored_col: Series indexed by hashtag; computed: vector over tag codes."""
    tag_to_code = pd.Series(np.arange(len(tags)), index=tags)
    s = stored_col[stored_col > 0]
    known = s.index.isin(tag_to_code.index)
    s = s[known]
    c_on_stored = computed[tag_to_code[s.index].to_numpy()]
    nonzero_computed = np.flatnonzero(computed > 1e-12)
    union = len(set(nonzero_computed) | set(tag_to_code[s.index].to_numpy()))
    diff = np.abs(c_on_stored - s.to_numpy())
    exact = int((diff < 1e-6).sum())
    corr = np.corrcoef(c_on_stored, s.to_numpy())[0, 1] if len(s) > 1 else np.nan
    return {'stored_tags': len(s), 'computed_tags': len(nonzero_computed), 'exact_matches': exact,
            'exact_fraction_of_union': exact / union if union else np.nan,
            'max_abs_diff': float(diff.max()) if len(diff) else np.nan, 'corr': corr,
            'sum_stored': float(s.sum()), 'sum_computed': float(computed.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--posts', required=True)
    ap.add_argument('--network', required=True)
    ap.add_argument('--stored', required=True, help='tag_likely_df_hour_..._date_hour_10min.pkl (index hashtag, columns window)')
    ap.add_argument('--windows', nargs='+', default=['2011-03-14 12:00', '2011-03-14 18:00'],
                    help='window start times (Tokyo local); keep them > 24 h away from the chunk boundaries of the original runs')
    ap.add_argument('--m-values', nargs='+', type=float, default=[1, 24, 48, 84])
    args = ap.parse_args()

    stored = pd.read_pickle(args.stored)
    stored.columns = to_naive_tokyo(stored.columns)
    data, tags, by_user, following = load(args.posts, args.network)

    hypotheses = [('trailing', m) for m in args.m_values] + [('legacy', m) for m in (1.0, 24.0)]
    for w in args.windows:
        w0 = pd.Timestamp(w)
        if w0 not in stored.columns:
            print(f'\n[{w}] not among the stored windows -- available range {stored.columns.min()} .. {stored.columns.max()}')
            continue
        t0 = np.int64(np.datetime64(w0, 's').astype(np.int64))
        window_posts = data[(data['t'] >= t0) & (data['t'] < t0 + 600)]
        col = stored[w0]
        print(f'\n===== window {w} : {len(window_posts):,} LCC posts; stored column: sum={col.sum():.1f}, '
              f'non-zero tags={int((col > 0).sum())} =====')
        if col.sum() > len(window_posts) + 1e-6:
            print('  !! stored sum exceeds the number of posts in the window -> check time zone / alignment')
        rows = []
        for mode, m in hypotheses:
            vec, n_exp = expected_counts(window_posts, by_user, following, len(tags), mode, m)
            r = compare(col, tags, vec)
            r.update({'hypothesis': f'{mode} m={m:g}h', 'posts_with_exposure': n_exp})
            rows.append(r)
            print(f"  {r['hypothesis']:>18}: posts_with_exposure={n_exp:6d}  sum={r['sum_computed']:9.1f}  "
                  f"exact={r['exact_matches']:6d}/{r['stored_tags']:6d}  union_frac={r['exact_fraction_of_union']:.3f}  "
                  f"max|diff|={r['max_abs_diff']:.3g}  corr={r['corr']:.4f}", flush=True)
        best = max(rows, key=lambda r: (r['exact_fraction_of_union'] if not np.isnan(r['exact_fraction_of_union']) else -1))
        print(f"  -> best match: {best['hypothesis']} (exact fraction {best['exact_fraction_of_union']:.3f})")


if __name__ == '__main__':
    main()
