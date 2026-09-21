"""postprocess.py -- convert raw simulation outputs into analysis-ready tables.

Inputs (written by simulation/hashtag_abm into <run_dir>/):
    posted_record.txt   one line per saved model day; comma-separated hashtag ids in posting order
    window_counts.txt   one line per 10-minute window (from window_output_start_day); "tag,count,tag,count,..."
    d_samples.txt       one line per sampled hashtag: "tag,d(0),d(1),..." (Model 1 only)

Outputs (pickled pandas objects in the same directory):
    daily_counts.pkl    DataFrame, index = hashtag id, columns = model day (0..D-1), value = daily usage count x_k(t)
                        (NaN where the hashtag was not used that day). This is the simulated analogue of the
                        empirical daily-count table hashtag_daily_counts.pkl.
    window_counts.pkl   long DataFrame with columns [time, tag, value]: 10-minute counts x_k(r)
    d_samples.pkl       DataFrame, index = hashtag id, columns = window index, value = d_k(r)

usage:  python analysis/postprocess.py <run_dir> [--window-start-day 20]
"""
import argparse
import os

import numpy as np
import pandas as pd


def daily_counts(path):
    """Daily usage counts of every hashtag from the posted-record file."""
    counts = {}
    with open(path) as f:
        for day, line in enumerate(f):
            tags = np.fromstring(line.strip(), dtype=np.int64, sep=',')
            counts[day] = pd.Series(tags).value_counts()
    df = pd.DataFrame(counts)          # index: hashtag id, columns: day; NaN = not used
    df.index.name = 'tag'
    return df.sort_index()


def window_counts_long(path, start_window):
    """10-minute counts as a long table (time, tag, value); time is the global window index."""
    frames = []
    with open(path) as f:
        for i, line in enumerate(f):
            v = np.fromstring(line.strip(), dtype=np.int64, sep=',')
            if v.size == 0:
                continue
            frames.append(pd.DataFrame({'time': start_window + i, 'tag': v[0::2], 'value': v[1::2]}))
    return pd.concat(frames, ignore_index=True)


def d_samples(path):
    rows, tags = [], []
    with open(path) as f:
        for line in f:
            v = np.fromstring(line.strip(), dtype=float, sep=',')
            tags.append(int(v[0]))
            rows.append(v[1:])
    return pd.DataFrame(rows, index=tags)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('--window-start-day', type=int, default=20,
                    help='window_output_start_day used in the simulation (default 20)')
    args = ap.parse_args()
    run = args.run_dir

    df = daily_counts(os.path.join(run, 'posted_record.txt'))
    df.to_pickle(os.path.join(run, 'daily_counts.pkl'))
    print(f'daily_counts.pkl: {df.shape[0]} hashtags x {df.shape[1]} days')

    wpath = os.path.join(run, 'window_counts.txt')
    if os.path.exists(wpath):
        w = window_counts_long(wpath, start_window=24 * 6 * args.window_start_day)
        w.to_pickle(os.path.join(run, 'window_counts.pkl'))
        print(f'window_counts.pkl: {len(w)} (window, hashtag) records')

    spath = os.path.join(run, 'd_samples.txt')
    if os.path.exists(spath):
        s = d_samples(spath)
        s.to_pickle(os.path.join(run, 'd_samples.pkl'))
        print(f'd_samples.pkl: {s.shape[0]} sampled hashtags x {s.shape[1]} windows')


if __name__ == '__main__':
    main()
