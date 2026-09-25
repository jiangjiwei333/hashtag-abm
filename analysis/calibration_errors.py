"""calibration_errors.py -- error components used to calibrate Model 1 (Methods, "Model 1 Simulation
and Parameter Calibration") and to draw the error heat map (Fig. 9b).

For one simulation run the four components are
    usage_err       Kolmogorov-Smirnov distance between the empirical and simulated daily usage-count distributions
    logb_err        RMSE between the empirical and simulated densities of the daily logarithmic growth rate
                    (30 common equal-width bins, densities averaged over days)
    bt_scaling_err  RMSE between the empirical and simulated size-dependent standard deviations of log b
                    (increasing and decreasing sides, logarithmic size bins)
    death_err       RMSE of log deactivation probabilities across usage-count bins
Across a parameter grid each component is min-max normalised to [0, 1] and the four are summed (err_sum_top4);
the parameter set with the smallest sum is selected. All four components are computed from the final --days
model days of each run (default 7, as in the Methods section) and compared with the 7-day empirical data.

usage:
    python analysis/calibration_errors.py --real data/hashtag_daily_counts.pkl --run output/model1_m84_theta0.007_d013
    python analysis/calibration_errors.py --real data/hashtag_daily_counts.pkl --grid output/grid --out calibration_errors.csv
"""
import argparse
import glob
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from result_analysis import (compare_log_growth_pdf, growth_rate_scaling_errorbar_prepare_data, ks_distance,
                           log_rmse, next_day_analysis_new, normalize_error, rmse)


def compute_errors(sim, real, days=7):
    """Four error components for one run. `sim`, `real`: daily-count tables (index hashtag, columns days);
    only the final `days` columns of `sim` are used."""
    plt.figure()
    sim = sim.iloc[:, -days:]
    usage_err = ks_distance(real, sim)

    bt_sim = np.log10(sim.shift(-1, axis=1) / sim)
    bt_real = np.log10(real.shift(-1, axis=1) / real)
    logb_err = compare_log_growth_pdf(bt_sim, bt_real)

    ge_real, ne_real = growth_rate_scaling_errorbar_prepare_data(
        real.fillna(0).T, stationaryPoint=0, base=20, step=0.5, minmal_sample=30, show_detail=False)
    ge_sim, ne_sim = growth_rate_scaling_errorbar_prepare_data(
        sim.fillna(0).T, stationaryPoint=0, base=20, step=0.5, minmal_sample=30, show_detail=False)
    ge = pd.merge(ge_real, ge_sim, on='bin', how='left', suffixes=('_real', '_sim'))
    ne = pd.merge(ne_real, ne_sim, on='bin', how='left', suffixes=('_real', '_sim'))
    bt_scaling_err = rmse(ge['std50_real'], ge['std50_sim']) + rmse(ne['std50_real'], ne['std50_sim'])

    death_real = next_day_analysis_new(real, logy=True, minvalue=0, return_=True)
    death_sim = next_day_analysis_new(sim, logy=True, minvalue=0, return_=True, fit_line=False)
    d = pd.merge(death_real, death_sim, on='current_value', how='left', suffixes=('_real', '_sim'))
    death_err = log_rmse(d['death_rate_q2_real'], d['death_rate_q2_sim'])
    plt.close('all')
    return {'usage_err': usage_err, 'logb_err': logb_err, 'bt_scaling_err': bt_scaling_err, 'death_err': death_err}


def read_params(run_dir):
    """theta, d0, sigma, memory_hours from config_used.txt."""
    params = {}
    with open(os.path.join(run_dir, 'config_used.txt')) as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                params[k] = v
    return {'theta': float(params['theta']), 'd0': float(params['d0']),
            'sigma': float(params['sigma']), 'm': float(params['memory_hours'])}


def grid_errors(real, grid_dir, **kw):
    rows = []
    for run in sorted(glob.glob(os.path.join(grid_dir, '*'))):
        path = os.path.join(run, 'daily_counts.pkl')
        if not os.path.exists(path):
            print(f'skip {run}: run postprocess.py first')
            continue
        sim = pd.read_pickle(path)
        row = read_params(run)
        row.update(compute_errors(sim, real, **kw))
        row['run'] = run
        rows.append(row)
        print(f"{run}: " + ', '.join(f'{k}={row[k]:.4f}' for k in ['usage_err', 'logb_err', 'bt_scaling_err', 'death_err']))
    df = pd.DataFrame(rows)
    normalize_error(df)                                       # min-max normalisation of each component over the grid
    df['err_sum_top4'] = df[['usage_err', 'logb_err', 'bt_scaling_err', 'death_err']].sum(axis=1)
    return df.sort_values('err_sum_top4')


def error_heatmap(df, value='err_sum_top4', out=None):
    """Heat map of the total normalised error over (theta, d0) -- Fig. 9(b)."""
    import seaborn as sns
    pivot = df.pivot_table(index='theta', columns='d0', values=value, aggfunc='mean')
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, cmap='YlGnBu', cbar=True)
    i, j = np.where(pivot.values == np.nanmin(pivot.values))
    plt.scatter(j[0] + 0.5, i[0] + 0.5, color='red', s=300, marker='*')
    plt.xlabel(r'$d_0$'); plt.ylabel(r'$\theta$')
    plt.tight_layout()
    if out:
        plt.savefig(out, bbox_inches='tight')


def main():
    matplotlib.use('Agg')   # headless when run as a script (the statistics functions draw as a side effect)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--real', required=True, help='empirical daily-count table (pickle)')
    ap.add_argument('--run', help='one simulation run directory (after postprocess.py)')
    ap.add_argument('--grid', help='directory containing one run directory per parameter set')
    ap.add_argument('--out', default='calibration_errors.csv')
    ap.add_argument('--days', type=int, default=7, help='use the final DAYS model days of each run (Methods: 7)')
    args = ap.parse_args()
    real = pd.read_pickle(args.real)
    kw = dict(days=args.days)

    if args.run:
        sim = pd.read_pickle(os.path.join(args.run, 'daily_counts.pkl'))
        for k, v in compute_errors(sim, real, **kw).items():
            print(f'{k}: {v:.4f}')
    if args.grid:
        df = grid_errors(real, args.grid, **kw)
        df.to_csv(args.out, index=False)
        print('best parameter set:\n', df.iloc[0][['theta', 'd0', 'sigma', 'm', 'err_sum_top4']])
        error_heatmap(df, out=os.path.splitext(args.out)[0] + '_heatmap.pdf')


if __name__ == '__main__':
    main()
