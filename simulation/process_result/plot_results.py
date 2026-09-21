import sys
import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import time

import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'analysis'))
from result_analysis import plot_growth_rate_cdf, plot_realdata_usage, plot_simulated_usage
from result_analysis import growthRateDistribution, compare_log_growth_pdf
from result_analysis import growth_rate_scaling_errorbar_plot, growth_rate_scaling_errorbar_prepare_data
from result_analysis import prefrential_attachment
from result_analysis import next_day_analysis_new, next_day_analysis_new_new
from result_analysis import ks_distance, rmse, log_rmse, plot_cdf


def main():
    # print("Plot result...", end = '')
    path = sys.argv[1]
    out_path = sys.argv[2]
    show_progress = int(sys.argv[3])
    diffu_example_path = sys.argv[4]
    usage_micro_path = sys.argv[5]
    network = sys.argv[6]

    path_series = ".".join(path.split('.')[:-1]) + '.pkl'
    usage_micro_path = ".".join(usage_micro_path.split('.')[:-1]) + '.pkl'

    word_fulltime_df = pd.read_pickle(path_series)
    hashtag_df = pd.read_pickle('./real_data/hashtag_counts_daily.pkl')
    bt = np.log10(word_fulltime_df.shift(-1, axis=1)/word_fulltime_df)
    real_bt = np.log10(hashtag_df.shift(-1, axis=1)/hashtag_df)
    try: xt_df = pd.read_pickle(usage_micro_path)
    except: pass
    else: pass

    #------------------ phase ------------------
    try: 
        order_para = word_fulltime_df.iloc[:, -7:].max(axis=0)/word_fulltime_df.iloc[:, -7:].sum(axis=0)
    except: 
        order_para_q50 = np.nan
        order_para_q25 = np.nan
        order_para_q75 = np.nan
    else:
        order_para_q50 = np.quantile(order_para, 0.5)
        order_para_q25 = np.quantile(order_para, 0.25)
        order_para_q75 = np.quantile(order_para, 0.75)

    #------------------ usage ------------------
    plot_realdata_usage(hashtag_df, newfigure=True, c = 'k')
    plot_simulated_usage(word_fulltime_df.iloc[:, -7:], c = 'C0', newfigure=False, 
                         day_interval = 1, errorbar=True)
    usage_ks_error = ks_distance(hashtag_df, word_fulltime_df.iloc[:, -7:])
    usage_mean = word_fulltime_df.iloc[:, -7:].mean().mean()
    usage_std = word_fulltime_df.iloc[:, -7:].std().mean()
    log_usage_std = np.log(word_fulltime_df.iloc[:, -7:]).std().mean()
    fig_path = "/".join(out_path.split('/')[:-1]) + "/1_1_usage_" + (out_path.split('/')[-1])[:-4] + '.pdf'
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.05, dpi = 300, format='pdf') # , format='pdf', dpi=300
    fig_path = "/".join(out_path.split('/')[:-1]) + "/1_1_usage_" + (out_path.split('/')[-1])[:-4] + '.png'
    plt.savefig(fig_path, bbox_inches='tight') # , format='pdf', dpi=
    

    from result_analysis import plot_realdata_usage_cummulative, plot_simulated_usage_cummulative
    plot_realdata_usage_cummulative(hashtag_df, newfigure=True, c = 'k')
    plot_simulated_usage_cummulative(word_fulltime_df.iloc[:, -7:], c = 'C0', newfigure=False, day_interval = 1)
    fig_path = "/".join(out_path.split('/')[:-1]) + "/1_2_usagecum_" + (out_path.split('/')[-1])[:-4] + '.png'
    plt.savefig(fig_path, bbox_inches='tight') # , format='pdf', dpi=300

    #------------------ growth rate ------------------
    # CDF
    plot_growth_rate_cdf(hashtag_df, day_interval=1, label = 'real data', c = 'k', newfigure = True)
    plot_growth_rate_cdf(word_fulltime_df.iloc[:, -10:], day_interval=1, label = 'simulated', c = 'C0', newfigure = False)
    fig_path = "/".join(out_path.split('/')[:-1]) + "/2_1_growth_" + (out_path.split('/')[-1])[:-4] + '.png'
    plt.savefig(fig_path, bbox_inches='tight') # , format='pdf', dpi=300

    # PDF
    plt.figure()
    logbt_error  = compare_log_growth_pdf(bt.iloc[:, -7:], real_bt, real_label = 'real', simu_label = 'simulation')
    fig_path = "/".join(out_path.split('/')[:-1]) + "/2_2_growth_" + (out_path.split('/')[-1])[:-4] + '.pdf'
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.05, dpi = 300, format='pdf') # , format='pdf', dpi=300
    fig_path = "/".join(out_path.split('/')[:-1]) + "/2_2_growth_" + (out_path.split('/')[-1])[:-4] + '.png'
    plt.savefig(fig_path, bbox_inches='tight') # , format='pdf', dpi=
    logb_ge_std = bt[bt>0].iloc[:, -7:].std().mean()
    logb_ne_std = bt[bt<0].iloc[:, -7:].std().mean()

    # size dependence growth rate
    paras = growthRateDistribution(word_fulltime_df.iloc[:, -20:].fillna(0).T, stationaryPoint = 0, base = 10, step = 0.5, 
                                logx2=1, logy2=1, minmal_sample=80, cumulated_norm=True, show_detail = show_progress)
    fig_path = "/".join(out_path.split('/')[:-1]) + "/3_growth_size_" + (out_path.split('/')[-1])[:-4] + '.pdf'
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.05, dpi = 300, format='pdf') # , format='pdf', dpi=300
    fig_path = "/".join(out_path.split('/')[:-1]) + "/3_growth_size_" + (out_path.split('/')[-1])[:-4] + '.png'
    plt.savefig(fig_path, bbox_inches='tight') # , format='pdf', dpi=

    # growth rate scaling
    # growthRateScaling(*paras)
    std_ge_result_real, std_ne_result_real = growth_rate_scaling_errorbar_prepare_data(hashtag_df.fillna(0).T, 
                                            stationaryPoint = 0, base = 20, step = 0.5, minmal_sample=30, show_detail = False)
    std_ge_result, std_ne_result = growth_rate_scaling_errorbar_prepare_data(word_fulltime_df.iloc[:, -20:].fillna(0).T, 
                                            stationaryPoint = 0, base = 20, step = 0.5, minmal_sample=30, show_detail = False)
    cal_err_ge = pd.merge(std_ge_result_real, std_ge_result, left_on = 'bin', right_on = 'bin', how = 'left', suffixes=('_real', '_simulated'))
    cal_err_ne = pd.merge(std_ne_result_real, std_ne_result, left_on = 'bin', right_on = 'bin', how = 'left', suffixes=('_real', '_simulated'))
    bt_scaling_err = rmse(cal_err_ge['std50_real'], cal_err_ge['std50_simulated']) + rmse(cal_err_ne['std50_real'], cal_err_ne['std50_simulated'])
    plt.figure(figsize=(8, 6))
    growth_rate_scaling_errorbar_plot(std_ge_result, std_ne_result, label = ' simulated', c = ['C0', 'C0'], markersize=13.5)
    growth_rate_scaling_errorbar_plot(std_ge_result_real, std_ne_result_real, label = ' real', c = ['k', 'k'], markersize=12.5)
    fig_path = "/".join(out_path.split('/')[:-1]) + "/3_growth_scaling_" + (out_path.split('/')[-1])[:-4] + '.pdf'
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.05, dpi = 300, format='pdf') # , format='pdf', dpi=300
    fig_path = "/".join(out_path.split('/')[:-1]) + "/3_growth_scaling_" + (out_path.split('/')[-1])[:-4] + '.png'
    plt.savefig(fig_path, bbox_inches='tight') # , format='pdf', dpi=
    # ------------------ Prefrential Attachment ------------------
    prefrential_attachment(word_fulltime_df.iloc[:, -20:])
    fig_path = "/".join(out_path.split('/')[:-1]) + "/5_prefrential_" + (out_path.split('/')[-1])[:-4] + '.png'
    plt.savefig(fig_path, bbox_inches='tight') # , format='pdf', dpi=300

    #------------------ timeseries ------------------
    word_fulltime_df_last10 = word_fulltime_df.iloc[:, -10:].copy()
    word_fulltime_df_last10.dropna(inplace=True, how='all')
    sum_top_series = word_fulltime_df_last10.sum(axis = 1).sort_values(ascending = False).index
    oneday_top_series = word_fulltime_df_last10.max(axis = 1).sort_values(ascending = False).index

    try:
        cnt = 0
        max_day = 30
        top_n = 6
        ax1 = plt.figure(figsize = (25, 8))
        while cnt < top_n:
            tag = sum_top_series[cnt]
            high_reso_series = xt_df[xt_df.tag == tag]
            if len(high_reso_series) <= 1: continue
            ax1 = plt.subplot(2, 3, cnt+1)
            ax2 = plt.twinx()
            ax1.plot(high_reso_series['time'], high_reso_series['value'], '-X', label = fr'$x_{{{tag}}}(t_{{10min}})$')
            ax1.legend(loc='upper left', fontsize = 20)

            low_reso_series = word_fulltime_df_last10.loc[tag, :]
            ax2.plot(low_reso_series.index*24*6 + 24 * 3, low_reso_series, '-o', label = fr'$x_{{{tag}}}(t_{{1day}})$', c = 'C1')
            ax2.legend(loc = 'upper right', fontsize = 20)
            for day in range(1, max_day): plt.axvline(x=24*6*day, color='gray', linestyle='--', alpha=0.3)
            plt.xlim(high_reso_series['time'].min() - 24*3, high_reso_series['time'].max() + 24*6)
            plt.xlim(low_reso_series.index.min()*24*6 - 24*3, low_reso_series.index.max()*24*6 + 24*6)
            cnt += 1
        plt.suptitle(f'Sum usage top {top_n} Series', fontsize=24)
        plt.tight_layout()
        fig_path = "/".join(out_path.split('/')[:-1]) + "/10_Series_1" + (out_path.split('/')[-1])[:-4] + '.pdf'
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.05, dpi = 300, format='pdf') # , format='pdf', dpi=300
        fig_path = "/".join(out_path.split('/')[:-1]) + "/10_Series_1" + (out_path.split('/')[-1])[:-4] + '.png'
        plt.savefig(fig_path, bbox_inches='tight') # , format='pdf', dpi=

        cnt = 0
        ax1 = plt.figure(figsize = (25, 8))
        while cnt < top_n:
            tag = oneday_top_series[cnt]
            high_reso_series = xt_df[xt_df.tag == tag]
            if len(high_reso_series) <= 1: continue
            ax1 = plt.subplot(2, 3, cnt+1)
            ax2 = plt.twinx()
            ax1.plot(high_reso_series['time'], high_reso_series['value'], '-X', label = fr'$x_{{{tag}}}(t_{{10min}})$')
            ax1.legend(loc='upper left', fontsize = 20)

            low_reso_series = word_fulltime_df_last10.loc[tag, :]
            ax2.plot(low_reso_series.index*24*6 + 24 * 3, low_reso_series, '-o', label = fr'$x_{{{tag}}}(t_{{1day}})$', c = 'C1')
            ax2.legend(loc = 'upper right', fontsize = 20)
            for day in range(1, max_day): plt.axvline(x=24*6*day, color='gray', linestyle='--', alpha=0.3)
            plt.xlim(high_reso_series['time'].min() - 24*3, high_reso_series['time'].max() + 24*6)
            cnt += 1
        plt.suptitle(f'Oneday usage top {top_n} Series', fontsize=24)
        plt.tight_layout()
        fig_path = "/".join(out_path.split('/')[:-1]) + "/10_Series_2" + (out_path.split('/')[-1])[:-4] + '.pdf'
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.05, dpi = 300, format='pdf') # , format='pdf', dpi=300
        fig_path = "/".join(out_path.split('/')[:-1]) + "/10_Series_2" + (out_path.split('/')[-1])[:-4] + '.png'
        plt.savefig(fig_path, bbox_inches='tight') # , format='pdf', dpi=
    except: pass
    else: pass

    #------------------ prob of become 0 ------------------
    try:
        plt.figure(figsize=(8, 6))
        error_real = next_day_analysis_new(hashtag_df, logy = True, minvalue=0, label = 'real', c = 'k', return_=True)
        error_simu = next_day_analysis_new(word_fulltime_df.iloc[:, -10:], logy = True, minvalue=0, label = 'simulated', c = 'C0', return_=True, fit_line = False)
        cal_err = pd.merge(error_real, error_simu, left_on = 'current_value', right_on = 'current_value', 
                        how = 'left', suffixes=('_real', '_simulated'))
        death_err = log_rmse(cal_err['death_rate_q2_real'], cal_err['death_rate_q2_simulated'])
        fig_path = "/".join(out_path.split('/')[:-1]) + "/11_a_Prob_become0_" + (out_path.split('/')[-1])[:-4] + '.pdf'
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.05, dpi = 300, format='pdf')
        fig_path = "/".join(out_path.split('/')[:-1]) + "/11_a_Prob_become0_" + (out_path.split('/')[-1])[:-4] + '.png'
        plt.savefig(fig_path, bbox_inches='tight')


        plt.figure(figsize=(8, 6))
        error_real = next_day_analysis_new_new(hashtag_df, logy = True, minvalue=0, label = 'real', c = 'k', return_=True)
        error_simu = next_day_analysis_new_new(word_fulltime_df.iloc[:, -10:], logy = True, minvalue=0, label = 'simulated', c = 'C0', return_=True, fit_line = False)
        cal_err = pd.merge(error_real, error_simu, left_on = 'current_value', right_on = 'current_value', 
                        how = 'left', suffixes=('_real', '_simulated'))
        death_err = log_rmse(cal_err['death_rate_q2_real'], cal_err['death_rate_q2_simulated'])
        fig_path = "/".join(out_path.split('/')[:-1]) + "/11_b_Prob_become0_" + (out_path.split('/')[-1])[:-4] + '.pdf'
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.05, dpi = 300, format='pdf')
        fig_path = "/".join(out_path.split('/')[:-1]) + "/11_b_Prob_become0_" + (out_path.split('/')[-1])[:-4] + '.png'
        plt.savefig(fig_path, bbox_inches='tight')
    except: death_err = np.nan
    else: pass


    #------------------ O-U simualtion example series ------------------
    try:
        with open(diffu_example_path, 'r') as f: lines = f.readlines()
        tag_name = []
        processed_data = []
        for i, line in enumerate(lines):
            dks = np.array(list(map(float, line.strip().split(','))))
            processed_data.append(dks[1:])
            tag_name.append(int(dks[0]))
        diffu_eg_series = pd.DataFrame(processed_data, index=tag_name)
    except: pass
    else:
        # Model 0 (const_dk) records a single value per sampled hashtag, because d_k never
        # changes; there is no trajectory to plot and the CDF panel has no columns to index.
        n_rows, n_cols = diffu_eg_series.shape
        if n_cols < 2:
            print(f"(d_k is constant in this run: {n_rows} sampled hashtags, 1 value each -- skipping the trajectory plot)")
        else:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            for i in range(min(10, n_rows)):
                tmp = diffu_eg_series.iloc[i, :]
                plt.plot(tmp[tmp>0], label = f'Tag {i}')
                plt.plot(tmp[tmp<0], c = 'gray', alpha = 0.3)
            plt.axhline(y=0, color='red', linestyle='--', linewidth = 3, alpha=0.8)
            maxday = 7
            for i in range(0, 24 * 6 * maxday, 6 * 24): plt.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
            plt.xlim([-10, 24 * 6 * maxday + 10])
            plt.legend(fontsize = 18, ncol=2)
            plt.xlabel(r'$t^{\prime}$')
            plt.ylabel(r'$d_k(t^{\prime})$')
            plt.subplot(1, 2, 2)
            for i in range(24 * 6 * 5, min(24 * 6 * 10, n_cols), 6):
                plot_cdf(diffu_eg_series.iloc[:, i], logx=0, logy=0)
            plt.xlabel(r'$d_k(t^{\prime})$')
            plt.ylabel('CDF')
            plt.tight_layout();
            fig_path = "/".join(out_path.split('/')[:-1]) + "/18_diffu_eg_series_" + (out_path.split('/')[-1])[:-4] + '.png'
            plt.savefig(fig_path, bbox_inches='tight')


    # flush figures and buffers before the calling shell reads the last line
    plt.close('all')
    time.sleep(10)
    
    import gc
    gc.collect()
    
    sys.stdout.flush()
    sys.stderr.flush()
    
    #------------------ Output error ------------------
    taylor_err = 0
    heaps_err = 0
    print(usage_ks_error, logbt_error, bt_scaling_err, death_err, heaps_err, taylor_err, end=' ')
    print(usage_mean, usage_std, log_usage_std, logb_ge_std, logb_ne_std, end=' ')
    for x in cal_err_ge['std50_simulated']: print(x, end = ' ')
    for x in cal_err_ne['std50_simulated']: print(x, end = ' ')
    print(word_fulltime_df.shape[0], order_para_q25, order_para_q50, order_para_q75)
    
    sys.stdout.flush()

if __name__ == "__main__":
    sys.exit(main())

