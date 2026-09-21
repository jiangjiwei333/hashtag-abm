"""result_analysis.py -- statistics and plotting routines for the paper figures.

Trimmed to the functions used by figures/paper_figures.ipynb and simulation/process_result/plot_results.py;
the function bodies are unchanged from the research code.

Daily hashtag counts are DataFrames indexed by hashtag with one column per day. Growth rates are
b_k(t) = x_k(t+1)/x_k(t) and are analysed on a log10 scale throughout.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from scipy import stats
from scipy.integrate import cumulative_trapezoid
from statsmodels.tsa.stattools import acf
from tqdm import tqdm, trange
from matplotlib.ticker import LogLocator

# figure style shared by all panels
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('default')
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["legend.framealpha"] = 0.2
plt.rcParams["legend.frameon"] = True
sns.set_style("ticks")
sns.set_context("notebook", font_scale=1.5,
                rc={'font.size': 10, 'axes.labelsize': 35, 'legend.fontsize': 10,
                    'xtick.labelsize': 25, 'ytick.labelsize': 25, 'figure.titlesize': 35})

# ----------------------------------------------------------------------------- error measures (calibration)
def rmse(fitted, real):
    """RMSE between two probability densities."""
    return np.sqrt(np.mean((fitted - real)**2))

def log_rmse(fitted, real):
    mask = (real > 0) & (fitted > 0)
    """RMSE between two log10 probability densities."""
    return np.sqrt(np.mean((np.log(fitted[mask]) - np.log(real[mask]))**2))

def ks_distance(fitted_data, real_data):
    """Kolmogorov-Smirnov statistic and p-value of two samples."""
    # flatten
    fitted_flat = fitted_data.values.flatten()
    real_flat = real_data.values.flatten()
    
    # drop NaN
    fitted_clean = fitted_flat[~np.isnan(fitted_flat)]
    real_clean = real_flat[~np.isnan(real_flat)]
    
    # two-sample KS test
    ks_stat, p_value = stats.ks_2samp(fitted_clean, real_clean)
    return ks_stat

def normalize(x): # normalize error to [0, 1]
    return (x - x.min()) / (x.max() - x.min())

def normalize_error(error_data):
    error_data['usage_err'] = normalize(error_data['usage_err'])
    error_data['logb_err'] = normalize(error_data['logb_err'])
    error_data['bt_scaling_err'] = normalize(error_data['bt_scaling_err'])
    error_data['death_err'] = normalize(error_data['death_err'])

def plot_realdata_usage(hashtag_df, newfigure = True, c = 'k', errorbar = False, raw = True, marker = 'o'):
    if newfigure: plt.figure(figsize = (8, 6))
    if raw == True:
        for i in range(hashtag_df.shape[1]): 
            plot_cdf(hashtag_df.iloc[:, i], c = c, label = 'real' if i == hashtag_df.shape[1] - 1 else None)

    if errorbar == True:
        cdf_df = pd.DataFrame(columns=['usage', 'cdf', 'day'])
        for i in range(hashtag_df.shape[1]):
            oneday = np.array(hashtag_df.iloc[:, i].dropna())
            cdf = 1 - np.arange(len(oneday))/len(oneday)
            oneday.sort()
            cdf_df = pd.concat([cdf_df, pd.DataFrame({'usage': oneday, 'cdf': cdf, 'day': i})], ignore_index=True)
        bins = np.array([(2)**i for i in range(17)])  # initial bins
        cdf_df['indeices'] = np.digitize(cdf_df.usage, bins)
        tmp = cdf_df.groupby(['indeices', 'day']).agg(cdf = ('cdf', 'max')).reset_index()
        cdf_error = tmp.groupby(['indeices']).agg(q1 = ('cdf', lambda x: x.quantile(0.05)),
                                        q2 = ('cdf', lambda x: x.quantile(0.5)),
                                        q3 = ('cdf', lambda x: x.quantile(0.95))).reset_index()
        plt.errorbar(bins[cdf_error.indeices - 1], cdf_error.q2,
                yerr=[cdf_error.q2 - cdf_error.q1, cdf_error.q3 - cdf_error.q2],
                capsize=8, fmt = marker, markersize = 10, ecolor= "C0", markeredgecolor = "k", color = 'C0',
                capthick=2, linewidth=2,
                label = 'real')
        plt.plot(bins[cdf_error.indeices - 1], cdf_error.q2, color=c)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize = 20)
    plt.xlabel(r'$x(t)$')
    plt.ylabel(r'$P(\geq x(t))$');

def plot_simulated_usage(word_fulltime_df, newfigure = True, day_interval = 1, c = 'C0', errorbar = False, raw = True, label = None, marker = 'o'):
    if newfigure: plt.figure(figsize = (8, 6))
    if raw == True:
        for i in range(0, word_fulltime_df.shape[1], day_interval):
            plot_cdf(word_fulltime_df.iloc[:, i].dropna(), c = c, alpha = 0.5)
    if errorbar == True:
        cdf_df = pd.DataFrame(columns=['usage', 'cdf', 'day'])
        for i in range(word_fulltime_df.shape[1]):
            oneday = np.array(word_fulltime_df.iloc[:, i].dropna())
            cdf = 1 - np.arange(len(oneday))/len(oneday)
            oneday.sort()
            cdf_df = pd.concat([cdf_df, pd.DataFrame({'usage': oneday, 'cdf': cdf, 'day': i})], ignore_index=True)
        bins = np.array([(2)**i for i in range(17)])  # initial bins
        cdf_df['indeices'] = np.digitize(cdf_df.usage, bins)
        tmp = cdf_df.groupby(['indeices', 'day']).agg(cdf = ('cdf', 'max')).reset_index()
        cdf_error = tmp.groupby(['indeices']).agg(q1 = ('cdf', lambda x: x.quantile(0.05)),
                                        q2 = ('cdf', lambda x: x.quantile(0.5)),
                                        q3 = ('cdf', lambda x: x.quantile(0.95))).reset_index()
        plt.errorbar(bins[cdf_error.indeices - 1], cdf_error.q2,
                yerr=[cdf_error.q2 - cdf_error.q1, cdf_error.q3 - cdf_error.q2],
                capsize=8, fmt = marker, markersize = 10, ecolor= "C1", markeredgecolor = "k", color = 'C1',
                capthick=2, linewidth=2,
                label = 'simulation' if label == None else label)
        plt.plot(bins[cdf_error.indeices - 1], cdf_error.q2, color=c)
    plt.legend(fontsize = 20, frameon = False)
    plt.xlabel(r'$x(t)$')
    plt.ylabel(r'$P(\geq x(t))$');
    plt.tight_layout()

def plot_realdata_usage_cummulative(hashtag_df, newfigure = True, c = 'k'):
    if newfigure: plt.figure(figsize = (10, 8))
    # if newfigure: plt.figure()
    for i in range(hashtag_df.shape[1]):
        plot_cdf(hashtag_df.iloc[:, :i+1].sum(axis = 1), c = c, 
                 label = 'real data' if i == hashtag_df.shape[1] - 1 else None)
    plt.legend(fontsize = 20)
    plt.xlabel(r'$x(t)$')
    plt.ylabel(r'$P(\geq x(t))$');

def plot_simulated_usage_cummulative(word_fulltime_df, newfigure = True, day_interval = 1, c = 'C0'):
    if newfigure: plt.figure(figsize = (10, 8))
    # row cdf line
    for i in range(0, word_fulltime_df.shape[1], day_interval):
        plot_cdf(word_fulltime_df.iloc[:, :i+1].sum(axis = 1).dropna(), c = c, label = 'simulated' if i == 0 else None, alpha = 0.5)
    plt.legend(fontsize = 20)
    plt.xlabel(r'$\sum_{t^{\prime}\leq t} x(t^{\prime})$')
    plt.ylabel(r'$P(\geq \sum_{t^{\prime}\leq t} x(t^{\prime}))$');

def plot_growth_rate_cdf(word_fulltime_df, logx = True, logy = True, 
                         plot_agg = False, day_interval = 10, 
                         label = None, c = None, newfigure = True):
    if newfigure: plt.figure(figsize=(10, 8))
    bt = word_fulltime_df.shift(-1, axis = 1)/word_fulltime_df
    cnt = 1
    for i in range(0, word_fulltime_df.shape[1]-1, day_interval):
        b_ls = np.array(bt.iloc[:, i]).flatten()
        b_ls = b_ls[~np.isnan(b_ls)]
        b_ls = b_ls[~np.isinf(b_ls)]
        plot_cdf(b_ls[b_ls>1], logx=logx, 
                 c = f'C{cnt}' if c==None else c, 
                #  label = label if i == 0 else None, alpha = 0.5)
                 label = word_fulltime_df.columns[i], alpha = 0.5)
        plot_cdf(b_ls[b_ls<1], logx=logx, left=True, 
                 alpha = 0.5, 
                 c = f'C{cnt}' if c==None else c)
                #  c = f'C{i}')
        cnt += 1
    if plot_agg == True:
        b_ls = np.array(bt).flatten()
        b_ls = b_ls[~np.isnan(b_ls)]
        b_ls = b_ls[~np.isinf(b_ls)]
        plot_cdf(b_ls[b_ls>1], logx=logx, logy = logy, c = 'k')
        plot_cdf(b_ls[b_ls<1], logx=logx, logy = logy, left=True, c = 'k')
    plt.xlabel(r'$b(t)$')
    plt.ylabel(r'CDF')
    if label != None: plt.legend(fontsize = 20)
    plt.grid(False)
    plt.tight_layout()


# ----------------------------------------------------------------------------- growth rates
def growthRateDistribution(hashtagDf, stationaryPoint=100, base=10, step=1, logx2=1, logy2=1, 
                           revise = True, 
                           minmal_sample = 80, 
                           cumulated_norm = True,
                           plot_jump_sample = 1, 
                           show_detail = True,
                           figsize = None):
    def get_max_exponent(hashtagDf):
        hashtagMax = hashtagDf.max().max()
        i = 0
        while (int(hashtagMax/base**i) != 0):
            i += 1
        return i
    hashtagDf.fillna(0, inplace = True)
    hashtagDf = hashtagDf.iloc[stationaryPoint:, :]
    growthRateDf = hashtagDf.shift(-1, axis=0)/hashtagDf
    growthRateInArr = []
    growthRateDeArr = []
    binLs = []
    muLs = []
    stdLs = []
    muGeLs = []
    stdGeLs = []
    binGeLs = []
    muNeLs = []
    stdNeLs = []
    binNeLs = []
    if figsize is not None: plt.figure(figsize=figsize)
    else: plt.figure(figsize=(8, 6))
    color = 0
    for cnt, i in enumerate(np.arange(0, get_max_exponent(hashtagDf), step)):
        growthRateDf_xt = growthRateDf[(hashtagDf >= base ** i) &
                                       (hashtagDf < base ** (i + step))]
        label = r'$[{%1.0f}^{%1.1f}, {%1.0f}^{%1.1f})$' % (base, i, base, i+step)

        growthRateDf_xt = np.array(growthRateDf_xt).flatten()
        growthRateDf_xt = growthRateDf_xt[~np.isnan(growthRateDf_xt)]
        growthRateDf_xt = growthRateDf_xt[~np.isinf(growthRateDf_xt)]
        if (len(growthRateDf_xt!=0) and show_detail == True):
            print(f'0 ratio = {len(growthRateDf_xt[growthRateDf_xt==0])/len(growthRateDf_xt)}', end = ', ')
        growthRateDf_xt = growthRateDf_xt[growthRateDf_xt!=0] # zero counts carry no growth rate



        increase = growthRateDf_xt[growthRateDf_xt > 1]
        increase = np.sort(increase)
        growthRateInArr.extend(increase)
        if cumulated_norm == True:
            cdf_inc = 1 - np.arange(len(increase))/len(increase)
        else:
            if len(growthRateDf_xt)>0:
                cdf_inc = 1 - ( (len(growthRateDf_xt) - len(increase))/len(growthRateDf_xt) + np.arange(len(increase))/len(growthRateDf_xt))

        decrease = growthRateDf_xt[growthRateDf_xt < 1]
        decrease = decrease[decrease != 0] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if revise == True: decrease = np.sort(decrease)
        else: decrease = np.sort(1/decrease)
        growthRateDeArr.extend(decrease)
        if revise == True: 
            if cumulated_norm == True:
                cdf_dec = (np.arange(len(decrease)) + 1)/len(decrease)
            else:
                if len(growthRateDf_xt)>0:
                    cdf_dec = (np.arange(len(decrease)) + 1)/len(growthRateDf_xt)
        else: cdf_dec = 1 - np.arange(len(decrease))/len(decrease)
        if show_detail == True:
            print(f'{label}: increase: {len(increase)}; decrease: {len(decrease)}')
        if len(growthRateDf_xt) > minmal_sample:
            # if if_all == True:
            binLs.append(base ** (i + step/2))
            muLs.append(np.mean(np.log10(growthRateDf_xt)))
            stdLs.append(np.std(np.log10(growthRateDf_xt)))
            if (len(increase) > minmal_sample) & (len(decrease) > minmal_sample):
                binGeLs.append(base ** (i + step/2))
                muGeLs.append(np.mean(np.log10(increase)))
                stdGeLs.append(np.std(np.log10(increase)))
                binNeLs.append(base ** (i + step/2))
                muNeLs.append(np.mean(np.log10(decrease)))
                stdNeLs.append(np.std(np.log10(decrease)))
                if (cnt+1)%plot_jump_sample==0:
                    # plt.subplot(1, 2, 1)
                    plt.plot(decrease, cdf_dec, '--.', c = f'C{color}', label=label, rasterized=True)
                    # plt.subplot(1, 2, 2)
                    plt.plot(increase, cdf_inc, '--.', rasterized=True
                            #  label=label, c = f'C{color}'
                             )
                    color += 1
    for i in range(2):
        # plt.subplot(1, 2, i+1)  
        plt.ylabel(r'CDF')
        plt.xlabel(r'$b(t|x(t))$')
        if logx2 == 1: plt.xscale('log')
        if logy2 == 1: plt.yscale('log')
        plt.yticks([10**(-i) for i in range(5)])
        if i == 0: 
            plt.xticks([10**(i) for i in range(-3, 1)])
            # plt.ylabel(r'$P(\leq \log (b(t)))$')
            plt.ylabel('CDF')
            # plt.legend(fontsize=20, ncol=1, bbox_to_anchor=(1.02, 0.1), loc=3, borderaxespad=0)
            plt.legend(fontsize = 16.5, framealpha=0, loc = 'upper right')
        else: 
            plt.xticks([10**(i) for i in range(0, 4)])
            # plt.ylabel(r'$P(\geq \log (b(t)))$')
            plt.ylabel('CDF')
        plt.xticks([10**(i) for i in range(-3, 4)])
        
    # x = np.linspace(1, 10**3.5, 100)
    # alpha = -1.3
    # y = 0.5 * (x/min(x))**alpha
    # plt.plot(x, y, '--', c='grey', linewidth = 3)

    # x = np.linspace(10**-(3.5), 10**(0), 100)
    # alpha = -1.5
    # y = 0.5 * (x/max(x))**(-alpha)
    # plt.plot(x, y, '--', c='grey', linewidth = 3)
    plt.tight_layout()
    return binLs, muLs, stdLs, binGeLs, muGeLs, stdGeLs, binNeLs, muNeLs, stdNeLs

def compare_log_growth_pdf(fitted_data, real_data, bins=30, 
                           real_bar = True, real_label = None, real_c = 'C0',
                           simu_label = None, simu_c = 'C1', simu_bar = True, simu_line = False, 
                           simu_marker = 's'
                           ):
    """
    Compare probability density functions of fitted and real log growth rate data with error bars
    showing variation across dates (columns)
    
    Parameters:
    fitted_data: DataFrame, fitted data (rows x days/columns)
    real_data: DataFrame, real data (rows x days/columns)
    bins: int, number of bins for probability density calculation
    figsize: tuple, figure size
    """
    
    # Get the minimum number of columns to ensure both DataFrames have the same columns
    n_cols_fitted = fitted_data.shape[1]
    n_cols_real = real_data.shape[1]
    n_dates = min(n_cols_fitted, n_cols_real)
    
    # Use integer indices instead of column names to avoid mismatch
    fitted_subset = fitted_data.iloc[:, :n_dates]
    real_subset = real_data.iloc[:, :n_dates]
    
    # Determine common bin edges
    all_data = np.concatenate([fitted_subset.values.flatten(), real_subset.values.flatten()])
    all_data_clean = all_data[~np.isnan(all_data)]
    bin_edges = np.linspace(all_data_clean.min(), all_data_clean.max(), bins + 1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    # Calculate probability density for each date (column)
    fitted_densities = []
    real_densities = []
    
    for i in range(n_dates):
        # Fitted data for this date
        fitted_col_data = fitted_subset.iloc[:, i].dropna().values
        if len(fitted_col_data) > 0:
            hist_fitted, _ = np.histogram(fitted_col_data, bins=bin_edges, density=True)
            fitted_densities.append(hist_fitted)
        else:
            fitted_densities.append(np.zeros(len(bin_centers)))
        
        # Real data for this date
        real_col_data = real_subset.iloc[:, i].dropna().values
        if len(real_col_data) > 0:
            hist_real, _ = np.histogram(real_col_data, bins=bin_edges, density=True)
            real_densities.append(hist_real)
        else:
            real_densities.append(np.zeros(len(bin_centers)))
    
    # Convert to arrays and calculate statistics
    fitted_densities = np.array(fitted_densities).T  # Shape: (bins, dates)
    real_densities = np.array(real_densities).T      # Shape: (bins, dates)
    
    # Calculate mean and std across dates for each bin
    fitted_mean = np.mean(fitted_densities, axis=1)
    fitted_std = np.std(fitted_densities, axis=1)
    real_mean = np.mean(real_densities, axis=1)
    real_std = np.std(real_densities, axis=1)
    
    # Create figure
    # fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot with error bars (add small offset to avoid overlapping)
    offset = (bin_centers[1] - bin_centers[0]) * 0.02

    if real_bar:
        line1 = plt.errorbar(bin_centers + offset, real_mean, yerr=real_std, 
                    # capsize=5, fmt = 's-', markersize = 8, 
                    fmt='o-', capsize=5, markersize=8, 
                    ecolor=real_c, markeredgecolor = "k", color = real_c,
                    capthick=2, linewidth=2, 
                    label = real_label, alpha=0.8)
    
    if simu_bar:
        line2 = plt.errorbar(bin_centers - offset, fitted_mean, yerr=fitted_std, 
                fmt=simu_marker if not simu_line else simu_marker + '-', capsize=5, markersize=8, 
                ecolor= simu_c, markeredgecolor = 'k', color = simu_c,
                capthick=2, linewidth=2, 
                label= simu_label, alpha=0.8)

    plt.yscale('log')
    plt.xlabel(r'$\log b(t)$')
    plt.ylabel(r'$p(\log b(t))$')
    plt.xticks(range(-4, 5, 1))
    if real_label is not None or simu_label is not None:
        plt.legend(fontsize=20, loc = 'upper right', frameon = False)

    plt.tight_layout()
    return rmse(fitted_mean, real_mean)

def growthRateScaling(binLs, muLs, stdLs, binGeLs, muGeLs, stdGeLs, binNeLs, muNeLs, stdNeLs, if_all=False):
    zoom = 2
    plt.figure(figsize=(4*2*zoom, 3*zoom))
    plt.subplot(1, 2, 1)

    plt.scatter(binGeLs, stdGeLs, label=r'$\log b>0$', s = 80)
    plt.scatter(binNeLs, stdNeLs, label=r'$\log b<0$', c='r', s = 80)
    if if_all == True:  plt.plot(binLs, stdLs, '--o', label='all', alpha = 0.5)
    plt.xscale('log')
    plt.legend(fontsize=20)
    plt.xlabel('$x(t)$')
    plt.ylabel(r'$\sigma(\log b)$')
    plt.xticks([10**i for i in range(6)])

    plt.subplot(1, 2, 2)
    plt.scatter(binGeLs, muGeLs, s = 80, label='Increase')
    plt.scatter(binNeLs, muNeLs, s = 80, label='Decrease', c='k')
    if if_all == True: plt.plot(binLs, muLs, '--o', label='all', alpha= 0.5)
    plt.xscale('log')
    plt.legend(fontsize=12)
    plt.xlabel('$x(t)$')
    plt.ylabel(r'$\mu(\log b)$')
    plt.tight_layout()

def growth_rate_scaling_errorbar_prepare_data(hashtagDf, stationaryPoint=0, base=10, step=1, 
                                 revise = True, minmal_sample = 80, show_detail = True):
    def get_max_exponent(hashtagDf):
        hashtagMax = hashtagDf.max().max()
        i = 0
        while (int(hashtagMax/base**i) != 0):
            i += 1
        return i
    hashtagDf.fillna(0, inplace = True)
    hashtagDf = hashtagDf.iloc[stationaryPoint:, :]
    growthRateDf = hashtagDf.shift(-1, axis=0)/hashtagDf

    std_ge_result = []
    std_ne_result = []
    for i in np.arange(0, get_max_exponent(hashtagDf), step):
        binLs = []
        muLs = []
        stdLs = []
        muGeLs = []
        stdGeLs = []
        binGeLs = []
        muNeLs = []
        stdNeLs = []
        binNeLs = []

        growthRateDf_xt_all = growthRateDf[(hashtagDf >= base ** i) &
                                       (hashtagDf < base ** (i + step))]
        label = r'$[{%1.0f}^{%1.1f}, {%1.0f}^{%1.1f})$' % (base, i, base, i+step)
        for day in range(growthRateDf_xt_all.shape[0]-1):
            growthRateDf_xt = np.array(growthRateDf_xt_all.iloc[day, :]).flatten()
            growthRateDf_xt = growthRateDf_xt[~np.isnan(growthRateDf_xt)]
            growthRateDf_xt = growthRateDf_xt[~np.isinf(growthRateDf_xt)]
            if (len(growthRateDf_xt!=0) and show_detail == True):
                print(f'0 ratio = {len(growthRateDf_xt[growthRateDf_xt==0])/len(growthRateDf_xt)}', end = ', ')
            growthRateDf_xt = growthRateDf_xt[growthRateDf_xt!=0] # zero counts carry no growth rate

            increase = growthRateDf_xt[growthRateDf_xt > 1] 
            increase = np.sort(increase)
            decrease = growthRateDf_xt[growthRateDf_xt < 1]
            decrease = decrease[decrease != 0] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if revise == True: decrease = np.sort(decrease)
            else: decrease = np.sort(1/decrease)
            if show_detail == True:
                print(f'{label}: increase: {len(increase)}; decrease: {len(decrease)}')
            if len(growthRateDf_xt) > minmal_sample:
                # if if_all == True:
                binLs.append(base ** (i + step/2))
                muLs.append(np.mean(np.log10(growthRateDf_xt)))
                stdLs.append(np.std(np.log10(growthRateDf_xt)))
                if len(increase) > minmal_sample:
                    binGeLs.append(base ** (i + step/2))
                    muGeLs.append(np.mean(np.log10(increase)))
                    stdGeLs.append(np.std(np.log10(increase)))
                if len(decrease) > minmal_sample:
                    binNeLs.append(base ** (i + step/2))
                    muNeLs.append(np.mean(np.log10(decrease)))
                    stdNeLs.append(np.std(np.log10(decrease)))
        if len(binGeLs) > 0:
            oneday_result = [binGeLs[0], np.percentile(stdGeLs, 50), np.percentile(stdGeLs, 25), np.percentile(stdGeLs, 75), len(binGeLs)]
            std_ge_result.append(oneday_result)
        if len(binNeLs) > 0:
            oneday_result = [binNeLs[0], np.percentile(stdNeLs, 50), np.percentile(stdNeLs, 25), np.percentile(stdNeLs, 75), len(binNeLs)]
            std_ne_result.append(oneday_result)        
    std_ge_result = pd.DataFrame(std_ge_result, columns=['bin', 'std50', 'std25', 'std75', 'sample_cnt'])
    std_ne_result = pd.DataFrame(std_ne_result, columns=['bin', 'std50', 'std25', 'std75', 'sample_cnt'])

    return std_ge_result, std_ne_result

def growth_rate_scaling_errorbar_plot(std_ge_result, std_ne_result, label = None, c = ['k', 'gray'], 
                                      plot_ge = True, plot_ne = True, markersize = 13):
    # zoom = 2
    # plt.figure(figsize=(4*zoom, 3*zoom))
    if plot_ge == True:
        plt.errorbar(std_ge_result.bin, std_ge_result.std50, 
                    yerr=[std_ge_result.std50 - std_ge_result.std25, 
                            std_ge_result.std75 - std_ge_result.std50], 
                    fmt='o', capsize=5, markersize=markersize, color=c[0], ecolor=c[0], 
                    markeredgecolor = 'k', elinewidth=3,
                    label=r'$\log b>0$' + label if label != None else r'$\log b>0$', 
                    alpha=0.7)
    if plot_ne == True:
        plt.errorbar(std_ne_result.bin, std_ne_result.std50, 
                    yerr=[std_ne_result.std50 - std_ne_result.std25, 
                            std_ne_result.std75 - std_ne_result.std50], 
                    fmt='^', capsize=5, markersize=markersize, color=c[1], ecolor=c[1], 
                    markeredgecolor = 'k', elinewidth=3,
                    label=r'$\log b<0$' + label if label != None else r'$\log b<0$', 
                    alpha=0.7)

    # plt.legend(fontsize=18, loc = 'upper left', framealpha = 0)
    plt.legend(fontsize=20, ncol=1, bbox_to_anchor=(1.02, 0.0), loc=3, borderaxespad=0)
    plt.ylim(0, 0.7)
    plt.xlim([1, 10**4])
    plt.xscale('log')
    plt.xlabel('$x(t)$')
    plt.ylabel(r'$\sigma(\log b)$')
    plt.xticks([10**i for i in range(5)])

def prefrential_attachment(hashtagDf, c = None, label = None, fit_line = True, scatter_c = 'gray', new_figure = True, marker = 'o'):
    tmp_df = pd.DataFrame(
        {'x': hashtagDf.iloc[:, :-1].to_numpy().flatten(), 
         'y': hashtagDf.iloc[:, 1:].to_numpy().flatten()}
    )
    bis_cnt = 15
    bins = np.linspace(np.log10(tmp_df['x']).min(), np.log10(tmp_df['x']).max(), bis_cnt)
    bins = 10 ** bins
    # mask none-nan
    mask = ~np.isnan(tmp_df['x']) & ~np.isnan(tmp_df['y'])
    tmp_df = tmp_df[mask]
    # tmp_df.fillna(0, inplace = True)
    tmp_df['indeices'] = np.digitize(tmp_df.x, bins)
    # bins = np.linspace(result['x'].min(), result['x'].max(), nbins)
    # result['indeices'] = np.digitize(result.x, bins)
    error = tmp_df.groupby('indeices').agg(
            x = ('x', 'median'), 
            y25 =  ('y', lambda x: np.percentile(x, 25)),
            y50 = ('y', lambda x: np.percentile(x, 50)),
            y75 = ('y', lambda x: np.percentile(x, 75)
                    )).reset_index(drop = True)
    error = error[error['x']>0]
    # linear regression of log error['x'] and error['y50']
    from scipy import stats as st
    slope, intercept, r_value, p_value, std_err = st.linregress(np.log(error['x']), np.log(error['y50']))
    
    if new_figure: plt.figure(figsize = (8, 6))
    plt.scatter(tmp_df['x'], tmp_df['y'], alpha = 0.1, c = scatter_c, s=20, rasterized=True)
    x = np.linspace(0, tmp_df['x'].max(), 20)
    y = np.exp(intercept) * x ** slope
    if fit_line: plt.plot(x, y, '--', c = "k", linewidth = 2, alpha = 0.8, label = r'$\alpha = %1.1f$'%slope)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$x(t)$')
    plt.ylabel('$n^{new}(t+1)$')
    # userdf = pd.DataFrame()
    # userdf['usageCount'] = np.array(x).flatten()
    # userdf['userCountNew'] = np.array(y).flatten()
    # trend_follow(userdf, x = 'usageCount', y = 'userCountNew', base = 10, step=0.5, plot = 1)
    plt.xlim([0.5, 4*10**5])
    plt.ylim([0.7, 2*10**5])
    plt.errorbar(error['x'], error['y50'], yerr=[error['y50'] - error['y25'], error['y75'] - error['y50']],
                fmt=marker, capsize=5, markersize=10, ecolor="k", markeredgecolor = "k", color=c, capthick=2, linewidth=2,
                label = label if label else None, alpha = 0.8)
    plt.legend(fontsize = 20, loc = 'upper left', framealpha = 0)
    plt.gca().xaxis.set_minor_locator(LogLocator(subs='all', numticks=100))

    plt.ylim([10**(-0.3), 2e5])
    plt.xlim([10**(-0.3), 1e5])

    plt.tight_layout()

def trend_follow(userDf, x = 'usageCount', y = 'userCountNew', base = 10, step=0.5, plot = 0, scatter = False):
    x_ls = []
    y_25 = []
    y_50 = []
    y_75 = []

    for i in np.arange(0, np.log10(userDf[x].max()), step):
        tmpY = userDf[(userDf[x] >= base ** i) & (userDf[x] < base ** (i+step))][y]
        if (len(tmpY) >= 5) & (np.percentile(tmpY, 50) != 0):
            # x_ls.append(base ** (i+step/2))
            x_ls.append(base ** (i))
            y_25.append(np.percentile(tmpY, 25))
            y_50.append(np.percentile(tmpY, 50))
            y_75.append(np.percentile(tmpY, 75))
    x_ls = np.array(x_ls)
    y_25 = np.array(y_25)
    y_50 = np.array(y_50)
    y_75 = np.array(y_75)

    slope, intercept, r_value, p_value, std_err = st.linregress(np.log(x_ls[x_ls!=1]), np.log(y_50[x_ls!=1]))
    print(slope)
    print(intercept)
    print(r_value ** 2)

    if plot == 1:
        plt.figure(figsize=(8, 6))
        # mask = (userDf[x]>=5) & (userDf[y]>=5)
        if scatter == True: plt.scatter(userDf[x], userDf[y], alpha = 0.2, c = 'grey', s=10)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$x(t)$')
        plt.ylabel(r'# New user (t+1)')
        x = np.arange(1, userDf[x].max())
        y = np.exp(intercept) * x ** slope
        plt.plot(x, y, '--', c = 'k', linewidth = 3, alpha = 1, label = r'$\alpha = %1.2f$'%slope)
        plt.errorbar(x = x_ls, y = y_50, yerr=[y_50 - y_25, y_75 - y_50], 
                    # capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w', 
                    capsize=8, fmt = 'o', markersize = 10, ecolor='k', markeredgecolor = "k", color = 'w',capthick=2, linewidth=2,
                    label = '[Q1, Q3]')
        plt.legend(fontsize = 20)
        plt.xticks([10**i for i in range(0, 6)])
        plt.yticks([10**i for i in range(0, 6)])


# ----------------------------------------------------------------------------- deactivation and new users
def next_day_analysis_new(data, minvalue=0, logy=False, label=None, c='C0',
                          fit_line=True, error_line=False, min_sample_size=20,
                          return_=False, return_raw=False, marker='o'):

    def extract_hashtag(df):
        frames = []
        for col, nxt in zip(df.columns[:-1], df.columns[1:]):
            sub = df.loc[df[col].notna(), [col, nxt]]
            frames.append(pd.DataFrame({'date': col, 'name': sub.index,
                                        'current_value': sub[col].to_numpy(),
                                        'next_day': sub[nxt].to_numpy()}))
        return pd.concat(frames, ignore_index=True).sort_values('date')

    step, base = 0.2, 10
    daily = extract_hashtag(data).fillna(0)

    rows = []
    for _, day in daily.groupby('date'):
        for i in np.arange(0, 3.5, step):
            g = day[(day.current_value >= base**i) & (day.current_value < base**(i+step))]
            if len(g) > 1:
                dead = g['next_day'] <= minvalue
                rows.append((base**i, dead.mean(), len(g), set(g['name'][dead])))
    result = pd.DataFrame(rows, columns=['current_value', 'death_rate', 'sample_size', 'name'])

    errbar = result.groupby('current_value').agg(
        death_rate_q1=('death_rate', lambda x: np.quantile(x, 0.25)),
        death_rate_q2=('death_rate', lambda x: np.quantile(x, 0.50)),
        death_rate_q3=('death_rate', lambda x: np.quantile(x, 0.75)),
        samples_size=('sample_size', 'mean')).reset_index()

    slope = None
    fit_mask = ((errbar.current_value > 0) & (errbar.death_rate_q2 > 0)
                & (errbar.samples_size >= min_sample_size))
    if fit_mask.sum() > 1:
        slope, intercept, *_ = stats.linregress(
            np.log10(errbar.current_value[fit_mask]),
            np.log10(errbar.death_rate_q2[fit_mask]))
        if fit_line:
            xs = np.logspace(np.log10(errbar.current_value.min()),
                             np.log10(errbar.current_value.max()), 100)
            plt.plot(xs, 10**(intercept + slope * np.log10(xs)), '--', color="k", linewidth=2, label=r'$\beta = %.1f$' % slope)

    errbar = errbar[errbar.samples_size >= min_sample_size]
    leg = None if label is None else label
    plt.errorbar(x=errbar.current_value, y=errbar.death_rate_q2,
                 yerr=[errbar.death_rate_q2 - errbar.death_rate_q1, 
                       errbar.death_rate_q3 - errbar.death_rate_q2],

                 capsize=8, fmt= marker + "-" if error_line else marker, markersize=10,
                 ecolor=c, markeredgecolor='k', color=c,
                 capthick=2, linewidth=2, alpha=0.9, label=leg)
    if not fit_line and error_line:
        plt.plot(errbar.current_value, errbar.death_rate_q2, '--', c='k', linewidth=2, alpha=0.8)

    plt.legend(fontsize=20, frameon=False)
    plt.xscale('log')
    plt.xticks([10**i for i in range(4)])
    if logy: plt.yscale('log')
    else: plt.yticks(np.arange(0, 1+0.1, 0.1))
    plt.xlabel(r'$x(t)$')
    plt.ylabel(r'$P(x(t+1)=0 | x(t))$')

    if return_: return errbar
    if return_raw: return result

def next_day_analysis_new_new(data, minvalue = 0, logy = False, label = None, c = 'C0', return_ = False, fit_line = True, min_sample_size = 20):
    def extract_hashtag(df):
        result = pd.DataFrame()
        df = df[df.iloc[:, 0].isna()] # keep only hashtags that appear for the first time
        for i in range(3, data.shape[1]-2):
            mask = (df.iloc[:, i].notna())
            # display(hashtag_df[mask].iloc[:, :(i+2)])
            cumulated_value = df[mask].iloc[:, :(i+1)].sum(axis = 1)
            current_value = df[mask].iloc[:, i+1]
            oneday_result =  pd.concat([cumulated_value, current_value], axis=1).rename(
                                        columns={0: 'current_value', df[mask].iloc[:, i+1].name: 'next_day'})
            oneday_result.reset_index(inplace=True)
            oneday_result.rename(columns={'hashtags': 'name'}, inplace=True)
            oneday_result['date'] = df.columns[i]
            result = pd.concat([result, oneday_result], ignore_index=True)
        return result
    old_hashtag_df = extract_hashtag(data)
    old_hashtag_df.fillna(0, inplace=True)
    
    step = 0.2
    base = 10
    result_current_value = []
    result_death_rate = []
    sample_size = []
    for date in old_hashtag_df.date.unique():
        result = old_hashtag_df[old_hashtag_df.date == date]
        birth_value_ls = []
        nan_ratio_ls = []
        sample_size_ls = []
        for i in np.arange(0, 3.5, step):
            tmp = result[(result.current_value >= base ** i) & (result.current_value < base ** (i+step))]['next_day']
            if len(tmp) > 1:
                # birth_value_ls.append(base**(i+step/2))
                birth_value_ls.append(base**(i))
                nan_ratio_ls.append(len(tmp[tmp<=minvalue])/(len(tmp)))
                sample_size_ls.append(len(tmp))

        # plt.plot(birth_value_ls, nan_ratio_ls, '--o', c = c, alpha = 0.8,  
        #         label = label if date == old_hashtag_df.date.max() else None )
        
        result_current_value.extend(birth_value_ls)
        result_death_rate.extend(nan_ratio_ls)
        sample_size.extend(sample_size_ls)

    
    # plot error bar
    result = pd.DataFrame([result_current_value, result_death_rate, sample_size]).T.rename(columns={0: 'current_value', 1: 'death_rate', 2: 'sample_size'})
    result_errbar = result.groupby(['current_value']).agg(
                    death_rate_q1 = ('death_rate', lambda x: np.quantile(x, 0.25)),
                    death_rate_q2 = ('death_rate', lambda x: np.quantile(x, 0.5)),
                    death_rate_q3 = ('death_rate', lambda x: np.quantile(x, 0.75)),
                    samples_size = ('sample_size', 'mean'),
                                ).reset_index()
    # plt.plot(result_errbar.current_value, result_errbar.death_rate_q2, '--', c = c, linewidth = 2, alpha = 0.8)

    valid_mask = (result_errbar.current_value > 0) & (result_errbar.death_rate_q2 > 0) & (result_errbar.samples_size >= min_sample_size)
    if valid_mask.sum() > 1:  # Need at least 2 points for regression
        drop_bottom = 5
        drop_top = 0
        log_x = np.log10(result_errbar.current_value[valid_mask][drop_top:-drop_bottom])
        log_y = np.log10(result_errbar.death_rate_q2[valid_mask][drop_top:-drop_bottom])
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

        x_range = np.logspace(np.log10(result_errbar.current_value.min()), 
                              np.log10(result_errbar.current_value.max()), 100)
        y_pred = 10**(intercept + slope * np.log10(x_range))
        
        if fit_line == True: plt.plot(x_range, y_pred, '--', color=c, linewidth=3)     

    valid_mask = result_errbar.samples_size >= min_sample_size
    result_errbar = result_errbar[valid_mask]
    if fit_line == True: plt.errorbar(x = result_errbar.current_value, 
                        y = result_errbar.death_rate_q2, 
                        yerr=[result_errbar.death_rate_q2 - result_errbar.death_rate_q1, 
                            result_errbar.death_rate_q3 - result_errbar.death_rate_q2],
                        capsize=8, fmt = 'o', markersize = 10, ecolor=c, markeredgecolor = c, color = 'w',
                        capthick=2, linewidth=2,
                        label = label + r' ($\beta = %.1f$)' % (slope) if label != None else None, alpha = 0.9
                        )
    else: plt.errorbar(x = result_errbar.current_value, 
                        y = result_errbar.death_rate_q2, 
                        yerr=[result_errbar.death_rate_q2 - result_errbar.death_rate_q1, 
                             result_errbar.death_rate_q3 - result_errbar.death_rate_q2],
                        capsize=8, fmt = 'o', markersize = 10, ecolor=c, markeredgecolor = c, color = 'w',
                        capthick=2, linewidth=2,
                        label = label if label != None else None, alpha = 0.9
                        )
    
    plt.legend(fontsize = 20)
    plt.xscale('log')
    plt.xticks([10 ** i for i in range(4)])
    if logy == True: plt.yscale('log')
    else: plt.yticks(np.arange(0, 1+0.1, 0.1))
    plt.xlabel(r'$\sum_{t^{\prime} \leq t} x(t^{\prime})$', fontsize = 25)
    plt.ylabel(r'$P(x(t+1)=0 | \sum_{t^{\prime} \leq t} x(t^{\prime}))$', fontsize = 25)

    if return_ == True:
        # return np.array(result_current_value).flatten(), np.array(result_death_rate).flatten()
        return result_errbar


# ----------------------------------------------------------------------------- distributions
def plot_cdf(data, logx=1, logy=1, label='0', c='0', alpha=1, markersize=2.5, linewidth=1.2, linestyle='o--', left=0, sort=1, del0=True):
    data = np.array(data)
    data = data[~np.isnan(data)]
    if del0 == True:
        data = data[data != 0]
        if sort == 1:
            data = np.sort(data)
        if left == 1:
            cdf = (np.arange(len(data))+1)/len(data)
        else:
            cdf = 1 - np.arange(len(data))/len(data)
        plt.xscale('log') if (logx == 1) else 0
        plt.yscale('log') if (logy == 1) else 0
        plt.grid(linestyle='-.')
        if ((label != '0') & (c != '0')):
            plt.plot(data, cdf, linestyle, markersize=markersize,
                     linewidth=linewidth, label=label, c=c, alpha=alpha, 
                     rasterized=True)
        elif ((label != '0') & (c == '0')):
            plt.plot(data, cdf, linestyle, markersize=markersize,
                     linewidth=linewidth, label=label, alpha=alpha, 
                     rasterized=True)
        elif ((label == '0') & (c != '0')):
            plt.plot(data, cdf, linestyle, markersize=markersize,
                     linewidth=linewidth, c=c, alpha=alpha, rasterized=True)
        else:
            plt.plot(data, cdf, linestyle, markersize=markersize,
                     linewidth=linewidth, alpha=alpha, rasterized=True)
    else:
        if sort == 1:
            data = np.sort(data)
        if left == 1:
            cdf = (np.arange(len(data))+1)/len(data)
        else:
            cdf = 1 - np.arange(len(data))/len(data)
        plt.xscale('log') if (logx == 1) else 0
        plt.yscale('log') if (logy == 1) else 0
        plt.grid(linestyle='-.')
        if ((label != '0') & (c != '0')):
            plt.plot(data, cdf, linestyle, markersize=markersize,
                     linewidth=linewidth, label=label, c=c, alpha=alpha, 
                     rasterized=True)
        elif ((label != '0') & (c == '0')):
            plt.plot(data, cdf, linestyle, markersize=markersize,
                     linewidth=linewidth, label=label, alpha=alpha,
                     rasterized=True)
        elif ((label == '0') & (c != '0')):
            plt.plot(data, cdf, linestyle, markersize=markersize,
                     linewidth=linewidth, c=c, alpha=alpha,
                     rasterized=True)
        else:
            plt.plot(data, cdf, linestyle, markersize=markersize,
                     linewidth=linewidth, alpha=alpha, 
                     rasterized=True)

def find_longest_continuous_subseq(series):
    """Longest run of consecutive non-NaN values."""
    mask = ~pd.isna(series)
    if not mask.any():
        return []
    
    # longest run of True
    changes = mask.diff().fillna(mask.iloc[0])
    groups = changes.cumsum()
    
    max_len = 0
    best_subseq = []
    
    for group_id in groups[mask].unique():
        subseq = series[groups == group_id].dropna()
        if len(subseq) > max_len:
            max_len = len(subseq)
            best_subseq = subseq.values
    
    return best_subseq


# ----------------------------------------------------------------------------- autocorrelation (Fig. 7b)
def batch_autocorr(df, min_length=3, max_lags=5):
    """Autocorrelation of every row up to max_lags, computed on its longest non-NaN run."""
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating autocorrelation"):
        # longest contiguous run
        subseq = find_longest_continuous_subseq(row)
        
        if len(subseq) >= min_length:
            # largest lag the run supports
            max_possible_lags = min(max_lags, len(subseq) - 1)
            acf_result = acf(subseq, nlags=max_possible_lags, fft=False)
            
            # lags 1..max_lags (lag 0 is always 1)
            row_result = [np.nan] * max_lags
            for i in range(1, len(acf_result)):
                if i <= max_lags:
                    row_result[i-1] = acf_result[i]
        else:
            # run too short: all NaN
            row_result = [np.nan] * max_lags
        
        results.append(row_result)
    
    # assemble
    columns = [f'lag_{i+1}' for i in range(max_lags)]
    return pd.DataFrame(results, index=df.index, columns=columns)


# ----------------------------------------------------------------------------- effective contagiousness (Fig. 5)
def visulize_diffusion_series_paper(hashtag_df_hour_lcc, tag_likely_df_hour, diffu_df, 
                              name, hashtag_df_noise_lcc = None):
    # observed count, expected count and effective contagiousness of one hashtag (Fig. 5a)
    # 1. time series
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(hashtag_df_hour_lcc.loc[name, :], '-', label = 'real')
    ax1.plot(tag_likely_df_hour.loc[name, :], '-', label = f'model 0')
    if hashtag_df_noise_lcc is not None: ax1.plot(hashtag_df_noise_lcc.loc[name, :], '-', c = 'k', label = f'noise', alpha=0.7)
    ax1.set_xticks([hashtag_df_hour_lcc.columns[i+24*3] for i in range(0, 24*6*7, 24*6)], [f'3/{i}' for i in range(11, 18)])
    ax1.set_ylabel(r'$x_k(r)$', fontsize = 40)
    # ax1.set_title(f'{name}')s
    ax1.legend(fontsize =20, framealpha = 0)
    ax1.set_xticks([], [])
    for i in range(24*6, 24*7*6, 24*6): ax1.axvline(x=hashtag_df_hour_lcc.columns[i], color='gray', linestyle='--', alpha=0.5)
    for i in range(24 * 6, 24*7*6 - 1, 24*6 * 2): ax1.axvspan(hashtag_df_hour_lcc.columns[i], hashtag_df_hour_lcc.columns[i + 24*6], color='gray', alpha=0.1) 
    width = pd.Timedelta(hours=3)
    ax1.set_xlim(hashtag_df_hour_lcc.columns[0]-width, hashtag_df_hour_lcc.columns[-1]+width)
    ax2 = plt.subplot(2, 1, 2)

    if hashtag_df_noise_lcc is not None:  label = r'$d_k(t)=$(real-noise)/model 0'
    else: label = r'$d_k(r)=$real/model 0'

    ax2.plot(diffu_df.loc[name, :], '-', color='k', label = label, alpha = 0.5)
    ax2.set_ylabel(r'$d_k(r)$', fontsize = 40)
    # ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.legend(fontsize = 20, framealpha = 0)
    ax2.set_xticks([hashtag_df_hour_lcc.columns[i+24*3] for i in range(0, 24*6*7, 24*6)], [f'3/{i}' for i in range(11, 18)])
    for i in range(24 * 6, 24*7*6, 24*6): ax2.axvline(x=hashtag_df_hour_lcc.columns[i], color='gray', linestyle='--', alpha=0.5)
    for i in range(24 * 6, 24*7*6 - 1, 24*6 * 2): ax2.axvspan(hashtag_df_hour_lcc.columns[i], hashtag_df_hour_lcc.columns[i + 24*6], color='gray', alpha=0.1) 
    ax2.set_xlim(hashtag_df_hour_lcc.columns[0]-width, hashtag_df_hour_lcc.columns[-1]+width)
    ax2.set_xlabel('Time', fontsize = 40)
    plt.tight_layout()

def potention_analysis_paper(diffu_df, center = 0, top_k = 200, nbins = 20, min_sample = 3, 
                            onesample = False, tag_name = None, 
                            plot_scatter = True, regression = True, show_mu = True, 
                            return_ = False
                            ):
    x_result = []
    y_result = []
    if onesample:
        loc = np.where(diffu_df.index == tag_name)[0][0]
        tag_ls = [loc]
    else:
        tag_ls = trange(top_k)
    plt.figure(figsize=(8, 6))
    for i in tag_ls:
        x_ = np.array(diffu_df.iloc[i, :-1]) - center
        y_ = np.array(diffu_df.iloc[i, 1:]) - np.array(diffu_df.iloc[i, :-1])
        mask = (x_ != np.inf) & (y_ != np.inf) & (~np.isnan(x_)) & (~np.isnan(y_)) # mask infinate value in x_, y_
        if plot_scatter: plt.scatter(x_[mask], y_[mask], s = 25, 
                                     rasterized=True, color = 'grey', alpha = 0.5)
        x_result.extend(x_[mask])
        y_result.extend(y_[mask])
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.ylabel(r'$d_k(r+1) - d_k(r)$', fontsize=40)
    plt.xlabel(r'$d_k(r)$', fontsize=40)

    
    result = pd.DataFrame([x_result, y_result], index=['x', 'y']).T.sort_values(by='x').reset_index(drop=True)
    result.dropna(inplace=True, how='any')
    bins = np.linspace(result['x'].min(), result['x'].max(), nbins)
    result['indeices'] = np.digitize(result.x, bins)
    result_bined = result.groupby('indeices').agg(
                    x_q2 = ('x', lambda x: np.percentile(x, 50)),
                    y_q2 = ('y', lambda x: np.percentile(x, 50)),
                    y_q1 = ('y', lambda x: np.percentile(x, 25)),
                    y_q3 = ('y', lambda x: np.percentile(x, 75)),
                    sample_size = ('y', 'count')).reset_index()
    result_bined = result_bined[result_bined['sample_size'] >= min_sample]
    result = result[result['x']<=result_bined['x_q2'].max()]

    bins = np.linspace(result['x'].min(), result['x'].max(), nbins)
    result['indeices'] = np.digitize(result.x, bins)
    try: result['interval'] = pd.cut(result['x'], bins=bins, include_lowest=True)
    except: result['interval'] = np.nan
    else: pass
    result_bined = result.groupby('indeices').agg(
                    x_q2 = ('x', lambda x: np.percentile(x, 50)),
                    y_q2 = ('y', lambda x: np.percentile(x, 50)),
                    y_q1 = ('y', lambda x: np.percentile(x, 25)),
                    y_q3 = ('y', lambda x: np.percentile(x, 75)),
                    sample_size = ('y', 'count')).reset_index()
    result_bined = result_bined[result_bined['sample_size'] >= min_sample]
    # plt.ylim([-1.8, 1.5])
    # plt.xlim([-0.5, 5])

    plt.errorbar(result_bined['x_q2'], result_bined['y_q2'],
                 yerr=[result_bined['y_q2'] - result_bined['y_q1'], result_bined['y_q3'] - result_bined['y_q2']],
                 capsize=5, fmt = 'o', markersize = 8, ecolor='k', 
                 markeredgecolor = 'k', color = 'w', 
                 label = r'[Q1, Q3]')
    if regression:
        try: 
            slope, intercept, k1, k2, k3= st.linregress(result_bined['x_q2'], result_bined['y_q2'])
            x_intersect = -intercept / slope if slope != 0 else np.nan
            y_pred = slope * result['x'] + intercept
            residuals = result['y'] - y_pred
            result['residuals'] = residuals
        except: slope, intercept, x_intersect = np.nan, np.nan, np.nan
        else:
            x = np.linspace(0, result_bined['x_q2'].max() * 1.2, 20)
            if show_mu: plt.axvline(x_intersect, color='r', linestyle='--', label = r'$\mu = %1.1f$'%x_intersect)
            plt.plot(x, intercept + slope * x, 'k--', label = rf'slope$ = {slope:.3f}$', linewidth=2, 
                     rasterized=True)
            plt.legend(fontsize=20, framealpha=0)

    plt.tight_layout()

    if return_: return result_bined, result, slope, intercept, x_intersect

def plot_potential(result, result_bined):
    # zoom = 0.8
    plt.figure(figsize=(8, 6))
    try: result['integral'] = cumulative_trapezoid(result['y'], -result['x'], initial=0)
    except: result['integral'] = np.nan
    else:  plt.plot(result['x'], result['integral'], '-o', markersize=3, label = 'raw data', 
                    rasterized=True)
    
    try: result_bined['integral'] = cumulative_trapezoid(result_bined['y_q2'], -result_bined['x_q2'], initial=0)
    except: result_bined['integral'] = np.nan
    else:  plt.plot(result_bined['x_q2'], result_bined['integral'], '-o', markersize=3, label = 'binned data', 
                    rasterized=True)
    
    plt.legend(fontsize=20, loc='upper left', frameon=False)
    plt.xlabel(r'$d_k(r)$', fontsize=40)
    plt.ylabel(r'$U$', fontsize=40)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
