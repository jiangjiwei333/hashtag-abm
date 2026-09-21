"""random_multiplicative_process.py -- macroscopic random multiplicative process (Results, "Connection from
Micro to Macro Model"; Fig. 7a).

    x_k(t+1) = b_k(t | x_k(t)) x_k(t) + f_k(t)

random_simulation(): b is drawn from the size-conditional growth-rate pool of the input daily-count table
(logarithmic size bins [10^{j/2}, 10^{(j+1)/2})), f is lognormal (mu, std) injection of new hashtags.
rm_model_solution(): theoretical power-law exponents per size bin.
Taken verbatim from the original analysis code (unused imports removed).
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import trange


def random_simulation(hashtag_df_real, days=29, size_dependence=1, base = 4, max_exponent=7, step = 1, 
                      lambda_ = 15, noise = 'poisson', del0 = True, mu = 0, std = 1.5, ifprint = True):
    """
    days: number of days that simulate
    size_dependence: simulate with size dependence?
    max_exponent: if there is size dependence, how to decompose the distribution
    """
    initial_vale = np.random.uniform(1, 10, size = hashtag_df_real.iloc[:, 0].dropna().shape[0])
    hashtag_df_real = hashtag_df_real.fillna(0).copy() # !!!!!!!!!!!!!!
    hashtag_simulated_result = [] # simulation result
    growth_df = hashtag_df_real.shift(-1, axis = 1)/hashtag_df_real
    if size_dependence == 1: # simulate with size dependence
        growth_rate_list = []
        for i in np.arange(0, max_exponent, step):
            if i < (max_exponent - step):
                if ifprint == True:
                    print(f'[{base}^{i}, {base}^{i+step})', end=': ')
                growth_rate = growth_df[(hashtag_df_real >= base ** i) & (hashtag_df_real < base ** (i+step))]
            else:
                if ifprint == True:
                    print(f'[{base}^{i}, infty)', end=': ')
                growth_rate = growth_df[hashtag_df_real >= base ** i]
            growth_rate = np.array(growth_rate).flatten()
            growth_rate = growth_rate[~np.isnan(growth_rate)]
            growth_rate = growth_rate[~np.isinf(growth_rate)]
            if del0 == True: growth_rate = growth_rate[growth_rate!=0]
            if ifprint == True:
                print(f'Data length = {len(growth_rate)}', end=', ')
                print(f'Mean(logb) = {np.log10(growth_rate[growth_rate!=0]).mean()}')
            if len(growth_rate) == 0: print(f'[{base**i}, {base**(i+1)}) no data!')
            growth_rate_list.append(growth_rate) # [nparray, nparray, ...]

        next = initial_vale
        for day in trange(0, days):
            hashtag_simulate = np.copy(next)
            if day >= days - 20000: hashtag_simulated_result.append(hashtag_simulate)
            cnt = 0
            for i in np.arange(0, max_exponent, step):
                if i < (max_exponent - step): # select distribution
                    index_ = np.where((hashtag_simulate >= base ** i) & (hashtag_simulate < base ** (i+1)))
                else: 
                    index_ = np.where(hashtag_simulate >= base ** i)
                # print(index_)
                growth_rate_sample = np.random.choice(growth_rate_list[cnt], len(index_[0]))
                next[index_] = growth_rate_sample * hashtag_simulate[index_]
                cnt += 1
            next = next.astype(int)
            if noise == 'poisson':
                next[np.where(next < 1)] += np.random.poisson(lam = lambda_, size = len(np.where(next < 1)[0])).astype(int)
            if noise == 'lognormal':
                next[np.where(next < 1)] += np.random.lognormal(mu, std, size = len(np.where(next < 1)[0])).astype(int)
            else: 
                next[np.where(next < 1)] += 1
            next = next.astype(int)
            next[np.where(next < 1)] = 1
    else:
        # growth_rate = np.array(growth_df).flatten() # simulate without size dependence
        # growth_rate = growth_rate[~np.isnan(growth_rate)]
        # growth_rate = growth_rate[~np.isinf(growth_rate)]
        # growth_rate_list = growth_rate
        # hashtag_simulate = initial_vale
        # for day in trange(0, days):
        #     if day >= 20000: hashtag_simulated_result.append(hashtag_simulate)
        #     growth_rate_sample = np.random.choice(growth_rate_list, len(hashtag_simulate))
        #     hashtag_simulate = growth_rate_sample * hashtag_simulate
        #     hashtag_simulate = hashtag_simulate.astype(int)
        #     hashtag_simulate[np.where(hashtag_simulate <= 1)] = 1
        pass
    return np.array(hashtag_simulated_result), growth_rate_list


def rm_model_solution(simulated_result, base = 10, max_exponent=5, step = 0.5, plot_curve = 0, compute = 0, del0 = True):
    """
    simulated_result: rows = time, columns = hashtags (must not contain NaN)
    """
    
    simulated_result_df = pd.DataFrame(simulated_result).T
    bt_df = simulated_result_df.shift(-1, axis=1)/simulated_result_df   # b(t) = x(t+1)/x(t)

    alpha_list = []
    alpha_list_simple = []
    cnt = 0
    zero_ratio = []
    for i in np.arange(0, max_exponent, step):
        
        if i != (max_exponent - step):
            grouped_bt = bt_df[(simulated_result_df >= base ** i) & 
                               (simulated_result_df < base ** (i+1))]
            label = r'[' + r'$%d^{%0.1f}$'%(base, i) + ',' + r'$%d^{%0.1f}$'%(base, i+step) + r')'
        else:
            grouped_bt = bt_df[(simulated_result_df >= base ** i)]
            label = r'[' + r'$%d^{%0.1f}$'%(base, i) + ',' + r'$\infty$' + r')'

        print(f'{label}', end=', ')
        if compute == 1:
            alpha_list.append(compute_alpha(grouped_bt, compute_all = 0))
        if plot_curve == 1: # Visualization
            grouped_bt_list = np.array(grouped_bt).flatten()
            grouped_bt_list = grouped_bt_list[~np.isnan(grouped_bt_list)]
            grouped_bt_list = grouped_bt_list[~np.isinf(grouped_bt_list)]
            if del0 == True: grouped_bt_list = grouped_bt_list[grouped_bt_list!=0]
            # grouped_bt_list = grouped_bt_list[~np.isinf(grouped_bt_list)]

            max_alpha = 3
            alpha_list_simple.append(steady_state_conditions_test(grouped_bt_list, label = label, max_alpha = max_alpha, 
                                                                  c=f'C{cnt}', marker='x'))
            # plt.legend(fontsize='18', loc = 'upper right')
            plt.legend(fontsize = 20, ncol = 1, bbox_to_anchor = (1.02, 0.0), loc=3, borderaxespad=0)
            plt.yscale('log')
            zero_ratio.append(len(grouped_bt_list[grouped_bt_list==0])/len(grouped_bt_list))
        cnt += 1
    if compute == 1:
        return pd.DataFrame(alpha_list)
    else:
        return pd.DataFrame(alpha_list_simple), zero_ratio


def steady_state_conditions_test(series, label = '0', max_alpha=1.5, c = 'k', marker = '.'):
    # serise = np.array(series)

    expectation = []
    for alpha in np.arange(0, max_alpha, 0.01):
        expectation.append(np.mean(series ** alpha))
        
    moment = np.abs(np.array(expectation[1:]) - 1)

    marker_interval = 6
    if label != '0':
        plt.plot(np.arange(0, max_alpha, 0.01), expectation, c = c)
        plt.scatter(np.arange(0, max_alpha, 0.01)[::marker_interval], expectation[::marker_interval], c = c, marker = marker, s = 70)
        plt.plot([], [], c = c, marker = marker, label = label, markersize = 9.2)
    else:
        plt.plot(np.arange(0, max_alpha, 0.01), expectation, c = c)
        plt.scatter(np.arange(0, max_alpha, 0.01)[::marker_interval], expectation[::marker_interval], c = c, marker = marker, s = 70)
    plt.plot(np.arange(0, max_alpha, 0.01), np.ones(len(np.arange(0, max_alpha, 0.01))), c = 'k')
    plt.ylim((0.7, 1.5))
    plt.yticks()
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\langle$ $b(t|x(t))^{\alpha}$ $\rangle$')
    
    # plt.ylim(0.9995, 1.0005)

    if moment.argmin() == 0:
        return 0
    else:
        return np.arange(0.01, max_alpha, 0.01)[moment.argmin()]


def compute_alpha(bt_df, sample_size = 100, compute_all = 0):
    """
    bt_df is an variable in function 'model_systeam_test'
    """
    alpha_list = []
    if compute_all == 1:
        bt_list = np.array(bt_df).flatten()
        bt_list = bt_list[~np.isnan(bt_list)]
        print(len(bt_list))
        if np.mean(np.log10(bt_list)) > 0:
            alpha_list.append(0)
        else:
            moment_list = []
            for alpha in np.arange(0.01, 3.50, 0.01):
                moment_list.append(np.abs(np.mean(bt_list ** alpha) - 1))
            moment_list = np.array(moment_list)
            alpha_list.append(np.arange(0.01, 3.50, 0.01)[moment_list.argmin()])
        return alpha_list
    for i in trange(0, 5000, sample_size):
        bt_list = np.array(bt_df.iloc[i:(i+sample_size), :]).flatten() # get b(t) list
        bt_list = bt_list[~np.isnan(bt_list)]
        if np.mean(np.log10(bt_list)) > 0: # stationarity condition
            alpha_list.append(0)
            continue
        else:
            moment_list = []
            for alpha in np.arange(0.01, 3.50, 0.01):
                moment_list.append(np.abs(np.mean(bt_list ** alpha) - 1))
            moment_list = np.array(moment_list)
            alpha_list.append(np.arange(0.01, 3.50, 0.01)[moment_list.argmin()])
    return alpha_list
