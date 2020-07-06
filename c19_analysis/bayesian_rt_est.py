# Extension of https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb to county-level, w/ the most salient
# change being as follows:
# Rather than using a prior of gamma-distributed generation intervals to estimate R (which seems totally reasonable),
# I'm experimenting with incorporating more locally-relevant information by calculating an R0 using initial incidence
# data from each locality.
from typing import Union, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps
from tqdm.notebook import tqdm
import datetime

import c19_analysis.cust_seir_model as cust_seir
import config


def highest_density_interval(pmf: Union[pd.DataFrame, pd.Series], p: float = 0.9) -> Union[pd.Series, pd.DataFrame]:
    # If we pass a DataFrame, just call this recursively on the columns
    if isinstance(pmf, pd.DataFrame):
        intervals = []
        for col in pmf:
            try:
                intervals.append(highest_density_interval(pmf[col], p=p))
            except ValueError:
                print(f"current date is {col} series is {pmf[col]}")
        return pd.DataFrame(intervals, index=pmf.columns)
    cumsum = np.cumsum(pmf.values)
    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]
    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()
    if not (len(highs) > 0 and len(lows) > 0):
        # usually due to data collection error (but also in other edge cases), no 90% credible interval within our range
        # so use the min and max r_t range
        return pd.Series([pmf.index[0], pmf.index[-1]], index=[f'{p * 100:.0f}_CrI_LB', f'{p * 100:.0f}_CrI_UB'])
    else:
        # Find the smallest range (highest density)
        best = (highs - lows).argmin()
        low = pmf.index[lows[best]]
        high = pmf.index[highs[best]]
        return pd.Series([low, high], index=[f'{p * 100:.0f}_CrI_LB', f'{p * 100:.0f}_CrI_UB'])


# This approach using equal-tailed credible interval instead of an HDI is about twice as fast
# which may be handy for daily county-level refreshes. If we were dealing with
# just Gaussian distributions (unimodal and symmetric), the equal-tailed credible interval [would overlap perfectly]
# (https://www.sciencedirect.com/topics/mathematics/credible-interval) with the highest density interval which would
# allow us to use ppf (inverse of cdf) to find the lower an upper bounds of our credible interval. Since we're using
# poisson pmfs, the equal-tailed credible interval diverges from the HDIs. Based on my experimentation though, it
# varies by < 5% in the vast majority of cases so the performance improvement may be worth it for some uses.
# Leaving this function in the notebook in case it's of future utility from a performance perspective
def equal_tail_interval(pmf: Union[pd.DataFrame, pd.Series], p: float = 0.9) -> Union[pd.DataFrame, pd.Series]:
    # If we pass a DataFrame, just call this recursively on the columns
    if isinstance(pmf, pd.DataFrame):
        intervals = []
        for col in pmf:
            try:
                intervals.append(equal_tail_interval(pmf[col], p=p))
            except ValueError:
                print(f"current date is {col} series is {pmf[col]}")
        return pd.DataFrame(intervals, index=pmf.columns)
        # define custom discrete distribution
    custom_dist = sps.rv_discrete(name='custm', values=(pmf.index.to_numpy(), pmf.to_numpy()))
    lb, ub = custom_dist.ppf((1 - p) / 2), custom_dist.ppf(1 - (1 - p) / 2)
    return pd.Series([lb, ub], index=[f'{p * 100:.0f}_CrI_LB', f'{p * 100:.0f}_CrI_UB'])


def get_posteriors(sr: pd.Series, gm_sigma: float, r_prior: float,
                   r_t_range: np.ndarray) -> Tuple[pd.DataFrame, float]:
    # Calculate $\lambda$ - the expected arrival rate for every day's poisson process
    lam = sr[:-1].values * np.exp(cust_seir.gamma * (r_t_range[:, None] - 1))
    # Calculate each day's likelihood distribution over all possible values of $R_t$
    likelihoods = pd.DataFrame(data=sps.poisson.pmf(sr[1:].values, lam), index=r_t_range, columns=sr.index[1:])
    # Calculate the Gaussian process matrix based on the value of $\sigma$
    process_matrix = sps.norm(loc=r_t_range, scale=gm_sigma).pdf(r_t_range[:, None])
    # Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
    # rather than using a prior of gamma-distributed generation intervals to estimate R, factor in more local variables
    # by using R0 calculated from initial incidence data
    prior0 = sps.norm.pdf(x=r_t_range, loc=r_prior, scale=3)
    prior0 /= prior0.sum()
    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(index=r_t_range, columns=sr.index, data={sr.index[0]: prior0})
    # We said we'd keep track of the sum of the log of the probability of the data for maximum likelihood calculation.
    log_likelihood = 0.0
    # Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):
        # Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]
        # Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
        # Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator) + config.epsilon
        # Execute full Bayes' Rule
        posteriors[current_day] = numerator / denominator
        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)
    return posteriors, log_likelihood


def build_rtdf(tmp_df: pd.DataFrame, rt_range: np.ndarray, test_mode: bool = False) -> pd.DataFrame:
    rt_df_tmps = []
    counties = tmp_df['name'].unique().tolist() if not test_mode else config.test_counties
    for c in tqdm(counties):
        rt_df_tmp = tmp_df.loc[(tmp_df['name'] == c)]
        if not rt_df_tmp.empty:
            r0_est = round(cust_seir.r_calc(rt_df_tmp['Total Estimated Cases'].iloc[0],
                                            rt_df_tmp['node_days'].iloc[0]), 2)
            posteriors, log_likelihood = get_posteriors(rt_df_tmp['daily new cases ma'], gm_sigma=.15, r_prior=r0_est,
                                                        r_t_range=rt_range)
            try:
                # see note above regarding ETI approach
                #etis = equal_tail_interval(posteriors, p=.9)
                hdis = highest_density_interval(posteriors, p=.9)
                most_likely = posteriors.idxmax().rename('Rt')
                #rt_df_tmps.append(pd.concat([rt_df_tmp, most_likely, etis], axis=1))
                rt_df_tmps.append(pd.concat([rt_df_tmp, most_likely, hdis], axis=1))
            except ValueError:
                print(f"Encountered Rt calculation error. Current county is {c} ")
    return pd.concat(rt_df_tmps, axis=0)


def gen_rt_df(covid_delta_df: pd.DataFrame, case_density: float = config.case_density) -> pd.DataFrame:
    # keep only counties with confirmed case density of > case_density per million people
    density = covid_delta_df['Total Estimated Cases'] / covid_delta_df.index.get_level_values('estimated_pop')
    density_condition = density * 1000000 > case_density
    covid_delta_df_tmp = covid_delta_df[(density_condition & (pd.notna(covid_delta_df['daily new cases ma'])))]
    covid_delta_df_tmp = covid_delta_df_tmp.reset_index()
    covid_delta_df_tmp = covid_delta_df_tmp.set_index(['Date'])
    if not config.county_rt_calc_zip.exists():
        rt_df = build_rtdf(covid_delta_df_tmp, config.r_t_range, test_mode=False)
        rt_df.to_csv(config.county_rt_calc_zip, compression='gzip')
    else:
        rt_df = pd.read_csv(config.county_rt_calc_zip, compression='gzip',
                            index_col=['id', 'estimated_pop', 'name', 'stateAbbr', 'Date'], parse_dates=True,
                            converters={'node_start_dt': pd.to_datetime, 'node_days': pd.to_timedelta})
        print('No update to core case data, loaded Rt estimates from cache')
    # reconfig index for downstream analysis
    rt_df = rt_df.reset_index()
    rt_df = rt_df.set_index(['id', 'estimated_pop', 'name', 'stateAbbr', 'Date'])
    rt_df = rt_df.sort_values(by=['id', 'Date'])
    rt_df['2nd_order_growth'] = rt_df['2nd_order_growth'].apply(lambda x: round(x * 100, 2))
    rt_df['naive_R0'] = rt_df.apply(lambda x: round(cust_seir.r_calc(x['Total Estimated Cases'], x['node_days']), 2)
                                    if x['daily new cases ma'] > 0 else None, axis=1)
    return rt_df
