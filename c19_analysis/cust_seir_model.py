from typing import Tuple, Optional, Union, Iterable, List
import math

import datetime
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import config

register_matplotlib_converters()
# mean latent period
time_l = 3  # https://www.ijidonline.com/article/S1201-9712(20)30117-X/fulltext,
# https://www.sciencedirect.com/science/article/pii/S0140673620302609
# infectious period
time_inf = 2.3  # https://www.nejm.org/doi/full/10.1056/NEJMoa2001316
# mean time to hospitalization
hospital_delay = 8  # https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30183-5/fulltext
# https://jamanetwork.com/journals/jama/fullarticle/2761044?guestAccessKey=f61bd430-07d8-4b86-a749-bec05bfffb65
hosp_period = 12
# https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/ \
# Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf (weighted by us demo by https://covidactnow.org/model)
hosp_rate = 0.073


# calculate R0
# N.B. that this approach only is appropriate for the initial exponential phase,
# subsequent R_t estimates are better done using the aforementioned Bayesian approach
def r_calc(confirmed_cases: int, d: pd.Timedelta) -> float:
    kappa = math.log(confirmed_cases) / d.days
    r = 1 + kappa * (time_l + time_inf) + pow(kappa, 2) * time_l * time_inf
    return r


# define system of ODEs that govern the classic compartmental SEIR model
# https://www.nature.com/articles/s41421-020-0148-0, https://idmod.org/docs/malaria/model-seir.html
# I've added a couple of simple equations governing quarantine and hospital recovery sub-compartments of recovery
def seir(initvals: Tuple, _, r: float, n: int) -> np.ndarray:
    beta = betafunc(r, time_inf)
    outputs = np.zeros(6)
    inputs = initvals
    outputs[0] = -(beta * inputs[0] * inputs[2]) / n  # dS/dt
    outputs[1] = ((beta * inputs[0] * inputs[2]) / n) - (sigma * inputs[1])  # dE/dt
    outputs[2] = (sigma * inputs[1]) - gamma * inputs[2]  # dI/dt
    outputs[3] = hosp_rate * gamma * inputs[2] - (1 / hospital_delay) * inputs[3]  # dpreHQ/dt - pre-hospital quarantine
    outputs[4] = (1 - hosp_rate) * gamma * inputs[2] + (1 / hosp_period) * inputs[5]  # dR/dt
    outputs[5] = (1 / hospital_delay) * inputs[3] - (1 / hosp_period) * inputs[5]  # dHospitalRecovery/dt
    return outputs


# sigma is the incubation rate calculated by the inverse of the mean latent or incubation period
# (latent period is actually different from incubation period often, allowing for pre-symptomatic transmission)
def sigmafunc(tl: Optional[float]) -> float:
    return 1 / tl


# beta is the infectious rate (R0*gamma)
def betafunc(curr_r: Optional[float], ti: Optional[float]) -> float:
    return curr_r / ti


# gamma is recovery rate, the inverse of the infectious period 1/time_inf
def gammafunc(ti: Optional[float]) -> float:
    return 1 / ti


# set aliases for gamma and sigma funcs
gamma = gammafunc(time_inf)
sigma = sigmafunc(time_l)


# wrapper for using the system of differential equations in different contexts
def exec_ode_full(s: float, e: float, i: float, r: float, hosp_rec: float, home_rec: float, time_steps: np.ndarray,
                  ktup: Tuple, mode: Optional[str] = 'full') -> Union[Iterable, int]:
    results = odeint(seir, (s, e, i, r, hosp_rec, home_rec), time_steps, ktup)
    if mode != 'full':
        # return hospitalizations scalar if not in full mode
        # noinspection PyTypeChecker
        result_scalar = results[-1, 5] if results.shape[0] > 0 else 0
        return result_scalar
    else:
        return results


def max_result(st_dt: datetime.datetime, results: np.ndarray, idx: int) -> Tuple:
    max_val = int(max(results[:, idx]))
    max_tdelta = datetime.timedelta(int(np.argmax(results[:, idx])))
    max_date = st_dt + max_tdelta
    return max_val, max_date


def plot_seir_results(county_dates, county_model, r0_est, rt_est, county_st_dt):
    # plt.plot(county_dates, county_model[:,0],color = 'darkblue',label = 'Susceptible')
    plt.plot(county_dates, county_model[:, 1], color='orange', label='Exposed')
    plt.plot(county_dates, county_model[:, 2], color='red', label='Infection')
    # plt.plot(county_dates, county_model[:,3],color = 'purple',label = 'Quarantine')
    # plt.plot(county_dates, county_model[:,4],color = 'green',label = 'Recovery')
    plt.plot(county_dates, county_model[:, 5], color='yellow', label='Hospital Recovery')
    plt.title(f'Simple System Evolution Using Historical Naive/Current Rs={r0_est}, {rt_est}')
    plt.suptitle(f'County SEIR Forecast: {config.seir_target_county}')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Number')
    plt.yscale('linear')
    plt.xticks(rotation=90)
    plt.show()
    max_infected = max_result(county_st_dt, county_model, 2)
    max_hospitalizations = max_result(county_st_dt, county_model, 5)
    print(f'max infected on initial peak: {max_infected[0]} ({max_infected[1]})')
    print(f'max hospitalizations on initial peak: {max_hospitalizations[0]} ({max_hospitalizations[1]})')


def build_seir_input(rt_df: pd.DataFrame) -> \
        [int, int, np.ndarray, np.ndarray, float, float, datetime.datetime, datetime.datetime]:
    target_county_df = rt_df[rt_df.index.get_level_values('name') == config.seir_target_county]
    county_st_dt = target_county_df.index.get_level_values('Date').min()
    county_f_dt = target_county_df.index.get_level_values('Date').max()
    target_county_est_pop = target_county_df.index.get_level_values('estimated_pop').max()
    target_county_st_tot_conf_cases = \
        target_county_df.loc[pd.IndexSlice[:, :, :, :, county_st_dt], ['Total Estimated Cases']].to_numpy()[0][0]
    target_county_f_growth_n = target_county_df.loc[pd.IndexSlice[:, :, :, :, county_f_dt],
                                                    ['growth_period_n']].to_numpy()[0][0]
    target_county_f_growth_2nd = target_county_df.loc[pd.IndexSlice[:, :, :, :, county_f_dt],
                                                      ['2nd_order_growth']].to_numpy()[0][0]
    r0_est = target_county_df.loc[pd.IndexSlice[:, :, :, :, county_f_dt], ['naive_R0']].to_numpy()[0][0].round(2)
    rt_est = target_county_df.loc[pd.IndexSlice[:, :, :, :, county_f_dt], ['Rt']].to_numpy()[0][0].round(2)
    if any(math.isnan(x) for x in [target_county_f_growth_2nd, target_county_f_growth_n]):
        print('Not enough data yet to run an SD intervention model for this county')
    return target_county_est_pop, target_county_st_tot_conf_cases, rt_est, r0_est, county_f_dt, county_st_dt


def model_temp_constraints(county_f_dt: datetime.datetime, county_st_dt: datetime.datetime,
                           t_total: int = config.t_total) -> [int, List, np.ndarray, np.ndarray]:
    history_dur = (county_f_dt - county_st_dt).days
    history_dates = [county_st_dt + datetime.timedelta(n) for n in range(history_dur + 1)]
    f_dates = [county_st_dt + datetime.timedelta(n) for n in range(history_dur + 1, t_total)]
    county_dates = history_dates + f_dates
    time_steps_hist = np.arange(0, history_dur + 1)
    # post-intervention
    time_steps_f = np.arange(history_dur, t_total)
    return history_dur, county_dates, time_steps_hist, time_steps_f


def gen_seir_viz(rt_df: pd.DataFrame) -> None:
    seir_inputs = build_seir_input(rt_df)
    county_f_dt, county_st_dt = seir_inputs[4], seir_inputs[5]
    temporal_constraints = model_temp_constraints(county_f_dt, county_st_dt)
    # define model inputs and constraints
    target_county_est_pop, target_county_st_tot_conf_cases, rt_est, r0_est = seir_inputs[0:4]
    history_dur, county_dates, time_steps_hist, time_steps_f = temporal_constraints
    # piecewise model (hist, future)
    county_model_hist = exec_ode_full(target_county_est_pop, target_county_st_tot_conf_cases,
                                      target_county_st_tot_conf_cases, 0, 0, 0, time_steps_hist,
                                      (r0_est, target_county_est_pop))
    county_model_f = exec_ode_full(*county_model_hist[-1, 0:6], time_steps=time_steps_f,
                                   ktup=(rt_est, target_county_est_pop))
    # remove dup row to facilitate historical to forecast transition
    county_model = np.delete(np.concatenate((county_model_hist, county_model_f), axis=0), history_dur, 0)
    plot_seir_results(county_dates, county_model, r0_est, rt_est, county_st_dt)
