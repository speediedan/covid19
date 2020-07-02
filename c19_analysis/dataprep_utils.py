import numpy as np
import pandas as pd
import shutil
import sys
import re

from typing import Any, List, Dict, Tuple
import json
from pathlib import Path
import os
from os import PathLike
import datetime

from tqdm.notebook import tqdm
import requests
from IPython.display import clear_output

import config as config


def create_dirs(dirs: List):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except OSError as err:
        print(f"Creating directories error: {err}")


def load_json(filename: str) -> Dict:
    # open the file as read only
    with open(filename, 'r') as f:
        # read all text
        json_data = json.load(f)
    return json_data


def datetime_serde(obj: Any) -> str:
    if isinstance(obj, datetime.date):
        return obj.__str__()


def save_json(var: Any, filename: str) -> None:
    json_data = json.dumps(var, indent=4, default=datetime_serde)
    with open(filename, 'w') as file:
        for line in json_data.split('\n'):
            file.write(line + '\n')


def patient_onset_stage(gzipped_file: Path, src_url: PathLike, target_dir: Path):
    # manually unzipping tar.gz. file due to header mismatch bug detected on 2020.06.30
    try:
        download_large_file(src_url, gzipped_file)
        clear_output(wait=True)
        print('Done downloading.')
    except:
        print('Something went wrong with patient onset download, cannot proceed with analysis. Exiting...')
        sys.exit(1)
    shutil.unpack_archive(str(gzipped_file), str(target_dir))


def read_cached_csv(cached_csv_loc: Path, src_url: PathLike = None, stream: bool = False, src_gzipped: bool = False,
                    read_csv_args: Dict = None) -> pd.DataFrame:
    if src_gzipped:
        read_csv_args['compression'] = 'gzip'
    if not cached_csv_loc.exists():
        if not src_url:
            print(f"The specified csv file doesn't exist and no download source provided. Exiting...")
            sys.exit(1)
        if stream:
            try:
                download_large_file(src_url, cached_csv_loc)
                clear_output(wait=True)
                print('Done downloading.')
            except:
                print('Something went wrong. Try again.')
            return pd.read_csv(cached_csv_loc, **read_csv_args)
        else:
            cached_df = pd.read_csv(src_url, **read_csv_args)
            cached_df.to_csv(cached_csv_loc, index=False)
        return cached_df
    else:
        return pd.read_csv(cached_csv_loc, **read_csv_args)


def download_large_file(url, local_filename):
    """From https://stackoverflow.com/questions/16694907/"""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    return local_filename


def save_bokeh_tags(tags: List[str], filename: str) -> None:
    with open(filename, 'w') as file:
        for t in tags:
            file.write(t)


def save_bokeh_ent(ent: str, filename: str) -> None:
    with open(filename, 'w') as file:
        file.write(ent)


def dailydata(df_subr: pd.DataFrame, oldname: str, newname: str) -> pd.DataFrame:
    subr_daily = df_subr.groupby(['id', 'name', 'stateAbbr']).diff().fillna(0)
    subr_daily = subr_daily.rename(columns={oldname: newname})
    subr_daily[newname] = subr_daily.apply(lambda x: 0 if x[newname] < 0 else x[newname], axis=1)
    return subr_daily


def date_xform_old(time_series_df: pd.DataFrame, dt_cnt: int) -> pd.DataFrame:
    usaf_start_dt = pd.to_datetime('1/22/2020', format='%m/%d/%Y')
    dates_us = [(usaf_start_dt + datetime.timedelta(n)).date() for n in range(dt_cnt)]
    time_series_df.rename(columns={'POPESTIMATE2018': 'estimated_pop'}, inplace=True)
    new_name_dict = dict(zip(time_series_df.columns[3:].tolist(), dates_us))
    time_series_df = time_series_df.rename(columns=new_name_dict)
    # unpivot date columns keeping identifiying columns specified, then set index based on renamed variable column
    time_series_df.reset_index(inplace=True)
    time_series_df = time_series_df.melt(id_vars=['id', 'estimated_pop', 'name', 'stateAbbr'], var_name='Date',
                                         value_name='Cases')
    time_series_df = time_series_df.set_index(['id', 'estimated_pop', 'name', 'stateAbbr', 'Date'])
    return time_series_df


def date_xform(time_series_df: pd.DataFrame) -> pd.DataFrame:
    time_series_df.rename(columns={'POPESTIMATE2018': 'estimated_pop', 'State': 'stateAbbr'}, inplace=True)
    # drop unnamed columns before parsing (data feed occasionally includes an errant comma)
    drop_unnamed = [c for c in time_series_df.columns if re.compile(r"Unnamed*").match(c)]
    if drop_unnamed:
        time_series_df.drop(columns=drop_unnamed, inplace=True)
    # unpivot date columns keeping identifiying columns specified, then set index based on renamed variable column
    time_series_df = time_series_df.melt(id_vars=['id', 'estimated_pop', 'name', 'stateAbbr'], var_name='Date',
                                         value_name='Cases')
    time_series_df['Date'] = time_series_df['Date'].apply(lambda x: pd.to_datetime(x, format='%m/%d/%y'))
    time_series_df = time_series_df.set_index(['id', 'estimated_pop', 'name', 'stateAbbr', 'Date'])
    return time_series_df


def prep_time_series(df: pd.DataFrame, cp: pd.DataFrame, cc: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=['countyFIPS', 'stateFIPS'])
    df['id'] = df.apply(lambda x: x['countyFIPS'] if x['countyFIPS'] else x['stateFIPS'] * 1000, axis=1)
    # df.drop(columns=['countyFIPS', 'county', 'stateFIPS', 'deaths', 'popul'], inplace=True)
    df.drop(columns=['countyFIPS', 'County Name', 'stateFIPS'], inplace=True)
    # Load the state and county codes
    df = pd.merge(cc, df, how='left', on='id')
    df.fillna(0, inplace=True)
    # df['confirmed'] = df.apply(lambda x: x['confirmed'] if x['confirmed'] else [0] * dt_cnt, axis=1)
    # time_series_dfs = [df, pd.DataFrame(df['confirmed'].tolist())]
    # time_series_df = pd.concat(time_series_dfs, axis=1).drop('confirmed', axis=1)
    # time_series_df.set_index(['id'], inplace=True)
    time_series_df = pd.merge(cp, df, how='left', on='id')
    time_series_df = date_xform(time_series_df)
    return time_series_df


def update_onset_xform(tmp_df: pd.DataFrame) -> pd.DataFrame:
    tmp_df.columns = ['Onset', 'Confirmed']
    # There's an errant reversed date
    tmp_df = tmp_df.replace('01.31.2020', '31.01.2020')
    # remove an errant date
    tmp_df = tmp_df.replace('31.04.2020', None)
    # Only keep if both values are present
    tmp_df = tmp_df.dropna()
    # Must have strings that look like individual dates
    # "2020.03.09" is 10 chars long
    is_ten_char = lambda x: x.str.len().eq(10)
    tmp_df = tmp_df[is_ten_char(tmp_df.Confirmed) & is_ten_char(tmp_df.Onset)]
    # Convert both to datetimes
    tmp_df.Confirmed = pd.to_datetime(tmp_df.Confirmed, format='%d.%m.%Y')
    try:
        tmp_df.Onset = pd.to_datetime(tmp_df.Onset, format="%d.%m.%Y")
    except ValueError as e:
        print(e)
    # Only keep records where confirmed > onset
    tmp_df = tmp_df[tmp_df.Confirmed >= tmp_df.Onset]
    return tmp_df


def generate_onset_dist(patients: pd.DataFrame) -> pd.Series:
    # Calculate the delta in days between onset and confirmation
    delay = (patients.Confirmed - patients.Onset).dt.days
    # Convert samples to an empirical distribution
    p_delay = delay.value_counts().sort_index()
    new_range = np.arange(0, p_delay.index.max() + 1)
    p_delay = p_delay.reindex(new_range, fill_value=0)
    p_delay /= p_delay.sum()
    return p_delay


def confirmed_to_onset(confirmed, p_delay):
    assert not confirmed.isna().any()
    # Reverse cases so that we convolve into the past
    convolved = np.convolve(confirmed[::-1].values, p_delay)
    # Calculate the new date range
    dr = pd.date_range(end=confirmed.index[-1], periods=len(convolved))
    # Flip the values and assign the date range
    onset = pd.Series(np.flip(convolved), index=dr)
    return onset


def adjust_onset_for_right_censorship(onset, p_delay):
    cumulative_p_delay = p_delay.cumsum()
    # Calculate the additional ones needed so shapes match
    ones_needed = len(onset) - len(cumulative_p_delay)
    padding_shape = (0, ones_needed)
    # Add ones and flip back
    cumulative_p_delay = np.pad(cumulative_p_delay, padding_shape, constant_values=1)
    cumulative_p_delay = np.flip(cumulative_p_delay)
    # Adjusts observed onset values to expected terminal onset values
    adjusted = onset / cumulative_p_delay
    return adjusted, cumulative_p_delay


def onset_shift_by_county(tmp_df: pd.DataFrame, onset_df: pd.Series, test_mode: bool = False) -> pd.DataFrame:
    onset_df_tmps = []
    tmp_df = tmp_df.reset_index()
    counties = tmp_df['name'].unique().tolist() if not test_mode else config.test_counties
    for c in tqdm(counties):
        onset_tmp_df = tmp_df.loc[(tmp_df['name'] == c)]
        index_values = onset_tmp_df['Date'].values.flatten()
        cases_tmp = onset_tmp_df['Daily New Cases'].values.flatten()
        onset_tmp = pd.Series(data=cases_tmp, index=index_values)
        onset = confirmed_to_onset(onset_tmp, onset_df)
        adjusted_onset, _ = adjust_onset_for_right_censorship(onset, onset_df)
        adjusted_onset = adjusted_onset.to_frame()
        adjusted_onset[['id', 'estimated_pop', 'name', 'stateAbbr']] = pd.DataFrame([onset_tmp_df.iloc[0].values[0:4]],
                                                                                    index=adjusted_onset.index)
        adjusted_onset.reset_index(inplace=True)
        adjusted_onset.rename(columns={"index": "Date", 0: "Estimated Onset Cases"}, inplace=True)
        adjusted_onset = adjusted_onset.set_index(['id', 'estimated_pop', 'name', 'stateAbbr', 'Date'])
        onset_df_tmps.append(adjusted_onset)
    return pd.concat(onset_df_tmps, axis=0)


def process_df(df_raw: pd.DataFrame, county_pops: pd.DataFrame, county_codes: pd.DataFrame,
               dt_cnt: int) -> pd.DataFrame:
    county_pops = county_pops.astype({'STATE': 'int64', 'COUNTY': 'int64'})
    county_pops['id'] = county_pops.apply(lambda x: x['STATE'] * 1000 + x['COUNTY'], axis=1)
    county_pops = county_pops.drop(columns=['STATE', 'COUNTY', 'CTYNAME']).set_index(['id'])
    time_series_df = prep_time_series(df_raw, county_pops, county_codes)
    time_series_df = dailydata(time_series_df, 'Cases', 'Daily New Cases')
    if not Path(config.repo_patient_onset_csv).exists():
        patient_onset_stage(config.repo_patient_onset_zip, config.PATIENT_ONSET_MAP_URL, config.eda_tmp_dir)
    onset_confirmed_df = read_cached_csv(config.repo_patient_onset_csv, read_csv_args=config.onset_args)
    onset_confirmed_df = update_onset_xform(onset_confirmed_df)
    onset_df = generate_onset_dist(onset_confirmed_df)
    adjusted_onset_df = onset_shift_by_county(time_series_df, onset_df, test_mode=False)
    time_series_df = time_series_df.reset_index().set_index(['id', 'estimated_pop', 'name', 'stateAbbr', 'Date'])[
        ~time_series_df.index.duplicated()]
    adjusted_onset_df = pd.concat([adjusted_onset_df, time_series_df['Daily New Cases']], axis=1).rename(
        columns={'Daily New Cases': 'Confirmed New Cases'}).fillna(0)
    return adjusted_onset_df


def add_columns(df_subr: pd.DataFrame) -> pd.DataFrame:
    df_subr['Total Estimated Cases'] = df_subr.groupby(['id', 'estimated_pop', 'name', 'stateAbbr'])[
        'Estimated Onset Cases'].cumsum().astype(int)
    df_subr = df_subr[df_subr['Total Estimated Cases'] > 0]
    df_subr['node_start_dt'] = df_subr.groupby(['id', 'estimated_pop', 'name', 'stateAbbr'])[
        'Total Estimated Cases'].idxmin().apply(lambda x: pd.to_datetime(x[-1]))
    df_subr.reset_index(inplace=True)
    df_subr['node_days'] = df_subr['Date'] - df_subr['node_start_dt']
    df_subr = df_subr.set_index(['id', 'estimated_pop', 'name', 'stateAbbr', 'Date'])
    df_subr = df_subr.sort_values(by=['id', 'Date'])
    df_subr['daily new cases ma'] = df_subr['Estimated Onset Cases'].rolling(window=4).mean().round()
    df_subr['growth_rate'] = df_subr['Total Estimated Cases'].pct_change().round(4)
    df_subr.loc[df_subr['node_days'] < np.timedelta64(1, 'D'), 'growth_rate'] = None
    df_subr['growth_period_n'] = df_subr['growth_rate'].rolling(4).mean()
    df_subr['growth_period_n-1'] = df_subr['growth_rate'].shift(4).rolling(4).mean()
    df_subr.loc[df_subr['node_days'] < np.timedelta64(3, 'D'), 'daily new cases ma'] = None
    df_subr.loc[df_subr['node_days'] < np.timedelta64(4, 'D'), 'growth_period_n'] = None
    df_subr.loc[df_subr['node_days'] < np.timedelta64(8, 'D'), 'growth_period_n-1'] = None
    df_subr['2nd_order_growth'] = (df_subr['growth_period_n'] / df_subr['growth_period_n-1']).round(4) - 1
    df_subr.loc[df_subr['node_days'] < np.timedelta64(8, 'D'), '2nd_order_growth'] = None
    df_subr = df_subr.loc[df_subr.groupby(['id', 'estimated_pop', 'name', 'stateAbbr'])['daily new cases ma'].apply(lambda x: x > 0)]
    df_subr = df_subr.loc[state_condition(df_subr)]
    return df_subr


def color_mask(col: pd.Series, thresh: float) -> List:
    good_color = 'green'
    bad_color = 'red'
    return [f'color: {good_color}' if v < thresh else f'color: {bad_color}' for v in col]


def state_condition(df: pd.DataFrame):
    return df.index.get_level_values('id') % 1000 != 0


def build_viz_dfs(rt_df: pd.DataFrame, viz_cols: List) -> Tuple[List, datetime.datetime, List]:
    # set horizon for national rt visualization
    viz_horizon = datetime.timedelta(13)
    county_f_st_dt = rt_df.index.get_level_values('Date').max()
    county_date_instances = [(county_f_st_dt - viz_horizon) + datetime.timedelta(n) for n in
                             range(viz_horizon.days + 1)]
    viz_df_instances = []
    for d in county_date_instances:
        tmpdf = rt_df.loc[pd.IndexSlice[:, :, :, :, d], viz_cols]
        tmpdf['confirmed %infected'] = round(
            (tmpdf['Total Estimated Cases'] / tmpdf.index.get_level_values('estimated_pop')) * 100, 2)
        tmpdf = tmpdf.sort_values(['confirmed %infected'], ascending=False)
        viz_df_instances.append(tmpdf)
    return viz_df_instances, county_f_st_dt, county_date_instances


def export_rtdf(df: pd.DataFrame) -> None:
    column_mask = ['Rt', '90_CrI_LB', '90_CrI_UB', '2nd_order_growth']
    exported_rtdf = df[column_mask]
    exported_rtdf.to_csv(config.exported_rtdf_csv, compression='gzip')
    exported_rtdf.reset_index().to_json(path_or_buf=config.exported_rtdf_json, date_format='iso', orient='records',
                                        compression='gzip')


def prep_dashboard_dfs(rt_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[pd.DataFrame], pd.DataFrame, List]:
    viz_cols = ['Estimated Onset Cases', 'Total Estimated Cases', 'node_start_dt', 'daily new cases ma',
                'Confirmed New Cases', 'growth_rate', 'growth_period_n', 'growth_period_n-1', '2nd_order_growth', 'Rt',
                '90_CrI_LB', '90_CrI_UB']
    rt_df = rt_df.loc[(rt_df['2nd_order_growth'] < np.inf) & state_condition(rt_df), :]
    export_rtdf(rt_df)
    viz_df_instances, county_f_st_dt, county_date_instances = build_viz_dfs(rt_df, viz_cols)
    status_df = rt_df.loc[pd.IndexSlice[:, :, :, :, county_f_st_dt], viz_cols]
    status_df['confirmed %infected'] = \
        round((status_df['Total Estimated Cases'] / status_df.index.get_level_values('estimated_pop')) * 100, 2)
    status_df = status_df.sort_values(['confirmed %infected'], ascending=False)
    return rt_df, viz_df_instances, status_df, county_date_instances
