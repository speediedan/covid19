from typing import Tuple
import os
import time
import datetime

import config
import c19_analysis.dataprep_utils as covid_utils
import c19_analysis.bayesian_rt_est as bayes_rt
# import c19_analysis.cust_seir_model as cust_seir
import dashboard.rt_explorer as rt_explorer
import dashboard.choropleth_explorer as choropleth_explorer
import dashboard.static_mpl_viz as static_mpl_viz
import pandas as pd
import faulthandler

faulthandler.enable()


def build_latest_case_data() -> Tuple[pd.DataFrame, bool]:
    covid_utils.create_dirs([config.eda_tmp_dir])
    covid_cases_df = pd.read_csv(config.JHU_CSSE_URL)
    county_pops_df = pd.read_csv(config.county_pops_csv)
    county_codes_df = pd.read_csv(config.county_codes_csv)
    latest_dt = covid_utils.latest_date(covid_cases_df)
    saved_dt = datetime.datetime.strptime(covid_utils.load_json(config.ds_meta), '%Y-%m-%d %H:%M:%S') \
        if config.ds_meta.exists() else datetime.datetime.strptime('01/01/2020', '%m/%d/%Y')
    updated = False
    if saved_dt < latest_dt or not config.latest_case_data_zip.exists():
        for cache in [config.repo_patient_onset_csv, config.county_rt_calc_zip]:
            # remove invalid downstream caches if built
            if cache.exists():
                os.remove(cache)
        covid_cases_df = covid_utils.process_df(covid_cases_df, county_pops_df, county_codes_df)
        covid_utils.save_json(latest_dt, config.ds_meta)
        covid_cases_df.to_csv(config.latest_case_data_zip, compression='gzip')
        updated = True
    else:
        print('No update to case data source, loading core case data from cache')
        covid_cases_df = pd.read_csv(config.latest_case_data_zip, compression='gzip',
                                     index_col=['id', 'estimated_pop', 'name', 'stateAbbr', 'Date'], parse_dates=True)
    covid_delta_df = covid_utils.add_columns(covid_cases_df)
    return covid_delta_df, updated


def main() -> None:
    dmode = int(os.environ['DMODE']) if 'DMODE' in os.environ.keys() else 0
    covid_delta_df = None
    if dmode:
        update_complete = False
        while not update_complete:
            covid_delta_df, updated = build_latest_case_data()
            if updated:
                update_complete = True
            else:
                time.sleep(config.case_src_check_interval)
    else:
        covid_delta_df, updated = build_latest_case_data()
    rt_df = bayes_rt.gen_rt_df(covid_delta_df)
    # cust_seir.gen_seir_viz(rt_df)
    rt_df, viz_df_instances, status_df, county_date_instances = covid_utils.prep_dashboard_dfs(rt_df)
    dashboard_dfs = rt_explorer.build_dashboard_doc(rt_df, status_df, debug_mode=False)
    primary_rt_plot_df, counties_df, main_plot_df, counties = dashboard_dfs
    choropleth_explorer.build_choropleth_doc(viz_df_instances, county_date_instances, debug_mode=False)
    static_mpl_viz.build_static_dashboards(primary_rt_plot_df, main_plot_df)


if __name__ == '__main__':
    main()
