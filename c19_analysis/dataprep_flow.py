from typing import Tuple
import os
import time

import config
import c19_analysis.dataprep_utils as covid_utils
import c19_analysis.bayesian_rt_est as bayes_rt
# import c19_analysis.cust_seir_model as cust_seir
import dashboard.rt_explorer as rt_explorer
import dashboard.choropleth_explorer as choropleth_explorer
import dashboard.static_mpl_viz as static_mpl_viz
import pandas as pd


def build_latest_case_data() -> Tuple[pd.DataFrame, bool]:
    covid_utils.create_dirs([config.eda_tmp_dir])
    # usafacts_df = pd.read_json(config.USAFACTS_URL)
    usafacts_df = pd.read_csv(config.USAFACTS_URL)
    county_pops_df = pd.read_csv(config.county_pops_csv)
    county_codes_df = pd.read_csv(config.county_codes_csv)
    dt_cnt = len(usafacts_df.columns)-4
    saved_cnt = covid_utils.load_json(config.ds_meta) if config.ds_meta.exists() else 0
    updated = False
    if saved_cnt < dt_cnt or not config.latest_case_data_zip.exists():
        for cache in [config.repo_patient_onset_csv, config.county_rt_calc_zip]:
            # remove invalid downstream caches if built
            if cache.exists():
                os.remove(cache)
        usafacts_df = covid_utils.process_df(usafacts_df, county_pops_df, county_codes_df, dt_cnt)
        covid_utils.save_json(dt_cnt, config.ds_meta)
        usafacts_df.to_csv(config.latest_case_data_zip, compression='gzip')
        updated = True
    else:
        print('No update to case data source, loading core case data from cache')
        usafacts_df = pd.read_csv(config.latest_case_data_zip, compression='gzip',
                                  index_col=['id', 'estimated_pop', 'name', 'stateAbbr', 'Date'], parse_dates=True)
    usafacts_delta_df = covid_utils.add_columns(usafacts_df)
    return usafacts_delta_df, updated


def main() -> None:
    dmode = int(os.environ['DMODE']) if 'DMODE' in os.environ.keys() else 0
    usafacts_delta_df = None
    if dmode:
        update_complete = False
        while not update_complete:
            usafacts_delta_df, updated = build_latest_case_data()
            if updated:
                update_complete = True
            else:
                time.sleep(config.case_src_check_interval)
    else:
        usafacts_delta_df, updated = build_latest_case_data()
    rt_df = bayes_rt.gen_rt_df(usafacts_delta_df)
    # cust_seir.gen_seir_viz(rt_df)
    rt_df, viz_df_instances, status_df, county_date_instances = covid_utils.prep_dashboard_dfs(rt_df)
    dashboard_dfs = rt_explorer.build_dashboard_doc(rt_df, status_df, debug_mode=False)
    primary_rt_plot_df, counties_df, main_plot_df, counties = dashboard_dfs
    choropleth_explorer.build_choropleth_doc(viz_df_instances, county_date_instances, debug_mode=False)
    static_mpl_viz.build_static_dashboards(primary_rt_plot_df, main_plot_df)


if __name__ == '__main__':
    main()
