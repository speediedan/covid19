from typing import List
from pathlib import Path
import os
import sys

import pandas as pd
import numpy as np
from bokeh.settings import settings


def create_dirs(dirs: List) -> None:
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success 1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except OSError as err:
        print(f"Creating directories error: {err}")
        sys.exit(0)


# pands/bokeh config
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_row', 2000)
settings.py_log_level = 'info'
settings.log_level = 'info'
pd.options.mode.chained_assignment = None

# remote files
#USAFACTS_URL = "https://usafactsstatic.blob.core.windows.net/public/2020/coronavirus-timeline/allData.json"
USAFACTS_URL = "https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv"
PATIENT_ONSET_MAP_URL = \
    "https://github.com/beoutbreakprepared/nCoV2019/blob/master/latest_data/latestdata.tar.gz?raw=true"

# remote file-specific config
onset_args = {"parse_dates": False, "usecols": ['date_confirmation', 'date_onset_symptoms'], "low_memory": False}

# local repo files
COVID_BASE = os.environ['COVID_BASE'] if 'COVID_BASE' in os.environ.keys() else \
    os.path.dirname(os.path.realpath(__file__))

us_counties_path = Path(f"{COVID_BASE}/static_datasets/us_counties.tar.gz")
county_pops_csv = Path(f"{COVID_BASE}/static_datasets/county_pops.csv")
county_codes_csv = Path(f"{COVID_BASE}/static_datasets/county_codes.csv")
state_fips_csv = Path(f"{COVID_BASE}/static_datasets/state_fips.csv")

# various staging/cache file locations
default_stage_dir = f"{os.environ['HOME']}/datasets/covid19/master"
curr_base = COVID_BASE.rsplit('/', 1)[1]
default_stage_dir = f"{os.environ['HOME']}/datasets/covid19/{curr_base}" if curr_base != "covid19" else \
    f"{os.environ['HOME']}/datasets/covid19/master"
eda_tmp_dir = os.environ['STAGE_DIR'] if 'STAGE_DIR' in os.environ.keys() else default_stage_dir
if not os.path.exists(eda_tmp_dir):
    create_dirs([eda_tmp_dir])
ds_meta = Path(f"{eda_tmp_dir}/ds_meta.json")
repo_patient_onset_zip = Path(f"{eda_tmp_dir}/latestdata.tar.gz")
repo_patient_onset_csv = Path(f"{eda_tmp_dir}/latestdata.csv")
latest_case_data_zip = Path(f"{eda_tmp_dir}/latest_case_data.tar.gz")
county_rt_calc_zip = Path(f"{eda_tmp_dir}/latest_county_rt_data.tar.gz")
county_covid_explorer_tags = Path(f"{eda_tmp_dir}/county_covid_explorer_tags.html")
choro_covid_explorer_tags = Path(f"{eda_tmp_dir}/choropleth_covid_explorer_tags.html")
national_layout_png_tmp = Path(f"{eda_tmp_dir}/national_layout_tmp.png")
cpath_counties_zip = Path(f"{eda_tmp_dir}/cpath_counties_df.tar.gz")
exported_rtdf_json = Path(f"{eda_tmp_dir}/rtdf_export_json.tar.gz")
exported_rtdf_csv = Path(f"{eda_tmp_dir}/rtdf_export_csv.tar.gz")

# Rt estimation constraints
# range of r_t to explore when calculating r_t using Bayesian approach
# based upon https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0002185
R_T_MAX = 10
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX * 100 + 1)
# keep only counties with confirmed case density of > n per million people
case_density = 300
epsilon = np.finfo(float).eps
# test counties
test_counties = ['King County, WA', 'Kings County, NY', 'Bullock County, AL']

# SEIR model defaults
seir_target_county = 'Kings County, NY'
t_total = 200  # days to forecast

# Dashboard config
# for main static status dashboard, rank counties with largest confirmed counts
min_case_cnt = 200  # min case count for main static status dashboards and rt_explorer datasource
min_days_over = 2

# when updating dashboard in daemon mode, wait this time between checks of the case data source
case_src_check_interval = 3600

# mail notification config
# must set MAIL_USER and MAIL_APP_PASSWORD env params
mail_subject = 'Covid19 Real-time Rt Dashboard Generation Results'