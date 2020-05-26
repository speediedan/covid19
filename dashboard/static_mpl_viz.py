from typing import List
import math
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.dates import DateFormatter, date2num
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
import config
register_matplotlib_converters()
plt.ioff()


def plot_rt(fig: plt.Figure, target_df: pd.DataFrame, ax: plt.Axes, county_name: str) -> None:
    above = [1, 0, 0]
    middle = [1, 1, 1]
    below = [0, 0, 0]
    cmap = ListedColormap(np.r_[np.linspace(below, middle, 25), np.linspace(middle, above, 25)])
    color_mapped = lambda y: np.clip(y, .5, 1.5) - .5
    target_df = target_df.loc[(target_df.index.get_level_values('name') == county_name)]
    start_dt = pd.to_datetime(target_df['Rt'].index.get_level_values('Date').min())
    index = pd.to_datetime(target_df['Rt'].index.get_level_values('Date'))
    values = target_df['Rt'].values
    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index, values, s=40, lw=.5, c=cmap(color_mapped(values)), edgecolors='k', zorder=2)
    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index), target_df['90_CrI_LB'].values, bounds_error=False, fill_value='extrapolate')
    highfn = interp1d(date2num(index), target_df['90_CrI_UB'].values, bounds_error=False, fill_value='extrapolate')
    extended = pd.date_range(start=start_dt - pd.Timedelta(days=3), end=index[-1] + pd.Timedelta(days=1))
    ax.fill_between(extended, lowfn(date2num(extended)), highfn(date2num(extended)),
                    color='k', alpha=.1, lw=0, zorder=3)
    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25)
    ax.set_title(f'{county_name}', loc='left', fontsize=20, fontweight=0, color='#375A97')
    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0, 5.0)
    ax.set_xlim(start_dt - pd.Timedelta(days=3), target_df.index.get_level_values('Date')[-1] + pd.Timedelta(days=1))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.set_facecolor('w')


def build_static_rtplot(target_df: pd.DataFrame, ttl: str) -> None:
    ncols = 5
    nrows = int(np.ceil(len(target_df.index.unique(level='name')) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(40, 30))
    for i, (name, county_df) in enumerate(target_df.groupby(level='name')):
        plot_rt(fig, county_df, axes.flat[i], name)
    plt.suptitle(f"{ttl}", fontsize=24, fontweight=0, color='#375A97', style='italic', y=0.92)
    fig.text(0.10, 0.5, ttl, ha='center', va='center', rotation='vertical', fontsize=20)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    image_dest = Path(f"{config.eda_tmp_dir}/{ttl}.jpg")
    fig.savefig(image_dest, dpi=150, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


def multline_plot(tmp_df: pd.DataFrame, ttl: str, ylims: List[float], sc: str, ythresh: float) -> None:
    plt.style.use('seaborn-darkgrid')
    f = plt.figure(figsize=(40, 50), dpi=50)
    cmap = plt.get_cmap('jet')
    num = 0
    axes = []
    colors = cmap(np.linspace(0, 1.0, 30))
    date_form = DateFormatter("%m-%d")
    for column in tmp_df:
        axes.append(plt.subplot(6, 5, num + 1))
        gpos = np.ma.masked_where(tmp_df[column] < ythresh, tmp_df[column])
        gneg = np.ma.masked_where(tmp_df[column] > ythresh, tmp_df[column])
        axes[num].plot(tmp_df.index, gpos, color='red', marker='', linewidth=1, alpha=0.9, label=column)
        axes[num].plot(tmp_df.index, gneg, color='green', marker='', linewidth=1, alpha=0.9, label=column)
        plt.xlim(datetime.datetime.today() - datetime.timedelta(30), datetime.datetime.today())
        axes[num].xaxis.set_major_formatter(date_form)
        plt.xticks(rotation=90, fontsize=16)
        plt.yticks(fontsize=16)
        plt.yscale(sc)
        plt.ylim(ylims)
        axes[num].yaxis.set_major_formatter(ScalarFormatter())
        axes[num].xaxis.grid(True, which='minor')
        if num in range(21):
            axes[num].tick_params(labelbottom='off')
        if num not in [1, 11, 21]:
            axes[num].tick_params(labelleft='off')
        plt.title(column, loc='left', fontsize=20, fontweight=0, color=colors[num])
        num += 1
    plt.suptitle(f"{ttl} of Counties w/ Highest Current Estimated Cases", fontsize=24, fontweight=0, color='#375A97',
                 style='italic', y=0.90)
    f.text(0.10, 0.5, ttl, ha='center', va='center', rotation='vertical', fontsize=20)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    image_dest = Path(f"{config.eda_tmp_dir}/{ttl}.jpg")
    f.savefig(image_dest, dpi=150, bbox_inches='tight', pad_inches=0.15)
    plt.close(f)


def build_static_dashboards(primary_rt_plot_df: pd.DataFrame, main_plot_df: pd.DataFrame) -> None:
    build_static_rtplot(primary_rt_plot_df, 'Rt (Top Total Estimated Cases)')
    sg_names = ['2nd Order Growth', 'Cumulative Case Growth (4-Day MA)', 'Estimated Onset Cases']
    sg_cols = ['2nd_order_growth', 'growth_period_n', 'Estimated Onset Cases']
    sg_lims = [[-3, 5], [0, 1], [0, 4000]]
    sg_axes = ['symlog', 'linear', 'linear']
    sg_thresholds = [0, .05, 100]
    for ttl, col, lims, sc, t in zip(sg_names, sg_cols, sg_lims, sg_axes, sg_thresholds):
        df_tmp = main_plot_df.pivot(index='Date', columns='name', values=col)
        df_tmp = df_tmp.applymap(lambda x: 0 if x == np.inf or math.isnan(x) or x == -1.0 else x)
        multline_plot(df_tmp, ttl, lims, sc, t)
