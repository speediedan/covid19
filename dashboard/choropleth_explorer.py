from typing import List, Dict, Tuple
import math
from pathlib import Path
import os
import re
from os import PathLike
import datetime

import numpy as np
import pandas as pd
from PIL import Image
from bokeh.models import (ColumnDataSource, CustomJS, HoverTool, Select, ColorBar, BasicTicker, LinearColorMapper, Div)
from bokeh.plotting import Figure, figure
from bokeh.io import show, curdoc, export_png
from bokeh.embed import components
from bokeh.layouts import column, grid, row
from bokeh.models.annotations import Title
from bokeh.transform import transform

import config
import c19_analysis.dataprep_utils as covid_utils
import dashboard.dashboard_constants as constants


def build_aspect_map(data_source: ColumnDataSource) -> Tuple:
    # dynamically define aspect ratios for given data_source
    state_dims = []
    for d in ['lats', 'lons']:
        mind = min([min(c) for c in data_source.data[d]])
        maxd = max([max(c) for c in data_source.data[d]])
        state_dims.append(maxd - mind)
    aspect_ratio = state_dims[1] / state_dims[0]
    min_width, min_height, max_width, max_height = constants.base_dims
    min_height = int(round(min_height / aspect_ratio, 0))
    max_height = int(round(max_height / aspect_ratio, 0))
    return tuple((min_width, min_height, max_width, max_height))


def build_national_sources(cpleth_dfs: List[pd.DataFrame]) -> [List[ColumnDataSource], List[pd.DataFrame]]:
    national_cpleth_sources = []
    for df in cpleth_dfs:
        df['confirmed %infected'] = df['confirmed %infected'].apply(
            lambda x: round(x, 1) if isinstance(x, float) else x)
        df['2nd_order_growth'] = df['2nd_order_growth'].apply(lambda x: round(x, 1) if isinstance(x, float) else x)
        df['Rt'] = df['Rt'].apply(lambda x: round(x, 2) if isinstance(x, float) else x)
        national_cpleth_df = df.copy()
        national_cpleth_df = national_cpleth_df.reset_index().drop(columns=['state', 'id'])
        national_dict = {}
        for new, old in zip(
                ['lats', 'lons', 'name', 'Rt', 'confirmed_infected', 'total_estimated_cases', 'daily_new_cases_ma',
                 '2nd_order_growth'], list(national_cpleth_df.columns)):
            national_dict[new] = national_cpleth_df[old].tolist()
        national_cpleth_sources.append(ColumnDataSource(national_dict))
    return national_cpleth_sources, cpleth_dfs


def build_cpleth_plot_df(states: pd.Index, cpleth_dfs: List[pd.DataFrame]) \
        -> [List[ColumnDataSource], Dict[ColumnDataSource, Tuple], Figure, List[Figure]]:
    national_cpleth_sources, cpleth_dfs = build_national_sources(cpleth_dfs)
    cpleth_sources = {}
    # use only the latest national snapshot for state-level views
    tmpdf = None
    for n in states:
        tmpdf = cpleth_dfs[-1].loc[cpleth_dfs[-1].index.get_level_values('state') == n]
        tmpdf = tmpdf.reset_index().drop(columns=['state', 'id'])
        state_dict = {}
        for new, old in zip(
                ['lats', 'lons', 'name', 'Rt', 'confirmed_infected', 'total_estimated_cases', 'daily_new_cases_ma',
                 '2nd_order_growth'], list(tmpdf.columns)):
            state_dict[new] = tmpdf[old].tolist()
        cpleth_sources[n] = [ColumnDataSource(state_dict), build_aspect_map(ColumnDataSource(state_dict))]
    state_dict = {}
    for new, old in zip(['lats', 'lons', 'name', 'Rt', 'confirmed_infected', 'total_estimated_cases',
                         'daily_new_cases_ma', '2nd_order_growth'], list(tmpdf.columns)):
        state_dict[new] = np.empty((1, 1))
    # set a dummy initial cds that will be hidden
    cpleth_sources['default'] = [ColumnDataSource(state_dict), constants.base_dims]
    stateplot = plot_config('state', cpleth_sources['default'][0])
    nationalplots = []
    for s in national_cpleth_sources:
        nationalplots.append(plot_config('national', s))
    return national_cpleth_sources, cpleth_sources, stateplot, nationalplots


def plot_config(ptype: str, data_source: ColumnDataSource) -> Figure:
    mapper = LinearColorMapper(palette=constants.cust_rg_palette, low=0.8, high=1.2, low_color='#33FF33',
                               high_color='#FF3333',
                               nan_color='#33FF33')
    color_bar = ColorBar(color_mapper=mapper, ticker=BasicTicker(),
                         label_standoff=12, border_line_color=None, location=(0, 0), background_fill_alpha=0,
                         background_fill_color=None, major_tick_line_alpha=0)
    custom_hover_format = CustomJS(
        args=dict(cust_tooltip_above=constants.cust_tooltip_above, cust_tooltip_below=constants.cust_tooltip_below),
        code=constants.custom_hover_code)
    national_args = dict(toolbar_location=None, plot_width=900, plot_height=600, frame_height=700, frame_width=1050)
    min_width, min_height, max_width, max_height = constants.base_dims
    state_args = {'width_policy': 'fit', 'height_policy': 'fit', 'min_height': min_height, 'min_width': min_width,
                  'max_height': max_height, 'max_width': max_width, 'min_border_left': 25,
                  'tools': 'pan,wheel_zoom,reset'}
    plot_tools_config = {'x_axis_location': None, 'y_axis_location': None, 'border_fill_alpha': 0,
                         'background_fill_alpha': 0,
                         'background_fill_color': None, 'border_fill_color': None, 'outline_line_alpha': 0,
                         'outline_line_width': 0}
    t = None
    if ptype == 'national':
        p = figure(**plot_tools_config, **national_args)
    else:
        p = figure(**plot_tools_config, **state_args)
        t = Title()
        t.text = 'Default'
        p.title = t
        p.title.text_font_size = "20pt"
        p.title.text_color = '#003566'
    p.add_tools(HoverTool(tooltips=None, callback=custom_hover_format,
                          formatters={'@2nd_order_growth': 'numeral', '@confirmed_infected': 'numeral',
                                      '@Rt': 'numeral'}))
    p.grid.grid_line_color = None
    p.hover.point_policy = "follow_mouse"
    p.hover.attachment = "above"
    p.toolbar.logo = None
    p.patches('lons', 'lats', source=data_source, fill_color=transform('Rt', mapper),
              fill_alpha=0.7, line_color='#003566', line_width=0.5, hover_color='#003566')
    if ptype != 'national':
        p.add_layout(color_bar, 'right')
        t.visible = False
    return p


def xform_transparent(src_img: PathLike, dest: PathLike) -> None:
    """
    Transforms a white background png into a transparent one.
    Despite using a transparent background configuration, bokeh function export_png is not exporting a transparent png.
    As such, I'm manually transforming the png using PIL here.
    """
    img = Image.open(src_img)
    img = img.convert("RGBA")
    img_tups = img.getdata()
    new_data = []
    for tup in img_tups:
        if tup[0] == 255 and tup[1] == 255 and tup[2] == 255:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(tup)
    img.putdata(new_data)
    img.save(dest, "PNG")


def gen_national_imgs(i: int, natp: Figure, county_date_instances: List) -> None:
    national_span = f"""
    <span style="font-size: 25px; font-weight:bold;color: #003566;position:absolute;top:20px;width:800px; left:75px;"
>Rt as of: {datetime.datetime.strftime(county_date_instances[i], '%B %-d, %Y')}</span>
"""
    national_ttl = Div(text=national_span, height_policy='min', width_policy='min', align='start')
    national_layout = column(national_ttl, natp)
    export_png(national_layout, filename=config.national_layout_png_tmp)
    national_layout_png = Path(f"{config.eda_tmp_dir}/national_layout_{i}.png")
    xform_transparent(config.national_layout_png_tmp, national_layout_png)
    os.remove(config.national_layout_png_tmp)


def build_cpleth_df(df_instances: List[pd.DataFrame]) -> [List[pd.DataFrame], pd.Index]:
    strip_apos = lambda x: [float(y) for y in re.sub(r'\'', '', x)[1:-1].split(',')]
    converters = {'lats': strip_apos, 'lons': strip_apos}
    us_counties_df = pd.read_csv(config.us_counties_path, compression='gzip', index_col=['id'], converters=converters)
    us_counties_df = us_counties_df[us_counties_df['state_id'] != 69]  # remove counties w/o a named state
    states_filter = us_counties_df['state'].isin(['ak', 'hi', 'pr', 'as', 'gu', 'vi'])
    us_counties_df = us_counties_df[~states_filter]
    merged_df_instances = []
    # create viz_horizon versions of county-level stats for independent visualizations
    # (ultimately a workaround for bokeh cdsview limitations wrt patches objects)
    cpleth_cols = ['id', 'lats', 'lons', 'name', 'Rt', 'confirmed %infected', 'Total Estimated Cases',
                   'daily new cases ma', '2nd_order_growth']
    for i, df in enumerate(df_instances):
        status_tmp = df.reset_index().drop(columns=['name'])
        cpleth_counties_df = pd.merge(us_counties_df, status_tmp, how='left', on='id')
        cpleth_counties_df = cpleth_counties_df.reset_index().sort_values('id', ascending=True).set_index(['state'])
        cpleth_counties_df = cpleth_counties_df[cpleth_cols]
        for col in ['Rt', 'confirmed %infected', 'Total Estimated Cases', 'daily new cases ma', '2nd_order_growth']:
            cpleth_counties_df[col] = cpleth_counties_df[col].apply(
                lambda x: 'Below Threshold' if x == np.inf or x == -np.inf or math.isnan(x) else x)
        merged_df_instances.append(cpleth_counties_df)
    states = merged_df_instances[-1].index.unique('state').dropna()
    return merged_df_instances, states


def build_choropleth_doc(cpleth_df_instances: List[pd.DataFrame], county_date_instances: List,
                         debug_mode: bool = False) -> None:
    df_instances, states = build_cpleth_df(cpleth_df_instances)
    curdoc().clear()
    national_cpleth_sources, cpleth_sources, stateplot, nationalplots = build_cpleth_plot_df(states, df_instances)
    ttldiv = Div(text="""Effective Reproduction Number ( R<sub>t</sub> )""", orientation='vertical',
                 height_policy='min',
                 width_policy='min', css_classes=['cbar_title'], align='center')
    ttldiv.visible = True
    menu = []
    for full_name, state in zip(constants.full_names, states):
        menu.append((state, full_name))
    select = Select(title="", value="Choose", options=menu, name='state_select', css_classes=['bk_select'],
                    width_policy='min', background=None)
    someargs = dict(plot=stateplot, select=select, ttldiv=ttldiv, cpleth_sources=cpleth_sources)
    select_callback = CustomJS(args=someargs, code=constants.select_callback_code)
    select.js_on_change('value', select_callback)
    row_layout = row(stateplot, ttldiv)
    gridlayout = grid([
        [select],
        [row_layout]
    ])
    if not debug_mode:
        for i, natp in enumerate(nationalplots):
            gen_national_imgs(i, natp, county_date_instances)
        script, div = components([gridlayout])
        covid_utils.save_bokeh_tags([div[0], script], config.choro_covid_explorer_tags)
    else:
        show(gridlayout)
