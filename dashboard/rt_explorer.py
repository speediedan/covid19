from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from bokeh.models import ColumnDataSource, CDSView, CustomJS, CustomJSFilter, Patch, Span, TableColumn, DataTable, AutocompleteInput
from bokeh.plotting import Figure
from bokeh.io import show, curdoc
from bokeh.embed import components
from bokeh.layouts import grid
from bokeh.transform import linear_cmap

import config
import c19_analysis.dataprep_utils as covid_utils
import dashboard.dashboard_constants as constants


def build_dashboard_dfs(rt_df: pd.DataFrame, status_df: pd.DataFrame) -> \
        [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Index]:
    # apply min case threshold
    status_df = status_df.loc[(status_df['Total Estimated Cases'] > config.min_case_cnt)]
    primary_ids = status_df.nlargest(30, 'Total Estimated Cases').index.get_level_values('id').tolist()
    # apply min case threshold
    rt_df = rt_df[rt_df['Total Estimated Cases'] > config.min_case_cnt]
    # apply min days over threshold
    rt_df = rt_df.groupby(level='name').filter(lambda x: len(x) >= config.min_days_over)
    main_plot_df = rt_df.loc[rt_df.index.isin(primary_ids, 'id')]
    all_counties_cols = ['Date', 'name', 'Rt', '90_CrI_LB', '90_CrI_UB', 'daily new cases ma', 'Confirmed New Cases',
                           '2nd_order_growth']
    rt_cols = ['Date', 'name', 'Rt', '90_CrI_LB', '90_CrI_UB']
    full_rt_df = rt_df.copy()
    for df in [main_plot_df, full_rt_df]:
        df.reset_index(inplace=True)
    primary_rt_plot_df = main_plot_df.loc[:, rt_cols].set_index(['name', 'Date'])
    all_counties_df = full_rt_df.loc[:, all_counties_cols].set_index(['name', 'Date'])
    counties = all_counties_df.index.unique(level='name')
    counties_df = all_counties_df.reset_index()
    return primary_rt_plot_df, counties_df, main_plot_df, counties


def build_patchsources(counties: pd.Index, counties_df: pd.DataFrame) -> Dict[str, Tuple]:
    patchsources = {}
    for n in counties:
        tmpdf = counties_df.loc[counties_df['name'] == n]
        patchsource = ColumnDataSource(dict(
            dates=np.concatenate((tmpdf['Date'], tmpdf['Date'][::-1])),
            rt=np.concatenate((tmpdf['90_CrI_LB'], tmpdf['90_CrI_UB'][::-1]))
        ))
        start_dt = pd.to_datetime(tmpdf['Date']).min()
        end_dt = pd.to_datetime(tmpdf['Date']).max()
        ymax_rt = (tmpdf['Rt'].mean()) * 3
        max_rt, min_rt = tmpdf['Rt'].max(), tmpdf['Rt'].min()
        patchsources[n] = tuple((patchsource, start_dt, end_dt, ymax_rt, max_rt, min_rt))
    return patchsources


def build_tablesource(status_df: pd.DataFrame) -> ColumnDataSource:
    datatable_cols = ['name', 'Rt', 'confirmed %infected', 'Total Estimated Cases', 'daily new cases ma',
                      'Confirmed New Cases', '2nd_order_growth']
    datatable_df = status_df.copy()
    datatable_df = datatable_df.loc[(datatable_df['daily new cases ma'] > 0) & (datatable_df['growth_period_n'] > 0) &
                                    (datatable_df['Total Estimated Cases'] > 200)]
    datatable_df['2nd_order_growth'] = datatable_df['2nd_order_growth'].apply(lambda x: round(x, 1))
    datatable_df['Rt'] = datatable_df['Rt'].apply(lambda x: round(x, 2))
    datatable_df['Estimated Onset Cases'] = datatable_df['Estimated Onset Cases'].apply(lambda x: round(x))
    datatable_df = datatable_df.reset_index().sort_values('Rt', ascending=False)
    datatable_df = datatable_df[datatable_cols]
    return ColumnDataSource(datatable_df)


def build_countytable(status_df: pd.DataFrame) -> Tuple[DataTable, ColumnDataSource]:
    countytable_cds = build_tablesource(status_df)
    countycolumns = [
        TableColumn(field="name", title="name", width=100),
        TableColumn(field="Rt", title="Rt", default_sort='descending', width=constants.d_col_width),
        TableColumn(field="confirmed %infected", title="Confirmed %Infected", width=constants.d_col_width),
        TableColumn(field="daily new cases ma", title="Est. New Cases Onset", width=constants.d_col_width),
        TableColumn(field="Confirmed New Cases", title="New Cases Confirmed", width=constants.d_col_width),
        TableColumn(field="2nd_order_growth", title="2nd Order Growth (%)", width=constants.d_col_width),
        TableColumn(field="Total Estimated Cases", title="Total Est. Cases", width=constants.d_col_width)
    ]
    countytable = DataTable(source=countytable_cds, columns=countycolumns, fit_columns=True, index_header="Rt Rk.",
                            width_policy='max', height_policy='min', min_height=150, max_height=300, index_width=60,
                            scroll_to_selection=True, css_classes=['bk_datatable'])
    return countytable, countytable_cds


def build_autocomplete_grph_driver(rtplot: Figure, plots: List, ms_plot: Figure, patchsources: Dict[str, Tuple],
                                   source: ColumnDataSource, default_county: str,
                                   counties: pd.Index) -> Tuple[CDSView, AutocompleteInput]:
    choices = AutocompleteInput(completions=counties.tolist(), case_sensitive=False, value=default_county,
                                   title='Search for county or select from table:', name="county_input", width_policy='fit',
                                   css_classes=['autocomplete_input'], min_width=250, align="start")
    someargs = dict(source=source, rtplot=rtplot, rtxaxis=rtplot.xaxis[0], rtyaxis=rtplot.yaxis[0],
                    ms_plot=ms_plot, ms_plot_xaxis=ms_plot.xaxis[0], ms_plot_yaxis0=ms_plot.yaxis[0],
                    plots=plots, choices=choices, patchsources=patchsources)
    someargs['xaxes'], someargs['yaxes'], someargs['plots'] = [], [], []
    for p in plots:
        someargs['xaxes'].append(p['plot'].xaxis[0])
        someargs['yaxes'].append(p['plot'].yaxis[0])
        someargs['plots'].append(p['plot'])

    callback = CustomJS(args=someargs, code=constants.autocomplete_input_code)
    choices.js_on_change('value', callback)
    js_filter = CustomJSFilter(args=dict(choices=choices), code=constants.cdsview_jsfilter_code)
    view = CDSView(source=source, filters=[js_filter])
    return view, choices


def init_rtplot(cust_palette: List, patchsources: Dict[str, Tuple], default_county: str, low: float = 0.5,
                high: float = 1.5) -> Tuple[Figure, Dict]:
    mapper = linear_cmap(field_name='Rt', palette=cust_palette, low=low, high=high, low_color='#33FF33',
                         high_color='#FF3333')
    plot_size_and_tools = {'height_policy': 'fit', 'width_policy': 'fit', 'plot_width': 300, 'plot_height': 250,
                           'min_width': 250, 'max_height': 300, 'min_height': 250, 'border_fill_alpha': 0,
                           'background_fill_alpha': 0, 'margin': 5, 'y_axis_label': 'Effective R',
                           'x_axis_type': 'datetime', 'tools': ['reset', 'zoom_in', 'zoom_out', 'pan']}
    plot = Figure(title='', **plot_size_and_tools)
    # need to pre-build separate dict of per-county patch columndatasources since bokeh doesn't allow patches to be
    # built from cdsviews at the moment
    patch = Patch(x="dates", y="rt", fill_color="#282e54", fill_alpha=0.3, line_alpha=0.1)
    initialcds = ColumnDataSource(dict(dates=np.empty((100, 100)), rt=np.empty((100, 100))))
    plot.add_glyph(initialcds, patch)
    hline = Span(location=1, dimension='width', line_color='#003566', line_alpha=0.8, line_width=3, line_dash='dashed')
    plot.renderers.extend([hline])
    plot.y_range.end = patchsources[default_county][3]
    plot.yaxis.axis_label_text_font_size = "12pt"
    plot.yaxis.axis_label_text_color = '#003566'
    plot.xaxis.axis_label_text_color = '#003566'
    plot.x_range.range_padding = 0
    plot.y_range.range_padding = 0
    plot.xaxis.bounds = (patchsources[default_county][1], patchsources[default_county][2])
    return plot, mapper


def init_simple_plot(target_field: str, y_axlabel: 'str', cust_palette_tup: Tuple = None, hline_loc: float = None,
                     ybounds: Tuple[float] = None) -> Tuple[Figure, Dict]:
    mapper = None
    if cust_palette_tup:
        cust_palette, low, high, low_color, high_color = cust_palette_tup
        mapper = linear_cmap(field_name=target_field, palette=cust_palette, low=low, high=high, low_color=low_color,
                             high_color=high_color)
    plot_size_and_tools = {'height_policy': 'fit', 'width_policy': 'fit', 'plot_width': 300, 'plot_height': 250,
                            'min_width': 250, 'max_height': 300, 'min_height': 250, 'margin': 5, 'border_fill_alpha': 0,
                            'background_fill_alpha': 0, 'y_axis_label': y_axlabel, 'x_axis_type': 'datetime',
                            'tools': ['reset', 'zoom_in', 'zoom_out', 'pan', 'hover']}
    plot = Figure(title='', **plot_size_and_tools)
    if hline_loc:
        hline = Span(location=hline_loc, dimension='width', line_color='blue', line_alpha=0.5, line_width=1,
                     line_dash='dashed')
        plot.renderers.extend([hline])
    if ybounds:
        plot.y_range.start = ybounds[0]
        plot.y_range.end = ybounds[1]
    plot.x_range.range_padding = 0
    plot.y_range.range_padding = 2
    plot.yaxis.axis_label_text_font_size = "12pt"
    plot.yaxis.axis_label_text_color = '#003566'
    plot.xaxis.axis_label_text_color = '#003566'
    return plot, mapper


def multi_series_plot(ybounds: List[Tuple]) -> Tuple[Figure, Optional[Dict]]:
    mapper = None
    plot_size_and_tools = {'height_policy': 'fit', 'width_policy': 'fit', 'plot_width': 300, 'plot_height': 250,
                           'min_width': 250, 'max_height': 300, 'min_height': 250, 'margin': 5, 'border_fill_alpha': 0,
                           'background_fill_alpha': 0, 'y_axis_label': 'Cases', 'x_axis_type': 'datetime',
                           'tools': ['reset', 'zoom_in', 'zoom_out', 'pan', 'hover']}
    plot = Figure(title='', **plot_size_and_tools)
    plot.y_range.start = ybounds[0][0]
    plot.y_range.end = ybounds[0][1]
    plot.x_range.range_padding = 0
    plot.y_range.range_padding = 2
    plot.yaxis.axis_label_text_font_size = "12pt"
    plot.yaxis.axis_label_text_color = '#003566'
    plot.xaxis.axis_label_text_color = '#003566'
    return plot, mapper


def build_dynamic_plots(cust_palette: List, patchsources: Dict[str, Tuple], source: ColumnDataSource,
                        counties: pd.Index, default_county: str) -> Tuple[Figure, List, Figure, AutocompleteInput]:
    rtplot, rtmapper = init_rtplot(cust_palette, patchsources, default_county)
    fields, labels, ttfmt = ['2nd_order_growth'],  ['2nd Order Growth'], ['{0.0}%']
    tups = [tuple((cust_palette, -50, 50, '#33FF33', '#FF3333'))]
    hlines = [0]
    ybounds = [None]
    plots = []
    for (f, l, ttf, tup, h, yb) in zip(fields, labels, ttfmt, tups, hlines, ybounds):
        plot, mapper = init_simple_plot(f, l, tup, h, yb)
        plot_dict = {'plot': plot, 'mapper': mapper, 'field': f, 'ttf': ttf}
        plots.append(plot_dict)
    # add multi_series_plot
    ms_plot, _ = multi_series_plot([(0, None), (0, None)])
    view, choices = build_autocomplete_grph_driver(rtplot, plots, ms_plot, patchsources, source, default_county,
                                                   counties)
    rtplot.circle(x='Date', y='Rt', source=source, view=view, color=rtmapper, size=constants.circle_marker_size,
                  fill_alpha=0.5, line_alpha=0.8, line_color='black')
    for p in plots:
        cmap = p['mapper'] or 'red'
        p['plot'].circle(x='Date', y=p['field'], source=source, view=view, color=cmap,
                         size=constants.circle_marker_size, fill_alpha=0.5, line_alpha=0.8, line_color='black')
        hoverv = f"@{p['field']}{p['ttf']}"
        cust_tooltip_p = constants.cust_tooltip_p_start + hoverv + constants.cust_tooltip_p_end
        p['plot'].hover.tooltips = cust_tooltip_p
    ms_plot.circle(x='Date', y='daily new cases ma', source=source, view=view, color='blue',
                   size=constants.circle_marker_size, fill_alpha=0.5, line_alpha=0.8, line_color='black',
                   legend_label='Est. Cases Onset')
    ms_plot.circle(x='Date', y='Confirmed New Cases', source=source, view=view, color='red',
                   size=constants.circle_marker_size, fill_alpha=0.5, line_alpha=0.8, line_color='black',
                   legend_label='Cases Confirmed')
    ms_plot.legend.location = "top_right"
    ms_plot.legend.background_fill_alpha = 0
    ms_plot.legend.border_line_alpha = 0
    ms_plot.hover.tooltips = constants.cust_tooltip_ms
    return rtplot, plots, ms_plot, choices


def set_tbl_logic(source: ColumnDataSource, choices: AutocompleteInput, patchsources: Dict[str, Tuple],
                  countytable_cds: ColumnDataSource, rtplot: Figure, plots: List[Dict], ms_plot: Figure) -> None:
    someargs = dict(source=source, rtplot=rtplot, rtxaxis=rtplot.xaxis[0], rtyaxis=rtplot.yaxis[0],
                    ms_plot=ms_plot, ms_plot_xaxis=ms_plot.xaxis[0], ms_plot_yaxis0=ms_plot.yaxis[0],
                    choices=choices, countytable_cds=countytable_cds, patchsources=patchsources)
    someargs['xaxes'], someargs['yaxes'], someargs['plots'] = [], [], []
    for p in plots:
        someargs['xaxes'].append(p['plot'].xaxis[0])
        someargs['yaxes'].append(p['plot'].yaxis[0])
        someargs['plots'].append(p['plot'])
    tblcallback = CustomJS(args=someargs, code=constants.tblcallback_code)
    countytable_cds.selected.js_on_change('indices', tblcallback)
    # set additional callback on autocompeleteinput linking table row selected
    someargs = dict(source=source, choices=choices, countytable_cds=countytable_cds)
    inputtbl_callback = CustomJS(args=someargs, code=constants.autocomplete_in_tbl_code)
    choices.js_on_change('value', inputtbl_callback)


def set_plots_blank(plots: List[Figure]) -> None:
    for plot in plots:
        # ensure initial display is blank
        plot.xaxis.visible = False
        plot.yaxis.visible = False
        for r in plot.renderers:
            r.visible = False
        plot.toolbar.logo = None


def build_dashboard_doc(rt_df: pd.DataFrame, status_df: pd.DataFrame, debug_mode: bool = False) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Index]:
    # define core document objects
    curdoc().clear()
    primary_rt_plot_df, counties_df, main_plot_df, counties = build_dashboard_dfs(rt_df, status_df)
    default_county = counties.tolist()[0]
    patchsources = build_patchsources(counties, counties_df)
    countytable, countytable_cds = build_countytable(status_df)
    rtsource = ColumnDataSource(counties_df)
    rtplot, plots, ms_plot, choices = build_dynamic_plots(constants.cust_rg_palette, patchsources, rtsource, counties,
                                                          default_county)
    set_tbl_logic(rtsource, choices, patchsources, countytable_cds, rtplot, plots, ms_plot)
    # set initial default
    plots = [p['plot'] for p in plots]
    plots.extend([rtplot, ms_plot])
    set_plots_blank(plots)
    gridlayout = grid([
        [None, None, choices],
        [plots[1], plots[0], plots[2]],
        [countytable]
    ])
    if not debug_mode:
        script, div = components([gridlayout])
        covid_utils.save_bokeh_tags([div[0], script], config.county_covid_explorer_tags)
    else:
        show(gridlayout)
    return primary_rt_plot_df, counties_df, main_plot_df, counties
