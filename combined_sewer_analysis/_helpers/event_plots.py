from itertools import pairwise

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil, floor

from .legend_helpers import add_custom_legend
from .events import START, span_table, event_duration, END, gap_table
from pandas.tseries.frequencies import to_offset

import warnings
warnings.filterwarnings('ignore',
                        message='.*VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences.*')


def event_line_axes(events, ax, y_bottom, bar_height, color, add_freq_to_duration=None, **kwargs):
    if events.empty:
        return
    ax.broken_barh(list(zip(events[START], event_duration(events, add_freq_to_duration=add_freq_to_duration))),
                   (y_bottom, bar_height),
                   **{**dict(facecolors=color, zorder=2, lw=0), **kwargs})


def add_nan_bar(ax, ts, color='red', label=' NaN', height_ratio=40):
    gaps = gap_table(ts)
    y1 = ax.get_ylim()[1]
    dy = (y1 - ax.get_ylim()[0]) / height_ratio
    event_line_axes(gaps, ax, y1 - dy, y1, color=color)
    ax.axhline(y1 - dy, color='black', lw=1)
    ax.text(ax.get_xlim()[1], y1 - dy / 2, label, ha='left', va='center')


def add_event_marker(ax: plt.Axes, events: pd.DataFrame, color='y', alpha=0.5, label=None, **kwargs):
    _label = label
    poly_collection = None
    for event in events.itertuples():
        poly_collection = ax.axvspan(event.start, event.end, facecolor=color, alpha=alpha, label=_label, **kwargs)
        _label = None
    return poly_collection  # for legend creation


def availability_axes(ts, ax=None, break_points=None, y_bottom=0, bar_height=1, legend=True, add_start_end_text=True, fillna=False, colors=None):
    # Timeseries must have NaNs for gaps
    if break_points is None:
        break_points = [100, 90, 60, 30]

    if colors is None:
        colors = ['#4CAF50', 'yellow', '#FFA500', '#ff5b5d']

    ts_ = ts.copy()
    if ts.index.tz is not None:
        from sww.libs.timeseries.timezone import TZ
        ts_.index = ts_.index.tz_convert(TZ.WINTER).tz_localize(None)
    _res = ts_.resample('1d')
    avail_daily = _res.count() / _res.size() * 100
    del _res

    if fillna:
        avail_daily = avail_daily.fillna(0)

    start, end = avail_daily.index[[0, -1]]

    if ax is None:
        fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
        # fig.set_dpi(90)
        # fig.set_size_inches(12, 2)

        new_figure = True

    else:
        new_figure = False
        fig = ax.get_figure()

    if new_figure or legend:
        pd.Series(index=pd.date_range(start, end, freq='D'), data=np.nan).plot(ax=ax)
        ax.set_xlim(start, end)

    legend_di = {}
    # tabs = []
    for color, (max_, min_) in zip(colors, pairwise(break_points + [-1])):
        legend_di[f'{max_}÷{max(min_, 0)}%'] = dict(lw=8, color=color, solid_capstyle='projecting')
        _tab = span_table((avail_daily <= max_) & (avail_daily > min_))
        event_line_axes(_tab, ax, y_bottom=y_bottom, bar_height=bar_height, color=color,
                        add_freq_to_duration=pd.Timedelta(days=1))
        # _tab['cat'] = f'{max_}÷{min_ if min_ > 0 else 0}%'
        # tabs.append(_tab)

    if legend:
        add_custom_legend(ax, legend_di, bbox_to_anchor=(0, 1, 1, 0),  # bbox (x, y, width, height)
                          loc='lower center', ncol=len(colors), borderaxespad=0., frameon=False, handlelength=0.2)

    # if freq is None:
    #     dur = ts.index[-1] - ts.index[0]
    #     if dur > pd.Timedelta(days=365 * 11):
    #         freq = '1YS'
    #     elif dur < pd.Timedelta(days=365 * 4):
    #         freq = '3MS'
    #     else:
    #         freq = '6MS'
    #
    # print(start, start.replace(day=1, month=ceil(start.month)).floor('D'))
    # print(end, end.replace(day=1, month=floor(end.month)).floor('D'))
    # ax.set_xticks([x for x in pd.date_range(
    #     start.replace(day=1, month=ceil(start.month)).floor('D'),
    #     end.replace(day=1, month=floor(end.month)).floor('D'), freq=freq)])
    # _ = ax.set_xticklabels([pd.to_datetime(x, unit='m').strftime('%Y\n%B') for x in ax.get_xticks()])

    # --------
    # new_years = [d for d in pd.date_range(start, end, freq='YS')]
    # ax.vlines(new_years, 0, 1, ls='--', color='darkgrey', lw=1.5)

    # for x in new_years:
    #     ax.text(x, 1, pd.to_datetime(x, unit='m').strftime('%Y'), ha='center', va='bottom')

    if new_figure:
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.grid()

    if add_start_end_text:
        y_text = y_bottom + bar_height / 2
        ax.text(start, y_text, start.strftime('%Y-%m-%d'), rotation=90, ha='right', va='center')
        ax.text(end, y_text, end.strftime('%Y-%m-%d'), rotation=90, ha='left', va='center')

    return fig, ax


def compare_avail_plot(list_of_timeseries, add_start_end_text=True, fillna=False, avail_lines_only=False, single_color=None):
    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    ax.set_ylim(0, len(list_of_timeseries))
    ax.set_yticks(list(range(len(list_of_timeseries))), minor=False)
    ax.set_yticklabels([], minor=False)
    ax.grid(True, ls=':')

    bar_height = 1
    y_offset = 0
    break_points = None
    colors = None
    add_separate_lines = True
    legend = True
    if avail_lines_only:
        break_points = [100, 75]
        colors = ['black' if single_color is None else single_color, (1, 1, 1, 0)]
        bar_height = 0.5
        y_offset = 0.25
        add_separate_lines = False
        legend = True

    y_ticks=[]
    y_ticklabels = []
    for i, ts in enumerate(list_of_timeseries):
        availability_axes(ts, ax, y_bottom=i+y_offset, bar_height=bar_height, legend=legend and (i == 0), add_start_end_text=add_start_end_text, fillna=fillna, break_points=break_points, colors=colors)
        # ax.text(ts.index[0], i+0.5, ts.name)
        y_ticks.append(i+.5)
        y_ticklabels.append(ts.name)
        #break
    ax.set_yticks(y_ticks, minor=True)
    ax.set_yticklabels(y_ticklabels, minor=True)
    ax.tick_params(axis='y', which='minor', length=0)
    # ax.grid(ls='-', zorder=1110, color='black', axis='y', which='minor', lw=5)
    # ax.grid(ls='-', zorder=1110, color='black', axis='y', which='major', lw=5)
    if add_separate_lines:
        for y in y_ticks[:-1]:
            ax.axhline(y+.5, color='black', lw=0.5, zorder=101, ls='-')
    return fig, ax
