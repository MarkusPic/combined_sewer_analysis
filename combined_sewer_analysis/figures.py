import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.colors as mcolors
from scipy.stats import norm

from ._helpers.event_plots import add_event_marker, event_line_axes
from ._helpers.legend_helpers import add_custom_legend, get_legend_dict
from ._helpers.axes_formatting import diurnal_axes, weekly_x_axes
from ._helpers.events import filter_events, span_table, event_duration
from ._helpers.pivot_tables import compare_week_table, compare_daily_times_table
from ._helpers.debug_helpers import check
from ._helpers.plot_helpers import daykind_color, XLABEL_DIURNAL

from ._class import AnalyseData, L
from ._helpers.calculation_helpers import calc_dry_mean, calc_dry_variation, _calc_dry_mean, mad
from .date_analysis import get_school_holidays, get_holidays, HOLIDAY, FAKE_FRIDAY, BRIDGE_DAY, DAY_KIND
from .definitions import MAD_TO_STD


########################################################################################################################
def diurnal_density(data: AnalyseData, day_series: pd.Series, ylim=None, ylab=None, smooth=20, show_rain=None, unit=None, title=None,
                    major_freq='h', minor_freq='15min', rasterized=True, alpha=0.05, ax=None) -> tuple[plt.Figure, plt.Axes]:
    # Statistical Tests for Normality
    # from scipy.stats import *
    # https://towardsdatascience.com/normality-tests-in-python-31e04aa4f411

    day = day_series.name

    s = day_series.resample('5min').mean()

    if isinstance(show_rain, pd.Series):
        crit = show_rain.copy()

        dry = s.copy()
        dry.loc[crit > 100] = np.nan
        dry.name = 'DRY_{}'.format(s.name)

        dry.dropna(inplace=True)
        data_table_dry = compare_daily_times_table(dry)
        ax = data_table_dry.T.plot(alpha=alpha, legend=False, color='black', rasterized=rasterized, ax=ax)

        wet = s[s.index.difference(dry.index)]
        data_table_wet = compare_daily_times_table(wet)
        ax = data_table_wet.T.plot(alpha=alpha, legend=False, color='cyan', ax=ax, rasterized=rasterized)
    else:
        data_table = compare_daily_times_table(s)
        ax = data_table.T.plot(alpha=alpha, legend=False, color='black', rasterized=rasterized, ax=ax)

    ax = diurnal_axes(ax, major_freq=major_freq, minor_freq=minor_freq)

    ax.set(ylim=ylim)

    if title:
        # title = make_title(title,
        #                    default=get_diurnal_title(name=data.name,
        #                                              day=s.name,
        #                                              kind=data.arithmetic,
        #                                              limit=data.limit,
        #                                              smooth=smooth))

        ax.set_title(title, fontsize=12, fontweight='bold')

    # if ylab:
    #     ax.set_ylabel(cst_label(data.name, unit=unit))

    # ax = data.get_dw_mean_table(smooth=smooth)[day].plot(ax=ax, color=daykind_color(s.name))

    m = ax.plot(data.get_dw_mean_table(smooth=smooth)[day], color=daykind_color(s.name))

    bounds = data.get_dw_bound_table_v2(smooth=smooth)

    upper, lower = bounds[L.UPPER], bounds[L.LOWER]

    bw = ax.fill_between(data.get_dw_mean_table(smooth=smooth).index,
                    lower[day],
                    upper[day],
                    alpha=.2, color=daykind_color(s.name))

    # add_custom_legend(ax, {'DW Bandwidth': bw})
    ax.legend(m + [bw], ['DW mean', 'DW bandwidth'])

    # ax.get_figure().show()
    # ax.legend().remove()
    return ax.get_figure(), ax


########################################################################################################################
def diurnal_density2(day_series: pd.Series, data: AnalyseData, dry_data=None, smooth=20, criterion=None, unit=None, no_calc=False,
                     down_scale='5T', add_bounds=True, title=None, ylim=None, two_label_lines=True, ylabel=None,
                     split=True, middle_y=None, rasterized=True) -> plt.Figure:
    day = day_series.name
    # if day != 'Saturday':
    #     return
    check(day)

    if middle_y is None:
        middle_y = day_series.median() * 3

    if add_bounds:
        upper = data.get_dw_bound_table(smooth=smooth)[L.UPPER]
        middle_y = max([middle_y, upper.max().max()]) * 1.1

    s = day_series.resample(down_scale).mean()

    if split:
        fig, (ax2, ax) = plt.subplots(2, 1, sharex=True, )
    else:
        fig, ax = plt.subplots(1, 1)
        ax2 = None

    if isinstance(criterion, pd.Series):
        crit = criterion.copy()

        dry = s.copy()
        dry.loc[crit > 100] = np.nan
        dry.name = 'DRY_{}'.format(s.name)

        dry.dropna(inplace=True)
        data_table_dry = compare_daily_times_table(dry)
        ax = data_table_dry.T.plot(alpha=0.15, legend=False, color='saddlebrown', lw=1, ax=ax, rasterized=rasterized)
        if split:
            ax2 = data_table_dry.T.plot(alpha=0.15, legend=False, color='saddlebrown', lw=1, ax=ax2, rasterized=rasterized)

        wet = s[s.index.difference(dry.index)]
        data_table_wet = compare_daily_times_table(wet)
        ax = data_table_wet.T.plot(alpha=0.15, legend=False, color='c', ax=ax, lw=1, rasterized=rasterized)
        if split:
            ax2 = data_table_wet.T.plot(alpha=0.15, legend=False, color='c', ax=ax2, lw=1, rasterized=rasterized)
        alpha = 0.3
    else:
        data_table = compare_daily_times_table(s)
        ax = data_table.T.plot(alpha=0.05, legend=False, color='black', ax=ax)
        if split:
            ax2 = data_table.T.plot(alpha=0.05, legend=False, color='black', ax=ax2)
        alpha = 0.2

    # title = make_title(title, default=get_diurnal_title(name=data.name, day=s.name))

    ax = diurnal_axes(ax, title=title, xlabel=XLABEL_DIURNAL)
    if split:
        ax2 = diurnal_axes(ax2, title=title, xlabel=XLABEL_DIURNAL)

    if not no_calc:
        ax = data.get_dw_mean_table(smooth=smooth)[day].plot(ax=ax, color=daykind_color(s.name), lw=2)
        agg_dry_bound = data.get_dw_bound_table(smooth=smooth)

        if add_bounds:
            ax.fill_between(data.get_dw_mean_table(smooth=smooth).index,
                            agg_dry_bound[(L.LOWER, day)],
                            agg_dry_bound[(L.UPPER, day)],
                            alpha=alpha, color=daykind_color(s.name), lw=2)

        if isinstance(dry_data, AnalyseData):
            ax = dry_data.get_dw_mean_table(smooth=smooth)[day].plot(ax=ax, color='yellow', linestyle='dotted', lw=2)
            if add_bounds:
                agg_dry_bound = data.get_dw_bound_table(smooth=smooth)

                ax = agg_dry_bound[(L.UPPER, day)].plot(ax=ax, color='yellow', linestyle='dashed', lw=1)
                ax = agg_dry_bound[(L.LOWER, day)].plot(ax=ax, color='yellow', linestyle='dashed', lw=1)

            # ax.fill_between(dry_data.agg_dry_mean(smooth=smooth).index,
            #                 lower[day],
            #                 upper[day],
            #                 alpha=.4, color='yellow')

    if split:
        ax2.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax2.xaxis.tick_top()
        ax2.tick_params(labeltop=False)  # don't put tick labels at the top
        # ax.set_frame_on(True)

        # ax.set_yticks(list(ax.get_yticks()) + [middle_y])
        ax2.set_yticks(list(ax2.get_yticks()) + [middle_y])
        ax2.set_ylim(bottom=middle_y)
        ax.set_ylim(bottom=0, top=middle_y)
        # ax.axhline(middle_y, color='black')
        # ax2.axhline(middle_y, color='black')
        ax2.set_ylabel(cst_label(data.name, unit=unit, two_lines=two_label_lines), color='white')

        ax2 = translate_ax(ax2, lang=lang)
    else:
        ax.set_ylim(ylim)
        ax.set_ylabel(cst_label(data.name, unit=unit, two_lines=two_label_lines, lang=lang))

    ax = translate_ax(ax, lang=lang)

    if split:
        fig.subplots_adjust(hspace=0.05)
        # ylabel
        fig.text(0.04, 0.5, cst_label(data.name, unit=unit, two_lines=two_label_lines, lang=lang), va='center',
                 rotation='vertical',
                 ha='center', size=plt.rcParams['axes.labelsize'], color=plt.rcParams['axes.labelcolor'])

    return fig


def diurnal_density_full(ts: pd.Series, major_freq='1h', minor_freq='15min', alpha=0.05, major_fmt='%-H', rasterized=True) -> tuple[plt.Figure, plt.Axes]:
    data_table = compare_daily_times_table(ts)
    ax = data_table.T.plot(alpha=alpha, legend=False, color='black', rasterized=rasterized)
    ax = diurnal_axes(ax, major_freq=major_freq, minor_freq=minor_freq, major_fmt=major_fmt)
    return ax.get_figure(), ax


def diurnal_density_full_heatmap(ts: pd.Series, major_freq='1h', minor_freq='15min', ymin=0, ymax=100, ybins=100):
    fig, ax = plt.subplots()
    _ = ax.hist2d(ts.index.hour * 60*60 + ts.index.minute*60, ts, norm=mcolors.LogNorm(), cmap='Greys',
                  range=[[xmin := 0, xmax := 1440*60], [ymin := ts.min(), ymax := ts.max()]],
                  bins=[288, ybins]
                  )
    ax.set_ylim(int(ts.min()), int(ts.max()))

    fig.colorbar(ax.collections[0], ax=ax, location='right',
                 label='Count', pad=0.01, shrink=0.5,  # aspect=20
                 # ticks=[0,1,2,3,4,5]
                 )
    ax.set_xticks(range(0, 1441*60, int(pd.Timedelta(major_freq) / pd.Timedelta(seconds=1))))
    # major_ticks = pd.date_range("00:00", "23:59", freq=major_freq).append(pd.DatetimeIndex(['23:59:59.999999']))
    # ax.set_xticklabels(major_ticks)

    from idf_analysis.little_helpers import duration_steps_readable
    ax.set_xticklabels(duration_steps_readable(ax.get_xticks()/60))

    ax.set_xticks(range(0, 1440*60, int(pd.Timedelta(minor_freq) / pd.Timedelta(seconds=1))), minor=True)
    return fig, ax


def weekly_density_plot(data: AnalyseData, ax=None, add_mean=True, color='black', smooth=None) -> tuple[plt.Figure, plt.Axes]:
    li_weekdays = list(calendar.day_name)
    ts_normal_week = data.ts[np.isin(data.get_day_category_index(level_of_detail=10, add_number=False), li_weekdays)].copy().dropna()
    data_table = compare_week_table(ts_normal_week, sunday_first=False)
    ax = data_table.T.plot(alpha=0.05, legend=False, color=color, ax=ax, rasterized=True)

    ax.set_title('Weekly')

    if add_mean:
        data7_mean = data.get_dw_mean_table(smooth=None).copy()
        # for c in (FAKE_FRIDAY, BRIDGE_DAY, HOLIDAY,
        #           f'5.1 {FAKE_FRIDAY}', f'6.1 {BRIDGE_DAY}', f'8 {HOLIDAY}'):
        #     if c in data7_mean:
        #         del data7_mean[c]

        # if '8 Holiday' in data7_mean:
        #     del data7_mean['8 Holiday']
        # if 'Holyday' in data7_mean:
        #     del data7_mean['Holiday']

        # if '7 Sunday' in data7_mean:
        #     data7_mean['0 Sunday'] = data7_mean['7 Sunday']
        #     del data7_mean['7 Sunday']

        # data7_mean = data7_mean.sort_index(axis=1)

        data7_mean2 = data7_mean.copy()
        data7_mean2.index = pd.to_timedelta([i.hour for i in data7_mean2.index], unit='h') + \
                            pd.to_timedelta([i.minute for i in data7_mean2.index], unit='m')

        data7_mean2 = data7_mean2[li_weekdays]

        data7_mean2.columns = pd.timedelta_range(start=pd.Timedelta(0), end=pd.Timedelta(days=6), freq='d')

        meas = data7_mean2.stack(0).sort_index(level=1)
        meas.index = meas.index.get_level_values(0) + meas.index.get_level_values(1)
        if smooth is not None:
            meas = meas.rolling(smooth).mean()
        ax = meas.rename('Mittelwert').plot(ax=ax, color='red', lw=2, legend=True)

    ax = weekly_x_axes(ax, sunday_first=False)  # , custom_switch_time='00:00', start = data7_mean.index[0], data7_mean.index[-1])
    # ylim = get_ylim(data.ts)
    # ax.set_ylim(ylim)
    return ax.get_figure(), ax


def weekly_density_plot_heatmap(data: AnalyseData, ax=None, add_mean=True, cmap='Greys', smooth=None, xbins=500, ybins=100, ymax=100, add_colorbar=False, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    li_weekdays = list(calendar.day_name)
    ts_normal_week = data.ts[np.isin(data.get_day_category_index(level_of_detail=10, add_number=False), li_weekdays)].copy().dropna()
    ts_normal_week = ts_normal_week[ts_normal_week < ymax]
    sec = ts_normal_week.index.weekday*24*60*60 + ts_normal_week.index.hour*60*60 + ts_normal_week.index.minute*60 + ts_normal_week.index.second

    if ax is None:
        fig, ax = plt.subplots()

    _ = ax.hist2d(sec*1e9, ts_normal_week, norm=mcolors.LogNorm(), cmap=cmap,
                  range=[[xmin:=0, xmax:=1440*60*7*1e9], [ymin:=ts_normal_week.min(), ymax:=ts_normal_week.max()]],
                  bins=[xbins, ybins]
                  )
    ax.set_ylim(int(ts_normal_week.min()), int(ts_normal_week.max()))

    if add_colorbar:
        fig.colorbar(ax.collections[0], ax=ax, location='right',
                     label='Count', pad=0.01, shrink=0.5,  # aspect=20
                     # ticks=[0,1,2,3,4,5]
                     )

    # ax.set_title('Weekly')

    if add_mean:
        data7_mean = data.get_dw_mean_table(smooth=None).copy()

        data7_mean2 = data7_mean.copy()
        data7_mean2.index = pd.to_timedelta([i.hour for i in data7_mean2.index], unit='h') + \
                            pd.to_timedelta([i.minute for i in data7_mean2.index], unit='m')

        data7_mean2 = data7_mean2[li_weekdays]

        data7_mean2.columns = pd.timedelta_range(start=pd.Timedelta(0), end=pd.Timedelta(days=6), freq='d')

        meas = data7_mean2.stack(0).sort_index(level=1)
        meas.index = meas.index.get_level_values(0) + meas.index.get_level_values(1)
        meas.index = meas.index.total_seconds()*1e9
        if smooth is not None:
            meas = meas.rolling(smooth).mean()
        ax = meas.rename('DW mean').plot(ax=ax, color='red', lw=2, legend=True)

    ax = weekly_x_axes(ax, sunday_first=False, **kwargs)  # , custom_switch_time='00:00', start = data7_mean.index[0], data7_mean.index[-1])

    return ax.get_figure(), ax


########################################################################################################################
def stability_analysis(data: AnalyseData, arithmetic=None, var=False) -> (plt.Figure, plt.Axes):
    """How much data is needed to achieve similar results as with all available data?"""
    """every day is thrown into one pit. -> not good"""
    if arithmetic is None:
        arithmetic = data.arithmetic

    cmap = plt.get_cmap('cividis')
    norm = mcolors.Normalize(vmin=1, vmax=3)
    color = cmap(norm(1))

    # ---
    res = {}
    for (day_kind, timestamp), series in data.analysis_grouper():
        # nur für volle Stunden
        if timestamp.minute != 0:
            continue

        s = series.dropna()

        res_day_kind = []

        final = calc_dry_mean(s, arithmetic)

        if var:  # make plot for variation -> MAD
            final = calc_dry_variation(s - final, kind=arithmetic)

        for seed in range(5):
            # random sorting of values using 5 seeds
            values_random = s.sample(frac=1, random_state=seed).values

            res_seed = {}

            for n_values in range(5, s.size, 2):
                # n_values = portion of the random values (minimum:5, step=2)
                values_random_pick = values_random[:n_values]

                recent = _calc_dry_mean(values_random_pick, arithmetic)
                if var:
                    recent = calc_dry_variation(values_random_pick - recent, arithmetic)

                res_seed[n_values] = (recent - final) / final

            res_day_kind.append(res_seed)
        res[(day_kind, timestamp)] = res_day_kind

    # ---
    time_dist = pd.concat([pd.DataFrame.from_records(r).rename_axis(index='random_state', columns='n_values').T.assign(day_kind=day_kind, timestamp=timestamp).set_index(['day_kind', 'timestamp'], append=True).unstack().unstack() for (day_kind, timestamp), r in res.items()], axis=1)

    # ---
    upper_ranges = time_dist.quantile([0.995, 0.975, 0.95], axis=1).T
    upper_ranges = upper_ranges.rolling(4, min_periods=1, center=True).median().rolling(5, min_periods=1,
                                                                                        center=True).mean()
    upper_ranges.columns = ['99%', '95%', '90%']

    lower_ranges = time_dist.quantile([0.05, 0.025, 0.005], axis=1).T
    lower_ranges = lower_ranges.rolling(4, min_periods=1, center=True).median().rolling(5, min_periods=1,
                                                                                        center=True).mean()

    ax: plt.Axes = upper_ranges.plot(color=[cmap(norm(i)) for i in range(3, 0, -1)], legend=True)
    ax.legend(title='Uncertainty')
    lower_ranges.plot(color=[cmap(norm(i)) for i in range(1, 4)], legend=False, ax=ax)

    # ---
    title = f'DW mean stability analysis\n[{arithmetic=}]'

    if var:
        title = title.replace('Mean', 'Variation')

    ax.set_title(title, fontsize=12, fontweight='bold')

    # ---
    ax.set_xlim(left=0)
    ax.set_ylabel('Deviation to result for full dataset')
    ax.set_xlabel('Sample size')

    ax.axhline(0, lw=0.7, color='black', zorder=0)

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # ax.get_figure().show()

    return ax.get_figure(), ax


########################################################################################################################
def compare_day(data: AnalyseData, smooth=20, unit=None, add_bounds=True, title=None, two_lines=True, major_freq='h', minor_freq='15min', major_fmt='%-H') -> tuple[plt.Figure, plt.Axes]:
    """

    Args:
        data:
        smooth:
        unit:
        add_bounds:
        title (str): title of the plot
        two_lines: for ylabel name and abbr+unit in 1 oder 2 lines

    Returns:
        plt.Figure:
    """
    mean = data.get_dw_mean_table(smooth=smooth)
    if add_bounds:
        agg_dry_bound = data.get_dw_bound_table(smooth=smooth)

    ax = None
    for day in DAY_KIND.sorter(mean.columns):
        print(day)
        ax = mean[day].plot(ax=ax, color=daykind_color(day, data.day_categorization), legend=True)
        if add_bounds:
            ax.fill_between(mean.index,
                            agg_dry_bound[(L.LOWER, day)],
                            agg_dry_bound[(L.UPPER, day)],
                            alpha=.25, color=daykind_color(day), lw=0)

    # ylim = get_ylim(data.ts)

    # title = make_title(title, default=get_compare_diurnal_title(name=data.name))
    # title = get_compare_diurnal_title(name=data.name, kind=data.arithmetic, limit=data.limit, smooth=smooth)
    # ax.set_title(title)
    ax = diurnal_axes(ax, major_freq=major_freq, minor_freq=minor_freq, major_fmt=major_fmt)
    # ax.set_ylabel(cst_label(data.name, unit=unit, two_lines=two_lines))
    # ax.set_ylim(ylim)
    return ax.get_figure(), ax


def compare_all_days(data: AnalyseData, smooth=None, major_freq='h', minor_freq='15min', ax=None):
    data10 = AnalyseData(data.ts, limit=data.limit, kind=data.arithmetic, day_categorization=10, smooth_window=data.smooth_window)

    mean = data10.get_dw_mean_table(smooth=data.smooth_window if smooth is None else smooth)

    for day in DAY_KIND.sorter(mean.columns):
        ax = mean[day].plot(color=daykind_color(day), legend=True, ax=ax)

    ax = diurnal_axes(ax, major_freq=major_freq, minor_freq=minor_freq)
    return ax.get_figure(), ax


########################################################################################################################
def dry_percentage(data: AnalyseData, unit=None, title=None):
    from scipy.stats import percentileofscore

    def calc_dry_perc(s):
        day, day_time = s.name
        upper = percentileofscore(s, data.upper_bound.loc[day_time, day])
        lower = percentileofscore(s, data.upper_bound.loc[day_time, day])
        return upper - lower

    res = data.analysis_grouper().apply(calc_dry_perc).unstack().T
    ax = res.plot()

    # title = make_title(title, default=get_compare_diurnal_title(name=data.name, kind=data.arithmetic, limit=data.limit))
    # ax.set_title(title)
    ax = diurnal_axes(ax)
    # ax.set_ylabel(cst_label(data.name, unit=unit))
    ax.set_ylim(0, 100)
    return ax.get_figure()


########################################################################################################################
def dry_trend(data: AnalyseData, smooth_window=pd.Timedelta(days=2), color=None, label='Dry-Weather Level',
              title=None, mark_holidays_school=False, mark_holidays_business=False, mark_gaps=False):
    _g = data.get_criterion_level_series(smooth_window=smooth_window).resample(smooth_window)
    level = _g.mean()
    level = level[_g.count() > 0][level != 0].asfreq(level.index.freq)
    ax = level.rename('DW level').plot(color=color)
    ax.set_xlim(level.index[0], level.index[-1])

    if mark_holidays_school:
        school_free = get_school_holidays().rename(columns={'Beginn': 'start', 'Ende': 'end'})
        school_free['start'] = school_free['start'].dt.tz_localize(level.index.tz)
        school_free['end'] = school_free['end'].dt.tz_localize(level.index.tz)
        school_free = filter_events(school_free, level.index[0], level.index[-1])

        school_free = school_free[event_duration(school_free) > pd.Timedelta(days=5)]

        school_holidays = school_free[school_free['Bemerkungen'] != 'Covid-19']
        add_event_marker(ax, school_holidays, '#e6e6a1', 0.7, 'School holiday')

        covid_free = school_free[school_free['Bemerkungen'] == 'Covid-19']
        add_event_marker(ax, covid_free, 'lightgray', 0.7, 'Covid-19')

    if mark_holidays_business:
        hd = get_holidays(list(range(level.index[0].year, level.index[-1].year)), state='ST')
        _label = 'Business holiday'
        for day, name in hd.items():
            ax.axvspan(day, day + pd.Timedelta(days=1, seconds=-1), color='red', alpha=0.6, label=_label)
            _label = None

    ax.axhline(0, color='black', linewidth=0.7)
    ax.axhline(100, color='darkgray', linewidth=0.7)
    ax.axhline(-100, color='darkgray', linewidth=0.7)

    if mark_gaps:
        y0, y1 = ax.get_ylim()
        y_max = y1
        dy = (y1 - y0) / 30
        nan_event = span_table(data.ts.isna().resample('7d').mean() > 0.5)
        event_line_axes(nan_event, ax, y1, dy, color='grey', label='Gaps')
        ax.axhline(y1, color='black', linewidth=0.7)
        # ax.text(ax.get_xlim()[0], y1+dy/2, 'Gaps  ', va='center_baseline', ha='right')
        ax.set_ylim(y0, y_max + dy)

    ax.set_ylabel(label)
    ax.set_xlabel('')
    ax.legend(handlelength=1.5)
    # ax.set_title(make_title(title, default=cst_label(data.name, unit=False)))
    # ax = translate_ax(ax, lang=lang)
    return ax.get_figure()


########################################################################################################################
def diurnal_uncertainty_density(data: AnalyseData, day_series, smooth=20, ylim=None,
                                major_freq='h', minor_freq='15min', rasterized=True) -> (plt.Figure, plt.Axes):
    day = day_series.name

    # ------------
    dw_bool_full = data.get_dw_bool_series(fill_na=False)
    dw_bool_day = dw_bool_full[day_series.index]
    ts_day_dw = day_series[dw_bool_day]

    dw_cont_day = data.get_dw_continuum_series()[day_series.index]
    dw_cont_day_dw = dw_cont_day[dw_bool_day]
    diff_day = ts_day_dw - dw_cont_day_dw

    dw_residuals_series_full = data.get_dw_residual_series(dw_bool_full)
    dw_uncertainty_table = data.get_dw_uncertainty_table()
    dw_residuals_series_day = dw_residuals_series_full[ts_day_dw.index]

    # ------------
    data_table = compare_daily_times_table(diff_day)
    ax = data_table.T.plot(alpha=0.05, legend=False, color='black', label='_nolegend_', rasterized=rasterized)
    # ax.legend().remove()
    # ------------
    std_raw = data_table.std()
    std = std_raw.rolling(smooth, center=True, min_periods=1).mean()
    interval_68 = std
    interval_95 = std * 2
    interval_99 = std * 3

    ax.plot(interval_68.index, -interval_68, color='orange', ls='--', lw=0.75, label=r'68.3% (1 $\sigma$)')
    ax.plot(interval_68.index, interval_68, color='orange', ls='--', lw=0.75)

    ax.plot(interval_95.index, -interval_95, color='red', ls='--', lw=0.75, label=r'95.4% (2 $\sigma$)')
    ax.plot(interval_95.index, interval_95, color='red', ls='--', lw=0.75)

    ax.plot(interval_99.index, -interval_99, color='darkviolet', ls='--', lw=0.75, label=r'99.7% (3 $\sigma$)')
    ax.plot(interval_99.index, interval_99, color='darkviolet', ls='--', lw=0.75)

    # ------------
    if ylim is None:
        ylim = interval_99.max()
    ax.set_ylim(-ylim, ylim)

    # ------------
    ax = diurnal_axes(ax, major_freq=major_freq, minor_freq=minor_freq)  # , ylab=cst_label(data.name, unit=unit), title=title)
    ax.legend()

    lines_dict_ = get_legend_dict(ax)
    lines_dict_ = {k: v for k, v in lines_dict_.items() if not k.startswith('20')}
    add_custom_legend(ax, lines_dict_, title='Confidence interval', bbox_to_anchor=(0, 0, 1, 0), loc='lower left', ncol=3, )

    # ------------
    ax.set_title(f'{day} - Uncertainty')

    # ax.get_figure().show()
    return ax.get_figure(), ax


########################################################################################################################
def compare_dw_uncertainty_day_absolute(data: AnalyseData, smooth=20,
                                        major_freq='h', minor_freq='15min', unit='L/s') -> tuple[plt.Figure, plt.Axes]:

    uncertainty = data.get_dw_uncertainty_table(smooth=smooth)

    # ---
    ax: plt.Axes = None
    for day in DAY_KIND.sorter(uncertainty.columns):
        # print(day)
        ax = uncertainty[day].plot(ax=ax, color=daykind_color(day), legend=True)

    ax = diurnal_axes(ax, major_freq=major_freq, minor_freq=minor_freq)

    ax.set_title('absolute Uncertainty')

    return ax.get_figure(), ax


def compare_dw_uncertainty_day_relative(data: AnalyseData, smooth=20,
                                        major_freq='h', minor_freq='15min') -> tuple[plt.Figure, plt.Axes]:

    mean = data.get_dw_mean_table(smooth=smooth)
    uncertainty = data.get_dw_uncertainty_table(smooth=smooth)

    uncertainty_rel = uncertainty / mean

    # ---
    ax: plt.Axes = None
    for day in DAY_KIND.sorter(uncertainty_rel.columns):
        # print(day)
        ax = uncertainty_rel[day].plot(ax=ax, color=daykind_color(day), legend=True)

    ax = diurnal_axes(ax, major_freq=major_freq, minor_freq=minor_freq)

    ax.set_title('relative Uncertainty')

    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    return ax.get_figure(), ax


def _single_timestamp_distribution(day_kind, timestamp, series, dw_bool, bin_width=5, set_title=True):
    import seaborn as sns
    from mp.libs import fitter
    import math
    # ---
    fig, (ax_dry, ax_wet) = plt.subplots(1, 2)  # type: plt.Figure, (plt.Axes, plt.Axes)

    series_dw = series[dw_bool].copy()
    bins = np.arange(math.floor(series_dw.min() / bin_width) * bin_width,
                     math.ceil(series_dw.max() / bin_width) * bin_width + bin_width, bin_width)
    # series_dw.hist(bins=bins, density=True, ax=ax_dry)

    sns.histplot(x=series_dw, kde=True, ax=ax_dry, bins=bins)
    # sns.kdeplot(data=tips, x="total_bill")
    sns.rugplot(x=series_dw, ax=ax_dry)

    # f = Fitter(deviations[day_kind],
    #            xmin=-abs_range, xmax=abs_range, bins=40,
    #            distributions=['chi2', 'lognorm', 'norm', ], timeout=60)
    # f.fit()
    # ax.plot(f.x, f.fitted_pdf['lognorm'], lw=1, label='lognorm', color='black')

    series_ww = series.loc[~dw_bool].copy()

    if not series_ww.empty:
        bins = np.arange(math.floor(series_ww.min() / bin_width) * bin_width,
                         math.ceil(series_ww.max() / bin_width) * bin_width + bin_width, bin_width)
        # series_ww.hist(bins=bins, density=True, ax=ax_wet)
        sns.histplot(x=series_ww, kde=True, ax=ax_wet, bins=bins)
        # sns.kdeplot(x=series_dw, ax=ax_wet, color='r')
        # sns.kdeplot(data=tips, x="total_bill")
        # sns.rugplot(x=series_ww, ax=ax_wet)

    # ---
    # stats
    mean_dw = series_dw.mean()
    std_dw = series_dw.std()
    median_all = series.median()
    mad_all = mad((series-median_all).dropna())
    std_robust_all = mad_all/MAD_TO_STD
    min_dw = series_dw.min()
    max_dw = series_dw.max()

    # ---
    # dw-border
    min_ww = series_ww.min()
    max_ww = series_ww.max()

    # ---
    # normal distribution plot with only DW data
    x = np.linspace(min_dw, max_dw, 100)
    p = norm.pdf(x, mean_dw, std_dw)
    upper_lim = ax_dry.get_ylim()[1]
    p = p/p.max()*upper_lim*0.9
    ax_dry.plot(x, p, 'red', lw=2)
    # ax_dry.axvline(mean_dw, color='r')

    # ---
    # normal distribution plot with robust metrics
    p = norm.pdf(x, median_all, std_robust_all)
    p = p/p.max()*ax_dry.get_ylim()[1]*0.9
    ax_dry.plot(x, p, 'goldenrod', lw=2)
    # ax_dry.axvline(median_all, color='goldenrod')

    # ---
    # 1,2,3 times standard deviation

    # for dw-metrics
    ax_dry.vlines([mean_dw + i*std_dw for i in range(-3, 4)], upper_lim*0.95, upper_lim, color='red')
    ax_dry.vlines(mean_dw, upper_lim*0.95, upper_lim, color='red', lw=2)
    # for robust-metrics
    ax_dry.vlines([median_all + i * std_robust_all for i in range(-3, 4)], upper_lim * 0.95, upper_lim, color='goldenrod')
    ax_dry.vlines(median_all, upper_lim * 0.95, upper_lim, color='goldenrod', lw=2)

    # ---
    ax_dry.set_ylim(top=upper_lim)
    if set_title:
        ax_dry.set_title(f'Dry\nn={series_dw.size}\n[{min_dw:0.0f} ... {max_dw:0.0f}]')

        ax_wet.set_title(f'Wet\nn={series_ww.size}\n[{min_ww:0.0f} ... {max_ww:0.0f}]')

        fig.suptitle(f'{day_kind} | {timestamp}\n'
                     f'$\\bar{{x}}_{{dw}}$={mean_dw:0.1f} | $\\sigma_{{dw}}$={std_dw:0.1f}\n'
                     f'median$\\hat{{x}}_{{all}}$={median_all:0.1f} | $\\sigma_{{robust,all}}$={std_robust_all:0.1f}')
    else:
        ax_dry.set_title('Dry weather')
        ax_wet.set_title('Wet weather')
        print(f'Dry\nn={series_dw.size}\n[{min_dw:0.0f} ... {max_dw:0.0f}]')

        print(f'Wet\nn={series_ww.size}\n[{min_ww:0.0f} ... {max_ww:0.0f}]')

        print(f'{day_kind} | {timestamp}\n'
                     f'$\\bar{{x}}_{{dw}}$={mean_dw:0.1f} | $\\sigma_{{dw}}$={std_dw:0.1f}\n'
                     f'median$\\hat{{x}}_{{all}}$={median_all:0.1f} | $\\sigma_{{robust,all}}$={std_robust_all:0.1f}')

    # fig.show()
    return fig


########################################################################################################################
AnalyseData.figure_diurnal_density = diurnal_density
AnalyseData.figure_compare_day = compare_day
AnalyseData.figure_dry_percentage = dry_percentage
AnalyseData.figure_dry_trend = dry_trend
AnalyseData.figure_stability_analysis = stability_analysis
