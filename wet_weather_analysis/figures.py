import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mp.projects.cst_monitoring.data_analysis.plots._helpers import args_to_string, get_compare_diurnal_title, get_diurnal_title, make_title
from mp.projects.cst_monitoring.misc.plot_helpers import daykind_color, get_ylim, cst_label, diurnal_xlabel, translate_ax, ENG

from sww.libs.timeseries.plots.plot_style import get_legend_dict, add_custom_legend
from sww.libs.timeseries.plots.axes_formatting import diurnal_axes, weekly_x_axes
from sww.libs.timeseries.stats.stats import compare_week_table, compare_daily_times_table

from . import MEDIAN__MAD_SPLIT_FULL
from ._helpers.debug_helpers import check
from .date_analysis import get_school_holidays, get_holidays
from ._class import AnalyseData, L
from ._helpers.calculation_helpers import calc_dry_mean, calc_dry_variation


LANG = ENG


########################################################################################################################
def diurnal_density(data: AnalyseData, day_series: pd.Series, ylim=None, ylab=None, smooth=20, show_rain=None, unit=None, title=None, lang=LANG,
                    major_freq='H', minor_freq='15T', rasterized=True) -> tuple[plt.Figure, plt.Axes]:
    # Statistical Tests for Normality
    # from scipy.stats import *
    # https://towardsdatascience.com/normality-tests-in-python-31e04aa4f411

    day = day_series.name

    s = day_series.resample('5T').mean()

    if isinstance(show_rain, pd.Series):
        crit = show_rain.copy()

        dry = s.copy()
        dry.loc[crit > 100] = np.NaN
        dry.name = 'DRY_{}'.format(s.name)

        dry.dropna(inplace=True)
        data_table_dry = compare_daily_times_table(dry)
        ax = data_table_dry.T.plot(alpha=0.05, legend=False, color='k', rasterized=rasterized)

        wet = s[s.index.difference(dry.index)]
        data_table_wet = compare_daily_times_table(wet)
        ax = data_table_wet.T.plot(alpha=0.05, legend=False, color='cyan', ax=ax, rasterized=rasterized)
    else:
        data_table = compare_daily_times_table(s)
        ax = data_table.T.plot(alpha=0.05, legend=False, color='k', rasterized=rasterized)

    ax = diurnal_axes(ax, major_freq=major_freq, minor_freq=minor_freq)

    ax.set(ylim=ylim)

    if title:
        title = make_title(title,
                           default=get_diurnal_title(name=data.name,
                                                     day=s.name,
                                                     kind=data.arithmetic,
                                                     limit=data.limit,
                                                     smooth=smooth))

        ax.set_title(title, fontsize=12, fontweight='bold')

    if ylab:
        ax.set_ylabel(cst_label(data.name, unit=unit))

    # ax = data.dw_mean_table(smooth=smooth)[day].plot(ax=ax, color=daykind_color(s.name))

    m = ax.plot(data.dw_mean_table(smooth=smooth)[day], color=daykind_color(s.name))

    bounds = data.get_dw_bound_table(smooth=smooth)

    upper, lower = bounds[L.UPPER], bounds[L.LOWER]

    bw = ax.fill_between(data.dw_mean_table(smooth=smooth).index,
                    lower[day],
                    upper[day],
                    alpha=.2, color=daykind_color(s.name))

    # add_custom_legend(ax, {'DW Bandwidth': bw})
    ax.legend(m + [bw], ['DW Mean', 'DW Bandwidth'])

    # ax.get_figure().show()
    # ax.legend().remove()
    return ax.get_figure(), ax


########################################################################################################################
def diurnal_density2(day_series: pd.Series, data: AnalyseData, dry_data=None, smooth=20, criterion=None, unit=None, no_calc=False,
                     down_scale='5T', add_bounds=True, title=None, ylim=None, two_label_lines=True, ylabel=None,
                     lang=LANG, split=True, middle_y=None, rasterized=True) -> plt.Figure:
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
        dry.loc[crit > 100] = np.NaN
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
        ax = data_table.T.plot(alpha=0.05, legend=False, color='k', ax=ax)
        if split:
            ax2 = data_table.T.plot(alpha=0.05, legend=False, color='k', ax=ax2)
        alpha = 0.2

    title = make_title(title, default=get_diurnal_title(name=data.name, day=s.name))

    ax = diurnal_axes(ax, title=title, xlabel=diurnal_xlabel(lang=lang))
    if split:
        ax2 = diurnal_axes(ax2, title=title, xlabel=diurnal_xlabel(lang=lang))

    if not no_calc:
        ax = data.dw_mean_table(smooth=smooth)[day].plot(ax=ax, color=daykind_color(s.name), lw=2)
        agg_dry_bound = data.get_dw_bound_table(smooth=smooth)

        if add_bounds:
            ax.fill_between(data.dw_mean_table(smooth=smooth).index,
                            agg_dry_bound[(L.LOWER, day)],
                            agg_dry_bound[(L.UPPER, day)],
                            alpha=alpha, color=daykind_color(s.name), lw=2)

        if isinstance(dry_data, AnalyseData):
            ax = dry_data.dw_mean_table(smooth=smooth)[day].plot(ax=ax, color='yellow', linestyle='dotted', lw=2)
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
        # ax.axhline(middle_y, color='k')
        # ax2.axhline(middle_y, color='k')
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


def diurnal_density_full(ts: pd.Series, major_freq='H', minor_freq='15T', alpha=0.05, rasterized=True) -> tuple[plt.Figure, plt.Axes]:
    data_table = compare_daily_times_table(ts)
    ax = data_table.T.plot(alpha=alpha, legend=False, color='k', rasterized=rasterized)
    ax = diurnal_axes(ax, major_freq=major_freq, minor_freq=minor_freq)
    return ax.get_figure(), ax


def weekly_density_plot(data: AnalyseData, ax=None, add_mean=True, color='k', smooth=None) -> tuple[plt.Figure, plt.Axes]:
    data_table = compare_week_table(data.ts[data.day_category_index != '8 Holiday'].copy())
    ax = data_table.T.plot(alpha=0.05, legend=False, color=color, ax=ax, rasterized=True)

    ax.set_title('Weekly')

    if add_mean:
        # data.set_number_day_labels()
        data7_mean = data.dw_mean_table(smooth=None).copy()
        if '8 Holiday' in data7_mean:
            del data7_mean['8 Holiday']
        if 'Holyday' in data7_mean:
            del data7_mean['Holiday']

        if '7 Sunday' in data7_mean:
            data7_mean['0 Sunday'] = data7_mean['7 Sunday']
            del data7_mean['7 Sunday']
        if 'Sunday' in data7_mean:
            data7_mean['Sunday'] = data7_mean['Sunday']
            del data7_mean['Sunday']

        data7_mean = data7_mean.sort_index(axis=1)

        data7_mean2 = data7_mean.copy()
        data7_mean2.index = pd.to_timedelta([i.hour for i in data7_mean2.index], unit='h') + \
                            pd.to_timedelta([i.minute for i in data7_mean2.index], unit='m')

        data7_mean2.columns = pd.timedelta_range(start=pd.Timedelta(0), end=pd.Timedelta(days=6), freq='d')

        meas = data7_mean2.stack(0).sort_index(level=1)
        meas.index = meas.index.get_level_values(0) + meas.index.get_level_values(1)
        if smooth is not None:
            meas = meas.rolling(smooth).mean()
        ax = meas.rename('Mittelwert').plot(ax=ax, color='red', lw=2, legend=True)

    ax = weekly_x_axes(ax)  # , custom_switch_time='00:00', start = data7_mean.index[0], data7_mean.index[-1])
    ylim = get_ylim(data.ts)
    ax.set_ylim(ylim)
    return ax.get_figure(), ax


########################################################################################################################
def stability_analysis(data: AnalyseData, kind=1, var=False, lang=LANG):
    time_dist = pd.DataFrame()

    groups = data.get_analysis_grouper()

    # ---------------------------------------------------------------------------------
    for g in groups.groups:
        _, timestamp = g
        if timestamp.minute != 0:
            continue

        series = groups.get_group(g)
        s = series.replace(0, np.NaN).dropna()

        new_dist = pd.Series(index=range(len(s)))

        final = calc_dry_mean(s, kind)
        if var:
            final = calc_dry_variation(s - final, kind=kind)

        for n in range(5):
            news = s.sample(frac=1, random_state=n).reset_index(drop=True)

            for i in np.arange(5, len(s), 2):
                x = news.iloc[0:i]

                recent = calc_dry_mean(x, kind)
                if var:
                    recent = calc_dry_variation(x - recent, kind)

                new_dist.iloc[i] = (recent - final) / final * 100

            time_dist = pd.concat([time_dist, new_dist], axis=1)
    # ---------------------------------------------------------------------------------

    upper_ranges = time_dist.quantile([0.995, 0.975, 0.95], axis=1).T
    upper_ranges = upper_ranges.rolling(4, min_periods=1, center=True).median().rolling(5, min_periods=1,
                                                                                        center=True).mean()
    upper_ranges.columns = ['99%', '95%', '90%']

    lower_ranges = time_dist.quantile([0.05, 0.025, 0.005], axis=1).T
    lower_ranges = lower_ranges.rolling(4, min_periods=1, center=True).median().rolling(5, min_periods=1,
                                                                                        center=True).mean()

    ax = upper_ranges.plot(color=['r', 'g', 'y'], legend=True)
    ax.legend(title='Described calculation steps')
    ax = lower_ranges.plot(color=['y', 'g', 'r'], legend=False, ax=ax)

    title = 'DW-Mean Stability Analysis\n' + args_to_string(kind=kind)

    if var:
        title = title.replace('Mean', 'Variation')

    ylim = (-20, 40)
    if var:
        ylim = (-100, 150)

    ax.set_ylim(ylim)
    ax.set_xlim(0)
    ax.set_ylabel('Divergence to Final Result [%]')
    ax.set_xlabel('Number of considered Data')
    ax.set_title(title, fontsize=12, fontweight='bold')

    return ax.get_figure()


########################################################################################################################
def compare_day(data: AnalyseData, smooth=20, unit=None, add_bounds=True, title=None, two_lines=True, lang=LANG, major_freq='H', minor_freq='15T') -> tuple[plt.Figure, plt.Axes]:
    """

    Args:
        data:
        smooth:
        unit:
        add_bounds:
        title (str): title of the plot
        two_lines: for ylabel name and abbr+unit in 1 oder 2 lines
        lang:

    Returns:
        plt.Figure:
    """
    if data.day_kind_detail == 8:
        data.set_number_day_labels()

    mean = data.dw_mean_table(smooth=smooth)
    if add_bounds:
        agg_dry_bound = data.get_dw_bound_table(smooth=smooth)

    ax = None
    for day in mean.columns:
        print(day)
        ax = mean[day].plot(ax=ax, color=daykind_color(day), legend=True)
        if add_bounds:
            ax.fill_between(mean.index,
                            agg_dry_bound[(L.LOWER, day)],
                            agg_dry_bound[(L.UPPER, day)],
                            alpha=.25, color=daykind_color(day), lw=2)

    ylim = get_ylim(data.ts)

    title = make_title(title, default=get_compare_diurnal_title(name=data.name))
    # title = get_compare_diurnal_title(name=data.name, kind=data.arithmetic, limit=data.limit, smooth=smooth)

    ax = diurnal_axes(ax, ylab=cst_label(data.name, unit=unit, two_lines=two_lines), title=title, major_freq=major_freq, minor_freq=minor_freq)
    ax.set_ylim(ylim)
    return ax.get_figure(), ax


def compare_all_days(data: AnalyseData, smooth=None, major_freq='H', minor_freq='15T'):
    data10 = AnalyseData(data.ts, limit=data.limit, kind=data.arithmetic, day_kind_detail=10, file_path=data.temp_file_path,
                         make_temp_files=False, est_best_shift_time=False, smooth_window=data.smooth_window).set_number_day_labels()

    mean = data10.dw_mean_table(smooth=data.smooth_window if smooth is None else smooth)

    ax = None
    for day in mean.columns:
        ax = mean[day].plot(color=daykind_color(day), legend=True, ax=ax)

    ax = diurnal_axes(ax, major_freq=major_freq, minor_freq=minor_freq)
    return ax.get_figure(), ax


########################################################################################################################
def dry_percentage(data: AnalyseData, unit=None, title=None, lang=LANG):
    from scipy.stats import percentileofscore

    def calc_dry_perc(s):
        day, day_time = s.name
        upper = percentileofscore(s, data.upper_bound.loc[day_time, day])
        lower = percentileofscore(s, data.upper_bound.loc[day_time, day])
        return upper - lower

    res = data.get_analysis_grouper().apply(calc_dry_perc).unstack().T
    ax = res.plot()

    title = make_title(title, default=get_compare_diurnal_title(name=data.name, kind=data.arithmetic, limit=data.limit))

    ax = diurnal_axes(ax, ylab=cst_label(data.name, unit=unit), title=title)
    ax.set_ylim(0, 100)
    return ax.get_figure()


########################################################################################################################
def dry_trend(data: AnalyseData, smooth_window=pd.Timedelta(days=2), color=None, label='Dry-Weather Level',
              title=None, lang=LANG, mark_holidays=False):
    level = data.get_criterion_level_series(smooth_window=smooth_window).resample('D').mean()
    ax = level.plot(color=color)
    ax.set_xlim(level.index[0], level.index[-1])

    if mark_holidays:
        school_holidays = get_school_holidays()

        for name, times in school_holidays.iterrows():
            ax.axvspan(times['Beginn'], times['Ende'], color='y', alpha=0.7)

        hd = get_holidays(list(range(level.index[0].year, level.index[-1].year)), state='ST')

        for day, name in hd.items():
            ax.axvspan(day, day + pd.Timedelta(days=1, seconds=-1), color='r', alpha=0.6)

    ax.axhline(0, color='black', linewidth=0.7)
    ax.axhline(100, color='darkgray', linewidth=0.7)
    ax.axhline(-100, color='darkgray', linewidth=0.7)
    ax.set_ylabel(label)
    ax.set_xlabel('')
    ax.set_title(make_title(title, default=cst_label(data.name, unit=False)))
    ax = translate_ax(ax, lang=lang)
    return ax.get_figure()


########################################################################################################################
def diurnal_uncertainty_density(data: AnalyseData, day_series, smooth=20, ylim=None, unit=None, title=None,
                                major_freq='H', minor_freq='15T', rasterized=True) -> (plt.Figure, plt.Axes):
    day = day_series.name

    # ------------
    dw_bool_full = data.get_dry_weather_bool().fillna(False)
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
    ax = data_table.T.plot(alpha=0.05, legend=False, color='k', label='_nolegend_', rasterized=rasterized)
    # ax.legend().remove()
    # ------------
    std_raw = data_table.std()
    std = std_raw.rolling(smooth, center=True, min_periods=1).mean()
    interval_68 = std
    interval_95 = std * 2
    interval_99 = std * 3

    ax.plot(interval_68.index, -interval_68, color='orange', ls='--', lw=0.75, label='68.3% (1 $\sigma$)')
    ax.plot(interval_68.index, interval_68, color='orange', ls='--', lw=0.75)

    ax.plot(interval_95.index, -interval_95, color='red', ls='--', lw=0.75, label='95.4% (2 $\sigma$)')
    ax.plot(interval_95.index, interval_95, color='red', ls='--', lw=0.75)

    ax.plot(interval_99.index, -interval_99, color='darkviolet', ls='--', lw=0.75, label='99.7% (3 $\sigma$)')
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
    # ax.get_figure().set_layout_engine('constrained')
    # ax.get_figure().show()
    return ax.get_figure(), ax


########################################################################################################################
def compare_dw_uncertainty_day_absolute(data: AnalyseData, smooth=20,
                                        major_freq='H', minor_freq='15T', unit='L/s') -> tuple[plt.Figure, plt.Axes]:

    if data.day_kind_detail == 8:
        data.set_number_day_labels()

    # ---
    uncertainty = data.get_dw_uncertainty_table(smooth=smooth)

    # ---
    ax: plt.Axes = None
    for day in uncertainty.columns:
        # print(day)
        ax = uncertainty[day].plot(ax=ax, color=daykind_color(day), legend=True)

    ax = diurnal_axes(ax, major_freq=major_freq, minor_freq=minor_freq)

    ax.set_title('absolute Uncertainty')

    # ax.get_figure().show()

    return ax.get_figure(), ax


def compare_dw_uncertainty_day_relative(data: AnalyseData, smooth=20,
                                        major_freq='H', minor_freq='15T') -> tuple[plt.Figure, plt.Axes]:

    if data.day_kind_detail == 8:
        data.set_number_day_labels()

    # ---
    mean = data.dw_mean_table(smooth=smooth)
    uncertainty = data.get_dw_uncertainty_table(smooth=smooth)

    uncertainty_rel = uncertainty / mean

    # ---
    ax: plt.Axes = None
    for day in uncertainty_rel.columns:
        # print(day)
        ax = uncertainty_rel[day].plot(ax=ax, color=daykind_color(day), legend=True)

    ax = diurnal_axes(ax, major_freq=major_freq, minor_freq=minor_freq)

    ax.set_title('relative Uncertainty')

    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # ax.get_figure().show()

    return ax.get_figure(), ax


########################################################################################################################
AnalyseData.figure_diurnal_density = diurnal_density
AnalyseData.figure_compare_day = compare_day
AnalyseData.figure_dry_percentage = dry_percentage
AnalyseData.figure_dry_trend = dry_trend
AnalyseData.figure_stability_analysis = stability_analysis


class AnalysePlots:
    def __init__(self, data: AnalyseData):
        self._data = data

    def diurnal_density(self, day_series, ylim, smooth=20, show_rain=None, unit=None, title=None, lang=LANG):
        return diurnal_density(self._data, day_series, ylim, smooth=smooth, show_rain=show_rain, unit=unit, title=title,
                               lang=lang)

    def compare_day(self, smooth=20, unit=None, add_bounds=True, title=None, two_lines=True, lang=LANG):
        return compare_day(self._data, smooth=smooth, unit=unit, add_bounds=add_bounds, title=title,
                           two_lines=two_lines,
                           lang=lang)

    def dry_percentage(self, unit=None, title=None, lang=LANG):
        return dry_percentage(self._data, unit=unit, title=title, lang=lang)

    def diurnal_density2(self, dry_data=None, smooth=20, criterion=None, unit=None, no_calc=False,
                         down_scale='5T', add_bounds=True, title=None, ylim=None, two_label_lines=True, ylabel=None,
                         lang=LANG):
        daily_groups = self._data.get_day_grouper()
        return [diurnal_density2(daily_groups.get_group(daily_group), self._data, dry_data=dry_data, smooth=smooth,
                                 criterion=criterion, unit=unit,
                                 no_calc=no_calc, down_scale=down_scale, add_bounds=add_bounds, title=title, ylim=ylim,
                                 two_label_lines=two_label_lines, ylabel=ylabel, lang=lang)
                for daily_group in daily_groups]

    def dry_trend(self, smooth_window=pd.Timedelta(days=2), color=None, label='Dry-Weather Level', title=None,
                  lang=LANG):
        return dry_trend(self._data, smooth_window=smooth_window, color=color, label=label, title=title, lang=lang)

    def stability_analysis(self, kind=1, var=False, lang=LANG):
        return stability_analysis(self._data, kind=kind, var=var, lang=lang)
