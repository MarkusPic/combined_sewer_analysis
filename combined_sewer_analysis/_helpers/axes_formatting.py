import calendar

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

# from .cmap_custom import custom_color_mapper
from .legend_helpers import add_custom_legend


def weekly_axes(ax, custom_switch_time, start, end):
    """
    the weekly axes decoration
    new x-axis ticks and labels based on the switch time

    Args:
        ax (matplotlib.pyplot.Axes):
        custom_switch_time (str): HH:MM
        start (pandas.Timestamp):
        end (pandas.Timestamp):

    Returns:

    """
    # ax.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='minor',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)  # labels along the bottom edge are off

    ax.set_xticks([], minor=True)

    # start = series.index[0]
    hour, minute = [int(t) for t in custom_switch_time.split(':')]
    start = start.replace(hour=hour, minute=minute)

    locs = pd.date_range(start, periods=8, freq='24h')

    ax.set_xticks(locs)
    ax.set_xticklabels([item.strftime('%a. %H:%M\n%d. %b\n%Y') for item in locs], rotation='horizontal', ha='center')
    ax.set_xlabel('')
    ax.set_xlim(locs[0], locs[-1])
    if start:
        ax.set_xlim(left=start)
    if end:
        ax.set_xlim(right=end)
    return ax


def get_efficient_datetime_labels(ticks,  fmt_month='%b.\n', fmt_day='%a. %d.\n', fmt_time='%H:%M\n'):
    labels = []

    last_year = None
    last_month = None
    last_day = None

    for t in ticks:
        year = t.year
        month = t.strftime(fmt_month)
        day = t.strftime(fmt_day)
        time = t.strftime(fmt_time)

        if year != last_year:
            label = f'{time}{day}{month}{year}'
        elif month != last_month:
            label = f'{time}{day}{month}'
        elif day != last_day:
            label = f'{time}{day}'
        else:
            label = f'{time}'

        last_year = year
        last_month = month
        last_day = day
        labels.append(label.strip())

    return labels


def timeseries_axes(ax, start=None, end=None):
    ax.set_xticks([], minor=True)
    ticks = pd.to_datetime(ax.get_xticks(), unit='D')
    # pd.to_datetime(ax.get_xticks(), unit='m')
    # t = ax.get_xticks()
    # l = ax.get_xticklabels()
    ax.set_xticks(ticks)

    labels = get_efficient_datetime_labels(ticks)
    ax.set_xticklabels(labels, rotation='horizontal', ha='center')
    ax.set_xlabel('')

    if start:
        ax.set_xlim(left=start)
    if end:
        ax.set_xlim(right=end)
    return ax


def diurnal_axes(ax, ylim=None, ylab=None, title=None, reference_time='00:00', xlabel=None, major_freq='h', minor_freq='15min', major_fmt='%-H'):
    """
    Make a diurnal axes.

    Args:
        ax (plt.Axes): plot axes
        ylim (tuple[int, int]):
        ylab (str):
        title (str):
        reference_time (str):
        xlabel (str):
        major_freq (str):
        minor_freq (str):

    Returns:
        plt.Axes: plot axes
    """
    if ylim:
        ax.set_ylim(ylim)

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')

    if ylab:
        ax.set_ylabel(ylab)

    ax.set_xlim(left='00:00:00', right='23:59:59.999999')

    if xlabel is None:
        ax.set_xlabel('Hours of the day')
    else:
        ax.set_xlabel(xlabel)

    # MAJOR
    major_ticks = pd.date_range("00:00", "23:59", freq=major_freq).append(pd.DatetimeIndex(['23:59:59.999999']))
    ax.set_xticks(major_ticks.time)
    hours = major_ticks.strftime(major_fmt).to_list()
    hours[-1] = hours[-1].replace('23', '24')
    ax.set_xticklabels(hours)

    # MINOR
    if isinstance(minor_freq, str):
        minor_ticks = pd.date_range("00:00", "23:59", freq=minor_freq)
        ax.set_xticks(minor_ticks.time, minor=True)
    elif minor_freq is None:
        ax.set_xticks([], minor=True)

    # split_times = [int(t) for t in reference_time.split(':')]
    # if reference_time == '00:00':
    #     switcher = False
    # else:
    #     switcher = True
    #
    # def format_date(x, pos=None):
    #     hour = int(x / 3600)
    #     if switcher:
    #         hour += split_times[0]
    #         if hour >= 24:
    #             hour -= 24
    #     return str(hour)  # + 'h'
    #
    # ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_date))

    # ax.grid(which='minor', alpha=0.2)
    # ax.grid(which='major', alpha=0.5)

    return ax


def to_ticks(t):
    return t.total_seconds() * 1e9


def from_ticks(t):
    return pd.to_timedelta(t, unit='ns')


def to_tick_label(t, fmt, sunday_first=True):
    today = datetime.today()
    date_time = pd.to_datetime(today.date() - timedelta(days=today.weekday() + int(sunday_first))) + t
    return date_time.strftime(fmt)


def set_xtick_labels(ax, t, minor=False, fmt=None, sunday_first=True):
    """

    Args:
        ax:
        t (pandas.TimedeltaIndex):
        minor (bool):
        fmt (str): https://strftime.org

    Returns:

    """
    if minor:
        major = from_ticks(ax.get_xticks(minor=False))
        t = t.difference(major)

    ax.set_xticks(to_ticks(t), minor=minor)
    ax.set_xticklabels(to_tick_label(t, fmt=fmt, sunday_first=sunday_first), minor=minor)


def weekly_x_axes(ax, #custom_switch_time, start, end,
                  minor_freq='6h', major_freq='24h', major_fmt='%Hh\n%A', minor_fmt='%Hh', sunday_first=True):
    set_xtick_labels(ax, pd.timedelta_range(pd.Timedelta(0), pd.Timedelta(days=7), freq=major_freq), fmt=major_fmt,
                     sunday_first=sunday_first)

    if minor_freq is not None:
        set_xtick_labels(ax, pd.timedelta_range(pd.Timedelta(0), pd.Timedelta(days=7), freq=minor_freq), minor=True,
                         fmt=minor_fmt, sunday_first=sunday_first)

    if hasattr(ax, 'left_ax'):
        ax.left_ax = weekly_x_axes(ax.left_ax)
    return ax


def set_default_rain_y_label(ax):
    """
    Args:
        ax (matplotlib.axes.Axes):

    Returns:
        matplotlib.axes.Axes:
    """
    ax.set_ylabel('Niederschlag\nP in mm/min')
    return ax


def time_ticks(df, freq):
    # import datetime
    # custom_tick_locs = [datetime.time(hour=8), datetime.time(hour=16)]
    # custom_tick_labels = map(lambda x: x.strftime('%H'), custom_tick_locs)
    # plt.xticks(custom_tick_locs, custom_tick_labels)

    ticklabels = [''] * len(df.index)

    ticklabels[::24 * 60] = [item.strftime('%a %H:%M\n%d.%b\n%Y') for item in df.index[::24 * 60]]
    print(ticklabels)
    return ticklabels

    print(len(df.index))
    exit()
    if 'M' in freq:
        # Every 4th ticklable shows the month and day
        # ticklabels[::3] = [item.strftime('%b') for item in df.index[::3]]
        # Every 12th ticklabel includes the year
        ticklabels[::12] = [item.strftime('%b\n%Y') for item in df.index[::12]]
    elif 'A' in freq:
        ticklabels[::1] = [item.strftime('%Y') for item in df.index[::1]]
    elif 'D' in freq:
        ticklabels[::1] = [item.strftime('%a\n%d. %b\n%Y') for item in df.index[::1]]
        plt.xlabel('Daily sum')
        # ticklabels[::1] = [item.strftime('%a\n%d') for item in df.index[::1]]

    # ax.xaxis.set_tick_params(reset=True)
    # ax.xaxis.set_major_locator(mpl.dates.YearLocator(1))
    # ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))

    return ticklabels


def set_y_ticks(ax, y_min, y_max, y_major=None, y_minor=None):
    if y_major is not None:
        ax.set_yticks(np.arange(y_min, y_max + y_major, y_major))
    if y_minor is not None:
        ax.set_yticks(np.arange(y_min, y_max + y_minor, y_minor), minor=True)
    ax.set_ylim(y_min, y_max)


def make_corr_axes(ax: plt.Axes):
    # synchronise axes
    ax.set_aspect('equal', 'box')

    lim_ = list(zip(ax.get_xlim(), ax.get_ylim()))
    lim = min(lim_[0]), max(lim_[1])

    ax.set_ylim(*lim)
    ax.set_xlim(*lim)

    yticks = [y for y in ax.get_yticks() if lim[0] < y < lim[1]]
    ax.set_yticks(yticks)
    ax.set_xticks(yticks)

    yticks_minor = [y for y in ax.get_yticks(minor=True) if lim[0] < y < lim[1]]
    ax.set_yticks(yticks_minor, minor=True)
    ax.set_xticks(yticks_minor, minor=True)

    ax.set_xlim(*lim)

    ax.grid(ls=':', which='both')
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k', zorder=-1, lw=0.75)


def plot_correlation_monthly_daily(a, b, lower_quantile=0.25, higher_quantile=0.75, b_lookup=False, show_all=False,
                                   language='eng'):
    """
    Scatter mit horizontalen IQR als Linie

    Messung vs. Regression

    Args:
        a:
        b:
        lower_quantile:
        higher_quantile:
        b_lookup:
        show_all:
        language:

    Returns:
        (plt.Figure, plt.Axes): plt
    """
    fig, ax = plt.subplots()

    # for the months
    cm = custom_color_mapper(cmap='rainbow', vmin=1, vmax=12, set_under='lightgray', set_bad='k')

    # for the days
    markers = ['o', 'v', 's', 'D', '*', 'P', '<']

    months = range(1, 13)
    weekdays = range(7)

    # ---------------
    legend1 = {}
    legend2 = {}

    # ---------------
    for month in months:  # type: int
        color = cm(month)
        legend1[f'{month:d} {calendar.month_abbr[month]}'] = dict(marker='.', color=color, linestyle='None')
        for day in weekdays:  # type: int
            marker = markers[day]
            legend2[f'{calendar.day_abbr[day]}'] = dict(marker=marker, color='black', linestyle='None')

            x = a[(a.index.month == month) & (a.index.weekday == day)]
            if b_lookup:
                y = [b.loc[(month, day)]] * len(x)
            else:
                y = b[(b.index.month == month) & (b.index.weekday == day)].dropna()

            if show_all:
                i = x.dropna().index.intersection(y.dropna().index)
                ax.scatter(x[i], y[i], color=color, marker=marker, alpha=0.5, lw=0)

            else:
                if len(x.dropna()) != 0:
                    ax.scatter(x.median(), y[0], color=color, marker=marker, alpha=0.5)

                if len(x) > 1:
                    ax.plot(x.quantile([lower_quantile, higher_quantile]), y[:2], color=color, alpha=0.7, lw=0.6)

                # if len(y) > 1:
                #     ax.plot(y.quantile([0.25, 0.75]), y[0:2], color=color, label=f'{m} {w}', alpha=0.7, lw=0.6)

    # ---------------
    ax.add_artist(add_custom_legend(ax, legend1, title={'eng': 'Month', 'ger': 'Monat'}[language],
                                    loc='upper left', bbox_to_anchor=(1, 1, 1, 0)))
    add_custom_legend(ax, legend2, title={'eng': 'Weekday', 'ger': 'Wochentag'}[language],
                      loc='lower left', bbox_to_anchor=(1, 0, 1, 0))

    # ---------------
    make_corr_axes(ax)
    return fig, ax


def init_zero_axes_figure():
    from mpl_toolkits.axisartist.axislines import AxesZero

    fig = plt.figure()
    ax = fig.add_subplot(axes_class=AxesZero)

    for direction in ["xzero", "yzero"]:
        # adds arrows at the ends of each axis
        ax.axis[direction].set_axisline_style("-|>")

        # adds X and Y-axis from the origin
        ax.axis[direction].set_visible(True)

    for direction in ["left", "right", "bottom", "top"]:
        # hides borders
        ax.axis[direction].set_visible(False)

    # ax.spines[["left", "bottom"]].set_position(("data", 0))
    # ax.spines[["top", "right"]].set_visible(False)
    # ax.set_aspect('equal')
    return fig, ax


def format_timedelta_xaxis(ax: plt.Axes, index: pd.TimedeltaIndex, *, short: bool = False, german: bool = False) -> None:
    """
    Format the x-axis of a matplotlib Axes for a TimedeltaIndex, setting ticks and labels at full intervals.

    The interval is chosen based on the total span of the index, and labels are rendered using readable timedelta strings.

    Args:
        ax (plt.Axes): The matplotlib Axes to format.
        index (pd.TimedeltaIndex): A sorted pandas TimedeltaIndex.
        short (bool): Whether to use short formatting for labels.
        german (bool): Whether to use German labels.
    """
    # Calculate the total range of the index
    delta = index[-1] - index[0]
    total_seconds = delta.total_seconds()

    # Determine a suitable tick interval
    if total_seconds <= 60:  # up to 1 min
        freq = pd.Timedelta(seconds=5)
        subfreq = pd.Timedelta(seconds=1)
    elif total_seconds <= 5 * 60:  # up to 5 min
        freq = pd.Timedelta(seconds=30)
        subfreq = pd.Timedelta(seconds=5)
    elif total_seconds <= 15 * 60:  # up to 15 min
        freq = pd.Timedelta(minutes=1)
        subfreq = pd.Timedelta(seconds=15)
    elif total_seconds <= 60 * 60:  # up to 1 hour
        freq = pd.Timedelta(minutes=5)
        subfreq = pd.Timedelta(minutes=1)
    elif total_seconds <= 3 * 60 * 60:  # up to 3 hours
        freq = pd.Timedelta(minutes=15)
        subfreq = pd.Timedelta(minutes=5)
    elif total_seconds <= 12 * 60 * 60:  # up to 12 hours
        freq = pd.Timedelta(minutes=30)
        subfreq = pd.Timedelta(minutes=5)
    elif total_seconds <= 24 * 60 * 60:  # up to 1 day
        freq = pd.Timedelta(hours=1)
        subfreq = pd.Timedelta(minutes=15)
    elif total_seconds <= 3 * 24 * 60 * 60:  # up to 3 days
        freq = pd.Timedelta(hours=6)
        subfreq = pd.Timedelta(hours=1)
    elif total_seconds <= 7 * 24 * 60 * 60:  # up to 1 week
        freq = pd.Timedelta(days=1)
        subfreq = pd.Timedelta(hours=6)
    elif total_seconds <= 3 * 7 * 24 * 60 * 60:  # up to 3 weeks
        freq = pd.Timedelta(days=2)
        subfreq = pd.Timedelta(hours=12)
    else:
        freq = pd.Timedelta(weeks=1)
        subfreq = pd.Timedelta(days=1)

    # Create ticks starting from the first full step after the min index
    start = (index[0] // freq) * freq
    end = index[-1]

    tick_locs = []
    tick_labels = []

    from sww.libs.timeseries.stats.freqs import timedelta_readable
    current = start
    while current <= end:
        tick_locs.append(current.total_seconds()*1e9)
        l = timedelta_readable(current, short=short, german=german, sep=';')
        l = '\n'.join(l.replace('and', ';').split(';')[::-1])
        tick_labels.append(l)
        current += freq

    tick_locs_minor = []
    current = start
    while current <= end:
        tick_locs_minor.append(current.total_seconds()*1e9)
        current += subfreq

    # ax.get_xticks()
    # ax.get_xticklabels()

    ax.set_xticks(tick_locs)
    ax.set_xticks(tick_locs_minor, minor=True)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(index[0].total_seconds()*1e9,
                index[-1].total_seconds()*1e9)
