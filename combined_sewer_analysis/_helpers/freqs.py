__author__ = "Markus Pichler"
__copyright__ = "Copyright 2017, University of Technology Graz"
__credits__ = ["Markus Pichler"]
__license__ = "LGPL"
__version__ = "1.0.0"
__maintainer__ = "Markus Pichler"

import datetime
import pandas as pd
from pandas import DatetimeIndex, Series, DateOffset, Timedelta
from pandas.tseries.frequencies import to_offset


def year_delta(years):
    return pd.Timedelta(days=365.2425 * years)


########################################################################################################################
def readable_offset(offset, short=False, german=False):
    """
    Converts DateOffset format "<x * freq>" to "x freq" for better readability

    Args:
        offset (pandas.DateOffset | pandas.Series[pandas.DateOffset] | pandas.Index[pandas.DateOffset]):
        short (bool): using only abbreviations
        german (bool): translate to german

    Returns:
        str | pandas.Series[str]:
    """
    if isinstance(offset, DateOffset):
        # '<10 * Minutes>' ->  '10 Minutes'
        o = str(offset).replace('<', '').replace('>', '').replace('* ', '')
        if short:
            return {'minute': 'min',
                    'hour': 'h',
                    'day': 'd'}.get(o.lower(), o)
        return o
    elif isinstance(offset, Series):
        return offset.apply(readable_offset, short=short, german=german)
    elif isinstance(offset, pd.Index):
        return readable_offset(offset.to_series())
    else:
        raise NotImplementedError(type(offset))


########################################################################################################################
def delta2offset(delta):
    """
    convert (Series/Index with) Timedelta values to Tick values

    Args:
        delta (pandas.Timedelta | pandas.Series[pandas.Timedelta] | pandas.Index[pandas.Timedelta] | str):

    Returns:
        pandas.DateOffset | pandas.Series[pandas.DateOffset]:
    """
    if isinstance(delta, Timedelta):
        if delta == Timedelta(days=0):
            return to_offset('0s')
        return to_offset(delta)
    elif isinstance(delta, str):
        return to_offset(delta)
    elif isinstance(delta, (Series, pd.arrays.TimedeltaArray)):
        try:
            return delta.apply(delta2offset)
        except AttributeError:
            return delta.map(delta2offset)
    elif isinstance(delta, pd.Index):
        return delta2offset(delta.to_series())
    else:
        raise NotImplementedError(type(delta))


########################################################################################################################
def delta2print(td, ceil_to='s'):
    """
    convert timedelta to a better readable string

    Args:
        td (pandas.Timedelta):
        ceil_to (str | DateOffset):

    Returns:
        str:
    """
    if ceil_to is not None:
        d = td.ceil(ceil_to)
    else:
        d = td
    di = d.components._asdict()
    components = ['{} {}'.format(item, key if item != 1 else key[:-1]) for key, item in di.items() if item != 0]
    return ', '.join(components[:-1]) + ' & ' + components[-1]


def delta_to_iso_light(td, sep=''):
    """
    convert timedelta to a better readable string

    Args:
        td (pandas.Timedelta):
        sep (str): seperator string

    Returns:
        str:
    """
    return sep.join(['{}{}'.format(item, key[0] + ('s' if ('seconds' in key and key != 'seconds') else '')) for key, item in td.components._asdict().items() if item != 0])


########################################################################################################################
def time_steps(date_time_index):
    """
    get all time steps existing within the DataFrame

    :type date_time_index: DatetimeIndex
    :rtype: Series[Timedelta]
    """
    return date_time_index.to_series().diff(periods=1).bfill()  # .fillna(method='backfill')


########################################################################################################################
def freq_steps(date_time_index, to_str=False):
    """
    get all time steps existing within the DataFrame

    :type date_time_index: DatetimeIndex
    :type to_str: bool
    :rtype: Series
    """
    delta_series = time_steps(date_time_index)
    freq_series = delta2offset(delta_series)
    if to_str:
        return readable_offset(freq_series)

    return freq_series


########################################################################################################################
def unique_frequencies(date_time_index, to_str=True):
    """
    return all gaps in form of a frequency

    :type date_time_index: DatetimeIndex
    :type to_str: bool
    :return: unique frequencies
    :rtype: list[str] | list[DateOffset]
    """
    delta_series = time_steps(date_time_index)
    unique_delta = delta_series.unique()
    unique_freqs = delta2offset(unique_delta)
    if to_str:
        return readable_offset(unique_freqs)
    return unique_freqs


########################################################################################################################
def guess_freq(date_time_index, default=pd.Timedelta(minutes=1)):
    """
    get most often frequency in the format [minutes]T eg: "1T" when the frequency is one minute

    Args:
        date_time_index (pandas.DatetimeIndex | pandas.Index): date-time-index of a time-series
        default (pandas.Timedelta):

    Returns:
        pandas.DateOffset: frequency of the date-time-index
    """
    # ---------------------------------
    # def _get_freq(freq):
    #     if isinstance(freq, str):
    #         freq = to_offset(freq)
    #     return freq

    # ---------------------------------
    freq = date_time_index.freq
    if pd.notnull(freq):
        return to_offset(freq)

    if len(date_time_index) > 3:
        freq = pd.infer_freq(date_time_index)  # 'T'

        if pd.notnull(freq):
            return to_offset(freq)

        delta_series = time_steps(date_time_index)
        counts = delta_series.value_counts()
        counts.drop(pd.Timedelta(minutes=0), errors='ignore')

        if counts.empty:
            delta = default
        else:
            delta = counts.index[0]
            if delta == pd.Timedelta(minutes=0):
                delta = default
    else:
        delta = default

    return to_offset(delta)


########################################################################################################################
def count_freq(date_time_index, to_str=True, german_str=False, short_str=False):
    """
    Counts all gaps and sums the appearances.

    Args:
        date_time_index (pandas.DatetimeIndex):
        to_str (bool):
        german_str (bool): translate string to german
        short_str (bool): use short string

    Returns:
        pandas.Series:
    """
    delta_series = time_steps(date_time_index)
    counts = delta_series.value_counts()
    counts.index = delta2offset(counts.index)

    # counts.index.to_series().apply(timedelta_readable, min_freq='s')
    if to_str:
        counts.index = readable_offset(counts.index, german=german_str, short=short_str)
        return counts

    return counts


def count_freq2(date_time_index, german_str=False, short_str=False, min_freq='s', sep=', '):
    """
    Counts all gaps and sums the appearances.

    Args:
        date_time_index (pandas.DatetimeIndex):
        german_str (bool): translate string to german
        short_str (bool): use short string

    Returns:
        pandas.Series:
    """
    delta_series = time_steps(date_time_index)
    counts = delta_series.value_counts()
    counts.index = counts.index.to_series().apply(timedelta_readable, min_freq=min_freq, short=short_str, sep=sep, german=german_str).values
    return counts


########################################################################################################################
def convert_freq(freq):
    """
    convert custom frequency abbreviation to python standard frequency

    Args:
        freq (str):

    Returns:
        str:
    """
    return {'D': '1D',
            'W': 'W-MON',
            'M': '1MS',
            'A': '1YS',
            'Y': '1YS'}.get(freq, freq)


def get_following_values(series):
    ev_bool = (series.diff() <= 0.01) & (series > 0.01)
    from .events import span_table, event_duration
    ev = span_table(ev_bool)
    ev['dur'] = event_duration(ev)
    ev2 = ev[ev['dur'] > pd.Timedelta(minutes=30)].copy()
    ev.sort_values('dur')


def check_freq_changes(series):
    from .following_values import remove_following_duplicates
    from numpy import nan
    import matplotlib.pyplot as plt

    ts = series.replace(0, nan)
    short = remove_following_duplicates(ts)
    # short = ts
    steps = time_steps(short.index).dt.round('10s')  # total_seconds().round(1)# / 60).round().astype(int)
    steps_fil = steps[~(short.shift().isna() & short.isna())]

    # full_steps = steps.resample('min').ffill()
    # full_steps_minutes = (full_steps.dt.total_seconds() / 60).round().astype(int)
    # full_steps_minutes[full_steps_minutes > 60*24] = NaN
    # full_steps_minutes[series.shift() == 0] = NaN

    # -----------------------------------
    import numpy as np

    c = steps_fil.value_counts()
    c.index = c.index.to_series().apply(to_offset)
    steps_fil_2 = steps_fil.copy()
    a = c.index.values
    unique_elements = np.unique(a)
    elements_left = np.sort(unique_elements)[::-1].copy()
    possible = []

    no_reminder = np.timedelta64(0, 'ns')

    while elements_left.size != 0:
        e, elements_left = elements_left[-1], elements_left[:-1]
        elements_left = elements_left[np.remainder(elements_left, e) != no_reminder]
        possible.append(e)
        steps_fil_2[(steps_fil_2.mod(e) == no_reminder) & (steps_fil_2 != e)] = np.nan

    b = remove_following_duplicates(steps_fil_2.ffill())
    res = {}
    for freq_delta, data in b.groupby(b):
        freq = to_offset(freq_delta)
        res[freq] = {}
        res[freq]['start'] = data.index.min()
        res[freq]['end'] = data.index.max()

    res_df = pd.DataFrame.from_dict(res, orient='index')
    steps_fil_2.plot(style='.', alpha=0.1, logy=True)
    # steps_fil.plot(style='.', alpha=0.1, logy=True)
    # plt.savefig('freq_series.png')
    plt.show()
    return res_df


def timedelta_components_plus(td, min_freq='min'):
    """Schaltjahre nicht wirklich miteinbezogen"""
    l = []

    if isinstance(td, datetime.timedelta):
        td = pd.to_timedelta(td)

    # years, weeks
    days_year = 365
    days_week = 7

    for component, value in td.round(min_freq).components._asdict().items():
        if component == 'days':
            years, value = value // days_year, value % days_year
            l.append([int(years), 'years'])

            value -= years // 4

            weeks, value = value // days_week, value % days_week
            l.append([int(weeks), 'weeks'])

        l.append([value, component])
    return l


DI_TRANSLATE_TD_COMPONENTS = {
    'years': 'Jahre',
    'weeks': 'Wochen',
    'days': 'Tage',
    'hours': 'Stunden',
    'minutes': 'Minuten',
    'seconds': 'Sekunden',
    'milliseconds': 'Millisekunden',
    'microseconds': 'Mikrosekunden',
    'nanoseconds': 'Nanosekunden',
}


def timedelta_components_readable(l, short=False, sep=', ', german=False):
    result = []
    for value, label_component in l:
        if german:
            label_component = DI_TRANSLATE_TD_COMPONENTS[label_component]
        if value > 0:
            if short:
                unit_sep = ''
                unit = label_component[0]
            else:
                unit_sep = ' '

                if value == 1:
                    unit = label_component[:-1]
                else:
                    unit = label_component

            result.append(f'{value}{unit_sep}{unit}')

    s = sep.join(result)

    if not short:
        last_sep = ' und ' if german else ' and '
        # replace last "," with "and"
        s = last_sep.join(s.rsplit(sep, 1))
    return s


def timedelta_readable(td, min_freq='min', short=False, sep=', ', german=False):
    """Schaltjahre nicht wirklich miteinbezogen"""
    if td == pd.Timedelta(0):
        if short:
            return '0m' if short else ' 0 m'
        return '0 Minuten' if german else '0 minutes'
    if td is pd.NaT:
        return '-'
    return timedelta_components_readable(timedelta_components_plus(td, min_freq), short=short, sep=sep, german=german)


def timedelta_readable2(d1, d2, min_freq='min', short=False, sep=', '):
    td = d2 - d1

    years = None
    if td > Timedelta(days=365):
        d2_new = d2.replace(year=d1.year)

        if d2_new < d1:
            d2_new = d2_new.replace(year=d1.year + 1)

        years = d2.year - d2_new.year

        td = d2_new - d1

    l = timedelta_components_plus(td, min_freq)

    if years is not None:
        l[0][0] = years

    return timedelta_components_readable(l, short=short, sep=sep)


def date_difference(date1: datetime.datetime, date2: datetime.datetime) -> str:
    """
    Calculate the difference between two dates in years, months, and days.
    The difference is calculated based on the calendar, not timedelta.

    Args:
        date1 (datetime.datetime): The earlier date.
        date2 (datetime.datetime): The later date.

    Returns:
        tuple[int, int, int]: A tuple of (years, months, days) difference.
    """
    if date1 > date2:
        date1, date2 = date2, date1  # Ensure date1 is earlier

    year_diff = date2.year - date1.year
    month_diff = date2.month - date1.month
    day_diff = date2.day - date1.day

    # Adjust days and months if necessary
    if day_diff < 0:
        month_diff -= 1
        # Get number of days in the previous month
        prev_month = (date2.month - 1) or 12
        prev_year = date2.year if date2.month != 1 else date2.year - 1
        days_in_prev_month = (datetime.datetime(prev_year, prev_month + 1, 1) - datetime.datetime(prev_year, prev_month, 1)).days
        day_diff += days_in_prev_month

    if month_diff < 0:
        year_diff -= 1
        month_diff += 12

    return f'{year_diff}y {month_diff}m {day_diff}d'
