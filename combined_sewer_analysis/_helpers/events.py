import pandas as pd
from pandas import Timedelta

from .freqs import guess_freq

__author__ = "Markus Pichler"
__copyright__ = "Copyright 2017, University of Technology Graz"
__credits__ = ["Markus Pichler"]
__license__ = "LGPL"
__version__ = "1.0.0"
__maintainer__ = "Markus Pichler"

START = 'start'
END = 'end'


def _index_series(date_time_index):
    """create a time series from a datetime index without losing the timezone info"""
    if isinstance(date_time_index, pd.DatetimeIndex) and date_time_index.tzinfo is not None:
        return pd.Series(data=date_time_index, index=date_time_index)
    else:
        return date_time_index.to_series()


def monotonic_error_table(date_time_index):
    """
    get the timedelta of data gaps in a DataFrame

    Args:
        date_time_index (pandas.DatetimeIndex):

    Returns:
        pandas.DataFrame: with the columns:
            'time_of_reset' = start-time,
            'reset_to' = end-time,
            'delta' = duration of the gap
    """
    temp = _index_series(date_time_index)

    timedelta = Timedelta(minutes=0)
    # start = temp[temp.diff(periods=-1) > -timedelta]
    start = temp[temp.diff(periods=-1).gt(-timedelta)]
    end = temp[temp.diff() < timedelta]

    events = pd.concat([start.reset_index(drop=True), end.reset_index(drop=True)], axis=1, ignore_index=True)
    events.columns = ['time_of_reset', 'reset_to']
    events['delta'] = events['time_of_reset'] - events['reset_to']

    return events


def fix_monotonic_errors(df, verbose=False):
    """Delete backward time jumps until index is monotonically increasing relative to index before jump"""
    df_clean = df.copy()

    from sww.libs.timeseries.stats import monotonic_error_table
    events_monotonic_errors = monotonic_error_table(df.index)
    if events_monotonic_errors.empty:
        if verbose:
            print('no monotonic errors found')
    else:
        for _, occurence in events_monotonic_errors.iterrows():
            i = df_clean.index.get_loc(occurence['time_of_reset'])
            index = df_clean.index
            if verbose:
                print('---')
                print(i, occurence['time_of_reset'])
            # j = index[i+1:][index[i+1:] > index[i]].idxmin()
            first_later_date = index[i + 1:][index[i + 1:] > index[i]].min()
            j = index.get_loc(first_later_date)

            if verbose:
                print(index[i:j + 1].map(str).to_list())
                print(j, first_later_date)
            k = i + 1
            while df_clean.iloc[k].name < occurence['time_of_reset']:
                if verbose: print('drop', k, df_clean.iloc[k].name)
                df_clean.drop(df_clean.iloc[k].name, inplace=True)

    if verbose:
        events_monotonic_errors = monotonic_error_table(df_clean.index)
        if not events_monotonic_errors.empty:
            print(events_monotonic_errors)
    return df_clean


def time_delta_table(date_time_index, timedelta=Timedelta(minutes=1)):
    """
    get the timedelta of data gaps in a DataFrame

    Args:
        date_time_index (pandas.DatetimeIndex):
        timedelta (pandas.Timedelta): at witch delta a gap is defined

    Returns:
        pandas.DataFrame: with the columns:
            'start' = start-time
            'end' = end-time
    """
    temp = _index_series(date_time_index)

    # start = temp[temp.diff(periods=-1) < -timedelta]
    start = temp[temp.diff(periods=-1).lt(-timedelta)]
    end = temp[temp.diff() > timedelta]

    events = pd.DataFrame()
    events[START] = start.to_list()
    events[END] = end.to_list()
    return events


def gap_table(data, min_gap=Timedelta(minutes=1)):
    """
    time gaps with consist 'NaN' with a minimum span of <min_gap> are the resulting events

    Args:
        data (pandas.DataFrame | pandas.Series):
        min_gap (pandas.Timedelta): minimum time range of an event

    Returns:
        pandas.DataFrame: with the columns:
            'start' = start-time
            'end' = end-time
    """
    if isinstance(data, pd.DataFrame):
        index = data.dropna(axis=0, how='any').index.copy()
        start, end = data.index[[0, -1]]
    elif isinstance(data, pd.Series):
        index = data.dropna().index.copy()
        start, end = data.index[[0, -1]]
    elif isinstance(data, pd.DatetimeIndex):
        index = data.copy()
        start, end = data[[0, -1]]
    else:
        raise NotImplementedError('Wrong data used in <gap_table>: DataFrame or Series - used "{}"'.format(type(data)))

    # to see NaN gaps at the start and the end of the data
    for i in (start, end):
        if i not in index:
            index = index.append(pd.DatetimeIndex([i]))
    index = index.sort_values()

    return time_delta_table(index, timedelta=min_gap)


def span_table(span_bool):
    """
    Create an events-table for time-spans with consistent ``True``-values.

    Args:
        span_bool (pandas.Series[bool]): "True"=Event

    Returns:
        pandas.DataFrame: with the columns:
            'start' = start-time,
            'end' = end-time,
    """
    if span_bool.empty or not span_bool.any():
        return pd.DataFrame(columns=[START, END])

    freq = guess_freq(span_bool.index)
    # minimum duration which is considered as one event

    # pandas.Series with DatetimeIndex as index AND data
    # only with rows where an event occurs
    temp = _index_series(span_bool.index[span_bool])

    # When the duration to previous event value is greater than `min_span` than an event starts.

    # first value in diff will default to NaN
    # fill value is set to double the value of the greater than operation = fixed true value

    # start_bool = temp.diff().gt(freq, fill_value=freq * 2)  # geht nicht bei Monatlichen/Jährlichen Intervall
    start_bool = temp > (temp.shift(fill_value=temp.index[0] - 2 * freq) + freq)

    end_bool = start_bool.shift(-1, fill_value=True)

    events = pd.DataFrame()
    events[START] = temp[start_bool].to_list()
    events[END] = temp[end_bool].to_list()

    return events


########################################################################################################################
def _expanding_time_max(array, ref_time=None):
    """
    calculate the expanding max of a series with datetime data

    Args:
        ref_time (pandas.Timedelta): reference time of the beginning of the series (needed for series with timezone)
                                     TODO: dosen't work with timezone
        array (numpy.ndarray): datetime data

    Returns:
        numpy.ndarray: datetime data
    """
    if array.size == 0:
        return array

    new_array = []
    for a in array:
        if new_array and (a < new_array[-1]):
            new_array.append(new_array[-1])
        else:
            new_array.append(a)
    return new_array


########################################################################################################################
def combine_events(events, new_event_after=Timedelta(hours=4), func=None):
    """
    combine narrow (less than <new_event_after>) respectively overlapping events in table

    old events will be combined by the function defined in <func>

    Args:
        events (pandas.DataFrame): table of events
        new_event_after (pandas.Timedelta): minimum duration between events - otherwise they get combined
        func (dict): dictionary of functions applies for each column, only columns which are in func will be returned

    Returns:
        pandas.DataFrame: table of new-combined events
    """
    if events.empty:
        return events
    if func is None:
        func = {}

    if isinstance(func, dict):
        if START not in func:
            func.update({START: 'min'})

        if END not in func:
            func.update({END: 'max'})

    event_table = events[list(func.keys())].copy()
    event_table['new_end'] = _expanding_time_max(event_table[END], ref_time=event_table[END].iloc[0])
    event_table['new_index'] = (event_table[START] - event_table['new_end'].shift()) > new_event_after
    del event_table['new_end']
    event_table['new_index'] = event_table['new_index'].cumsum() + 1
    event_table = event_table.groupby('new_index').agg(func)
    event_table.reset_index(drop=True, inplace=True)
    return event_table


def combine_events_dwd(events):
    # print('run')
    events = combine_events(events, new_event_after=pd.Timedelta(hours=4))
    events['duration'] = event_duration(events, add_freq_to_duration=pd.Timedelta(minutes=5))
    events['gap_to_previous'] = events[START] - events[END].shift()
    events['gap_to_next'] = events[START].shift(-1) - events[END]
    events['min_duration_next'] = pd.concat([events['duration'], events['duration'].shift(-1)], axis=1).min(axis=1)
    events['combine_with_next_event'] = events['min_duration_next'] > events['gap_to_next']

    if events['combine_with_next_event'].sum() > 0:
        events.loc[events['combine_with_next_event'].shift(fill_value=False), START] = events.loc[events['combine_with_next_event'], START].values
        events = events.loc[~events['combine_with_next_event']]
        # copy start of true to next
        # delete true rows
        return combine_events_dwd(events)

    return events[[START, END]]


########################################################################################################################
def combine_events2(events, new_event_after=Timedelta(hours=4)):
    """
    Combine narrow (less than <new_event_after>) respectively overlapping events in table by adding(!) a new index
    combined events get the same index

    new index is on level=0

    Args:
        events (pandas.DataFrame): table of events
        new_event_after (pandas.Timedelta): minimum duration between events - otherwise they get combined

    Returns:
        pandas.DataFrame: table of events with new top-level index
    """
    new_end = _expanding_time_max(events[END])
    new_end = pd.Series(index=events.index, data=new_end)
    new_index = (events[START] - new_end.shift()) > new_event_after
    return events.set_index(new_index.cumsum().rename('No') + 1, append=True)


########################################################################################################################
def connect_tables(events_1, events_2, source_column, func, preface=pd.Timedelta(minutes=0), **kwargs):
    """
    connect two tables by placing the source_column of events_2 to the events_1

    TODO: do not use

    Args:
        events_1 (pandas.DataFrame): table of events with 'start' and 'end' times
        events_2 (pandas.DataFrame): table of events with 'start' and 'end' times
        source_column (str): column name in <events_2> from where the data should be used
        func (function): function to apply on <events_2>[<source_column>]
        preface (pandas.Timedelta): connect tables where the table2 is the time-span "preface" earlier
        **kwargs: keyword arguments of the function

    Returns:
        pd.Series:
    """
    # is_column_type_str = events_2[source_column].apply(type).isin([str]).all()

    source_is_index = (source_column == 'index') and ('index' not in events_2)
    func_is_iterable = func in (list, set, tuple)

    def _connect_tables(e):
        common_bool = (events_2[START] < e[END]) & (events_2[END] + preface > e[START])

        if source_is_index:
            return common_bool[common_bool].index.to_series().agg(func, **kwargs)
        else:
            values = events_2.loc[common_bool, source_column].values
            if len(values) == 0:
                return
            res = func(values, **kwargs)
            if func_is_iterable and len(res) == 1:
                return res[0]
            # res = events_2.loc[common_bool, source_column].agg(func, **kwargs)
            return res

    return events_1.agg(_connect_tables, axis=1)


def connect_tables2(events_dict, new_event_after=pd.Timedelta(minutes=1)):
    if not events_dict:
        return pd.DataFrame()
    events_frame = None

    for events_name, events_table in events_dict.items():
        events_tab = events_table.copy()
        events_tab.index = pd.MultiIndex.from_product([[events_name], events_tab.index], names=['Event', 'No'])

        if events_tab.empty:
            continue
        elif events_frame is None:
            events_frame = events_tab.copy()
        else:
            events_frame = pd.concat([events_frame, events_tab], axis=0)

    if events_frame is None:
        # every events_table empty
        return pd.DataFrame()

    events_frame = events_frame.sort_values(START)
    events_frame = combine_events2(events_frame, new_event_after=new_event_after).reorder_levels([2, 0, 1])
    return events_frame


def expand_events(events, delta, which='both'):
    """

    Args:
        events (pandas.DataFrame): table of events with 'start' and 'end' times
        delta (pandas.Timedelta):
        which (str): 'start', 'end' or 'both'

    Returns:
        pandas.DataFrame:
    """
    new_events = events.copy()
    if which in [START, 'both']:
        new_events[START] -= delta
    if which in [END, 'both']:
        new_events[END] += delta

    return new_events


########################################################################################################################
def check_times(events, additional=None):
    """
    convert columns to datetime
    ie: when importing the table from a csv file

    Args:
        events (pandas.DataFrame): events table with datetime columns ie. start and end times
        additional (list[str]): list of additional datetime columns

    Returns:
        pandas.DataFrame: events with converted columns
    """
    if additional is None:
        additional = []
    for col in [START, END] + additional:
        if col in events:
            events[col] = pd.to_datetime(events[col])
    return events


def check_deltas(events, additional=None):
    """
    convert columns to timedelta
    ie: when importing the table from a csv file

    Args:
        events (pandas.DataFrame): events table with timedelta columns
        additional (list[str]): list of additional timedelta columns

    Returns:
        pandas.DataFrame: events with converted columns
    """
    if additional is None:
        additional = []
    for col in additional:
        if col in events:
            events[col] = pd.to_timedelta(events[col])
    return events


def event_duration(events, add_freq_to_duration=None):
    """
    calculate the event duration

    Args:
        events (pandas.DataFrame): table of events with 'start' and 'end' times
        add_freq_to_duration (pd.Timedelta): add the frequency of the data to the duration

    Returns:
        pandas.Series: duration of each event
    """
    dur = events[END] - events[START]
    if add_freq_to_duration is not None:
        dur += add_freq_to_duration
    return dur


def duration_to_minutes(duration):
    """
    Convert the duration from Timedelta to minutes as float.

    Args:
        duration (pandas.Series[pandas.Timedelta]):

    Returns:
        pandas.Series[float]: duration in minutes as float
    """
    return duration.dt.total_seconds() / 60


def agg_events(events, series, agg='sum', **kwargs):
    """
    aggregate the series data over the single events

    Args:
        events (pandas.DataFrame): table of events with 'start' and 'end' times
        series (pandas.Series): data
        agg (str | function): aggregation of timeseries

    Returns:
        pandas.Series: result of function of every event
    """
    if isinstance(agg, str):
        def _agg_event(event):
            return series[event[START]:event[END]].agg(agg, **kwargs)
    else:
        def _agg_event(event):
            return agg(series[event[START]:event[END]], **kwargs)

    if events.empty:
        return pd.Series(dtype=object)

    return events.apply(_agg_event, axis=1)


def append_agg_events_frame(events, frame, agg='sum', naming=None):
    if naming is None:
        naming = lambda i: f'sum/{i}'
    events = pd.concat([events, agg_events(events, frame, agg).rename(columns=naming)], axis=1)
    return events


def apply_events(events, data, func, **kwargs):
    """
    apply a function on the data during the single events

    Args:
        events (pandas.DataFrame): table of events with 'start' and 'end' times
        data (pandas.Series | pandas.DataFrame): data
        func (function): function which return one value
        **kwargs: keyword arguments of the function

    Returns:
        pandas.Series: result of function of every event
    """

    def _apply_event(event):
        return func(data[event[START]:event[END]], **kwargs)

    return events.apply(_apply_event, axis=1)


def avail_events(events, series, precision=1):
    """

    Args:
        events (pandas.DataFrame): table of events with 'start' and 'end' times
        series (pandas.Series): data, filled gaps with nans
        precision (int): precision of the percentages

    Returns:
        pandas.Series: result of availability for every event
    """
    return apply_events(events, series.notna(), func=lambda s: s.sum() / s.count()).round(2 + precision) * 100


def filter_events(events, start, end):
    return events[(events[START] < end) & (events[END] > start)]


def iter_events(events):
    for event_no, event in events.iterrows():
        yield event[START], event[END], event.to_dict()


def iter_events_of_series(events, series):
    for start, end, _ in iter_events(events):
        yield series[start:end]


########################################################################################################################
########################################################################################################################
def gap_table4frame(frame, min_gap=Timedelta(minutes=1), new_event_after=None):
    if new_event_after is None:
        new_event_after = min_gap
    events_dict = {}
    for col in frame:
        s = frame[col]
        events_dict[col] = gap_table(s, min_gap=min_gap)

    final_gaps = connect_tables2(events_dict, new_event_after=new_event_after)
    return final_gaps


def add_sub_events_stats(events, event_bool,
                         min_duration=pd.Timedelta(minutes=30),
                         new_event_after=pd.Timedelta(hours=1),
                         prefix=None):
    if prefix is None:
        prefix = ''

    events[f'{prefix}duration'] = None
    events[f'{prefix}#sub_events'] = None
    freq = guess_freq(event_bool.index)
    for event_no, event in events.iterrows():
        sub_events = span_table(event_bool[event.start:event.end])

        sub_events = combine_events(sub_events, new_event_after=new_event_after)

        sub_events['duration'] = event_duration(sub_events, add_freq_to_duration=pd.Timedelta(freq))

        # filter event with just one value
        if not sub_events.empty:
            sub_events = sub_events[sub_events['duration'] > min_duration]

        events.loc[event_no, f'{prefix}#sub_events'] = sub_events.index.size

        if not sub_events.empty:
            events.loc[event_no, f'{prefix}duration'] = sub_events['duration'].sum()
            events.loc[event_no, f'{prefix}start'] = sub_events.start.iloc[0]
            events.loc[event_no, f'{prefix}end'] = sub_events.end.iloc[-1]
        else:
            events.loc[event_no, f'{prefix}duration'] = pd.NaT
            events.loc[event_no, f'{prefix}start'] = pd.NaT
            events.loc[event_no, f'{prefix}end'] = pd.NaT

    # events[f'{prefix}duration'] = events[f'{prefix}duration'].replace(0, pd.NaT)
