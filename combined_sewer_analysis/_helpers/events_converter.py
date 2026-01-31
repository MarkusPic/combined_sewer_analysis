import pandas as pd
from numpy import nan, int64, float64

from .freqs import guess_freq
from .events import START, END


def mark_event_bool_v0(events, index):
    """
    TODO: very slow | Alternative: event_to_series()
    get a bool series where True is during events

    Args:
        events (pandas.DataFrame): table with event times, columns=[start, end]
        index (pandas.DatetimeIndex): of the returning series

    Returns:
        pandas.Series: of bool
    """
    events_bool = pd.Series(index=index)
    events_dict = events.to_dict(orient='index')
    for _, event in events_dict.items():
        events_bool[event[START]: event[END]] = 1
    return events_bool == 1


def mark_event_bool_v1(events, index):
    """
    get a bool series where True is during events

    Args:
        events (pandas.DataFrame): table with event times, columns=[start, end]
        index (pandas.DatetimeIndex): of the returning series

    Returns:
        pandas.Series: of bool
    """
    events_bool = pd.Series(index=index, data=0)
    # events during the period of the index
    consider_events = events[(events[END] > index[0]) & (events[START] < index[-1])].copy()

    # if no event is during the period of the index
    if consider_events.empty:
        return events_bool == 1

    shift = guess_freq(index)

    # ______________________
    # define start times

    event_start_times = pd.DatetimeIndex(consider_events[START])
    # if the first event started before the fist index
    if index[0] > event_start_times[0]:
        events_bool.iloc[0] = 1
        event_start_times = event_start_times[event_start_times >= index[0]]

    if index.tzinfo is None:
        events_bool[event_start_times] = 1
    else:
        # locs = [index.get_loc(i) for i in event_start_times]
        events_bool.iloc[[index.get_loc(i) for i in event_start_times]] = 1

    # ______________________
    # define end times

    event_end_times = pd.DatetimeIndex(consider_events[END]) + shift
    # if the last event ends after the last index
    # if index[-1] < consider_events[END].iloc[-1]:
    #     events_bool.iloc[-1] += 1
    event_end_times = event_end_times[event_end_times <= index[-1]]

    if index.tzinfo is None:
        events_bool[event_end_times] += -1
    else:
        events_bool.iloc[[index.get_loc(i) for i in event_end_times]] += -1

    # ______________________
    # cumsum == 1 wenn ein event ist und > 1 wenn mehrere gleichzeitig sind
    return events_bool.cumsum() >= 1


def event_to_series_v0(events, index, data=True, alt=None):
    """
    TODO: very slow | Alternative: event_to_series()

    make a time-series where the value of the <column> in events is paste to the <index>

    Args:
        events (pandas.DataFrame):
        column (str):
        index (pandas.DatetimeIndex):

    Returns:
        pandas.Series:
    """
    to_event_bool = isinstance(data, bool)
    if to_event_bool:
        alt = not data
        value = 1
    ts = pd.Series(index=index, dtype=bool if to_event_bool else float)

    # for event_no, event in events.iterrows():
    events_dict = events.to_dict(orient='index')
    for event_no, event in events_dict.items():
        if to_event_bool:
            ts[event[START]: event[END]] = value
        else:
            ts[event[START]: event[END]] = event[data]
            # ts[event[START]: event[END]] = event_no

    # if not to_event_bool:
    #     ts = events.loc[ts.values, data]

    if to_event_bool:
        ts = ts == value

    if alt is not None:
        return ts.fillna(alt)

    return ts


def event_to_series_v1(events, index, data=True):
    """
    get a bool series where True is during events

    Args:
        events (pandas.DataFrame): table with event times, columns=[start, end]
        index (pandas.DatetimeIndex): of the returning series
        data (str | bool):
                    bool: events will be returned as true others as false
                    str: events will be returned as value of the events column = data (only numeric data) others are 0

    Returns:
        pandas.Series: of bool or values
    """
    start_times = pd.DatetimeIndex(events[START])
    freq = guess_freq(index)
    end_times = pd.DatetimeIndex(events[END]) + freq
    appendix = start_times.append(end_times)

    if _ := 0:
        if appendix.isin(index).all():
            index_all = index
        else:
            index_all = index.union(appendix)  # FASTER than index.append(appendix).drop_duplicates().sort_values()
    else:
        index_all = index.union(appendix)

    bool_series = pd.Series(index=index_all, data=0)

    if isinstance(data, bool):
        if _ := 1:
            bool_series.iloc[[index_all.get_loc(i) for i in start_times]] = 1
        else:
            bool_series[start_times] = 1

        if _ := 1:
            bool_series.iloc[[index_all.get_loc(i) for i in end_times]] = -1
        else:
            bool_series[end_times] = -1

        bool_series = bool_series.cumsum() >= 1
        if not data:
            bool_series = ~bool_series

    elif isinstance(data, str):
        assert data in events

        bool_series[start_times] = events[data]
        bool_series[end_times] = -1 * events[data]
        bool_series = bool_series.cumsum().replace(0, nan)

    else:
        raise NotImplementedError(str(type(data)) + ' is not implement!')

    return bool_series[index].copy()


def event_to_series_v2(events, index, data=True, alt=None):
    if isinstance(data, bool):
        alt = not data

    events_bool = pd.Series(index=index, data=alt)

    event_index = pd.DatetimeIndex([])
    if isinstance(data, bool):
        event_values = True
    else:
        event_values = []
    events_dict = events.to_dict(orient='index')
    for _, event in events_dict.items():
        new_ind = events_bool[event[START]: event[END]].index
        if not isinstance(data, bool):
            event_values += [event[data]] * len(new_ind)
        event_index = event_index.append(new_ind)
    events_bool.loc[event_index] = event_values
    return events_bool


def event_to_series(events, index, data=True, alt=None):
    if index.size > 1000000 and events.index.size < 2000:
        return event_to_series_v0(events, index, data)

    if isinstance(data, bool):
        #     if index.size > 1000000:
        #         return mark_event_bool_v0(events, index)
        #     else:
        return mark_event_bool_v1(events, index)

    if events[data].dtype in [int64, float64]:
        return event_to_series_v1(events, index, data)
    else:
        return event_to_series_v2(events, index, data)


########################################################################################################################


# def event_to_series_v3(events, index, data=True, alt=None):
#     event_index = []
#     event_values = []
#     index_l = index.tolist()
#     to_event_bool = isinstance(data, bool)
#     for event in events.to_dict(orient='index').values():
#         new_ind = index_l[index_l.index(event[START]): index_l.index(event[END])]
#         if not to_event_bool:
#             event_values += [event[data]] * len(new_ind)
#         event_index += new_ind
#
#     if to_event_bool:
#         event_values = data
#
#     events_bool = pd.Series(index=event_index, data=event_values)
#
#     if to_event_bool:
#         events_bool = events_bool.fillna(not event_values)
#     return events_bool

mark_event_bool = mark_event_bool_v1
