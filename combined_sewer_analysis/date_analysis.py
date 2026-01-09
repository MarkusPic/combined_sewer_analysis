from pathlib import Path
import calendar
from datetime import date, datetime
import pandas as pd

from pandas import DatetimeIndex, Series, Timedelta, Timestamp, Index, date_range, to_datetime, read_csv

_states = {
    'B': 1,
    'K': 2,
    'N': 3,
    'O': 4,
    'S': 5,
    'ST': 6,
    'T': 7,
    'V': 8,
    'W': 9
}
HOLIDAY_CONFIG = {'country': 'AT',
                  'subdiv': _states['ST']
                  }


class DAY_KIND:
    ALL_DAYS = 'Day'
    WORKDAY = 'Workday'
    NO_WORKDAY = 'Non-working'
    WEEKEND = 'Weekend'
    SATURDAY = 'Saturday'
    SUN_HOLIDAY = 'Sun-&Holiday'
    HOLIDAY = 'Holiday'
    BRIDGE_DAY = 'Bridge Day'
    FAKE_FRIDAY = 'Fake Friday'  # day before a holiday

    @staticmethod
    def sorter(li):
        order = [ALL_DAYS, WORKDAY, *list(calendar.day_name), NO_WORKDAY, WEEKEND, SUN_HOLIDAY, HOLIDAY, BRIDGE_DAY, FAKE_FRIDAY]
        reordered = [item for item in order if item in li] + [item for item in li if item not in order]
        return reordered

ALL_DAYS = 'Day'
WORKDAY = 'Workday'
NO_WORKDAY = 'Non-working'
WEEKEND = 'Weekend'
SATURDAY = 'Saturday'
SUN_HOLIDAY = 'Sun-&Holiday'
HOLIDAY = 'Holiday'
BRIDGE_DAY = 'Bridge Day'
FAKE_FRIDAY = 'Fake Friday'  # day before a holiday

"""
austrian national holidays
"""

"""
https://www.feiertagskalender.ch/ferien.php?geo=3129
"""


def get_school_holidays():
    df = read_csv(Path(__file__).parent / '_helpers' / 'school_holidays_styria_2005-2023.csv',
                  skip_blank_lines=True, skipinitialspace=True, sep=';')
    # "Beginn";"Ende";"Bezeichnung";"Bemerkungen";
    df['Beginn'] = to_datetime(df['Beginn'], format='%d.%m.%Y')
    df['Ende'] = to_datetime(df['Ende'], format='%d.%m.%Y')
    df['Ende'] += Timedelta(days=1, seconds=-1)
    return df


def get_school_holidays_as_index(freq):
    school_holidays = get_school_holidays()
    index = pd.date_range(school_holidays['Beginn'][0], school_holidays['Ende'][-1], freq=freq)
    bool_series = is_school_holiday(index)
    return bool_series[bool_series].index


def is_school_holiday(time_data):
    school_holidays = get_school_holidays()
    if isinstance(time_data, (date, Timestamp, datetime)):
        return ((school_holidays['Beginn'] >= time_data) & (school_holidays['Ende'] <= time_data)).any()

    elif isinstance(time_data, DatetimeIndex):
        if time_data.tz is not None:
            school_holidays['Beginn'] = school_holidays['Beginn'].dt.tz_localize(time_data.tz)
            school_holidays['Ende'] = school_holidays['Ende'].dt.tz_localize(time_data.tz)
        bool_series = pd.Series(index=time_data, data=False)
        for _, holiday_period in school_holidays.iterrows():
            bool_series[holiday_period['Beginn']:holiday_period['Ende']] = True
        return bool_series


def get_holidays(year, **kwargs):
    """
    Get the dates of the austrian national holidays.

    Links:
        https://en.wikipedia.org/wiki/ISO_3166-2:AT
        https://pypi.org/project/holidays/

    Args:
        year (list[int] | int): year(s)

    Returns:
        dict[date,str]: dictionary with the timestamp as the key and the name of the holiday as the value
    """
    import holidays
    return holidays.country_holidays(**{**HOLIDAY_CONFIG, **kwargs}, years=year)


def get_holidays_as_index(year, **kwargs):
    """
    Get the austrian national holidays as date-time-index.

    Args:
        year (list[int] | int): year(s)

    Returns:
        pandas.DatetimeIndex: austrian holidays as index
    """
    return DatetimeIndex(get_holidays(year, **kwargs))


def is_holiday(time_data, **kwargs):
    """
    detect holidays

    Args:
        time_data (date | pandas.DatetimeIndex):

    Returns:
        bool | pandas.Series[bool]:
    """
    if isinstance(time_data, date):
        if isinstance(time_data, Timestamp):
            time_data = time_data.date()
        return time_data in get_holidays(year=time_data.year, **kwargs)

    elif isinstance(time_data, DatetimeIndex):
        years = time_data.year.unique().tolist()
        return Index(time_data.date).isin(get_holidays(years, **kwargs))


def is_fake_friday(time_data, **kwargs):
    """
    detect days before holidays

    Args:
        time_data (date | pandas.DatetimeIndex):

    Returns:
        bool | pandas.Series[bool]:
    """
    return is_holiday(time_data + Timedelta(days=1), **kwargs)


def is_bridge_day(time_data, within_days=1, **kwargs):
    """
    detects days between:
        - two holidays
        - the weekend and a holiday

    Args:
        time_data (date | pandas.DatetimeIndex):
        within_days (int):

    Returns:
        bool | pandas.Series[bool]:
    """
    single_date = False
    if isinstance(time_data, date):
        single_date = True
        date_ = time_data
        if isinstance(date_, Timestamp):
            date_ = date_.date()

        time_data = DatetimeIndex([date_ + Timedelta(days=d) for d in range(-within_days, within_days + 1)])

    if isinstance(time_data, DatetimeIndex):
        time_data_date = time_data.date
        days = date_range(time_data_date.min(), time_data_date.max(), freq='D')
        # days = Series(index=time_data).asfreq('D').index
        weekends = days.dayofweek >= 5
        holidays = is_holiday(days, **kwargs)
        free_days = Series(index=days, data=weekends | holidays)
        bridge_days = (free_days.rolling((within_days * 2 + 1), center=True).sum() == 2) & ~free_days

        if single_date:
            return bridge_days[date_]
        else:
            return Index(time_data_date).isin(bridge_days[bridge_days].index.date)


def get_kind_of_day(time_stamp, level_of_detail=3):
    """
    get the category of the day

    Args:
        time_stamp (pandas.Timestamp):
        level_of_detail (int): in how many categories the dates will be differentiated

    Returns:
        str: category of the day
    """
    if level_of_detail == 1:
        return DAY_KIND.ALL_DAYS

    elif level_of_detail == 2:
        if is_holiday(time_stamp) or (time_stamp.dayofweek >= 5):
            return DAY_KIND.NO_WORKDAY
        else:
            return DAY_KIND.WORKDAY

    elif level_of_detail == 3:
        if is_holiday(time_stamp):
            return DAY_KIND.HOLIDAY
        elif time_stamp.dayofweek >= 5:
            return DAY_KIND.WEEKEND
        else:
            return DAY_KIND.WORKDAY

    elif level_of_detail == 3.1:
        if is_holiday(time_stamp) | (time_stamp.dayofweek == 6):
            return DAY_KIND.SUN_HOLIDAY
        elif time_stamp.dayofweek == 5:
            return DAY_KIND.SATURDAY
        else:
            return DAY_KIND.WORKDAY

    elif level_of_detail == 3.11:  # Metadier et. al 2011
        day = get_kind_of_day(time_stamp, level_of_detail=2)
        if day == DAY_KIND.WORKDAY:
            if is_school_holiday(day):
                return day + ' (inside school holidays)'
            else:
                return day + ' (outside school holidays)'
        else:
            return day

    elif level_of_detail == 8:
        if is_holiday(time_stamp):
            return DAY_KIND.HOLIDAY
        else:
            return time_stamp.day_name()

    elif level_of_detail == 9:
        if is_holiday(time_stamp):
            return DAY_KIND.HOLIDAY
        elif is_fake_friday(time_stamp):
            return DAY_KIND.FAKE_FRIDAY
        else:
            return time_stamp.day_name()

    elif level_of_detail == 10:
        if is_holiday(time_stamp):
            return DAY_KIND.HOLIDAY
        elif is_fake_friday(time_stamp):
            return DAY_KIND.FAKE_FRIDAY
        elif is_bridge_day(time_stamp):
            return DAY_KIND.BRIDGE_DAY
        else:
            return time_stamp.day_name()


def diff_day_type(index, level_of_detail=3., add_number=False, as_series=False):
    """
    Get labels for the kind of the day.

    Args:
        index (pandas.Timestamp | pandas.DatetimeIndex):
        level_of_detail (int | float): in how many categories the dates will be differentiated
        add_number (bool): if the number of the day should be added if the dates will be differentiated in the weekdays
        as_series (bool): if True, return a Series object

    Returns:
        str | pandas.Series: categories of the dates
    """
    # check('DayType0')
    if isinstance(index, Timestamp):
        return get_kind_of_day(index, level_of_detail=level_of_detail)

    elif isinstance(index, DatetimeIndex):
        if level_of_detail == 1:
            return Series(index=index, data=DAY_KIND.ALL_DAYS).values

        holidays = is_holiday(index)
        dayofweek = index.dayofweek.values

        if level_of_detail == 2:
            days = Series(index=index.floor('d'), data=DAY_KIND.WORKDAY)
            days.loc[holidays | (dayofweek >= 5)] = DAY_KIND.NO_WORKDAY

        elif level_of_detail == 3:
            days = Series(index=index.floor('d'), data=DAY_KIND.WORKDAY)
            days.loc[dayofweek >= 5] = DAY_KIND.WEEKEND
            days.loc[holidays] = DAY_KIND.HOLIDAY

        elif level_of_detail == 3.1:
            days = Series(index=index.floor('d'), data=DAY_KIND.WORKDAY)
            days.loc[dayofweek == 5] = DAY_KIND.SATURDAY
            days.loc[holidays | (dayofweek == 6)] = DAY_KIND.SUN_HOLIDAY

        elif level_of_detail == 3.11:  # Metadier et. al 2011
            days = diff_day_type(index, level_of_detail=2)
            bool_school_holiday = is_school_holiday(index)
            days[bool_school_holiday & (days == 'Workday')] += ' (inside school holidays)'
            days[~bool_school_holiday & (days == 'Workday')] += ' (outside school holidays)'
            days = pd.Series(pd.Categorical(days), index=index)

        elif level_of_detail == 7:
            days = Series(index=index.floor('d'), data=index.day_name())

        elif level_of_detail == 8:
            days = Series(index=index.floor('d'), data=index.day_name())
            if add_number:
                days = Series(index=days.index, data=dayofweek).add(1).astype(str).str.cat(days, sep=' ')
                days.loc[holidays] = '8 ' + DAY_KIND.HOLIDAY
            else:
                days.loc[holidays] = DAY_KIND.HOLIDAY

        elif level_of_detail == 9:
            days = Series(index=index.floor('d'), data=index.day_name())
            if add_number:
                days = Series(index=days.index, data=dayofweek).add(1).astype(str).str.cat(days, sep=' ')
                days.loc[holidays] = '8 ' + DAY_KIND.HOLIDAY
                days.loc[is_fake_friday(index)] = '5.1 ' + DAY_KIND.FAKE_FRIDAY
            else:
                days.loc[holidays] = DAY_KIND.HOLIDAY
                days.loc[is_fake_friday(index)] = DAY_KIND.FAKE_FRIDAY

        elif level_of_detail == 10:
            days = Series(index=index.floor('d'), data=index.day_name())
            if add_number:
                days = Series(index=days.index, data=dayofweek).add(1).astype(str).str.cat(days, sep=' ')
                days.loc[holidays] = '8 ' + DAY_KIND.HOLIDAY
                days.loc[is_fake_friday(index)] = '5.1 ' + DAY_KIND.FAKE_FRIDAY
                days.loc[is_bridge_day(index)] = '6.1 ' + DAY_KIND.BRIDGE_DAY
            else:
                days.loc[holidays] = DAY_KIND.HOLIDAY
                days.loc[is_fake_friday(index)] = DAY_KIND.FAKE_FRIDAY
                days.loc[is_bridge_day(index)] = DAY_KIND.BRIDGE_DAY

        else:
            days = Series(index=index.floor('d'))
        # check('DayType1')
        # return pd.CategoricalIndex(days.values)
        if as_series:
            return pd.Series(index=index, data=days.values)

        return days.values
