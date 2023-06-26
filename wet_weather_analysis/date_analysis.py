from pathlib import Path

from datetime import date
import holidays

from pandas import DatetimeIndex, Series, Timedelta, Timestamp, Index, date_range, to_datetime, read_csv


class DAY_KIND:
    ALL_DAYS = 'Day'
    HOLIDAY = 'Holiday'
    WORKDAY = 'Workday'
    WEEKEND = 'Weekend'
    NO_WORKDAY = 'Non-working'
    SUN_HOLIDAY = 'Sun-&Holiday'
    SATURDAY = 'Saturday'
    BRIDGE_DAY = 'Bridge Day'
    FAKE_FRIDAY = 'Fake Friday'  # day before a holiday

ALL_DAYS = 'Day'
HOLIDAY = 'Holiday'
WORKDAY = 'Workday'
WEEKEND = 'Weekend'
NO_WORKDAY = 'Non-working'
SUN_HOLIDAY = 'Sun-&Holiday'
SATURDAY = 'Saturday'
BRIDGE_DAY = 'Bridge Day'
FAKE_FRIDAY = 'Fake Friday'  # day before a holiday

"""
austrian national holidays
"""


"""
https://www.feiertagskalender.ch/ferien.php?geo=3129
"""

def get_school_holidays():
    df = read_csv(Path(__file__) / '_helpers' / 'school_holidays_styria_2005-2023.csv',
                  skip_blank_lines=True, skipinitialspace=True, sep=';')
    # "Beginn";"Ende";"Bezeichnung";"Bemerkungen";
    df['Beginn'] = to_datetime(df['Beginn'], format='%d.%m.%Y')
    df['Ende'] = to_datetime(df['Ende'], format='%d.%m.%Y')
    df['Ende'] += Timedelta(days=1, seconds=-1)
    return df


def get_holidays(year, state='ST'):
    """
    the dates of the austrian national holidays

    Args:
        year (list[int] | int): year(s)
        state (str): which of the nine austrian provinces
                        ['B', 'K', 'N', 'O', 'S', 'ST', 'T', 'V', 'W']

    Returns:
        dict[date,str]: dictionary with the timestamp as the key and the name of the holiday as the value
    """
    return holidays.Austria(state=state, years=year)


def get_holidays_as_index(year, state='ST'):
    """
    the the dates a the austrian national holidays

    Args:
        year (list[int] | int): year(s)
        state (str): which of the nine austrian provinces
                        ['B', 'K', 'N', 'O', 'S', 'ST', 'T', 'V', 'W']

    Returns:
        pandas.DatetimeIndex: austrian holidays as index
    """
    return DatetimeIndex(get_holidays(year, state))


def is_holiday(time_data):
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
        return time_data in get_holidays(year=time_data.year)

    elif isinstance(time_data, DatetimeIndex):
        years = time_data.year.unique().tolist()
        return Index(time_data.date).isin(get_holidays(years))


def is_fake_friday(time_data):
    """
    detect days before holidays

    Args:
        time_data (date | pandas.DatetimeIndex):

    Returns:
        bool | pandas.Series[bool]:
    """
    return is_holiday(time_data + Timedelta(days=1))


def is_bridge_day(time_data, within_days=1):
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
        days = date_range(time_data.min(), time_data.max(), freq='D')
        # days = Series(index=time_data).asfreq('D').index
        weekends = days.dayofweek >= 5
        holidays = is_holiday(days)
        free_days = Series(index=days, data=weekends | holidays)
        bridge_days = (free_days.rolling((within_days * 2 + 1), center=True).sum() == 2) & ~free_days

        if single_date:
            return bridge_days[date_]
        else:
            return Index(time_data.date).isin(bridge_days[bridge_days].index.date)


def get_kind_of_day(time_stamp, level_of_detail=3):
    """
    get the category of the day

    Args:
        time_stamp (pandas.Timestamp):
        level_of_detail (int): in how much categories the dates will be differentiated

    Returns:
        str: category of the day
    """
    if level_of_detail == 1:
        return ALL_DAYS

    elif level_of_detail == 2:
        if is_holiday(time_stamp) or (time_stamp.dayofweek >= 5):
            return NO_WORKDAY
        else:
            return WORKDAY

    elif level_of_detail == 3:
        if is_holiday(time_stamp):
            return HOLIDAY
        elif time_stamp.dayofweek >= 5:
            return WEEKEND
        else:
            return WORKDAY

    elif level_of_detail == 3.1:
        if is_holiday(time_stamp) | (time_stamp.dayofweek == 6):
            return SUN_HOLIDAY
        elif time_stamp.dayofweek == 5:
            return SATURDAY
        else:
            return WORKDAY

    elif level_of_detail == 8:
        if is_holiday(time_stamp):
            return HOLIDAY
        else:
            return time_stamp.day_name()

    elif level_of_detail == 9:
        if is_holiday(time_stamp):
            return HOLIDAY
        elif is_fake_friday(time_stamp):
            return FAKE_FRIDAY
        else:
            return time_stamp.day_name()

    elif level_of_detail == 10:
        if is_holiday(time_stamp):
            return HOLIDAY
        elif is_fake_friday(time_stamp):
            return FAKE_FRIDAY
        elif is_bridge_day(time_stamp):
            return BRIDGE_DAY
        else:
            return time_stamp.day_name()


def diff_day_type(index, level_of_detail=3., add_number=False):
    """
    Get labels for the kind of the day.

    Args:
        index (pandas.Timestamp | pandas.DatetimeIndex):
        level_of_detail (int | float): in how many categories the dates will be differentiated
        add_number (bool): if the number of the day should be added if the dates will be differentiated in the weekdays

    Returns:
        str | pandas.Series: categories of the dates
    """
    # check('DayType0')
    if isinstance(index, Timestamp):
        return get_kind_of_day(index, level_of_detail=level_of_detail)

    elif isinstance(index, DatetimeIndex):
        if level_of_detail == 1:
            return Series(index=index, data=ALL_DAYS).values

        holidays = is_holiday(index)
        dayofweek = index.dayofweek.values

        if level_of_detail == 2:
            days = Series(index=index.floor('D'), data=WORKDAY)
            days.loc[holidays | (dayofweek >= 5)] = NO_WORKDAY

        elif level_of_detail == 3:
            days = Series(index=index.floor('D'), data=WORKDAY)
            days.loc[dayofweek >= 5] = WEEKEND
            days.loc[holidays] = HOLIDAY

        elif level_of_detail == 3.1:
            days = Series(index=index.floor('D'), data=WORKDAY)
            days.loc[dayofweek == 5] = SATURDAY
            days.loc[holidays | (dayofweek == 6)] = SUN_HOLIDAY

        elif level_of_detail == 7:
            days = Series(index=index.floor('D'), data=index.day_name())

        elif level_of_detail == 8:
            days = Series(index=index.floor('D'), data=index.day_name())
            if add_number:
                days = Series(index=days.index, data=dayofweek).add(1).astype(str).str.cat(days, sep=' ')
                days.loc[holidays] = '8 ' + HOLIDAY

            else:
                days.loc[holidays] = HOLIDAY

        elif level_of_detail == 9:
            days = Series(index=index.floor('D'), data=index.day_name())
            days.loc[holidays] = HOLIDAY
            days.loc[is_fake_friday(index)] = FAKE_FRIDAY

        elif level_of_detail == 10:
            days = Series(index=index.floor('D'), data=index.day_name())
            days.loc[holidays] = HOLIDAY
            days.loc[is_fake_friday(index)] = FAKE_FRIDAY
            days.loc[is_bridge_day(index)] = BRIDGE_DAY

        else:
            days = Series(index=index.floor('D'))
        # check('DayType1')
        # return pd.CategoricalIndex(days.values)
        return days.values