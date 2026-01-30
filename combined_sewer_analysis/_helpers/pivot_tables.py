import pandas as pd


def compare_daily_times_table(series):
    """
    create a table where the index is a day and the columns are the times during a day

    Args:
        series (pandas.Series):

    Returns:
        pandas.DataFrame:
    """
    if series.name is None:
        series.name = 'values'
    daily_times_table = series.to_frame()

    daily_times_table['time'] = series.index.time
    daily_times_table['date'] = series.index.date

    if (series.index.tzinfo is not None) and series.index.tz_localize(None).duplicated().any():
        daily_times_table['date'] = series.index.strftime('%Y-%m-%d %Z')

    return daily_times_table.pivot(index='date', columns='time', values=series.name)


def compare_week_table(series, sunday_first=True):
    """
    create a table where the index is a calendarweek of a year and the column is the timedelta since the start of the week

    Args:
        series (pandas.Series):
        sunday_first (bool):

    Returns:
        pandas.DataFrame:
    """
    if series.name is None:
        series.name = 'data'
    daily_times_table = series.to_frame()

    if sunday_first:
        w = series.index.strftime('%w')
    else:
        w = series.index.weekday.astype(str)
    daily_times_table['week_time'] = pd.to_timedelta(w + ' days ' + series.index.strftime('%H:%M:%S'))
    daily_times_table['week'] = series.index.strftime('%Y - KW%W')

    if (series.index.tzinfo is not None) and series.index.tz_localize(None).duplicated().any():
        daily_times_table['week'] = series.index.strftime('%Y - KW%W  %Z')

    return daily_times_table.pivot(index='week', columns='week_time', values=series.name)


def compare_months_table(series):
    """
    create a table where the index is a year and the column is the month

    Args:
        series (pandas.Series):

    Returns:
        pandas.DataFrame:
    """
    monthly_table = series.to_frame()

    monthly_table['year'] = series.index.year
    monthly_table['month'] = series.index.month
    return monthly_table.pivot(index='year', columns='month', values=series.name)
