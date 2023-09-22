from functools import wraps
from os import path
from pathlib import Path
from sys import platform
from typing import Literal

import numpy as np
import pandas as pd
from numpy import NaN
from pandas import Timedelta
from pandas.core.groupby import GroupBy

from ._helpers.debug_helpers import timeit, lev
from ._helpers.calculation_helpers import calc_dry_mean, calc_dry_variation_split
from ._helpers.pickle_helpers import read_pickle, write_pickle
# from .definitions import *
from .definitions import MEDIAN__MAD
from .date_analysis import diff_day_type

from mp.libs.timeseries.stats.events import combine_events, span_table, event_duration
from mp.libs.timeseries.stats.events_converter import mark_event_bool
from mp.libs.timeseries.stats.freqs import guess_freq
from mp.libs.timeseries.stats.wastewater import calculate_load_rate

# from mp.projects.cst_monitoring.helpers import check
# from .plots.figures import AnalysePlots
# import pyarrow as pa
# import pyarrow.parquet as pq


def isfile(fn):
    return path.isfile(f'{fn}.parq') or path.isfile(f'{fn}_{platform}.pkl')


# FILETYPE = ''


DW_CRITERION = 'DW-CRITERION'
DW_MEAN = 'DW-MEAN'
DW_LEVEL = 'DW-LEVEL'
DW_CONTINUUM = 'DW-CONTINUUM'
DW_BOOL = 'DW-BOOL'
DW_AVAILABILITY = 'DW-AVAILABILITY'
LOWER = 'LOWER'
MEAN = 'MEAN'
UPPER = 'UPPER'
AUTO = 'auto'

# def shift2delta(reference_time):
#     if reference_time == '00:00':
#         return None
#     split_times = [int(t) for t in reference_time.split(':')]
#     return pd.Timedelta(hours=split_times[0], minutes=split_times[1])


def smoother(method):
    """Smooth results of function. Default: 20min center rolling mean"""
    @wraps(method)
    def _smoother(*args, **kwargs):
        if 'smooth' in kwargs:
            smooth = kwargs.pop('smooth')
        else:
            smooth = args[0].smooth_window

        result = method(*args, **kwargs)  # type: pd.Series

        if smooth is None or smooth == 1:
            return result
        else:
            return result.rolling(window=smooth, center=True, min_periods=1).mean()

    return _smoother


########################################################################################################################
class AnalyseData:
    """
    Attributes:
        shift_delta (Timedelta): delta to shift the time-series. Time when the values of all days are the most simular.
        smooth_window (int): Number of values used to smooth results. Default=20 -> with a frequency of 1 min -> smooth-windows=20 min
    """
    @staticmethod
    def from_pickle(fn):
        """
        :type fn: str
        :rtype: AnalyseData
        """
        return read_pickle(fn)

    def __str__(self):
        return 'AnalyseData({}, kind={}, dkd={})'.format(self.name, self.arithmetic, self.day_kind_detail)

    def __init__(self, ts, kind=MEDIAN__MAD, limit=2, day_kind_detail=None, ww_crit_limit=100, dw_crit_limit=100,
                 make_temp_files=False, file_path='.',
                 est_best_shift_time=True,
                 min_rain_period=Timedelta(hours=2),
                 trail_period=Timedelta(hours=4)):  # day_kind_detail=3.1, reference_time='00:00'
        """

        Args:
            ts (pd.Seres):
            kind (int): 0,6,8,97,98,1,2,7,99
            limit (float): multiplicative of MAD (median of absolute difference) which is stiff dry-weather.
            day_kind_detail (int | float): 1,2,3,3.1,7,8,9,10 | weekdays, holiday, bridge-day, fake-friday, weekend,
            ... | default=automated
            make_temp_files (bool): Whether to make temporary files.
            file_path (str | Path): Path where the temporary files should be saved.
            est_best_shift_time (bool): If the time-shift should be automatically estimated.
            min_rain_period (Timedelta): Minimum duration from which it is a rain event. Shorter events will be ignored.
            trail_period (Timedelta): Duration for combining rain events + duration after an event to restore dw-conditions.
        """
        self.ts = ts.replace(0, NaN)
        # kind of calculation method for the dw-mean and the dw-variance
        self.arithmetic = kind
        self.limit = limit
        self.name = ts.name
        self.dw_crit_limit = dw_crit_limit
        self.ww_crit_limit = ww_crit_limit

        # temporary file path
        self.temp_file_path = Path(file_path)
        self.make_temp_files = make_temp_files

        # ------
        # per day and time
        # columns(day) index(day_time)
        self._agg_dw_mean = None  # DataFrame - <daykinds>
        self._agg_dw_variance = None  # {}  # of DataFrames 'UPPER', 'LOWER', 'MEAN' - <daykinds>
        self._dw_uncertainty = None  # DataFrame - <daykinds>
        # ----
        # time series
        self.dw_mean = None  # pd.Series(name=DW_MEAN)
        self.dw_var = None  # pd.DataFrame(columns=[UPPER, LOWER])
        self.dw_range = None  # pd.DataFrame(columns=[UPPER, LOWER])

        self.criterion = None  # pd.Series(name=DW_CRITERION)

        self.criterion_level = None  # pd.Series(name=DW_LEVEL)
        self.dry_continuum = None  # pd.Series(name=DW_CONTINUUM)
        self.dry_weather_bool = None  # pd.Series(name=DW_BOOL)
        self.dry_weather_avail = None  # pd.Series(name=DW_AVAILABILITY)

        self.cont = None  # pd.DataFrame()  # upper, mean, lower

        # ----
        # table
        self.dry_weather_span_table = None
        self.wet_weather_span_table = None

        # # analyse helpers
        # add a number to the day labels
        self._number_day_labels = False  # bool

        # over time and day-kind
        self._grouper_analysis = None  # pd.Grouper
        # over day-kind
        self._grouper_daily = None  # pd.Grouper

        # 1:(day)
        # 2:(workday / non-working-day)
        # 3:(workday / weekend / holiday)
        # 3.1:(weekdays / Saturday / Sun-& Holiday
        # 8:("weekday_name" / holiday)

        if day_kind_detail is None:
            day_kind_detail = 'best'

        self.day_kind_detail = day_kind_detail
        self._day_category_index = None  # pd.CategoricalIndex

        # time series
        self.shift_delta: Timedelta = None  # pd.Timedelta(minutes=0)  # self.shift2delta(reference_time)
        self._shifted_ts = self.ts.copy()  # ts only for day-kind calculation

        if (day_kind_detail != 1) and est_best_shift_time:
            from .estimate_parameters import est_best_shift_time
            est_best_shift_time(self)
        # self.day_category_index
        # print()
        # self.est_best_daily_grouping()

        # self.plot = AnalysePlots(self)

        # internal
        self._lookup_index = None
        self._lookup_columns = None

        # ---------------
        self.smooth_window = 20

        # ----------------
        # duration for event detection
        self.min_rain_period = min_rain_period  # Minimum duration from which it is a rain event. Shorter events will be ignored.
        self.trail_period = trail_period  # Duration for combining rain events + duration after an event to restore dw-conditions.
        self.min_dry_period = self.trail_period  # Minimum duration from which it is a dry period. Shorter periods will be ignored.

    # HELPERS-----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def check_args(self, **kwargs):  # kind, limit
        args_bool = False

        if 'kind' in kwargs:
            kind = kwargs['kind']
            kind_bool = bool(kind) and (self.arithmetic != kind)
            if kind_bool:
                self.arithmetic = kind
            args_bool |= kind_bool

        return args_bool

    def filename(self, basename):
        if not self.make_temp_files:
            return ''

        else:
            folder = f'{self.name} # kind={self.day_kind_detail} # shift={self.shift_label}'.replace('.', '-')
            dir_ = self.temp_file_path / folder
            if not dir_.exists():
                dir_.mkdir()  # parents=True

            return dir_ / basename

    # ------------------------------------------------------------------------------------------------------------------
    def set_number_day_labels(self):
        self._number_day_labels = True
        return self

    # ------------------------------------------------------------------------------------------------------------------
    def shift_times(self):
        if self.shift_delta is None:
            self._shifted_ts.index = self.ts.index.copy()
        else:
            self._shifted_ts.index = self.ts.index.copy() - self.shift_delta

    def manual_time_shift(self, shift_delta):
        self.shift_delta = shift_delta
        self.shift_times()
        return self

    @property
    def shift_label(self):
        if (self.shift_delta is None) or (self.shift_delta == Timedelta(hours=0)):
            return '-'
        return ''.join(['{}{}'.format(item, key[0] + ('s' if ('seconds' in key and key != 'seconds') else '')) for key, item in self.shift_delta.components._asdict().items() if item != 0])

    @property
    def guessed_freq(self):
        return guess_freq(self.ts.index)

    def get_window_size(self, window: pd.Timedelta):
        if isinstance(window, int):
            return window
        return int(round(window / self.guessed_freq))

    # ------------------------------------------------------------------------------------------------------------------
    @timeit
    def get_diff_day_type(self, index):
        return diff_day_type(index,
                             level_of_detail=self.day_kind_detail,
                             add_number=self._number_day_labels)

    @property
    def day_category_index(self) -> pd.CategoricalIndex:
        if self._day_category_index is None:
            fn = self.filename(f'day_category_index # num={self._number_day_labels}')

            if isfile(fn):
                self._day_category_index = self._read(fn, dtype=pd.Series).values
            else:
                if self.day_kind_detail == 'best':
                    from wet_weather_analysis.estimate_parameters import est_best_daily_grouping
                    est_best_daily_grouping(self)

                else:
                    # self.est_best_shift_time()
                    self._day_category_index = self.get_diff_day_type(self._shifted_ts.index)
                    # self._day_category_index = diff_day_type(self._shifted_ts.index,
                    #                                          level_of_detail=self.day_kind_detail,
                    #                                          add_number=self._number_day_labels)
                self._write(pd.Series(self._day_category_index, name='Day Category Index'), fn)

        return self._day_category_index

    def get_day_categories(self):
        return self.day_category_index.categories.tolist()

    @timeit
    def get_analysis_grouper(self):
        """
        Groups data in [day, time] groups.

        Returns:
            GroupBy: day-kind and time groups
        """
        if self._grouper_analysis is None:
            fn = self.filename('analysis_group')
            if isfile(fn):
                self._grouper_analysis = self._read(fn)
            else:
                self._grouper_analysis = self.ts.groupby([self.day_category_index, self.ts.index.time])
                self._write(self._grouper_analysis, fn)
        return self._grouper_analysis

    @timeit
    def get_day_grouper(self):
        """
        Get the daily groups depending from the level of detail of the analysis.

        Returns:
            GroupBy: Data grouped by day-kind.
        """
        if self._grouper_daily is None:
            fn = self.filename('day_group')
            if isfile(fn):
                self._grouper_daily = self._read(fn)
            else:
                self._grouper_daily = self.ts.groupby(self.day_category_index)
                self._write(self._grouper_daily, fn)

        return self._grouper_daily

    # AGGREGATIONS------------------------------------------------------------------------------------------------------
    @timeit
    def get_dw_mean_table(self, arithmetic):
        """
        Aggregate data für analysis groups and calculate the dry-mean.

        Read from file or calculate and write to file.

        Args:
            arithmetic (float): kind of dry-mean arithmetic.

        Returns:
            pd.DataFrame: index=day-times | columns=day-kinds
        """
        fn = self.filename(f'agg_dry_mean # arithmetic={arithmetic}')
        if isfile(fn):
            return self._read(fn)
        else:
            agg_dw_mean = self.get_analysis_grouper().agg(calc_dry_mean, kind=arithmetic).unstack(0)
            self._write(agg_dw_mean, fn)
            return agg_dw_mean

    @smoother
    def dw_mean_table(self, arithmetic=None):
        """
        Aggregate data für analysis groups and calculate the dry-mean.

        Args:
            arithmetic (float): kind of dry-mean arithmetic.

        Returns:
            pd.DataFrame: index=day-times | columns=day-kinds
        """
        if arithmetic is not None:
            return self.get_dw_mean_table(arithmetic=arithmetic)

        if self._agg_dw_mean is None:
            self._agg_dw_mean = self.get_dw_mean_table(arithmetic=self.arithmetic)
        return self._agg_dw_mean

    # ------------------------------------------------------------------------------------------------------------------
    @timeit
    def get_dw_variance_table(self, arithmetic):
        fn = self.filename(f'agg_dry_var # arithmetic={arithmetic}')
        if isfile(fn):
            return self._read(fn)
        else:
            agg_dw_mean = self.get_dw_mean_table(arithmetic=arithmetic)

            def _vars(s):
                upper, lower = calc_dry_variation_split(s, kind=self.arithmetic,
                                                        infer_mean=float(agg_dw_mean.loc[s.name[1], s.name[0]]),
                                                        time_stamp=s.name[1])
                return {UPPER: upper, LOWER: lower, MEAN: np.mean([upper, lower])}

            variances = self.get_analysis_grouper().apply(_vars)  # multiindex: day - time - (lower/upper)
            variances = variances.unstack([2, 0])

            self._write(variances, fn)
            return variances

    @smoother
    def dw_variance_table(self, arithmetic=None):
        if arithmetic is not None:
            return self.get_dw_variance_table(arithmetic=arithmetic)

        if self._agg_dw_variance is None:
            self._agg_dw_variance = self.get_dw_variance_table(arithmetic=self.arithmetic)
        return self._agg_dw_variance

    # ------------------------------------------------------------------------------------------------------------------
    @timeit
    @smoother
    def get_dw_bound_table(self, arithmetic=None, limit=None):
        if arithmetic is not None:
            dry_mean = self.get_dw_mean_table(arithmetic=arithmetic)
            variance = self.get_dw_variance_table(arithmetic=arithmetic)

        else:
            dry_mean = self.dw_mean_table(smooth=1)
            variance = self.dw_variance_table(smooth=1)

        if limit is None:
            limit = self.limit

        bound = variance.copy()
        del bound[MEAN]
        bound.columns = bound.columns.remove_unused_levels()
        for sign, side in [(-1, LOWER), (1, UPPER)]:
            bound[side] = dry_mean + (variance[side] + variance[MEAN] * (limit - 1)) * sign

        return bound

    # TRANSFORMATIONS---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _lookup(self, data):
        if self._lookup_index is None:
            idx1, cols1 = pd.factorize(self.day_category_index)
            self._lookup_index = (idx1, cols1)
        else:
            idx1, cols1 = self._lookup_index

        if self._lookup_columns is None:
            idx0, cols0 = pd.factorize(self.ts.index.time)
            self._lookup_columns = (idx0, cols0)
        else:
            idx0, cols0 = self._lookup_columns

        return data.reindex(cols1, axis=1).reindex(cols0, axis=0).to_numpy()[idx0, idx1]

    # ------------------------------------------------------------------------------------------------------------------
    # @timeit
    @smoother
    def get_dw_mean_series(self, arithmetic=None) -> pd.Series:
        if arithmetic is not None:
            return pd.Series(index=self.ts.index, data=self._lookup(self.get_dw_mean_table(arithmetic=arithmetic)), name=DW_MEAN)

        if self.dw_mean is None:
            dw_mean_table = self.dw_mean_table(smooth=1)

            # self.dw_mean = pd.Series(index=self.ts.index,
            #                          data=dw_mean_table.lookup(self.ts.index.time, self.day_category_index),
            #                          name=DW_MEAN)

            # self.dw_mean = pd.Series(index=self.ts.index,
            #                          data=dw_mean_table.unstack().loc[zip(self.day_category_index, self.ts.index.time)].values,
            #                          name=DW_MEAN)

            self.dw_mean = pd.Series(index=self.ts.index,
                                     data=self._lookup(dw_mean_table),
                                     name=DW_MEAN)

        return self.dw_mean

    # ------------------------------------------------------------------------------------------------------------------
    # @timeit
    @smoother
    def get_dw_variance_series(self, arithmetic=None) -> pd.DataFrame:
        if arithmetic is not None:
            dw_variance_table = self.get_dw_variance_table(arithmetic=arithmetic)
            dw_var = pd.DataFrame(index=self.ts.index)
            for col in dw_variance_table.columns.levels[0]:
                dw_var[col] = self._lookup(dw_variance_table[col])
            return dw_var

        if self.dw_var is None:
            dw_variance_table = self.dw_variance_table(smooth=1)
            self.dw_var = pd.DataFrame(index=self.ts.index)
            for col in dw_variance_table.columns.levels[0]:
                self.dw_var[col] = self._lookup(dw_variance_table[col])

        return self.dw_var

    # ------------------------------------------------------------------------------------------------------------------
    # @timeit
    @smoother
    def get_dw_range_series(self, arithmetic=None, limit=None) -> pd.DataFrame:
        if arithmetic is not None:
            dw_bound_table = self.get_dw_bound_table(arithmetic=arithmetic, limit=limit)
            dw_range = pd.DataFrame(index=self.ts.index)
            for col in dw_bound_table.columns.levels[0]:
                dw_range[col] = self._lookup(dw_bound_table[col])
            return dw_range

        if self.dw_range is None:
            dw_bound_table = self.get_dw_bound_table(limit=limit, smooth=1)
            self.dw_range = pd.DataFrame(index=self.ts.index)
            for col in dw_bound_table.columns.levels[0]:
                self.dw_range[col] = self._lookup(dw_bound_table[col])

        return self.dw_range

    @timeit
    def _calc_criterion(self, arithmetic=None, accuracy=1e-5) -> pd.Series:
        fn = self.filename(f'crit # arithmetic={self.arithmetic if arithmetic is None else arithmetic}')
        if isfile(fn):
            return self._read(fn, dtype=pd.Series)
        else:
            diff = self.ts - self.get_dw_mean_series(arithmetic, smooth=None)
            var = self.get_dw_variance_series(arithmetic, smooth=None)

            lower = diff < -accuracy
            higher = diff > accuracy
            crit = diff.copy()
            crit[(diff > -accuracy) & (diff < accuracy)] = 0
            crit[lower] /= var.loc[lower, LOWER]
            crit[higher] /= var.loc[higher, UPPER]
            crit *= 100
            criterion = crit.rename(DW_CRITERION)
            self._write(criterion, fn)
            return criterion

    @smoother
    def get_criterion_series(self, arithmetic=None, limit=None) -> pd.Series:
        """float between 0 (=DW-Mean) and inf where 100 is estimated the maximum DW. NaN if NaN in timeseries!"""
        if arithmetic is not None:
            # calculate custom crit
            return self._calc_criterion(arithmetic=arithmetic).divide(self.limit if limit is None else limit)

        if self.criterion is None:
            self.criterion = self._calc_criterion()
        return self.criterion.divide(self.limit if limit is None else limit)

    # ------------------------------------------------------------------------------------------------------------------
    def get_wet_weather_span_table(self, min_rain_period=None, trail_period=None):
        """
        Get table with wet weather events with a minimum period and combine events which are closer than a tail period.

        Gaps (NaN) in the timeseries will be defaulted to dry weather.

        Args:
            min_rain_period (Timedelta): Minimum duration that counts as a rain event / wet weather period.
            trail_period (Timedelta): Nachlaufzeit | minimum duration to separate following events.

        Returns:
            pd.DataFrame: events with start and end times
        """
        if self.wet_weather_span_table is None:
            # NaNs are assumed to be dry weather
            criterion_bool = self.get_criterion_series().fillna(0) > self.ww_crit_limit

            wet_weather_table = span_table(span_bool=criterion_bool)
            # it is only a wet-weather-event when it is longer than "min_rain_period"

            if min_rain_period is None:
                min_rain_period = self.min_rain_period

            if trail_period is None:
                trail_period = self.trail_period

            wet_weather_table = wet_weather_table[event_duration(wet_weather_table) >= min_rain_period]

            # combine close events
            self.wet_weather_span_table = combine_events(wet_weather_table, new_event_after=trail_period)
        return self.wet_weather_span_table

    def get_wet_weather_span_table2(self, min_rain_period=None, trail_period=None):
        """
        Event definition:
            Value must be greater than DW-continuum + 2 x DW-uncertainty

        Args:
            min_rain_period:
            trail_period:

        Returns:

        """
        if self.wet_weather_span_table is None:
            # NaNs are assumed to be dry weather
            criterion_bool = self.ts > (self.get_dw_continuum_series() + self.get_dw_uncertainty_series()*2)

            # criterion_bool = self.get_criterion_series().fillna(0) > self.ww_crit_limit

            wet_weather_table = span_table(span_bool=criterion_bool)
            # it is only a wet-weather-event when it is longer than "min_rain_period"

            if min_rain_period is None:
                min_rain_period = self.min_rain_period

            if trail_period is None:
                trail_period = self.trail_period

            wet_weather_table = wet_weather_table[event_duration(wet_weather_table) >= min_rain_period]

            # combine close events
            self.wet_weather_span_table = combine_events(wet_weather_table, new_event_after=trail_period)
        return self.wet_weather_span_table

    def get_dry_weather_table(self, min_dry_period=None):
        """
        Get table with dry weather period with a minimum period.

        Gaps (NaN) in the timeseries will be defaulted to wet weather.

        Args:
            min_dry_period (Timedelta): Minimum duration that counts as a dry period.

        Returns:
            pd.DataFrame: events with start and end times
        """
        if self.dry_weather_span_table is None:
            # NaNs are assumed to be wet weather
            criterion_bool = self.get_criterion_series().fillna(self.dw_crit_limit+1) < self.dw_crit_limit

            dry_weather_table = span_table(span_bool=criterion_bool)
            # it is only a dw event (period) when it is longer than "min_dry_period" dry.

            if min_dry_period is None:
                min_dry_period = self.min_dry_period

            dry_weather_table = dry_weather_table[event_duration(dry_weather_table) >= min_dry_period]

            self.dry_weather_span_table = dry_weather_table
        return self.dry_weather_span_table

    @timeit
    def get_dry_weather_bool(self, min_rain_period=None, extra_range=None, fill_na=NaN):
        """
        Mark wet weather periods (including a tail = extra_range) with False and dry weather periods as True.

        NaN are equal to NaN in Timeseries.

        Short events will not have trail-periods and will be ignored.

        Args:
            min_rain_period (pandas.Timedelta): minimum period to count as a rain-event_analysis
            extra_range (pandas.Timedelta): extra time between a dry period and wet weather (no more influence)
            fill_na: value to fill na values in original series

        Returns:
            pd.Series[bool]: condition of DW-period
        """
        if self.dry_weather_bool is None:

            if min_rain_period is None:
                min_rain_period = self.min_rain_period

            if extra_range is None:
                extra_range = self.min_dry_period

            ww_period_table = self.get_wet_weather_span_table(min_rain_period=min_rain_period, trail_period=extra_range).copy()

            # extend rain events
            ww_period_table['end'] += extra_range

            index = self.ts.index

            # so the extended end is not longer than the series
            ww_period_table['end'] = ww_period_table['end'].clip(upper=index[-1])

            dw_period_table = self.get_dry_weather_table(min_dry_period=extra_range).copy()

            # make dry period bool series
            # dry_weather_bool = ~mark_event_bool(ww_period_table, index)
            wet_weather_bool = mark_event_bool(ww_period_table, index)
            dry_weather_bool = mark_event_bool(dw_period_table, index)

            dry_weather_bool &= ~wet_weather_bool

            # control
            # potential_error = ww_period_table.index.size * Timedelta(self.ts.index.freq)
            # ww_dur = (ww_period_table['end'] - ww_period_table['start']).sum() + potential_error
            # ww_dur2 = (dry_weather_bool.size - dry_weather_bool.sum()) * Timedelta(self.ts.index.freq)

            # setting boolean values to NaN will make the series to a type('0') and will convert the boolean to float!
            dry_weather_bool[self.ts.isna()] = fill_na
            self.dry_weather_bool = dry_weather_bool.rename(DW_BOOL)

        return self.dry_weather_bool

    # ------------------------------------------------------------------------------------------------------------------
    def set_dry_weather_bool(self, bool_series):
        self.dry_weather_bool = bool_series.rename(DW_BOOL)

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _smooth_criterion(criterion, smooth):
        # prediction over <smooth>/4 with last values
        criterion_smooth = criterion.rolling(smooth, center=True, min_periods=int(smooth / 4)).mean()
        # interpolate between the predictions
        criterion_level = criterion_smooth.interpolate(limit=smooth * 2, limit_direction='both', limit_area='inside')
        # fill the rest with 0

        if testing := False:
            import matplotlib.pyplot as plt

            selection = slice('2010-09-03', '2010-09-09')
            selection = slice(None)
            fig, ax = plt.subplots()
            criterion.loc[selection].plot(ax=ax)
            # fig.show()

            criterion_level[selection].plot(ax=ax)
            # ax.set_ylim(-100, 70)
            fig.show()

            c = AnalyseData(criterion, day_kind_detail=1)


            c.dw_mean_table().plot().get_figure().show()
            c.dw_variance_table().plot().get_figure().show()

            from wet_weather_analysis.figures import diurnal_density_full, weekly_density_plot
            fig, ax = diurnal_density_full(criterion)
            fig.show()

            fig, ax = weekly_density_plot(AnalyseData(criterion, day_kind_detail=8, est_best_shift_time=False))
            ax.set_ylim(-100, 100)
            fig.set_size_inches(12,8)
            fig.show()

            from statsmodels.tsa.api import ExponentialSmoothing

            # Identify missing values (NaNs)
            missing_mask = criterion.isnull()

            # Split your data into two parts: one with missing values and one without
            data_with_missing = criterion[missing_mask]
            data_without_missing = criterion[~missing_mask]

            # Define your ExponentialSmoothing model (you can choose the appropriate settings)
            model = ExponentialSmoothing(criterion.interpolate().dropna(),
                                         seasonal='add',
                                         seasonal_periods=12*24, freq='5min')

            # Fit the model to the data without missing values
            model_fit = model.fit()

            # Forecast the missing values
            forecasted_values = model_fit.forecast(steps=len(data_with_missing))

            # Fill in the missing values in the original DataFrame
            # df.loc[missing_mask, 'your_column_with_time_series'] = forecasted_values

        return criterion_level.fillna(0)

    @timeit
    def get_criterion_level_series(self, smooth_window=pd.Timedelta(days=2)):
        """
        Smoothed dry-weather-criterion only considering dw-periods.

        Wet-weather periods will have a criterion of 0.
        Smoothed criterion over <smooth_window> duration.
        Interpolation between <smooth_window * 2> long gaps.

        Args:
            smooth_window (pd.Timedelta): duration of smooth window.

        Returns:
            pd.Series: Criterion level
        """
        smooth = self.get_window_size(smooth_window)
        if self.criterion_level is None:
            criterion = self.get_criterion_series(smooth=1)
            dw_bool = self.get_dry_weather_bool().fillna(False)
            criterion[~dw_bool] = np.NAN

            self.criterion_level = self._smooth_criterion(criterion, smooth).rename(DW_LEVEL)
        return self.criterion_level.rolling(smooth, center=True, min_periods=1).mean()

    # ------------------------------------------------------------------------------------------------------------------
    # @timeit
    @smoother
    def get_dw_continuum_series(self) -> pd.Series:
        """
        Get a dry-weather-continuum time-series based on the criterion-level.

        CONTINUUM = MEAN + CRITERION * VARIANCE

        Returns:
            pd.Series: dry-weather continuum.
        """
        if self.dry_continuum is None:
            regular = self.get_dw_mean_series(smooth=1)
            # criterion = self.get_criterion(smooth=1)  # .round(1)
            level = self.get_criterion_level_series()  # .round(1)
            var = self.get_dw_variance_series(smooth=1)

            lower = level < 0
            higher = level > 0

            cont = regular.copy()
            cont[lower] += var[LOWER] * level[lower] / 100 * self.limit
            cont[higher] += var[UPPER] * level[higher] / 100 * self.limit

            # self.dry_continuum = (regular + (self.ts - regular) * level / criterion).rename(DW_CONTINUUM)
            self.dry_continuum = cont.rename(DW_CONTINUUM)
        return self.dry_continuum

    def get_dw_event_series(self, start, end):
        regular = self.get_dw_mean_series(smooth=1).loc[start:end]
        criterion = self.get_criterion_series(smooth=1)
        # level = self.get_criterion_level_series()  # .round(1)
        var = self.get_dw_variance_series(smooth=1).loc[start:end]

        criterion[start:end] = np.NAN

        level = self._smooth_criterion(criterion, smooth=self.get_window_size(pd.Timedelta(days=2))).loc[start:end]

        lower = level < 0
        higher = level > 0

        cont = regular.copy()
        cont[lower] += var[LOWER] * level[lower] / 100 * self.limit
        cont[higher] += var[UPPER] * level[higher] / 100 * self.limit

        return cont.rename(DW_CONTINUUM)

    # ------------------------------------------------------------------------------------------------------------------
    # @timeit
    @smoother
    def get_dry_filling_series(self, which: Literal[UPPER, LOWER, MEAN, AUTO]=MEAN):
        """
        Uses the measured time-series and fills wet-weather periods with estimated dry-weather values.

        BUT ignore NaNs

        Args:
            which (str):  'UPPER', 'LOWER'(-DW-Bound), (DW-)'MEAN', 'AUTO' (=DW-Continuum)
            smooth (int): smoothing window in minutes

        Returns:
            pd.Series: continuously dry weather values
        """
        fn = self.filename(f'dry-fill # arithmetic={self.arithmetic} # limit={self.limit} # crit_limit={self.ww_crit_limit}')

        # Is there an existing file with a DataFrame
        if self.cont is None:
            if isfile(fn):
                self.cont = self._read(fn)
            else:
                self.cont = pd.DataFrame(index=self.ts.index)

        if which not in self.cont:
            self.cont[which] = self.ts.values

            criterion = self.get_criterion_series(smooth=1)

            factor = {MEAN: 0,
                      UPPER: 100,
                      LOWER: -100,
                      AUTO: self.get_criterion_level_series()
                      }[which]

            out = criterion.sub(factor).abs() > self.ww_crit_limit

            fill_series = {MEAN: self.get_dw_mean_series(smooth=1),
                           UPPER: self.get_dw_range_series(smooth=1)[UPPER],
                           LOWER: self.get_dw_range_series(smooth=1)[LOWER],
                           AUTO: self.get_dw_continuum_series(smooth=1)
                           }[which]

            self.cont.loc[out, which] = fill_series[out]
            self._write(self.cont, fn)

        return self.cont[which].rename(DW_CONTINUUM)

    # ------------------------------------------------------------------------------------------------------------------
    @timeit
    def get_dw_avail(self, window=pd.Timedelta(days=2), crit_limit=100):
        if self.dry_weather_avail is None:
            dw_bool = self.get_dry_weather_bool(crit_limit=crit_limit)
            window_num = self.get_window_size(window) # int(round(window / guess_freq(dw_bool.index)))
            roll = dw_bool.fillna(False).rolling(window_num, center=True, min_periods=int(window_num / 4))
            dry_weather_avail = roll.sum() / roll.count() * 100
            self.dry_weather_avail = dry_weather_avail.rename(DW_AVAILABILITY)
        return self.dry_weather_avail

    def new_level(self):
        regular = self.get_dw_mean_series(smooth=1)
        cont = self.get_dw_continuum_series()
        return cont.div(regular)

    # ------------------------------------------------------------------------------------------------------------------
    def group_day_count(self):
        return self.get_day_grouper().apply(lambda x: round(len(x) / 60 / 24, 1))

    # ------------------------------------------------------------------------------------------------------------------
    @timeit
    def get_dw_uncertainty_table(self, min_rain_period=Timedelta(hours=2), extra_range=Timedelta(hours=4)):
        """
        Aggregate data für analysis groups and calculate the dry-weather uncertainty.

        Read from file or calculate and write to file.

        Args:
            min_rain_period (pandas.Timedelta): minimum period to count as a rain-event_analysis
            extra_range (pandas.Timedelta): extra time between a dry period and wet weather (no more influence)

        Returns:
            pd.DataFrame: index=day-times | columns=day-kinds
        """
        fn = self.filename(f'dw_uncertainty')
        if isfile(fn):
            return self._read(fn)
        else:
            dw_bool = self.get_dry_weather_bool(min_rain_period=min_rain_period, extra_range=extra_range).fillna(False)
            dw_cont = self.get_dw_continuum_series()
            ts_dw = self.ts[dw_bool]
            dw_cont_dw = dw_cont[dw_bool]
            diff = ts_dw - dw_cont_dw
            grouper = diff.groupby([self.day_category_index[dw_bool], self.ts.index.time[dw_bool]])
            dw_uncertainty = grouper.std().unstack(0)
            self._write(dw_uncertainty, fn)
            return dw_uncertainty

    @smoother
    def dw_uncertainty_table(self):
        """
        Aggregate data für analysis groups and calculate the dry-weather uncertainty.

        Returns:
            pd.DataFrame: index=day-times | columns=day-kinds
        """
        if self._dw_uncertainty is None:
            self._dw_uncertainty = self.get_dw_uncertainty_table()
        return self._dw_uncertainty

    # @timeit
    @smoother
    def get_dw_uncertainty_series(self) -> pd.Series:
        dw_uncertainty_table = self.dw_uncertainty_table(smooth=1)
        dw_uncertainty_series = pd.Series(index=self.ts.index,
                                 data=self._lookup(dw_uncertainty_table),
                                 name='DW-UNCERTAINTY')
        return dw_uncertainty_series

    ####################################################################################################################
    def _write(self, data, fn):
        if self.make_temp_files:
            if isinstance(data, (pd.DataFrame, pd.Series)):
                if isinstance(data, pd.Series):
                    d = data.to_frame()
                elif isinstance(data, pd.DataFrame):
                    d = data.copy()
                    if isinstance(d.columns, pd.MultiIndex):
                        d.columns = ['/'.join(str(c) for c in col).strip() for col in d.columns]
                else:
                    d = data.copy()

                d.to_parquet(f'{fn}.parq', compression='brotli')

            else:
                write_pickle(data, f'{fn}_{platform}.pkl')

            # global lev
            print(f'{lev}written: {fn}')

    ####################################################################################################################
    def _read(self, fn, dtype=None):
        if self.make_temp_files:
            # check('read: ' + fn)

            if path.isfile(f'{fn}.parq'):
                data = pd.read_parquet(f'{fn}.parq')
                if data.columns.size == 1 and ((dtype is not None) and (dtype is pd.Series)):
                    data = data.iloc[:, 0].copy()
                elif all(['/' in col for col in data.columns]):
                    data.columns = pd.MultiIndex.from_tuples([col.split('/') for col in data.columns])
                return data

            elif path.isfile(f'{fn}_{platform}.pkl'):
                try:
                    return read_pickle(f'{fn}_{platform}.pkl')
                except AttributeError as e:
                    raise IOError(f'Can\'t read "{fn}_{platform}.pkl": has an erroneous attribute!')

    # def dw_periods(self, allowed_anomalies=4, minimum_duration=pd.Timedelta(hours=5)):
    #     criterion = self.get_criterion(limit=self.limit)
    #     criterion = criterion.fillna(1000)
    #     dw = self.get_dry_weather_bool()
    #     freq = guess_freq(self.ts)
    #     dw_period_table = span_table(dw.index, span_bool=dw, min_span=freq)
    #     dw_period_table = combine_events(dw_period_table, new_event_after=freq * allowed_anomalies)
    #     dw_period_table = calc_event_duration(dw_period_table)
    #     dw_period_table = dw_period_table[dw_period_table[DELTA_min] > minimum_duration].copy()
    #     return dw_period_table

    # def save_state(self, fn=''):
    #     self.get_dry_continuum()
    #     self.get_dry_filling(which='auto')
    #     self.get_dry_filling(which='mean')
    #     self.get_dry_filling(which='upper')
    #     self.get_dry_filling(which='lower')
    #     self.get_day_group()
    #     self.get_dry_range()
    #     self.agg_dry_bound()
    # write_pickle(self, fn)


########################################################################################################################
class AnalyseLoadRate(AnalyseData):
    def __init__(self, flow, concentration, label_parameter, **kwargs):
        # flow in [L/s]
        # concentration in [mg/L]
        self.flow = flow
        self.concentration = concentration
        self.label_parameter = label_parameter
        lr = calculate_load_rate(concentration, flow)  # kg/[FREQ of flow]
        lr.index.name = ''

        AnalyseData.__init__(self, lr, **kwargs)
