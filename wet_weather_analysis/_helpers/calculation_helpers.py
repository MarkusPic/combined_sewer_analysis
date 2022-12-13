from pandas import Series, DataFrame
from scipy.stats import iqr, gaussian_kde
from scipy.optimize import minimize_scalar
import numpy as np
from datetime import time


def mad(variation):
    """
    median absolute deviation
    :param variation: _np.array_ = values - mean_of_values
    :return:
    """
    return np.median(np.abs(variation))


def split_mad(variation):
    """
    calculate split median absolute deviation
    median deviation split for values above and below the mean value
    :param variation: _np.array_ = values - mean_of_values
    :return:
    """
    return np.abs(np.median(variation[variation > 0])), np.abs(np.median(variation[variation < 0]))


def est_bins(a):
    # maximum of the 'Sturges' and 'FD'(Freedman Diaconis) estimators
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram.html
    num = len(a)
    return np.max([2 * iqr(a) / num ** (1 / 3), np.log2(num) + 2])


def find_nearest(a, x):
    """
    get index of closest value in array
    :param a: _np.array_
    :param x: _num_
    :return: _int_ index in array
    """
    return np.abs(a - x).argmin()


def robust_variance(variation):
    """

    :param variation: _np.array_ = values - mean_of_values
    :return:
    """
    a = np.sort(variation)

    def f(x):
        return np.abs(np.mean(a[:int(x)]))

    start_index = find_nearest(a, 0)

    # np.percentile(a, 25) - 1.5 * (np.percentile(a, 50) - np.percentile(a, 25))

    # the representative bound must include the mean value
    res = minimize_scalar(f, bounds=(start_index, len(a) - 1), method='bounded').x
    return np.std(a[:int(res)])


def robust_mean(_array):
    data = _array
    clip = (np.percentile(data, 1, interpolation='nearest'),
            np.percentile(data, 99, interpolation='lower'))
    data = np.clip(data, *clip)
    kde = gaussian_kde(data)
    bw = kde.factor
    gridsize = max(100, _array.size)
    cut = 3
    support_min = max(data.min() - bw * cut, clip[0])
    support_max = min(data.max() + bw * cut, clip[1])
    support = np.linspace(support_min, support_max, gridsize)

    # np.percentile(data, 1)
    # np.percentile(data, 99, interpolation='lower')

    densities = kde.evaluate(support)
    return support[np.argmax(densities)]

    # import seaborn as sns
    # x = sns.distplot(_array, rug=True)
    # x = sns.distplot(array, rug=True, ax=x)
    #
    # f.plot(x=0, y=1, ax=x)
    #
    # f = pd.DataFrame([support, densities]).T
    # f1 = pd.DataFrame([new_array, densities]).T
    # f = pd.DataFrame([_array, densities]).T.sort_values(0)
    #
    # densities = kde.evaluate(_array)
    # return _array[np.argmax(densities)]
    #
    # densities = gaussian_kde(array).evaluate(array)
    # return _array[np.argmax(densities)]
    #
    # import statsmodels.nonparametric.api as smnp
    # bw = "scott"
    # kernel = "gau"
    # gridsize = 128
    # cut = -1
    # data = _array
    # kde = smnp.KDEUnivariate(data)
    # kde.fit(kernel, bw, True, gridsize=gridsize, cut=cut, clip=(data.min()+1, data.max()-1))
    # if cumulative:
    #     grid, y = kde.support, kde.cdf
    # else:
    #     grid, y = kde.support, kde.density


# @numba.jit
def _calc_dry_mean(_array, kind):
    array = _array[~np.isnan(_array)]

    if kind in [0, 6, 8, 97, 98]:
        return np.median(array)

    elif kind in [1, 2, 7]:
        return robust_mean(array)

    elif kind == 99:
        return np.mean(array)

    else:
        return np.NaN


def calc_dry_mean(s, kind):
    return _calc_dry_mean(s.values, kind)


# @numba.jit
def _calc_dry_variation(_variation, kind):
    variation = _variation[~np.isnan(_variation)]

    if kind in [0, 2, 6, 7, 8]:
        return mad(variation)

    elif kind == 1:
        return robust_variance(variation)

    elif kind == 99:
        return np.std(variation)

    elif kind == 97:
        # IQR
        return (np.percentile(variation, 75) - np.percentile(variation, 25)) / 2

    else:
        return np.NaN


def calc_dry_variation(s, kind):
    return _calc_dry_variation(s.values, kind)


def calc_dry_variation_split(_variation, kind, time_stamp=None, infer_mean=False):
    if isinstance(infer_mean, bool) and infer_mean:
        mean = calc_dry_mean(_variation, kind)
        variation = _variation - mean
    elif isinstance(infer_mean, Series) or isinstance(infer_mean, float):
        variation = _variation - infer_mean
    else:
        variation = _variation
        # assert NotImplementedError

    if not time_stamp:
        time_stamp = variation.index[0].time()

    if kind in [6, 7] and (time_stamp > time(hour=7)) & (time_stamp < time(hour=12)):
        return split_mad(variation)
    elif kind == 8:
        return split_mad(variation)

    elif kind == 98:  # IQR
        return variation.quantile(0.75), variation.quantile(0.25)

    else:
        if isinstance(variation, np.ndarray):
            var = _calc_dry_variation(variation, kind)
        else:
            var = calc_dry_variation(variation, kind)
        return var, var


def func_dry_variation(variation, kind, time_stamp=None, infer_mean=False):
    upper, lower = calc_dry_variation_split(variation, kind, time_stamp=time_stamp, infer_mean=infer_mean)
    return DataFrame({'UPPER': upper, 'LOWER': lower}, index=variation.index)


# def calc_criterion(s, limit, dry_mean, dry_upper_variance, dry_lower_variance, day_kind_detail):
#     try:
#         day = get_kind_of_day(s.index[0], level_of_detail=day_kind_detail)
#         index = s.index.time
#         # new = new.reindex(dry_mean.index)
#         diff = s.values - dry_mean[day][index]
#         criteria = dry_mean[day][index].copy()
#         upper = diff > 0
#         lower = diff < 0
#         criteria[upper] = diff[upper] / dry_upper_variance[day][index][upper]
#         criteria[lower] = diff[lower] / dry_lower_variance[day][index][lower]
#         criteria = criteria * 100 / limit
#     except:
#         print(s, limit, dry_mean, dry_upper_variance, dry_lower_variance, sep='\n\n' + '-'*100 + '\n\n')
#         exit()
#     return Series(index=s.index, data=criteria.values)


# @numba.jit
def _calc_criterion(_array, kind, time_stamp, limit=1):
    """

    :rtype: np.ndarray
    :type limit: float
    :type time_stamp: datetime.time
    :type kind: int
    :type _array: np.ndarray
    :param _array:

    :param kind:
    :param time_stamp:
    :param limit:
    :return:
    """
    dry_mean = _calc_dry_mean(_array, kind=kind)
    diff = _array - dry_mean
    upper_var, lower_var = calc_dry_variation_split(diff, kind=kind, time_stamp=time_stamp)
    lower = np.invert(_array.astype(bool))
    upper = lower.copy()
    notna = ~np.isnan(_array)
    lower[notna] = diff[notna] < 0
    upper[notna] = diff[notna] > 0
    crit = diff.copy()
    crit[lower] = crit[lower] / lower_var
    crit[upper] = crit[upper] / upper_var
    return crit * 100 / limit


def calc_criterion(s, kind, limit=1):
    # return _calc_criterion(s.fillna(0).values, kind, limit=limit, time_stamp=s.index[0].time())
    return _calc_criterion(s.values, kind, limit=limit, time_stamp=s.index[0].time())
