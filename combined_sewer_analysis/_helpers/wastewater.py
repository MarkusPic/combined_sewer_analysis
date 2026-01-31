import numpy as np
import pandas as pd
from pandas._libs.tslibs import to_offset

from .freqs import guess_freq


def match_freq(one, two):
    """
    fit index of both pandas to smaller frequency of both
    :param one: _pd.Series_ or _pd.DataFrame_
    :param two: _pd.Series_ or _pd.DataFrame_
    :return: (_pd.Series_ or _pd.DataFrame_) same as input
    """
    freq_one = guess_freq(one.index)
    freq_two = guess_freq(two.index)
    delta_one = pd.Timedelta(freq_one)
    delta_two = pd.Timedelta(freq_two)

    if freq_one != freq_two:
        new_freq = min(freq_one, freq_two)
        # only fill gaps of new index freq
        fill_limit = int(max(delta_one, delta_two) // min(delta_one, delta_two) + 1)
        fill_limit = to_offset(fill_limit)
        if freq_one != new_freq:
            return one.asfreq(new_freq).interpolate(method='time', limit=fill_limit), two
        elif freq_two != new_freq:
            return one, two.asfreq(new_freq).interpolate(method='time', limit=fill_limit)
    else:
        return one.copy(), two.copy()


def calculate_load(_concentration, _flow, freq, custom_Q_time='00:00', custom_substance_time='00:00',
                   volume_proportions=None):
    """
    calculate freights for a given frequency
    opt: custom flow split time or custom substance split time, where a new day/week/month begins
    :param concentration: _pandas.DataFrame_ or _pandas.Series_ of the quality measurements (in mg/L)
    :param flow: _pandas.Series_ of the flow measurement in (L/s)
    :param freq: _str_ frequency of the results
    :param custom_Q_time: _str_ time in HH:MM
    :param custom_substance_time: time in HH:MM
    :param volume_proportions: _int_ volume steps in cubic meters [m³]
    :return: _pandas.DataFrame_ freights
    """
    # C .. Concentration for quality measurements (mg/L)
    # LR ... Load/Freight Rate for quality measurement (mg/s)
    # Q ... flow (L/s)
    # L ... Load / Freight (kg/d)
    # L ... daily sum (xx/d) LQ flow sum

    flow, concentration = match_freq(_flow, _concentration)

    if custom_Q_time != '00:00' or custom_substance_time != '00:00' or volume_proportions:
        if volume_proportions:
            concentration = volume_proportional_sample(flow, concentration, volume_proportions=volume_proportions)

        flow = convert_flow(flow, per_step=True)  # [L/s] to [L/min]
        flow_sum = aggregate_data(flow, freq, 'sum', custom_switch_time=custom_Q_time)  # [L/min] to [L/freq]
        load = pd.DataFrame()

        for name in concentration.columns:
            concentration_mean = aggregate_data(concentration[name], freq, 'mean',
                                                custom_switch_time=custom_substance_time)  # [mg/L]

            if len(flow_sum) != len(concentration_mean):
                join_index = flow_sum.index.join(concentration_mean.index, how='inner', sort=True)
                load['L_' + name] = np.multiply(flow_sum[join_index], concentration_mean[join_index])
            else:
                load['L_' + name] = np.multiply(flow, concentration_mean)  # [mg/freq]

            # aggregated load from (mg) to (kg)
            load = load.multiply(1 / 1000000)

    else:
        load_rate = calculate_load_rate(concentration, flow)  # kg/min
        load = aggregate_data(load_rate, freq, 'sum')  # kg/freq

        def rename_col(x):
            if x.startswith('LR_') and x.endswith('_sum'):
                x = x.replace('LR_', 'L_').replace('_sum', '')
            return x

        load.rename(columns=rename_col, inplace=True)

        # load.columns = [rename_col(c) for c in load.columns]

    return load


def convert_load_to_ew(load):
    """
    convert freight to ew (=population equivalent)
    only BOD5 and COD implemented
    :param load: _pandas.DataFrame_ freight in (kg/FREQ)
    :return: _pandas.DataFrame_ ew-factor
    """
    # kg/d
    ew_factor = {'BOD5': 0.06,
                 'COD': 0.12}
    ew = pd.DataFrame()
    freq = load.index.freq
    freq_factor = freq / pd.Timedelta(days=1)

    for f in load.columns:
        substance = f.replace('L_', '')
        substance = substance.split('_')[0]
        if substance in ew_factor.keys():
            ew[f.replace('L_', 'EW_')] = load[f] / ew_factor[substance] * freq_factor
    return ew


def convert_flow(_flow, per_step=True, cubic_meter=True):
    """
    convert from (L/s) to (m³/"Time-Step")

    :param _flow:
    :type _flow: pd.Series

    :param per_step: if to convert from [x/s] to [x/'freq'] i.e. [x/min], ...
    :type per_step: bool

    :param cubic_meter: if to convert from [L] to [m³]
    :type cubic_meter: bool

    :return: of converted unit
    :rtype: pd.Series
    """
    flow = _flow.copy()
    if per_step:
        flow *= guess_freq(flow.index) / pd.Timedelta(seconds=1)
    if cubic_meter:
        flow = flow / 1000
    return flow


def volume_proportional_sample(_flow, _concentration, volume_proportions=500):
    """
    get a volume proportional sample out of a quality measurement
    the flow and the every-parameter must have the same unit
    :param _flow: _pandas.Series_ of the flow in [L/s]
    :param _concentration: _pandas.Series_ or _pandas.DataFrame_ of quality measurements
    :param volume_proportions: _int_ volume steps in cubic meters [m³]
    :return: _same as qual_ sample of the quality measurement
    """
    flow, concentration = match_freq(_flow, _concentration)
    flow = convert_flow(flow, per_step=True, cubic_meter=True)
    prop = flow.to_frame().join(concentration, how='left')
    prop['flow_cum'] = prop[flow.name].cumsum()
    prop['flow_round'] = (prop['flow_cum'] / volume_proportions).floor()  # .astype(int)
    prop['sample'] = ~prop.duplicated('flow_round', keep='first')
    # print(prop)
    if isinstance(concentration, pd.Series):
        names = concentration.name
    elif isinstance(concentration, pd.DataFrame):
        names = concentration.columns
    else:
        names = None
    return prop.loc[prop['sample'], names]


def volume_proportional_aggragation(flow, concentration, freq, agg, custom_switch_time='00:00',
                                    unavailability_marker=None,
                                    volume_proportions=500):
    """

    :param flow: _pandas.Series_ of the flow in [L/s]
    :param concentration: _pandas.Series_ or _pandas.DataFrame_ of quality measurements
    :param freq:
    :param agg:
    :param custom_switch_time:
    :param unavailability_marker:
    :param volume_proportions: _int_ volume steps in cubic meters [m³]
    :return:
    """
    vol_prop_concentration = volume_proportional_sample(flow, concentration, volume_proportions)
    return aggregate_data(vol_prop_concentration, freq, agg, custom_switch_time, unavailability_marker)


def volume_proportional_exact_mean(flow, concentration):
    return (flow * concentration).sum() / flow.sum()


def calculate_load_rate(concentration, flow):
    """
    calculate load rate
    
    Args:
        concentration (pd.Series): quality measurements in [mg/L]
        flow (pd.Series): flow measurement (in L/s)

    Returns:
        pd.Series: load rate in [kg/min]
    """
    c, q = match_freq(concentration, flow)

    # L/s to m³/min
    q = convert_flow(q, per_step=True, cubic_meter=True)

    # mg/L * m³/min = g/m³ * m³/min = g/min
    load_rate = c.multiply(q, axis=0)

    # g/min to kg/min
    load_rate = load_rate.divide(1000, axis=0)

    rename_string = 'LR_{}'

    if isinstance(c, pd.DataFrame):
        load_rate.columns = [rename_string.format(c) for c in load_rate.columns]

    elif isinstance(c, pd.Series):
        load_rate.name = rename_string.format(c.name)

    return load_rate