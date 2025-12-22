import pandas as pd
from ._class import L


def dw_loads_table(data, is_flow=False, aggs=None, name='TW-Tages-{}'):
    mean = data.dw_mean_table()
    agg_dry_bound = data.get_dw_bound_table()
    upper = agg_dry_bound[L.UPPER]
    lower = agg_dry_bound[L.LOWER]

    if isinstance(mean, pd.DataFrame) and mean.columns.size == 1:
        mean = mean.iloc[:, 0].copy()
        lower = lower.iloc[:, 0].copy()
        upper = upper.iloc[:, 0].copy()
        one_day = True
    else:
        one_day = False

    res = None

    if aggs is None:
        aggs = ['sum', 'mean']
        aggs = {'sum': 'Summe',
                'mean': 'Mittel'}

    for agg in aggs:

        if agg == 'sum':
            if is_flow:
                mean_ = mean.copy() / 1000 * 60
                upper_ = upper.copy() / 1000 * 60
                lower_ = lower.copy() / 1000 * 60
            else:
                mean_ = mean.copy() / 1000
                upper_ = upper.copy() / 1000
                lower_ = lower.copy() / 1000
        else:
            mean_ = mean
            upper_ = upper
            lower_ = lower

        if one_day:
            table = pd.Series({'Min': lower_.apply(agg),
                               'Mittel': mean_.apply(agg),
                               'Max': upper_.apply(agg)}, name=name.format(agg))
            table = table[['Min', 'Mittel', 'Max']].copy()
        else:
            table = pd.concat([
                lower_.apply(agg).rename('Min'),
                mean_.apply(agg).rename('Mittel'),
                upper_.apply(agg).rename('Max')
                               ], axis=1).T
            # table = table.applymap(round_sig)
            table['Art'] = name.format(agg)

            table = table.set_index('Art', append=True).swaplevel(0, 1)
            table.index = table.index.rename(None, level=0)

            # use short weekdayname
            new_cols = []
            for col in table.columns:
                names = col.split(',')
                new_cols.append(','.join([''.join([s[:2] for s in name.split(' ')]) for name in names]))
            table.columns = new_cols

        if res is None:
            res = table
        else:
            res = pd.concat([res, table])

    return res
