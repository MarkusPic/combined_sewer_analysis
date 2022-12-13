import pandas as pd

from mp.helpers import timeit, check
from wet_weather_analysis import AnalyseData


@timeit
def est_best_daily_grouping(data: AnalyseData):
    data.day_kind_detail = 9  # temporary
    est_best_shift_time(data)
    data.get_analysis_grouper()
    all_means = data.dw_mean_table()

    """
    from mp.libs.timeseries.plots.plotly_interface import PlotlyAxes
    axes = PlotlyAxes()
    axes.plot(all_means, row=1)
    fig = axes.get_figure()
    fig.save('/home/markus/Downloads/all_mean', auto_open=True)
    """

    # http://seaborn.pydata.org/generated/seaborn.clustermap.html#seaborn.clustermap

    df = all_means.corr()
    # dx = df.where(~np.triu(np.ones(df.shape)).astype(np.bool))

    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import pdist

    X = df.values
    d = pdist(X)  # vector of ('55' choose 2) pairwise distances
    L = sch.linkage(d, method='complete')
    # ind = sch.fcluster(L, 0.2 * d.max(), 'distance')

    ind = sch.fcluster(L, d.min() + (d.max() - d.min()) / 2, 'distance')

    group_no = pd.Series(data=ind, index=df.columns.tolist())
    groups = group_no.groupby(group_no).agg(lambda a: ','.join(a.index.tolist()))

    group_label = group_no.map(groups)
    data._day_category_index = pd.CategoricalIndex(data.day_category_index.map(group_label))

    check('Best grouping with:  {}'.format(' -- '.join(data.get_day_categories())))

    # ----------------------

    # all_means.plot()

    # import seaborn as sns
    # sns.heatmap(df, annot=True)
    #
    # d2 = all_means.reindex(df.columns[np.argsort(ind)], axis=1).corr()
    # g = sns.clustermap(d2)
    # g.get_figure().show()
    #
    # sns.clustermap(all_means,  metric="correlation", robust=True, row_cluster=False, method='complete')
    #
    #
    # # ----------------------
    #
    # from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
    #
    # x = s.copy().values
    # x.shape = (x.shape[0], 1)
    #
    # # n = len(x)
    # #
    # # def llf(id):
    # #     if id < n:
    # #         return '{:0.2f}'.format(x[id][0])
    # #     return ''
    #
    # Z = linkage(x, 'ward')  # ward single complete
    #
    # cluster = cut_tree(Z
    #                    # , n_clusters=2
    #                    , height=[1.0])
    # dendrogram(
    #     Z,
    #     leaf_rotation=90,  # rotates the x axis labels
    #     leaf_font_size=8,  # font size for the x axis labels
    #     # leaf_label_func=llf
    # )
    #
    # # ----------------------
    #
    # df[df == 1] = NaN
    #
    # dr = 100 - df * 100 / df.max()
    #
    #
    # df[(df!= 1) & (df >0.99)]
    #
    # df.idxmax()

    data._grouper_analysis = None
    data._agg_dw_mean = None
    data.day_kind_detail = 'best'
    # self._day_category_index = None
    est_best_shift_time(data)


@timeit
def est_best_shift_time(data: AnalyseData):
    # TODO: est_best_shift_time: beta testing

    # fn = path('{}_best_time_shift_{}'.format(self.name, self.day_kind_detail))

    # reset_day_kind = False
    # if self.day_kind_detail == 'best':
    #     reset_day_kind = True
    #     self.day_kind_detail = 9  # temporary

    fn = data.filename('best_time_shift')
    if isfile(fn):
        data.shift_delta = data._read(fn)
    else:

        data.get_analysis_grouper()
        all_means = data.dw_mean_table()
        diff = pd.Series(index=all_means.index, data=0)
        cols = all_means.columns.tolist()
        for col in cols:
            cols.remove(col)
            if cols:
                diff = diff.add(all_means[cols].sub(all_means[col], axis=0).abs().sum(axis=1), axis=0)

        rank = diff.rank()
        ranks = pd.concat([rank, rank, rank], axis=0)
        rm = ranks.rolling(30, center=True).mean().dropna()
        day_time = rm.idxmin()

        """
        ri = 100 - rm * 100 / rm.max()
        ri.plot(secondary_y=True, ls=':', label='Rank', legend=True)
        day_time = ri.idxmax()
        """  # only for displaying purpose

        """
        dm = all_means.max(axis=1) - all_means.min(axis=1)
        di = 100 - dm * 100 / dm.max()
        di = di.rolling(30, center=True).mean().dropna().plot(secondary_y=True, ls=':', label='Range', legend=True)
        di.idxmax()
        """  # may be easier calculated

        check('Ideal shift time at {}'.format(day_time))

        """
        ax = all_means.plot()
        rm.plot(secondary_y=True, ls=':', label='30', legend=True)
        ranks.rolling(60, center=True).mean().dropna().plot(secondary_y=True, label='60', legend=True, ls=':')
        ranks.rolling(30, center=True).mean().dropna().plot(secondary_y=True, label='30', legend=True, ls=':')
        ranks.rolling(10, center=True).mean().dropna().plot(secondary_y=True, label='10', legend=True, ls=':')
        ranks.dropna().plot(secondary_y=True, label='1', legend=True)
        import matplotlib.pyplot as plt
        plt.close()

        all_means.index = pd.DatetimeIndex([datetime.datetime.combine(pd.to_datetime('2018-01-01').date(),i) for i in all_means.index.values])
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.HourLocator(interval = 1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 15))
        ax.set_xlim(all_means.index[0], all_means.index[-1])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H h'))
        ax.get_figure().autofmt_xdate()
        plt.savefig('test.pdf')
        """  # testing

        data.shift_delta = pd.Timedelta(hours=day_time.hour, minutes=day_time.minute)

        data._write(data.shift_delta, fn)

        data._grouper_analysis = None
        data._agg_dw_mean = None
        data.shift_times()
        data._day_category_index = None
    # if reset_day_kind:
    #     self.day_kind_detail = 'best'
    #     self._day_category_index = None
