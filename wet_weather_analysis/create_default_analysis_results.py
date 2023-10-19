from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import EngFormatter

from wet_weather_analysis import AnalyseData
from wet_weather_analysis.figures import (diurnal_density_full, compare_all_days, weekly_density_plot, diurnal_density,
                                          diurnal_uncertainty_density, compare_dw_uncertainty_day_relative,
                                          compare_dw_uncertainty_day_absolute)


def create_default_analysis_results(data_class: AnalyseData, pth: Path, ylim: tuple[float | int], unit='L/s'):
    pth.mkdir(exist_ok=True)

    override = False

    # ===
    # plot alle tagesgang kurven übereinandergelegt
    fn = pth / 'diurnal_density_full.png'
    if override or not fn.is_file():
        fig, ax = diurnal_density_full(data_class.ts, major_freq='H', minor_freq='15T', alpha=0.01)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(EngFormatter(unit=unit))  # , places=0
        fig.savefig(fn)
        plt.close()

    # ===
    # plot tagesgang mittelwert für alle tageskategorien
    fn = pth / 'compare_all_days.png'
    if override or not fn.is_file():
        fig, ax = compare_all_days(data_class, major_freq='H', minor_freq='15T')
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(EngFormatter(unit=unit))  # , places=0
        fig.savefig(fn)
        plt.close()

    # ===
    # pro gewählter tageskategorie einen diurnal density plot
    fn = pth / 'diurnal_density.pdf'
    if override or not fn.is_file():
        with PdfPages(fn) as pdf:
            for daily_group, daily_series in data_class.get_day_grouper():
                fig, ax = diurnal_density(data_class, daily_series.rename(daily_group))
                ax.set_ylim(*ylim)
                ax.yaxis.set_major_formatter(EngFormatter(unit=unit))  # , places=0
                pdf.savefig(fig)
                plt.close(fig)

    # ===
    # plot tagesgang mittelwert für alle tageskategorien
    fn = pth / 'weekly_density_plot.png'
    if override or not fn.is_file():
        data8 = AnalyseData(data_class.ts, limit=data_class.limit, kind=data_class.arithmetic, day_kind_detail=8,
                            file_path=data_class.temp_file_path,
                            make_temp_files=False, est_best_shift_time=False,
                            smooth_window=data_class.smooth_window).set_number_day_labels()

        fig, ax = weekly_density_plot(data8)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(EngFormatter(unit=unit))  # , places=0
        fig.set_size_inches(h=7.5, w=15)
        fig.savefig(fn)
        plt.close()

    # ===
    # diurnal uncertainty plot
    fn = pth / 'diurnal_uncertainty_density.pdf'
    if override or not fn.is_file():
        with PdfPages(fn) as pdf:
            for daily_group, daily_series in data_class.get_day_grouper():
                fig, ax = diurnal_uncertainty_density(data_class, daily_series.rename(daily_group), ylim=40)
                ax.yaxis.set_major_formatter(EngFormatter(unit=unit))  # , places=0
                pdf.savefig(fig)
                plt.close(fig)

    # ===
    # plot absolute TW-Unsicherheit für alle tageskategorien
    fn = pth / 'compare_dw_uncertainty_day_absolute.png'
    if override or not fn.is_file():
        fig, ax = compare_dw_uncertainty_day_absolute(data_class, major_freq='H', minor_freq='15T')
        ax.yaxis.set_major_formatter(EngFormatter(unit=unit))  # , places=0
        fig.savefig(fn)
        plt.close()

    # ===
    # plot relative TW-Unsicherheit für alle tageskategorien
    fn = pth / 'compare_dw_uncertainty_day_relative.png'
    if override or not fn.is_file():
        fig, ax = compare_dw_uncertainty_day_relative(data_class, major_freq='H', minor_freq='15T')
        fig.savefig(fn)
        plt.close()
