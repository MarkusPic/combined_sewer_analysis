from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ._class import AnalyseData
from .figures import _single_timestamp_distribution


def compare_timestamp_distribution(data: AnalyseData, pth: Path):
    # wie ist die verteilung der daten
    # pro Zeitpunkt und Tageskategorie
    # TW und RW getrennt

    dw_bool = data.get_dw_bool_series(fill_na=False).astype(bool)
    # crit = data.get_criterion_series()

    groupby = data.ts.groupby(
        [data.get_diff_day_type(data._shifted_ts.index, level_of_detail=2), data.ts.index.time])

    bin_width = 5

    pdfs = {}
    from tqdm import tqdm
    progress = tqdm(groupby)

    # ---
    for (day_kind, timestamp), series in progress:
        if timestamp.minute != 0:
            continue
        progress.set_postfix_str(f'{day_kind} | {timestamp}')

        if day_kind not in pdfs:
            pdfs[day_kind] = PdfPages(pth / f'compare_timestamp_distribution - {day_kind}.pdf')

        dw_bool_ = dw_bool[series.index].copy()
        fig = _single_timestamp_distribution(day_kind, timestamp, series, dw_bool_, bin_width=bin_width)

        pdfs[day_kind].savefig(fig)
        plt.close()

    for pdf in pdfs.values():
        pdf.close()
