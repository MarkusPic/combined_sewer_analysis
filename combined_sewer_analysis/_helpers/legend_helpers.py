import matplotlib as mpl
from matplotlib.artist import Artist
from matplotlib.container import Container
from matplotlib.lines import Line2D


def add_custom_legend(ax, lines_dict, **kwargs):
    """
    lines_dict:
    { label in legend: {line_dict ie: color, marker, linewidth, linestyle, ...) }
    kwargs: for legend
    """
    lines = []
    labels = []
    for label, line in lines_dict.items():
        labels.append(label)
        if isinstance(line, (Artist, Container)):
            lines.append(line)
        elif isinstance(line, tuple) and (len(line) == 1) and isinstance(line[0], (Artist, Container)):
            lines.append(line[0])
        else:
            lines.append(Line2D([0], [0], **line))
    return ax.legend(lines, labels, **kwargs)


def get_legend_dict(ax):
    handles, labels = ax.get_legend_handles_labels()
    return dict(zip(labels, handles))


def get_legend_dict_from_legend(legend: mpl.legend.Legend):
    return dict(zip([t._text for t in legend.texts], legend.legend_handles))


def get_legend_dict_displayed(ax):
    if ax.legend_ is None:
        return get_legend_dict(ax)
    return get_legend_dict_from_legend(ax.legend_)


def combine_legends(*axes, **kwargs):
    handles, labels = [], []
    for axi in axes:
        for handle, label in zip(*axi.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)
    axes[0].legend(handles, labels, **kwargs
                   # bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.
                   )


def append_custom_legend(ax, lines_dict, **kwargs):
    lines_dict_ = get_legend_dict(ax)
    lines_dict_.update(lines_dict)
    return add_custom_legend(ax, lines_dict_, **kwargs)
