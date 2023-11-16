from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np

def stacked_barplot(df, bar_width='auto', colors='bgrcky', labels='index', rotation=90, legend_loc='best'):
    if False:
        return 10
    "\n    Function to plot stacked barplots\n\n    Parameters\n    ----------\n    df : pandas.DataFrame\n        A pandas DataFrame where the index denotes the\n        x-axis labels, and the columns contain the different\n        measurements for each row.\n    bar_width: 'auto' or float (default: 'auto')\n        Parameter to set the widths of the bars. if\n        'auto', the width is automatically determined by\n        the number of columns in the dataset.\n    colors: str (default: 'bgrcky')\n        The colors of the bars.\n    labels: 'index' or iterable (default: 'index')\n        If 'index', the DataFrame index will be used as\n        x-tick labels.\n    rotation: int (default: 90)\n        Parameter to rotate the x-axis labels.\n    legend_loc : str (default: 'best')\n        Location of the plot legend\n        {best, upper left, upper right, lower left, lower right}\n        No legend if legend_loc=False\n\n    Returns\n    ---------\n    fig : matplotlib.pyplot figure object\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/plotting/stacked_barplot/\n\n    "
    pos = np.array(range(len(df.index)))
    if bar_width == 'auto':
        width = 1 / (len(df.columns[1:]) * 2)
    else:
        width = bar_width
    if labels == 'index':
        labels = df.index
    color_gen = cycle(colors)
    label_pos = [pos]
    (fig, ax) = plt.subplots(figsize=(12, 6))
    plt.bar(pos, df.iloc[:, 0], width, alpha=0.8, color=next(color_gen), label=df.columns[0])
    for (i, c) in enumerate(df.columns[1:]):
        bar_pos = [p + width * (i + 1) for p in pos]
        label_pos.append(bar_pos)
        plt.bar(bar_pos, df.iloc[:, i + 1], width, alpha=0.5, color=next(color_gen), label=c)
    label_pos = np.asarray(label_pos).mean(axis=0) + width * 0.5
    ax.set_xticks(label_pos)
    ax.set_xticklabels(labels, rotation=rotation, horizontalalignment='center')
    plt.xlim(min(pos) - width, max(pos) + width * 7)
    if legend_loc:
        plt.legend(loc=legend_loc, scatterpoints=1)
    return fig