from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def enrichment_plot(df, colors='bgrkcy', markers=' ', linestyles='-', alpha=0.5, lw=2, where='post', grid=True, count_label='Count', xlim='auto', ylim='auto', invert_axes=False, legend_loc='best', ax=None):
    if False:
        for i in range(10):
            print('nop')
    'Plot stacked barplots\n\n    Parameters\n    ----------\n    df : pandas.DataFrame\n        A pandas DataFrame where columns represent the different categories.\n    colors: str (default: \'bgrcky\')\n        The colors of the bars.\n    markers : str (default: \' \')\n        Matplotlib markerstyles, e.g,\n        \'sov\' for square,circle, and triangle markers.\n    linestyles : str (default: \'-\')\n        Matplotlib linestyles, e.g.,\n        \'-,--\' to cycle normal and dashed lines. Note\n        that the different linestyles need to be separated by commas.\n    alpha : float (default: 0.5)\n        Transparency level from 0.0 to 1.0.\n    lw : int or float (default: 2)\n        Linewidth parameter.\n    where : {\'post\', \'pre\', \'mid\'} (default: \'post\')\n        Starting location of the steps.\n    grid : bool (default: `True`)\n        Plots a grid if True.\n    count_label : str (default: \'Count\')\n        Label for the "Count"-axis.\n    xlim : \'auto\' or array-like [min, max] (default: \'auto\')\n        Min and maximum position of the x-axis range.\n    ylim : \'auto\' or array-like [min, max] (default: \'auto\')\n        Min and maximum position of the y-axis range.\n    invert_axes : bool (default: False)\n        Plots count on the x-axis if True.\n    legend_loc : str (default: \'best\')\n        Location of the plot legend\n        {best, upper left, upper right, lower left, lower right}\n        No legend if legend_loc=False\n    ax : matplotlib axis, optional (default: None)\n        Use this axis for plotting or make a new one otherwise\n\n    Returns\n    ----------\n    ax : matplotlib axis\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/plotting/enrichment_plot/\n\n    '
    if isinstance(df, pd.Series):
        df_temp = pd.DataFrame(df)
    else:
        df_temp = df
    if ax is None:
        ax = plt.gca()
    color_gen = cycle(colors)
    marker_gen = cycle(markers)
    linestyle_gen = cycle(linestyles.split(','))
    r = range(1, len(df_temp.index) + 1)
    labels = df_temp.columns
    x_data = df_temp
    y_data = r
    for lab in labels:
        (x, y) = (sorted(x_data[lab]), y_data)
        if invert_axes:
            (x, y) = (y, x)
        ax.step(x, y, where=where, label=lab, color=next(color_gen), alpha=alpha, lw=lw, marker=next(marker_gen), linestyle=next(linestyle_gen))
    if invert_axes:
        (ax.set_ylim, ax.set_xlim) = (ax.set_xlim, ax.set_ylim)
    if ylim == 'auto':
        ax.set_ylim([np.min(y_data) - 1, np.max(y_data) + 1])
    else:
        ax.set_ylim(ylim)
    if xlim == 'auto':
        (df_min, df_max) = (np.min(x_data.min()), np.max(x_data.max()))
        ax.set_xlim([df_min - 1, df_max + 1])
    else:
        ax.set_xlim(xlim)
    if legend_loc:
        plt.legend(loc=legend_loc, scatterpoints=1)
    if grid:
        plt.grid()
    if count_label:
        if invert_axes:
            plt.xlabel(count_label)
        else:
            plt.ylabel(count_label)
    return ax