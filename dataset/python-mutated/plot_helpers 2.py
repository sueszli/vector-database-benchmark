import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap
cm_cycle = ListedColormap(['#0000aa', '#ff5050', '#50ff50', '#9040a0', '#fff000'])
cm3 = ListedColormap(['#0000aa', '#ff2020', '#50ff50'])
cm2 = ListedColormap(['#0000aa', '#ff2020'])
cdict = {'red': [(0.0, 0.0, cm2(0)[0]), (1.0, cm2(1)[0], 1.0)], 'green': [(0.0, 0.0, cm2(0)[1]), (1.0, cm2(1)[1], 1.0)], 'blue': [(0.0, 0.0, cm2(0)[2]), (1.0, cm2(1)[2], 1.0)]}
ReBl = LinearSegmentedColormap('ReBl', cdict)

def discrete_scatter(x1, x2, y=None, markers=None, s=10, ax=None, labels=None, padding=0.2, alpha=1, c=None, markeredgewidth=None):
    if False:
        for i in range(10):
            print('nop')
    "Adaption of matplotlib.pyplot.scatter to plot classes or clusters.\n    Parameters\n    ----------\n    x1 : nd-array\n        input data, first axis\n    x2 : nd-array\n        input data, second axis\n    y : nd-array\n        input data, discrete labels\n    cmap : colormap\n        Colormap to use.\n    markers : list of string\n        List of markers to use, or None (which defaults to 'o').\n    s : int or float\n        Size of the marker\n    padding : float\n        Fraction of the dataset range to use for padding the axes.\n    alpha : float\n        Alpha value for all points.\n    "
    if ax is None:
        ax = plt.gca()
    if y is None:
        y = np.zeros(len(x1))
    unique_y = np.unique(y)
    if markers is None:
        markers = ['o', '^', 'v', 'D', 's', '*', 'p', 'h', 'H', '8', '<', '>'] * 10
    if len(markers) == 1:
        markers = markers * len(unique_y)
    if labels is None:
        labels = unique_y
    lines = []
    current_cycler = mpl.rcParams['axes.prop_cycle']
    for (i, (yy, cycle)) in enumerate(zip(unique_y, current_cycler())):
        mask = y == yy
        if c is None:
            color = cycle['color']
        elif len(c) > 1:
            color = c[i]
        else:
            color = c
        if np.mean(colorConverter.to_rgb(color)) < 0.4:
            markeredgecolor = 'grey'
        else:
            markeredgecolor = 'black'
        lines.append(ax.plot(x1[mask], x2[mask], markers[i], markersize=s, label=labels[i], alpha=alpha, c=color, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor)[0])
    if padding != 0:
        pad1 = x1.std() * padding
        pad2 = x2.std() * padding
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(min(x1.min() - pad1, xlim[0]), max(x1.max() + pad1, xlim[1]))
        ax.set_ylim(min(x2.min() - pad2, ylim[0]), max(x2.max() + pad2, ylim[1]))
    return lines