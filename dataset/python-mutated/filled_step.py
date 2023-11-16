"""
=========================
Hatch-filled histograms
=========================

Hatching capabilities for plotting histograms.
"""
from functools import partial
import itertools
from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

def filled_hist(ax, edges, values, bottoms=None, orientation='v', **kwargs):
    if False:
        return 10
    "\n    Draw a histogram as a stepped patch.\n\n    Parameters\n    ----------\n    ax : Axes\n        The axes to plot to\n\n    edges : array\n        A length n+1 array giving the left edges of each bin and the\n        right edge of the last bin.\n\n    values : array\n        A length n array of bin counts or values\n\n    bottoms : float or array, optional\n        A length n array of the bottom of the bars.  If None, zero is used.\n\n    orientation : {'v', 'h'}\n       Orientation of the histogram.  'v' (default) has\n       the bars increasing in the positive y-direction.\n\n    **kwargs\n        Extra keyword arguments are passed through to `.fill_between`.\n\n    Returns\n    -------\n    ret : PolyCollection\n        Artist added to the Axes\n    "
    print(orientation)
    if orientation not in 'hv':
        raise ValueError(f"orientation must be in {{'h', 'v'}} not {orientation}")
    kwargs.setdefault('step', 'post')
    kwargs.setdefault('alpha', 0.7)
    edges = np.asarray(edges)
    values = np.asarray(values)
    if len(edges) - 1 != len(values):
        raise ValueError(f'Must provide one more bin edge than value not: len(edges)={len(edges)!r} len(values)={len(values)!r}')
    if bottoms is None:
        bottoms = 0
    bottoms = np.broadcast_to(bottoms, values.shape)
    values = np.append(values, values[-1])
    bottoms = np.append(bottoms, bottoms[-1])
    if orientation == 'h':
        return ax.fill_betweenx(edges, values, bottoms, **kwargs)
    elif orientation == 'v':
        return ax.fill_between(edges, values, bottoms, **kwargs)
    else:
        raise AssertionError('you should never be here')

def stack_hist(ax, stacked_data, sty_cycle, bottoms=None, hist_func=None, labels=None, plot_func=None, plot_kwargs=None):
    if False:
        while True:
            i = 10
    "\n    Parameters\n    ----------\n    ax : axes.Axes\n        The axes to add artists too\n\n    stacked_data : array or Mapping\n        A (M, N) shaped array.  The first dimension will be iterated over to\n        compute histograms row-wise\n\n    sty_cycle : Cycler or operable of dict\n        Style to apply to each set\n\n    bottoms : array, default: 0\n        The initial positions of the bottoms.\n\n    hist_func : callable, optional\n        Must have signature `bin_vals, bin_edges = f(data)`.\n        `bin_edges` expected to be one longer than `bin_vals`\n\n    labels : list of str, optional\n        The label for each set.\n\n        If not given and stacked data is an array defaults to 'default set {n}'\n\n        If *stacked_data* is a mapping, and *labels* is None, default to the\n        keys.\n\n        If *stacked_data* is a mapping and *labels* is given then only the\n        columns listed will be plotted.\n\n    plot_func : callable, optional\n        Function to call to draw the histogram must have signature:\n\n          ret = plot_func(ax, edges, top, bottoms=bottoms,\n                          label=label, **kwargs)\n\n    plot_kwargs : dict, optional\n        Any extra keyword arguments to pass through to the plotting function.\n        This will be the same for all calls to the plotting function and will\n        override the values in *sty_cycle*.\n\n    Returns\n    -------\n    arts : dict\n        Dictionary of artists keyed on their labels\n    "
    if hist_func is None:
        hist_func = np.histogram
    if plot_func is None:
        plot_func = filled_hist
    if plot_kwargs is None:
        plot_kwargs = {}
    print(plot_kwargs)
    try:
        l_keys = stacked_data.keys()
        label_data = True
        if labels is None:
            labels = l_keys
    except AttributeError:
        label_data = False
        if labels is None:
            labels = itertools.repeat(None)
    if label_data:
        loop_iter = enumerate(((stacked_data[lab], lab, s) for (lab, s) in zip(labels, sty_cycle)))
    else:
        loop_iter = enumerate(zip(stacked_data, labels, sty_cycle))
    arts = {}
    for (j, (data, label, sty)) in loop_iter:
        if label is None:
            label = f'dflt set {j}'
        label = sty.pop('label', label)
        (vals, edges) = hist_func(data)
        if bottoms is None:
            bottoms = np.zeros_like(vals)
        top = bottoms + vals
        print(sty)
        sty.update(plot_kwargs)
        print(sty)
        ret = plot_func(ax, edges, top, bottoms=bottoms, label=label, **sty)
        bottoms = top
        arts[label] = ret
    ax.legend(fontsize=10)
    return arts
edges = np.linspace(-3, 3, 20, endpoint=True)
hist_func = partial(np.histogram, bins=edges)
color_cycle = cycler(facecolor=plt.rcParams['axes.prop_cycle'][:4])
label_cycle = cycler(label=[f'set {n}' for n in range(4)])
hatch_cycle = cycler(hatch=['/', '*', '+', '|'])
np.random.seed(19680801)
stack_data = np.random.randn(4, 12250)
dict_data = dict(zip((c['label'] for c in label_cycle), stack_data))
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
arts = stack_hist(ax1, stack_data, color_cycle + label_cycle + hatch_cycle, hist_func=hist_func)
arts = stack_hist(ax2, stack_data, color_cycle, hist_func=hist_func, plot_kwargs=dict(edgecolor='w', orientation='h'))
ax1.set_ylabel('counts')
ax1.set_xlabel('x')
ax2.set_xlabel('counts')
ax2.set_ylabel('x')
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True, sharey=True)
arts = stack_hist(ax1, dict_data, color_cycle + hatch_cycle, hist_func=hist_func)
arts = stack_hist(ax2, dict_data, color_cycle + hatch_cycle, hist_func=hist_func, labels=['set 0', 'set 3'])
ax1.xaxis.set_major_locator(mticker.MaxNLocator(5))
ax1.set_xlabel('counts')
ax1.set_ylabel('x')
ax2.set_ylabel('x')
plt.show()