import multiprocessing as mp
from itertools import cycle
from math import ceil, floor
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.utils import check_Xy, format_kwarg_dictionaries

def get_feature_range_mask(X, filler_feature_values=None, filler_feature_ranges=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Function that constucts a boolean array to get rid of samples\n    in X that are outside the feature range specified by filler_feature_values\n    and filler_feature_ranges\n    '
    if not isinstance(X, np.ndarray) or not len(X.shape) == 2:
        raise ValueError('X must be a 2D array')
    elif filler_feature_values is None:
        raise ValueError('filler_feature_values must not be None')
    elif filler_feature_ranges is None:
        raise ValueError('filler_feature_ranges must not be None')
    mask = np.ones(X.shape[0], dtype=bool)
    for feature_idx in filler_feature_ranges:
        feature_value = filler_feature_values[feature_idx]
        feature_width = filler_feature_ranges[feature_idx]
        upp_limit = feature_value + feature_width
        low_limit = feature_value - feature_width
        feature_mask = (X[:, feature_idx] > low_limit) & (X[:, feature_idx] < upp_limit)
        mask = mask & feature_mask
    return mask

def parallel(X_predict, clf, xtype):
    if False:
        print('Hello World!')
    Z = clf.predict(X_predict.astype(xtype))
    return Z

def plot_decision_regions(X, y, clf, feature_index=None, filler_feature_values=None, filler_feature_ranges=None, ax=None, X_highlight=None, zoom_factor=1.0, legend=1, hide_spines=True, markers='s^oxv<>', colors='#1f77b4,#ff7f0e,#3ca02c,#d62728,#9467bd,#8c564b,#e377c2,#7f7f7f,#bcbd22,#17becf', scatter_kwargs=None, contourf_kwargs=None, contour_kwargs=None, scatter_highlight_kwargs=None, n_jobs=None):
    if False:
        return 10
    "Plot decision regions of a classifier.\n\n    Please note that this functions assumes that class labels are\n    labeled consecutively, e.g,. 0, 1, 2, 3, 4, and 5. If you have class\n    labels with integer labels > 4, you may want to provide additional colors\n    and/or markers as `colors` and `markers` arguments.\n    See https://matplotlib.org/examples/color/named_colors.html for more\n    information.\n\n    Parameters\n    ----------\n    X : array-like, shape = [n_samples, n_features]\n        Feature Matrix.\n\n    y : array-like, shape = [n_samples]\n        True class labels.\n\n    clf : Classifier object.\n        Must have a .predict method.\n\n    feature_index : array-like (default: (0,) for 1D, (0, 1) otherwise)\n        Feature indices to use for plotting. The first index in\n        `feature_index` will be on the x-axis, the second index will be\n        on the y-axis.\n\n    filler_feature_values : dict (default: None)\n        Only needed for number features > 2. Dictionary of feature\n        index-value pairs for the features not being plotted.\n\n    filler_feature_ranges : dict (default: None)\n        Only needed for number features > 2. Dictionary of feature\n        index-value pairs for the features not being plotted. Will use the\n        ranges provided to select training samples for plotting.\n\n    ax : matplotlib.axes.Axes (default: None)\n        An existing matplotlib Axes. Creates\n        one if ax=None.\n\n    X_highlight : array-like, shape = [n_samples, n_features] (default: None)\n        An array with data points that are used to highlight samples in `X`.\n\n    zoom_factor : float (default: 1.0)\n        Controls the scale of the x- and y-axis of the decision plot.\n\n    hide_spines : bool (default: True)\n        Hide axis spines if True.\n\n    legend : int (default: 1)\n        Integer to specify the legend location.\n        No legend if legend is 0.\n\n    markers : str (default: 's^oxv<>')\n        Scatterplot markers.\n\n    colors : str (default: 'red,blue,limegreen,gray,cyan')\n        Comma separated list of colors.\n\n    scatter_kwargs : dict (default: None)\n        Keyword arguments for underlying matplotlib scatter function.\n\n    contourf_kwargs : dict (default: None)\n        Keyword arguments for underlying matplotlib contourf function.\n\n    contour_kwargs : dict (default: None)\n        Keyword arguments for underlying matplotlib contour function\n        (which draws the lines between decision regions).\n\n    scatter_highlight_kwargs : dict (default: None)\n        Keyword arguments for underlying matplotlib scatter function.\n\n    n_jobs : int or None, optional (default=None)\n        The number of CPUs to use to do the computation using Python's\n        multiprocessing library.\n        `None` means 1.\n        `-1` means using all processors. New in v0.22.0.\n\n    Returns\n    ---------\n    ax : matplotlib.axes.Axes object\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/\n\n    "
    check_Xy(X, y, y_int=True)
    dim = X.shape[1]
    if n_jobs is None:
        n_jobs = 1
    if ax is None:
        ax = plt.gca()
    plot_testdata = True
    if not isinstance(X_highlight, np.ndarray):
        if X_highlight is not None:
            raise ValueError('X_highlight must be a NumPy array or None')
        else:
            plot_testdata = False
    elif len(X_highlight.shape) < 2:
        raise ValueError('X_highlight must be a 2D array')
    if feature_index is not None:
        if dim == 1:
            raise ValueError('feature_index requires more than one training feature')
        try:
            (x_index, y_index) = feature_index
        except ValueError:
            raise ValueError('Unable to unpack feature_index. Make sure feature_index only has two dimensions.')
        try:
            (X[:, x_index], X[:, y_index])
        except IndexError:
            raise IndexError('feature_index values out of range. X.shape is {}, but feature_index is {}'.format(X.shape, feature_index))
    else:
        feature_index = (0, 1)
        (x_index, y_index) = feature_index
    if dim > 2:
        if filler_feature_values is None:
            raise ValueError('Filler values must be provided when X has more than 2 training features.')
        if filler_feature_ranges is not None:
            if not set(filler_feature_values) == set(filler_feature_ranges):
                raise ValueError('filler_feature_values and filler_feature_ranges must have the same keys')
        column_check = np.zeros(dim, dtype=bool)
        for idx in filler_feature_values:
            column_check[idx] = True
        for idx in feature_index:
            column_check[idx] = True
        if not all(column_check):
            missing_cols = np.argwhere(~column_check).flatten()
            raise ValueError('Column(s) {} need to be accounted for in either feature_index or filler_feature_values'.format(missing_cols))
    if n_jobs > mp.cpu_count():
        raise ValueError('Number of defined CPU cores is more than the available resources {} '.format(mp.cpu_count()))
    marker_gen = cycle(list(markers))
    n_classes = np.unique(y).shape[0]
    colors = colors.split(',')
    colors_gen = cycle(colors)
    colors = [next(colors_gen) for c in range(n_classes)]
    (x_min, x_max) = (X[:, x_index].min() - 1.0 / zoom_factor, X[:, x_index].max() + 1.0 / zoom_factor)
    if dim == 1:
        (y_min, y_max) = (-1, 1)
    else:
        (y_min, y_max) = (X[:, y_index].min() - 1.0 / zoom_factor, X[:, y_index].max() + 1.0 / zoom_factor)
    (xnum, ynum) = plt.gcf().dpi * plt.gcf().get_size_inches()
    (xnum, ynum) = (floor(xnum), ceil(ynum))
    (xx, yy) = np.meshgrid(np.linspace(x_min, x_max, num=xnum), np.linspace(y_min, y_max, num=ynum))
    if dim == 1:
        X_predict = np.array([xx.ravel()]).T
    else:
        X_grid = np.array([xx.ravel(), yy.ravel()]).T
        X_predict = np.zeros((X_grid.shape[0], dim))
        X_predict[:, x_index] = X_grid[:, 0]
        X_predict[:, y_index] = X_grid[:, 1]
        if dim > 2:
            for feature_idx in filler_feature_values:
                X_predict[:, feature_idx] = filler_feature_values[feature_idx]
    if n_jobs == 1:
        Z = clf.predict(X_predict.astype(X.dtype))
        Z = Z.reshape(xx.shape)
    else:
        if n_jobs == -1:
            cpus = mp.cpu_count()
        else:
            cpus = n_jobs
        pool = mp.Pool(cpus)
        partQuant = len(X_predict) / cpus
        partitions = []
        for n in range(cpus - 1):
            (start, end) = (np.floor(partQuant * n).astype(int), np.floor(partQuant * (n + 1)).astype(int))
            partitions.append(X_predict[start:end])
        partitions.append(X_predict[end:])
        xtype = X.dtype
        Z = pool.starmap(parallel, [(x, clf, xtype) for x in partitions])
        pool.close()
        Z = np.concatenate(Z)
        Z = Z.reshape(xx.shape)
    contourf_kwargs_default = {'alpha': 0.45, 'antialiased': True}
    contourf_kwargs = format_kwarg_dictionaries(default_kwargs=contourf_kwargs_default, user_kwargs=contourf_kwargs, protected_keys=['colors', 'levels'])
    cset = ax.contourf(xx, yy, Z, colors=colors, levels=np.arange(Z.max() + 2) - 0.5, **contourf_kwargs)
    contour_kwargs_default = {'linewidths': 0.5, 'colors': 'k', 'antialiased': True}
    contour_kwargs = format_kwarg_dictionaries(default_kwargs=contour_kwargs_default, user_kwargs=contour_kwargs, protected_keys=[])
    ax.contour(xx, yy, Z, cset.levels, **contour_kwargs)
    ax.axis([xx.min(), xx.max(), yy.min(), yy.max()])
    scatter_kwargs_default = {'alpha': 0.8, 'edgecolor': 'black'}
    scatter_kwargs = format_kwarg_dictionaries(default_kwargs=scatter_kwargs_default, user_kwargs=scatter_kwargs, protected_keys=['c', 'marker', 'label'])
    for (idx, c) in enumerate(np.unique(y)):
        if dim == 1:
            y_data = [0 for i in X[y == c]]
            x_data = X[y == c]
        elif dim == 2:
            y_data = X[y == c, y_index]
            x_data = X[y == c, x_index]
        elif dim > 2 and filler_feature_ranges is not None:
            class_mask = y == c
            feature_range_mask = get_feature_range_mask(X, filler_feature_values=filler_feature_values, filler_feature_ranges=filler_feature_ranges)
            y_data = X[class_mask & feature_range_mask, y_index]
            x_data = X[class_mask & feature_range_mask, x_index]
        else:
            continue
        ax.scatter(x=x_data, y=y_data, c=colors[idx], marker=next(marker_gen), label=c, **scatter_kwargs)
    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if dim == 1:
        ax.axes.get_yaxis().set_ticks([])
    if plot_testdata:
        if dim == 1:
            x_data = X_highlight
            y_data = [0 for i in X_highlight]
        elif dim == 2:
            x_data = X_highlight[:, x_index]
            y_data = X_highlight[:, y_index]
        else:
            feature_range_mask = get_feature_range_mask(X_highlight, filler_feature_values=filler_feature_values, filler_feature_ranges=filler_feature_ranges)
            y_data = X_highlight[feature_range_mask, y_index]
            x_data = X_highlight[feature_range_mask, x_index]
        scatter_highlight_defaults = {'c': 'none', 'edgecolor': 'black', 'alpha': 1.0, 'linewidths': 1, 'marker': 'o', 's': 80}
        scatter_highlight_kwargs = format_kwarg_dictionaries(default_kwargs=scatter_highlight_defaults, user_kwargs=scatter_highlight_kwargs)
        ax.scatter(x_data, y_data, **scatter_highlight_kwargs)
    if legend:
        if dim > 2 and filler_feature_ranges is None:
            pass
        else:
            (handles, labels) = ax.get_legend_handles_labels()
            ax.legend(handles, labels, framealpha=0.3, scatterpoints=1, loc=legend)
    return ax