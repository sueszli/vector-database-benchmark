import matplotlib.pyplot as plt
import numpy as np

def ecdf(x, y_label='ECDF', x_label=None, ax=None, percentile=None, ecdf_color=None, ecdf_marker='o', percentile_color='black', percentile_linestyle='--'):
    if False:
        return 10
    "Plots an Empirical Cumulative Distribution Function\n\n    Parameters\n    ----------\n    x : array or list, shape=[n_samples,]\n        Array-like object containing the feature values\n    y_label : str (default='ECDF')\n        Text label for the y-axis\n    x_label : str (default=None)\n        Text label for the x-axis\n    ax : matplotlib.axes.Axes (default: None)\n        An existing matplotlib Axes. Creates\n        one if ax=None\n    percentile : float (default=None)\n        Float between 0 and 1 for plotting a percentile\n        threshold line\n    ecdf_color : matplotlib color (default=None)\n        Color for the ECDF plot; uses matplotlib defaults\n        if None\n    ecdf_marker : matplotlib marker (default='o')\n        Marker style for the ECDF plot\n    percentile_color : matplotlib color (default='black')\n        Color for the percentile threshold if percentile is not None\n    percentile_linestyle : matplotlib linestyle (default='--')\n        Line style for the percentile threshold if percentile is not None\n\n    Returns\n    ---------\n    ax : matplotlib.axes.Axes object\n    percentile_threshold : float\n        Feature threshold at the percentile or None if `percentile=None`\n    percentile_count : Number of if percentile is not None\n        Number of samples that have a feature less or equal than\n        the feature threshold at a percentile threshold\n        or None if `percentile=None`\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/plotting/ecdf/\n\n    "
    if ax is None:
        ax = plt.gca()
    x = np.sort(x)
    y = np.arange(1, x.shape[0] + 1) / float(x.shape[0])
    ax.plot(x, y, marker=ecdf_marker, linestyle='', color=ecdf_color)
    ax.set_ylabel('ECDF')
    if x_label is not None:
        ax.set_xlabel(x_label)
    if percentile:
        targets = x[y <= percentile]
        percentile_threshold = targets.max()
        percentile_count = targets.shape[0]
        ax.axvline(percentile_threshold, color=percentile_color, linestyle=percentile_linestyle)
    else:
        percentile_threshold = None
        percentile_count = None
    return (ax, percentile_threshold, percentile_count)