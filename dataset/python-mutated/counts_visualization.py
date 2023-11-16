"""
Visualization functions for measurement counts.
"""
from collections import OrderedDict
import functools
import numpy as np
from qiskit.utils import optionals as _optionals
from qiskit.utils.deprecation import deprecate_arg
from qiskit.result import QuasiDistribution, ProbDistribution
from .exceptions import VisualizationError
from .utils import matplotlib_close_if_inline

def hamming_distance(str1, str2):
    if False:
        while True:
            i = 10
    'Calculate the Hamming distance between two bit strings\n\n    Args:\n        str1 (str): First string.\n        str2 (str): Second string.\n    Returns:\n        int: Distance between strings.\n    Raises:\n        VisualizationError: Strings not same length\n    '
    if len(str1) != len(str2):
        raise VisualizationError('Strings not same length.')
    return sum((s1 != s2 for (s1, s2) in zip(str1, str2)))
VALID_SORTS = ['asc', 'desc', 'hamming', 'value', 'value_desc']
DIST_MEAS = {'hamming': hamming_distance}

def _is_deprecated_data_format(data) -> bool:
    if False:
        i = 10
        return i + 15
    if not isinstance(data, list):
        data = [data]
    for dat in data:
        if isinstance(dat, (QuasiDistribution, ProbDistribution)) or isinstance(next(iter(dat.values())), float):
            return True
    return False

@deprecate_arg('data', deprecation_description='Using plot_histogram() ``data`` argument with QuasiDistribution, ProbDistribution, or a distribution dictionary', since='0.22.0', additional_msg='Instead, use ``plot_distribution()``.', predicate=_is_deprecated_data_format, pending=True, package_name='qiskit-terra')
def plot_histogram(data, figsize=(7, 5), color=None, number_to_keep=None, sort='asc', target_string=None, legend=None, bar_labels=True, title=None, ax=None, filename=None):
    if False:
        while True:
            i = 10
    'Plot a histogram of input counts data.\n\n    Args:\n        data (list or dict): This is either a list of dictionaries or a single\n            dict containing the values to represent (ex ``{\'001\': 130}``)\n        figsize (tuple): Figure size in inches.\n        color (list or str): String or list of strings for histogram bar colors.\n        number_to_keep (int): The number of terms to plot per dataset.  The rest is made into a\n            single bar called \'rest\'.  If multiple datasets are given, the ``number_to_keep``\n            applies to each dataset individually, which may result in more bars than\n            ``number_to_keep + 1``.  The ``number_to_keep`` applies to the total values, rather than\n            the x-axis sort.\n        sort (string): Could be `\'asc\'`, `\'desc\'`, `\'hamming\'`, `\'value\'`, or\n            `\'value_desc\'`. If set to `\'value\'` or `\'value_desc\'` the x axis\n            will be sorted by the number of counts for each bitstring.\n            Defaults to `\'asc\'`.\n        target_string (str): Target string if \'sort\' is a distance measure.\n        legend(list): A list of strings to use for labels of the data.\n            The number of entries must match the length of data (if data is a\n            list or 1 if it\'s a dict)\n        bar_labels (bool): Label each bar in histogram with counts value.\n        title (str): A string to use for the plot title\n        ax (matplotlib.axes.Axes): An optional Axes object to be used for\n            the visualization output. If none is specified a new matplotlib\n            Figure will be created and used. Additionally, if specified there\n            will be no returned Figure since it is redundant.\n        filename (str): file path to save image to.\n\n    Returns:\n        matplotlib.Figure:\n            A figure for the rendered histogram, if the ``ax``\n            kwarg is not set.\n\n    Raises:\n        MissingOptionalLibraryError: Matplotlib not available.\n        VisualizationError: When legend is provided and the length doesn\'t\n            match the input data.\n        VisualizationError: Input must be Counts or a dict\n\n    Examples:\n        .. plot::\n           :include-source:\n\n            # Plot two counts in the same figure with legends and colors specified.\n\n            from qiskit.visualization import plot_histogram\n\n            counts1 = {\'00\': 525, \'11\': 499}\n            counts2 = {\'00\': 511, \'11\': 514}\n\n            legend = [\'First execution\', \'Second execution\']\n\n            plot_histogram([counts1, counts2], legend=legend, color=[\'crimson\',\'midnightblue\'],\n                            title="New Histogram")\n\n            # You can sort the bitstrings using different methods.\n\n            counts = {\'001\': 596, \'011\': 211, \'010\': 50, \'000\': 117, \'101\': 33, \'111\': 8,\n                    \'100\': 6, \'110\': 3}\n\n            # Sort by the counts in descending order\n            hist1 = plot_histogram(counts, sort=\'value_desc\')\n\n            # Sort by the hamming distance (the number of bit flips to change from\n            # one bitstring to the other) from a target string.\n            hist2 = plot_histogram(counts, sort=\'hamming\', target_string=\'001\')\n    '
    if not isinstance(data, list):
        data = [data]
    kind = 'counts'
    for dat in data:
        if isinstance(dat, (QuasiDistribution, ProbDistribution)) or isinstance(next(iter(dat.values())), float):
            kind = 'distribution'
    return _plotting_core(data, figsize, color, number_to_keep, sort, target_string, legend, bar_labels, title, ax, filename, kind=kind)

def plot_distribution(data, figsize=(7, 5), color=None, number_to_keep=None, sort='asc', target_string=None, legend=None, bar_labels=True, title=None, ax=None, filename=None):
    if False:
        for i in range(10):
            print('nop')
    'Plot a distribution from input sampled data.\n\n    Args:\n        data (list or dict): This is either a list of dictionaries or a single\n            dict containing the values to represent (ex {\'001\': 130})\n        figsize (tuple): Figure size in inches.\n        color (list or str): String or list of strings for distribution bar colors.\n        number_to_keep (int): The number of terms to plot per dataset.  The rest is made into a\n            single bar called \'rest\'.  If multiple datasets are given, the ``number_to_keep``\n            applies to each dataset individually, which may result in more bars than\n            ``number_to_keep + 1``.  The ``number_to_keep`` applies to the total values, rather than\n            the x-axis sort.\n        sort (string): Could be `\'asc\'`, `\'desc\'`, `\'hamming\'`, `\'value\'`, or\n            `\'value_desc\'`. If set to `\'value\'` or `\'value_desc\'` the x axis\n            will be sorted by the maximum probability for each bitstring.\n            Defaults to `\'asc\'`.\n        target_string (str): Target string if \'sort\' is a distance measure.\n        legend(list): A list of strings to use for labels of the data.\n            The number of entries must match the length of data (if data is a\n            list or 1 if it\'s a dict)\n        bar_labels (bool): Label each bar in histogram with probability value.\n        title (str): A string to use for the plot title\n        ax (matplotlib.axes.Axes): An optional Axes object to be used for\n            the visualization output. If none is specified a new matplotlib\n            Figure will be created and used. Additionally, if specified there\n            will be no returned Figure since it is redundant.\n        filename (str): file path to save image to.\n\n    Returns:\n        matplotlib.Figure:\n            A figure for the rendered distribution, if the ``ax``\n            kwarg is not set.\n\n    Raises:\n        MissingOptionalLibraryError: Matplotlib not available.\n        VisualizationError: When legend is provided and the length doesn\'t\n            match the input data.\n\n    Examples:\n        .. plot::\n           :include-source:\n\n            # Plot two counts in the same figure with legends and colors specified.\n\n            from qiskit.visualization import plot_distribution\n\n            counts1 = {\'00\': 525, \'11\': 499}\n            counts2 = {\'00\': 511, \'11\': 514}\n\n            legend = [\'First execution\', \'Second execution\']\n\n            plot_distribution([counts1, counts2], legend=legend, color=[\'crimson\',\'midnightblue\'],\n                            title="New Distribution")\n\n            # You can sort the bitstrings using different methods.\n\n            counts = {\'001\': 596, \'011\': 211, \'010\': 50, \'000\': 117, \'101\': 33, \'111\': 8,\n                    \'100\': 6, \'110\': 3}\n\n            # Sort by the counts in descending order\n            dist1 = plot_distribution(counts, sort=\'value_desc\')\n\n            # Sort by the hamming distance (the number of bit flips to change from\n            # one bitstring to the other) from a target string.\n            dist2 = plot_distribution(counts, sort=\'hamming\', target_string=\'001\')\n\n    '
    return _plotting_core(data, figsize, color, number_to_keep, sort, target_string, legend, bar_labels, title, ax, filename, kind='distribution')

@_optionals.HAS_MATPLOTLIB.require_in_call
def _plotting_core(data, figsize=(7, 5), color=None, number_to_keep=None, sort='asc', target_string=None, legend=None, bar_labels=True, title=None, ax=None, filename=None, kind='counts'):
    if False:
        i = 10
        return i + 15
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    if sort not in VALID_SORTS:
        raise VisualizationError("Value of sort option, %s, isn't a valid choice. Must be 'asc', 'desc', 'hamming', 'value', 'value_desc'")
    if sort in DIST_MEAS and target_string is None:
        err_msg = 'Must define target_string when using distance measure.'
        raise VisualizationError(err_msg)
    if isinstance(data, dict):
        data = [data]
    if legend and len(legend) != len(data):
        raise VisualizationError(f"Length of legend ({len(legend)}) doesn't match number of input executions ({len(data)}).")
    if color is None:
        color = ['#648fff', '#dc267f', '#785ef0', '#ffb000', '#fe6100']
    elif isinstance(color, str):
        color = [color]
    if ax is None:
        (fig, ax) = plt.subplots(figsize=figsize)
    else:
        fig = None
    labels = sorted(functools.reduce(lambda x, y: x.union(y.keys()), data, set()))
    if number_to_keep is not None:
        labels.append('rest')
    if sort in DIST_MEAS:
        dist = []
        for item in labels:
            dist.append(DIST_MEAS[sort](item, target_string) if item != 'rest' else 0)
        labels = [list(x) for x in zip(*sorted(zip(dist, labels), key=lambda pair: pair[0]))][1]
    elif 'value' in sort:
        combined_counts = {}
        if isinstance(data, dict):
            combined_counts = data
        else:
            for counts in data:
                for count in counts:
                    prev_count = combined_counts.get(count, 0)
                    combined_counts[count] = max(prev_count, counts[count])
        labels = sorted(combined_counts.keys(), key=lambda key: combined_counts[key])
    length = len(data)
    width = 1 / (len(data) + 1)
    (labels_dict, all_pvalues, all_inds) = _plot_data(data, labels, number_to_keep, kind=kind)
    rects = []
    for (item, _) in enumerate(data):
        label = None
        for (idx, val) in enumerate(all_pvalues[item]):
            if not idx and legend:
                label = legend[item]
            if val > 0:
                rects.append(ax.bar(idx + item * width, val, width, label=label, color=color[item % len(color)], zorder=2))
                label = None
        bar_center = width / 2 * (length - 1)
        ax.set_xticks(all_inds[item] + bar_center)
        ax.set_xticklabels(labels_dict.keys(), fontsize=14, rotation=70)
        if bar_labels:
            for rect in rects:
                for rec in rect:
                    height = rec.get_height()
                    if kind == 'distribution':
                        height = round(height, 3)
                    if height >= 0.001:
                        ax.text(rec.get_x() + rec.get_width() / 2.0, 1.05 * height, str(height), ha='center', va='bottom', zorder=3)
                    else:
                        ax.text(rec.get_x() + rec.get_width() / 2.0, 1.05 * height, '0', ha='center', va='bottom', zorder=3)
    if kind == 'counts':
        ax.set_ylabel('Count', fontsize=14)
    else:
        ax.set_ylabel('Quasi-probability', fontsize=14)
    all_vals = np.concatenate(all_pvalues).ravel()
    min_ylim = 0.0
    if kind == 'distribution':
        min_ylim = min(0.0, min((1.1 * val for val in all_vals)))
    ax.set_ylim([min_ylim, min([1.1 * sum(all_vals), max((1.1 * val for val in all_vals))])])
    if 'desc' in sort:
        ax.invert_xaxis()
    ax.yaxis.set_major_locator(MaxNLocator(5))
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
    plt.grid(which='major', axis='y', zorder=0, linestyle='--')
    if title:
        plt.title(title)
    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), ncol=1, borderaxespad=0, frameon=True, fontsize=12)
    if fig:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)

def _keep_largest_items(execution, number_to_keep):
    if False:
        for i in range(10):
            print('nop')
    "Keep only the largest values in a dictionary, and sum the rest into a new key 'rest'."
    sorted_counts = sorted(execution.items(), key=lambda p: p[1])
    rest = sum((count for (key, count) in sorted_counts[:-number_to_keep]))
    return dict(sorted_counts[-number_to_keep:], rest=rest)

def _unify_labels(data):
    if False:
        print('Hello World!')
    'Make all dictionaries in data have the same set of keys, using 0 for missing values.'
    data = tuple(data)
    all_labels = set().union(*(execution.keys() for execution in data))
    base = {label: 0 for label in all_labels}
    out = []
    for execution in data:
        new_execution = base.copy()
        new_execution.update(execution)
        out.append(new_execution)
    return out

def _plot_data(data, labels, number_to_keep, kind='counts'):
    if False:
        i = 10
        return i + 15
    "Generate the data needed for plotting counts.\n\n    Parameters:\n        data (list or dict): This is either a list of dictionaries or a single\n            dict containing the values to represent (ex {'001': 130})\n        labels (list): The list of bitstring labels for the plot.\n        number_to_keep (int): The number of terms to plot and rest\n            is made into a single bar called 'rest'.\n        kind (str): One of 'counts' or 'distribution`\n\n    Returns:\n        tuple: tuple containing:\n            (dict): The labels actually used in the plotting.\n            (list): List of ndarrays for the bars in each experiment.\n            (list): Indices for the locations of the bars for each\n                    experiment.\n    "
    labels_dict = OrderedDict()
    all_pvalues = []
    all_inds = []
    if isinstance(data, dict):
        data = [data]
    if number_to_keep is not None:
        data = _unify_labels((_keep_largest_items(execution, number_to_keep) for execution in data))
    for execution in data:
        values = []
        for key in labels:
            if key not in execution:
                if number_to_keep is None:
                    labels_dict[key] = 1
                    values.append(0)
            else:
                labels_dict[key] = 1
                values.append(execution[key])
        if kind == 'counts':
            pvalues = np.array(values, dtype=int)
        else:
            pvalues = np.array(values, dtype=float)
            pvalues /= np.sum(pvalues)
        all_pvalues.append(pvalues)
        numelem = len(values)
        ind = np.arange(numelem)
        all_inds.append(ind)
    return (labels_dict, all_pvalues, all_inds)