"""Variations on boxplots."""
import numpy as np
from scipy.stats import gaussian_kde
from . import utils
__all__ = ['violinplot', 'beanplot']

def violinplot(data, ax=None, labels=None, positions=None, side='both', show_boxplot=True, plot_opts=None):
    if False:
        i = 10
        return i + 15
    '\n    Make a violin plot of each dataset in the `data` sequence.\n\n    A violin plot is a boxplot combined with a kernel density estimate of the\n    probability density function per point.\n\n    Parameters\n    ----------\n    data : sequence[array_like]\n        Data arrays, one array per value in `positions`.\n    ax : AxesSubplot, optional\n        If given, this subplot is used to plot in instead of a new figure being\n        created.\n    labels : list[str], optional\n        Tick labels for the horizontal axis.  If not given, integers\n        ``1..len(data)`` are used.\n    positions : array_like, optional\n        Position array, used as the horizontal axis of the plot.  If not given,\n        spacing of the violins will be equidistant.\n    side : {\'both\', \'left\', \'right\'}, optional\n        How to plot the violin.  Default is \'both\'.  The \'left\', \'right\'\n        options can be used to create asymmetric violin plots.\n    show_boxplot : bool, optional\n        Whether or not to show normal box plots on top of the violins.\n        Default is True.\n    plot_opts : dict, optional\n        A dictionary with plotting options.  Any of the following can be\n        provided, if not present in `plot_opts` the defaults will be used::\n\n          - \'violin_fc\', MPL color.  Fill color for violins.  Default is \'y\'.\n          - \'violin_ec\', MPL color.  Edge color for violins.  Default is \'k\'.\n          - \'violin_lw\', scalar.  Edge linewidth for violins.  Default is 1.\n          - \'violin_alpha\', float.  Transparancy of violins.  Default is 0.5.\n          - \'cutoff\', bool.  If True, limit violin range to data range.\n                Default is False.\n          - \'cutoff_val\', scalar.  Where to cut off violins if `cutoff` is\n                True.  Default is 1.5 standard deviations.\n          - \'cutoff_type\', {\'std\', \'abs\'}.  Whether cutoff value is absolute,\n                or in standard deviations.  Default is \'std\'.\n          - \'violin_width\' : float.  Relative width of violins.  Max available\n                space is 1, default is 0.8.\n          - \'label_fontsize\', MPL fontsize.  Adjusts fontsize only if given.\n          - \'label_rotation\', scalar.  Adjusts label rotation only if given.\n                Specify in degrees.\n          - \'bw_factor\', Adjusts the scipy gaussian_kde kernel. default: None.\n                Options for scalar or callable.\n\n    Returns\n    -------\n    Figure\n        If `ax` is None, the created figure.  Otherwise the figure to which\n        `ax` is connected.\n\n    See Also\n    --------\n    beanplot : Bean plot, builds on `violinplot`.\n    matplotlib.pyplot.boxplot : Standard boxplot.\n\n    Notes\n    -----\n    The appearance of violins can be customized with `plot_opts`.  If\n    customization of boxplot elements is required, set `show_boxplot` to False\n    and plot it on top of the violins by calling the Matplotlib `boxplot`\n    function directly.  For example::\n\n        violinplot(data, ax=ax, show_boxplot=False)\n        ax.boxplot(data, sym=\'cv\', whis=2.5)\n\n    It can happen that the axis labels or tick labels fall outside the plot\n    area, especially with rotated labels on the horizontal axis.  With\n    Matplotlib 1.1 or higher, this can easily be fixed by calling\n    ``ax.tight_layout()``.  With older Matplotlib one has to use ``plt.rc`` or\n    ``plt.rcParams`` to fix this, for example::\n\n        plt.rc(\'figure.subplot\', bottom=0.25)\n        violinplot(data, ax=ax)\n\n    References\n    ----------\n    J.L. Hintze and R.D. Nelson, "Violin Plots: A Box Plot-Density Trace\n    Synergism", The American Statistician, Vol. 52, pp.181-84, 1998.\n\n    Examples\n    --------\n    We use the American National Election Survey 1996 dataset, which has Party\n    Identification of respondents as independent variable and (among other\n    data) age as dependent variable.\n\n    >>> data = sm.datasets.anes96.load_pandas()\n    >>> party_ID = np.arange(7)\n    >>> labels = ["Strong Democrat", "Weak Democrat", "Independent-Democrat",\n    ...           "Independent-Indpendent", "Independent-Republican",\n    ...           "Weak Republican", "Strong Republican"]\n\n    Group age by party ID, and create a violin plot with it:\n\n    >>> plt.rcParams[\'figure.subplot.bottom\'] = 0.23  # keep labels visible\n    >>> age = [data.exog[\'age\'][data.endog == id] for id in party_ID]\n    >>> fig = plt.figure()\n    >>> ax = fig.add_subplot(111)\n    >>> sm.graphics.violinplot(age, ax=ax, labels=labels,\n    ...                        plot_opts={\'cutoff_val\':5, \'cutoff_type\':\'abs\',\n    ...                                   \'label_fontsize\':\'small\',\n    ...                                   \'label_rotation\':30})\n    >>> ax.set_xlabel("Party identification of respondent.")\n    >>> ax.set_ylabel("Age")\n    >>> plt.show()\n\n    .. plot:: plots/graphics_boxplot_violinplot.py\n    '
    plot_opts = {} if plot_opts is None else plot_opts
    if max([np.size(arr) for arr in data]) == 0:
        msg = 'No Data to make Violin: Try again!'
        raise ValueError(msg)
    (fig, ax) = utils.create_mpl_ax(ax)
    data = list(map(np.asarray, data))
    if positions is None:
        positions = np.arange(len(data)) + 1
    pos_span = np.max(positions) - np.min(positions)
    width = np.min([0.15 * np.max([pos_span, 1.0]), plot_opts.get('violin_width', 0.8) / 2.0])
    for (pos_data, pos) in zip(data, positions):
        _single_violin(ax, pos, pos_data, width, side, plot_opts)
    if show_boxplot:
        ax.boxplot(data, notch=1, positions=positions, vert=1)
    _set_ticks_labels(ax, data, labels, positions, plot_opts)
    return fig

def _single_violin(ax, pos, pos_data, width, side, plot_opts):
    if False:
        for i in range(10):
            print('nop')
    ''
    bw_factor = plot_opts.get('bw_factor', None)

    def _violin_range(pos_data, plot_opts):
        if False:
            i = 10
            return i + 15
        'Return array with correct range, with which violins can be plotted.'
        cutoff = plot_opts.get('cutoff', False)
        cutoff_type = plot_opts.get('cutoff_type', 'std')
        cutoff_val = plot_opts.get('cutoff_val', 1.5)
        s = 0.0
        if not cutoff:
            if cutoff_type == 'std':
                s = cutoff_val * np.std(pos_data)
            else:
                s = cutoff_val
        x_lower = kde.dataset.min() - s
        x_upper = kde.dataset.max() + s
        return np.linspace(x_lower, x_upper, 100)
    pos_data = np.asarray(pos_data)
    kde = gaussian_kde(pos_data, bw_method=bw_factor)
    xvals = _violin_range(pos_data, plot_opts)
    violin = kde.evaluate(xvals)
    violin = width * violin / violin.max()
    if side == 'both':
        (envelope_l, envelope_r) = (-violin + pos, violin + pos)
    elif side == 'right':
        (envelope_l, envelope_r) = (pos, violin + pos)
    elif side == 'left':
        (envelope_l, envelope_r) = (-violin + pos, pos)
    else:
        msg = "`side` parameter should be one of {'left', 'right', 'both'}."
        raise ValueError(msg)
    ax.fill_betweenx(xvals, envelope_l, envelope_r, facecolor=plot_opts.get('violin_fc', '#66c2a5'), edgecolor=plot_opts.get('violin_ec', 'k'), lw=plot_opts.get('violin_lw', 1), alpha=plot_opts.get('violin_alpha', 0.5))
    return (xvals, violin)

def _set_ticks_labels(ax, data, labels, positions, plot_opts):
    if False:
        print('Hello World!')
    'Set ticks and labels on horizontal axis.'
    ax.set_xlim([np.min(positions) - 0.5, np.max(positions) + 0.5])
    ax.set_xticks(positions)
    label_fontsize = plot_opts.get('label_fontsize')
    label_rotation = plot_opts.get('label_rotation')
    if label_fontsize or label_rotation:
        from matplotlib.artist import setp
    if labels is not None:
        if not len(labels) == len(data):
            msg = 'Length of `labels` should equal length of `data`.'
            raise ValueError(msg)
        xticknames = ax.set_xticklabels(labels)
        if label_fontsize:
            setp(xticknames, fontsize=label_fontsize)
        if label_rotation:
            setp(xticknames, rotation=label_rotation)
    return

def beanplot(data, ax=None, labels=None, positions=None, side='both', jitter=False, plot_opts={}):
    if False:
        for i in range(10):
            print('nop')
    '\n    Bean plot of each dataset in a sequence.\n\n    A bean plot is a combination of a `violinplot` (kernel density estimate of\n    the probability density function per point) with a line-scatter plot of all\n    individual data points.\n\n    Parameters\n    ----------\n    data : sequence[array_like]\n        Data arrays, one array per value in `positions`.\n    ax : AxesSubplot\n        If given, this subplot is used to plot in instead of a new figure being\n        created.\n    labels : list[str], optional\n        Tick labels for the horizontal axis.  If not given, integers\n        ``1..len(data)`` are used.\n    positions : array_like, optional\n        Position array, used as the horizontal axis of the plot.  If not given,\n        spacing of the violins will be equidistant.\n    side : {\'both\', \'left\', \'right\'}, optional\n        How to plot the violin.  Default is \'both\'.  The \'left\', \'right\'\n        options can be used to create asymmetric violin plots.\n    jitter : bool, optional\n        If True, jitter markers within violin instead of plotting regular lines\n        around the center.  This can be useful if the data is very dense.\n    plot_opts : dict, optional\n        A dictionary with plotting options.  All the options for `violinplot`\n        can be specified, they will simply be passed to `violinplot`.  Options\n        specific to `beanplot` are:\n\n          - \'violin_width\' : float.  Relative width of violins.  Max available\n                space is 1, default is 0.8.\n          - \'bean_color\', MPL color.  Color of bean plot lines.  Default is \'k\'.\n                Also used for jitter marker edge color if `jitter` is True.\n          - \'bean_size\', scalar.  Line length as a fraction of maximum length.\n                Default is 0.5.\n          - \'bean_lw\', scalar.  Linewidth, default is 0.5.\n          - \'bean_show_mean\', bool.  If True (default), show mean as a line.\n          - \'bean_show_median\', bool.  If True (default), show median as a\n                marker.\n          - \'bean_mean_color\', MPL color.  Color of mean line.  Default is \'b\'.\n          - \'bean_mean_lw\', scalar.  Linewidth of mean line, default is 2.\n          - \'bean_mean_size\', scalar.  Line length as a fraction of maximum length.\n                Default is 0.5.\n          - \'bean_median_color\', MPL color.  Color of median marker.  Default\n                is \'r\'.\n          - \'bean_median_marker\', MPL marker.  Marker type, default is \'+\'.\n          - \'jitter_marker\', MPL marker.  Marker type for ``jitter=True``.\n                Default is \'o\'.\n          - \'jitter_marker_size\', int.  Marker size.  Default is 4.\n          - \'jitter_fc\', MPL color.  Jitter marker face color.  Default is None.\n          - \'bean_legend_text\', str.  If given, add a legend with given text.\n\n    Returns\n    -------\n    Figure\n        If `ax` is None, the created figure.  Otherwise the figure to which\n        `ax` is connected.\n\n    See Also\n    --------\n    violinplot : Violin plot, also used internally in `beanplot`.\n    matplotlib.pyplot.boxplot : Standard boxplot.\n\n    References\n    ----------\n    P. Kampstra, "Beanplot: A Boxplot Alternative for Visual Comparison of\n    Distributions", J. Stat. Soft., Vol. 28, pp. 1-9, 2008.\n\n    Examples\n    --------\n    We use the American National Election Survey 1996 dataset, which has Party\n    Identification of respondents as independent variable and (among other\n    data) age as dependent variable.\n\n    >>> data = sm.datasets.anes96.load_pandas()\n    >>> party_ID = np.arange(7)\n    >>> labels = ["Strong Democrat", "Weak Democrat", "Independent-Democrat",\n    ...           "Independent-Indpendent", "Independent-Republican",\n    ...           "Weak Republican", "Strong Republican"]\n\n    Group age by party ID, and create a violin plot with it:\n\n    >>> plt.rcParams[\'figure.subplot.bottom\'] = 0.23  # keep labels visible\n    >>> age = [data.exog[\'age\'][data.endog == id] for id in party_ID]\n    >>> fig = plt.figure()\n    >>> ax = fig.add_subplot(111)\n    >>> sm.graphics.beanplot(age, ax=ax, labels=labels,\n    ...                      plot_opts={\'cutoff_val\':5, \'cutoff_type\':\'abs\',\n    ...                                 \'label_fontsize\':\'small\',\n    ...                                 \'label_rotation\':30})\n    >>> ax.set_xlabel("Party identification of respondent.")\n    >>> ax.set_ylabel("Age")\n    >>> plt.show()\n\n    .. plot:: plots/graphics_boxplot_beanplot.py\n    '
    (fig, ax) = utils.create_mpl_ax(ax)
    data = list(map(np.asarray, data))
    if positions is None:
        positions = np.arange(len(data)) + 1
    pos_span = np.max(positions) - np.min(positions)
    violin_width = np.min([0.15 * np.max([pos_span, 1.0]), plot_opts.get('violin_width', 0.8) / 2.0])
    bean_width = np.min([0.15 * np.max([pos_span, 1.0]), plot_opts.get('bean_size', 0.5) / 2.0])
    bean_mean_width = np.min([0.15 * np.max([pos_span, 1.0]), plot_opts.get('bean_mean_size', 0.5) / 2.0])
    legend_txt = plot_opts.get('bean_legend_text', None)
    for (pos_data, pos) in zip(data, positions):
        (xvals, violin) = _single_violin(ax, pos, pos_data, violin_width, side, plot_opts)
        if jitter:
            jitter_coord = pos + _jitter_envelope(pos_data, xvals, violin, side)
            ax.plot(jitter_coord, pos_data, ls='', marker=plot_opts.get('jitter_marker', 'o'), ms=plot_opts.get('jitter_marker_size', 4), mec=plot_opts.get('bean_color', 'k'), mew=1, mfc=plot_opts.get('jitter_fc', 'none'), label=legend_txt)
        else:
            ax.hlines(pos_data, pos - bean_width, pos + bean_width, lw=plot_opts.get('bean_lw', 0.5), color=plot_opts.get('bean_color', 'k'), label=legend_txt)
        if legend_txt is not None:
            _show_legend(ax)
            legend_txt = None
        if plot_opts.get('bean_show_mean', True):
            ax.hlines(np.mean(pos_data), pos - bean_mean_width, pos + bean_mean_width, lw=plot_opts.get('bean_mean_lw', 2.0), color=plot_opts.get('bean_mean_color', 'b'))
        if plot_opts.get('bean_show_median', True):
            ax.plot(pos, np.median(pos_data), marker=plot_opts.get('bean_median_marker', '+'), color=plot_opts.get('bean_median_color', 'r'))
    _set_ticks_labels(ax, data, labels, positions, plot_opts)
    return fig

def _jitter_envelope(pos_data, xvals, violin, side):
    if False:
        while True:
            i = 10
    'Determine envelope for jitter markers.'
    if side == 'both':
        (low, high) = (-1.0, 1.0)
    elif side == 'right':
        (low, high) = (0, 1.0)
    elif side == 'left':
        (low, high) = (-1.0, 0)
    else:
        raise ValueError('`side` input incorrect: %s' % side)
    jitter_envelope = np.interp(pos_data, xvals, violin)
    jitter_coord = jitter_envelope * np.random.uniform(low=low, high=high, size=pos_data.size)
    return jitter_coord

def _show_legend(ax):
    if False:
        while True:
            i = 10
    'Utility function to show legend.'
    leg = ax.legend(loc=1, shadow=True, fancybox=True, labelspacing=0.2, borderpad=0.15)
    ltext = leg.get_texts()
    llines = leg.get_lines()
    frame = leg.get_frame()
    from matplotlib.artist import setp
    setp(ltext, fontsize='small')
    setp(llines, linewidth=1)