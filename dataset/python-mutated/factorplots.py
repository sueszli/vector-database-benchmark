"""
Authors:    Josef Perktold, Skipper Seabold, Denis A. Engemann
"""
from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.graphics.plottools import rainbow
import statsmodels.graphics.utils as utils

def interaction_plot(x, trace, response, func='mean', ax=None, plottype='b', xlabel=None, ylabel=None, colors=None, markers=None, linestyles=None, legendloc='best', legendtitle=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Interaction plot for factor level statistics.\n\n    Note. If categorial factors are supplied levels will be internally\n    recoded to integers. This ensures matplotlib compatibility. Uses\n    a DataFrame to calculate an `aggregate` statistic for each level of the\n    factor or group given by `trace`.\n\n    Parameters\n    ----------\n    x : array_like\n        The `x` factor levels constitute the x-axis. If a `pandas.Series` is\n        given its name will be used in `xlabel` if `xlabel` is None.\n    trace : array_like\n        The `trace` factor levels will be drawn as lines in the plot.\n        If `trace` is a `pandas.Series` its name will be used as the\n        `legendtitle` if `legendtitle` is None.\n    response : array_like\n        The reponse or dependent variable. If a `pandas.Series` is given\n        its name will be used in `ylabel` if `ylabel` is None.\n    func : function\n        Anything accepted by `pandas.DataFrame.aggregate`. This is applied to\n        the response variable grouped by the trace levels.\n    ax : axes, optional\n        Matplotlib axes instance\n    plottype : str {'line', 'scatter', 'both'}, optional\n        The type of plot to return. Can be 'l', 's', or 'b'\n    xlabel : str, optional\n        Label to use for `x`. Default is 'X'. If `x` is a `pandas.Series` it\n        will use the series names.\n    ylabel : str, optional\n        Label to use for `response`. Default is 'func of response'. If\n        `response` is a `pandas.Series` it will use the series names.\n    colors : list, optional\n        If given, must have length == number of levels in trace.\n    markers : list, optional\n        If given, must have length == number of levels in trace\n    linestyles : list, optional\n        If given, must have length == number of levels in trace.\n    legendloc : {None, str, int}\n        Location passed to the legend command.\n    legendtitle : {None, str}\n        Title of the legend.\n    **kwargs\n        These will be passed to the plot command used either plot or scatter.\n        If you want to control the overall plotting options, use kwargs.\n\n    Returns\n    -------\n    Figure\n        The figure given by `ax.figure` or a new instance.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> np.random.seed(12345)\n    >>> weight = np.random.randint(1,4,size=60)\n    >>> duration = np.random.randint(1,3,size=60)\n    >>> days = np.log(np.random.randint(1,30, size=60))\n    >>> fig = interaction_plot(weight, duration, days,\n    ...             colors=['red','blue'], markers=['D','^'], ms=10)\n    >>> import matplotlib.pyplot as plt\n    >>> plt.show()\n\n    .. plot::\n\n       import numpy as np\n       from statsmodels.graphics.factorplots import interaction_plot\n       np.random.seed(12345)\n       weight = np.random.randint(1,4,size=60)\n       duration = np.random.randint(1,3,size=60)\n       days = np.log(np.random.randint(1,30, size=60))\n       fig = interaction_plot(weight, duration, days,\n                   colors=['red','blue'], markers=['D','^'], ms=10)\n       import matplotlib.pyplot as plt\n       #plt.show()\n    "
    from pandas import DataFrame
    (fig, ax) = utils.create_mpl_ax(ax)
    response_name = ylabel or getattr(response, 'name', 'response')
    func_name = getattr(func, '__name__', str(func))
    ylabel = '%s of %s' % (func_name, response_name)
    xlabel = xlabel or getattr(x, 'name', 'X')
    legendtitle = legendtitle or getattr(trace, 'name', 'Trace')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    x_values = x_levels = None
    if isinstance(x[0], str):
        x_levels = [l for l in np.unique(x)]
        x_values = lrange(len(x_levels))
        x = _recode(x, dict(zip(x_levels, x_values)))
    data = DataFrame(dict(x=x, trace=trace, response=response))
    plot_data = data.groupby(['trace', 'x']).aggregate(func).reset_index()
    n_trace = len(plot_data['trace'].unique())
    linestyles = ['-'] * n_trace if linestyles is None else linestyles
    markers = ['.'] * n_trace if markers is None else markers
    colors = rainbow(n_trace) if colors is None else colors
    if len(linestyles) != n_trace:
        raise ValueError('Must be a linestyle for each trace level')
    if len(markers) != n_trace:
        raise ValueError('Must be a marker for each trace level')
    if len(colors) != n_trace:
        raise ValueError('Must be a color for each trace level')
    if plottype == 'both' or plottype == 'b':
        for (i, (values, group)) in enumerate(plot_data.groupby('trace')):
            label = str(group['trace'].values[0])
            ax.plot(group['x'], group['response'], color=colors[i], marker=markers[i], label=label, linestyle=linestyles[i], **kwargs)
    elif plottype == 'line' or plottype == 'l':
        for (i, (values, group)) in enumerate(plot_data.groupby('trace')):
            label = str(group['trace'].values[0])
            ax.plot(group['x'], group['response'], color=colors[i], label=label, linestyle=linestyles[i], **kwargs)
    elif plottype == 'scatter' or plottype == 's':
        for (i, (values, group)) in enumerate(plot_data.groupby('trace')):
            label = str(group['trace'].values[0])
            ax.scatter(group['x'], group['response'], color=colors[i], label=label, marker=markers[i], **kwargs)
    else:
        raise ValueError('Plot type %s not understood' % plottype)
    ax.legend(loc=legendloc, title=legendtitle)
    ax.margins(0.1)
    if all([x_levels, x_values]):
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_levels)
    return fig

def _recode(x, levels):
    if False:
        i = 10
        return i + 15
    ' Recode categorial data to int factor.\n\n    Parameters\n    ----------\n    x : array_like\n        array like object supporting with numpy array methods of categorially\n        coded data.\n    levels : dict\n        mapping of labels to integer-codings\n\n    Returns\n    -------\n    out : instance numpy.ndarray\n    '
    from pandas import Series
    name = None
    index = None
    if isinstance(x, Series):
        name = x.name
        index = x.index
        x = x.values
    if x.dtype.type not in [np.str_, np.object_]:
        raise ValueError('This is not a categorial factor. Array of str type required.')
    elif not isinstance(levels, dict):
        raise ValueError('This is not a valid value for levels. Dict required.')
    elif not (np.unique(x) == np.unique(list(levels.keys()))).all():
        raise ValueError('The levels do not match the array values.')
    else:
        out = np.empty(x.shape[0], dtype=int)
        for (level, coding) in levels.items():
            out[x == level] = coding
        if name:
            out = Series(out, name=name, index=index)
        return out