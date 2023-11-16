import ast
import functools
import numpy as np
import logging
from vaex.dataframe import DataFrame, _hidden
from vaex.docstrings import docsubst
from vaex.utils import _parse_n, _parse_f, _ensure_strings_from_expressions, _ensure_list, _expand_limits, _expand_shape, _expand, _parse_reduction, _issequence, listify
import vaex.image
from .vector import plot2d_vector
from .tensor import plot2d_tensor
from .contour import plot2d_contour
import warnings
logger = logging.getLogger('vaex.viz')

def add_plugin():
    if False:
        while True:
            i = 10
    pass

def patch(f):
    if False:
        for i in range(10):
            print('nop')
    'Adds method f to the DataFrame class'
    name = f.__name__
    setattr(DataFrame, name, _hidden(f))
    return f

def viz_method(f):
    if False:
        print('Hello World!')

    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return f(self.df, *args, **kwargs)
    from . import DataFrameAccessorViz as cls
    setattr(cls, f.__name__, wrapper)
patch(plot2d_vector)
patch(plot2d_tensor)
patch(plot2d_contour)
max_labels = 10

@patch
def plot1d(self, *args, **kwargs):
    if False:
        print('Hello World!')
    warnings.warn('`plot1d` is deprecated and it will be removed in version 5.x. Please use `df.viz.histogram` instead.')
    self.viz.histogram(*args, **kwargs)

@viz_method
@docsubst
def histogram(self, x=None, what='count(*)', grid=None, shape=64, facet=None, limits=None, figsize=None, f='identity', n=None, normalize_axis=None, xlabel=None, ylabel=None, label=None, selection=None, show=False, tight_layout=True, hardcopy=None, progress=None, **kwargs):
    if False:
        while True:
            i = 10
    "Plot a histogram.\n\n    Example:\n\n    >>> df.histogram(df.x)\n    >>> df.histogram(df.x, limits=[0, 100], shape=100)\n    >>> df.histogram(df.x, what='mean(y)', limits=[0, 100], shape=100)\n\n    If you want to do a computation yourself, pass the grid argument, but you are responsible for passing the\n    same limits arguments:\n\n    >>> counts = df.mean(df.y, binby=df.x, limits=[0, 100], shape=100)/100.\n    >>> df.histogram(df.x, limits=[0, 100], shape=100, grid=means, label='mean(y)/100')\n\n    :param x: {expression_one}\n    :param what: What to plot, count(*) will show a N-d histogram, mean('x'), the mean of the x column, sum('x') the sum\n    :param grid: {grid}\n    :param shape: {shape}\n    :param facet: Expression to produce facetted plots ( facet='x:0,1,12' will produce 12 plots with x in a range between 0 and 1)\n    :param limits: {limits}\n    :param figsize: (x, y) tuple passed to plt.figure for setting the figure size\n    :param f: transform values by: 'identity' does nothing 'log' or 'log10' will show the log of the value\n    :param n: normalization function, currently only 'normalize' is supported, or None for no normalization\n    :param normalize_axis: which axes to normalize on, None means normalize by the global maximum.\n    :param selection: {selection}\n    :param xlabel: String for label on x axis (may contain latex)\n    :param ylabel: Same for y axis\n    :param: tight_layout: call plt.tight_layout or not\n    :param kwargs: extra argument passed to plt.plot\n    :return:\n    "
    import matplotlib.pyplot as plt
    f = _parse_f(f)
    n = _parse_n(n)
    if type(shape) == int:
        shape = (shape,)
    binby = []
    x = _ensure_strings_from_expressions(x)
    for expression in [x]:
        if expression is not None:
            binby = [expression] + binby
    limits = self.limits(binby, limits)
    if figsize is not None:
        plt.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
    fig = plt.gcf()
    import re
    if facet is not None:
        match = re.match('(.*):(.*),(.*),(.*)', facet)
        if match:
            groups = match.groups()
            facet_expression = groups[0]
            facet_limits = [ast.literal_eval(groups[1]), ast.literal_eval(groups[2])]
            facet_count = ast.literal_eval(groups[3])
            limits.append(facet_limits)
            binby.append(facet_expression)
            shape = (facet_count,) + shape
        else:
            raise ValueError("Could not understand 'facet' argument %r, expected something in form: 'column:-1,10:5'" % facet)
    if grid is None:
        if what:
            if isinstance(what, vaex.stat.Expression):
                grid = what.calculate(self, binby=binby, limits=limits, shape=shape, selection=selection)
            else:
                what = what.strip()
                index = what.index('(')
                import re
                groups = re.match('(.*)\\((.*)\\)', what).groups()
                if groups and len(groups) == 2:
                    function = groups[0]
                    arguments = groups[1].strip()
                    functions = ['mean', 'sum', 'std', 'count']
                    if function in functions:
                        grid = getattr(vaex.stat, function)(arguments).calculate(self, binby=binby, limits=limits, shape=shape, selection=selection, progress=progress)
                    elif function == 'count' and arguments == '*':
                        grid = self.count(binby=binby, shape=shape, limits=limits, selection=selection, progress=progress)
                    elif function == 'cumulative' and arguments == '*':
                        grid = self.count(binby=binby, shape=shape, limits=limits, selection=selection, progress=progress)
                        grid = np.cumsum(grid)
                    else:
                        raise ValueError("Could not understand method: %s, expected one of %r'" % (function, functions))
                else:
                    raise ValueError("Could not understand 'what' argument %r, expected something in form: 'count(*)', 'mean(x)'" % what)
        else:
            grid = self.histogram(binby, size=shape, limits=limits, selection=selection)
    if len(grid.shape) > 1 and _issequence(selection):
        grid = grid.T
    fgrid = f(grid)
    if n is not None:
        ngrid = fgrid / fgrid.sum()
    else:
        ngrid = fgrid
    (xmin, xmax) = limits[-1]
    if facet:
        N = len(grid[-1])
    else:
        N = len(grid)
    xexpression = binby[0]
    xar = np.arange(N + 1) / (N - 0.0) * (xmax - xmin) + xmin
    label = str(label or selection or x)
    if facet:
        import math
        (rows, columns) = (int(math.ceil(facet_count / 4.0)), 4)
        values = np.linspace(facet_limits[0], facet_limits[1], facet_count + 1)
        for i in range(facet_count):
            ax = plt.subplot(rows, columns, i + 1)
            value = ax.plot(xar, ngrid[i], drawstyle='steps-mid', label=label, **kwargs)
            (v1, v2) = (values[i], values[i + 1])
            plt.xlabel(xlabel or x)
            plt.ylabel(ylabel or what)
            ax.set_title('%3f <= %s < %3f' % (v1, facet_expression, v2))
            if self.iscategory(xexpression):
                labels = self.category_labels(xexpression)
                step = len(labels) // max_labels
                plt.xticks(range(len(labels))[::step], labels[::step], size='small')
    else:
        plt.xlabel(xlabel or self.label(x))
        plt.ylabel(ylabel or what)
        g = np.concatenate([ngrid[0:1], ngrid])
        value = plt.plot(xar, g, drawstyle='steps-pre', label=label, **kwargs)
        if self.iscategory(xexpression):
            labels = self.category_labels(xexpression)
            step = len(labels) // max_labels
            plt.xticks(range(len(labels))[::step], labels[::step], size='small')
    if tight_layout:
        plt.tight_layout()
    if hardcopy:
        plt.savefig(hardcopy)
    if show:
        plt.show()
    return value

@patch
def scatter(self, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    warnings.warn('`scatter` is deprecated and it will be removed in version 5.x. Please use `df.viz.scatter` instead.')
    self.viz.scatter(*args, **kwargs)

@viz_method
@docsubst
def scatter(self, x, y, xerr=None, yerr=None, cov=None, corr=None, s_expr=None, c_expr=None, labels=None, selection=None, length_limit=50000, length_check=True, label=None, xlabel=None, ylabel=None, errorbar_kwargs={}, ellipse_kwargs={}, **kwargs):
    if False:
        print('Hello World!')
    'Viz (small amounts) of data in 2d using a scatter plot\n\n    Convenience wrapper around plt.scatter when for working with small DataFrames or selections\n\n    :param x: {expression_one}\n    :param y: {expression_one}\n    :param s_expr: When given, use if for the s (size) argument of plt.scatter\n    :param c_expr: When given, use if for the c (color) argument of plt.scatter\n    :param labels: Annotate the points with these text values\n    :param selection: {selection1}\n    :param length_limit: maximum number of rows it will plot\n    :param length_check: should we do the maximum row check or not?\n    :param label: label for the legend\n    :param xlabel: label for x axis, if None .label(x) is used\n    :param ylabel: label for y axis, if None .label(y) is used\n    :param errorbar_kwargs: extra dict with arguments passed to plt.errorbar\n    :param kwargs: extra arguments passed to plt.scatter\n    :return:\n    '
    import matplotlib.pyplot as plt
    x = _ensure_strings_from_expressions(x)
    y = _ensure_strings_from_expressions(y)
    label = str(label or selection)
    selection = _ensure_strings_from_expressions(selection)
    if length_check:
        count = self.count(selection=selection)
        if count > length_limit:
            raise ValueError('the number of rows (%d) is above the limit (%d), pass length_check=False, or increase length_limit' % (count, length_limit))
    x_values = self.evaluate(x, selection=selection)
    y_values = self.evaluate(y, selection=selection)
    if s_expr:
        kwargs['s'] = self.evaluate(s_expr, selection=selection)
    if c_expr:
        kwargs['c'] = self.evaluate(c_expr, selection=selection)
    plt.xlabel(xlabel or self.label(x))
    plt.ylabel(ylabel or self.label(y))
    s = plt.scatter(x_values, y_values, label=label, **kwargs)
    if labels:
        label_values = self.evaluate(labels, selection=selection)
        for (i, label_value) in enumerate(label_values):
            plt.annotate(label_value, (x_values[i], y_values[i]))
    xerr_values = None
    yerr_values = None
    if cov is not None or corr is not None:
        from matplotlib.patches import Ellipse
        sx = self.evaluate(xerr, selection=selection)
        sy = self.evaluate(yerr, selection=selection)
        if corr is not None:
            sxy = self.evaluate(corr, selection=selection) * sx * sy
        elif cov is not None:
            sxy = self.evaluate(cov, selection=selection)
        cov_matrix = np.zeros((len(sx), 2, 2))
        cov_matrix[:, 0, 0] = sx ** 2
        cov_matrix[:, 1, 1] = sy ** 2
        cov_matrix[:, 0, 1] = cov_matrix[:, 1, 0] = sxy
        ax = plt.gca()
        ellipse_kwargs = dict(ellipse_kwargs)
        ellipse_kwargs['facecolor'] = ellipse_kwargs.get('facecolor', 'none')
        ellipse_kwargs['edgecolor'] = ellipse_kwargs.get('edgecolor', 'black')
        for i in range(len(sx)):
            (eigen_values, eigen_vectors) = np.linalg.eig(cov_matrix[i])
            indices = np.argsort(eigen_values)[::-1]
            eigen_values = eigen_values[indices]
            eigen_vectors = eigen_vectors[:, indices]
            v1 = eigen_vectors[:, 0]
            v2 = eigen_vectors[:, 1]
            varx = cov_matrix[i, 0, 0]
            vary = cov_matrix[i, 1, 1]
            angle = np.arctan2(v1[1], v1[0])
            if eigen_values[1] < 0 and abs(eigen_values[1] / eigen_values[0]) < 1e-10:
                eigen_values[1] = 0
            if eigen_values[0] < 0 or eigen_values[1] < 0:
                raise ValueError('neg val')
            (width, height) = (np.sqrt(np.max(eigen_values)), np.sqrt(np.min(eigen_values)))
            e = Ellipse(xy=(x_values[i], y_values[i]), width=width, height=height, angle=np.degrees(angle), **ellipse_kwargs)
            ax.add_artist(e)
    else:
        if xerr is not None:
            if _issequence(xerr):
                assert len(xerr) == 2, 'if xerr is a sequence it should be of length 2'
                xerr_values = [self.evaluate(xerr[0], selection=selection), self.evaluate(xerr[1], selection=selection)]
            else:
                xerr_values = self.evaluate(xerr, selection=selection)
        if yerr is not None:
            if _issequence(yerr):
                assert len(yerr) == 2, 'if yerr is a sequence it should be of length 2'
                yerr_values = [self.evaluate(yerr[0], selection=selection), self.evaluate(yerr[1], selection=selection)]
            else:
                yerr_values = self.evaluate(yerr, selection=selection)
        if xerr_values is not None or yerr_values is not None:
            errorbar_kwargs = dict(errorbar_kwargs)
            errorbar_kwargs['fmt'] = errorbar_kwargs.get('fmt', 'none')
            plt.errorbar(x_values, y_values, yerr=yerr_values, xerr=xerr_values, **errorbar_kwargs)
    return s

@patch
def plot(self, *args, **kwargs):
    if False:
        print('Hello World!')
    warnings.warn('`plot` is deprecated and it will be removed in version 5.x. Please `df.viz.heatmap` instead.')
    self.viz.heatmap(*args, **kwargs)

@viz_method
@docsubst
def heatmap(self, x=None, y=None, z=None, what='count(*)', vwhat=None, reduce=['colormap'], f=None, normalize='normalize', normalize_axis='what', vmin=None, vmax=None, shape=256, vshape=32, limits=None, grid=None, colormap='afmhot', figsize=None, xlabel=None, ylabel=None, aspect='auto', tight_layout=True, interpolation='nearest', show=False, colorbar=True, colorbar_label=None, selection=None, selection_labels=None, title=None, background_color='white', pre_blend=False, background_alpha=1.0, visual=dict(x='x', y='y', layer='z', fade='selection', row='subspace', column='what'), smooth_pre=None, smooth_post=None, wrap=True, wrap_columns=4, return_extra=False, hardcopy=None):
    if False:
        print('Hello World!')
    "Viz data in a 2d histogram/heatmap.\n\n    Declarative plotting of statistical plots using matplotlib, supports subplots, selections, layers.\n\n    Instead of passing x and y, pass a list as x argument for multiple panels. Give what a list of options to have multiple\n    panels. When both are present then will be origanized in a column/row order.\n\n    This methods creates a 6 dimensional 'grid', where each dimension can map the a visual dimension.\n    The grid dimensions are:\n\n    * x: shape determined by shape, content by x argument or the first dimension of each space\n    * y:   ,,\n    * z:  related to the z argument\n    * selection: shape equals length of selection argument\n    * what: shape equals length of what argument\n    * space: shape equals length of x argument if multiple values are given\n\n     By default, this its shape is (1, 1, 1, 1, shape, shape) (where x is the last dimension)\n\n    The visual dimensions are\n\n    * x: x coordinate on a plot / image (default maps to grid's x)\n    * y: y   ,,                         (default maps to grid's y)\n    * layer: each image in this dimension is blended togeher to one image (default maps to z)\n    * fade: each image is shown faded after the next image (default mapt to selection)\n    * row: rows of subplots (default maps to space)\n    * columns: columns of subplot (default maps to what)\n\n    All these mappings can be changes by the visual argument, some examples:\n\n    >>> df.viz.heatmap('x', 'y', what=['mean(x)', 'correlation(vx, vy)'])\n\n    Will plot each 'what' as a column.\n\n    >>> df.viz.heatmap('x', 'y', selection=['FeH < -3', '(FeH >= -3) & (FeH < -2)'], visual=dict(column='selection'))\n\n    Will plot each selection as a column, instead of a faded on top of each other.\n\n\n\n\n\n    :param x: Expression to bin in the x direction (by default maps to x), or list of pairs, like [['x', 'y'], ['x', 'z']], if multiple pairs are given, this dimension maps to rows by default\n    :param y:                          y           (by default maps to y)\n    :param z: Expression to bin in the z direction, followed by a :start,end,shape  signature, like 'FeH:-3,1:5' will produce 5 layers between -10 and 10 (by default maps to layer)\n    :param what: What to plot, count(*) will show a N-d histogram, mean('x'), the mean of the x column, sum('x') the sum, std('x') the standard deviation, correlation('vx', 'vy') the correlation coefficient. Can also be a list of values, like ['count(x)', std('vx')], (by default maps to column)\n    :param reduce:\n    :param f: transform values by: 'identity' does nothing 'log' or 'log10' will show the log of the value\n    :param normalize: normalization function, currently only 'normalize' is supported\n    :param normalize_axis: which axes to normalize on, None means normalize by the global maximum.\n    :param vmin: instead of automatic normalization, (using normalize and normalization_axis) scale the data between vmin and vmax to [0, 1]\n    :param vmax: see vmin\n    :param shape: {shape}\n    :param limits: {limits}\n    :param grid: {grid}\n    :param colormap: matplotlib colormap to use\n    :param figsize: (x, y) tuple passed to plt.figure for setting the figure size\n    :param xlabel:\n    :param ylabel:\n    :param aspect:\n    :param tight_layout: call plt.tight_layout or not\n    :param colorbar: plot a colorbar or not\n    :param selection: {selection1}\n    :param interpolation: interpolation for imshow, possible options are: 'nearest', 'bilinear', 'bicubic', see matplotlib for more\n    :param return_extra:\n    :return:\n    "
    import matplotlib
    import matplotlib.pyplot as plt
    n = _parse_n(normalize)
    if type(shape) == int:
        shape = (shape,) * 2
    binby = []
    x = _ensure_strings_from_expressions(x)
    y = _ensure_strings_from_expressions(y)
    for expression in [y, x]:
        if expression is not None:
            binby = [expression] + binby
    fig = plt.gcf()
    if figsize is not None:
        fig.set_size_inches(*figsize)
    import re
    what_units = None
    whats = _ensure_list(what)
    selections = _ensure_list(selection)
    selections = _ensure_strings_from_expressions(selections)
    if y is None:
        (waslist, [x]) = listify(x)
    else:
        (waslist, [x, y]) = listify(x, y)
        x = list(zip(x, y))
        limits = [limits]
    vwhats = _expand_limits(vwhat, len(x))
    logger.debug('x: %s', x)
    (limits, shape) = self.limits(x, limits, shape=shape)
    shape = shape[0]
    logger.debug('limits: %r', limits)
    labels = {}
    shape = _expand_shape(shape, 2)
    vshape = _expand_shape(shape, 2)
    if z is not None:
        match = re.match('(.*):(.*),(.*),(.*)', z)
        if match:
            groups = match.groups()
            import ast
            z_expression = groups[0]
            logger.debug('found groups: %r', list(groups))
            z_limits = [ast.literal_eval(groups[1]), ast.literal_eval(groups[2])]
            z_shape = ast.literal_eval(groups[3])
            x = [[z_expression] + list(k) for k in x]
            limits = np.array([[z_limits] + list(k) for k in limits])
            shape = (z_shape,) + shape
            vshape = (z_shape,) + vshape
            logger.debug('x = %r', x)
            values = np.linspace(z_limits[0], z_limits[1], num=z_shape + 1)
            labels['z'] = list(['%s <= %s < %s' % (v1, z_expression, v2) for (v1, v2) in zip(values[:-1], values[1:])])
        else:
            raise ValueError("Could not understand 'z' argument %r, expected something in form: 'column:-1,10:5'" % facet)
    else:
        z_shape = 1
    if z is None:
        total_grid = np.zeros((len(x), len(whats), len(selections), 1) + shape, dtype=float)
        total_vgrid = np.zeros((len(x), len(whats), len(selections), 1) + vshape, dtype=float)
    else:
        total_grid = np.zeros((len(x), len(whats), len(selections)) + shape, dtype=float)
        total_vgrid = np.zeros((len(x), len(whats), len(selections)) + vshape, dtype=float)
    logger.debug('shape of total grid: %r', total_grid.shape)
    axis = dict(plot=0, what=1, selection=2)
    xlimits = limits
    grid_axes = dict(x=-1, y=-2, z=-3, selection=-4, what=-5, subspace=-6)
    visual_axes = dict(x=-1, y=-2, layer=-3, fade=-4, column=-5, row=-6)
    visual_default = dict(x='x', y='y', layer='z', fade='selection', row='subspace', column='what')

    def invert(x):
        if False:
            return 10
        return dict(((v, k) for (k, v) in x.items()))
    free_visual_axes = list(visual_default.keys())
    logger.debug('1: %r %r', visual, free_visual_axes)
    for (visual_name, grid_name) in visual.items():
        if visual_name in free_visual_axes:
            free_visual_axes.remove(visual_name)
        else:
            raise ValueError('visual axes %s used multiple times' % visual_name)
    logger.debug('2: %r %r', visual, free_visual_axes)
    for (visual_name, grid_name) in visual_default.items():
        if visual_name in free_visual_axes and grid_name not in visual.values():
            free_visual_axes.remove(visual_name)
            visual[visual_name] = grid_name
    logger.debug('3: %r %r', visual, free_visual_axes)
    for (visual_name, grid_name) in visual_default.items():
        if visual_name not in free_visual_axes and grid_name not in visual.values():
            visual[free_visual_axes.pop(0)] = grid_name
    logger.debug('4: %r %r', visual, free_visual_axes)
    visual_reverse = invert(visual)
    (visual, visual_reverse) = (visual_reverse, visual)
    move = {}
    for (grid_name, visual_name) in visual.items():
        if visual_axes[visual_name] in visual.values():
            index = visual.values().find(visual_name)
            key = visual.keys()[index]
            raise ValueError('trying to map %s to %s while, it is already mapped by %s' % (grid_name, visual_name, key))
        move[grid_axes[grid_name]] = visual_axes[visual_name]
    fs = _expand(f, total_grid.shape[grid_axes[normalize_axis]])
    what_labels = []
    if grid is None:
        grid_of_grids = []
        for (i, (binby, limits)) in enumerate(zip(x, xlimits)):
            grid_of_grids.append([])
            for (j, what) in enumerate(whats):
                if isinstance(what, vaex.stat.Expression):
                    grid = what.calculate(self, binby=binby, shape=shape, limits=limits, selection=selections, delay=True)
                else:
                    what = what.strip()
                    index = what.index('(')
                    import re
                    groups = re.match('(.*)\\((.*)\\)', what).groups()
                    if groups and len(groups) == 2:
                        function = groups[0]
                        arguments = groups[1].strip()
                        if ',' in arguments:
                            arguments = arguments.split(',')
                        functions = ['mean', 'sum', 'std', 'var', 'correlation', 'covar', 'min', 'max', 'median_approx']
                        unit_expression = None
                        if function in ['mean', 'sum', 'std', 'min', 'max', 'median']:
                            unit_expression = arguments
                        if function in ['var']:
                            unit_expression = '(%s) * (%s)' % (arguments, arguments)
                        if function in ['covar']:
                            unit_expression = '(%s) * (%s)' % arguments
                        if unit_expression:
                            unit = self.unit(unit_expression)
                            if unit:
                                what_units = unit.to_string('latex_inline')
                        if function in functions:
                            grid = getattr(self, function)(arguments, binby=binby, limits=limits, shape=shape, selection=selections, delay=True)
                        elif function == 'count':
                            grid = self.count(arguments, binby, shape=shape, limits=limits, selection=selections, delay=True)
                        else:
                            raise ValueError("Could not understand method: %s, expected one of %r'" % (function, functions))
                    else:
                        raise ValueError("Could not understand 'what' argument %r, expected something in form: 'count(*)', 'mean(x)'" % what)
                if i == 0:
                    what_label = str(whats[j])
                    if what_units:
                        what_label += ' (%s)' % what_units
                    if fs[j]:
                        what_label = fs[j] + ' ' + what_label
                    what_labels.append(what_label)
                grid_of_grids[-1].append(grid)
        self.execute()
        for (i, (binby, limits)) in enumerate(zip(x, xlimits)):
            for (j, what) in enumerate(whats):
                grid = grid_of_grids[i][j].get()
                total_grid[i, j, :, :] = grid[:, None, ...]
        labels['what'] = what_labels
    else:
        dims_left = 6 - len(grid.shape)
        total_grid = np.broadcast_to(grid, (1,) * dims_left + grid.shape)

    def _selection_name(name):
        if False:
            for i in range(10):
                print('nop')
        if name in [None, False]:
            return 'selection: all'
        elif name in ['default', True]:
            return 'selection: default'
        else:
            return 'selection: %s' % name
    if selection_labels is None:
        labels['selection'] = list([_selection_name(k) for k in selections])
    else:
        labels['selection'] = selection_labels
    axes = [None] * len(move)
    for (key, value) in move.items():
        axes[value] = key
    visual_grid = np.transpose(total_grid, axes)
    logger.debug('grid shape: %r', total_grid.shape)
    logger.debug('visual: %r', visual.items())
    logger.debug('move: %r', move)
    logger.debug('visual grid shape: %r', visual_grid.shape)
    xexpressions = []
    yexpressions = []
    for (i, (binby, limits)) in enumerate(zip(x, xlimits)):
        xexpressions.append(binby[0])
        yexpressions.append(binby[1])
    if xlabel is None:
        xlabels = []
        ylabels = []
        for (i, (binby, limits)) in enumerate(zip(x, xlimits)):
            if z is not None:
                xlabels.append(self.label(binby[1]))
                ylabels.append(self.label(binby[2]))
            else:
                xlabels.append(self.label(binby[0]))
                ylabels.append(self.label(binby[1]))
    else:
        Nl = visual_grid.shape[visual_axes['row']]
        xlabels = _expand(xlabel, Nl)
        ylabels = _expand(ylabel, Nl)
    labels['x'] = xlabels
    labels['y'] = ylabels
    axes = []
    background_color = np.array(matplotlib.colors.colorConverter.to_rgb(background_color))
    import math
    facet_columns = None
    facets = visual_grid.shape[visual_axes['row']] * visual_grid.shape[visual_axes['column']]
    if visual_grid.shape[visual_axes['column']] == 1 and wrap:
        facet_columns = min(wrap_columns, visual_grid.shape[visual_axes['row']])
        wrapped = True
    elif visual_grid.shape[visual_axes['row']] == 1 and wrap:
        facet_columns = min(wrap_columns, visual_grid.shape[visual_axes['column']])
        wrapped = True
    else:
        wrapped = False
        facet_columns = visual_grid.shape[visual_axes['column']]
    facet_rows = int(math.ceil(facets / facet_columns))
    logger.debug('facet_rows: %r', facet_rows)
    logger.debug('facet_columns: %r', facet_columns)
    grid = visual_grid * 1.0
    fgrid = visual_grid * 1.0
    ngrid = visual_grid * 1.0
    vmins = _expand(vmin, visual_grid.shape[visual_axes[visual[normalize_axis]]], type=list)
    vmaxs = _expand(vmax, visual_grid.shape[visual_axes[visual[normalize_axis]]], type=list)
    visual_grid
    if smooth_pre:
        grid = vaex.grids.gf(grid, smooth_pre)
    if 1:
        axis = visual_axes[visual[normalize_axis]]
        for i in range(visual_grid.shape[axis]):
            item = [slice(None, None, None)] * len(visual_grid.shape)
            item[axis] = i
            item = tuple(item)
            f = _parse_f(fs[i])
            with np.errstate(divide='ignore', invalid='ignore'):
                fgrid.__setitem__(item, f(grid.__getitem__(item)))
            if vmins[i] is not None and vmaxs[i] is not None:
                nsubgrid = fgrid.__getitem__(item) * 1
                nsubgrid -= vmins[i]
                nsubgrid /= vmaxs[i] - vmins[i]
            else:
                (nsubgrid, vmin, vmax) = n(fgrid.__getitem__(item))
                vmins[i] = vmin
                vmaxs[i] = vmax
            ngrid.__setitem__(item, nsubgrid)
    if 0:
        grid = visual_grid[i]
        f = _parse_f(fs[i])
        fgrid = f(grid)
        finite_mask = np.isfinite(grid)
        finite_mask = np.any(finite_mask, axis=0)
        if vmin is not None and vmax is not None:
            ngrid = fgrid * 1
            ngrid -= vmin
            ngrid /= vmax - vmin
            ngrid = np.clip(ngrid, 0, 1)
        else:
            (ngrid, vmin, vmax) = n(fgrid)
    (rows, columns) = (int(math.ceil(facets / float(facet_columns))), facet_columns)
    colorbar_location = 'individual'
    if visual['what'] == 'row' and visual_grid.shape[1] == facet_columns:
        colorbar_location = 'per_row'
    if visual['what'] == 'column' and visual_grid.shape[0] == facet_rows:
        colorbar_location = 'per_column'
    logger.debug('rows: %r, columns: %r', rows, columns)
    import matplotlib.gridspec as gridspec
    column_scale = 1
    row_scale = 1
    row_offset = 0
    if facets > 1:
        if colorbar_location == 'per_row':
            column_scale = 4
            gs = gridspec.GridSpec(rows, columns * column_scale + 1)
        elif colorbar_location == 'per_column':
            row_offset = 1
            row_scale = 4
            gs = gridspec.GridSpec(rows * row_scale + 1, columns)
        else:
            gs = gridspec.GridSpec(rows, columns)
    facet_index = 0
    fs = _expand(f, len(whats))
    colormaps = _expand(colormap, len(whats))
    for i in range(visual_grid.shape[0]):
        for j in range(visual_grid.shape[1]):
            if colorbar and colorbar_location == 'per_column' and (i == 0):
                norm = matplotlib.colors.Normalize(vmins[j], vmaxs[j])
                sm = matplotlib.cm.ScalarMappable(norm, colormaps[j])
                sm.set_array(1)
                if facets > 1:
                    ax = plt.subplot(gs[0, j])
                    colorbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
                else:
                    colorbar = fig.colorbar(sm, ax=plt.gca())
                if 'what' in labels:
                    label = labels['what'][j]
                    if facets > 1:
                        colorbar.ax.set_title(label)
                    else:
                        colorbar.ax.set_ylabel(colorbar_label or label)
            if colorbar and colorbar_location == 'per_row' and (j == 0):
                norm = matplotlib.colors.Normalize(vmins[i], vmaxs[i])
                sm = matplotlib.cm.ScalarMappable(norm, colormaps[i])
                sm.set_array(1)
                if facets > 1:
                    ax = plt.subplot(gs[i, -1])
                    colorbar = fig.colorbar(sm, cax=ax)
                else:
                    colorbar = fig.colorbar(sm, ax=plt.gca())
                label = labels['what'][i]
                colorbar.ax.set_ylabel(colorbar_label or label)
            rgrid = ngrid[i, j] * 1.0
            for k in range(rgrid.shape[0]):
                for l in range(rgrid.shape[0]):
                    if smooth_post is not None:
                        rgrid[k, l] = vaex.grids.gf(rgrid, smooth_post)
            if visual['what'] == 'column':
                what_index = j
            elif visual['what'] == 'row':
                what_index = i
            else:
                what_index = 0
            if visual[normalize_axis] == 'column':
                normalize_index = j
            elif visual[normalize_axis] == 'row':
                normalize_index = i
            else:
                normalize_index = 0
            for r in reduce:
                r = _parse_reduction(r, colormaps[what_index], [])
                rgrid = r(rgrid)
            row = facet_index // facet_columns
            column = facet_index % facet_columns
            if colorbar and colorbar_location == 'individual':
                norm = matplotlib.colors.Normalize(vmins[normalize_index], vmaxs[normalize_index])
                sm = matplotlib.cm.ScalarMappable(norm, colormaps[what_index])
                sm.set_array(1)
                if facets > 1:
                    ax = plt.subplot(gs[row, column])
                    colorbar = fig.colorbar(sm, ax=ax)
                else:
                    colorbar = fig.colorbar(sm, ax=plt.gca())
                label = labels['what'][what_index]
                colorbar.ax.set_ylabel(colorbar_label or label)
            if facets > 1:
                ax = plt.subplot(gs[row_offset + row * row_scale:row_offset + (row + 1) * row_scale, column * column_scale:(column + 1) * column_scale])
            else:
                ax = plt.gca()
            axes.append(ax)
            logger.debug('rgrid: %r', rgrid.shape)
            plot_rgrid = rgrid
            assert plot_rgrid.shape[1] == 1, 'no layers supported yet'
            plot_rgrid = plot_rgrid[:, 0]
            if plot_rgrid.shape[0] > 1:
                plot_rgrid = vaex.image.fade(plot_rgrid[::-1])
            else:
                plot_rgrid = plot_rgrid[0]
            extend = None
            if visual['subspace'] == 'row':
                subplot_index = i
            elif visual['subspace'] == 'column':
                subplot_index = j
            else:
                subplot_index = 0
            extend = np.array(xlimits[subplot_index][-2:]).flatten()
            logger.debug('plot rgrid: %r', plot_rgrid.shape)
            plot_rgrid = np.transpose(plot_rgrid, (1, 0, 2))
            im = ax.imshow(plot_rgrid, extent=extend.tolist(), origin='lower', aspect=aspect, interpolation=interpolation)

            def label(index, label, expression):
                if False:
                    for i in range(10):
                        print('nop')
                if label and _issequence(label):
                    return label[i]
                else:
                    return self.label(expression)
            if visual_reverse['x'] == 'x':
                labelsx = labels['x']
                plt.xlabel(labelsx[subplot_index])
            if visual_reverse['x'] == 'x':
                labelsy = labels['y']
                plt.ylabel(labelsy[subplot_index])
            if visual['z'] in ['row']:
                labelsz = labels['z']
                ax.set_title(labelsz[i])
            if visual['z'] in ['column']:
                labelsz = labels['z']
                ax.set_title(labelsz[j])
            max_labels = 10
            xexpression = xexpressions[subplot_index]
            if self.iscategory(xexpression):
                labels = self.category_labels(xexpression)
                step = max(len(labels) // max_labels, 1)
                plt.xticks(np.arange(len(labels))[::step], labels[::step], size='small')
            yexpression = yexpressions[subplot_index]
            if self.iscategory(yexpression):
                labels = self.category_labels(yexpression)
                step = max(len(labels) // max_labels, 1)
                plt.yticks(np.arange(len(labels))[::step], labels[::step], size='small')
            facet_index += 1
    if title:
        fig.suptitle(title, fontsize='x-large')
    if tight_layout:
        if title:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            plt.tight_layout()
    if hardcopy:
        plt.savefig(hardcopy)
    if show:
        plt.show()
    if return_extra:
        return (im, grid, fgrid, ngrid, rgrid)
    else:
        return im

@patch
def healpix_plot(self, *args, **kwargs):
    if False:
        while True:
            i = 10
    warnings.warn('`healpix_plot` is deprecated and it will be removed in version 5.x. Please `df.viz.healpix_heatmap` instead.')
    self.viz.healpix_heatmap(*args, **kwargs)

@viz_method
def healpix_heatmap(self, healpix_expression='source_id/34359738368', healpix_max_level=12, healpix_level=8, what='count(*)', selection=None, grid=None, healpix_input='equatorial', healpix_output='galactic', f=None, colormap='afmhot', grid_limits=None, image_size=800, nest=True, figsize=None, interactive=False, title='', smooth=None, show=False, colorbar=True, rotation=(0, 0, 0), **kwargs):
    if False:
        print('Hello World!')
    'Viz data in 2d using a healpix column.\n\n    :param healpix_expression: {healpix_max_level}\n    :param healpix_max_level: {healpix_max_level}\n    :param healpix_level: {healpix_level}\n    :param what: {what}\n    :param selection: {selection}\n    :param grid: {grid}\n    :param healpix_input: Specificy if the healpix index is in "equatorial", "galactic" or "ecliptic".\n    :param healpix_output: Plot in "equatorial", "galactic" or "ecliptic".\n    :param f: function to apply to the data\n    :param colormap: matplotlib colormap\n    :param grid_limits: Optional sequence [minvalue, maxvalue] that determine the min and max value that map to the colormap (values below and above these are clipped to the the min/max). (default is [min(f(grid)), max(f(grid)))\n    :param image_size: size for the image that healpy uses for rendering\n    :param nest: If the healpix data is in nested (True) or ring (False)\n    :param figsize: If given, modify the matplotlib figure size. Example (14,9)\n    :param interactive: (Experimental, uses healpy.mollzoom is True)\n    :param title: Title of figure\n    :param smooth: apply gaussian smoothing, in degrees\n    :param show: Call matplotlib\'s show (True) or not (False, defaut)\n    :param rotation: Rotatate the plot, in format (lon, lat, psi) such that (lon, lat) is the center, and rotate on the screen by angle psi. All angles are degrees.\n    :return:\n    '
    import healpy as hp
    import matplotlib.pyplot as plt
    if grid is None:
        reduce_level = healpix_max_level - healpix_level
        NSIDE = 2 ** healpix_level
        nmax = hp.nside2npix(NSIDE)
        scaling = 4 ** reduce_level
        epsilon = 1.0 / scaling / 2
        grid = self._stat(what=what, binby='%s/%s' % (healpix_expression, scaling), limits=[-epsilon, nmax - epsilon], shape=nmax, selection=selection)
    if grid_limits:
        (grid_min, grid_max) = grid_limits
    else:
        grid_min = grid_max = None
    f_org = f
    f = _parse_f(f)
    if smooth:
        if nest:
            grid = hp.reorder(grid, inp='NEST', out='RING')
            nest = False
        grid = hp.smoothing(grid, sigma=np.radians(smooth))
    fgrid = f(grid)
    coord_map = dict(equatorial='C', galactic='G', ecliptic='E')
    fig = plt.gcf()
    if figsize is not None:
        fig.set_size_inches(*figsize)
    what_label = what
    if f_org:
        what_label = f_org + ' ' + what_label
    f = hp.mollzoom if interactive else hp.mollview
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        coord = (coord_map[healpix_input], coord_map[healpix_output])
        if coord_map[healpix_input] == coord_map[healpix_output]:
            coord = None
        f(fgrid, unit=what_label, rot=rotation, nest=nest, title=title, coord=coord, cmap=colormap, hold=True, xsize=image_size, min=grid_min, max=grid_max, cbar=colorbar, **kwargs)
    if show:
        plt.show()