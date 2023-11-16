from __future__ import annotations
import random
import warnings
from bisect import bisect_left
from itertools import cycle
from operator import add, itemgetter
from tlz import accumulate, groupby, pluck, unique
from dask.core import istask
from dask.utils import apply, funcname, import_required

def BOKEH_VERSION():
    if False:
        print('Hello World!')
    import bokeh
    from packaging.version import parse as parse_version
    return parse_version(bokeh.__version__)
_BOKEH_MISSING_MSG = 'Diagnostics plots require `bokeh` to be installed'

def unquote(expr):
    if False:
        i = 10
        return i + 15
    if istask(expr):
        if expr[0] in (tuple, list, set):
            return expr[0](map(unquote, expr[1]))
        elif expr[0] == dict and isinstance(expr[1], list) and isinstance(expr[1][0], list):
            return dict(map(unquote, expr[1]))
    return expr

def pprint_task(task, keys, label_size=60):
    if False:
        i = 10
        return i + 15
    "Return a nicely formatted string for a task.\n\n    Parameters\n    ----------\n    task:\n        Value within dask graph to render as text\n    keys: iterable\n        List of keys within dask graph\n    label_size: int (optional)\n        Maximum size of output label, defaults to 60\n\n    Examples\n    --------\n    >>> from operator import add, mul\n    >>> dsk = {'a': 1,\n    ...        'b': 2,\n    ...        'c': (add, 'a', 'b'),\n    ...        'd': (add, (mul, 'a', 'b'), 'c'),\n    ...        'e': (sum, ['a', 'b', 5]),\n    ...        'f': (add,),\n    ...        'g': []}\n\n    >>> pprint_task(dsk['c'], dsk)\n    'add(_, _)'\n    >>> pprint_task(dsk['d'], dsk)\n    'add(mul(_, _), _)'\n    >>> pprint_task(dsk['e'], dsk)\n    'sum([_, _, *])'\n    >>> pprint_task(dsk['f'], dsk)\n    'add()'\n    >>> pprint_task(dsk['g'], dsk)\n    '[]'\n    "
    if istask(task):
        func = task[0]
        if func is apply:
            head = funcname(task[1])
            tail = ')'
            args = unquote(task[2]) if len(task) > 2 else ()
            kwargs = unquote(task[3]) if len(task) > 3 else {}
        else:
            if hasattr(func, 'funcs'):
                head = '('.join((funcname(f) for f in func.funcs))
                tail = ')' * len(func.funcs)
            else:
                head = funcname(task[0])
                tail = ')'
            args = task[1:]
            kwargs = {}
        if args or kwargs:
            label_size2 = int((label_size - len(head) - len(tail)) // (len(args) + len(kwargs)))
            pprint = lambda t: pprint_task(t, keys, label_size2)
        if args:
            if label_size2 > 5:
                args = ', '.join((pprint(t) for t in args))
            else:
                args = '...'
        else:
            args = ''
        if kwargs:
            if label_size2 > 5:
                kwargs = ', ' + ', '.join((f'{k}={pprint(v)}' for (k, v) in sorted(kwargs.items())))
            else:
                kwargs = ', ...'
        else:
            kwargs = ''
        return f'{head}({args}{kwargs}{tail}'
    elif isinstance(task, list):
        if not task:
            return '[]'
        elif len(task) > 3:
            result = pprint_task(task[:3], keys, label_size)
            return result[:-1] + ', ...]'
        else:
            label_size2 = int((label_size - 2 - 2 * len(task)) // len(task))
            args = ', '.join((pprint_task(t, keys, label_size2) for t in task))
            return f'[{args}]'
    else:
        try:
            if task in keys:
                return '_'
            else:
                return '*'
        except TypeError:
            return '*'

def get_colors(palette, funcs):
    if False:
        return 10
    'Get a dict mapping funcs to colors from palette.\n\n    Parameters\n    ----------\n    palette : string\n        Name of the bokeh palette to use, must be a member of\n        bokeh.palettes.all_palettes.\n    funcs : iterable\n        Iterable of function names\n    '
    palettes = import_required('bokeh.palettes', _BOKEH_MISSING_MSG)
    unique_funcs = sorted(unique(funcs))
    n_funcs = len(unique_funcs)
    palette_lookup = palettes.all_palettes[palette]
    keys = list(sorted(palette_lookup.keys()))
    index = keys[min(bisect_left(keys, n_funcs), len(keys) - 1)]
    palette = palette_lookup[index]
    palette = list(unique(palette))
    if len(palette) > n_funcs:
        random.Random(42).shuffle(palette)
    color_lookup = dict(zip(unique_funcs, cycle(palette)))
    return [color_lookup[n] for n in funcs]

def visualize(profilers, filename='profile.html', show=True, save=None, mode=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Visualize the results of profiling in a bokeh plot.\n\n    If multiple profilers are passed in, the plots are stacked vertically.\n\n    Parameters\n    ----------\n    profilers : profiler or list\n        Profiler or list of profilers.\n    filename : string, optional\n        Name of the plot output file.\n    show : boolean, optional\n        If True (default), the plot is opened in a browser.\n    save : boolean, optional\n        If True (default when not in notebook), the plot is saved to disk.\n    mode : str, optional\n        Mode passed to bokeh.output_file()\n    **kwargs\n        Other keyword arguments, passed to bokeh.figure. These will override\n        all defaults set by visualize.\n\n    Returns\n    -------\n    The completed bokeh plot object.\n    '
    bp = import_required('bokeh.plotting', _BOKEH_MISSING_MSG)
    from bokeh.io import state
    if 'file_path' in kwargs:
        warnings.warn('The file_path keyword argument is deprecated and will be removed in a future release. Please use filename instead.', category=FutureWarning, stacklevel=2)
        filename = kwargs.pop('file_path')
    if save is None:
        save = not state.curstate().notebook
    if not isinstance(profilers, list):
        profilers = [profilers]
    figs = [prof._plot(**kwargs) for prof in profilers]
    if len(figs) == 1:
        p = figs[0]
    else:
        top = figs[0]
        for f in figs[1:]:
            f.x_range = top.x_range
            f.title = None
            f.min_border_top = 20
            if BOKEH_VERSION().major < 3:
                f.plot_height -= 30
            else:
                f.height -= 30
        for f in figs[:-1]:
            f.xaxis.axis_label = None
            f.min_border_bottom = 20
            if BOKEH_VERSION().major < 3:
                f.plot_height -= 30
            else:
                f.height -= 30
        for f in figs:
            f.min_border_left = 75
            f.min_border_right = 75
        p = bp.gridplot([[f] for f in figs])
    if show:
        bp.show(p)
    if save:
        bp.output_file(filename, mode=mode)
        bp.save(p)
    return p

def plot_tasks(results, dsk, start_time, end_time, palette='Viridis', label_size=60, **kwargs):
    if False:
        i = 10
        return i + 15
    'Visualize the results of profiling in a bokeh plot.\n\n    Parameters\n    ----------\n    results : sequence\n        Output of Profiler.results\n    dsk : dict\n        The dask graph being profiled.\n    start_time : float\n        Start time of the profile in seconds\n    end_time : float\n        End time of the profile in seconds\n    palette : string, optional\n        Name of the bokeh palette to use, must be a member of\n        bokeh.palettes.all_palettes.\n    label_size: int (optional)\n        Maximum size of output labels in plot, defaults to 60\n    **kwargs\n        Other keyword arguments, passed to bokeh.figure. These will override\n        all defaults set by visualize.\n\n    Returns\n    -------\n    The completed bokeh plot object.\n    '
    bp = import_required('bokeh.plotting', _BOKEH_MISSING_MSG)
    from bokeh.models import HoverTool
    defaults = dict(title='Profile Results', tools='hover,save,reset,xwheel_zoom,xpan', toolbar_location='above', width=800, height=300)
    if 'plot_width' in kwargs:
        kwargs['width'] = kwargs.pop('plot_width')
    if 'plot_height' in kwargs:
        kwargs['height'] = kwargs.pop('plot_height')
    defaults.update(**kwargs)
    if results:
        (keys, tasks, starts, ends, ids) = zip(*results)
        id_group = groupby(itemgetter(4), results)
        timings = {k: [i.end_time - i.start_time for i in v] for (k, v) in id_group.items()}
        id_lk = {t[0]: n for (n, t) in enumerate(sorted(timings.items(), key=itemgetter(1), reverse=True))}
        p = bp.figure(y_range=[str(i) for i in range(len(id_lk))], x_range=[0, end_time - start_time], **defaults)
        data = {}
        data['width'] = width = [e - s for (s, e) in zip(starts, ends)]
        data['x'] = [w / 2 + s - start_time for (w, s) in zip(width, starts)]
        data['y'] = [id_lk[i] + 1 for i in ids]
        data['function'] = funcs = [pprint_task(i, dsk, label_size) for i in tasks]
        data['color'] = get_colors(palette, funcs)
        data['key'] = [str(i) for i in keys]
        source = bp.ColumnDataSource(data=data)
        p.rect(source=source, x='x', y='y', height=1, width='width', color='color', line_color='gray')
    else:
        p = bp.figure(y_range=[str(i) for i in range(8)], x_range=[0, 10], **defaults)
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.yaxis.axis_label = 'Worker ID'
    p.xaxis.axis_label = 'Time (s)'
    hover = p.select(HoverTool)
    hover.tooltips = '\n    <div>\n        <span style="font-size: 14px; font-weight: bold;">Key:</span>&nbsp;\n        <span style="font-size: 10px; font-family: Monaco, monospace;">@key</span>\n    </div>\n    <div>\n        <span style="font-size: 14px; font-weight: bold;">Task:</span>&nbsp;\n        <span style="font-size: 10px; font-family: Monaco, monospace;">@function</span>\n    </div>\n    '
    hover.point_policy = 'follow_mouse'
    return p

def plot_resources(results, start_time, end_time, palette='Viridis', **kwargs):
    if False:
        i = 10
        return i + 15
    'Plot resource usage in a bokeh plot.\n\n    Parameters\n    ----------\n    results : sequence\n        Output of ResourceProfiler.results\n    start_time : float\n        Start time of the profile in seconds\n    end_time : float\n        End time of the profile in seconds\n    palette : string, optional\n        Name of the bokeh palette to use, must be a member of\n        bokeh.palettes.all_palettes.\n    **kwargs\n        Other keyword arguments, passed to bokeh.figure. These will override\n        all defaults set by plot_resources.\n\n    Returns\n    -------\n    The completed bokeh plot object.\n    '
    bp = import_required('bokeh.plotting', _BOKEH_MISSING_MSG)
    from bokeh import palettes
    from bokeh.models import LinearAxis, Range1d
    defaults = dict(title='Profile Results', tools='save,reset,xwheel_zoom,xpan', toolbar_location='above', width=800, height=300)
    if 'plot_width' in kwargs:
        kwargs['width'] = kwargs.pop('plot_width')
        if BOKEH_VERSION().major >= 3:
            warnings.warn('Use width instead of plot_width with Bokeh >= 3')
    if 'plot_height' in kwargs:
        kwargs['height'] = kwargs.pop('plot_height')
        if BOKEH_VERSION().major >= 3:
            warnings.warn('Use height instead of plot_height with Bokeh >= 3')
    if 'label_size' in kwargs:
        kwargs.pop('label_size')
    defaults.update(**kwargs)
    if results:
        (t, mem, cpu) = zip(*results)
        left = start_time
        right = end_time
        t = [i - left for i in t]
        p = bp.figure(y_range=fix_bounds(0, max(cpu), 100), x_range=fix_bounds(0, right - left, 1), **defaults)
    else:
        t = mem = cpu = []
        p = bp.figure(y_range=(0, 100), x_range=(0, 1), **defaults)
    colors = palettes.all_palettes[palette][6]
    p.line(t, cpu, color=colors[0], line_width=4, legend_label='% CPU')
    p.yaxis.axis_label = '% CPU'
    p.extra_y_ranges = {'memory': Range1d(*fix_bounds(min(mem) if mem else 0, max(mem) if mem else 100, 100))}
    p.line(t, mem, color=colors[2], y_range_name='memory', line_width=4, legend_label='Memory')
    p.add_layout(LinearAxis(y_range_name='memory', axis_label='Memory (MB)'), 'right')
    p.xaxis.axis_label = 'Time (s)'
    return p

def fix_bounds(start, end, min_span):
    if False:
        for i in range(10):
            print('nop')
    'Adjust end point to ensure span of at least `min_span`'
    return (start, max(end, start + min_span))

def plot_cache(results, dsk, start_time, end_time, metric_name, palette='Viridis', label_size=60, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Visualize the results of profiling in a bokeh plot.\n\n    Parameters\n    ----------\n    results : sequence\n        Output of CacheProfiler.results\n    dsk : dict\n        The dask graph being profiled.\n    start_time : float\n        Start time of the profile in seconds\n    end_time : float\n        End time of the profile in seconds\n    metric_name : string\n        Metric used to measure cache size\n    palette : string, optional\n        Name of the bokeh palette to use, must be a member of\n        bokeh.palettes.all_palettes.\n    label_size: int (optional)\n        Maximum size of output labels in plot, defaults to 60\n    **kwargs\n        Other keyword arguments, passed to bokeh.figure. These will override\n        all defaults set by visualize.\n\n    Returns\n    -------\n    The completed bokeh plot object.\n    '
    bp = import_required('bokeh.plotting', _BOKEH_MISSING_MSG)
    from bokeh.models import HoverTool
    defaults = dict(title='Profile Results', tools='hover,save,reset,wheel_zoom,xpan', toolbar_location='above', width=800, height=300)
    if 'plot_width' in kwargs:
        kwargs['width'] = kwargs.pop('plot_width')
        if BOKEH_VERSION().major >= 3:
            warnings.warn('Use width instead of plot_width with Bokeh >= 3')
    if 'plot_height' in kwargs:
        kwargs['height'] = kwargs.pop('plot_height')
        if BOKEH_VERSION().major >= 3:
            warnings.warn('Use height instead of plot_height with Bokeh >= 3')
    defaults.update(**kwargs)
    if results:
        (starts, ends) = list(zip(*results))[3:]
        tics = sorted(unique(starts + ends))
        groups = groupby(lambda d: pprint_task(d[1], dsk, label_size), results)
        data = {}
        for (k, vals) in groups.items():
            cnts = dict.fromkeys(tics, 0)
            for v in vals:
                cnts[v.cache_time] += v.metric
                cnts[v.free_time] -= v.metric
            data[k] = [0] + list(accumulate(add, pluck(1, sorted(cnts.items()))))
        tics = [0] + [i - start_time for i in tics]
        p = bp.figure(x_range=[0, end_time - start_time], **defaults)
        for ((key, val), color) in zip(data.items(), get_colors(palette, data.keys())):
            p.line('x', 'y', line_color=color, line_width=3, source=bp.ColumnDataSource({'x': tics, 'y': val, 'label': [key for i in val]}))
    else:
        p = bp.figure(y_range=[0, 10], x_range=[0, 10], **defaults)
    p.yaxis.axis_label = f'Cache Size ({metric_name})'
    p.xaxis.axis_label = 'Time (s)'
    hover = p.select(HoverTool)
    hover.tooltips = '\n    <div>\n        <span style="font-size: 14px; font-weight: bold;">Task:</span>&nbsp;\n        <span style="font-size: 10px; font-family: Monaco, monospace;">@label</span>\n    </div>\n    '
    return p