from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots
pd = optional_imports.get_module('pandas')
DIAG_CHOICES = ['scatter', 'histogram', 'box']
VALID_COLORMAP_TYPES = ['cat', 'seq']

def endpts_to_intervals(endpts):
    if False:
        while True:
            i = 10
    '\n    Returns a list of intervals for categorical colormaps\n\n    Accepts a list or tuple of sequentially increasing numbers and returns\n    a list representation of the mathematical intervals with these numbers\n    as endpoints. For example, [1, 6] returns [[-inf, 1], [1, 6], [6, inf]]\n\n    :raises: (PlotlyError) If input is not a list or tuple\n    :raises: (PlotlyError) If the input contains a string\n    :raises: (PlotlyError) If any number does not increase after the\n        previous one in the sequence\n    '
    length = len(endpts)
    if not (isinstance(endpts, tuple) or isinstance(endpts, list)):
        raise exceptions.PlotlyError('The intervals_endpts argument must be a list or tuple of a sequence of increasing numbers.')
    for item in endpts:
        if isinstance(item, str):
            raise exceptions.PlotlyError('The intervals_endpts argument must be a list or tuple of a sequence of increasing numbers.')
    for k in range(length - 1):
        if endpts[k] >= endpts[k + 1]:
            raise exceptions.PlotlyError('The intervals_endpts argument must be a list or tuple of a sequence of increasing numbers.')
    else:
        intervals = []
        intervals.append([float('-inf'), endpts[0]])
        for k in range(length - 1):
            interval = []
            interval.append(endpts[k])
            interval.append(endpts[k + 1])
            intervals.append(interval)
        intervals.append([endpts[length - 1], float('inf')])
        return intervals

def hide_tick_labels_from_box_subplots(fig):
    if False:
        while True:
            i = 10
    '\n    Hides tick labels for box plots in scatterplotmatrix subplots.\n    '
    boxplot_xaxes = []
    for trace in fig['data']:
        if trace['type'] == 'box':
            boxplot_xaxes.append('xaxis{}'.format(trace['xaxis'][1:]))
    for xaxis in boxplot_xaxes:
        fig['layout'][xaxis]['showticklabels'] = False

def validate_scatterplotmatrix(df, index, diag, colormap_type, **kwargs):
    if False:
        print('Hello World!')
    "\n    Validates basic inputs for FigureFactory.create_scatterplotmatrix()\n\n    :raises: (PlotlyError) If pandas is not imported\n    :raises: (PlotlyError) If pandas dataframe is not inputted\n    :raises: (PlotlyError) If pandas dataframe has <= 1 columns\n    :raises: (PlotlyError) If diagonal plot choice (diag) is not one of\n        the viable options\n    :raises: (PlotlyError) If colormap_type is not a valid choice\n    :raises: (PlotlyError) If kwargs contains 'size', 'color' or\n        'colorscale'\n    "
    if not pd:
        raise ImportError('FigureFactory.scatterplotmatrix requires a pandas DataFrame.')
    if not isinstance(df, pd.core.frame.DataFrame):
        raise exceptions.PlotlyError('Dataframe not inputed. Please use a pandas dataframe to produce a scatterplot matrix.')
    if len(df.columns) <= 1:
        raise exceptions.PlotlyError('Dataframe has only one column. To use the scatterplot matrix, use at least 2 columns.')
    if diag not in DIAG_CHOICES:
        raise exceptions.PlotlyError('Make sure diag is set to one of {}'.format(DIAG_CHOICES))
    if colormap_type not in VALID_COLORMAP_TYPES:
        raise exceptions.PlotlyError("Must choose a valid colormap type. Either 'cat' or 'seq' for a categorical and sequential colormap respectively.")
    if 'marker' in kwargs:
        FORBIDDEN_PARAMS = ['size', 'color', 'colorscale']
        if any((param in kwargs['marker'] for param in FORBIDDEN_PARAMS)):
            raise exceptions.PlotlyError("Your kwargs dictionary cannot include the 'size', 'color' or 'colorscale' key words inside the marker dict since 'size' is already an argument of the scatterplot matrix function and both 'color' and 'colorscale are set internally.")

def scatterplot(dataframe, headers, diag, size, height, width, title, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Refer to FigureFactory.create_scatterplotmatrix() for docstring\n\n    Returns fig for scatterplotmatrix without index\n\n    '
    dim = len(dataframe)
    fig = make_subplots(rows=dim, cols=dim, print_grid=False)
    trace_list = []
    for listy in dataframe:
        for listx in dataframe:
            if listx == listy and diag == 'histogram':
                trace = graph_objs.Histogram(x=listx, showlegend=False)
            elif listx == listy and diag == 'box':
                trace = graph_objs.Box(y=listx, name=None, showlegend=False)
            elif 'marker' in kwargs:
                kwargs['marker']['size'] = size
                trace = graph_objs.Scatter(x=listx, y=listy, mode='markers', showlegend=False, **kwargs)
                trace_list.append(trace)
            else:
                trace = graph_objs.Scatter(x=listx, y=listy, mode='markers', marker=dict(size=size), showlegend=False, **kwargs)
            trace_list.append(trace)
    trace_index = 0
    indices = range(1, dim + 1)
    for y_index in indices:
        for x_index in indices:
            fig.append_trace(trace_list[trace_index], y_index, x_index)
            trace_index += 1
    for j in range(dim):
        xaxis_key = 'xaxis{}'.format(dim * dim - dim + 1 + j)
        fig['layout'][xaxis_key].update(title=headers[j])
    for j in range(dim):
        yaxis_key = 'yaxis{}'.format(1 + dim * j)
        fig['layout'][yaxis_key].update(title=headers[j])
    fig['layout'].update(height=height, width=width, title=title, showlegend=True)
    hide_tick_labels_from_box_subplots(fig)
    return fig

def scatterplot_dict(dataframe, headers, diag, size, height, width, title, index, index_vals, endpts, colormap, colormap_type, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Refer to FigureFactory.create_scatterplotmatrix() for docstring\n\n    Returns fig for scatterplotmatrix with both index and colormap picked.\n    Used if colormap is a dictionary with index values as keys pointing to\n    colors. Forces colormap_type to behave categorically because it would\n    not make sense colors are assigned to each index value and thus\n    implies that a categorical approach should be taken\n\n    '
    theme = colormap
    dim = len(dataframe)
    fig = make_subplots(rows=dim, cols=dim, print_grid=False)
    trace_list = []
    legend_param = 0
    for listy in dataframe:
        for listx in dataframe:
            unique_index_vals = {}
            for name in index_vals:
                if name not in unique_index_vals:
                    unique_index_vals[name] = []
            for name in sorted(unique_index_vals.keys()):
                new_listx = []
                new_listy = []
                for j in range(len(index_vals)):
                    if index_vals[j] == name:
                        new_listx.append(listx[j])
                        new_listy.append(listy[j])
                if legend_param == 1:
                    if listx == listy and diag == 'histogram':
                        trace = graph_objs.Histogram(x=new_listx, marker=dict(color=theme[name]), showlegend=True)
                    elif listx == listy and diag == 'box':
                        trace = graph_objs.Box(y=new_listx, name=None, marker=dict(color=theme[name]), showlegend=True)
                    elif 'marker' in kwargs:
                        kwargs['marker']['size'] = size
                        kwargs['marker']['color'] = theme[name]
                        trace = graph_objs.Scatter(x=new_listx, y=new_listy, mode='markers', name=name, showlegend=True, **kwargs)
                    else:
                        trace = graph_objs.Scatter(x=new_listx, y=new_listy, mode='markers', name=name, marker=dict(size=size, color=theme[name]), showlegend=True, **kwargs)
                elif listx == listy and diag == 'histogram':
                    trace = graph_objs.Histogram(x=new_listx, marker=dict(color=theme[name]), showlegend=False)
                elif listx == listy and diag == 'box':
                    trace = graph_objs.Box(y=new_listx, name=None, marker=dict(color=theme[name]), showlegend=False)
                elif 'marker' in kwargs:
                    kwargs['marker']['size'] = size
                    kwargs['marker']['color'] = theme[name]
                    trace = graph_objs.Scatter(x=new_listx, y=new_listy, mode='markers', name=name, showlegend=False, **kwargs)
                else:
                    trace = graph_objs.Scatter(x=new_listx, y=new_listy, mode='markers', name=name, marker=dict(size=size, color=theme[name]), showlegend=False, **kwargs)
                unique_index_vals[name] = trace
            trace_list.append(unique_index_vals)
            legend_param += 1
    trace_index = 0
    indices = range(1, dim + 1)
    for y_index in indices:
        for x_index in indices:
            for name in sorted(trace_list[trace_index].keys()):
                fig.append_trace(trace_list[trace_index][name], y_index, x_index)
            trace_index += 1
    for j in range(dim):
        xaxis_key = 'xaxis{}'.format(dim * dim - dim + 1 + j)
        fig['layout'][xaxis_key].update(title=headers[j])
    for j in range(dim):
        yaxis_key = 'yaxis{}'.format(1 + dim * j)
        fig['layout'][yaxis_key].update(title=headers[j])
    hide_tick_labels_from_box_subplots(fig)
    if diag == 'histogram':
        fig['layout'].update(height=height, width=width, title=title, showlegend=True, barmode='stack')
        return fig
    else:
        fig['layout'].update(height=height, width=width, title=title, showlegend=True)
        return fig

def scatterplot_theme(dataframe, headers, diag, size, height, width, title, index, index_vals, endpts, colormap, colormap_type, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Refer to FigureFactory.create_scatterplotmatrix() for docstring\n\n    Returns fig for scatterplotmatrix with both index and colormap picked\n\n    '
    if isinstance(index_vals[0], str):
        unique_index_vals = []
        for name in index_vals:
            if name not in unique_index_vals:
                unique_index_vals.append(name)
        n_colors_len = len(unique_index_vals)
        if colormap_type == 'seq':
            foo = clrs.color_parser(colormap, clrs.unlabel_rgb)
            foo = clrs.n_colors(foo[0], foo[1], n_colors_len)
            theme = clrs.color_parser(foo, clrs.label_rgb)
        if colormap_type == 'cat':
            theme = colormap
        dim = len(dataframe)
        fig = make_subplots(rows=dim, cols=dim, print_grid=False)
        trace_list = []
        legend_param = 0
        for listy in dataframe:
            for listx in dataframe:
                unique_index_vals = {}
                for name in index_vals:
                    if name not in unique_index_vals:
                        unique_index_vals[name] = []
                c_indx = 0
                for name in sorted(unique_index_vals.keys()):
                    new_listx = []
                    new_listy = []
                    for j in range(len(index_vals)):
                        if index_vals[j] == name:
                            new_listx.append(listx[j])
                            new_listy.append(listy[j])
                    if legend_param == 1:
                        if listx == listy and diag == 'histogram':
                            trace = graph_objs.Histogram(x=new_listx, marker=dict(color=theme[c_indx]), showlegend=True)
                        elif listx == listy and diag == 'box':
                            trace = graph_objs.Box(y=new_listx, name=None, marker=dict(color=theme[c_indx]), showlegend=True)
                        elif 'marker' in kwargs:
                            kwargs['marker']['size'] = size
                            kwargs['marker']['color'] = theme[c_indx]
                            trace = graph_objs.Scatter(x=new_listx, y=new_listy, mode='markers', name=name, showlegend=True, **kwargs)
                        else:
                            trace = graph_objs.Scatter(x=new_listx, y=new_listy, mode='markers', name=name, marker=dict(size=size, color=theme[c_indx]), showlegend=True, **kwargs)
                    elif listx == listy and diag == 'histogram':
                        trace = graph_objs.Histogram(x=new_listx, marker=dict(color=theme[c_indx]), showlegend=False)
                    elif listx == listy and diag == 'box':
                        trace = graph_objs.Box(y=new_listx, name=None, marker=dict(color=theme[c_indx]), showlegend=False)
                    elif 'marker' in kwargs:
                        kwargs['marker']['size'] = size
                        kwargs['marker']['color'] = theme[c_indx]
                        trace = graph_objs.Scatter(x=new_listx, y=new_listy, mode='markers', name=name, showlegend=False, **kwargs)
                    else:
                        trace = graph_objs.Scatter(x=new_listx, y=new_listy, mode='markers', name=name, marker=dict(size=size, color=theme[c_indx]), showlegend=False, **kwargs)
                    unique_index_vals[name] = trace
                    if c_indx >= len(theme) - 1:
                        c_indx = -1
                    c_indx += 1
                trace_list.append(unique_index_vals)
                legend_param += 1
        trace_index = 0
        indices = range(1, dim + 1)
        for y_index in indices:
            for x_index in indices:
                for name in sorted(trace_list[trace_index].keys()):
                    fig.append_trace(trace_list[trace_index][name], y_index, x_index)
                trace_index += 1
        for j in range(dim):
            xaxis_key = 'xaxis{}'.format(dim * dim - dim + 1 + j)
            fig['layout'][xaxis_key].update(title=headers[j])
        for j in range(dim):
            yaxis_key = 'yaxis{}'.format(1 + dim * j)
            fig['layout'][yaxis_key].update(title=headers[j])
        hide_tick_labels_from_box_subplots(fig)
        if diag == 'histogram':
            fig['layout'].update(height=height, width=width, title=title, showlegend=True, barmode='stack')
            return fig
        elif diag == 'box':
            fig['layout'].update(height=height, width=width, title=title, showlegend=True)
            return fig
        else:
            fig['layout'].update(height=height, width=width, title=title, showlegend=True)
            return fig
    elif endpts:
        intervals = utils.endpts_to_intervals(endpts)
        if colormap_type == 'seq':
            foo = clrs.color_parser(colormap, clrs.unlabel_rgb)
            foo = clrs.n_colors(foo[0], foo[1], len(intervals))
            theme = clrs.color_parser(foo, clrs.label_rgb)
        if colormap_type == 'cat':
            theme = colormap
        dim = len(dataframe)
        fig = make_subplots(rows=dim, cols=dim, print_grid=False)
        trace_list = []
        legend_param = 0
        for listy in dataframe:
            for listx in dataframe:
                interval_labels = {}
                for interval in intervals:
                    interval_labels[str(interval)] = []
                c_indx = 0
                for interval in intervals:
                    new_listx = []
                    new_listy = []
                    for j in range(len(index_vals)):
                        if interval[0] < index_vals[j] <= interval[1]:
                            new_listx.append(listx[j])
                            new_listy.append(listy[j])
                    if legend_param == 1:
                        if listx == listy and diag == 'histogram':
                            trace = graph_objs.Histogram(x=new_listx, marker=dict(color=theme[c_indx]), showlegend=True)
                        elif listx == listy and diag == 'box':
                            trace = graph_objs.Box(y=new_listx, name=None, marker=dict(color=theme[c_indx]), showlegend=True)
                        elif 'marker' in kwargs:
                            kwargs['marker']['size'] = size
                            kwargs['marker']['color'] = theme[c_indx]
                            trace = graph_objs.Scatter(x=new_listx, y=new_listy, mode='markers', name=str(interval), showlegend=True, **kwargs)
                        else:
                            trace = graph_objs.Scatter(x=new_listx, y=new_listy, mode='markers', name=str(interval), marker=dict(size=size, color=theme[c_indx]), showlegend=True, **kwargs)
                    elif listx == listy and diag == 'histogram':
                        trace = graph_objs.Histogram(x=new_listx, marker=dict(color=theme[c_indx]), showlegend=False)
                    elif listx == listy and diag == 'box':
                        trace = graph_objs.Box(y=new_listx, name=None, marker=dict(color=theme[c_indx]), showlegend=False)
                    elif 'marker' in kwargs:
                        kwargs['marker']['size'] = size
                        kwargs['marker']['color'] = theme[c_indx]
                        trace = graph_objs.Scatter(x=new_listx, y=new_listy, mode='markers', name=str(interval), showlegend=False, **kwargs)
                    else:
                        trace = graph_objs.Scatter(x=new_listx, y=new_listy, mode='markers', name=str(interval), marker=dict(size=size, color=theme[c_indx]), showlegend=False, **kwargs)
                    interval_labels[str(interval)] = trace
                    if c_indx >= len(theme) - 1:
                        c_indx = -1
                    c_indx += 1
                trace_list.append(interval_labels)
                legend_param += 1
        trace_index = 0
        indices = range(1, dim + 1)
        for y_index in indices:
            for x_index in indices:
                for interval in intervals:
                    fig.append_trace(trace_list[trace_index][str(interval)], y_index, x_index)
                trace_index += 1
        for j in range(dim):
            xaxis_key = 'xaxis{}'.format(dim * dim - dim + 1 + j)
            fig['layout'][xaxis_key].update(title=headers[j])
        for j in range(dim):
            yaxis_key = 'yaxis{}'.format(1 + dim * j)
            fig['layout'][yaxis_key].update(title=headers[j])
        hide_tick_labels_from_box_subplots(fig)
        if diag == 'histogram':
            fig['layout'].update(height=height, width=width, title=title, showlegend=True, barmode='stack')
            return fig
        elif diag == 'box':
            fig['layout'].update(height=height, width=width, title=title, showlegend=True)
            return fig
        else:
            fig['layout'].update(height=height, width=width, title=title, showlegend=True)
            return fig
    else:
        theme = colormap
        if len(theme) <= 1:
            theme.append(theme[0])
        color = []
        for incr in range(len(theme)):
            color.append([1.0 / (len(theme) - 1) * incr, theme[incr]])
        dim = len(dataframe)
        fig = make_subplots(rows=dim, cols=dim, print_grid=False)
        trace_list = []
        legend_param = 0
        for listy in dataframe:
            for listx in dataframe:
                if legend_param == 1:
                    if listx == listy and diag == 'histogram':
                        trace = graph_objs.Histogram(x=listx, marker=dict(color=theme[0]), showlegend=False)
                    elif listx == listy and diag == 'box':
                        trace = graph_objs.Box(y=listx, marker=dict(color=theme[0]), showlegend=False)
                    elif 'marker' in kwargs:
                        kwargs['marker']['size'] = size
                        kwargs['marker']['color'] = index_vals
                        kwargs['marker']['colorscale'] = color
                        kwargs['marker']['showscale'] = True
                        trace = graph_objs.Scatter(x=listx, y=listy, mode='markers', showlegend=False, **kwargs)
                    else:
                        trace = graph_objs.Scatter(x=listx, y=listy, mode='markers', marker=dict(size=size, color=index_vals, colorscale=color, showscale=True), showlegend=False, **kwargs)
                elif listx == listy and diag == 'histogram':
                    trace = graph_objs.Histogram(x=listx, marker=dict(color=theme[0]), showlegend=False)
                elif listx == listy and diag == 'box':
                    trace = graph_objs.Box(y=listx, marker=dict(color=theme[0]), showlegend=False)
                elif 'marker' in kwargs:
                    kwargs['marker']['size'] = size
                    kwargs['marker']['color'] = index_vals
                    kwargs['marker']['colorscale'] = color
                    kwargs['marker']['showscale'] = False
                    trace = graph_objs.Scatter(x=listx, y=listy, mode='markers', showlegend=False, **kwargs)
                else:
                    trace = graph_objs.Scatter(x=listx, y=listy, mode='markers', marker=dict(size=size, color=index_vals, colorscale=color, showscale=False), showlegend=False, **kwargs)
                trace_list.append(trace)
                legend_param += 1
        trace_index = 0
        indices = range(1, dim + 1)
        for y_index in indices:
            for x_index in indices:
                fig.append_trace(trace_list[trace_index], y_index, x_index)
                trace_index += 1
        for j in range(dim):
            xaxis_key = 'xaxis{}'.format(dim * dim - dim + 1 + j)
            fig['layout'][xaxis_key].update(title=headers[j])
        for j in range(dim):
            yaxis_key = 'yaxis{}'.format(1 + dim * j)
            fig['layout'][yaxis_key].update(title=headers[j])
        hide_tick_labels_from_box_subplots(fig)
        if diag == 'histogram':
            fig['layout'].update(height=height, width=width, title=title, showlegend=True, barmode='stack')
            return fig
        elif diag == 'box':
            fig['layout'].update(height=height, width=width, title=title, showlegend=True)
            return fig
        else:
            fig['layout'].update(height=height, width=width, title=title, showlegend=True)
            return fig

def create_scatterplotmatrix(df, index=None, endpts=None, diag='scatter', height=500, width=500, size=6, title='Scatterplot Matrix', colormap=None, colormap_type='cat', dataframe=None, headers=None, index_vals=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns data for a scatterplot matrix;\n    **deprecated**,\n    use instead the plotly.graph_objects trace\n    :class:`plotly.graph_objects.Splom`.\n\n    :param (array) df: array of the data with column headers\n    :param (str) index: name of the index column in data array\n    :param (list|tuple) endpts: takes an increasing sequece of numbers\n        that defines intervals on the real line. They are used to group\n        the entries in an index of numbers into their corresponding\n        interval and therefore can be treated as categorical data\n    :param (str) diag: sets the chart type for the main diagonal plots.\n        The options are 'scatter', 'histogram' and 'box'.\n    :param (int|float) height: sets the height of the chart\n    :param (int|float) width: sets the width of the chart\n    :param (float) size: sets the marker size (in px)\n    :param (str) title: the title label of the scatterplot matrix\n    :param (str|tuple|list|dict) colormap: either a plotly scale name,\n        an rgb or hex color, a color tuple, a list of colors or a\n        dictionary. An rgb color is of the form 'rgb(x, y, z)' where\n        x, y and z belong to the interval [0, 255] and a color tuple is a\n        tuple of the form (a, b, c) where a, b and c belong to [0, 1].\n        If colormap is a list, it must contain valid color types as its\n        members.\n        If colormap is a dictionary, all the string entries in\n        the index column must be a key in colormap. In this case, the\n        colormap_type is forced to 'cat' or categorical\n    :param (str) colormap_type: determines how colormap is interpreted.\n        Valid choices are 'seq' (sequential) and 'cat' (categorical). If\n        'seq' is selected, only the first two colors in colormap will be\n        considered (when colormap is a list) and the index values will be\n        linearly interpolated between those two colors. This option is\n        forced if all index values are numeric.\n        If 'cat' is selected, a color from colormap will be assigned to\n        each category from index, including the intervals if endpts is\n        being used\n    :param (dict) **kwargs: a dictionary of scatterplot arguments\n        The only forbidden parameters are 'size', 'color' and\n        'colorscale' in 'marker'\n\n    Example 1: Vanilla Scatterplot Matrix\n\n    >>> from plotly.graph_objs import graph_objs\n    >>> from plotly.figure_factory import create_scatterplotmatrix\n\n    >>> import numpy as np\n    >>> import pandas as pd\n\n    >>> # Create dataframe\n    >>> df = pd.DataFrame(np.random.randn(10, 2),\n    ...                 columns=['Column 1', 'Column 2'])\n\n    >>> # Create scatterplot matrix\n    >>> fig = create_scatterplotmatrix(df)\n    >>> fig.show()\n\n\n    Example 2: Indexing a Column\n\n    >>> from plotly.graph_objs import graph_objs\n    >>> from plotly.figure_factory import create_scatterplotmatrix\n\n    >>> import numpy as np\n    >>> import pandas as pd\n\n    >>> # Create dataframe with index\n    >>> df = pd.DataFrame(np.random.randn(10, 2),\n    ...                    columns=['A', 'B'])\n\n    >>> # Add another column of strings to the dataframe\n    >>> df['Fruit'] = pd.Series(['apple', 'apple', 'grape', 'apple', 'apple',\n    ...                          'grape', 'pear', 'pear', 'apple', 'pear'])\n\n    >>> # Create scatterplot matrix\n    >>> fig = create_scatterplotmatrix(df, index='Fruit', size=10)\n    >>> fig.show()\n\n\n    Example 3: Styling the Diagonal Subplots\n\n    >>> from plotly.graph_objs import graph_objs\n    >>> from plotly.figure_factory import create_scatterplotmatrix\n\n    >>> import numpy as np\n    >>> import pandas as pd\n\n    >>> # Create dataframe with index\n    >>> df = pd.DataFrame(np.random.randn(10, 4),\n    ...                    columns=['A', 'B', 'C', 'D'])\n\n    >>> # Add another column of strings to the dataframe\n    >>> df['Fruit'] = pd.Series(['apple', 'apple', 'grape', 'apple', 'apple',\n    ...                          'grape', 'pear', 'pear', 'apple', 'pear'])\n\n    >>> # Create scatterplot matrix\n    >>> fig = create_scatterplotmatrix(df, diag='box', index='Fruit', height=1000,\n    ...                                width=1000)\n    >>> fig.show()\n\n\n    Example 4: Use a Theme to Style the Subplots\n\n    >>> from plotly.graph_objs import graph_objs\n    >>> from plotly.figure_factory import create_scatterplotmatrix\n\n    >>> import numpy as np\n    >>> import pandas as pd\n\n    >>> # Create dataframe with random data\n    >>> df = pd.DataFrame(np.random.randn(100, 3),\n    ...                    columns=['A', 'B', 'C'])\n\n    >>> # Create scatterplot matrix using a built-in\n    >>> # Plotly palette scale and indexing column 'A'\n    >>> fig = create_scatterplotmatrix(df, diag='histogram', index='A',\n    ...                                colormap='Blues', height=800, width=800)\n    >>> fig.show()\n\n\n    Example 5: Example 4 with Interval Factoring\n\n    >>> from plotly.graph_objs import graph_objs\n    >>> from plotly.figure_factory import create_scatterplotmatrix\n\n    >>> import numpy as np\n    >>> import pandas as pd\n\n    >>> # Create dataframe with random data\n    >>> df = pd.DataFrame(np.random.randn(100, 3),\n    ...                    columns=['A', 'B', 'C'])\n\n    >>> # Create scatterplot matrix using a list of 2 rgb tuples\n    >>> # and endpoints at -1, 0 and 1\n    >>> fig = create_scatterplotmatrix(df, diag='histogram', index='A',\n    ...                                colormap=['rgb(140, 255, 50)',\n    ...                                          'rgb(170, 60, 115)', '#6c4774',\n    ...                                          (0.5, 0.1, 0.8)],\n    ...                                endpts=[-1, 0, 1], height=800, width=800)\n    >>> fig.show()\n\n\n    Example 6: Using the colormap as a Dictionary\n\n    >>> from plotly.graph_objs import graph_objs\n    >>> from plotly.figure_factory import create_scatterplotmatrix\n\n    >>> import numpy as np\n    >>> import pandas as pd\n    >>> import random\n\n    >>> # Create dataframe with random data\n    >>> df = pd.DataFrame(np.random.randn(100, 3),\n    ...                    columns=['Column A',\n    ...                             'Column B',\n    ...                             'Column C'])\n\n    >>> # Add new color column to dataframe\n    >>> new_column = []\n    >>> strange_colors = ['turquoise', 'limegreen', 'goldenrod']\n\n    >>> for j in range(100):\n    ...     new_column.append(random.choice(strange_colors))\n    >>> df['Colors'] = pd.Series(new_column, index=df.index)\n\n    >>> # Create scatterplot matrix using a dictionary of hex color values\n    >>> # which correspond to actual color names in 'Colors' column\n    >>> fig = create_scatterplotmatrix(\n    ...     df, diag='box', index='Colors',\n    ...     colormap= dict(\n    ...         turquoise = '#00F5FF',\n    ...         limegreen = '#32CD32',\n    ...         goldenrod = '#DAA520'\n    ...     ),\n    ...     colormap_type='cat',\n    ...     height=800, width=800\n    ... )\n    >>> fig.show()\n    "
    if dataframe is None:
        dataframe = []
    if headers is None:
        headers = []
    if index_vals is None:
        index_vals = []
    validate_scatterplotmatrix(df, index, diag, colormap_type, **kwargs)
    if isinstance(colormap, dict):
        colormap = clrs.validate_colors_dict(colormap, 'rgb')
    elif isinstance(colormap, str) and 'rgb' not in colormap and ('#' not in colormap):
        if colormap not in clrs.PLOTLY_SCALES.keys():
            raise exceptions.PlotlyError("If 'colormap' is a string, it must be the name of a Plotly Colorscale. The available colorscale names are {}".format(clrs.PLOTLY_SCALES.keys()))
        else:
            colormap = clrs.colorscale_to_colors(clrs.PLOTLY_SCALES[colormap])
            colormap = [colormap[0]] + [colormap[-1]]
        colormap = clrs.validate_colors(colormap, 'rgb')
    else:
        colormap = clrs.validate_colors(colormap, 'rgb')
    if not index:
        for name in df:
            headers.append(name)
        for name in headers:
            dataframe.append(df[name].values.tolist())
        utils.validate_dataframe(dataframe)
        figure = scatterplot(dataframe, headers, diag, size, height, width, title, **kwargs)
        return figure
    else:
        if index not in df:
            raise exceptions.PlotlyError('Make sure you set the index input variable to one of the column names of your dataframe.')
        index_vals = df[index].values.tolist()
        for name in df:
            if name != index:
                headers.append(name)
        for name in headers:
            dataframe.append(df[name].values.tolist())
        utils.validate_dataframe(dataframe)
        utils.validate_index(index_vals)
        if isinstance(colormap, dict):
            for key in colormap:
                if not all((index in colormap for index in index_vals)):
                    raise exceptions.PlotlyError('If colormap is a dictionary, all the names in the index must be keys.')
            figure = scatterplot_dict(dataframe, headers, diag, size, height, width, title, index, index_vals, endpts, colormap, colormap_type, **kwargs)
            return figure
        else:
            figure = scatterplot_theme(dataframe, headers, diag, size, height, width, title, index, index_vals, endpts, colormap, colormap_type, **kwargs)
            return figure