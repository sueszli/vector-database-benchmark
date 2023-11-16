from numbers import Number
import copy
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
import plotly.graph_objects as go
pd = optional_imports.get_module('pandas')
REQUIRED_GANTT_KEYS = ['Task', 'Start', 'Finish']

def _get_corner_points(x0, y0, x1, y1):
    if False:
        while True:
            i = 10
    '\n    Returns the corner points of a scatter rectangle\n\n    :param x0: x-start\n    :param y0: y-lower\n    :param x1: x-end\n    :param y1: y-upper\n    :return: ([x], [y]), tuple of lists containing the x and y values\n    '
    return ([x0, x1, x1, x0], [y0, y0, y1, y1])

def validate_gantt(df):
    if False:
        print('Hello World!')
    '\n    Validates the inputted dataframe or list\n    '
    if pd and isinstance(df, pd.core.frame.DataFrame):
        for key in REQUIRED_GANTT_KEYS:
            if key not in df:
                raise exceptions.PlotlyError('The columns in your dataframe must include the following keys: {0}'.format(', '.join(REQUIRED_GANTT_KEYS)))
        num_of_rows = len(df.index)
        chart = []
        for index in range(num_of_rows):
            task_dict = {}
            for key in df:
                task_dict[key] = df.iloc[index][key]
            chart.append(task_dict)
        return chart
    if not isinstance(df, list):
        raise exceptions.PlotlyError('You must input either a dataframe or a list of dictionaries.')
    if len(df) <= 0:
        raise exceptions.PlotlyError('Your list is empty. It must contain at least one dictionary.')
    if not isinstance(df[0], dict):
        raise exceptions.PlotlyError('Your list must only include dictionaries.')
    return df

def gantt(chart, colors, title, bar_width, showgrid_x, showgrid_y, height, width, tasks=None, task_names=None, data=None, group_tasks=False, show_hover_fill=True, show_colorbar=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Refer to create_gantt() for docstring\n    '
    if tasks is None:
        tasks = []
    if task_names is None:
        task_names = []
    if data is None:
        data = []
    for index in range(len(chart)):
        task = dict(x0=chart[index]['Start'], x1=chart[index]['Finish'], name=chart[index]['Task'])
        if 'Description' in chart[index]:
            task['description'] = chart[index]['Description']
        tasks.append(task)
    scatter_data_dict = dict()
    marker_data_dict = dict()
    if show_hover_fill:
        hoverinfo = 'name'
    else:
        hoverinfo = 'skip'
    scatter_data_template = {'x': [], 'y': [], 'mode': 'none', 'fill': 'toself', 'hoverinfo': hoverinfo}
    marker_data_template = {'x': [], 'y': [], 'mode': 'markers', 'text': [], 'marker': dict(color='', size=1, opacity=0), 'name': '', 'showlegend': False}
    for index in range(len(tasks)):
        tn = tasks[index]['name']
        if not group_tasks or tn not in task_names:
            task_names.append(tn)
    if group_tasks:
        task_names.reverse()
    color_index = 0
    for index in range(len(tasks)):
        tn = tasks[index]['name']
        del tasks[index]['name']
        groupID = index
        if group_tasks:
            groupID = task_names.index(tn)
        tasks[index]['y0'] = groupID - bar_width
        tasks[index]['y1'] = groupID + bar_width
        if color_index >= len(colors):
            color_index = 0
        tasks[index]['fillcolor'] = colors[color_index]
        color_id = tasks[index]['fillcolor']
        if color_id not in scatter_data_dict:
            scatter_data_dict[color_id] = copy.deepcopy(scatter_data_template)
        scatter_data_dict[color_id]['fillcolor'] = color_id
        scatter_data_dict[color_id]['name'] = str(tn)
        scatter_data_dict[color_id]['legendgroup'] = color_id
        if len(scatter_data_dict[color_id]['x']) > 0:
            scatter_data_dict[color_id]['x'].append(scatter_data_dict[color_id]['x'][-1])
            scatter_data_dict[color_id]['y'].append(None)
        (xs, ys) = _get_corner_points(tasks[index]['x0'], tasks[index]['y0'], tasks[index]['x1'], tasks[index]['y1'])
        scatter_data_dict[color_id]['x'] += xs
        scatter_data_dict[color_id]['y'] += ys
        if color_id not in marker_data_dict:
            marker_data_dict[color_id] = copy.deepcopy(marker_data_template)
            marker_data_dict[color_id]['marker']['color'] = color_id
            marker_data_dict[color_id]['legendgroup'] = color_id
        marker_data_dict[color_id]['x'].append(tasks[index]['x0'])
        marker_data_dict[color_id]['x'].append(tasks[index]['x1'])
        marker_data_dict[color_id]['y'].append(groupID)
        marker_data_dict[color_id]['y'].append(groupID)
        if 'description' in tasks[index]:
            marker_data_dict[color_id]['text'].append(tasks[index]['description'])
            marker_data_dict[color_id]['text'].append(tasks[index]['description'])
            del tasks[index]['description']
        else:
            marker_data_dict[color_id]['text'].append(None)
            marker_data_dict[color_id]['text'].append(None)
        color_index += 1
    showlegend = show_colorbar
    layout = dict(title=title, showlegend=showlegend, height=height, width=width, shapes=[], hovermode='closest', yaxis=dict(showgrid=showgrid_y, ticktext=task_names, tickvals=list(range(len(task_names))), range=[-1, len(task_names) + 1], autorange=False, zeroline=False), xaxis=dict(showgrid=showgrid_x, zeroline=False, rangeselector=dict(buttons=list([dict(count=7, label='1w', step='day', stepmode='backward'), dict(count=1, label='1m', step='month', stepmode='backward'), dict(count=6, label='6m', step='month', stepmode='backward'), dict(count=1, label='YTD', step='year', stepmode='todate'), dict(count=1, label='1y', step='year', stepmode='backward'), dict(step='all')])), type='date'))
    data = [scatter_data_dict[k] for k in sorted(scatter_data_dict)]
    data += [marker_data_dict[k] for k in sorted(marker_data_dict)]
    fig = go.Figure(data=data, layout=layout)
    return fig

def gantt_colorscale(chart, colors, title, index_col, show_colorbar, bar_width, showgrid_x, showgrid_y, height, width, tasks=None, task_names=None, data=None, group_tasks=False, show_hover_fill=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Refer to FigureFactory.create_gantt() for docstring\n    '
    if tasks is None:
        tasks = []
    if task_names is None:
        task_names = []
    if data is None:
        data = []
    showlegend = False
    for index in range(len(chart)):
        task = dict(x0=chart[index]['Start'], x1=chart[index]['Finish'], name=chart[index]['Task'])
        if 'Description' in chart[index]:
            task['description'] = chart[index]['Description']
        tasks.append(task)
    scatter_data_dict = dict()
    marker_data_dict = dict()
    if show_hover_fill:
        hoverinfo = 'name'
    else:
        hoverinfo = 'skip'
    scatter_data_template = {'x': [], 'y': [], 'mode': 'none', 'fill': 'toself', 'showlegend': False, 'hoverinfo': hoverinfo, 'legendgroup': ''}
    marker_data_template = {'x': [], 'y': [], 'mode': 'markers', 'text': [], 'marker': dict(color='', size=1, opacity=0), 'name': '', 'showlegend': False, 'legendgroup': ''}
    index_vals = []
    for row in range(len(tasks)):
        if chart[row][index_col] not in index_vals:
            index_vals.append(chart[row][index_col])
    index_vals.sort()
    if isinstance(chart[0][index_col], Number):
        if len(colors) < 2:
            raise exceptions.PlotlyError("You must use at least 2 colors in 'colors' if you are using a colorscale. However only the first two colors given will be used for the lower and upper bounds on the colormap.")
        for index in range(len(tasks)):
            tn = tasks[index]['name']
            if not group_tasks or tn not in task_names:
                task_names.append(tn)
        if group_tasks:
            task_names.reverse()
        for index in range(len(tasks)):
            tn = tasks[index]['name']
            del tasks[index]['name']
            groupID = index
            if group_tasks:
                groupID = task_names.index(tn)
            tasks[index]['y0'] = groupID - bar_width
            tasks[index]['y1'] = groupID + bar_width
            colors = clrs.color_parser(colors, clrs.unlabel_rgb)
            lowcolor = colors[0]
            highcolor = colors[1]
            intermed = chart[index][index_col] / 100.0
            intermed_color = clrs.find_intermediate_color(lowcolor, highcolor, intermed)
            intermed_color = clrs.color_parser(intermed_color, clrs.label_rgb)
            tasks[index]['fillcolor'] = intermed_color
            color_id = tasks[index]['fillcolor']
            if color_id not in scatter_data_dict:
                scatter_data_dict[color_id] = copy.deepcopy(scatter_data_template)
            scatter_data_dict[color_id]['fillcolor'] = color_id
            scatter_data_dict[color_id]['name'] = str(chart[index][index_col])
            scatter_data_dict[color_id]['legendgroup'] = color_id
            colors = clrs.color_parser(colors, clrs.label_rgb)
            if len(scatter_data_dict[color_id]['x']) > 0:
                scatter_data_dict[color_id]['x'].append(scatter_data_dict[color_id]['x'][-1])
                scatter_data_dict[color_id]['y'].append(None)
            (xs, ys) = _get_corner_points(tasks[index]['x0'], tasks[index]['y0'], tasks[index]['x1'], tasks[index]['y1'])
            scatter_data_dict[color_id]['x'] += xs
            scatter_data_dict[color_id]['y'] += ys
            if color_id not in marker_data_dict:
                marker_data_dict[color_id] = copy.deepcopy(marker_data_template)
                marker_data_dict[color_id]['marker']['color'] = color_id
                marker_data_dict[color_id]['legendgroup'] = color_id
            marker_data_dict[color_id]['x'].append(tasks[index]['x0'])
            marker_data_dict[color_id]['x'].append(tasks[index]['x1'])
            marker_data_dict[color_id]['y'].append(groupID)
            marker_data_dict[color_id]['y'].append(groupID)
            if 'description' in tasks[index]:
                marker_data_dict[color_id]['text'].append(tasks[index]['description'])
                marker_data_dict[color_id]['text'].append(tasks[index]['description'])
                del tasks[index]['description']
            else:
                marker_data_dict[color_id]['text'].append(None)
                marker_data_dict[color_id]['text'].append(None)
        if show_colorbar is True:
            k = list(marker_data_dict.keys())[0]
            marker_data_dict[k]['marker'].update(dict(colorscale=[[0, colors[0]], [1, colors[1]]], showscale=True, cmax=100, cmin=0))
    if isinstance(chart[0][index_col], str):
        index_vals = []
        for row in range(len(tasks)):
            if chart[row][index_col] not in index_vals:
                index_vals.append(chart[row][index_col])
        index_vals.sort()
        if len(colors) < len(index_vals):
            raise exceptions.PlotlyError("Error. The number of colors in 'colors' must be no less than the number of unique index values in your group column.")
        index_vals_dict = {}
        c_index = 0
        for key in index_vals:
            if c_index > len(colors) - 1:
                c_index = 0
            index_vals_dict[key] = colors[c_index]
            c_index += 1
        for index in range(len(tasks)):
            tn = tasks[index]['name']
            if not group_tasks or tn not in task_names:
                task_names.append(tn)
        if group_tasks:
            task_names.reverse()
        for index in range(len(tasks)):
            tn = tasks[index]['name']
            del tasks[index]['name']
            groupID = index
            if group_tasks:
                groupID = task_names.index(tn)
            tasks[index]['y0'] = groupID - bar_width
            tasks[index]['y1'] = groupID + bar_width
            tasks[index]['fillcolor'] = index_vals_dict[chart[index][index_col]]
            color_id = tasks[index]['fillcolor']
            if color_id not in scatter_data_dict:
                scatter_data_dict[color_id] = copy.deepcopy(scatter_data_template)
            scatter_data_dict[color_id]['fillcolor'] = color_id
            scatter_data_dict[color_id]['legendgroup'] = color_id
            scatter_data_dict[color_id]['name'] = str(chart[index][index_col])
            colors = clrs.color_parser(colors, clrs.label_rgb)
            if len(scatter_data_dict[color_id]['x']) > 0:
                scatter_data_dict[color_id]['x'].append(scatter_data_dict[color_id]['x'][-1])
                scatter_data_dict[color_id]['y'].append(None)
            (xs, ys) = _get_corner_points(tasks[index]['x0'], tasks[index]['y0'], tasks[index]['x1'], tasks[index]['y1'])
            scatter_data_dict[color_id]['x'] += xs
            scatter_data_dict[color_id]['y'] += ys
            if color_id not in marker_data_dict:
                marker_data_dict[color_id] = copy.deepcopy(marker_data_template)
                marker_data_dict[color_id]['marker']['color'] = color_id
                marker_data_dict[color_id]['legendgroup'] = color_id
            marker_data_dict[color_id]['x'].append(tasks[index]['x0'])
            marker_data_dict[color_id]['x'].append(tasks[index]['x1'])
            marker_data_dict[color_id]['y'].append(groupID)
            marker_data_dict[color_id]['y'].append(groupID)
            if 'description' in tasks[index]:
                marker_data_dict[color_id]['text'].append(tasks[index]['description'])
                marker_data_dict[color_id]['text'].append(tasks[index]['description'])
                del tasks[index]['description']
            else:
                marker_data_dict[color_id]['text'].append(None)
                marker_data_dict[color_id]['text'].append(None)
        if show_colorbar is True:
            showlegend = True
            for k in scatter_data_dict:
                scatter_data_dict[k]['showlegend'] = showlegend
    layout = dict(title=title, showlegend=showlegend, height=height, width=width, shapes=[], hovermode='closest', yaxis=dict(showgrid=showgrid_y, ticktext=task_names, tickvals=list(range(len(task_names))), range=[-1, len(task_names) + 1], autorange=False, zeroline=False), xaxis=dict(showgrid=showgrid_x, zeroline=False, rangeselector=dict(buttons=list([dict(count=7, label='1w', step='day', stepmode='backward'), dict(count=1, label='1m', step='month', stepmode='backward'), dict(count=6, label='6m', step='month', stepmode='backward'), dict(count=1, label='YTD', step='year', stepmode='todate'), dict(count=1, label='1y', step='year', stepmode='backward'), dict(step='all')])), type='date'))
    data = [scatter_data_dict[k] for k in sorted(scatter_data_dict)]
    data += [marker_data_dict[k] for k in sorted(marker_data_dict)]
    fig = go.Figure(data=data, layout=layout)
    return fig

def gantt_dict(chart, colors, title, index_col, show_colorbar, bar_width, showgrid_x, showgrid_y, height, width, tasks=None, task_names=None, data=None, group_tasks=False, show_hover_fill=True):
    if False:
        print('Hello World!')
    '\n    Refer to FigureFactory.create_gantt() for docstring\n    '
    if tasks is None:
        tasks = []
    if task_names is None:
        task_names = []
    if data is None:
        data = []
    showlegend = False
    for index in range(len(chart)):
        task = dict(x0=chart[index]['Start'], x1=chart[index]['Finish'], name=chart[index]['Task'])
        if 'Description' in chart[index]:
            task['description'] = chart[index]['Description']
        tasks.append(task)
    scatter_data_dict = dict()
    marker_data_dict = dict()
    if show_hover_fill:
        hoverinfo = 'name'
    else:
        hoverinfo = 'skip'
    scatter_data_template = {'x': [], 'y': [], 'mode': 'none', 'fill': 'toself', 'hoverinfo': hoverinfo, 'legendgroup': ''}
    marker_data_template = {'x': [], 'y': [], 'mode': 'markers', 'text': [], 'marker': dict(color='', size=1, opacity=0), 'name': '', 'showlegend': False}
    index_vals = []
    for row in range(len(tasks)):
        if chart[row][index_col] not in index_vals:
            index_vals.append(chart[row][index_col])
    index_vals.sort()
    for key in index_vals:
        if key not in colors:
            raise exceptions.PlotlyError('If you are using colors as a dictionary, all of its keys must be all the values in the index column.')
    for index in range(len(tasks)):
        tn = tasks[index]['name']
        if not group_tasks or tn not in task_names:
            task_names.append(tn)
    if group_tasks:
        task_names.reverse()
    for index in range(len(tasks)):
        tn = tasks[index]['name']
        del tasks[index]['name']
        groupID = index
        if group_tasks:
            groupID = task_names.index(tn)
        tasks[index]['y0'] = groupID - bar_width
        tasks[index]['y1'] = groupID + bar_width
        tasks[index]['fillcolor'] = colors[chart[index][index_col]]
        color_id = tasks[index]['fillcolor']
        if color_id not in scatter_data_dict:
            scatter_data_dict[color_id] = copy.deepcopy(scatter_data_template)
        scatter_data_dict[color_id]['legendgroup'] = color_id
        scatter_data_dict[color_id]['fillcolor'] = color_id
        if len(scatter_data_dict[color_id]['x']) > 0:
            scatter_data_dict[color_id]['x'].append(scatter_data_dict[color_id]['x'][-1])
            scatter_data_dict[color_id]['y'].append(None)
        (xs, ys) = _get_corner_points(tasks[index]['x0'], tasks[index]['y0'], tasks[index]['x1'], tasks[index]['y1'])
        scatter_data_dict[color_id]['x'] += xs
        scatter_data_dict[color_id]['y'] += ys
        if color_id not in marker_data_dict:
            marker_data_dict[color_id] = copy.deepcopy(marker_data_template)
            marker_data_dict[color_id]['marker']['color'] = color_id
            marker_data_dict[color_id]['legendgroup'] = color_id
        marker_data_dict[color_id]['x'].append(tasks[index]['x0'])
        marker_data_dict[color_id]['x'].append(tasks[index]['x1'])
        marker_data_dict[color_id]['y'].append(groupID)
        marker_data_dict[color_id]['y'].append(groupID)
        if 'description' in tasks[index]:
            marker_data_dict[color_id]['text'].append(tasks[index]['description'])
            marker_data_dict[color_id]['text'].append(tasks[index]['description'])
            del tasks[index]['description']
        else:
            marker_data_dict[color_id]['text'].append(None)
            marker_data_dict[color_id]['text'].append(None)
    if show_colorbar is True:
        showlegend = True
    for index_value in index_vals:
        scatter_data_dict[colors[index_value]]['name'] = str(index_value)
    layout = dict(title=title, showlegend=showlegend, height=height, width=width, shapes=[], hovermode='closest', yaxis=dict(showgrid=showgrid_y, ticktext=task_names, tickvals=list(range(len(task_names))), range=[-1, len(task_names) + 1], autorange=False, zeroline=False), xaxis=dict(showgrid=showgrid_x, zeroline=False, rangeselector=dict(buttons=list([dict(count=7, label='1w', step='day', stepmode='backward'), dict(count=1, label='1m', step='month', stepmode='backward'), dict(count=6, label='6m', step='month', stepmode='backward'), dict(count=1, label='YTD', step='year', stepmode='todate'), dict(count=1, label='1y', step='year', stepmode='backward'), dict(step='all')])), type='date'))
    data = [scatter_data_dict[k] for k in sorted(scatter_data_dict)]
    data += [marker_data_dict[k] for k in sorted(marker_data_dict)]
    fig = go.Figure(data=data, layout=layout)
    return fig

def create_gantt(df, colors=None, index_col=None, show_colorbar=False, reverse_colors=False, title='Gantt Chart', bar_width=0.2, showgrid_x=False, showgrid_y=False, height=600, width=None, tasks=None, task_names=None, data=None, group_tasks=False, show_hover_fill=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    **deprecated**, use instead\n    :func:`plotly.express.timeline`.\n\n    Returns figure for a gantt chart\n\n    :param (array|list) df: input data for gantt chart. Must be either a\n        a dataframe or a list. If dataframe, the columns must include\n        \'Task\', \'Start\' and \'Finish\'. Other columns can be included and\n        used for indexing. If a list, its elements must be dictionaries\n        with the same required column headers: \'Task\', \'Start\' and\n        \'Finish\'.\n    :param (str|list|dict|tuple) colors: either a plotly scale name, an\n        rgb or hex color, a color tuple or a list of colors. An rgb color\n        is of the form \'rgb(x, y, z)\' where x, y, z belong to the interval\n        [0, 255] and a color tuple is a tuple of the form (a, b, c) where\n        a, b and c belong to [0, 1]. If colors is a list, it must\n        contain the valid color types aforementioned as its members.\n        If a dictionary, all values of the indexing column must be keys in\n        colors.\n    :param (str|float) index_col: the column header (if df is a data\n        frame) that will function as the indexing column. If df is a list,\n        index_col must be one of the keys in all the items of df.\n    :param (bool) show_colorbar: determines if colorbar will be visible.\n        Only applies if values in the index column are numeric.\n    :param (bool) show_hover_fill: enables/disables the hovertext for the\n        filled area of the chart.\n    :param (bool) reverse_colors: reverses the order of selected colors\n    :param (str) title: the title of the chart\n    :param (float) bar_width: the width of the horizontal bars in the plot\n    :param (bool) showgrid_x: show/hide the x-axis grid\n    :param (bool) showgrid_y: show/hide the y-axis grid\n    :param (float) height: the height of the chart\n    :param (float) width: the width of the chart\n\n    Example 1: Simple Gantt Chart\n\n    >>> from plotly.figure_factory import create_gantt\n\n    >>> # Make data for chart\n    >>> df = [dict(Task="Job A", Start=\'2009-01-01\', Finish=\'2009-02-30\'),\n    ...       dict(Task="Job B", Start=\'2009-03-05\', Finish=\'2009-04-15\'),\n    ...       dict(Task="Job C", Start=\'2009-02-20\', Finish=\'2009-05-30\')]\n\n    >>> # Create a figure\n    >>> fig = create_gantt(df)\n    >>> fig.show()\n\n\n    Example 2: Index by Column with Numerical Entries\n\n    >>> from plotly.figure_factory import create_gantt\n\n    >>> # Make data for chart\n    >>> df = [dict(Task="Job A", Start=\'2009-01-01\',\n    ...            Finish=\'2009-02-30\', Complete=10),\n    ...       dict(Task="Job B", Start=\'2009-03-05\',\n    ...            Finish=\'2009-04-15\', Complete=60),\n    ...       dict(Task="Job C", Start=\'2009-02-20\',\n    ...            Finish=\'2009-05-30\', Complete=95)]\n\n    >>> # Create a figure with Plotly colorscale\n    >>> fig = create_gantt(df, colors=\'Blues\', index_col=\'Complete\',\n    ...                    show_colorbar=True, bar_width=0.5,\n    ...                    showgrid_x=True, showgrid_y=True)\n    >>> fig.show()\n\n\n    Example 3: Index by Column with String Entries\n\n    >>> from plotly.figure_factory import create_gantt\n\n    >>> # Make data for chart\n    >>> df = [dict(Task="Job A", Start=\'2009-01-01\',\n    ...            Finish=\'2009-02-30\', Resource=\'Apple\'),\n    ...       dict(Task="Job B", Start=\'2009-03-05\',\n    ...            Finish=\'2009-04-15\', Resource=\'Grape\'),\n    ...       dict(Task="Job C", Start=\'2009-02-20\',\n    ...            Finish=\'2009-05-30\', Resource=\'Banana\')]\n\n    >>> # Create a figure with Plotly colorscale\n    >>> fig = create_gantt(df, colors=[\'rgb(200, 50, 25)\', (1, 0, 1), \'#6c4774\'],\n    ...                    index_col=\'Resource\', reverse_colors=True,\n    ...                    show_colorbar=True)\n    >>> fig.show()\n\n\n    Example 4: Use a dictionary for colors\n\n    >>> from plotly.figure_factory import create_gantt\n    >>> # Make data for chart\n    >>> df = [dict(Task="Job A", Start=\'2009-01-01\',\n    ...            Finish=\'2009-02-30\', Resource=\'Apple\'),\n    ...       dict(Task="Job B", Start=\'2009-03-05\',\n    ...            Finish=\'2009-04-15\', Resource=\'Grape\'),\n    ...       dict(Task="Job C", Start=\'2009-02-20\',\n    ...            Finish=\'2009-05-30\', Resource=\'Banana\')]\n\n    >>> # Make a dictionary of colors\n    >>> colors = {\'Apple\': \'rgb(255, 0, 0)\',\n    ...           \'Grape\': \'rgb(170, 14, 200)\',\n    ...           \'Banana\': (1, 1, 0.2)}\n\n    >>> # Create a figure with Plotly colorscale\n    >>> fig = create_gantt(df, colors=colors, index_col=\'Resource\',\n    ...                    show_colorbar=True)\n\n    >>> fig.show()\n\n    Example 5: Use a pandas dataframe\n\n    >>> from plotly.figure_factory import create_gantt\n    >>> import pandas as pd\n\n    >>> # Make data as a dataframe\n    >>> df = pd.DataFrame([[\'Run\', \'2010-01-01\', \'2011-02-02\', 10],\n    ...                    [\'Fast\', \'2011-01-01\', \'2012-06-05\', 55],\n    ...                    [\'Eat\', \'2012-01-05\', \'2013-07-05\', 94]],\n    ...                   columns=[\'Task\', \'Start\', \'Finish\', \'Complete\'])\n\n    >>> # Create a figure with Plotly colorscale\n    >>> fig = create_gantt(df, colors=\'Blues\', index_col=\'Complete\',\n    ...                    show_colorbar=True, bar_width=0.5,\n    ...                    showgrid_x=True, showgrid_y=True)\n    >>> fig.show()\n    '
    chart = validate_gantt(df)
    if index_col:
        if index_col not in chart[0]:
            raise exceptions.PlotlyError('In order to use an indexing column and assign colors to the values of the index, you must choose an actual column name in the dataframe or key if a list of dictionaries is being used.')
        index_list = []
        for dictionary in chart:
            index_list.append(dictionary[index_col])
        utils.validate_index(index_list)
    if isinstance(colors, dict):
        colors = clrs.validate_colors_dict(colors, 'rgb')
    else:
        colors = clrs.validate_colors(colors, 'rgb')
    if reverse_colors is True:
        colors.reverse()
    if not index_col:
        if isinstance(colors, dict):
            raise exceptions.PlotlyError('Error. You have set colors to a dictionary but have not picked an index. An index is required if you are assigning colors to particular values in a dictionary.')
        fig = gantt(chart, colors, title, bar_width, showgrid_x, showgrid_y, height, width, tasks=None, task_names=None, data=None, group_tasks=group_tasks, show_hover_fill=show_hover_fill, show_colorbar=show_colorbar)
        return fig
    elif not isinstance(colors, dict):
        fig = gantt_colorscale(chart, colors, title, index_col, show_colorbar, bar_width, showgrid_x, showgrid_y, height, width, tasks=None, task_names=None, data=None, group_tasks=group_tasks, show_hover_fill=show_hover_fill)
        return fig
    else:
        fig = gantt_dict(chart, colors, title, index_col, show_colorbar, bar_width, showgrid_x, showgrid_y, height, width, tasks=None, task_names=None, data=None, group_tasks=group_tasks, show_hover_fill=show_hover_fill)
        return fig