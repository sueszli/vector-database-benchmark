import math
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
import plotly
import plotly.graph_objs as go
pd = optional_imports.get_module('pandas')

def _bullet(df, markers, measures, ranges, subtitles, titles, orientation, range_colors, measure_colors, horizontal_spacing, vertical_spacing, scatter_options, layout_options):
    if False:
        for i in range(10):
            print('nop')
    num_of_lanes = len(df)
    num_of_rows = num_of_lanes if orientation == 'h' else 1
    num_of_cols = 1 if orientation == 'h' else num_of_lanes
    if not horizontal_spacing:
        horizontal_spacing = 1.0 / num_of_lanes
    if not vertical_spacing:
        vertical_spacing = 1.0 / num_of_lanes
    fig = plotly.subplots.make_subplots(num_of_rows, num_of_cols, print_grid=False, horizontal_spacing=horizontal_spacing, vertical_spacing=vertical_spacing)
    fig['layout'].update(dict(shapes=[]), title='Bullet Chart', height=600, width=1000, showlegend=False, barmode='stack', annotations=[], margin=dict(l=120 if orientation == 'h' else 80))
    fig['layout'].update(layout_options)
    if orientation == 'h':
        width_axis = 'yaxis'
        length_axis = 'xaxis'
    else:
        width_axis = 'xaxis'
        length_axis = 'yaxis'
    for key in fig['layout']:
        if 'xaxis' in key or 'yaxis' in key:
            fig['layout'][key]['showgrid'] = False
            fig['layout'][key]['zeroline'] = False
        if length_axis in key:
            fig['layout'][key]['tickwidth'] = 1
        if width_axis in key:
            fig['layout'][key]['showticklabels'] = False
            fig['layout'][key]['range'] = [0, 1]
    if num_of_lanes <= 1:
        fig['layout'][width_axis + '1']['domain'] = [0.4, 0.6]
    if not range_colors:
        range_colors = ['rgb(200, 200, 200)', 'rgb(245, 245, 245)']
    if not measure_colors:
        measure_colors = ['rgb(31, 119, 180)', 'rgb(176, 196, 221)']
    for row in range(num_of_lanes):
        for idx in range(len(df.iloc[row]['ranges'])):
            inter_colors = clrs.n_colors(range_colors[0], range_colors[1], len(df.iloc[row]['ranges']), 'rgb')
            x = [sorted(df.iloc[row]['ranges'])[-1 - idx]] if orientation == 'h' else [0]
            y = [0] if orientation == 'h' else [sorted(df.iloc[row]['ranges'])[-1 - idx]]
            bar = go.Bar(x=x, y=y, marker=dict(color=inter_colors[-1 - idx]), name='ranges', hoverinfo='x' if orientation == 'h' else 'y', orientation=orientation, width=2, base=0, xaxis='x{}'.format(row + 1), yaxis='y{}'.format(row + 1))
            fig.add_trace(bar)
        for idx in range(len(df.iloc[row]['measures'])):
            inter_colors = clrs.n_colors(measure_colors[0], measure_colors[1], len(df.iloc[row]['measures']), 'rgb')
            x = [sorted(df.iloc[row]['measures'])[-1 - idx]] if orientation == 'h' else [0.5]
            y = [0.5] if orientation == 'h' else [sorted(df.iloc[row]['measures'])[-1 - idx]]
            bar = go.Bar(x=x, y=y, marker=dict(color=inter_colors[-1 - idx]), name='measures', hoverinfo='x' if orientation == 'h' else 'y', orientation=orientation, width=0.4, base=0, xaxis='x{}'.format(row + 1), yaxis='y{}'.format(row + 1))
            fig.add_trace(bar)
        x = df.iloc[row]['markers'] if orientation == 'h' else [0.5]
        y = [0.5] if orientation == 'h' else df.iloc[row]['markers']
        markers = go.Scatter(x=x, y=y, name='markers', hoverinfo='x' if orientation == 'h' else 'y', xaxis='x{}'.format(row + 1), yaxis='y{}'.format(row + 1), **scatter_options)
        fig.add_trace(markers)
        title = df.iloc[row]['titles']
        if 'subtitles' in df:
            subtitle = '<br>{}'.format(df.iloc[row]['subtitles'])
        else:
            subtitle = ''
        label = '<b>{}</b>'.format(title) + subtitle
        annot = utils.annotation_dict_for_label(label, num_of_lanes - row if orientation == 'h' else row + 1, num_of_lanes, vertical_spacing if orientation == 'h' else horizontal_spacing, 'row' if orientation == 'h' else 'col', True if orientation == 'h' else False, False)
        fig['layout']['annotations'] += (annot,)
    return fig

def create_bullet(data, markers=None, measures=None, ranges=None, subtitles=None, titles=None, orientation='h', range_colors=('rgb(200, 200, 200)', 'rgb(245, 245, 245)'), measure_colors=('rgb(31, 119, 180)', 'rgb(176, 196, 221)'), horizontal_spacing=None, vertical_spacing=None, scatter_options={}, **layout_options):
    if False:
        i = 10
        return i + 15
    '\n    **deprecated**, use instead the plotly.graph_objects trace\n    :class:`plotly.graph_objects.Indicator`.\n\n    :param (pd.DataFrame | list | tuple) data: either a list/tuple of\n        dictionaries or a pandas DataFrame.\n    :param (str) markers: the column name or dictionary key for the markers in\n        each subplot.\n    :param (str) measures: the column name or dictionary key for the measure\n        bars in each subplot. This bar usually represents the quantitative\n        measure of performance, usually a list of two values [a, b] and are\n        the blue bars in the foreground of each subplot by default.\n    :param (str) ranges: the column name or dictionary key for the qualitative\n        ranges of performance, usually a 3-item list [bad, okay, good]. They\n        correspond to the grey bars in the background of each chart.\n    :param (str) subtitles: the column name or dictionary key for the subtitle\n        of each subplot chart. The subplots are displayed right underneath\n        each title.\n    :param (str) titles: the column name or dictionary key for the main label\n        of each subplot chart.\n    :param (bool) orientation: if \'h\', the bars are placed horizontally as\n        rows. If \'v\' the bars are placed vertically in the chart.\n    :param (list) range_colors: a tuple of two colors between which all\n        the rectangles for the range are drawn. These rectangles are meant to\n        be qualitative indicators against which the marker and measure bars\n        are compared.\n        Default=(\'rgb(200, 200, 200)\', \'rgb(245, 245, 245)\')\n    :param (list) measure_colors: a tuple of two colors which is used to color\n        the thin quantitative bars in the bullet chart.\n        Default=(\'rgb(31, 119, 180)\', \'rgb(176, 196, 221)\')\n    :param (float) horizontal_spacing: see the \'horizontal_spacing\' param in\n        plotly.tools.make_subplots. Ranges between 0 and 1.\n    :param (float) vertical_spacing: see the \'vertical_spacing\' param in\n        plotly.tools.make_subplots. Ranges between 0 and 1.\n    :param (dict) scatter_options: describes attributes for the scatter trace\n        in each subplot such as name and marker size. Call\n        help(plotly.graph_objs.Scatter) for more information on valid params.\n    :param layout_options: describes attributes for the layout of the figure\n        such as title, height and width. Call help(plotly.graph_objs.Layout)\n        for more information on valid params.\n\n    Example 1: Use a Dictionary\n\n    >>> import plotly.figure_factory as ff\n\n    >>> data = [\n    ...   {"label": "revenue", "sublabel": "us$, in thousands",\n    ...    "range": [150, 225, 300], "performance": [220,270], "point": [250]},\n    ...   {"label": "Profit", "sublabel": "%", "range": [20, 25, 30],\n    ...    "performance": [21, 23], "point": [26]},\n    ...   {"label": "Order Size", "sublabel":"US$, average","range": [350, 500, 600],\n    ...    "performance": [100,320],"point": [550]},\n    ...   {"label": "New Customers", "sublabel": "count", "range": [1400, 2000, 2500],\n    ...    "performance": [1000, 1650],"point": [2100]},\n    ...   {"label": "Satisfaction", "sublabel": "out of 5","range": [3.5, 4.25, 5],\n    ...    "performance": [3.2, 4.7], "point": [4.4]}\n    ... ]\n\n    >>> fig = ff.create_bullet(\n    ...     data, titles=\'label\', subtitles=\'sublabel\', markers=\'point\',\n    ...     measures=\'performance\', ranges=\'range\', orientation=\'h\',\n    ...     title=\'my simple bullet chart\'\n    ... )\n    >>> fig.show()\n\n    Example 2: Use a DataFrame with Custom Colors\n\n    >>> import plotly.figure_factory as ff\n    >>> import pandas as pd\n    >>> data = pd.read_json(\'https://cdn.rawgit.com/plotly/datasets/master/BulletData.json\')\n\n    >>> fig = ff.create_bullet(\n    ...     data, titles=\'title\', markers=\'markers\', measures=\'measures\',\n    ...     orientation=\'v\', measure_colors=[\'rgb(14, 52, 75)\', \'rgb(31, 141, 127)\'],\n    ...     scatter_options={\'marker\': {\'symbol\': \'circle\'}}, width=700)\n    >>> fig.show()\n    '
    if not pd:
        raise ImportError("'pandas' must be installed for this figure factory.")
    if utils.is_sequence(data):
        if not all((isinstance(item, dict) for item in data)):
            raise exceptions.PlotlyError('Every entry of the data argument list, tuple, etc must be a dictionary.')
    elif not isinstance(data, pd.DataFrame):
        raise exceptions.PlotlyError('You must input a pandas DataFrame, or a list of dictionaries.')
    col_names = ['titles', 'subtitle', 'markers', 'measures', 'ranges']
    if utils.is_sequence(data):
        df = pd.DataFrame([[d[titles] for d in data] if titles else [''] * len(data), [d[subtitles] for d in data] if subtitles else [''] * len(data), [d[markers] for d in data] if markers else [[]] * len(data), [d[measures] for d in data] if measures else [[]] * len(data), [d[ranges] for d in data] if ranges else [[]] * len(data)], index=col_names)
    elif isinstance(data, pd.DataFrame):
        df = pd.DataFrame([data[titles].tolist() if titles else [''] * len(data), data[subtitles].tolist() if subtitles else [''] * len(data), data[markers].tolist() if markers else [[]] * len(data), data[measures].tolist() if measures else [[]] * len(data), data[ranges].tolist() if ranges else [[]] * len(data)], index=col_names)
    df = pd.DataFrame.transpose(df)
    for needed_key in ['ranges', 'measures', 'markers']:
        for (idx, r) in enumerate(df[needed_key]):
            try:
                r_is_nan = math.isnan(r)
                if r_is_nan or r is None:
                    df[needed_key][idx] = []
            except TypeError:
                pass
    for colors_list in [range_colors, measure_colors]:
        if colors_list:
            if len(colors_list) != 2:
                raise exceptions.PlotlyError("Both 'range_colors' or 'measure_colors' must be a list of two valid colors.")
            clrs.validate_colors(colors_list)
            colors_list = clrs.convert_colors_to_same_type(colors_list, 'rgb')[0]
    default_scatter = {'marker': {'size': 12, 'symbol': 'diamond-tall', 'color': 'rgb(0, 0, 0)'}}
    if scatter_options == {}:
        scatter_options.update(default_scatter)
    else:
        for k in default_scatter['marker']:
            if k not in scatter_options['marker']:
                scatter_options['marker'][k] = default_scatter['marker'][k]
    fig = _bullet(df, markers, measures, ranges, subtitles, titles, orientation, range_colors, measure_colors, horizontal_spacing, vertical_spacing, scatter_options, layout_options)
    return fig