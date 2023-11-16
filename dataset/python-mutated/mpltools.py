"""
Tools

A module for converting from mpl language to plotly language.

"""
import math
import warnings
import matplotlib.dates

def check_bar_match(old_bar, new_bar):
    if False:
        return 10
    'Check if two bars belong in the same collection (bar chart).\n\n    Positional arguments:\n    old_bar -- a previously sorted bar dictionary.\n    new_bar -- a new bar dictionary that needs to be sorted.\n\n    '
    tests = []
    tests += (new_bar['orientation'] == old_bar['orientation'],)
    tests += (new_bar['facecolor'] == old_bar['facecolor'],)
    if new_bar['orientation'] == 'v':
        new_width = new_bar['x1'] - new_bar['x0']
        old_width = old_bar['x1'] - old_bar['x0']
        tests += (new_width - old_width < 1e-06,)
        tests += (new_bar['y0'] == old_bar['y0'],)
    elif new_bar['orientation'] == 'h':
        new_height = new_bar['y1'] - new_bar['y0']
        old_height = old_bar['y1'] - old_bar['y0']
        tests += (new_height - old_height < 1e-06,)
        tests += (new_bar['x0'] == old_bar['x0'],)
    if all(tests):
        return True
    else:
        return False

def check_corners(inner_obj, outer_obj):
    if False:
        return 10
    inner_corners = inner_obj.get_window_extent().corners()
    outer_corners = outer_obj.get_window_extent().corners()
    if inner_corners[0][0] < outer_corners[0][0]:
        return False
    elif inner_corners[0][1] < outer_corners[0][1]:
        return False
    elif inner_corners[3][0] > outer_corners[3][0]:
        return False
    elif inner_corners[3][1] > outer_corners[3][1]:
        return False
    else:
        return True

def convert_dash(mpl_dash):
    if False:
        for i in range(10):
            print('nop')
    'Convert mpl line symbol to plotly line symbol and return symbol.'
    if mpl_dash in DASH_MAP:
        return DASH_MAP[mpl_dash]
    else:
        dash_array = mpl_dash.split(',')
        if len(dash_array) < 2:
            return 'solid'
        if math.isclose(float(dash_array[1]), 0.0):
            return 'solid'
        dashpx = ','.join([x + 'px' for x in dash_array])
        if dashpx == '7.4px,3.2px':
            dashpx = 'dashed'
        elif dashpx == '12.8px,3.2px,2.0px,3.2px':
            dashpx = 'dashdot'
        elif dashpx == '2.0px,3.3px':
            dashpx = 'dotted'
        return dashpx

def convert_path(path):
    if False:
        print('Hello World!')
    verts = path[0]
    code = tuple(path[1])
    if code in PATH_MAP:
        return PATH_MAP[code]
    else:
        return None

def convert_symbol(mpl_symbol):
    if False:
        print('Hello World!')
    'Convert mpl marker symbol to plotly symbol and return symbol.'
    if isinstance(mpl_symbol, list):
        symbol = list()
        for s in mpl_symbol:
            symbol += [convert_symbol(s)]
        return symbol
    elif mpl_symbol in SYMBOL_MAP:
        return SYMBOL_MAP[mpl_symbol]
    else:
        return 'circle'

def hex_to_rgb(value):
    if False:
        i = 10
        return i + 15
    "\n    Change a hex color to an rgb tuple\n\n    :param (str|unicode) value: The hex string we want to convert.\n    :return: (int, int, int) The red, green, blue int-tuple.\n\n    Example:\n\n        '#FFFFFF' --> (255, 255, 255)\n\n    "
    value = value.lstrip('#')
    lv = len(value)
    return tuple((int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)))

def merge_color_and_opacity(color, opacity):
    if False:
        print('Hello World!')
    "\n    Merge hex color with an alpha (opacity) to get an rgba tuple.\n\n    :param (str|unicode) color: A hex color string.\n    :param (float|int) opacity: A value [0, 1] for the 'a' in 'rgba'.\n    :return: (int, int, int, float) The rgba color and alpha tuple.\n\n    "
    if color is None:
        return None
    rgb_tup = hex_to_rgb(color)
    if opacity is None:
        return 'rgb {}'.format(rgb_tup)
    rgba_tup = rgb_tup + (opacity,)
    return 'rgba {}'.format(rgba_tup)

def convert_va(mpl_va):
    if False:
        return 10
    'Convert mpl vertical alignment word to equivalent HTML word.\n\n    Text alignment specifiers from mpl differ very slightly from those used\n    in HTML. See the VA_MAP for more details.\n\n    Positional arguments:\n    mpl_va -- vertical mpl text alignment spec.\n\n    '
    if mpl_va in VA_MAP:
        return VA_MAP[mpl_va]
    else:
        return None

def convert_x_domain(mpl_plot_bounds, mpl_max_x_bounds):
    if False:
        while True:
            i = 10
    "Map x dimension of current plot to plotly's domain space.\n\n    The bbox used to locate an axes object in mpl differs from the\n    method used to locate axes in plotly. The mpl version locates each\n    axes in the figure so that axes in a single-plot figure might have\n    the bounds, [0.125, 0.125, 0.775, 0.775] (x0, y0, width, height),\n    in mpl's figure coordinates. However, the axes all share one space in\n    plotly such that the domain will always be [0, 0, 1, 1]\n    (x0, y0, x1, y1). To convert between the two, the mpl figure bounds\n    need to be mapped to a [0, 1] domain for x and y. The margins set\n    upon opening a new figure will appropriately match the mpl margins.\n\n    Optionally, setting margins=0 and simply copying the domains from\n    mpl to plotly would place axes appropriately. However,\n    this would throw off axis and title labeling.\n\n    Positional arguments:\n    mpl_plot_bounds -- the (x0, y0, width, height) params for current ax **\n    mpl_max_x_bounds -- overall (x0, x1) bounds for all axes **\n\n    ** these are all specified in mpl figure coordinates\n\n    "
    mpl_x_dom = [mpl_plot_bounds[0], mpl_plot_bounds[0] + mpl_plot_bounds[2]]
    plotting_width = mpl_max_x_bounds[1] - mpl_max_x_bounds[0]
    x0 = (mpl_x_dom[0] - mpl_max_x_bounds[0]) / plotting_width
    x1 = (mpl_x_dom[1] - mpl_max_x_bounds[0]) / plotting_width
    return [x0, x1]

def convert_y_domain(mpl_plot_bounds, mpl_max_y_bounds):
    if False:
        while True:
            i = 10
    "Map y dimension of current plot to plotly's domain space.\n\n    The bbox used to locate an axes object in mpl differs from the\n    method used to locate axes in plotly. The mpl version locates each\n    axes in the figure so that axes in a single-plot figure might have\n    the bounds, [0.125, 0.125, 0.775, 0.775] (x0, y0, width, height),\n    in mpl's figure coordinates. However, the axes all share one space in\n    plotly such that the domain will always be [0, 0, 1, 1]\n    (x0, y0, x1, y1). To convert between the two, the mpl figure bounds\n    need to be mapped to a [0, 1] domain for x and y. The margins set\n    upon opening a new figure will appropriately match the mpl margins.\n\n    Optionally, setting margins=0 and simply copying the domains from\n    mpl to plotly would place axes appropriately. However,\n    this would throw off axis and title labeling.\n\n    Positional arguments:\n    mpl_plot_bounds -- the (x0, y0, width, height) params for current ax **\n    mpl_max_y_bounds -- overall (y0, y1) bounds for all axes **\n\n    ** these are all specified in mpl figure coordinates\n\n    "
    mpl_y_dom = [mpl_plot_bounds[1], mpl_plot_bounds[1] + mpl_plot_bounds[3]]
    plotting_height = mpl_max_y_bounds[1] - mpl_max_y_bounds[0]
    y0 = (mpl_y_dom[0] - mpl_max_y_bounds[0]) / plotting_height
    y1 = (mpl_y_dom[1] - mpl_max_y_bounds[0]) / plotting_height
    return [y0, y1]

def display_to_paper(x, y, layout):
    if False:
        i = 10
        return i + 15
    "Convert mpl display coordinates to plotly paper coordinates.\n\n    Plotly references object positions with an (x, y) coordinate pair in either\n    'data' or 'paper' coordinates which reference actual data in a plot or\n    the entire plotly axes space where the bottom-left of the bottom-left\n    plot has the location (x, y) = (0, 0) and the top-right of the top-right\n    plot has the location (x, y) = (1, 1). Display coordinates in mpl reference\n    objects with an (x, y) pair in pixel coordinates, where the bottom-left\n    corner is at the location (x, y) = (0, 0) and the top-right corner is at\n    the location (x, y) = (figwidth*dpi, figheight*dpi). Here, figwidth and\n    figheight are in inches and dpi are the dots per inch resolution.\n\n    "
    num_x = x - layout['margin']['l']
    den_x = layout['width'] - (layout['margin']['l'] + layout['margin']['r'])
    num_y = y - layout['margin']['b']
    den_y = layout['height'] - (layout['margin']['b'] + layout['margin']['t'])
    return (num_x / den_x, num_y / den_y)

def get_axes_bounds(fig):
    if False:
        for i in range(10):
            print('nop')
    "Return the entire axes space for figure.\n\n    An axes object in mpl is specified by its relation to the figure where\n    (0,0) corresponds to the bottom-left part of the figure and (1,1)\n    corresponds to the top-right. Margins exist in matplotlib because axes\n    objects normally don't go to the edges of the figure.\n\n    In plotly, the axes area (where all subplots go) is always specified with\n    the domain [0,1] for both x and y. This function finds the smallest box,\n    specified by two points, that all of the mpl axes objects fit into. This\n    box is then used to map mpl axes domains to plotly axes domains.\n\n    "
    (x_min, x_max, y_min, y_max) = ([], [], [], [])
    for axes_obj in fig.get_axes():
        bounds = axes_obj.get_position().bounds
        x_min.append(bounds[0])
        x_max.append(bounds[0] + bounds[2])
        y_min.append(bounds[1])
        y_max.append(bounds[1] + bounds[3])
    (x_min, y_min, x_max, y_max) = (min(x_min), min(y_min), max(x_max), max(y_max))
    return ((x_min, x_max), (y_min, y_max))

def get_axis_mirror(main_spine, mirror_spine):
    if False:
        print('Hello World!')
    if main_spine and mirror_spine:
        return 'ticks'
    elif main_spine and (not mirror_spine):
        return False
    elif not main_spine and mirror_spine:
        return False
    else:
        return False

def get_bar_gap(bar_starts, bar_ends, tol=1e-10):
    if False:
        return 10
    if len(bar_starts) == len(bar_ends) and len(bar_starts) > 1:
        sides1 = bar_starts[1:]
        sides2 = bar_ends[:-1]
        gaps = [s2 - s1 for (s2, s1) in zip(sides1, sides2)]
        gap0 = gaps[0]
        uniform = all([abs(gap0 - gap) < tol for gap in gaps])
        if uniform:
            return gap0

def convert_rgba_array(color_list):
    if False:
        return 10
    clean_color_list = list()
    for c in color_list:
        clean_color_list += [dict(r=int(c[0] * 255), g=int(c[1] * 255), b=int(c[2] * 255), a=c[3])]
    plotly_colors = list()
    for rgba in clean_color_list:
        plotly_colors += ['rgba({r},{g},{b},{a})'.format(**rgba)]
    if len(plotly_colors) == 1:
        return plotly_colors[0]
    else:
        return plotly_colors

def convert_path_array(path_array):
    if False:
        i = 10
        return i + 15
    symbols = list()
    for path in path_array:
        symbols += [convert_path(path)]
    if len(symbols) == 1:
        return symbols[0]
    else:
        return symbols

def convert_linewidth_array(width_array):
    if False:
        i = 10
        return i + 15
    if len(width_array) == 1:
        return width_array[0]
    else:
        return width_array

def convert_size_array(size_array):
    if False:
        for i in range(10):
            print('nop')
    size = [math.sqrt(s) for s in size_array]
    if len(size) == 1:
        return size[0]
    else:
        return size

def get_markerstyle_from_collection(props):
    if False:
        for i in range(10):
            print('nop')
    markerstyle = dict(alpha=None, facecolor=convert_rgba_array(props['styles']['facecolor']), marker=convert_path_array(props['paths']), edgewidth=convert_linewidth_array(props['styles']['linewidth']), markersize=convert_size_array(props['mplobj'].get_sizes()), edgecolor=convert_rgba_array(props['styles']['edgecolor']))
    return markerstyle

def get_rect_xmin(data):
    if False:
        print('Hello World!')
    'Find minimum x value from four (x,y) vertices.'
    return min(data[0][0], data[1][0], data[2][0], data[3][0])

def get_rect_xmax(data):
    if False:
        while True:
            i = 10
    'Find maximum x value from four (x,y) vertices.'
    return max(data[0][0], data[1][0], data[2][0], data[3][0])

def get_rect_ymin(data):
    if False:
        while True:
            i = 10
    'Find minimum y value from four (x,y) vertices.'
    return min(data[0][1], data[1][1], data[2][1], data[3][1])

def get_rect_ymax(data):
    if False:
        print('Hello World!')
    'Find maximum y value from four (x,y) vertices.'
    return max(data[0][1], data[1][1], data[2][1], data[3][1])

def get_spine_visible(ax, spine_key):
    if False:
        i = 10
        return i + 15
    'Return some spine parameters for the spine, `spine_key`.'
    spine = ax.spines[spine_key]
    ax_frame_on = ax.get_frame_on()
    position = spine._position or ('outward', 0.0)
    if isinstance(position, str):
        if position == 'center':
            position = ('axes', 0.5)
        elif position == 'zero':
            position = ('data', 0)
    (position_type, amount) = position
    if position_type == 'outward' and amount == 0:
        spine_frame_like = True
    else:
        spine_frame_like = False
    if not spine.get_visible():
        return False
    elif not spine._edgecolor[-1]:
        return False
    elif not ax_frame_on and spine_frame_like:
        return False
    elif ax_frame_on and spine_frame_like:
        return True
    elif not ax_frame_on and (not spine_frame_like):
        return True
    else:
        return False

def is_bar(bar_containers, **props):
    if False:
        for i in range(10):
            print('nop')
    'A test to decide whether a path is a bar from a vertical bar chart.'
    for container in bar_containers:
        if props['mplobj'] in container:
            return True
    return False

def make_bar(**props):
    if False:
        while True:
            i = 10
    "Make an intermediate bar dictionary.\n\n    This creates a bar dictionary which aids in the comparison of new bars to\n    old bars from other bar chart (patch) collections. This is not the\n    dictionary that needs to get passed to plotly as a data dictionary. That\n    happens in PlotlyRenderer in that class's draw_bar method. In other\n    words, this dictionary describes a SINGLE bar, whereas, plotly will\n    require a set of bars to be passed in a data dictionary.\n\n    "
    return {'bar': props['mplobj'], 'x0': get_rect_xmin(props['data']), 'y0': get_rect_ymin(props['data']), 'x1': get_rect_xmax(props['data']), 'y1': get_rect_ymax(props['data']), 'alpha': props['style']['alpha'], 'edgecolor': props['style']['edgecolor'], 'facecolor': props['style']['facecolor'], 'edgewidth': props['style']['edgewidth'], 'dasharray': props['style']['dasharray'], 'zorder': props['style']['zorder']}

def prep_ticks(ax, index, ax_type, props):
    if False:
        return 10
    "Prepare axis obj belonging to axes obj.\n\n    positional arguments:\n    ax - the mpl axes instance\n    index - the index of the axis in `props`\n    ax_type - 'x' or 'y' (for now)\n    props - an mplexporter poperties dictionary\n\n    "
    axis_dict = dict()
    if ax_type == 'x':
        axis = ax.get_xaxis()
    elif ax_type == 'y':
        axis = ax.get_yaxis()
    else:
        return dict()
    scale = props['axes'][index]['scale']
    if scale == 'linear':
        try:
            tickvalues = props['axes'][index]['tickvalues']
            tick0 = tickvalues[0]
            dticks = [round(tickvalues[i] - tickvalues[i - 1], 12) for i in range(1, len(tickvalues) - 1)]
            if all([dticks[i] == dticks[i - 1] for i in range(1, len(dticks) - 1)]):
                dtick = tickvalues[1] - tickvalues[0]
            else:
                warnings.warn("'linear' {0}-axis tick spacing not even, ignoring mpl tick formatting.".format(ax_type))
                raise TypeError
        except (IndexError, TypeError):
            axis_dict['nticks'] = props['axes'][index]['nticks']
        else:
            axis_dict['tick0'] = tick0
            axis_dict['dtick'] = dtick
            axis_dict['tickmode'] = None
    elif scale == 'log':
        try:
            axis_dict['tick0'] = props['axes'][index]['tickvalues'][0]
            axis_dict['dtick'] = props['axes'][index]['tickvalues'][1] - props['axes'][index]['tickvalues'][0]
            axis_dict['tickmode'] = None
        except (IndexError, TypeError):
            axis_dict = dict(nticks=props['axes'][index]['nticks'])
        base = axis.get_transform().base
        if base == 10:
            if ax_type == 'x':
                axis_dict['range'] = [math.log10(props['xlim'][0]), math.log10(props['xlim'][1])]
            elif ax_type == 'y':
                axis_dict['range'] = [math.log10(props['ylim'][0]), math.log10(props['ylim'][1])]
        else:
            axis_dict = dict(range=None, type='linear')
            warnings.warn("Converted non-base10 {0}-axis log scale to 'linear'".format(ax_type))
    else:
        return dict()
    formatter = axis.get_major_formatter().__class__.__name__
    if ax_type == 'x' and 'DateFormatter' in formatter:
        axis_dict['type'] = 'date'
        try:
            axis_dict['tick0'] = mpl_dates_to_datestrings(axis_dict['tick0'], formatter)
        except KeyError:
            pass
        finally:
            axis_dict.pop('dtick', None)
            axis_dict.pop('tickmode', None)
            axis_dict['range'] = mpl_dates_to_datestrings(props['xlim'], formatter)
    if formatter == 'LogFormatterMathtext':
        axis_dict['exponentformat'] = 'e'
    return axis_dict

def prep_xy_axis(ax, props, x_bounds, y_bounds):
    if False:
        print('Hello World!')
    xaxis = dict(type=props['axes'][0]['scale'], range=list(props['xlim']), showgrid=props['axes'][0]['grid']['gridOn'], domain=convert_x_domain(props['bounds'], x_bounds), side=props['axes'][0]['position'], tickfont=dict(size=props['axes'][0]['fontsize']))
    xaxis.update(prep_ticks(ax, 0, 'x', props))
    yaxis = dict(type=props['axes'][1]['scale'], range=list(props['ylim']), showgrid=props['axes'][1]['grid']['gridOn'], domain=convert_y_domain(props['bounds'], y_bounds), side=props['axes'][1]['position'], tickfont=dict(size=props['axes'][1]['fontsize']))
    yaxis.update(prep_ticks(ax, 1, 'y', props))
    return (xaxis, yaxis)

def mpl_dates_to_datestrings(dates, mpl_formatter):
    if False:
        i = 10
        return i + 15
    'Convert matplotlib dates to iso-formatted-like time strings.\n\n    Plotly\'s accepted format: "YYYY-MM-DD HH:MM:SS" (e.g., 2001-01-01 00:00:00)\n\n    Info on mpl dates: http://matplotlib.org/api/dates_api.html\n\n    '
    _dates = dates
    if mpl_formatter == 'TimeSeries_DateFormatter':
        try:
            dates = matplotlib.dates.epoch2num([date * 24 * 60 * 60 for date in dates])
            dates = matplotlib.dates.num2date(dates)
        except:
            return _dates
    else:
        try:
            dates = matplotlib.dates.num2date(dates)
        except:
            return _dates
    time_stings = [' '.join(date.isoformat().split('+')[0].split('T')) for date in dates]
    return time_stings
DASH_MAP = {'10,0': 'solid', '6,6': 'dash', '2,2': 'circle', '4,4,2,4': 'dashdot', 'none': 'solid', '7.4,3.2': 'dash'}
PATH_MAP = {('M', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'Z'): 'o', ('M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'Z'): '*', ('M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'Z'): '8', ('M', 'L', 'L', 'L', 'L', 'L', 'Z'): 'h', ('M', 'L', 'L', 'L', 'L', 'Z'): 'p', ('M', 'L', 'M', 'L', 'M', 'L'): '1', ('M', 'L', 'L', 'L', 'Z'): 's', ('M', 'L', 'M', 'L'): '+', ('M', 'L', 'L', 'Z'): '^', ('M', 'L'): '|'}
SYMBOL_MAP = {'o': 'circle', 'v': 'triangle-down', '^': 'triangle-up', '<': 'triangle-left', '>': 'triangle-right', 's': 'square', '+': 'cross', 'x': 'x', '*': 'star', 'D': 'diamond', 'd': 'diamond'}
VA_MAP = {'center': 'middle', 'baseline': 'bottom', 'top': 'top'}