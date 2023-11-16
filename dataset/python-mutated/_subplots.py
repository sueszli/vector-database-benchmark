import collections
_single_subplot_types = {'scene', 'geo', 'polar', 'ternary', 'mapbox'}
_subplot_types = set.union(_single_subplot_types, {'xy', 'domain'})
_subplot_prop_named_subplot = {'polar', 'ternary', 'mapbox'}
SubplotXY = collections.namedtuple('SubplotXY', ('xaxis', 'yaxis'))
SubplotDomain = collections.namedtuple('SubplotDomain', ('x', 'y'))
SubplotRef = collections.namedtuple('SubplotRef', ('subplot_type', 'layout_keys', 'trace_kwargs'))

def _get_initial_max_subplot_ids():
    if False:
        i = 10
        return i + 15
    max_subplot_ids = {subplot_type: 0 for subplot_type in _single_subplot_types}
    max_subplot_ids['xaxis'] = 0
    max_subplot_ids['yaxis'] = 0
    return max_subplot_ids

def make_subplots(rows=1, cols=1, shared_xaxes=False, shared_yaxes=False, start_cell='top-left', print_grid=False, horizontal_spacing=None, vertical_spacing=None, subplot_titles=None, column_widths=None, row_heights=None, specs=None, insets=None, column_titles=None, row_titles=None, x_title=None, y_title=None, figure=None, **kwargs):
    if False:
        return 10
    '\n    Return an instance of plotly.graph_objs.Figure with predefined subplots\n    configured in \'layout\'.\n\n    Parameters\n    ----------\n    rows: int (default 1)\n        Number of rows in the subplot grid. Must be greater than zero.\n\n    cols: int (default 1)\n        Number of columns in the subplot grid. Must be greater than zero.\n\n    shared_xaxes: boolean or str (default False)\n        Assign shared (linked) x-axes for 2D cartesian subplots\n\n          - True or \'columns\': Share axes among subplots in the same column\n          - \'rows\': Share axes among subplots in the same row\n          - \'all\': Share axes across all subplots in the grid.\n\n    shared_yaxes: boolean or str (default False)\n        Assign shared (linked) y-axes for 2D cartesian subplots\n\n          - \'columns\': Share axes among subplots in the same column\n          - True or \'rows\': Share axes among subplots in the same row\n          - \'all\': Share axes across all subplots in the grid.\n\n    start_cell: \'bottom-left\' or \'top-left\' (default \'top-left\')\n        Choose the starting cell in the subplot grid used to set the\n        domains_grid of the subplots.\n\n          - \'top-left\': Subplots are numbered with (1, 1) in the top\n                        left corner\n          - \'bottom-left\': Subplots are numbererd with (1, 1) in the bottom\n                           left corner\n\n    print_grid: boolean (default True):\n        If True, prints a string representation of the plot grid.  Grid may\n        also be printed using the `Figure.print_grid()` method on the\n        resulting figure.\n\n    horizontal_spacing: float (default 0.2 / cols)\n        Space between subplot columns in normalized plot coordinates. Must be\n        a float between 0 and 1.\n\n        Applies to all columns (use \'specs\' subplot-dependents spacing)\n\n    vertical_spacing: float (default 0.3 / rows)\n        Space between subplot rows in normalized plot coordinates. Must be\n        a float between 0 and 1.\n\n        Applies to all rows (use \'specs\' subplot-dependents spacing)\n\n    subplot_titles: list of str or None (default None)\n        Title of each subplot as a list in row-major ordering.\n\n        Empty strings ("") can be included in the list if no subplot title\n        is desired in that space so that the titles are properly indexed.\n\n    specs: list of lists of dict or None (default None)\n        Per subplot specifications of subplot type, row/column spanning, and\n        spacing.\n\n        ex1: specs=[[{}, {}], [{\'colspan\': 2}, None]]\n\n        ex2: specs=[[{\'rowspan\': 2}, {}], [None, {}]]\n\n        - Indices of the outer list correspond to subplot grid rows\n          starting from the top, if start_cell=\'top-left\',\n          or bottom, if start_cell=\'bottom-left\'.\n          The number of rows in \'specs\' must be equal to \'rows\'.\n\n        - Indices of the inner lists correspond to subplot grid columns\n          starting from the left. The number of columns in \'specs\'\n          must be equal to \'cols\'.\n\n        - Each item in the \'specs\' list corresponds to one subplot\n          in a subplot grid. (N.B. The subplot grid has exactly \'rows\'\n          times \'cols\' cells.)\n\n        - Use None for a blank a subplot cell (or to move past a col/row span).\n\n        - Note that specs[0][0] has the specs of the \'start_cell\' subplot.\n\n        - Each item in \'specs\' is a dictionary.\n            The available keys are:\n            * type (string, default \'xy\'): Subplot type. One of\n                - \'xy\': 2D Cartesian subplot type for scatter, bar, etc.\n                - \'scene\': 3D Cartesian subplot for scatter3d, cone, etc.\n                - \'polar\': Polar subplot for scatterpolar, barpolar, etc.\n                - \'ternary\': Ternary subplot for scatterternary\n                - \'mapbox\': Mapbox subplot for scattermapbox\n                - \'domain\': Subplot type for traces that are individually\n                            positioned. pie, parcoords, parcats, etc.\n                - trace type: A trace type which will be used to determine\n                              the appropriate subplot type for that trace\n\n            * secondary_y (bool, default False): If True, create a secondary\n                y-axis positioned on the right side of the subplot. Only valid\n                if type=\'xy\'.\n            * colspan (int, default 1): number of subplot columns\n                for this subplot to span.\n            * rowspan (int, default 1): number of subplot rows\n                for this subplot to span.\n            * l (float, default 0.0): padding left of cell\n            * r (float, default 0.0): padding right of cell\n            * t (float, default 0.0): padding right of cell\n            * b (float, default 0.0): padding bottom of cell\n\n        - Note: Use \'horizontal_spacing\' and \'vertical_spacing\' to adjust\n          the spacing in between the subplots.\n\n    insets: list of dict or None (default None):\n        Inset specifications.  Insets are subplots that overlay grid subplots\n\n        - Each item in \'insets\' is a dictionary.\n            The available keys are:\n\n            * cell (tuple, default=(1,1)): (row, col) index of the\n                subplot cell to overlay inset axes onto.\n            * type (string, default \'xy\'): Subplot type\n            * l (float, default=0.0): padding left of inset\n                  in fraction of cell width\n            * w (float or \'to_end\', default=\'to_end\') inset width\n                  in fraction of cell width (\'to_end\': to cell right edge)\n            * b (float, default=0.0): padding bottom of inset\n                  in fraction of cell height\n            * h (float or \'to_end\', default=\'to_end\') inset height\n                  in fraction of cell height (\'to_end\': to cell top edge)\n\n    column_widths: list of numbers or None (default None)\n        list of length `cols` of the relative widths of each column of suplots.\n        Values are normalized internally and used to distribute overall width\n        of the figure (excluding padding) among the columns.\n\n        For backward compatibility, may also be specified using the\n        `column_width` keyword argument.\n\n    row_heights: list of numbers or None (default None)\n        list of length `rows` of the relative heights of each row of subplots.\n        If start_cell=\'top-left\' then row heights are applied top to bottom.\n        Otherwise, if start_cell=\'bottom-left\' then row heights are applied\n        bottom to top.\n\n        For backward compatibility, may also be specified using the\n        `row_width` kwarg. If specified as `row_width`, then the width values\n        are applied from bottom to top regardless of the value of start_cell.\n        This matches the legacy behavior of the `row_width` argument.\n\n    column_titles: list of str or None (default None)\n        list of length `cols` of titles to place above the top subplot in\n        each column.\n\n    row_titles: list of str or None (default None)\n        list of length `rows` of titles to place on the right side of each\n        row of subplots. If start_cell=\'top-left\' then row titles are\n        applied top to bottom. Otherwise, if start_cell=\'bottom-left\' then\n        row titles are applied bottom to top.\n\n    x_title: str or None (default None)\n        Title to place below the bottom row of subplots,\n        centered horizontally\n\n    y_title: str or None (default None)\n        Title to place to the left of the left column of subplots,\n        centered vertically\n\n    figure: go.Figure or None (default None)\n        If None, a new go.Figure instance will be created and its axes will be\n        populated with those corresponding to the requested subplot geometry and\n        this new figure will be returned.\n        If a go.Figure instance, the axes will be added to the\n        layout of this figure and this figure will be returned. If the figure\n        already contains axes, they will be overwritten.\n\n    Examples\n    --------\n\n    Example 1:\n\n    >>> # Stack two subplots vertically, and add a scatter trace to each\n    >>> from plotly.subplots import make_subplots\n    >>> import plotly.graph_objects as go\n    >>> fig = make_subplots(rows=2)\n\n    This is the format of your plot grid:\n    [ (1,1) xaxis1,yaxis1 ]\n    [ (2,1) xaxis2,yaxis2 ]\n\n    >>> fig.add_scatter(y=[2, 1, 3], row=1, col=1) # doctest: +ELLIPSIS\n    Figure(...)\n    >>> fig.add_scatter(y=[1, 3, 2], row=2, col=1) # doctest: +ELLIPSIS\n    Figure(...)\n\n    or see Figure.append_trace\n\n    Example 2:\n\n    >>> # Stack a scatter plot\n    >>> fig = make_subplots(rows=2, shared_xaxes=True)\n\n    This is the format of your plot grid:\n    [ (1,1) xaxis1,yaxis1 ]\n    [ (2,1) xaxis2,yaxis2 ]\n\n    >>> fig.add_scatter(y=[2, 1, 3], row=1, col=1) # doctest: +ELLIPSIS\n    Figure(...)\n    >>> fig.add_scatter(y=[1, 3, 2], row=2, col=1) # doctest: +ELLIPSIS\n    Figure(...)\n\n    Example 3:\n\n    >>> # irregular subplot layout (more examples below under \'specs\')\n    >>> fig = make_subplots(rows=2, cols=2,\n    ...                     specs=[[{}, {}],\n    ...                     [{\'colspan\': 2}, None]])\n\n    This is the format of your plot grid:\n    [ (1,1) xaxis1,yaxis1 ]  [ (1,2) xaxis2,yaxis2 ]\n    [ (2,1) xaxis3,yaxis3           -              ]\n\n    >>> fig.add_trace(go.Scatter(x=[1,2,3], y=[2,1,2]), row=1, col=1) # doctest: +ELLIPSIS\n    Figure(...)\n    >>> fig.add_trace(go.Scatter(x=[1,2,3], y=[2,1,2]), row=1, col=2) # doctest: +ELLIPSIS\n    Figure(...)\n    >>> fig.add_trace(go.Scatter(x=[1,2,3], y=[2,1,2]), row=2, col=1) # doctest: +ELLIPSIS\n    Figure(...)\n\n    Example 4:\n\n    >>> # insets\n    >>> fig = make_subplots(insets=[{\'cell\': (1,1), \'l\': 0.7, \'b\': 0.3}])\n\n    This is the format of your plot grid:\n    [ (1,1) xaxis1,yaxis1 ]\n\n    With insets:\n    [ xaxis2,yaxis2 ] over [ (1,1) xaxis1,yaxis1 ]\n\n    >>> fig.add_scatter(x=[1,2,3], y=[2,1,1]) # doctest: +ELLIPSIS\n    Figure(...)\n    >>> fig.add_scatter(x=[1,2,3], y=[2,1,2], xaxis=\'x2\', yaxis=\'y2\') # doctest: +ELLIPSIS\n    Figure(...)\n\n    Example 5:\n\n    >>> # include subplot titles\n    >>> fig = make_subplots(rows=2, subplot_titles=(\'Plot 1\',\'Plot 2\'))\n\n    This is the format of your plot grid:\n    [ (1,1) x1,y1 ]\n    [ (2,1) x2,y2 ]\n\n    >>> fig.add_scatter(x=[1,2,3], y=[2,1,2], row=1, col=1) # doctest: +ELLIPSIS\n    Figure(...)\n    >>> fig.add_bar(x=[1,2,3], y=[2,1,2], row=2, col=1) # doctest: +ELLIPSIS\n    Figure(...)\n\n    Example 6:\n\n    Subplot with mixed subplot types\n\n    >>> fig = make_subplots(rows=2, cols=2,\n    ...                     specs=[[{\'type\': \'xy\'},    {\'type\': \'polar\'}],\n    ...                            [{\'type\': \'scene\'}, {\'type\': \'ternary\'}]])\n\n    >>> fig.add_traces(\n    ...     [go.Scatter(y=[2, 3, 1]),\n    ...      go.Scatterpolar(r=[1, 3, 2], theta=[0, 45, 90]),\n    ...      go.Scatter3d(x=[1, 2, 1], y=[2, 3, 1], z=[0, 3, 5]),\n    ...      go.Scatterternary(a=[0.1, 0.2, 0.1],\n    ...                        b=[0.2, 0.3, 0.1],\n    ...                        c=[0.7, 0.5, 0.8])],\n    ...     rows=[1, 1, 2, 2],\n    ...     cols=[1, 2, 1, 2]) # doctest: +ELLIPSIS\n    Figure(...)\n    '
    import plotly.graph_objs as go
    use_legacy_row_heights_order = 'row_width' in kwargs
    row_heights = kwargs.pop('row_width', row_heights)
    column_widths = kwargs.pop('column_width', column_widths)
    if kwargs:
        raise TypeError('make_subplots() got unexpected keyword argument(s): {}'.format(list(kwargs)))
    if not isinstance(rows, int) or rows <= 0:
        raise ValueError("\nThe 'rows' argument to make_suplots must be an int greater than 0.\n    Received value of type {typ}: {val}".format(typ=type(rows), val=repr(rows)))
    if not isinstance(cols, int) or cols <= 0:
        raise ValueError("\nThe 'cols' argument to make_suplots must be an int greater than 0.\n    Received value of type {typ}: {val}".format(typ=type(cols), val=repr(cols)))
    if start_cell == 'bottom-left':
        col_dir = 1
        row_dir = 1
    elif start_cell == 'top-left':
        col_dir = 1
        row_dir = -1
    else:
        raise ValueError("\nThe 'start_cell` argument to make_subplots must be one of ['bottom-left', 'top-left']\n    Received value of type {typ}: {val}".format(typ=type(start_cell), val=repr(start_cell)))

    def _check_keys_and_fill(name, arg, defaults):
        if False:
            i = 10
            return i + 15

        def _checks(item, defaults):
            if False:
                print('Hello World!')
            if item is None:
                return
            if not isinstance(item, dict):
                raise ValueError("\nElements of the '{name}' argument to make_suplots must be dictionaries or None.\n    Received value of type {typ}: {val}".format(name=name, typ=type(item), val=repr(item)))
            for k in item:
                if k not in defaults:
                    raise ValueError("\nInvalid key specified in an element of the '{name}' argument to make_subplots: {k}\n    Valid keys include: {valid_keys}".format(k=repr(k), name=name, valid_keys=repr(list(defaults))))
            for (k, v) in defaults.items():
                item.setdefault(k, v)
        for arg_i in arg:
            if isinstance(arg_i, (list, tuple)):
                for arg_ii in arg_i:
                    _checks(arg_ii, defaults)
            elif isinstance(arg_i, dict):
                _checks(arg_i, defaults)
    if specs is None:
        specs = [[{} for c in range(cols)] for r in range(rows)]
    elif not (isinstance(specs, (list, tuple)) and specs and all((isinstance(row, (list, tuple)) for row in specs)) and (len(specs) == rows) and all((len(row) == cols for row in specs)) and all((all((v is None or isinstance(v, dict) for v in row)) for row in specs))):
        raise ValueError("\nThe 'specs' argument to make_subplots must be a 2D list of dictionaries with dimensions ({rows} x {cols}).\n    Received value of type {typ}: {val}".format(rows=rows, cols=cols, typ=type(specs), val=repr(specs)))
    for row in specs:
        for spec in row:
            if spec and spec.pop('is_3d', None):
                spec['type'] = 'scene'
    spec_defaults = dict(type='xy', secondary_y=False, colspan=1, rowspan=1, l=0.0, r=0.0, b=0.0, t=0.0)
    _check_keys_and_fill('specs', specs, spec_defaults)
    has_secondary_y = False
    for row in specs:
        for spec in row:
            if spec is not None:
                has_secondary_y = has_secondary_y or spec['secondary_y']
            if spec and spec['type'] != 'xy' and spec['secondary_y']:
                raise ValueError("\nThe 'secondary_y' spec property is not supported for subplot of type '{s_typ}'\n     'secondary_y' is only supported for subplots of type 'xy'\n".format(s_typ=spec['type']))
    if insets is None or insets is False:
        insets = []
    elif not (isinstance(insets, (list, tuple)) and all((isinstance(v, dict) for v in insets))):
        raise ValueError("\nThe 'insets' argument to make_suplots must be a list of dictionaries.\n    Received value of type {typ}: {val}".format(typ=type(insets), val=repr(insets)))
    if insets:
        for inset in insets:
            if inset and inset.pop('is_3d', None):
                inset['type'] = 'scene'
        inset_defaults = dict(cell=(1, 1), type='xy', l=0.0, w='to_end', b=0.0, h='to_end')
        _check_keys_and_fill('insets', insets, inset_defaults)
    valid_shared_vals = [None, True, False, 'rows', 'columns', 'all']
    shared_err_msg = '\nThe {arg} argument to make_subplots must be one of: {valid_vals}\n    Received value of type {typ}: {val}'
    if shared_xaxes not in valid_shared_vals:
        val = shared_xaxes
        raise ValueError(shared_err_msg.format(arg='shared_xaxes', valid_vals=valid_shared_vals, typ=type(val), val=repr(val)))
    if shared_yaxes not in valid_shared_vals:
        val = shared_yaxes
        raise ValueError(shared_err_msg.format(arg='shared_yaxes', valid_vals=valid_shared_vals, typ=type(val), val=repr(val)))

    def _check_hv_spacing(dimsize, spacing, name, dimvarname, dimname):
        if False:
            while True:
                i = 10
        if spacing < 0 or spacing > 1:
            raise ValueError('%s spacing must be between 0 and 1.' % (name,))
        if dimsize <= 1:
            return
        max_spacing = 1.0 / float(dimsize - 1)
        if spacing > max_spacing:
            raise ValueError('{name} spacing cannot be greater than (1 / ({dimvarname} - 1)) = {max_spacing:f}.\nThe resulting plot would have {dimsize} {dimname} ({dimvarname}={dimsize}).'.format(dimvarname=dimvarname, name=name, dimname=dimname, max_spacing=max_spacing, dimsize=dimsize))
    if horizontal_spacing is None:
        if has_secondary_y:
            horizontal_spacing = 0.4 / cols
        else:
            horizontal_spacing = 0.2 / cols
    _check_hv_spacing(cols, horizontal_spacing, 'Horizontal', 'cols', 'columns')
    if vertical_spacing is None:
        if subplot_titles is not None:
            vertical_spacing = 0.5 / rows
        else:
            vertical_spacing = 0.3 / rows
    _check_hv_spacing(rows, vertical_spacing, 'Vertical', 'rows', 'rows')
    if subplot_titles is None:
        subplot_titles = [''] * rows * cols
    if has_secondary_y:
        max_width = 0.94
    elif row_titles:
        max_width = 0.98
    else:
        max_width = 1.0
    if column_widths is None:
        widths = [(max_width - horizontal_spacing * (cols - 1)) / cols] * cols
    elif isinstance(column_widths, (list, tuple)) and len(column_widths) == cols:
        cum_sum = float(sum(column_widths))
        widths = []
        for w in column_widths:
            widths.append((max_width - horizontal_spacing * (cols - 1)) * (w / cum_sum))
    else:
        raise ValueError("\nThe 'column_widths' argument to make_suplots must be a list of numbers of length {cols}.\n    Received value of type {typ}: {val}".format(cols=cols, typ=type(column_widths), val=repr(column_widths)))
    if row_heights is None:
        heights = [(1.0 - vertical_spacing * (rows - 1)) / rows] * rows
    elif isinstance(row_heights, (list, tuple)) and len(row_heights) == rows:
        cum_sum = float(sum(row_heights))
        heights = []
        for h in row_heights:
            heights.append((1.0 - vertical_spacing * (rows - 1)) * (h / cum_sum))
        if row_dir < 0 and (not use_legacy_row_heights_order):
            heights = list(reversed(heights))
    else:
        raise ValueError("\nThe 'row_heights' argument to make_suplots must be a list of numbers of length {rows}.\n    Received value of type {typ}: {val}".format(rows=rows, typ=type(row_heights), val=repr(row_heights)))
    if column_titles and (not isinstance(column_titles, (list, tuple))):
        raise ValueError('\nThe column_titles argument to make_subplots must be a list or tuple\n    Received value of type {typ}: {val}'.format(typ=type(column_titles), val=repr(column_titles)))
    if row_titles and (not isinstance(row_titles, (list, tuple))):
        raise ValueError('\nThe row_titles argument to make_subplots must be a list or tuple\n    Received value of type {typ}: {val}'.format(typ=type(row_titles), val=repr(row_titles)))
    layout = go.Layout()
    col_seq = range(cols)[::col_dir]
    row_seq = range(rows)[::row_dir]
    grid = [[(sum(widths[:c]) + c * horizontal_spacing, sum(heights[:r]) + r * vertical_spacing) for c in col_seq] for r in row_seq]
    domains_grid = [[None for _ in range(cols)] for _ in range(rows)]
    grid_ref = [[None for c in range(cols)] for r in range(rows)]
    list_of_domains = []
    max_subplot_ids = _get_initial_max_subplot_ids()
    for (r, spec_row) in enumerate(specs):
        for (c, spec) in enumerate(spec_row):
            if spec is None:
                continue
            c_spanned = c + spec['colspan'] - 1
            r_spanned = r + spec['rowspan'] - 1
            if c_spanned >= cols:
                raise Exception("Some 'colspan' value is too large for this subplot grid.")
            if r_spanned >= rows:
                raise Exception("Some 'rowspan' value is too large for this subplot grid.")
            x_s = grid[r][c][0] + spec['l']
            x_e = grid[r][c_spanned][0] + widths[c_spanned] - spec['r']
            x_domain = [x_s, x_e]
            if row_dir > 0:
                y_s = grid[r][c][1] + spec['b']
                y_e = grid[r_spanned][c][1] + heights[r_spanned] - spec['t']
            else:
                y_s = grid[r_spanned][c][1] + spec['b']
                y_e = grid[r][c][1] + heights[-1 - r] - spec['t']
            if y_s < 0.0:
                if y_s > -0.01:
                    y_s = 0.0
                else:
                    raise Exception("A combination of the 'b' values, heights, and number of subplots too large for this subplot grid.")
            if y_s > 1.0:
                if y_s < 1.01:
                    y_s = 1.0
                else:
                    raise Exception("A combination of the 'b' values, heights, and number of subplots too large for this subplot grid.")
            if y_e < 0.0:
                if y_e > -0.01:
                    y_e = 0.0
                else:
                    raise Exception("A combination of the 't' values, heights, and number of subplots too large for this subplot grid.")
            if y_e > 1.0:
                if y_e < 1.01:
                    y_e = 1.0
                else:
                    raise Exception("A combination of the 't' values, heights, and number of subplots too large for this subplot grid.")
            y_domain = [y_s, y_e]
            list_of_domains.append(x_domain)
            list_of_domains.append(y_domain)
            domains_grid[r][c] = [x_domain, y_domain]
            subplot_type = spec['type']
            secondary_y = spec['secondary_y']
            subplot_refs = _init_subplot(layout, subplot_type, secondary_y, x_domain, y_domain, max_subplot_ids)
            grid_ref[r][c] = subplot_refs
    _configure_shared_axes(layout, grid_ref, specs, 'x', shared_xaxes, row_dir)
    _configure_shared_axes(layout, grid_ref, specs, 'y', shared_yaxes, row_dir)
    insets_ref = [None for inset in range(len(insets))] if insets else None
    if insets:
        for (i_inset, inset) in enumerate(insets):
            r = inset['cell'][0] - 1
            c = inset['cell'][1] - 1
            if not 0 <= r < rows:
                raise Exception("Some 'cell' row value is out of range. Note: the starting cell is (1, 1)")
            if not 0 <= c < cols:
                raise Exception("Some 'cell' col value is out of range. Note: the starting cell is (1, 1)")
            x_s = grid[r][c][0] + inset['l'] * widths[c]
            if inset['w'] == 'to_end':
                x_e = grid[r][c][0] + widths[c]
            else:
                x_e = x_s + inset['w'] * widths[c]
            x_domain = [x_s, x_e]
            y_s = grid[r][c][1] + inset['b'] * heights[-1 - r]
            if inset['h'] == 'to_end':
                y_e = grid[r][c][1] + heights[-1 - r]
            else:
                y_e = y_s + inset['h'] * heights[-1 - r]
            y_domain = [y_s, y_e]
            list_of_domains.append(x_domain)
            list_of_domains.append(y_domain)
            subplot_type = inset['type']
            subplot_refs = _init_subplot(layout, subplot_type, False, x_domain, y_domain, max_subplot_ids)
            insets_ref[i_inset] = subplot_refs
    grid_str = _build_grid_str(specs, grid_ref, insets, insets_ref, row_seq)
    plot_title_annotations = _build_subplot_title_annotations(subplot_titles, list_of_domains)
    layout['annotations'] = plot_title_annotations
    if column_titles:
        domains_list = []
        if row_dir > 0:
            for c in range(cols):
                domain_pair = domains_grid[-1][c]
                if domain_pair:
                    domains_list.extend(domain_pair)
        else:
            for c in range(cols):
                domain_pair = domains_grid[0][c]
                if domain_pair:
                    domains_list.extend(domain_pair)
        column_title_annotations = _build_subplot_title_annotations(column_titles, domains_list)
        layout['annotations'] += tuple(column_title_annotations)
    if row_titles:
        domains_list = []
        for r in range(rows):
            domain_pair = domains_grid[r][-1]
            if domain_pair:
                domains_list.extend(domain_pair)
        column_title_annotations = _build_subplot_title_annotations(row_titles, domains_list, title_edge='right')
        layout['annotations'] += tuple(column_title_annotations)
    if x_title:
        domains_list = [(0, max_width), (0, 1)]
        column_title_annotations = _build_subplot_title_annotations([x_title], domains_list, title_edge='bottom', offset=30)
        layout['annotations'] += tuple(column_title_annotations)
    if y_title:
        domains_list = [(0, 1), (0, 1)]
        column_title_annotations = _build_subplot_title_annotations([y_title], domains_list, title_edge='left', offset=40)
        layout['annotations'] += tuple(column_title_annotations)
    if print_grid:
        print(grid_str)
    if figure is None:
        figure = go.Figure()
    figure.update_layout(layout)
    figure.__dict__['_grid_ref'] = grid_ref
    figure.__dict__['_grid_str'] = grid_str
    return figure

def _configure_shared_axes(layout, grid_ref, specs, x_or_y, shared, row_dir):
    if False:
        while True:
            i = 10
    rows = len(grid_ref)
    cols = len(grid_ref[0])
    layout_key_ind = ['x', 'y'].index(x_or_y)
    if row_dir < 0:
        rows_iter = range(rows - 1, -1, -1)
    else:
        rows_iter = range(rows)

    def update_axis_matches(first_axis_id, subplot_ref, spec, remove_label):
        if False:
            i = 10
            return i + 15
        if subplot_ref is None:
            return first_axis_id
        if x_or_y == 'x':
            span = spec['colspan']
        else:
            span = spec['rowspan']
        if subplot_ref.subplot_type == 'xy' and span == 1:
            if first_axis_id is None:
                first_axis_name = subplot_ref.layout_keys[layout_key_ind]
                first_axis_id = first_axis_name.replace('axis', '')
            else:
                axis_name = subplot_ref.layout_keys[layout_key_ind]
                axis_to_match = layout[axis_name]
                axis_to_match.matches = first_axis_id
                if remove_label:
                    axis_to_match.showticklabels = False
        return first_axis_id
    if shared == 'columns' or (x_or_y == 'x' and shared is True):
        for c in range(cols):
            first_axis_id = None
            ok_to_remove_label = x_or_y == 'x'
            for r in rows_iter:
                if not grid_ref[r][c]:
                    continue
                subplot_ref = grid_ref[r][c][0]
                spec = specs[r][c]
                first_axis_id = update_axis_matches(first_axis_id, subplot_ref, spec, ok_to_remove_label)
    elif shared == 'rows' or (x_or_y == 'y' and shared is True):
        for r in rows_iter:
            first_axis_id = None
            ok_to_remove_label = x_or_y == 'y'
            for c in range(cols):
                if not grid_ref[r][c]:
                    continue
                subplot_ref = grid_ref[r][c][0]
                spec = specs[r][c]
                first_axis_id = update_axis_matches(first_axis_id, subplot_ref, spec, ok_to_remove_label)
    elif shared == 'all':
        first_axis_id = None
        for c in range(cols):
            for (ri, r) in enumerate(rows_iter):
                if not grid_ref[r][c]:
                    continue
                subplot_ref = grid_ref[r][c][0]
                spec = specs[r][c]
                if x_or_y == 'y':
                    ok_to_remove_label = c > 0
                else:
                    ok_to_remove_label = ri > 0 if row_dir > 0 else r < rows - 1
                first_axis_id = update_axis_matches(first_axis_id, subplot_ref, spec, ok_to_remove_label)

def _init_subplot_xy(layout, secondary_y, x_domain, y_domain, max_subplot_ids=None):
    if False:
        while True:
            i = 10
    if max_subplot_ids is None:
        max_subplot_ids = _get_initial_max_subplot_ids()
    x_cnt = max_subplot_ids['xaxis'] + 1
    y_cnt = max_subplot_ids['yaxis'] + 1
    x_label = 'x{cnt}'.format(cnt=x_cnt if x_cnt > 1 else '')
    y_label = 'y{cnt}'.format(cnt=y_cnt if y_cnt > 1 else '')
    (x_anchor, y_anchor) = (y_label, x_label)
    xaxis_name = 'xaxis{cnt}'.format(cnt=x_cnt if x_cnt > 1 else '')
    yaxis_name = 'yaxis{cnt}'.format(cnt=y_cnt if y_cnt > 1 else '')
    x_axis = {'domain': x_domain, 'anchor': x_anchor}
    y_axis = {'domain': y_domain, 'anchor': y_anchor}
    layout[xaxis_name] = x_axis
    layout[yaxis_name] = y_axis
    subplot_refs = [SubplotRef(subplot_type='xy', layout_keys=(xaxis_name, yaxis_name), trace_kwargs={'xaxis': x_label, 'yaxis': y_label})]
    if secondary_y:
        y_cnt += 1
        secondary_yaxis_name = 'yaxis{cnt}'.format(cnt=y_cnt if y_cnt > 1 else '')
        secondary_y_label = 'y{cnt}'.format(cnt=y_cnt)
        subplot_refs.append(SubplotRef(subplot_type='xy', layout_keys=(xaxis_name, secondary_yaxis_name), trace_kwargs={'xaxis': x_label, 'yaxis': secondary_y_label}))
        secondary_y_axis = {'anchor': y_anchor, 'overlaying': y_label, 'side': 'right'}
        layout[secondary_yaxis_name] = secondary_y_axis
    max_subplot_ids['xaxis'] = x_cnt
    max_subplot_ids['yaxis'] = y_cnt
    return tuple(subplot_refs)

def _init_subplot_single(layout, subplot_type, x_domain, y_domain, max_subplot_ids=None):
    if False:
        for i in range(10):
            print('nop')
    if max_subplot_ids is None:
        max_subplot_ids = _get_initial_max_subplot_ids()
    cnt = max_subplot_ids[subplot_type] + 1
    label = '{subplot_type}{cnt}'.format(subplot_type=subplot_type, cnt=cnt if cnt > 1 else '')
    scene = dict(domain={'x': x_domain, 'y': y_domain})
    layout[label] = scene
    trace_key = 'subplot' if subplot_type in _subplot_prop_named_subplot else subplot_type
    subplot_ref = SubplotRef(subplot_type=subplot_type, layout_keys=(label,), trace_kwargs={trace_key: label})
    max_subplot_ids[subplot_type] = cnt
    return (subplot_ref,)

def _init_subplot_domain(x_domain, y_domain):
    if False:
        for i in range(10):
            print('nop')
    subplot_ref = SubplotRef(subplot_type='domain', layout_keys=(), trace_kwargs={'domain': {'x': tuple(x_domain), 'y': tuple(y_domain)}})
    return (subplot_ref,)

def _subplot_type_for_trace_type(trace_type):
    if False:
        return 10
    from plotly.validators import DataValidator
    trace_validator = DataValidator()
    if trace_type in trace_validator.class_strs_map:
        trace = trace_validator.validate_coerce([{'type': trace_type}])[0]
        if 'domain' in trace:
            return 'domain'
        elif 'xaxis' in trace and 'yaxis' in trace:
            return 'xy'
        elif 'geo' in trace:
            return 'geo'
        elif 'scene' in trace:
            return 'scene'
        elif 'subplot' in trace:
            for t in _subplot_prop_named_subplot:
                try:
                    trace.subplot = t
                    return t
                except ValueError:
                    pass
    return None

def _validate_coerce_subplot_type(subplot_type):
    if False:
        for i in range(10):
            print('nop')
    orig_subplot_type = subplot_type
    subplot_type = subplot_type.lower()
    if subplot_type in _subplot_types:
        return subplot_type
    subplot_type = _subplot_type_for_trace_type(subplot_type)
    if subplot_type is None:
        raise ValueError('Unsupported subplot type: {}'.format(repr(orig_subplot_type)))
    else:
        return subplot_type

def _init_subplot(layout, subplot_type, secondary_y, x_domain, y_domain, max_subplot_ids=None):
    if False:
        print('Hello World!')
    subplot_type = _validate_coerce_subplot_type(subplot_type)
    if max_subplot_ids is None:
        max_subplot_ids = _get_initial_max_subplot_ids()
    x_domain = [max(0.0, x_domain[0]), min(1.0, x_domain[1])]
    y_domain = [max(0.0, y_domain[0]), min(1.0, y_domain[1])]
    if subplot_type == 'xy':
        subplot_refs = _init_subplot_xy(layout, secondary_y, x_domain, y_domain, max_subplot_ids)
    elif subplot_type in _single_subplot_types:
        subplot_refs = _init_subplot_single(layout, subplot_type, x_domain, y_domain, max_subplot_ids)
    elif subplot_type == 'domain':
        subplot_refs = _init_subplot_domain(x_domain, y_domain)
    else:
        raise ValueError('Unsupported subplot type: {}'.format(repr(subplot_type)))
    return subplot_refs

def _get_cartesian_label(x_or_y, r, c, cnt):
    if False:
        return 10
    label = '{x_or_y}{cnt}'.format(x_or_y=x_or_y, cnt=cnt)
    return label

def _build_subplot_title_annotations(subplot_titles, list_of_domains, title_edge='top', offset=0):
    if False:
        return 10
    x_dom = list_of_domains[::2]
    y_dom = list_of_domains[1::2]
    subtitle_pos_x = []
    subtitle_pos_y = []
    if title_edge == 'top':
        text_angle = 0
        xanchor = 'center'
        yanchor = 'bottom'
        for x_domains in x_dom:
            subtitle_pos_x.append(sum(x_domains) / 2.0)
        for y_domains in y_dom:
            subtitle_pos_y.append(y_domains[1])
        yshift = offset
        xshift = 0
    elif title_edge == 'bottom':
        text_angle = 0
        xanchor = 'center'
        yanchor = 'top'
        for x_domains in x_dom:
            subtitle_pos_x.append(sum(x_domains) / 2.0)
        for y_domains in y_dom:
            subtitle_pos_y.append(y_domains[0])
        yshift = -offset
        xshift = 0
    elif title_edge == 'right':
        text_angle = 90
        xanchor = 'left'
        yanchor = 'middle'
        for x_domains in x_dom:
            subtitle_pos_x.append(x_domains[1])
        for y_domains in y_dom:
            subtitle_pos_y.append(sum(y_domains) / 2.0)
        yshift = 0
        xshift = offset
    elif title_edge == 'left':
        text_angle = -90
        xanchor = 'right'
        yanchor = 'middle'
        for x_domains in x_dom:
            subtitle_pos_x.append(x_domains[0])
        for y_domains in y_dom:
            subtitle_pos_y.append(sum(y_domains) / 2.0)
        yshift = 0
        xshift = -offset
    else:
        raise ValueError("Invalid annotation edge '{edge}'".format(edge=title_edge))
    plot_titles = []
    for index in range(len(subplot_titles)):
        if not subplot_titles[index] or index >= len(subtitle_pos_y):
            pass
        else:
            annot = {'y': subtitle_pos_y[index], 'xref': 'paper', 'x': subtitle_pos_x[index], 'yref': 'paper', 'text': subplot_titles[index], 'showarrow': False, 'font': dict(size=16), 'xanchor': xanchor, 'yanchor': yanchor}
            if xshift != 0:
                annot['xshift'] = xshift
            if yshift != 0:
                annot['yshift'] = yshift
            if text_angle != 0:
                annot['textangle'] = text_angle
            plot_titles.append(annot)
    return plot_titles

def _build_grid_str(specs, grid_ref, insets, insets_ref, row_seq):
    if False:
        return 10
    rows = len(specs)
    cols = len(specs[0])
    sp = '  '
    s_str = '[ '
    e_str = ' ]'
    s_top = '⎡ '
    s_mid = '⎢ '
    s_bot = '⎣ '
    e_top = ' ⎤'
    e_mid = ' ⎟'
    e_bot = ' ⎦'
    colspan_str = '       -'
    rowspan_str = '       :'
    empty_str = '    (empty) '
    grid_str = 'This is the format of your plot grid:\n'
    _tmp = [['' for c in range(cols)] for r in range(rows)]

    def _get_cell_str(r, c, subplot_refs):
        if False:
            i = 10
            return i + 15
        layout_keys = sorted({k for ref in subplot_refs for k in ref.layout_keys})
        ref_str = ','.join(layout_keys)
        ref_str = ref_str.replace('axis', '')
        return '({r},{c}) {ref}'.format(r=r + 1, c=c + 1, ref=ref_str)
    cell_len = max([len(_get_cell_str(r, c, ref)) for (r, row_ref) in enumerate(grid_ref) for (c, ref) in enumerate(row_ref) if ref]) + len(s_str) + len(e_str)

    def _pad(s, cell_len=cell_len):
        if False:
            print('Hello World!')
        return ' ' * (cell_len - len(s))
    for (r, spec_row) in enumerate(specs):
        for (c, spec) in enumerate(spec_row):
            ref = grid_ref[r][c]
            if ref is None:
                if _tmp[r][c] == '':
                    _tmp[r][c] = empty_str + _pad(empty_str)
                continue
            if spec['rowspan'] > 1:
                cell_str = s_top + _get_cell_str(r, c, ref)
            else:
                cell_str = s_str + _get_cell_str(r, c, ref)
            if spec['colspan'] > 1:
                for cc in range(1, spec['colspan'] - 1):
                    _tmp[r][c + cc] = colspan_str + _pad(colspan_str)
                if spec['rowspan'] > 1:
                    _tmp[r][c + spec['colspan'] - 1] = colspan_str + _pad(colspan_str + e_str) + e_top
                else:
                    _tmp[r][c + spec['colspan'] - 1] = colspan_str + _pad(colspan_str + e_str) + e_str
            else:
                padding = ' ' * (cell_len - len(cell_str) - 2)
                if spec['rowspan'] > 1:
                    cell_str += padding + e_top
                else:
                    cell_str += padding + e_str
            if spec['rowspan'] > 1:
                for cc in range(spec['colspan']):
                    for rr in range(1, spec['rowspan']):
                        row_str = rowspan_str + _pad(rowspan_str)
                        if cc == 0:
                            if rr < spec['rowspan'] - 1:
                                row_str = s_mid + row_str[2:]
                            else:
                                row_str = s_bot + row_str[2:]
                        if cc == spec['colspan'] - 1:
                            if rr < spec['rowspan'] - 1:
                                row_str = row_str[:-2] + e_mid
                            else:
                                row_str = row_str[:-2] + e_bot
                        _tmp[r + rr][c + cc] = row_str
            _tmp[r][c] = cell_str + _pad(cell_str)
    for r in row_seq[::-1]:
        grid_str += sp.join(_tmp[r]) + '\n'
    if insets:
        grid_str += '\nWith insets:\n'
        for (i_inset, inset) in enumerate(insets):
            r = inset['cell'][0] - 1
            c = inset['cell'][1] - 1
            ref = grid_ref[r][c]
            subplot_labels_str = ','.join(insets_ref[i_inset][0].layout_keys)
            subplot_labels_str = subplot_labels_str.replace('axis', '')
            grid_str += s_str + subplot_labels_str + e_str + ' over ' + s_str + _get_cell_str(r, c, ref) + e_str + '\n'
    return grid_str

def _set_trace_grid_reference(trace, layout, grid_ref, row, col, secondary_y=False):
    if False:
        i = 10
        return i + 15
    if row <= 0:
        raise Exception('Row value is out of range. Note: the starting cell is (1, 1)')
    if col <= 0:
        raise Exception('Col value is out of range. Note: the starting cell is (1, 1)')
    try:
        subplot_refs = grid_ref[row - 1][col - 1]
    except IndexError:
        raise Exception('The (row, col) pair sent is out of range. Use Figure.print_grid to view the subplot grid. ')
    if not subplot_refs:
        raise ValueError('\nNo subplot specified at grid position ({row}, {col})'.format(row=row, col=col))
    if secondary_y:
        if len(subplot_refs) < 2:
            raise ValueError("\nSubplot with type '{subplot_type}' at grid position ({row}, {col}) was not\ncreated with the secondary_y spec property set to True. See the docstring\nfor the specs argument to plotly.subplots.make_subplots for more information.\n")
        trace_kwargs = subplot_refs[1].trace_kwargs
    else:
        trace_kwargs = subplot_refs[0].trace_kwargs
    for k in trace_kwargs:
        if k not in trace:
            raise ValueError("Trace type '{typ}' is not compatible with subplot type '{subplot_type}'\nat grid position ({row}, {col})\n\nSee the docstring for the specs argument to plotly.subplots.make_subplots\nfor more information on subplot types".format(typ=trace.type, subplot_type=subplot_refs[0].subplot_type, row=row, col=col))
    trace.update(trace_kwargs)

def _get_grid_subplot(fig, row, col, secondary_y=False):
    if False:
        return 10
    try:
        grid_ref = fig._grid_ref
    except AttributeError:
        raise Exception('In order to reference traces by row and column, you must first use plotly.tools.make_subplots to create the figure with a subplot grid.')
    rows = len(grid_ref)
    cols = len(grid_ref[0])
    if not isinstance(row, int) or row < 1 or rows < row:
        raise ValueError('\nThe row argument to get_subplot must be an integer where 1 <= row <= {rows}\n    Received value of type {typ}: {val}'.format(rows=rows, typ=type(row), val=repr(row)))
    if not isinstance(col, int) or col < 1 or cols < col:
        raise ValueError('\nThe col argument to get_subplot must be an integer where 1 <= row <= {cols}\n    Received value of type {typ}: {val}'.format(cols=cols, typ=type(col), val=repr(col)))
    subplot_refs = fig._grid_ref[row - 1][col - 1]
    if not subplot_refs:
        return None
    if secondary_y:
        if len(subplot_refs) > 1:
            layout_keys = subplot_refs[1].layout_keys
        else:
            return None
    else:
        layout_keys = subplot_refs[0].layout_keys
    if len(layout_keys) == 0:
        return SubplotDomain(**subplot_refs[0].trace_kwargs['domain'])
    elif len(layout_keys) == 1:
        return fig.layout[layout_keys[0]]
    elif len(layout_keys) == 2:
        return SubplotXY(xaxis=fig.layout[layout_keys[0]], yaxis=fig.layout[layout_keys[1]])
    else:
        raise ValueError('\nUnexpected subplot type with layout_keys of {}'.format(layout_keys))

def _get_subplot_ref_for_trace(trace):
    if False:
        print('Hello World!')
    if 'domain' in trace:
        return SubplotRef(subplot_type='domain', layout_keys=(), trace_kwargs={'domain': {'x': trace.domain.x, 'y': trace.domain.y}})
    elif 'xaxis' in trace and 'yaxis' in trace:
        xaxis_name = 'xaxis' + trace.xaxis[1:] if trace.xaxis else 'xaxis'
        yaxis_name = 'yaxis' + trace.yaxis[1:] if trace.yaxis else 'yaxis'
        return SubplotRef(subplot_type='xy', layout_keys=(xaxis_name, yaxis_name), trace_kwargs={'xaxis': trace.xaxis, 'yaxis': trace.yaxis})
    elif 'geo' in trace:
        return SubplotRef(subplot_type='geo', layout_keys=(trace.geo,), trace_kwargs={'geo': trace.geo})
    elif 'scene' in trace:
        return SubplotRef(subplot_type='scene', layout_keys=(trace.scene,), trace_kwargs={'scene': trace.scene})
    elif 'subplot' in trace:
        for t in _subplot_prop_named_subplot:
            try:
                validator = trace._get_prop_validator('subplot')
                validator.validate_coerce(t)
                return SubplotRef(subplot_type=t, layout_keys=(trace.subplot,), trace_kwargs={'subplot': trace.subplot})
            except ValueError:
                pass
    return None