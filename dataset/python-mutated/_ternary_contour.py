import plotly.colors as clrs
from plotly.graph_objs import graph_objs as go
from plotly import exceptions
from plotly import optional_imports
np = optional_imports.get_module('numpy')
scipy_interp = optional_imports.get_module('scipy.interpolate')
from skimage import measure

def _ternary_layout(title='Ternary contour plot', width=550, height=525, pole_labels=['a', 'b', 'c']):
    if False:
        for i in range(10):
            print('nop')
    "\n    Layout of ternary contour plot, to be passed to ``go.FigureWidget``\n    object.\n\n    Parameters\n    ==========\n    title : str or None\n        Title of ternary plot\n    width : int\n        Figure width.\n    height : int\n        Figure height.\n    pole_labels : str, default ['a', 'b', 'c']\n        Names of the three poles of the triangle.\n    "
    return dict(title=title, width=width, height=height, ternary=dict(sum=1, aaxis=dict(title=dict(text=pole_labels[0]), min=0.01, linewidth=2, ticks='outside'), baxis=dict(title=dict(text=pole_labels[1]), min=0.01, linewidth=2, ticks='outside'), caxis=dict(title=dict(text=pole_labels[2]), min=0.01, linewidth=2, ticks='outside')), showlegend=False)

def _replace_zero_coords(ternary_data, delta=0.0005):
    if False:
        while True:
            i = 10
    '\n    Replaces zero ternary coordinates with delta and normalize the new\n    triplets (a, b, c).\n\n    Parameters\n    ----------\n\n    ternary_data : ndarray of shape (N, 3)\n\n    delta : float\n        Small float to regularize logarithm.\n\n    Notes\n    -----\n    Implements a method\n    by J. A. Martin-Fernandez,  C. Barcelo-Vidal, V. Pawlowsky-Glahn,\n    Dealing with zeros and missing values in compositional data sets\n    using nonparametric imputation, Mathematical Geology 35 (2003),\n    pp 253-278.\n    '
    zero_mask = ternary_data == 0
    is_any_coord_zero = np.any(zero_mask, axis=0)
    unity_complement = 1 - delta * is_any_coord_zero
    if np.any(unity_complement) < 0:
        raise ValueError('The provided value of delta led to negativeternary coords.Set a smaller delta')
    ternary_data = np.where(zero_mask, delta, unity_complement * ternary_data)
    return ternary_data

def _ilr_transform(barycentric):
    if False:
        return 10
    '\n    Perform Isometric Log-Ratio on barycentric (compositional) data.\n\n    Parameters\n    ----------\n    barycentric: ndarray of shape (3, N)\n        Barycentric coordinates.\n\n    References\n    ----------\n    "An algebraic method to compute isometric logratio transformation and\n    back transformation of compositional data", Jarauta-Bragulat, E.,\n    Buenestado, P.; Hervada-Sala, C., in Proc. of the Annual Conf. of the\n    Intl Assoc for Math Geology, 2003, pp 31-30.\n    '
    barycentric = np.asarray(barycentric)
    x_0 = np.log(barycentric[0] / barycentric[1]) / np.sqrt(2)
    x_1 = 1.0 / np.sqrt(6) * np.log(barycentric[0] * barycentric[1] / barycentric[2] ** 2)
    ilr_tdata = np.stack((x_0, x_1))
    return ilr_tdata

def _ilr_inverse(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Perform inverse Isometric Log-Ratio (ILR) transform to retrieve\n    barycentric (compositional) data.\n\n    Parameters\n    ----------\n    x : array of shape (2, N)\n        Coordinates in ILR space.\n\n    References\n    ----------\n    "An algebraic method to compute isometric logratio transformation and\n    back transformation of compositional data", Jarauta-Bragulat, E.,\n    Buenestado, P.; Hervada-Sala, C., in Proc. of the Annual Conf. of the\n    Intl Assoc for Math Geology, 2003, pp 31-30.\n    '
    x = np.array(x)
    matrix = np.array([[0.5, 1, 1.0], [-0.5, 1, 1.0], [0.0, 0.0, 1.0]])
    s = np.sqrt(2) / 2
    t = np.sqrt(3 / 2)
    Sk = np.einsum('ik, kj -> ij', np.array([[s, t], [-s, t]]), x)
    Z = -np.log(1 + np.exp(Sk).sum(axis=0))
    log_barycentric = np.einsum('ik, kj -> ij', matrix, np.stack((2 * s * x[0], t * x[1], Z)))
    iilr_tdata = np.exp(log_barycentric)
    return iilr_tdata

def _transform_barycentric_cartesian():
    if False:
        return 10
    '\n    Returns the transformation matrix from barycentric to Cartesian\n    coordinates and conversely.\n    '
    tri_verts = np.array([[0.5, np.sqrt(3) / 2], [0, 0], [1, 0]])
    M = np.array([tri_verts[:, 0], tri_verts[:, 1], np.ones(3)])
    return (M, np.linalg.inv(M))

def _prepare_barycentric_coord(b_coords):
    if False:
        while True:
            i = 10
    '\n    Check ternary coordinates and return the right barycentric coordinates.\n    '
    if not isinstance(b_coords, (list, np.ndarray)):
        raise ValueError('Data  should be either an array of shape (n,m),or a list of n m-lists, m=2 or 3')
    b_coords = np.asarray(b_coords)
    if b_coords.shape[0] not in (2, 3):
        raise ValueError('A point should have  2 (a, b) or 3 (a, b, c)barycentric coordinates')
    if len(b_coords) == 3 and (not np.allclose(b_coords.sum(axis=0), 1, rtol=0.01)) and (not np.allclose(b_coords.sum(axis=0), 100, rtol=0.01)):
        msg = 'The sum of coordinates should be 1 or 100 for all data points'
        raise ValueError(msg)
    if len(b_coords) == 2:
        (A, B) = b_coords
        C = 1 - (A + B)
    else:
        (A, B, C) = b_coords / b_coords.sum(axis=0)
    if np.any(np.stack((A, B, C)) < 0):
        raise ValueError('Barycentric coordinates should be positive.')
    return np.stack((A, B, C))

def _compute_grid(coordinates, values, interp_mode='ilr'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Transform data points with Cartesian or ILR mapping, then Compute\n    interpolation on a regular grid.\n\n    Parameters\n    ==========\n\n    coordinates : array-like\n        Barycentric coordinates of data points.\n    values : 1-d array-like\n        Data points, field to be represented as contours.\n    interp_mode : 'ilr' (default) or 'cartesian'\n        Defines how data are interpolated to compute contours.\n    "
    if interp_mode == 'cartesian':
        (M, invM) = _transform_barycentric_cartesian()
        coord_points = np.einsum('ik, kj -> ij', M, coordinates)
    elif interp_mode == 'ilr':
        coordinates = _replace_zero_coords(coordinates)
        coord_points = _ilr_transform(coordinates)
    else:
        raise ValueError('interp_mode should be cartesian or ilr')
    (xx, yy) = coord_points[:2]
    (x_min, x_max) = (xx.min(), xx.max())
    (y_min, y_max) = (yy.min(), yy.max())
    n_interp = max(200, int(np.sqrt(len(values))))
    gr_x = np.linspace(x_min, x_max, n_interp)
    gr_y = np.linspace(y_min, y_max, n_interp)
    (grid_x, grid_y) = np.meshgrid(gr_x, gr_y)
    grid_z = scipy_interp.griddata(coord_points[:2].T, values, (grid_x, grid_y), method='cubic')
    grid_z_other = scipy_interp.griddata(coord_points[:2].T, values, (grid_x, grid_y), method='nearest')
    return (grid_z, gr_x, gr_y)

def _polygon_area(x, y):
    if False:
        print('Hello World!')
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def _colors(ncontours, colormap=None):
    if False:
        return 10
    '\n    Return a list of ``ncontours`` colors from the ``colormap`` colorscale.\n    '
    if colormap in clrs.PLOTLY_SCALES.keys():
        cmap = clrs.PLOTLY_SCALES[colormap]
    else:
        raise exceptions.PlotlyError('Colorscale must be a valid Plotly Colorscale.The available colorscale names are {}'.format(clrs.PLOTLY_SCALES.keys()))
    values = np.linspace(0, 1, ncontours)
    vals_cmap = np.array([pair[0] for pair in cmap])
    cols = np.array([pair[1] for pair in cmap])
    inds = np.searchsorted(vals_cmap, values)
    if '#' in cols[0]:
        cols = [clrs.label_rgb(clrs.hex_to_rgb(col)) for col in cols]
    colors = [cols[0]]
    for (ind, val) in zip(inds[1:], values[1:]):
        (val1, val2) = (vals_cmap[ind - 1], vals_cmap[ind])
        interm = (val - val1) / (val2 - val1)
        col = clrs.find_intermediate_color(cols[ind - 1], cols[ind], interm, colortype='rgb')
        colors.append(col)
    return colors

def _is_invalid_contour(x, y):
    if False:
        return 10
    '\n    Utility function for _contour_trace\n\n    Contours with an area of the order as 1 pixel are considered spurious.\n    '
    too_small = np.all(np.abs(x - x[0]) < 2) and np.all(np.abs(y - y[0]) < 2)
    return too_small

def _extract_contours(im, values, colors):
    if False:
        for i in range(10):
            print('nop')
    '\n    Utility function for _contour_trace.\n\n    In ``im`` only one part of the domain has valid values (corresponding\n    to a subdomain where barycentric coordinates are well defined). When\n    computing contours, we need to assign values outside of this domain.\n    We can choose a value either smaller than all the values inside the\n    valid domain, or larger. This value must be chose with caution so that\n    no spurious contours are added. For example, if the boundary of the valid\n    domain has large values and the outer value is set to a small one, all\n    intermediate contours will be added at the boundary.\n\n    Therefore, we compute the two sets of contours (with an outer value\n    smaller of larger than all values in the valid domain), and choose\n    the value resulting in a smaller total number of contours. There might\n    be a faster way to do this, but it works...\n    '
    mask_nan = np.isnan(im)
    (im_min, im_max) = (im[np.logical_not(mask_nan)].min(), im[np.logical_not(mask_nan)].max())
    zz_min = np.copy(im)
    zz_min[mask_nan] = 2 * im_min
    zz_max = np.copy(im)
    zz_max[mask_nan] = 2 * im_max
    (all_contours1, all_values1, all_areas1, all_colors1) = ([], [], [], [])
    (all_contours2, all_values2, all_areas2, all_colors2) = ([], [], [], [])
    for (i, val) in enumerate(values):
        contour_level1 = measure.find_contours(zz_min, val)
        contour_level2 = measure.find_contours(zz_max, val)
        all_contours1.extend(contour_level1)
        all_contours2.extend(contour_level2)
        all_values1.extend([val] * len(contour_level1))
        all_values2.extend([val] * len(contour_level2))
        all_areas1.extend([_polygon_area(contour.T[1], contour.T[0]) for contour in contour_level1])
        all_areas2.extend([_polygon_area(contour.T[1], contour.T[0]) for contour in contour_level2])
        all_colors1.extend([colors[i]] * len(contour_level1))
        all_colors2.extend([colors[i]] * len(contour_level2))
    if len(all_contours1) <= len(all_contours2):
        return (all_contours1, all_values1, all_areas1, all_colors1)
    else:
        return (all_contours2, all_values2, all_areas2, all_colors2)

def _add_outer_contour(all_contours, all_values, all_areas, all_colors, values, val_outer, v_min, v_max, colors, color_min, color_max):
    if False:
        print('Hello World!')
    '\n    Utility function for _contour_trace\n\n    Adds the background color to fill gaps outside of computed contours.\n\n    To compute the background color, the color of the contour with largest\n    area (``val_outer``) is used. As background color, we choose the next\n    color value in the direction of the extrema of the colormap.\n\n    Then we add information for the outer contour for the different lists\n    provided as arguments.\n\n    A discrete colormap with all used colors is also returned (to be used\n    by colorscale trace).\n    '
    outer_contour = 20 * np.array([[0, 0, 1], [0, 1, 0.5]]).T
    all_contours = [outer_contour] + all_contours
    delta_values = np.diff(values)[0]
    values = np.concatenate(([values[0] - delta_values], values, [values[-1] + delta_values]))
    colors = np.concatenate(([color_min], colors, [color_max]))
    index = np.nonzero(values == val_outer)[0][0]
    if index < len(values) / 2:
        index -= 1
    else:
        index += 1
    all_colors = [colors[index]] + all_colors
    all_values = [values[index]] + all_values
    all_areas = [0] + all_areas
    used_colors = [color for color in colors if color in all_colors]
    color_number = len(used_colors)
    scale = np.linspace(0, 1, color_number + 1)
    discrete_cm = []
    for (i, color) in enumerate(used_colors):
        discrete_cm.append([scale[i], used_colors[i]])
        discrete_cm.append([scale[i + 1], used_colors[i]])
    discrete_cm.append([scale[color_number], used_colors[color_number - 1]])
    return (all_contours, all_values, all_areas, all_colors, discrete_cm)

def _contour_trace(x, y, z, ncontours=None, colorscale='Electric', linecolor='rgb(150,150,150)', interp_mode='llr', coloring=None, v_min=0, v_max=1):
    if False:
        for i in range(10):
            print('nop')
    "\n    Contour trace in Cartesian coordinates.\n\n    Parameters\n    ==========\n\n    x, y : array-like\n        Cartesian coordinates\n    z : array-like\n        Field to be represented as contours.\n    ncontours : int or None\n        Number of contours to display (determined automatically if None).\n    colorscale : None or str (Plotly colormap)\n        colorscale of the contours.\n    linecolor : rgb color\n        Color used for lines. If ``colorscale`` is not None, line colors are\n        determined from ``colorscale`` instead.\n    interp_mode : 'ilr' (default) or 'cartesian'\n        Defines how data are interpolated to compute contours. If 'irl',\n        ILR (Isometric Log-Ratio) of compositional data is performed. If\n        'cartesian', contours are determined in Cartesian space.\n    coloring : None or 'lines'\n        How to display contour. Filled contours if None, lines if ``lines``.\n    vmin, vmax : float\n        Bounds of interval of values used for the colorspace\n\n    Notes\n    =====\n    "
    colors = _colors(ncontours + 2, colorscale)
    values = np.linspace(v_min, v_max, ncontours + 2)
    (color_min, color_max) = (colors[0], colors[-1])
    colors = colors[1:-1]
    values = values[1:-1]
    if linecolor is None:
        linecolor = 'rgb(150, 150, 150)'
    else:
        colors = [linecolor] * ncontours
    (all_contours, all_values, all_areas, all_colors) = _extract_contours(z, values, colors)
    order = np.argsort(all_areas)[::-1]
    (all_contours, all_values, all_areas, all_colors, discrete_cm) = _add_outer_contour(all_contours, all_values, all_areas, all_colors, values, all_values[order[0]], v_min, v_max, colors, color_min, color_max)
    order = np.concatenate(([0], order + 1))
    traces = []
    (M, invM) = _transform_barycentric_cartesian()
    dx = (x.max() - x.min()) / x.size
    dy = (y.max() - y.min()) / y.size
    for index in order:
        (y_contour, x_contour) = all_contours[index].T
        val = all_values[index]
        if interp_mode == 'cartesian':
            bar_coords = np.dot(invM, np.stack((dx * x_contour, dy * y_contour, np.ones(x_contour.shape))))
        elif interp_mode == 'ilr':
            bar_coords = _ilr_inverse(np.stack((dx * x_contour + x.min(), dy * y_contour + y.min())))
        if index == 0:
            a = np.array([1, 0, 0])
            b = np.array([0, 1, 0])
            c = np.array([0, 0, 1])
        else:
            (a, b, c) = bar_coords
        if _is_invalid_contour(x_contour, y_contour):
            continue
        _col = all_colors[index] if coloring == 'lines' else linecolor
        trace = dict(type='scatterternary', a=a, b=b, c=c, mode='lines', line=dict(color=_col, shape='spline', width=1), fill='toself', fillcolor=all_colors[index], showlegend=True, hoverinfo='skip', name='%.3f' % val)
        if coloring == 'lines':
            trace['fill'] = None
        traces.append(trace)
    return (traces, discrete_cm)

def create_ternary_contour(coordinates, values, pole_labels=['a', 'b', 'c'], width=500, height=500, ncontours=None, showscale=False, coloring=None, colorscale='Bluered', linecolor=None, title=None, interp_mode='ilr', showmarkers=False):
    if False:
        print('Hello World!')
    "\n    Ternary contour plot.\n\n    Parameters\n    ----------\n\n    coordinates : list or ndarray\n        Barycentric coordinates of shape (2, N) or (3, N) where N is the\n        number of data points. The sum of the 3 coordinates is expected\n        to be 1 for all data points.\n    values : array-like\n        Data points of field to be represented as contours.\n    pole_labels : str, default ['a', 'b', 'c']\n        Names of the three poles of the triangle.\n    width : int\n        Figure width.\n    height : int\n        Figure height.\n    ncontours : int or None\n        Number of contours to display (determined automatically if None).\n    showscale : bool, default False\n        If True, a colorbar showing the color scale is displayed.\n    coloring : None or 'lines'\n        How to display contour. Filled contours if None, lines if ``lines``.\n    colorscale : None or str (Plotly colormap)\n        colorscale of the contours.\n    linecolor : None or rgb color\n        Color used for lines. ``colorscale`` has to be set to None, otherwise\n        line colors are determined from ``colorscale``.\n    title : str or None\n        Title of ternary plot\n    interp_mode : 'ilr' (default) or 'cartesian'\n        Defines how data are interpolated to compute contours. If 'irl',\n        ILR (Isometric Log-Ratio) of compositional data is performed. If\n        'cartesian', contours are determined in Cartesian space.\n    showmarkers : bool, default False\n        If True, markers corresponding to input compositional points are\n        superimposed on contours, using the same colorscale.\n\n    Examples\n    ========\n\n    Example 1: ternary contour plot with filled contours\n\n    >>> import plotly.figure_factory as ff\n    >>> import numpy as np\n    >>> # Define coordinates\n    >>> a, b = np.mgrid[0:1:20j, 0:1:20j]\n    >>> mask = a + b <= 1\n    >>> a = a[mask].ravel()\n    >>> b = b[mask].ravel()\n    >>> c = 1 - a - b\n    >>> # Values to be displayed as contours\n    >>> z = a * b * c\n    >>> fig = ff.create_ternary_contour(np.stack((a, b, c)), z)\n    >>> fig.show()\n\n    It is also possible to give only two barycentric coordinates for each\n    point, since the sum of the three coordinates is one:\n\n    >>> fig = ff.create_ternary_contour(np.stack((a, b)), z)\n\n\n    Example 2: ternary contour plot with line contours\n\n    >>> fig = ff.create_ternary_contour(np.stack((a, b, c)), z, coloring='lines')\n\n    Example 3: customize number of contours\n\n    >>> fig = ff.create_ternary_contour(np.stack((a, b, c)), z, ncontours=8)\n\n    Example 4: superimpose contour plot and original data as markers\n\n    >>> fig = ff.create_ternary_contour(np.stack((a, b, c)), z, coloring='lines',\n    ...                                 showmarkers=True)\n\n    Example 5: customize title and pole labels\n\n    >>> fig = ff.create_ternary_contour(np.stack((a, b, c)), z,\n    ...                                 title='Ternary plot',\n    ...                                 pole_labels=['clay', 'quartz', 'fledspar'])\n    "
    if scipy_interp is None:
        raise ImportError('    The create_ternary_contour figure factory requires the scipy package')
    sk_measure = optional_imports.get_module('skimage')
    if sk_measure is None:
        raise ImportError('    The create_ternary_contour figure factory requires the scikit-image\n    package')
    if colorscale is None:
        showscale = False
    if ncontours is None:
        ncontours = 5
    coordinates = _prepare_barycentric_coord(coordinates)
    (v_min, v_max) = (values.min(), values.max())
    (grid_z, gr_x, gr_y) = _compute_grid(coordinates, values, interp_mode=interp_mode)
    layout = _ternary_layout(pole_labels=pole_labels, width=width, height=height, title=title)
    (contour_trace, discrete_cm) = _contour_trace(gr_x, gr_y, grid_z, ncontours=ncontours, colorscale=colorscale, linecolor=linecolor, interp_mode=interp_mode, coloring=coloring, v_min=v_min, v_max=v_max)
    fig = go.Figure(data=contour_trace, layout=layout)
    opacity = 1 if showmarkers else 0
    (a, b, c) = coordinates
    hovertemplate = pole_labels[0] + ': %{a:.3f}<br>' + pole_labels[1] + ': %{b:.3f}<br>' + pole_labels[2] + ': %{c:.3f}<br>z: %{marker.color:.3f}<extra></extra>'
    fig.add_scatterternary(a=a, b=b, c=c, mode='markers', marker={'color': values, 'colorscale': colorscale, 'line': {'color': 'rgb(120, 120, 120)', 'width': int(coloring != 'lines')}}, opacity=opacity, hovertemplate=hovertemplate)
    if showscale:
        if not showmarkers:
            colorscale = discrete_cm
        colorbar = dict({'type': 'scatterternary', 'a': [None], 'b': [None], 'c': [None], 'marker': {'cmin': values.min(), 'cmax': values.max(), 'colorscale': colorscale, 'showscale': True}, 'mode': 'markers'})
        fig.add_trace(colorbar)
    return fig