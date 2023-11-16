from plotly.express._core import build_dataframe
from plotly.express._doc import make_docstring
from plotly.express._chart_types import choropleth_mapbox, scatter_mapbox
import numpy as np
import pandas as pd

def _project_latlon_to_wgs84(lat, lon):
    if False:
        print('Hello World!')
    '\n    Projects lat and lon to WGS84, used to get regular hexagons on a mapbox map\n    '
    x = lon * np.pi / 180
    y = np.arctanh(np.sin(lat * np.pi / 180))
    return (x, y)

def _project_wgs84_to_latlon(x, y):
    if False:
        i = 10
        return i + 15
    '\n    Projects WGS84 to lat and lon, used to get regular hexagons on a mapbox map\n    '
    lon = x * 180 / np.pi
    lat = (2 * np.arctan(np.exp(y)) - np.pi / 2) * 180 / np.pi
    return (lat, lon)

def _getBoundsZoomLevel(lon_min, lon_max, lat_min, lat_max, mapDim):
    if False:
        while True:
            i = 10
    '\n    Get the mapbox zoom level given bounds and a figure dimension\n    Source: https://stackoverflow.com/questions/6048975/google-maps-v3-how-to-calculate-the-zoom-level-for-a-given-bounds\n    '
    scale = 2
    WORLD_DIM = {'height': 256 * scale, 'width': 256 * scale}
    ZOOM_MAX = 18

    def latRad(lat):
        if False:
            for i in range(10):
                print('nop')
        sin = np.sin(lat * np.pi / 180)
        radX2 = np.log((1 + sin) / (1 - sin)) / 2
        return max(min(radX2, np.pi), -np.pi) / 2

    def zoom(mapPx, worldPx, fraction):
        if False:
            i = 10
            return i + 15
        return 0.95 * np.log(mapPx / worldPx / fraction) / np.log(2)
    latFraction = (latRad(lat_max) - latRad(lat_min)) / np.pi
    lngDiff = lon_max - lon_min
    lngFraction = (lngDiff + 360 if lngDiff < 0 else lngDiff) / 360
    latZoom = zoom(mapDim['height'], WORLD_DIM['height'], latFraction)
    lngZoom = zoom(mapDim['width'], WORLD_DIM['width'], lngFraction)
    return min(latZoom, lngZoom, ZOOM_MAX)

def _compute_hexbin(x, y, x_range, y_range, color, nx, agg_func, min_count):
    if False:
        i = 10
        return i + 15
    "\n    Computes the aggregation at hexagonal bin level.\n    Also defines the coordinates of the hexagons for plotting.\n    The binning is inspired by matplotlib's implementation.\n\n    Parameters\n    ----------\n    x : np.ndarray\n        Array of x values (shape N)\n    y : np.ndarray\n        Array of y values (shape N)\n    x_range : np.ndarray\n        Min and max x (shape 2)\n    y_range : np.ndarray\n        Min and max y (shape 2)\n    color : np.ndarray\n        Metric to aggregate at hexagon level (shape N)\n    nx : int\n        Number of hexagons horizontally\n    agg_func : function\n        Numpy compatible aggregator, this function must take a one-dimensional\n        np.ndarray as input and output a scalar\n    min_count : int\n        Minimum number of points in the hexagon for the hexagon to be displayed\n\n    Returns\n    -------\n    np.ndarray\n        X coordinates of each hexagon (shape M x 6)\n    np.ndarray\n        Y coordinates of each hexagon (shape M x 6)\n    np.ndarray\n        Centers of the hexagons (shape M x 2)\n    np.ndarray\n        Aggregated value in each hexagon (shape M)\n\n    "
    xmin = x_range.min()
    xmax = x_range.max()
    ymin = y_range.min()
    ymax = y_range.max()
    padding = 1e-09 * (xmax - xmin)
    xmin -= padding
    xmax += padding
    Dx = xmax - xmin
    Dy = ymax - ymin
    if Dx == 0 and Dy > 0:
        dx = Dy / nx
    elif Dx == 0 and Dy == 0:
        (dx, _) = _project_latlon_to_wgs84(1, 1)
    else:
        dx = Dx / nx
    dy = dx * np.sqrt(3)
    ny = np.ceil(Dy / dy).astype(int)
    ymin -= (ymin + dy * ny - ymax) / 2
    x = (x - xmin) / dx
    y = (y - ymin) / dy
    ix1 = np.round(x).astype(int)
    iy1 = np.round(y).astype(int)
    ix2 = np.floor(x).astype(int)
    iy2 = np.floor(y).astype(int)
    nx1 = nx + 1
    ny1 = ny + 1
    nx2 = nx
    ny2 = ny
    n = nx1 * ny1 + nx2 * ny2
    d1 = (x - ix1) ** 2 + 3.0 * (y - iy1) ** 2
    d2 = (x - ix2 - 0.5) ** 2 + 3.0 * (y - iy2 - 0.5) ** 2
    bdist = d1 < d2
    if color is None:
        lattice1 = np.zeros((nx1, ny1))
        lattice2 = np.zeros((nx2, ny2))
        c1 = (0 <= ix1) & (ix1 < nx1) & (0 <= iy1) & (iy1 < ny1) & bdist
        c2 = (0 <= ix2) & (ix2 < nx2) & (0 <= iy2) & (iy2 < ny2) & ~bdist
        np.add.at(lattice1, (ix1[c1], iy1[c1]), 1)
        np.add.at(lattice2, (ix2[c2], iy2[c2]), 1)
        if min_count is not None:
            lattice1[lattice1 < min_count] = np.nan
            lattice2[lattice2 < min_count] = np.nan
        accum = np.concatenate([lattice1.ravel(), lattice2.ravel()])
        good_idxs = ~np.isnan(accum)
    else:
        if min_count is None:
            min_count = 1
        lattice1 = np.empty((nx1, ny1), dtype=object)
        for i in range(nx1):
            for j in range(ny1):
                lattice1[i, j] = []
        lattice2 = np.empty((nx2, ny2), dtype=object)
        for i in range(nx2):
            for j in range(ny2):
                lattice2[i, j] = []
        for i in range(len(x)):
            if bdist[i]:
                if 0 <= ix1[i] < nx1 and 0 <= iy1[i] < ny1:
                    lattice1[ix1[i], iy1[i]].append(color[i])
            elif 0 <= ix2[i] < nx2 and 0 <= iy2[i] < ny2:
                lattice2[ix2[i], iy2[i]].append(color[i])
        for i in range(nx1):
            for j in range(ny1):
                vals = lattice1[i, j]
                if len(vals) >= min_count:
                    lattice1[i, j] = agg_func(vals)
                else:
                    lattice1[i, j] = np.nan
        for i in range(nx2):
            for j in range(ny2):
                vals = lattice2[i, j]
                if len(vals) >= min_count:
                    lattice2[i, j] = agg_func(vals)
                else:
                    lattice2[i, j] = np.nan
        accum = np.hstack((lattice1.astype(float).ravel(), lattice2.astype(float).ravel()))
        good_idxs = ~np.isnan(accum)
    agreggated_value = accum[good_idxs]
    centers = np.zeros((n, 2), float)
    centers[:nx1 * ny1, 0] = np.repeat(np.arange(nx1), ny1)
    centers[:nx1 * ny1, 1] = np.tile(np.arange(ny1), nx1)
    centers[nx1 * ny1:, 0] = np.repeat(np.arange(nx2) + 0.5, ny2)
    centers[nx1 * ny1:, 1] = np.tile(np.arange(ny2), nx2) + 0.5
    centers[:, 0] *= dx
    centers[:, 1] *= dy
    centers[:, 0] += xmin
    centers[:, 1] += ymin
    centers = centers[good_idxs]
    hx = [0, 0.5, 0.5, 0, -0.5, -0.5]
    hy = [-0.5 / np.cos(np.pi / 6), -0.5 * np.tan(np.pi / 6), 0.5 * np.tan(np.pi / 6), 0.5 / np.cos(np.pi / 6), 0.5 * np.tan(np.pi / 6), -0.5 * np.tan(np.pi / 6)]
    m = len(centers)
    hxs = np.array([hx] * m) * dx + np.vstack(centers[:, 0])
    hys = np.array([hy] * m) * dy / np.sqrt(3) + np.vstack(centers[:, 1])
    return (hxs, hys, centers, agreggated_value)

def _compute_wgs84_hexbin(lat=None, lon=None, lat_range=None, lon_range=None, color=None, nx=None, agg_func=None, min_count=None):
    if False:
        print('Hello World!')
    '\n    Computes the lat-lon aggregation at hexagonal bin level.\n    Latitude and longitude need to be projected to WGS84 before aggregating\n    in order to display regular hexagons on the map.\n\n    Parameters\n    ----------\n    lat : np.ndarray\n        Array of latitudes (shape N)\n    lon : np.ndarray\n        Array of longitudes (shape N)\n    lat_range : np.ndarray\n        Min and max latitudes (shape 2)\n    lon_range : np.ndarray\n        Min and max longitudes (shape 2)\n    color : np.ndarray\n        Metric to aggregate at hexagon level (shape N)\n    nx : int\n        Number of hexagons horizontally\n    agg_func : function\n        Numpy compatible aggregator, this function must take a one-dimensional\n        np.ndarray as input and output a scalar\n    min_count : int\n        Minimum number of points in the hexagon for the hexagon to be displayed\n\n    Returns\n    -------\n    np.ndarray\n        Lat coordinates of each hexagon (shape M x 6)\n    np.ndarray\n        Lon coordinates of each hexagon (shape M x 6)\n    pd.Series\n        Unique id for each hexagon, to be used in the geojson data (shape M)\n    np.ndarray\n        Aggregated value in each hexagon (shape M)\n\n    '
    (x, y) = _project_latlon_to_wgs84(lat, lon)
    if lat_range is None:
        lat_range = np.array([lat.min(), lat.max()])
    if lon_range is None:
        lon_range = np.array([lon.min(), lon.max()])
    (x_range, y_range) = _project_latlon_to_wgs84(lat_range, lon_range)
    (hxs, hys, centers, agreggated_value) = _compute_hexbin(x, y, x_range, y_range, color, nx, agg_func, min_count)
    (hexagons_lats, hexagons_lons) = _project_wgs84_to_latlon(hxs, hys)
    centers = centers.astype(str)
    hexagons_ids = pd.Series(centers[:, 0]) + ',' + pd.Series(centers[:, 1])
    return (hexagons_lats, hexagons_lons, hexagons_ids, agreggated_value)

def _hexagons_to_geojson(hexagons_lats, hexagons_lons, ids=None):
    if False:
        i = 10
        return i + 15
    '\n    Creates a geojson of hexagonal features based on the outputs of\n    _compute_wgs84_hexbin\n    '
    features = []
    if ids is None:
        ids = np.arange(len(hexagons_lats))
    for (lat, lon, idx) in zip(hexagons_lats, hexagons_lons, ids):
        points = np.array([lon, lat]).T.tolist()
        points.append(points[0])
        features.append(dict(type='Feature', id=idx, geometry=dict(type='Polygon', coordinates=[points])))
    return dict(type='FeatureCollection', features=features)

def create_hexbin_mapbox(data_frame=None, lat=None, lon=None, color=None, nx_hexagon=5, agg_func=None, animation_frame=None, color_discrete_sequence=None, color_discrete_map={}, labels={}, color_continuous_scale=None, range_color=None, color_continuous_midpoint=None, opacity=None, zoom=None, center=None, mapbox_style=None, title=None, template=None, width=None, height=None, min_count=None, show_original_data=False, original_data_marker=None):
    if False:
        print('Hello World!')
    '\n    Returns a figure aggregating scattered points into connected hexagons\n    '
    args = build_dataframe(args=locals(), constructor=None)
    if agg_func is None:
        agg_func = np.mean
    lat_range = args['data_frame'][args['lat']].agg(['min', 'max']).values
    lon_range = args['data_frame'][args['lon']].agg(['min', 'max']).values
    (hexagons_lats, hexagons_lons, hexagons_ids, count) = _compute_wgs84_hexbin(lat=args['data_frame'][args['lat']].values, lon=args['data_frame'][args['lon']].values, lat_range=lat_range, lon_range=lon_range, color=None, nx=nx_hexagon, agg_func=agg_func, min_count=min_count)
    geojson = _hexagons_to_geojson(hexagons_lats, hexagons_lons, hexagons_ids)
    if zoom is None:
        if height is None and width is None:
            mapDim = dict(height=450, width=450)
        elif height is None and width is not None:
            mapDim = dict(height=450, width=width)
        elif height is not None and width is None:
            mapDim = dict(height=height, width=height)
        else:
            mapDim = dict(height=height, width=width)
        zoom = _getBoundsZoomLevel(lon_range[0], lon_range[1], lat_range[0], lat_range[1], mapDim)
    if center is None:
        center = dict(lat=lat_range.mean(), lon=lon_range.mean())
    if args['animation_frame'] is not None:
        groups = args['data_frame'].groupby(args['animation_frame']).groups
    else:
        groups = {0: args['data_frame'].index}
    agg_data_frame_list = []
    for (frame, index) in groups.items():
        df = args['data_frame'].loc[index]
        (_, _, hexagons_ids, aggregated_value) = _compute_wgs84_hexbin(lat=df[args['lat']].values, lon=df[args['lon']].values, lat_range=lat_range, lon_range=lon_range, color=df[args['color']].values if args['color'] else None, nx=nx_hexagon, agg_func=agg_func, min_count=min_count)
        agg_data_frame_list.append(pd.DataFrame(np.c_[hexagons_ids, aggregated_value], columns=['locations', 'color']))
    agg_data_frame = pd.concat(agg_data_frame_list, axis=0, keys=groups.keys()).rename_axis(index=('frame', 'index')).reset_index('frame')
    agg_data_frame['color'] = pd.to_numeric(agg_data_frame['color'])
    if range_color is None:
        range_color = [agg_data_frame['color'].min(), agg_data_frame['color'].max()]
    fig = choropleth_mapbox(data_frame=agg_data_frame, geojson=geojson, locations='locations', color='color', hover_data={'color': True, 'locations': False, 'frame': False}, animation_frame='frame' if args['animation_frame'] is not None else None, color_discrete_sequence=color_discrete_sequence, color_discrete_map=color_discrete_map, labels=labels, color_continuous_scale=color_continuous_scale, range_color=range_color, color_continuous_midpoint=color_continuous_midpoint, opacity=opacity, zoom=zoom, center=center, mapbox_style=mapbox_style, title=title, template=template, width=width, height=height)
    if show_original_data:
        original_fig = scatter_mapbox(data_frame=args['data_frame'].sort_values(by=args['animation_frame']) if args['animation_frame'] is not None else args['data_frame'], lat=args['lat'], lon=args['lon'], animation_frame=args['animation_frame'])
        original_fig.data[0].hoverinfo = 'skip'
        original_fig.data[0].hovertemplate = None
        original_fig.data[0].marker = original_data_marker
        fig.add_trace(original_fig.data[0])
        if args['animation_frame'] is not None:
            for i in range(len(original_fig.frames)):
                original_fig.frames[i].data[0].hoverinfo = 'skip'
                original_fig.frames[i].data[0].hovertemplate = None
                original_fig.frames[i].data[0].marker = original_data_marker
                fig.frames[i].data = [fig.frames[i].data[0], original_fig.frames[i].data[0]]
    return fig
create_hexbin_mapbox.__doc__ = make_docstring(create_hexbin_mapbox, override_dict=dict(nx_hexagon=['int', 'Number of hexagons (horizontally) to be created'], agg_func=['function', 'Numpy array aggregator, it must take as input a 1D array', 'and output a scalar value.'], min_count=['int', 'Minimum number of points in a hexagon for it to be displayed.', 'If None and color is not set, display all hexagons.', 'If None and color is set, only display hexagons that contain points.'], show_original_data=['bool', 'Whether to show the original data on top of the hexbin aggregation.'], original_data_marker=['dict', 'Scattermapbox marker options.']))