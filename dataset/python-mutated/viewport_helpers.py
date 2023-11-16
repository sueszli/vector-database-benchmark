"""
Functions that make it easier to provide a default centering
for a view state
"""
import math
from ..bindings.view_state import ViewState
from .type_checking import is_pandas_df

def _squared_diff(x, x0):
    if False:
        return 10
    return (x0 - x) * (x0 - x)

def euclidean(y, y1):
    if False:
        while True:
            i = 10
    'Euclidean distance in n-dimensions\n\n    Parameters\n    ----------\n    y : tuple of float\n        A point in n-dimensions\n    y1 : tuple of float\n        A point in n-dimensions\n\n    Examples\n    --------\n    >>> EPSILON = 0.001\n    >>> euclidean((3, 6, 5), (7, -5, 1)) - 12.369 < EPSILON\n    True\n    '
    if not len(y) == len(y1):
        raise Exception('Input coordinates must be of the same length')
    return math.sqrt(sum([_squared_diff(x, x0) for (x, x0) in zip(y, y1)]))

def geometric_mean(points):
    if False:
        print('Hello World!')
    'Gets centroid in a series of points\n\n    Parameters\n    ----------\n    points : list of list of float\n        List of (x, y) coordinates\n\n    Returns\n    -------\n    tuple\n        The centroid of a list of points\n    '
    avg_x = sum([float(p[0]) for p in points]) / len(points)
    avg_y = sum([float(p[1]) for p in points]) / len(points)
    return (avg_x, avg_y)

def get_bbox(points):
    if False:
        i = 10
        return i + 15
    'Get the bounding box around the data,\n\n    Parameters\n    ----------\n    points : list of list of float\n        List of (x, y) coordinates\n\n    Returns\n    -------\n    dict\n        Dictionary containing the top left and bottom right points of a bounding box\n    '
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    max_x = max(xs)
    max_y = max(ys)
    min_x = min(xs)
    min_y = min(ys)
    return ((min_x, max_y), (max_x, min_y))

def k_nearest_neighbors(points, center, k):
    if False:
        i = 10
        return i + 15
    'Gets the k furthest points from the center\n\n    Parameters\n    ----------\n    points : list of list of float\n        List of (x, y) coordinates\n    center : list of list of float\n        Center point\n    k : int\n        Number of points\n\n    Returns\n    -------\n    list\n        Index of the k furthest points\n\n    Todo\n    ---\n    Currently implemently naively, needs to be more efficient\n    '
    pts_with_distance = [(pt, euclidean(pt, center)) for pt in points]
    sorted_pts = sorted(pts_with_distance, key=lambda x: x[1])
    return [x[0] for x in sorted_pts][:int(k)]

def get_n_pct(points, proportion=1):
    if False:
        i = 10
        return i + 15
    'Computes the bounding box of the maximum zoom for the specified list of points\n\n    Parameters\n    ----------\n    points : list of list of float\n        List of (x, y) coordinates\n    proportion : float, default 1\n        Value between 0 and 1 representing the minimum proportion of data to be captured\n\n    Returns\n    -------\n    list\n        k nearest data points\n    '
    if proportion == 1:
        return points
    centroid = geometric_mean(points)
    n_to_keep = math.floor(proportion * len(points))
    return k_nearest_neighbors(points, centroid, n_to_keep)

def bbox_to_zoom_level(bbox):
    if False:
        return 10
    'Computes the zoom level of a lat/lng bounding box\n\n    Parameters\n    ----------\n    bbox : list of list of float\n        Northwest and southeast corners of a bounding box, given as two points in a list\n\n    Returns\n    -------\n    int\n        Zoom level of map in a WGS84 Mercator projection (e.g., like that of Google Maps)\n    '
    lat_diff = max(bbox[0][0], bbox[1][0]) - min(bbox[0][0], bbox[1][0])
    lng_diff = max(bbox[0][1], bbox[1][1]) - min(bbox[0][1], bbox[1][1])
    max_diff = max(lng_diff, lat_diff)
    zoom_level = None
    if max_diff < 360.0 / math.pow(2, 20):
        zoom_level = 21
    else:
        zoom_level = int(-1 * (math.log(max_diff) / math.log(2.0) - math.log(360.0) / math.log(2)))
        if zoom_level < 1:
            zoom_level = 1
    return zoom_level

def compute_view(points, view_proportion=1, view_type=ViewState):
    if False:
        print('Hello World!')
    'Automatically computes a zoom level for the points passed in.\n\n    Parameters\n    ----------\n    points : list of list of float or pandas.DataFrame\n        A list of points\n    view_propotion : float, default 1\n        Proportion of the data that is meaningful to plot\n    view_type : class constructor for pydeck.ViewState, default :class:`pydeck.bindings.view_state.ViewState`\n        Class constructor for a viewport. In the current version of pydeck,\n        users most likely do not have to modify this attribute.\n\n    Returns\n    -------\n    pydeck.Viewport\n        Viewport fitted to the data\n    '
    if is_pandas_df(points):
        points = points.to_records(index=False)
    bbox = get_bbox(get_n_pct(points, view_proportion))
    zoom = bbox_to_zoom_level(bbox)
    center = geometric_mean(points)
    instance = view_type(latitude=center[1], longitude=center[0], zoom=zoom)
    return instance