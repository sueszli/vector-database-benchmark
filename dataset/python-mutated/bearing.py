"""Calculate graph edge bearings."""
from warnings import warn
import networkx as nx
import numpy as np
from . import plot
from . import projection
try:
    import scipy
except ImportError:
    scipy = None

def calculate_bearing(lat1, lon1, lat2, lon2):
    if False:
        i = 10
        return i + 15
    "\n    Calculate the compass bearing(s) between pairs of lat-lon points.\n\n    Vectorized function to calculate initial bearings between two points'\n    coordinates or between arrays of points' coordinates. Expects coordinates\n    in decimal degrees. Bearing represents the clockwise angle in degrees\n    between north and the geodesic line from (lat1, lon1) to (lat2, lon2).\n\n    Parameters\n    ----------\n    lat1 : float or numpy.array of float\n        first point's latitude coordinate\n    lon1 : float or numpy.array of float\n        first point's longitude coordinate\n    lat2 : float or numpy.array of float\n        second point's latitude coordinate\n    lon2 : float or numpy.array of float\n        second point's longitude coordinate\n\n    Returns\n    -------\n    bearing : float or numpy.array of float\n        the bearing(s) in decimal degrees\n    "
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    delta_lon = np.radians(lon2 - lon1)
    y = np.sin(delta_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    initial_bearing = np.degrees(np.arctan2(y, x))
    return initial_bearing % 360

def add_edge_bearings(G, precision=None):
    if False:
        while True:
            i = 10
    '\n    Add compass `bearing` attributes to all graph edges.\n\n    Vectorized function to calculate (initial) bearing from origin node to\n    destination node for each edge in a directed, unprojected graph then add\n    these bearings as new edge attributes. Bearing represents angle in degrees\n    (clockwise) between north and the geodesic line from the origin node to\n    the destination node. Ignores self-loop edges as their bearings are\n    undefined.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        unprojected graph\n    precision : int\n        deprecated, do not use\n\n    Returns\n    -------\n    G : networkx.MultiDiGraph\n        graph with edge bearing attributes\n    '
    if precision is None:
        precision = 1
    else:
        warn('the `precision` parameter is deprecated and will be removed in a future release', stacklevel=2)
    if projection.is_projected(G.graph['crs']):
        msg = 'graph must be unprojected to add edge bearings'
        raise ValueError(msg)
    uvk = [(u, v, k) for (u, v, k) in G.edges if u != v]
    x = G.nodes(data='x')
    y = G.nodes(data='y')
    coords = np.array([(y[u], x[u], y[v], x[v]) for (u, v, k) in uvk])
    bearings = calculate_bearing(coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3])
    values = zip(uvk, bearings.round(precision))
    nx.set_edge_attributes(G, dict(values), name='bearing')
    return G

def orientation_entropy(Gu, num_bins=36, min_length=0, weight=None):
    if False:
        print('Hello World!')
    '\n    Calculate undirected graph\'s orientation entropy.\n\n    Orientation entropy is the entropy of its edges\' bidirectional bearings\n    across evenly spaced bins. Ignores self-loop edges as their bearings are\n    undefined.\n\n    For more info see: Boeing, G. 2019. "Urban Spatial Order: Street Network\n    Orientation, Configuration, and Entropy." Applied Network Science, 4 (1),\n    67. https://doi.org/10.1007/s41109-019-0189-1\n\n    Parameters\n    ----------\n    Gu : networkx.MultiGraph\n        undirected, unprojected graph with `bearing` attributes on each edge\n    num_bins : int\n        number of bins; for example, if `num_bins=36` is provided, then each\n        bin will represent 10 degrees around the compass\n    min_length : float\n        ignore edges with `length` attributes less than `min_length`; useful\n        to ignore the noise of many very short edges\n    weight : string\n        if not None, weight edges\' bearings by this (non-null) edge attribute.\n        for example, if "length" is provided, this will return 1 bearing\n        observation per meter per street, which could result in a very large\n        `bearings` array.\n\n    Returns\n    -------\n    entropy : float\n        the graph\'s orientation entropy\n    '
    if scipy is None:
        msg = 'scipy must be installed to calculate entropy'
        raise ImportError(msg)
    (bin_counts, _) = _bearings_distribution(Gu, num_bins, min_length, weight)
    return scipy.stats.entropy(bin_counts)

def _extract_edge_bearings(Gu, min_length=0, weight=None):
    if False:
        return 10
    '\n    Extract undirected graph\'s bidirectional edge bearings.\n\n    For example, if an edge has a bearing of 90 degrees then we will record\n    bearings of both 90 degrees and 270 degrees for this edge.\n\n    Parameters\n    ----------\n    Gu : networkx.MultiGraph\n        undirected, unprojected graph with `bearing` attributes on each edge\n    min_length : float\n        ignore edges with `length` attributes less than `min_length`; useful\n        to ignore the noise of many very short edges\n    weight : string\n        if not None, weight edges\' bearings by this (non-null) edge attribute.\n        for example, if "length" is provided, this will return 1 bearing\n        observation per meter per street, which could result in a very large\n        `bearings` array.\n\n    Returns\n    -------\n    bearings : numpy.array\n        the graph\'s bidirectional edge bearings\n    '
    if nx.is_directed(Gu) or projection.is_projected(Gu.graph['crs']):
        msg = 'graph must be undirected and unprojected to analyze edge bearings'
        raise ValueError(msg)
    bearings = []
    for (u, v, data) in Gu.edges(data=True):
        if u != v and data['length'] >= min_length:
            if weight:
                bearings.extend([data['bearing']] * int(data[weight]))
            else:
                bearings.append(data['bearing'])
    bearings = np.array(bearings)
    bearings = bearings[~np.isnan(bearings)]
    bearings_r = (bearings - 180) % 360
    return np.concatenate([bearings, bearings_r])

def _bearings_distribution(Gu, num_bins, min_length=0, weight=None):
    if False:
        while True:
            i = 10
    '\n    Compute distribution of bearings across evenly spaced bins.\n\n    Prevents bin-edge effects around common values like 0 degrees and 90\n    degrees by initially creating twice as many bins as desired, then merging\n    them in pairs. For example, if `num_bins=36` is provided, then each bin\n    will represent 10 degrees around the compass, with the first bin\n    representing 355 degrees to 5 degrees.\n\n    Parameters\n    ----------\n    Gu : networkx.MultiGraph\n        undirected, unprojected graph with `bearing` attributes on each edge\n    num_bins : int\n        number of bins for the bearings histogram\n    min_length : float\n        ignore edges with `length` attributes less than `min_length`; useful\n        to ignore the noise of many very short edges\n    weight : string\n        if not None, weight edges\' bearings by this (non-null) edge attribute.\n        for example, if "length" is provided, this will return 1 bearing\n        observation per meter per street, which could result in a very large\n        `bearings` array.\n\n    Returns\n    -------\n    bin_counts, bin_edges : tuple of numpy.array\n        counts of bearings per bin and the bins edges\n    '
    n = num_bins * 2
    bins = np.arange(n + 1) * 360 / n
    bearings = _extract_edge_bearings(Gu, min_length, weight)
    (count, bin_edges) = np.histogram(bearings, bins=bins)
    count = np.roll(count, 1)
    bin_counts = count[::2] + count[1::2]
    bin_edges = bin_edges[range(0, len(bin_edges), 2)]
    return (bin_counts, bin_edges)

def plot_orientation(Gu, num_bins=36, min_length=0, weight=None, ax=None, figsize=(5, 5), area=True, color='#003366', edgecolor='k', linewidth=0.5, alpha=0.7, title=None, title_y=1.05, title_font=None, xtick_font=None):
    if False:
        print('Hello World!')
    '\n    Do not use: deprecated.\n\n    The plot_orientation function moved to the plot module. Calling it via the\n    bearing module will raise an error in a future release.\n\n    Parameters\n    ----------\n    Gu : networkx.MultiGraph\n        deprecated, do not use\n    num_bins : int\n        deprecated, do not use\n    min_length : float\n        deprecated, do not use\n    weight : string\n        deprecated, do not use\n    ax : matplotlib.axes.PolarAxesSubplot\n        deprecated, do not use\n    figsize : tuple\n        deprecated, do not use\n    area : bool\n        deprecated, do not use\n    color : string\n        deprecated, do not use\n    edgecolor : string\n        deprecated, do not use\n    linewidth : float\n        deprecated, do not use\n    alpha : float\n        deprecated, do not use\n    title : string\n        deprecated, do not use\n    title_y : float\n        deprecated, do not use\n    title_font : dict\n        deprecated, do not use\n    xtick_font : dict\n        deprecated, do not use\n\n    Returns\n    -------\n    fig, ax : tuple\n        matplotlib figure, axis\n    '
    warn('The `plot_orientation` function moved to the `plot` module. Calling it via the `bearing` module will cause an exception in a future release.', stacklevel=2)
    return plot.plot_orientation(Gu, num_bins=num_bins, min_length=min_length, weight=weight, ax=ax, figsize=figsize, area=area, color=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, title=title, title_y=title_y, title_font=title_font, xtick_font=xtick_font)