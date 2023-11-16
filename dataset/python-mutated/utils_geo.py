"""Geospatial utility functions."""
from warnings import warn
import networkx as nx
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPoint
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import split
from . import projection
from . import settings
from . import utils
from . import utils_graph

def sample_points(G, n):
    if False:
        for i in range(10):
            print('nop')
    "\n    Randomly sample points constrained to a spatial graph.\n\n    This generates a graph-constrained uniform random sample of points. Unlike\n    typical spatially uniform random sampling, this method accounts for the\n    graph's geometry. And unlike equal-length edge segmenting, this method\n    guarantees uniform randomness.\n\n    Parameters\n    ----------\n    G : networkx.MultiGraph\n        graph to sample points from; should be undirected (to not oversample\n        bidirectional edges) and projected (for accurate point interpolation)\n    n : int\n        how many points to sample\n\n    Returns\n    -------\n    points : geopandas.GeoSeries\n        the sampled points, multi-indexed by (u, v, key) of the edge from\n        which each point was drawn\n    "
    if nx.is_directed(G):
        warn('graph should be undirected to not oversample bidirectional edges', stacklevel=2)
    gdf_edges = utils_graph.graph_to_gdfs(G, nodes=False)[['geometry', 'length']]
    weights = gdf_edges['length'] / gdf_edges['length'].sum()
    idx = np.random.choice(gdf_edges.index, size=n, p=weights)
    lines = gdf_edges.loc[idx, 'geometry']
    return lines.interpolate(np.random.rand(n), normalized=True)

def interpolate_points(geom, dist):
    if False:
        while True:
            i = 10
    "\n    Interpolate evenly spaced points along a LineString.\n\n    The spacing is approximate because the LineString's length may not be\n    evenly divisible by it.\n\n    Parameters\n    ----------\n    geom : shapely.geometry.LineString\n        a LineString geometry\n    dist : float\n        spacing distance between interpolated points, in same units as `geom`.\n        smaller values generate more points.\n\n    Yields\n    ------\n    point : tuple of floats\n        a generator of (x, y) tuples of the interpolated points' coordinates\n    "
    if isinstance(geom, LineString):
        num_vert = max(round(geom.length / dist), 1)
        for n in range(num_vert + 1):
            point = geom.interpolate(n / num_vert, normalized=True)
            yield (point.x, point.y)
    else:
        msg = f'unhandled geometry type {geom.geom_type}'
        raise TypeError(msg)

def _round_polygon_coords(p, precision):
    if False:
        return 10
    '\n    Round the coordinates of a shapely Polygon to some decimal precision.\n\n    Parameters\n    ----------\n    p : shapely.geometry.Polygon\n        the polygon to round the coordinates of\n    precision : int\n        decimal precision to round coordinates to\n\n    Returns\n    -------\n    shapely.geometry.Polygon\n    '
    shell = [[round(x, precision) for x in c] for c in p.exterior.coords]
    holes = [[[round(x, precision) for x in c] for c in i.coords] for i in p.interiors]
    return Polygon(shell=shell, holes=holes).buffer(0)

def _round_multipolygon_coords(mp, precision):
    if False:
        print('Hello World!')
    '\n    Round the coordinates of a shapely MultiPolygon to some decimal precision.\n\n    Parameters\n    ----------\n    mp : shapely.geometry.MultiPolygon\n        the MultiPolygon to round the coordinates of\n    precision : int\n        decimal precision to round coordinates to\n\n    Returns\n    -------\n    shapely.geometry.MultiPolygon\n    '
    return MultiPolygon([_round_polygon_coords(p, precision) for p in mp.geoms])

def _round_point_coords(pt, precision):
    if False:
        while True:
            i = 10
    '\n    Round the coordinates of a shapely Point to some decimal precision.\n\n    Parameters\n    ----------\n    pt : shapely.geometry.Point\n        the Point to round the coordinates of\n    precision : int\n        decimal precision to round coordinates to\n\n    Returns\n    -------\n    shapely.geometry.Point\n    '
    return Point([round(x, precision) for x in pt.coords[0]])

def _round_multipoint_coords(mpt, precision):
    if False:
        for i in range(10):
            print('nop')
    '\n    Round the coordinates of a shapely MultiPoint to some decimal precision.\n\n    Parameters\n    ----------\n    mpt : shapely.geometry.MultiPoint\n        the MultiPoint to round the coordinates of\n    precision : int\n        decimal precision to round coordinates to\n\n    Returns\n    -------\n    shapely.geometry.MultiPoint\n    '
    return MultiPoint([_round_point_coords(pt, precision) for pt in mpt.geoms])

def _round_linestring_coords(ls, precision):
    if False:
        return 10
    '\n    Round the coordinates of a shapely LineString to some decimal precision.\n\n    Parameters\n    ----------\n    ls : shapely.geometry.LineString\n        the LineString to round the coordinates of\n    precision : int\n        decimal precision to round coordinates to\n\n    Returns\n    -------\n    shapely.geometry.LineString\n    '
    return LineString([[round(x, precision) for x in c] for c in ls.coords])

def _round_multilinestring_coords(mls, precision):
    if False:
        print('Hello World!')
    '\n    Round the coordinates of a shapely MultiLineString to some decimal precision.\n\n    Parameters\n    ----------\n    mls : shapely.geometry.MultiLineString\n        the MultiLineString to round the coordinates of\n    precision : int\n        decimal precision to round coordinates to\n\n    Returns\n    -------\n    shapely.geometry.MultiLineString\n    '
    return MultiLineString([_round_linestring_coords(ls, precision) for ls in mls.geoms])

def round_geometry_coords(geom, precision):
    if False:
        for i in range(10):
            print('nop')
    '\n    Do not use: deprecated.\n\n    Parameters\n    ----------\n    geom : shapely.geometry.geometry {Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon}\n        deprecated, do not use\n    precision : int\n        deprecated, do not use\n\n    Returns\n    -------\n    shapely.geometry.geometry\n    '
    warn('the `round_geometry_coords` function is deprecated and will be removed in a future release', stacklevel=2)
    if isinstance(geom, Point):
        return _round_point_coords(geom, precision)
    if isinstance(geom, MultiPoint):
        return _round_multipoint_coords(geom, precision)
    if isinstance(geom, LineString):
        return _round_linestring_coords(geom, precision)
    if isinstance(geom, MultiLineString):
        return _round_multilinestring_coords(geom, precision)
    if isinstance(geom, Polygon):
        return _round_polygon_coords(geom, precision)
    if isinstance(geom, MultiPolygon):
        return _round_multipolygon_coords(geom, precision)
    msg = f'cannot round coordinates of unhandled geometry type: {type(geom)}'
    raise TypeError(msg)

def _consolidate_subdivide_geometry(geometry, max_query_area_size=None):
    if False:
        print('Hello World!')
    "\n    Consolidate and subdivide some geometry.\n\n    Consolidate a geometry into a convex hull, then subdivide it into smaller\n    sub-polygons if its area exceeds max size (in geometry's units). Configure\n    the max size via max_query_area_size in the settings module.\n\n    When the geometry has a very large area relative to its vertex count,\n    the resulting MultiPolygon's boundary may differ somewhat from the input,\n    due to the way long straight lines are projected. You can interpolate\n    additional vertices along your input geometry's exterior to mitigate this.\n\n    Parameters\n    ----------\n    geometry : shapely.geometry.Polygon or shapely.geometry.MultiPolygon\n        the geometry to consolidate and subdivide\n    max_query_area_size : int\n        maximum area for any part of the geometry in meters: any polygon\n        bigger than this will get divided up for multiple queries to API\n        (default 50km x 50km). if None, use settings.max_query_area_size\n\n    Returns\n    -------\n    geometry : shapely.geometry.MultiPolygon\n    "
    if max_query_area_size is None:
        max_query_area_size = settings.max_query_area_size
    quadrat_width = np.sqrt(max_query_area_size)
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        msg = 'Geometry must be a shapely Polygon or MultiPolygon'
        raise TypeError(msg)
    if isinstance(geometry, MultiPolygon) or (isinstance(geometry, Polygon) and geometry.area > max_query_area_size):
        geometry = geometry.convex_hull
    if geometry.area > max_query_area_size:
        geometry = _quadrat_cut_geometry(geometry, quadrat_width=quadrat_width)
    if isinstance(geometry, Polygon):
        geometry = MultiPolygon([geometry])
    return geometry

def _get_polygons_coordinates(geometry):
    if False:
        i = 10
        return i + 15
    '\n    Extract exterior coordinates from polygon(s) to pass to OSM.\n\n    Ignore the interior ("holes") coordinates.\n\n    Parameters\n    ----------\n    geometry : shapely.geometry.Polygon or shapely.geometry.MultiPolygon\n        the geometry to extract exterior coordinates from\n\n    Returns\n    -------\n    polygon_coord_strs : list\n    '
    if not isinstance(geometry, MultiPolygon):
        msg = 'Geometry must be a shapely MultiPolygon'
        raise TypeError(msg)
    polygons_coords = []
    for polygon in geometry.geoms:
        (x, y) = polygon.exterior.xy
        polygons_coords.append(list(zip(x, y)))
    polygon_coord_strs = []
    for coords in polygons_coords:
        s = ''
        separator = ' '
        for coord in list(coords):
            s = f'{s}{separator}{coord[1]:.6f}{separator}{coord[0]:.6f}'
        polygon_coord_strs.append(s.strip(separator))
    return polygon_coord_strs

def _quadrat_cut_geometry(geometry, quadrat_width, min_num=3):
    if False:
        print('Hello World!')
    '\n    Split a Polygon or MultiPolygon up into sub-polygons of a specified size.\n\n    Parameters\n    ----------\n    geometry : shapely.geometry.Polygon or shapely.geometry.MultiPolygon\n        the geometry to split up into smaller sub-polygons\n    quadrat_width : numeric\n        the linear width of the quadrats with which to cut up the geometry (in\n        the units the geometry is in)\n    min_num : int\n        the minimum number of linear quadrat lines (e.g., min_num=3 would\n        produce a quadrat grid of 4 squares)\n\n    Returns\n    -------\n    geometry : shapely.geometry.MultiPolygon\n    '
    (west, south, east, north) = geometry.bounds
    x_num = int(np.ceil((east - west) / quadrat_width) + 1)
    y_num = int(np.ceil((north - south) / quadrat_width) + 1)
    x_points = np.linspace(west, east, num=max(x_num, min_num))
    y_points = np.linspace(south, north, num=max(y_num, min_num))
    vertical_lines = [LineString([(x, y_points[0]), (x, y_points[-1])]) for x in x_points]
    horizont_lines = [LineString([(x_points[0], y), (x_points[-1], y)]) for y in y_points]
    lines = vertical_lines + horizont_lines
    geometries = [geometry]
    for line in lines:
        split_geoms = [split(g, line).geoms if g.intersects(line) else [g] for g in geometries]
        geometries = [g for g_list in split_geoms for g in g_list]
    return MultiPolygon(geometries)

def _intersect_index_quadrats(geometries, polygon, quadrat_width=0.05, min_num=3):
    if False:
        print('Hello World!')
    "\n    Identify geometries that intersect a (multi)polygon.\n\n    Uses an r-tree spatial index and cuts polygon up into smaller sub-polygons\n    for r-tree acceleration. Ensure that geometries and polygon are in the\n    same coordinate reference system.\n\n    Parameters\n    ----------\n    geometries : geopandas.GeoSeries\n        the geometries to intersect with the polygon\n    polygon : shapely.geometry.Polygon or shapely.geometry.MultiPolygon\n        the polygon to intersect with the geometries\n    quadrat_width : numeric\n        linear length (in polygon's units) of quadrat lines with which to cut\n        up the polygon (default = 0.05 degrees, approx 4km at NYC's latitude)\n    min_num : int\n        the minimum number of linear quadrat lines (e.g., min_num=3 would\n        produce a quadrat grid of 4 squares)\n\n    Returns\n    -------\n    geoms_in_poly : set\n        index labels of geometries that intersected polygon\n    "
    sindex = geometries.sindex
    utils.log(f'Created r-tree spatial index for {len(geometries):,} geometries')
    multipoly = _quadrat_cut_geometry(polygon, quadrat_width=quadrat_width, min_num=min_num)
    geoms_in_poly = set()
    for poly in multipoly.geoms:
        poly_buff = poly.buffer(0)
        if poly_buff.is_valid and poly_buff.area > 0:
            possible_matches_iloc = sindex.intersection(poly_buff.bounds)
            possible_matches = geometries.iloc[list(possible_matches_iloc)]
            precise_matches = possible_matches[possible_matches.intersects(poly_buff)]
            geoms_in_poly.update(precise_matches.index)
    utils.log(f'Identified {len(geoms_in_poly):,} geometries inside polygon')
    return geoms_in_poly

def bbox_from_point(point, dist=1000, project_utm=False, return_crs=False):
    if False:
        i = 10
        return i + 15
    '\n    Create a bounding box from a (lat, lon) center point.\n\n    Create a bounding box some distance in each direction (north, south, east,\n    and west) from the center point and optionally project it.\n\n    Parameters\n    ----------\n    point : tuple\n        the (lat, lon) center point to create the bounding box around\n    dist : int\n        bounding box distance in meters from the center point\n    project_utm : bool\n        if True, return bounding box as UTM-projected coordinates\n    return_crs : bool\n        if True, and project_utm=True, return the projected CRS too\n\n    Returns\n    -------\n    tuple\n        (north, south, east, west) or (north, south, east, west, crs_proj)\n    '
    earth_radius = 6371009
    (lat, lon) = point
    delta_lat = dist / earth_radius * (180 / np.pi)
    delta_lon = dist / earth_radius * (180 / np.pi) / np.cos(lat * np.pi / 180)
    north = lat + delta_lat
    south = lat - delta_lat
    east = lon + delta_lon
    west = lon - delta_lon
    if project_utm:
        bbox_poly = bbox_to_poly(north, south, east, west)
        (bbox_proj, crs_proj) = projection.project_geometry(bbox_poly)
        (west, south, east, north) = bbox_proj.bounds
    utils.log(f'Created bbox {dist} m from {point}: {north},{south},{east},{west}')
    if project_utm and return_crs:
        return (north, south, east, west, crs_proj)
    return (north, south, east, west)

def bbox_to_poly(north, south, east, west):
    if False:
        print('Hello World!')
    '\n    Convert bounding box coordinates to shapely Polygon.\n\n    Parameters\n    ----------\n    north : float\n        northern coordinate\n    south : float\n        southern coordinate\n    east : float\n        eastern coordinate\n    west : float\n        western coordinate\n\n    Returns\n    -------\n    shapely.geometry.Polygon\n    '
    return Polygon([(west, south), (east, south), (east, north), (west, north)])