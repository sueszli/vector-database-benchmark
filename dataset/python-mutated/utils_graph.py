"""Graph utility functions."""
import itertools
from warnings import warn
import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry import Point
from . import utils

def graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True):
    if False:
        while True:
            i = 10
    '\n    Convert a MultiDiGraph to node and/or edge GeoDataFrames.\n\n    This function is the inverse of `graph_from_gdfs`.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        input graph\n    nodes : bool\n        if True, convert graph nodes to a GeoDataFrame and return it\n    edges : bool\n        if True, convert graph edges to a GeoDataFrame and return it\n    node_geometry : bool\n        if True, create a geometry column from node x and y attributes\n    fill_edge_geometry : bool\n        if True, fill in missing edge geometry fields using nodes u and v\n\n    Returns\n    -------\n    geopandas.GeoDataFrame or tuple\n        gdf_nodes or gdf_edges or tuple of (gdf_nodes, gdf_edges). gdf_nodes\n        is indexed by osmid and gdf_edges is multi-indexed by u, v, key\n        following normal MultiDiGraph structure.\n    '
    crs = G.graph['crs']
    if nodes:
        if not G.nodes:
            msg = 'graph contains no nodes'
            raise ValueError(msg)
        (nodes, data) = zip(*G.nodes(data=True))
        if node_geometry:
            geom = (Point(d['x'], d['y']) for d in data)
            gdf_nodes = gpd.GeoDataFrame(data, index=nodes, crs=crs, geometry=list(geom))
        else:
            gdf_nodes = gpd.GeoDataFrame(data, index=nodes)
        gdf_nodes.index.rename('osmid', inplace=True)
        utils.log('Created nodes GeoDataFrame from graph')
    if edges:
        if not G.edges:
            msg = 'graph contains no edges'
            raise ValueError(msg)
        (u, v, k, data) = zip(*G.edges(keys=True, data=True))
        if fill_edge_geometry:
            x_lookup = nx.get_node_attributes(G, 'x')
            y_lookup = nx.get_node_attributes(G, 'y')

            def _make_geom(u, v, data, x=x_lookup, y=y_lookup):
                if False:
                    for i in range(10):
                        print('nop')
                if 'geometry' in data:
                    return data['geometry']
                return LineString((Point((x[u], y[u])), Point((x[v], y[v]))))
            geom = map(_make_geom, u, v, data)
            gdf_edges = gpd.GeoDataFrame(data, crs=crs, geometry=list(geom))
        else:
            gdf_edges = gpd.GeoDataFrame(data)
            if 'geometry' not in gdf_edges.columns:
                gdf_edges = gdf_edges.set_geometry([None] * len(gdf_edges))
            gdf_edges = gdf_edges.set_crs(crs)
        gdf_edges['u'] = u
        gdf_edges['v'] = v
        gdf_edges['key'] = k
        gdf_edges.set_index(['u', 'v', 'key'], inplace=True)
        utils.log('Created edges GeoDataFrame from graph')
    if nodes and edges:
        return (gdf_nodes, gdf_edges)
    if nodes:
        return gdf_nodes
    if edges:
        return gdf_edges
    msg = 'you must request nodes or edges or both'
    raise ValueError(msg)

def graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=None):
    if False:
        while True:
            i = 10
    '\n    Convert node and edge GeoDataFrames to a MultiDiGraph.\n\n    This function is the inverse of `graph_to_gdfs` and is designed to work in\n    conjunction with it.\n\n    However, you can convert arbitrary node and edge GeoDataFrames as long as\n    1) `gdf_nodes` is uniquely indexed by `osmid`, 2) `gdf_nodes` contains `x`\n    and `y` coordinate columns representing node geometries, 3) `gdf_edges` is\n    uniquely multi-indexed by `u`, `v`, `key` (following normal MultiDiGraph\n    structure). This allows you to load any node/edge shapefiles or GeoPackage\n    layers as GeoDataFrames then convert them to a MultiDiGraph for graph\n    analysis. Note that any `geometry` attribute on `gdf_nodes` is discarded\n    since `x` and `y` provide the necessary node geometry information instead.\n\n    Parameters\n    ----------\n    gdf_nodes : geopandas.GeoDataFrame\n        GeoDataFrame of graph nodes uniquely indexed by osmid\n    gdf_edges : geopandas.GeoDataFrame\n        GeoDataFrame of graph edges uniquely multi-indexed by u, v, key\n    graph_attrs : dict\n        the new G.graph attribute dict. if None, use crs from gdf_edges as the\n        only graph-level attribute (gdf_edges must have crs attribute set)\n\n    Returns\n    -------\n    G : networkx.MultiDiGraph\n    '
    if not ('x' in gdf_nodes.columns and 'y' in gdf_nodes.columns):
        msg = 'gdf_nodes must contain x and y columns'
        raise ValueError(msg)
    if hasattr(gdf_nodes, 'geometry'):
        try:
            all_x_match = (gdf_nodes.geometry.x == gdf_nodes['x']).all()
            all_y_match = (gdf_nodes.geometry.y == gdf_nodes['y']).all()
            assert all_x_match
            assert all_y_match
        except (AssertionError, ValueError):
            warn('discarding the gdf_nodes geometry column, though its values differ from the coordinates in the x and y columns', stacklevel=2)
        gdf_nodes = gdf_nodes.drop(columns=gdf_nodes.geometry.name)
    if graph_attrs is None:
        graph_attrs = {'crs': gdf_edges.crs}
    G = nx.MultiDiGraph(**graph_attrs)
    attr_names = gdf_edges.columns.to_list()
    for ((u, v, k), attr_vals) in zip(gdf_edges.index, gdf_edges.values):
        data_all = zip(attr_names, attr_vals)
        data = {name: val for (name, val) in data_all if isinstance(val, list) or pd.notnull(val)}
        G.add_edge(u, v, key=k, **data)
    G.add_nodes_from(set(gdf_nodes.index) - set(G.nodes))
    for col in gdf_nodes.columns:
        nx.set_node_attributes(G, name=col, values=gdf_nodes[col].dropna())
    utils.log('Created graph from node/edge GeoDataFrames')
    return G

def route_to_gdf(G, route, weight='length'):
    if False:
        print('Hello World!')
    '\n    Return a GeoDataFrame of the edges in a path, in order.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        input graph\n    route : list\n        list of node IDs constituting the path\n    weight : string\n        if there are parallel edges between two nodes, choose lowest weight\n\n    Returns\n    -------\n    gdf_edges : geopandas.GeoDataFrame\n        GeoDataFrame of the edges\n    '
    pairs = zip(route[:-1], route[1:])
    uvk = ((u, v, min(G[u][v].items(), key=lambda i: i[1][weight])[0]) for (u, v) in pairs)
    return graph_to_gdfs(G.subgraph(route), nodes=False).loc[uvk]

def get_route_edge_attributes(G, route, attribute=None, minimize_key='length', retrieve_default=None):
    if False:
        i = 10
        return i + 15
    '\n    Do not use: deprecated.\n\n    Use the `route_to_gdf` function instead.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        deprecated\n    route : list\n        deprecated\n    attribute : string\n        deprecated\n    minimize_key : string\n        deprecated\n    retrieve_default : Callable[Tuple[Any, Any], Any]\n        deprecated\n\n    Returns\n    -------\n    attribute_values : list\n        deprecated\n    '
    warn('The `get_route_edge_attributes` function has been deprecated and will be removed in a future release. Use the `route_to_gdf` function instead.', stacklevel=2)
    attribute_values = []
    for (u, v) in zip(route[:-1], route[1:]):
        data = min(G.get_edge_data(u, v).values(), key=lambda x: x[minimize_key])
        if attribute is None:
            attribute_value = data
        elif retrieve_default is not None:
            attribute_value = data.get(attribute, retrieve_default(u, v))
        else:
            attribute_value = data[attribute]
        attribute_values.append(attribute_value)
    return attribute_values

def remove_isolated_nodes(G):
    if False:
        return 10
    '\n    Remove from a graph all nodes that have no incident edges.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        graph from which to remove isolated nodes\n\n    Returns\n    -------\n    G : networkx.MultiDiGraph\n        graph with all isolated nodes removed\n    '
    G = G.copy()
    isolated_nodes = {node for (node, degree) in G.degree() if degree < 1}
    G.remove_nodes_from(isolated_nodes)
    utils.log(f'Removed {len(isolated_nodes):,} isolated nodes')
    return G

def get_largest_component(G, strongly=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get subgraph of G's largest weakly/strongly connected component.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        input graph\n    strongly : bool\n        if True, return the largest strongly instead of weakly connected\n        component\n\n    Returns\n    -------\n    G : networkx.MultiDiGraph\n        the largest connected component subgraph of the original graph\n    "
    if strongly:
        kind = 'strongly'
        is_connected = nx.is_strongly_connected
        connected_components = nx.strongly_connected_components
    else:
        kind = 'weakly'
        is_connected = nx.is_weakly_connected
        connected_components = nx.weakly_connected_components
    if not is_connected(G):
        largest_cc = max(connected_components(G), key=len)
        n = len(G)
        G = nx.MultiDiGraph(G.subgraph(largest_cc))
        utils.log(f'Got largest {kind} connected component ({len(G):,} of {n:,} total nodes)')
    return G

def get_digraph(G, weight='length'):
    if False:
        return 10
    '\n    Convert MultiDiGraph to DiGraph.\n\n    Chooses between parallel edges by minimizing `weight` attribute value.\n    Note: see also `get_undirected` to convert MultiDiGraph to MultiGraph.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        input graph\n    weight : string\n        attribute value to minimize when choosing between parallel edges\n\n    Returns\n    -------\n    networkx.DiGraph\n    '
    G = G.copy()
    to_remove = []
    parallels = ((u, v) for (u, v) in G.edges(keys=False) if len(G.get_edge_data(u, v)) > 1)
    for (u, v) in set(parallels):
        (k_min, _) = min(G.get_edge_data(u, v).items(), key=lambda x: x[1][weight])
        to_remove.extend(((u, v, k) for k in G[u][v] if k != k_min))
    G.remove_edges_from(to_remove)
    utils.log('Converted MultiDiGraph to DiGraph')
    return nx.DiGraph(G)

def get_undirected(G):
    if False:
        while True:
            i = 10
    '\n    Convert MultiDiGraph to undirected MultiGraph.\n\n    Maintains parallel edges only if their geometries differ. Note: see also\n    `get_digraph` to convert MultiDiGraph to DiGraph.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        input graph\n\n    Returns\n    -------\n    networkx.MultiGraph\n    '
    G = G.copy()
    for (u, v, d) in G.edges(data=True):
        d['from'] = u
        d['to'] = v
        if 'geometry' not in d:
            point_u = (G.nodes[u]['x'], G.nodes[u]['y'])
            point_v = (G.nodes[v]['x'], G.nodes[v]['y'])
            d['geometry'] = LineString([point_u, point_v])
    G = _update_edge_keys(G)
    H = nx.MultiGraph(**G.graph)
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from(G.edges(keys=True, data=True))
    duplicate_edges = set()
    for (u1, v1, key1, data1) in H.edges(keys=True, data=True):
        if (u1, v1, key1) not in duplicate_edges:
            for key2 in H[u1][v1]:
                if key1 != key2:
                    data2 = H.edges[u1, v1, key2]
                    if _is_duplicate_edge(data1, data2):
                        duplicate_edges.add((u1, v1, key2))
    H.remove_edges_from(duplicate_edges)
    utils.log('Converted MultiDiGraph to undirected MultiGraph')
    return H

def _is_duplicate_edge(data1, data2):
    if False:
        return 10
    "\n    Check if two graph edge data dicts have the same osmid and geometry.\n\n    Parameters\n    ----------\n    data1: dict\n        the first edge's data\n    data2 : dict\n        the second edge's data\n\n    Returns\n    -------\n    is_dupe : bool\n    "
    is_dupe = False
    osmid1 = set(data1['osmid']) if isinstance(data1['osmid'], list) else data1['osmid']
    osmid2 = set(data2['osmid']) if isinstance(data2['osmid'], list) else data2['osmid']
    if osmid1 == osmid2:
        if 'geometry' in data1 and 'geometry' in data2:
            if _is_same_geometry(data1['geometry'], data2['geometry']):
                is_dupe = True
        elif 'geometry' not in data1 and 'geometry' not in data2:
            is_dupe = True
        else:
            pass
    return is_dupe

def _is_same_geometry(ls1, ls2):
    if False:
        return 10
    '\n    Determine if two LineString geometries are the same (in either direction).\n\n    Check both the normal and reversed orders of their constituent points.\n\n    Parameters\n    ----------\n    ls1 : shapely.geometry.LineString\n        the first LineString geometry\n    ls2 : shapely.geometry.LineString\n        the second LineString geometry\n\n    Returns\n    -------\n    bool\n    '
    geom1 = [tuple(coords) for coords in ls1.xy]
    geom2 = [tuple(coords) for coords in ls2.xy]
    geom1_r = [tuple(reversed(coords)) for coords in ls1.xy]
    return geom2 in (geom1, geom1_r)

def _update_edge_keys(G):
    if False:
        return 10
    "\n    Increment key of one edge of parallel edges that differ in geometry.\n\n    For example, two streets from u to v that bow away from each other as\n    separate streets, rather than opposite direction edges of a single street.\n    Increment one of these edge's keys so that they do not match across u, v,\n    k or v, u, k so we can add both to an undirected MultiGraph.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        input graph\n\n    Returns\n    -------\n    G : networkx.MultiDiGraph\n    "
    edges = graph_to_gdfs(G, nodes=False, fill_edge_geometry=False)
    edges['uvk'] = ['_'.join(sorted([str(u), str(v)]) + [str(k)]) for (u, v, k) in edges.index]
    mask = edges['uvk'].duplicated(keep=False)
    dupes = edges[mask].dropna(subset=['geometry'])
    different_streets = []
    groups = dupes[['geometry', 'uvk']].groupby('uvk')
    for (_, group) in groups:
        for (geom1, geom2) in itertools.combinations(group['geometry'], 2):
            if not _is_same_geometry(geom1, geom2):
                different_streets.append(group.index[0])
    for (u, v, k) in set(different_streets):
        new_key = max(list(G[u][v]) + list(G[v][u])) + 1
        G.add_edge(u, v, key=new_key, **G.get_edge_data(u, v, k))
        G.remove_edge(u, v, key=k)
    return G