"""Simplify, correct, and consolidate network topology."""
import logging as lg
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from . import stats
from . import utils
from . import utils_graph
from ._errors import GraphSimplificationError

def _is_endpoint(G, node, strict=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Determine if a node is a true endpoint of an edge.\n\n    Return True if the node is a "real" endpoint of an edge in the network,\n    otherwise False. OSM data includes lots of nodes that exist only as points\n    to help streets bend around curves. An end point is a node that either:\n    1) is its own neighbor, ie, it self-loops.\n    2) or, has no incoming edges or no outgoing edges, ie, all its incident\n    edges point inward or all its incident edges point outward.\n    3) or, it does not have exactly two neighbors and degree of 2 or 4.\n    4) or, if strict mode is false, if its edges have different OSM IDs.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        input graph\n    node : int\n        the node to examine\n    strict : bool\n        if False, allow nodes to be end points even if they fail all other rules\n        but have edges with different OSM IDs\n\n    Returns\n    -------\n    bool\n    '
    neighbors = set(list(G.predecessors(node)) + list(G.successors(node)))
    n = len(neighbors)
    d = G.degree(node)
    if node in neighbors:
        return True
    if G.out_degree(node) == 0 or G.in_degree(node) == 0:
        return True
    if not (n == 2 and d in {2, 4}):
        return True
    if not strict:
        incoming = [G.edges[u, node, k]['osmid'] for u in G.predecessors(node) for k in G[u][node]]
        outgoing = [G.edges[node, v, k]['osmid'] for v in G.successors(node) for k in G[node][v]]
        return len(set(incoming + outgoing)) > 1
    return False

def _build_path(G, endpoint, endpoint_successor, endpoints):
    if False:
        for i in range(10):
            print('nop')
    '\n    Build a path of nodes from one endpoint node to next endpoint node.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        input graph\n    endpoint : int\n        the endpoint node from which to start the path\n    endpoint_successor : int\n        the successor of endpoint through which the path to the next endpoint\n        will be built\n    endpoints : set\n        the set of all nodes in the graph that are endpoints\n\n    Returns\n    -------\n    path : list\n        the first and last items in the resulting path list are endpoint\n        nodes, and all other items are interstitial nodes that can be removed\n        subsequently\n    '
    path = [endpoint, endpoint_successor]
    for this_successor in G.successors(endpoint_successor):
        successor = this_successor
        if successor not in path:
            path.append(successor)
            while successor not in endpoints:
                successors = [n for n in G.successors(successor) if n not in path]
                if len(successors) == 1:
                    successor = successors[0]
                    path.append(successor)
                elif len(successors) == 0:
                    if endpoint in G.successors(successor):
                        return path + [endpoint]
                    msg = f'Unexpected simplify pattern handled near {successor}'
                    utils.log(msg, level=lg.WARN)
                    return path
                else:
                    msg = f'Impossible simplify pattern failed near {successor}'
                    raise GraphSimplificationError(msg)
            return path
    return path

def _get_paths_to_simplify(G, strict=True):
    if False:
        return 10
    '\n    Generate all the paths to be simplified between endpoint nodes.\n\n    The path is ordered from the first endpoint, through the interstitial nodes,\n    to the second endpoint.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        input graph\n    strict : bool\n        if False, allow nodes to be end points even if they fail all other rules\n        but have edges with different OSM IDs\n\n    Yields\n    ------\n    path_to_simplify : list\n        a generator of paths to simplify\n    '
    endpoints = {n for n in G.nodes if _is_endpoint(G, n, strict=strict)}
    utils.log(f'Identified {len(endpoints):,} edge endpoints')
    for endpoint in endpoints:
        for successor in G.successors(endpoint):
            if successor not in endpoints:
                yield _build_path(G, endpoint, successor, endpoints)

def _remove_rings(G):
    if False:
        i = 10
        return i + 15
    '\n    Remove all self-contained rings from a graph.\n\n    This identifies any connected components that form a self-contained ring\n    without any endpoints, and removes them from the graph.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        input graph\n\n    Returns\n    -------\n    G : networkx.MultiDiGraph\n        graph with self-contained rings removed\n    '
    nodes_in_rings = set()
    for wcc in nx.weakly_connected_components(G):
        if not any((_is_endpoint(G, n) for n in wcc)):
            nodes_in_rings.update(wcc)
    G.remove_nodes_from(nodes_in_rings)
    return G

def simplify_graph(G, strict=True, remove_rings=True, track_merged=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Simplify a graph's topology by removing interstitial nodes.\n\n    Simplifies graph topology by removing all nodes that are not intersections\n    or dead-ends. Create an edge directly between the end points that\n    encapsulate them, but retain the geometry of the original edges, saved as\n    a new `geometry` attribute on the new edge. Note that only simplified\n    edges receive a `geometry` attribute. Some of the resulting consolidated\n    edges may comprise multiple OSM ways, and if so, their multiple attribute\n    values are stored as a list. Optionally, the simplified edges can receive\n    a `merged_edges` attribute that contains a list of all the (u, v) node\n    pairs that were merged together.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        input graph\n    strict : bool\n        if False, allow nodes to be end points even if they fail all other\n        rules but have incident edges with different OSM IDs. Lets you keep\n        nodes at elbow two-way intersections, but sometimes individual blocks\n        have multiple OSM IDs within them too.\n    remove_rings : bool\n        if True, remove isolated self-contained rings that have no endpoints\n    track_merged : bool\n        if True, add `merged_edges` attribute on simplified edges, containing\n        a list of all the (u, v) node pairs that were merged together\n\n    Returns\n    -------\n    G : networkx.MultiDiGraph\n        topologically simplified graph, with a new `geometry` attribute on\n        each simplified edge\n    "
    if 'simplified' in G.graph and G.graph['simplified']:
        msg = 'This graph has already been simplified, cannot simplify it again.'
        raise GraphSimplificationError(msg)
    utils.log('Begin topologically simplifying the graph...')
    attrs_to_sum = {'length', 'travel_time'}
    G = G.copy()
    initial_node_count = len(G)
    initial_edge_count = len(G.edges)
    all_nodes_to_remove = []
    all_edges_to_add = []
    for path in _get_paths_to_simplify(G, strict=strict):
        merged_edges = []
        path_attributes = {}
        for (u, v) in zip(path[:-1], path[1:]):
            if track_merged:
                merged_edges.append((u, v))
            edge_count = G.number_of_edges(u, v)
            if edge_count != 1:
                utils.log(f'Found {edge_count} edges between {u} and {v} when simplifying')
            edge_data = list(G.get_edge_data(u, v).values())[0]
            for attr in edge_data:
                if attr in path_attributes:
                    path_attributes[attr].append(edge_data[attr])
                else:
                    path_attributes[attr] = [edge_data[attr]]
        for attr in path_attributes:
            if attr in attrs_to_sum:
                path_attributes[attr] = sum(path_attributes[attr])
            elif len(set(path_attributes[attr])) == 1:
                path_attributes[attr] = path_attributes[attr][0]
            else:
                path_attributes[attr] = list(set(path_attributes[attr]))
        path_attributes['geometry'] = LineString([Point((G.nodes[node]['x'], G.nodes[node]['y'])) for node in path])
        if track_merged:
            path_attributes['merged_edges'] = merged_edges
        all_nodes_to_remove.extend(path[1:-1])
        all_edges_to_add.append({'origin': path[0], 'destination': path[-1], 'attr_dict': path_attributes})
    for edge in all_edges_to_add:
        G.add_edge(edge['origin'], edge['destination'], **edge['attr_dict'])
    G.remove_nodes_from(set(all_nodes_to_remove))
    if remove_rings:
        G = _remove_rings(G)
    G.graph['simplified'] = True
    msg = f'Simplified graph: {initial_node_count:,} to {len(G):,} nodes, {initial_edge_count:,} to {len(G.edges):,} edges'
    utils.log(msg)
    return G

def consolidate_intersections(G, tolerance=10, rebuild_graph=True, dead_ends=False, reconnect_edges=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Consolidate intersections comprising clusters of nearby nodes.\n\n    Merges nearby nodes and returns either their centroids or a rebuilt graph\n    with consolidated intersections and reconnected edge geometries. The\n    tolerance argument should be adjusted to approximately match street design\n    standards in the specific street network, and you should always use a\n    projected graph to work in meaningful and consistent units like meters.\n    Note the tolerance represents a per-node buffering radius: for example, to\n    consolidate nodes within 10 meters of each other, use tolerance=5.\n\n    When rebuild_graph=False, it uses a purely geometrical (and relatively\n    fast) algorithm to identify "geometrically close" nodes, merge them, and\n    return just the merged intersections\' centroids. When rebuild_graph=True,\n    it uses a topological (and slower but more accurate) algorithm to identify\n    "topologically close" nodes, merge them, then rebuild/return the graph.\n    Returned graph\'s node IDs represent clusters rather than osmids. Refer to\n    nodes\' osmid_original attributes for original osmids. If multiple nodes\n    were merged together, the osmid_original attribute is a list of merged\n    nodes\' osmids.\n\n    Divided roads are often represented by separate centerline edges. The\n    intersection of two divided roads thus creates 4 nodes, representing where\n    each edge intersects a perpendicular edge. These 4 nodes represent a\n    single intersection in the real world. A similar situation occurs with\n    roundabouts and traffic circles. This function consolidates nearby nodes\n    by buffering them to an arbitrary distance, merging overlapping buffers,\n    and taking their centroid.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        a projected graph\n    tolerance : float\n        nodes are buffered to this distance (in graph\'s geometry\'s units) and\n        subsequent overlaps are dissolved into a single node\n    rebuild_graph : bool\n        if True, consolidate the nodes topologically, rebuild the graph, and\n        return as networkx.MultiDiGraph. if False, consolidate the nodes\n        geometrically and return the consolidated node points as\n        geopandas.GeoSeries\n    dead_ends : bool\n        if False, discard dead-end nodes to return only street-intersection\n        points\n    reconnect_edges : bool\n        ignored if rebuild_graph is not True. if True, reconnect edges and\n        their geometries in rebuilt graph to the consolidated nodes and update\n        edge length attributes; if False, returned graph has no edges (which\n        is faster if you just need topologically consolidated intersection\n        counts).\n\n    Returns\n    -------\n    networkx.MultiDiGraph or geopandas.GeoSeries\n        if rebuild_graph=True, returns MultiDiGraph with consolidated\n        intersections and reconnected edge geometries. if rebuild_graph=False,\n        returns GeoSeries of shapely Points representing the centroids of\n        street intersections\n    '
    if not dead_ends:
        spn = stats.streets_per_node(G)
        dead_end_nodes = [node for (node, count) in spn.items() if count <= 1]
        G = G.copy()
        G.remove_nodes_from(dead_end_nodes)
    if rebuild_graph:
        if not G or not G.edges:
            return G
        return _consolidate_intersections_rebuild_graph(G, tolerance, reconnect_edges)
    if not G:
        return gpd.GeoSeries(crs=G.graph['crs'])
    return _merge_nodes_geometric(G, tolerance).centroid

def _merge_nodes_geometric(G, tolerance):
    if False:
        i = 10
        return i + 15
    "\n    Geometrically merge nodes within some distance of each other.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        a projected graph\n    tolerance : float\n        buffer nodes to this distance (in graph's geometry's units) then merge\n        overlapping polygons into a single polygon via a unary union operation\n\n    Returns\n    -------\n    merged : GeoSeries\n        the merged overlapping polygons of the buffered nodes\n    "
    merged = utils_graph.graph_to_gdfs(G, edges=False)['geometry'].buffer(tolerance).unary_union
    merged = MultiPolygon([merged]) if isinstance(merged, Polygon) else merged
    return gpd.GeoSeries(merged.geoms, crs=G.graph['crs'])

def _consolidate_intersections_rebuild_graph(G, tolerance=10, reconnect_edges=True):
    if False:
        i = 10
        return i + 15
    "\n    Consolidate intersections comprising clusters of nearby nodes.\n\n    Merge nodes and return a rebuilt graph with consolidated intersections and\n    reconnected edge geometries.\n\n    The tolerance argument should be adjusted to approximately match street\n    design standards in the specific street network, and you should always use\n    a projected graph to work in meaningful and consistent units like meters.\n\n    Returned graph's node IDs represent clusters rather than osmids. Refer to\n    nodes' osmid_original attributes for original osmids. If multiple nodes\n    were merged together, the osmid_original attribute is a list of merged\n    nodes' osmids.\n\n    Parameters\n    ----------\n    G : networkx.MultiDiGraph\n        a projected graph\n    tolerance : float\n        nodes are buffered to this distance (in graph's geometry's units) and\n        subsequent overlaps are dissolved into a single node\n    reconnect_edges : bool\n        ignored if rebuild_graph is not True. if True, reconnect edges and\n        their geometries in rebuilt graph to the consolidated nodes and update\n        edge length attributes; if False, returned graph has no edges (which\n        is faster if you just need topologically consolidated intersection\n        counts).\n\n    Returns\n    -------\n    H : networkx.MultiDiGraph\n        a rebuilt graph with consolidated intersections and reconnected\n        edge geometries\n    "
    node_clusters = gpd.GeoDataFrame(geometry=_merge_nodes_geometric(G, tolerance))
    centroids = node_clusters.centroid
    node_clusters['x'] = centroids.x
    node_clusters['y'] = centroids.y
    node_points = utils_graph.graph_to_gdfs(G, edges=False)[['geometry']]
    gdf = gpd.sjoin(node_points, node_clusters, how='left', predicate='within')
    gdf = gdf.drop(columns='geometry').rename(columns={'index_right': 'cluster'})
    groups = gdf.groupby('cluster')
    for (cluster_label, nodes_subset) in groups:
        if len(nodes_subset) > 1:
            wccs = list(nx.weakly_connected_components(G.subgraph(nodes_subset.index)))
            if len(wccs) > 1:
                suffix = 0
                for wcc in wccs:
                    idx = list(wcc)
                    subcluster_centroid = node_points.loc[idx].unary_union.centroid
                    gdf.loc[idx, 'x'] = subcluster_centroid.x
                    gdf.loc[idx, 'y'] = subcluster_centroid.y
                    gdf.loc[idx, 'cluster'] = f'{cluster_label}-{suffix}'
                    suffix += 1
    gdf['cluster'] = gdf['cluster'].factorize()[0]
    H = nx.MultiDiGraph()
    H.graph = G.graph
    groups = gdf.groupby('cluster')
    for (cluster_label, nodes_subset) in groups:
        osmids = nodes_subset.index.to_list()
        if len(osmids) == 1:
            osmid = osmids[0]
            H.add_node(cluster_label, osmid_original=osmid, **G.nodes[osmid])
        else:
            H.add_node(cluster_label, osmid_original=str(osmids), x=nodes_subset['x'].iloc[0], y=nodes_subset['y'].iloc[0])
    null_nodes = [n for (n, sc) in H.nodes(data='street_count') if sc is None]
    street_count = stats.count_streets_per_node(H, nodes=null_nodes)
    nx.set_node_attributes(H, street_count, name='street_count')
    if not G.edges or not reconnect_edges:
        return H
    gdf_edges = utils_graph.graph_to_gdfs(G, nodes=False)
    for (u, v, k, data) in G.edges(keys=True, data=True):
        u2 = gdf.loc[u, 'cluster']
        v2 = gdf.loc[v, 'cluster']
        if u2 != v2 or u == v:
            data['u_original'] = u
            data['v_original'] = v
            if 'geometry' not in data:
                data['geometry'] = gdf_edges.loc[(u, v, k), 'geometry']
            H.add_edge(u2, v2, **data)
    for (cluster_label, nodes_subset) in groups:
        if len(nodes_subset) > 1:
            x = H.nodes[cluster_label]['x']
            y = H.nodes[cluster_label]['y']
            xy = [(x, y)]
            in_edges = set(H.in_edges(cluster_label, keys=True))
            out_edges = set(H.out_edges(cluster_label, keys=True))
            for (u, v, k) in in_edges | out_edges:
                old_coords = list(H.edges[u, v, k]['geometry'].coords)
                new_coords = xy + old_coords if cluster_label == u else old_coords + xy
                new_geom = LineString(new_coords)
                H.edges[u, v, k]['geometry'] = new_geom
                H.edges[u, v, k]['length'] = new_geom.length
    return H