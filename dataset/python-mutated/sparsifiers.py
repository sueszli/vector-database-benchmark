"""Functions for computing sparsifiers of graphs."""
import math
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
__all__ = ['spanner']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@py_random_state(3)
@nx._dispatch(edge_attrs='weight')
def spanner(G, stretch, weight=None, seed=None):
    if False:
        while True:
            i = 10
    'Returns a spanner of the given graph with the given stretch.\n\n    A spanner of a graph G = (V, E) with stretch t is a subgraph\n    H = (V, E_S) such that E_S is a subset of E and the distance between\n    any pair of nodes in H is at most t times the distance between the\n    nodes in G.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        An undirected simple graph.\n\n    stretch : float\n        The stretch of the spanner.\n\n    weight : object\n        The edge attribute to use as distance.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    NetworkX graph\n        A spanner of the given graph with the given stretch.\n\n    Raises\n    ------\n    ValueError\n        If a stretch less than 1 is given.\n\n    Notes\n    -----\n    This function implements the spanner algorithm by Baswana and Sen,\n    see [1].\n\n    This algorithm is a randomized las vegas algorithm: The expected\n    running time is O(km) where k = (stretch + 1) // 2 and m is the\n    number of edges in G. The returned graph is always a spanner of the\n    given graph with the specified stretch. For weighted graphs the\n    number of edges in the spanner is O(k * n^(1 + 1 / k)) where k is\n    defined as above and n is the number of nodes in G. For unweighted\n    graphs the number of edges is O(n^(1 + 1 / k) + kn).\n\n    References\n    ----------\n    [1] S. Baswana, S. Sen. A Simple and Linear Time Randomized\n    Algorithm for Computing Sparse Spanners in Weighted Graphs.\n    Random Struct. Algorithms 30(4): 532-563 (2007).\n    '
    if stretch < 1:
        raise ValueError('stretch must be at least 1')
    k = (stretch + 1) // 2
    H = nx.empty_graph()
    H.add_nodes_from(G.nodes)
    residual_graph = _setup_residual_graph(G, weight)
    clustering = {v: v for v in G.nodes}
    sample_prob = math.pow(G.number_of_nodes(), -1 / k)
    size_limit = 2 * math.pow(G.number_of_nodes(), 1 + 1 / k)
    i = 0
    while i < k - 1:
        sampled_centers = set()
        for center in set(clustering.values()):
            if seed.random() < sample_prob:
                sampled_centers.add(center)
        edges_to_add = set()
        edges_to_remove = set()
        new_clustering = {}
        for v in residual_graph.nodes:
            if clustering[v] in sampled_centers:
                continue
            (lightest_edge_neighbor, lightest_edge_weight) = _lightest_edge_dicts(residual_graph, clustering, v)
            neighboring_sampled_centers = set(lightest_edge_weight.keys()) & sampled_centers
            if not neighboring_sampled_centers:
                for neighbor in lightest_edge_neighbor.values():
                    edges_to_add.add((v, neighbor))
                for neighbor in residual_graph.adj[v]:
                    edges_to_remove.add((v, neighbor))
            else:
                closest_center = min(neighboring_sampled_centers, key=lightest_edge_weight.get)
                closest_center_weight = lightest_edge_weight[closest_center]
                closest_center_neighbor = lightest_edge_neighbor[closest_center]
                edges_to_add.add((v, closest_center_neighbor))
                new_clustering[v] = closest_center
                for (center, edge_weight) in lightest_edge_weight.items():
                    if edge_weight < closest_center_weight:
                        neighbor = lightest_edge_neighbor[center]
                        edges_to_add.add((v, neighbor))
                for neighbor in residual_graph.adj[v]:
                    neighbor_cluster = clustering[neighbor]
                    neighbor_weight = lightest_edge_weight[neighbor_cluster]
                    if neighbor_cluster == closest_center or neighbor_weight < closest_center_weight:
                        edges_to_remove.add((v, neighbor))
        if len(edges_to_add) > size_limit:
            continue
        i = i + 1
        for (u, v) in edges_to_add:
            _add_edge_to_spanner(H, residual_graph, u, v, weight)
        residual_graph.remove_edges_from(edges_to_remove)
        for (node, center) in clustering.items():
            if center in sampled_centers:
                new_clustering[node] = center
        clustering = new_clustering
        for u in residual_graph.nodes:
            for v in list(residual_graph.adj[u]):
                if clustering[u] == clustering[v]:
                    residual_graph.remove_edge(u, v)
        for v in list(residual_graph.nodes):
            if v not in clustering:
                residual_graph.remove_node(v)
    for v in residual_graph.nodes:
        (lightest_edge_neighbor, _) = _lightest_edge_dicts(residual_graph, clustering, v)
        for neighbor in lightest_edge_neighbor.values():
            _add_edge_to_spanner(H, residual_graph, v, neighbor, weight)
    return H

def _setup_residual_graph(G, weight):
    if False:
        i = 10
        return i + 15
    "Setup residual graph as a copy of G with unique edges weights.\n\n    The node set of the residual graph corresponds to the set V' from\n    the Baswana-Sen paper and the edge set corresponds to the set E'\n    from the paper.\n\n    This function associates distinct weights to the edges of the\n    residual graph (even for unweighted input graphs), as required by\n    the algorithm.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        An undirected simple graph.\n\n    weight : object\n        The edge attribute to use as distance.\n\n    Returns\n    -------\n    NetworkX graph\n        The residual graph used for the Baswana-Sen algorithm.\n    "
    residual_graph = G.copy()
    for (u, v) in G.edges():
        if not weight:
            residual_graph[u][v]['weight'] = (id(u), id(v))
        else:
            residual_graph[u][v]['weight'] = (G[u][v][weight], id(u), id(v))
    return residual_graph

def _lightest_edge_dicts(residual_graph, clustering, node):
    if False:
        while True:
            i = 10
    'Find the lightest edge to each cluster.\n\n    Searches for the minimum-weight edge to each cluster adjacent to\n    the given node.\n\n    Parameters\n    ----------\n    residual_graph : NetworkX graph\n        The residual graph used by the Baswana-Sen algorithm.\n\n    clustering : dictionary\n        The current clustering of the nodes.\n\n    node : node\n        The node from which the search originates.\n\n    Returns\n    -------\n    lightest_edge_neighbor, lightest_edge_weight : dictionary, dictionary\n        lightest_edge_neighbor is a dictionary that maps a center C to\n        a node v in the corresponding cluster such that the edge from\n        the given node to v is the lightest edge from the given node to\n        any node in cluster. lightest_edge_weight maps a center C to the\n        weight of the aforementioned edge.\n\n    Notes\n    -----\n    If a cluster has no node that is adjacent to the given node in the\n    residual graph then the center of the cluster is not a key in the\n    returned dictionaries.\n    '
    lightest_edge_neighbor = {}
    lightest_edge_weight = {}
    for neighbor in residual_graph.adj[node]:
        neighbor_center = clustering[neighbor]
        weight = residual_graph[node][neighbor]['weight']
        if neighbor_center not in lightest_edge_weight or weight < lightest_edge_weight[neighbor_center]:
            lightest_edge_neighbor[neighbor_center] = neighbor
            lightest_edge_weight[neighbor_center] = weight
    return (lightest_edge_neighbor, lightest_edge_weight)

def _add_edge_to_spanner(H, residual_graph, u, v, weight):
    if False:
        print('Hello World!')
    'Add the edge {u, v} to the spanner H and take weight from\n    the residual graph.\n\n    Parameters\n    ----------\n    H : NetworkX graph\n        The spanner under construction.\n\n    residual_graph : NetworkX graph\n        The residual graph used by the Baswana-Sen algorithm. The weight\n        for the edge is taken from this graph.\n\n    u : node\n        One endpoint of the edge.\n\n    v : node\n        The other endpoint of the edge.\n\n    weight : object\n        The edge attribute to use as distance.\n    '
    H.add_edge(u, v)
    if weight:
        H[u][v][weight] = residual_graph[u][v]['weight'][0]