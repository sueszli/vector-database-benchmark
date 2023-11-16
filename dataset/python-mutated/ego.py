"""
Ego graph.
"""
__all__ = ['ego_graph']
import networkx as nx

@nx._dispatch(edge_attrs='distance')
def ego_graph(G, n, radius=1, center=True, undirected=False, distance=None):
    if False:
        while True:
            i = 10
    'Returns induced subgraph of neighbors centered at node n within\n    a given radius.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX Graph or DiGraph\n\n    n : node\n      A single node\n\n    radius : number, optional\n      Include all neighbors of distance<=radius from n.\n\n    center : bool, optional\n      If False, do not include center node in graph\n\n    undirected : bool, optional\n      If True use both in- and out-neighbors of directed graphs.\n\n    distance : key, optional\n      Use specified edge data key as distance.  For example, setting\n      distance=\'weight\' will use the edge weight to measure the\n      distance from the node n.\n\n    Notes\n    -----\n    For directed graphs D this produces the "out" neighborhood\n    or successors.  If you want the neighborhood of predecessors\n    first reverse the graph with D.reverse().  If you want both\n    directions use the keyword argument undirected=True.\n\n    Node, edge, and graph attributes are copied to the returned subgraph.\n    '
    if undirected:
        if distance is not None:
            (sp, _) = nx.single_source_dijkstra(G.to_undirected(), n, cutoff=radius, weight=distance)
        else:
            sp = dict(nx.single_source_shortest_path_length(G.to_undirected(), n, cutoff=radius))
    elif distance is not None:
        (sp, _) = nx.single_source_dijkstra(G, n, cutoff=radius, weight=distance)
    else:
        sp = dict(nx.single_source_shortest_path_length(G, n, cutoff=radius))
    H = G.subgraph(sp).copy()
    if not center:
        H.remove_node(n)
    return H