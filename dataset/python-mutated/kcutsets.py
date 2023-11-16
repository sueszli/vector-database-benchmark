"""
Kanevsky all minimum node k cutsets algorithm.
"""
import copy
from collections import defaultdict
from itertools import combinations
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow import build_residual_network, edmonds_karp, shortest_augmenting_path
from .utils import build_auxiliary_node_connectivity
default_flow_func = edmonds_karp
__all__ = ['all_node_cuts']

@nx._dispatch
def all_node_cuts(G, k=None, flow_func=None):
    if False:
        for i in range(10):
            print('nop')
    "Returns all minimum k cutsets of an undirected graph G.\n\n    This implementation is based on Kanevsky's algorithm [1]_ for finding all\n    minimum-size node cut-sets of an undirected graph G; ie the set (or sets)\n    of nodes of cardinality equal to the node connectivity of G. Thus if\n    removed, would break G into two or more connected components.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Undirected graph\n\n    k : Integer\n        Node connectivity of the input graph. If k is None, then it is\n        computed. Default value: None.\n\n    flow_func : function\n        Function to perform the underlying flow computations. Default value is\n        :func:`~networkx.algorithms.flow.edmonds_karp`. This function performs\n        better in sparse graphs with right tailed degree distributions.\n        :func:`~networkx.algorithms.flow.shortest_augmenting_path` will\n        perform better in denser graphs.\n\n\n    Returns\n    -------\n    cuts : a generator of node cutsets\n        Each node cutset has cardinality equal to the node connectivity of\n        the input graph.\n\n    Examples\n    --------\n    >>> # A two-dimensional grid graph has 4 cutsets of cardinality 2\n    >>> G = nx.grid_2d_graph(5, 5)\n    >>> cutsets = list(nx.all_node_cuts(G))\n    >>> len(cutsets)\n    4\n    >>> all(2 == len(cutset) for cutset in cutsets)\n    True\n    >>> nx.node_connectivity(G)\n    2\n\n    Notes\n    -----\n    This implementation is based on the sequential algorithm for finding all\n    minimum-size separating vertex sets in a graph [1]_. The main idea is to\n    compute minimum cuts using local maximum flow computations among a set\n    of nodes of highest degree and all other non-adjacent nodes in the Graph.\n    Once we find a minimum cut, we add an edge between the high degree\n    node and the target node of the local maximum flow computation to make\n    sure that we will not find that minimum cut again.\n\n    See also\n    --------\n    node_connectivity\n    edmonds_karp\n    shortest_augmenting_path\n\n    References\n    ----------\n    .. [1]  Kanevsky, A. (1993). Finding all minimum-size separating vertex\n            sets in a graph. Networks 23(6), 533--541.\n            http://onlinelibrary.wiley.com/doi/10.1002/net.3230230604/abstract\n\n    "
    if not nx.is_connected(G):
        raise nx.NetworkXError('Input graph is disconnected.')
    if nx.density(G) == 1:
        for cut_set in combinations(G, len(G) - 1):
            yield set(cut_set)
        return
    seen = []
    H = build_auxiliary_node_connectivity(G)
    H_nodes = H.nodes
    mapping = H.graph['mapping']
    original_H_pred = copy.copy(H._pred)
    R = build_residual_network(H, 'capacity')
    kwargs = {'capacity': 'capacity', 'residual': R}
    if flow_func is None:
        flow_func = default_flow_func
    if flow_func is shortest_augmenting_path:
        kwargs['two_phase'] = True
    if k is None:
        k = nx.node_connectivity(G, flow_func=flow_func)
    X = {n for (n, d) in sorted(G.degree(), key=itemgetter(1), reverse=True)[:k]}
    if _is_separating_set(G, X):
        seen.append(X)
        yield X
    for x in X:
        non_adjacent = set(G) - X - set(G[x])
        for v in non_adjacent:
            R = flow_func(H, f'{mapping[x]}B', f'{mapping[v]}A', **kwargs)
            flow_value = R.graph['flow_value']
            if flow_value == k:
                E1 = flowed_edges = [(u, w) for (u, w, d) in R.edges(data=True) if d['flow'] != 0]
                VE1 = incident_nodes = {n for edge in E1 for n in edge}
                saturated_edges = [(u, w, d) for (u, w, d) in R.edges(data=True) if d['capacity'] == d['flow'] or d['capacity'] == 0]
                R.remove_edges_from(saturated_edges)
                R_closure = nx.transitive_closure(R)
                L = nx.condensation(R)
                cmap = L.graph['mapping']
                inv_cmap = defaultdict(list)
                for (n, scc) in cmap.items():
                    inv_cmap[scc].append(n)
                VE1 = {cmap[n] for n in VE1}
                for antichain in nx.antichains(L):
                    if not set(antichain).issubset(VE1):
                        continue
                    S = set()
                    for scc in antichain:
                        S.update(inv_cmap[scc])
                    S_ancestors = set()
                    for n in S:
                        S_ancestors.update(R_closure._pred[n])
                    S.update(S_ancestors)
                    if f'{mapping[x]}B' not in S or f'{mapping[v]}A' in S:
                        continue
                    cutset = set()
                    for u in S:
                        cutset.update(((u, w) for w in original_H_pred[u] if w not in S))
                    if any((H_nodes[u]['id'] != H_nodes[w]['id'] for (u, w) in cutset)):
                        continue
                    node_cut = {H_nodes[u]['id'] for (u, _) in cutset}
                    if len(node_cut) == k:
                        if x in node_cut or v in node_cut:
                            continue
                        if node_cut not in seen:
                            yield node_cut
                            seen.append(node_cut)
                H.add_edge(f'{mapping[x]}B', f'{mapping[v]}A', capacity=1)
                H.add_edge(f'{mapping[v]}B', f'{mapping[x]}A', capacity=1)
                R.add_edge(f'{mapping[x]}B', f'{mapping[v]}A', capacity=1)
                R.add_edge(f'{mapping[v]}A', f'{mapping[x]}B', capacity=0)
                R.add_edge(f'{mapping[v]}B', f'{mapping[x]}A', capacity=1)
                R.add_edge(f'{mapping[x]}A', f'{mapping[v]}B', capacity=0)
                R.add_edges_from(saturated_edges)

def _is_separating_set(G, cut):
    if False:
        return 10
    'Assumes that the input graph is connected'
    if len(cut) == len(G) - 1:
        return True
    H = nx.restricted_view(G, cut, [])
    if nx.is_connected(H):
        return False
    return True