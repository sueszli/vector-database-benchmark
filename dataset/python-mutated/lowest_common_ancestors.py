"""Algorithms for finding the lowest common ancestor of trees and DAGs."""
from collections import defaultdict
from collections.abc import Mapping, Set
from itertools import combinations_with_replacement
import networkx as nx
from networkx.utils import UnionFind, arbitrary_element, not_implemented_for
__all__ = ['all_pairs_lowest_common_ancestor', 'tree_all_pairs_lowest_common_ancestor', 'lowest_common_ancestor']

@not_implemented_for('undirected')
@nx._dispatch
def all_pairs_lowest_common_ancestor(G, pairs=None):
    if False:
        while True:
            i = 10
    'Return the lowest common ancestor of all pairs or the provided pairs\n\n    Parameters\n    ----------\n    G : NetworkX directed graph\n\n    pairs : iterable of pairs of nodes, optional (default: all pairs)\n        The pairs of nodes of interest.\n        If None, will find the LCA of all pairs of nodes.\n\n    Yields\n    ------\n    ((node1, node2), lca) : 2-tuple\n        Where lca is least common ancestor of node1 and node2.\n        Note that for the default case, the order of the node pair is not considered,\n        e.g. you will not get both ``(a, b)`` and ``(b, a)``\n\n    Raises\n    ------\n    NetworkXPointlessConcept\n        If `G` is null.\n    NetworkXError\n        If `G` is not a DAG.\n\n    Examples\n    --------\n    The default behavior is to yield the lowest common ancestor for all\n    possible combinations of nodes in `G`, including self-pairings:\n\n    >>> G = nx.DiGraph([(0, 1), (0, 3), (1, 2)])\n    >>> dict(nx.all_pairs_lowest_common_ancestor(G))\n    {(0, 0): 0, (0, 1): 0, (0, 3): 0, (0, 2): 0, (1, 1): 1, (1, 3): 0, (1, 2): 1, (3, 3): 3, (3, 2): 0, (2, 2): 2}\n\n    The pairs argument can be used to limit the output to only the\n    specified node pairings:\n\n    >>> dict(nx.all_pairs_lowest_common_ancestor(G, pairs=[(1, 2), (2, 3)]))\n    {(1, 2): 1, (2, 3): 0}\n\n    Notes\n    -----\n    Only defined on non-null directed acyclic graphs.\n\n    See Also\n    --------\n    lowest_common_ancestor\n    '
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError('LCA only defined on directed acyclic graphs.')
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('LCA meaningless on null graphs.')
    if pairs is None:
        pairs = combinations_with_replacement(G, 2)
    else:
        pairs = dict.fromkeys(pairs)
        nodeset = set(G)
        for pair in pairs:
            if set(pair) - nodeset:
                raise nx.NodeNotFound(f'Node(s) {set(pair) - nodeset} from pair {pair} not in G.')

    def generate_lca_from_pairs(G, pairs):
        if False:
            while True:
                i = 10
        ancestor_cache = {}
        for (v, w) in pairs:
            if v not in ancestor_cache:
                ancestor_cache[v] = nx.ancestors(G, v)
                ancestor_cache[v].add(v)
            if w not in ancestor_cache:
                ancestor_cache[w] = nx.ancestors(G, w)
                ancestor_cache[w].add(w)
            common_ancestors = ancestor_cache[v] & ancestor_cache[w]
            if common_ancestors:
                common_ancestor = next(iter(common_ancestors))
                while True:
                    successor = None
                    for lower_ancestor in G.successors(common_ancestor):
                        if lower_ancestor in common_ancestors:
                            successor = lower_ancestor
                            break
                    if successor is None:
                        break
                    common_ancestor = successor
                yield ((v, w), common_ancestor)
    return generate_lca_from_pairs(G, pairs)

@not_implemented_for('undirected')
@nx._dispatch
def lowest_common_ancestor(G, node1, node2, default=None):
    if False:
        i = 10
        return i + 15
    'Compute the lowest common ancestor of the given pair of nodes.\n\n    Parameters\n    ----------\n    G : NetworkX directed graph\n\n    node1, node2 : nodes in the graph.\n\n    default : object\n        Returned if no common ancestor between `node1` and `node2`\n\n    Returns\n    -------\n    The lowest common ancestor of node1 and node2,\n    or default if they have no common ancestors.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph()\n    >>> nx.add_path(G, (0, 1, 2, 3))\n    >>> nx.add_path(G, (0, 4, 3))\n    >>> nx.lowest_common_ancestor(G, 2, 4)\n    0\n\n    See Also\n    --------\n    all_pairs_lowest_common_ancestor'
    ans = list(all_pairs_lowest_common_ancestor(G, pairs=[(node1, node2)]))
    if ans:
        assert len(ans) == 1
        return ans[0][1]
    return default

@not_implemented_for('undirected')
@nx._dispatch
def tree_all_pairs_lowest_common_ancestor(G, root=None, pairs=None):
    if False:
        for i in range(10):
            print('nop')
    'Yield the lowest common ancestor for sets of pairs in a tree.\n\n    Parameters\n    ----------\n    G : NetworkX directed graph (must be a tree)\n\n    root : node, optional (default: None)\n        The root of the subtree to operate on.\n        If None, assume the entire graph has exactly one source and use that.\n\n    pairs : iterable or iterator of pairs of nodes, optional (default: None)\n        The pairs of interest. If None, Defaults to all pairs of nodes\n        under `root` that have a lowest common ancestor.\n\n    Returns\n    -------\n    lcas : generator of tuples `((u, v), lca)` where `u` and `v` are nodes\n        in `pairs` and `lca` is their lowest common ancestor.\n\n    Examples\n    --------\n    >>> import pprint\n    >>> G = nx.DiGraph([(1, 3), (2, 4), (1, 2)])\n    >>> pprint.pprint(dict(nx.tree_all_pairs_lowest_common_ancestor(G)))\n    {(1, 1): 1,\n     (2, 1): 1,\n     (2, 2): 2,\n     (3, 1): 1,\n     (3, 2): 1,\n     (3, 3): 3,\n     (3, 4): 1,\n     (4, 1): 1,\n     (4, 2): 2,\n     (4, 4): 4}\n\n    We can also use `pairs` argument to specify the pairs of nodes for which we\n    want to compute lowest common ancestors. Here is an example:\n\n    >>> dict(nx.tree_all_pairs_lowest_common_ancestor(G, pairs=[(1, 4), (2, 3)]))\n    {(2, 3): 1, (1, 4): 1}\n\n    Notes\n    -----\n    Only defined on non-null trees represented with directed edges from\n    parents to children. Uses Tarjan\'s off-line lowest-common-ancestors\n    algorithm. Runs in time $O(4 \\times (V + E + P))$ time, where 4 is the largest\n    value of the inverse Ackermann function likely to ever come up in actual\n    use, and $P$ is the number of pairs requested (or $V^2$ if all are needed).\n\n    Tarjan, R. E. (1979), "Applications of path compression on balanced trees",\n    Journal of the ACM 26 (4): 690-715, doi:10.1145/322154.322161.\n\n    See Also\n    --------\n    all_pairs_lowest_common_ancestor: similar routine for general DAGs\n    lowest_common_ancestor: just a single pair for general DAGs\n    '
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('LCA meaningless on null graphs.')
    if pairs is not None:
        pair_dict = defaultdict(set)
        if not isinstance(pairs, Mapping | Set):
            pairs = set(pairs)
        for (u, v) in pairs:
            for n in (u, v):
                if n not in G:
                    msg = f'The node {str(n)} is not in the digraph.'
                    raise nx.NodeNotFound(msg)
            pair_dict[u].add(v)
            pair_dict[v].add(u)
    if root is None:
        for (n, deg) in G.in_degree:
            if deg == 0:
                if root is not None:
                    msg = 'No root specified and tree has multiple sources.'
                    raise nx.NetworkXError(msg)
                root = n
            elif deg > 1 and len(G.pred[n]) > 1:
                msg = 'Tree LCA only defined on trees; use DAG routine.'
                raise nx.NetworkXError(msg)
    if root is None:
        raise nx.NetworkXError('Graph contains a cycle.')
    uf = UnionFind()
    ancestors = {}
    for node in G:
        ancestors[node] = uf[node]
    colors = defaultdict(bool)
    for node in nx.dfs_postorder_nodes(G, root):
        colors[node] = True
        for v in pair_dict[node] if pairs is not None else G:
            if colors[v]:
                if pairs is not None and (node, v) in pairs:
                    yield ((node, v), ancestors[uf[v]])
                if pairs is None or (v, node) in pairs:
                    yield ((v, node), ancestors[uf[v]])
        if node != root:
            parent = arbitrary_element(G.pred[node])
            uf.union(parent, node)
            ancestors[uf[parent]] = parent