import networkx as nx
__all__ = ['convert_node_labels_to_integers', 'relabel_nodes']

@nx._dispatch(preserve_all_attrs=True)
def relabel_nodes(G, mapping, copy=True):
    if False:
        i = 10
        return i + 15
    'Relabel the nodes of the graph G according to a given mapping.\n\n    The original node ordering may not be preserved if `copy` is `False` and the\n    mapping includes overlap between old and new labels.\n\n    Parameters\n    ----------\n    G : graph\n       A NetworkX graph\n\n    mapping : dictionary\n       A dictionary with the old labels as keys and new labels as values.\n       A partial mapping is allowed. Mapping 2 nodes to a single node is allowed.\n       Any non-node keys in the mapping are ignored.\n\n    copy : bool (optional, default=True)\n       If True return a copy, or if False relabel the nodes in place.\n\n    Examples\n    --------\n    To create a new graph with nodes relabeled according to a given\n    dictionary:\n\n    >>> G = nx.path_graph(3)\n    >>> sorted(G)\n    [0, 1, 2]\n    >>> mapping = {0: "a", 1: "b", 2: "c"}\n    >>> H = nx.relabel_nodes(G, mapping)\n    >>> sorted(H)\n    [\'a\', \'b\', \'c\']\n\n    Nodes can be relabeled with any hashable object, including numbers\n    and strings:\n\n    >>> import string\n    >>> G = nx.path_graph(26)  # nodes are integers 0 through 25\n    >>> sorted(G)[:3]\n    [0, 1, 2]\n    >>> mapping = dict(zip(G, string.ascii_lowercase))\n    >>> G = nx.relabel_nodes(G, mapping)  # nodes are characters a through z\n    >>> sorted(G)[:3]\n    [\'a\', \'b\', \'c\']\n    >>> mapping = dict(zip(G, range(1, 27)))\n    >>> G = nx.relabel_nodes(G, mapping)  # nodes are integers 1 through 26\n    >>> sorted(G)[:3]\n    [1, 2, 3]\n\n    To perform a partial in-place relabeling, provide a dictionary\n    mapping only a subset of the nodes, and set the `copy` keyword\n    argument to False:\n\n    >>> G = nx.path_graph(3)  # nodes 0-1-2\n    >>> mapping = {0: "a", 1: "b"}  # 0->\'a\' and 1->\'b\'\n    >>> G = nx.relabel_nodes(G, mapping, copy=False)\n    >>> sorted(G, key=str)\n    [2, \'a\', \'b\']\n\n    A mapping can also be given as a function:\n\n    >>> G = nx.path_graph(3)\n    >>> H = nx.relabel_nodes(G, lambda x: x ** 2)\n    >>> list(H)\n    [0, 1, 4]\n\n    In a multigraph, relabeling two or more nodes to the same new node\n    will retain all edges, but may change the edge keys in the process:\n\n    >>> G = nx.MultiGraph()\n    >>> G.add_edge(0, 1, value="a")  # returns the key for this edge\n    0\n    >>> G.add_edge(0, 2, value="b")\n    0\n    >>> G.add_edge(0, 3, value="c")\n    0\n    >>> mapping = {1: 4, 2: 4, 3: 4}\n    >>> H = nx.relabel_nodes(G, mapping, copy=True)\n    >>> print(H[0])\n    {4: {0: {\'value\': \'a\'}, 1: {\'value\': \'b\'}, 2: {\'value\': \'c\'}}}\n\n    This works for in-place relabeling too:\n\n    >>> G = nx.relabel_nodes(G, mapping, copy=False)\n    >>> print(G[0])\n    {4: {0: {\'value\': \'a\'}, 1: {\'value\': \'b\'}, 2: {\'value\': \'c\'}}}\n\n    Notes\n    -----\n    Only the nodes specified in the mapping will be relabeled.\n    Any non-node keys in the mapping are ignored.\n\n    The keyword setting copy=False modifies the graph in place.\n    Relabel_nodes avoids naming collisions by building a\n    directed graph from ``mapping`` which specifies the order of\n    relabelings. Naming collisions, such as a->b, b->c, are ordered\n    such that "b" gets renamed to "c" before "a" gets renamed "b".\n    In cases of circular mappings (e.g. a->b, b->a), modifying the\n    graph is not possible in-place and an exception is raised.\n    In that case, use copy=True.\n\n    If a relabel operation on a multigraph would cause two or more\n    edges to have the same source, target and key, the second edge must\n    be assigned a new key to retain all edges. The new key is set\n    to the lowest non-negative integer not already used as a key\n    for edges between these two nodes. Note that this means non-numeric\n    keys may be replaced by numeric keys.\n\n    See Also\n    --------\n    convert_node_labels_to_integers\n    '
    m = {n: mapping(n) for n in G} if callable(mapping) else mapping
    if copy:
        return _relabel_copy(G, m)
    else:
        return _relabel_inplace(G, m)

def _relabel_inplace(G, mapping):
    if False:
        while True:
            i = 10
    if len(mapping.keys() & mapping.values()) > 0:
        D = nx.DiGraph(list(mapping.items()))
        D.remove_edges_from(nx.selfloop_edges(D))
        try:
            nodes = reversed(list(nx.topological_sort(D)))
        except nx.NetworkXUnfeasible as err:
            raise nx.NetworkXUnfeasible('The node label sets are overlapping and no ordering can resolve the mapping. Use copy=True.') from err
    else:
        nodes = [n for n in G if n in mapping]
    multigraph = G.is_multigraph()
    directed = G.is_directed()
    for old in nodes:
        try:
            new = mapping[old]
            G.add_node(new, **G.nodes[old])
        except KeyError:
            continue
        if new == old:
            continue
        if multigraph:
            new_edges = [(new, new if old == target else target, key, data) for (_, target, key, data) in G.edges(old, data=True, keys=True)]
            if directed:
                new_edges += [(new if old == source else source, new, key, data) for (source, _, key, data) in G.in_edges(old, data=True, keys=True)]
            seen = set()
            for (i, (source, target, key, data)) in enumerate(new_edges):
                if target in G[source] and key in G[source][target]:
                    new_key = 0 if not isinstance(key, int | float) else key
                    while new_key in G[source][target] or (target, new_key) in seen:
                        new_key += 1
                    new_edges[i] = (source, target, new_key, data)
                    seen.add((target, new_key))
        else:
            new_edges = [(new, new if old == target else target, data) for (_, target, data) in G.edges(old, data=True)]
            if directed:
                new_edges += [(new if old == source else source, new, data) for (source, _, data) in G.in_edges(old, data=True)]
        G.remove_node(old)
        G.add_edges_from(new_edges)
    return G

def _relabel_copy(G, mapping):
    if False:
        for i in range(10):
            print('nop')
    H = G.__class__()
    H.add_nodes_from((mapping.get(n, n) for n in G))
    H._node.update(((mapping.get(n, n), d.copy()) for (n, d) in G.nodes.items()))
    if G.is_multigraph():
        new_edges = [(mapping.get(n1, n1), mapping.get(n2, n2), k, d.copy()) for (n1, n2, k, d) in G.edges(keys=True, data=True)]
        undirected = not G.is_directed()
        seen_edges = set()
        for (i, (source, target, key, data)) in enumerate(new_edges):
            while (source, target, key) in seen_edges:
                if not isinstance(key, int | float):
                    key = 0
                key += 1
            seen_edges.add((source, target, key))
            if undirected:
                seen_edges.add((target, source, key))
            new_edges[i] = (source, target, key, data)
        H.add_edges_from(new_edges)
    else:
        H.add_edges_from(((mapping.get(n1, n1), mapping.get(n2, n2), d.copy()) for (n1, n2, d) in G.edges(data=True)))
    H.graph.update(G.graph)
    return H

@nx._dispatch(preserve_edge_attrs=True, preserve_node_attrs=True, preserve_graph_attrs=True)
def convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None):
    if False:
        while True:
            i = 10
    'Returns a copy of the graph G with the nodes relabeled using\n    consecutive integers.\n\n    Parameters\n    ----------\n    G : graph\n       A NetworkX graph\n\n    first_label : int, optional (default=0)\n       An integer specifying the starting offset in numbering nodes.\n       The new integer labels are numbered first_label, ..., n-1+first_label.\n\n    ordering : string\n       "default" : inherit node ordering from G.nodes()\n       "sorted"  : inherit node ordering from sorted(G.nodes())\n       "increasing degree" : nodes are sorted by increasing degree\n       "decreasing degree" : nodes are sorted by decreasing degree\n\n    label_attribute : string, optional (default=None)\n       Name of node attribute to store old label.  If None no attribute\n       is created.\n\n    Notes\n    -----\n    Node and edge attribute data are copied to the new (relabeled) graph.\n\n    There is no guarantee that the relabeling of nodes to integers will\n    give the same two integers for two (even identical graphs).\n    Use the `ordering` argument to try to preserve the order.\n\n    See Also\n    --------\n    relabel_nodes\n    '
    N = G.number_of_nodes() + first_label
    if ordering == 'default':
        mapping = dict(zip(G.nodes(), range(first_label, N)))
    elif ordering == 'sorted':
        nlist = sorted(G.nodes())
        mapping = dict(zip(nlist, range(first_label, N)))
    elif ordering == 'increasing degree':
        dv_pairs = [(d, n) for (n, d) in G.degree()]
        dv_pairs.sort()
        mapping = dict(zip([n for (d, n) in dv_pairs], range(first_label, N)))
    elif ordering == 'decreasing degree':
        dv_pairs = [(d, n) for (n, d) in G.degree()]
        dv_pairs.sort()
        dv_pairs.reverse()
        mapping = dict(zip([n for (d, n) in dv_pairs], range(first_label, N)))
    else:
        raise nx.NetworkXError(f'Unknown node ordering: {ordering}')
    H = relabel_nodes(G, mapping)
    if label_attribute is not None:
        nx.set_node_attributes(H, {v: k for (k, v) in mapping.items()}, label_attribute)
    return H