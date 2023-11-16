"""
Algorithms for finding optimum branchings and spanning arborescences.

This implementation is based on:

    J. Edmonds, Optimum branchings, J. Res. Natl. Bur. Standards 71B (1967),
    233–240. URL: http://archive.org/details/jresv71Bn4p233

"""
import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
__all__ = ['branching_weight', 'greedy_branching', 'maximum_branching', 'minimum_branching', 'minimal_branching', 'maximum_spanning_arborescence', 'minimum_spanning_arborescence', 'ArborescenceIterator', 'Edmonds']
KINDS = {'max', 'min'}
STYLES = {'branching': 'branching', 'arborescence': 'arborescence', 'spanning arborescence': 'arborescence'}
INF = float('inf')

@py_random_state(1)
def random_string(L=15, seed=None):
    if False:
        for i in range(10):
            print('nop')
    return ''.join([seed.choice(string.ascii_letters) for n in range(L)])

def _min_weight(weight):
    if False:
        while True:
            i = 10
    return -weight

def _max_weight(weight):
    if False:
        for i in range(10):
            print('nop')
    return weight

@nx._dispatch(edge_attrs={'attr': 'default'})
def branching_weight(G, attr='weight', default=1):
    if False:
        while True:
            i = 10
    '\n    Returns the total weight of a branching.\n\n    You must access this function through the networkx.algorithms.tree module.\n\n    Parameters\n    ----------\n    G : DiGraph\n        The directed graph.\n    attr : str\n        The attribute to use as weights. If None, then each edge will be\n        treated equally with a weight of 1.\n    default : float\n        When `attr` is not None, then if an edge does not have that attribute,\n        `default` specifies what value it should take.\n\n    Returns\n    -------\n    weight: int or float\n        The total weight of the branching.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph()\n    >>> G.add_weighted_edges_from([(0, 1, 2), (1, 2, 4), (2, 3, 3), (3, 4, 2)])\n    >>> nx.tree.branching_weight(G)\n    11\n\n    '
    return sum((edge[2].get(attr, default) for edge in G.edges(data=True)))

@py_random_state(4)
@nx._dispatch(edge_attrs={'attr': 'default'})
def greedy_branching(G, attr='weight', default=1, kind='max', seed=None):
    if False:
        while True:
            i = 10
    "\n    Returns a branching obtained through a greedy algorithm.\n\n    This algorithm is wrong, and cannot give a proper optimal branching.\n    However, we include it for pedagogical reasons, as it can be helpful to\n    see what its outputs are.\n\n    The output is a branching, and possibly, a spanning arborescence. However,\n    it is not guaranteed to be optimal in either case.\n\n    Parameters\n    ----------\n    G : DiGraph\n        The directed graph to scan.\n    attr : str\n        The attribute to use as weights. If None, then each edge will be\n        treated equally with a weight of 1.\n    default : float\n        When `attr` is not None, then if an edge does not have that attribute,\n        `default` specifies what value it should take.\n    kind : str\n        The type of optimum to search for: 'min' or 'max' greedy branching.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    B : directed graph\n        The greedily obtained branching.\n\n    "
    if kind not in KINDS:
        raise nx.NetworkXException('Unknown value for `kind`.')
    if kind == 'min':
        reverse = False
    else:
        reverse = True
    if attr is None:
        attr = random_string(seed=seed)
    edges = [(u, v, data.get(attr, default)) for (u, v, data) in G.edges(data=True)]
    try:
        edges.sort(key=itemgetter(2, 0, 1), reverse=reverse)
    except TypeError:
        edges.sort(key=itemgetter(2), reverse=reverse)
    B = nx.DiGraph()
    B.add_nodes_from(G)
    uf = nx.utils.UnionFind()
    for (i, (u, v, w)) in enumerate(edges):
        if uf[u] == uf[v]:
            continue
        elif B.in_degree(v) == 1:
            continue
        else:
            data = {}
            if attr is not None:
                data[attr] = w
            B.add_edge(u, v, **data)
            uf.union(u, v)
    return B

class MultiDiGraph_EdgeKey(nx.MultiDiGraph):
    """
    MultiDiGraph which assigns unique keys to every edge.

    Adds a dictionary edge_index which maps edge keys to (u, v, data) tuples.

    This is not a complete implementation. For Edmonds algorithm, we only use
    add_node and add_edge, so that is all that is implemented here. During
    additions, any specified keys are ignored---this means that you also
    cannot update edge attributes through add_node and add_edge.

    Why do we need this? Edmonds algorithm requires that we track edges, even
    as we change the head and tail of an edge, and even changing the weight
    of edges. We must reliably track edges across graph mutations.
    """

    def __init__(self, incoming_graph_data=None, **attr):
        if False:
            for i in range(10):
                print('nop')
        cls = super()
        cls.__init__(incoming_graph_data=incoming_graph_data, **attr)
        self._cls = cls
        self.edge_index = {}
        import warnings
        msg = 'MultiDiGraph_EdgeKey has been deprecated and will be removed in NetworkX 3.4.'
        warnings.warn(msg, DeprecationWarning)

    def remove_node(self, n):
        if False:
            return 10
        keys = set()
        for keydict in self.pred[n].values():
            keys.update(keydict)
        for keydict in self.succ[n].values():
            keys.update(keydict)
        for key in keys:
            del self.edge_index[key]
        self._cls.remove_node(n)

    def remove_nodes_from(self, nbunch):
        if False:
            print('Hello World!')
        for n in nbunch:
            self.remove_node(n)

    def add_edge(self, u_for_edge, v_for_edge, key_for_edge, **attr):
        if False:
            for i in range(10):
                print('nop')
        '\n        Key is now required.\n\n        '
        (u, v, key) = (u_for_edge, v_for_edge, key_for_edge)
        if key in self.edge_index:
            (uu, vv, _) = self.edge_index[key]
            if u != uu or v != vv:
                raise Exception(f'Key {key!r} is already in use.')
        self._cls.add_edge(u, v, key, **attr)
        self.edge_index[key] = (u, v, self.succ[u][v][key])

    def add_edges_from(self, ebunch_to_add, **attr):
        if False:
            print('Hello World!')
        for (u, v, k, d) in ebunch_to_add:
            self.add_edge(u, v, k, **d)

    def remove_edge_with_key(self, key):
        if False:
            return 10
        try:
            (u, v, _) = self.edge_index[key]
        except KeyError as err:
            raise KeyError(f'Invalid edge key {key!r}') from err
        else:
            del self.edge_index[key]
            self._cls.remove_edge(u, v, key)

    def remove_edges_from(self, ebunch):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

def get_path(G, u, v):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the edge keys of the unique path between u and v.\n\n    This is not a generic function. G must be a branching and an instance of\n    MultiDiGraph_EdgeKey.\n\n    '
    nodes = nx.shortest_path(G, u, v)

    def first_key(i, vv):
        if False:
            for i in range(10):
                print('nop')
        keys = G[nodes[i]][vv].keys()
        keys = list(keys)
        return keys[0]
    edges = [first_key(i, vv) for (i, vv) in enumerate(nodes[1:])]
    return (nodes, edges)

class Edmonds:
    """
    Edmonds algorithm [1]_ for finding optimal branchings and spanning
    arborescences.

    This algorithm can find both minimum and maximum spanning arborescences and
    branchings.

    Notes
    -----
    While this algorithm can find a minimum branching, since it isn't required
    to be spanning, the minimum branching is always from the set of negative
    weight edges which is most likely the empty set for most graphs.

    References
    ----------
    .. [1] J. Edmonds, Optimum Branchings, Journal of Research of the National
           Bureau of Standards, 1967, Vol. 71B, p.233-240,
           https://archive.org/details/jresv71Bn4p233

    """

    def __init__(self, G, seed=None):
        if False:
            i = 10
            return i + 15
        self.G_original = G
        self.store = True
        self.edges = []
        self.template = random_string(seed=seed) + '_{0}'
        import warnings
        msg = 'Edmonds has been deprecated and will be removed in NetworkX 3.4. Please use the appropriate minimum or maximum branching or arborescence function directly.'
        warnings.warn(msg, DeprecationWarning)

    def _init(self, attr, default, kind, style, preserve_attrs, seed, partition):
        if False:
            print('Hello World!')
        '\n        So we need the code in _init and find_optimum to successfully run edmonds algorithm.\n        Responsibilities of the _init function:\n        - Check that the kind argument is in {min, max} or raise a NetworkXException.\n        - Transform the graph if we need a minimum arborescence/branching.\n          - The current method is to map weight -> -weight. This is NOT a good approach since\n            the algorithm can and does choose to ignore negative weights when creating a branching\n            since that is always optimal when maximzing the weights. I think we should set the edge\n            weights to be (max_weight + 1) - edge_weight.\n        - Transform the graph into a MultiDiGraph, adding the partition information and potoentially\n          other edge attributes if we set preserve_attrs = True.\n        - Setup the buckets and union find data structures required for the algorithm.\n        '
        if kind not in KINDS:
            raise nx.NetworkXException('Unknown value for `kind`.')
        self.attr = attr
        self.default = default
        self.kind = kind
        self.style = style
        if kind == 'min':
            self.trans = trans = _min_weight
        else:
            self.trans = trans = _max_weight
        if attr is None:
            attr = random_string(seed=seed)
        self._attr = attr
        self.candidate_attr = 'candidate_' + random_string(seed=seed)
        self.G = G = MultiDiGraph_EdgeKey()
        for (key, (u, v, data)) in enumerate(self.G_original.edges(data=True)):
            d = {attr: trans(data.get(attr, default))}
            if data.get(partition) is not None:
                d[partition] = data.get(partition)
            if preserve_attrs:
                for (d_k, d_v) in data.items():
                    if d_k != attr:
                        d[d_k] = d_v
            G.add_edge(u, v, key, **d)
        self.level = 0
        self.B = MultiDiGraph_EdgeKey()
        self.B.edge_index = {}
        self.graphs = []
        self.branchings = []
        self.uf = nx.utils.UnionFind()
        self.circuits = []
        self.minedge_circuit = []

    def find_optimum(self, attr='weight', default=1, kind='max', style='branching', preserve_attrs=False, partition=None, seed=None):
        if False:
            print('Hello World!')
        "\n        Returns a branching from G.\n\n        Parameters\n        ----------\n        attr : str\n            The edge attribute used to in determining optimality.\n        default : float\n            The value of the edge attribute used if an edge does not have\n            the attribute `attr`.\n        kind : {'min', 'max'}\n            The type of optimum to search for, either 'min' or 'max'.\n        style : {'branching', 'arborescence'}\n            If 'branching', then an optimal branching is found. If `style` is\n            'arborescence', then a branching is found, such that if the\n            branching is also an arborescence, then the branching is an\n            optimal spanning arborescences. A given graph G need not have\n            an optimal spanning arborescence.\n        preserve_attrs : bool\n            If True, preserve the other edge attributes of the original\n            graph (that are not the one passed to `attr`)\n        partition : str\n            The edge attribute holding edge partition data. Used in the\n            spanning arborescence iterator.\n        seed : integer, random_state, or None (default)\n            Indicator of random number generation state.\n            See :ref:`Randomness<randomness>`.\n\n        Returns\n        -------\n        H : (multi)digraph\n            The branching.\n\n        "
        self._init(attr, default, kind, style, preserve_attrs, seed, partition)
        uf = self.uf
        (G, B) = (self.G, self.B)
        D = set()
        nodes = iter(list(G.nodes()))
        attr = self._attr
        G_pred = G.pred

        def desired_edge(v):
            if False:
                return 10
            '\n            Find the edge directed toward v with maximal weight.\n\n            If an edge partition exists in this graph, return the included edge\n            if it exists and no not return any excluded edges. There can only\n            be one included edge for each vertex otherwise the edge partition is\n            empty.\n            '
            edge = None
            weight = -INF
            for (u, _, key, data) in G.in_edges(v, data=True, keys=True):
                if data.get(partition) == nx.EdgePartition.EXCLUDED:
                    continue
                new_weight = data[attr]
                if data.get(partition) == nx.EdgePartition.INCLUDED:
                    weight = new_weight
                    edge = (u, v, key, new_weight, data)
                    return (edge, weight)
                if new_weight > weight:
                    weight = new_weight
                    edge = (u, v, key, new_weight, data)
            return (edge, weight)
        while True:
            try:
                v = next(nodes)
            except StopIteration:
                assert len(G) == len(B)
                if len(B):
                    assert is_branching(B)
                if self.store:
                    self.graphs.append(G.copy())
                    self.branchings.append(B.copy())
                    self.circuits.append([])
                    self.minedge_circuit.append(None)
                break
            else:
                if v in D:
                    continue
            D.add(v)
            B.add_node(v)
            (edge, weight) = desired_edge(v)
            if edge is None:
                continue
            else:
                u = edge[0]
                if uf[u] == uf[v]:
                    (Q_nodes, Q_edges) = get_path(B, v, u)
                    Q_edges.append(edge[2])
                else:
                    (Q_nodes, Q_edges) = (None, None)
                if self.style == 'branching' and weight <= 0:
                    acceptable = False
                else:
                    acceptable = True
                if acceptable:
                    dd = {attr: weight}
                    if edge[4].get(partition) is not None:
                        dd[partition] = edge[4].get(partition)
                    B.add_edge(u, v, edge[2], **dd)
                    G[u][v][edge[2]][self.candidate_attr] = True
                    uf.union(u, v)
                    if Q_edges is not None:
                        minweight = INF
                        minedge = None
                        Q_incoming_weight = {}
                        for edge_key in Q_edges:
                            (u, v, data) = B.edge_index[edge_key]
                            w = data[attr]
                            Q_incoming_weight[v] = w
                            if data.get(partition) == nx.EdgePartition.INCLUDED:
                                continue
                            if w < minweight:
                                minweight = w
                                minedge = edge_key
                        self.circuits.append(Q_edges)
                        self.minedge_circuit.append(minedge)
                        if self.store:
                            self.graphs.append(G.copy())
                        self.branchings.append(B.copy())
                        new_node = self.template.format(self.level)
                        G.add_node(new_node)
                        new_edges = []
                        for (u, v, key, data) in G.edges(data=True, keys=True):
                            if u in Q_incoming_weight:
                                if v in Q_incoming_weight:
                                    continue
                                else:
                                    dd = data.copy()
                                    new_edges.append((new_node, v, key, dd))
                            elif v in Q_incoming_weight:
                                w = data[attr]
                                w += minweight - Q_incoming_weight[v]
                                dd = data.copy()
                                dd[attr] = w
                                new_edges.append((u, new_node, key, dd))
                            else:
                                continue
                        G.remove_nodes_from(Q_nodes)
                        B.remove_nodes_from(Q_nodes)
                        D.difference_update(set(Q_nodes))
                        for (u, v, key, data) in new_edges:
                            G.add_edge(u, v, key, **data)
                            if self.candidate_attr in data:
                                del data[self.candidate_attr]
                                B.add_edge(u, v, key, **data)
                                uf.union(u, v)
                        nodes = iter(list(G.nodes()))
                        self.level += 1
        H = self.G_original.__class__()

        def is_root(G, u, edgekeys):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Returns True if `u` is a root node in G.\n\n            Node `u` will be a root node if its in-degree, restricted to the\n            specified edges, is equal to 0.\n\n            '
            if u not in G:
                raise Exception(f'{u!r} not in G')
            for v in G.pred[u]:
                for edgekey in G.pred[u][v]:
                    if edgekey in edgekeys:
                        return (False, edgekey)
            else:
                return (True, None)
        edges = set(self.branchings[self.level].edge_index)
        while self.level > 0:
            self.level -= 1
            merged_node = self.template.format(self.level)
            circuit = self.circuits[self.level]
            (isroot, edgekey) = is_root(self.graphs[self.level + 1], merged_node, edges)
            edges.update(circuit)
            if isroot:
                minedge = self.minedge_circuit[self.level]
                if minedge is None:
                    raise Exception
                edges.remove(minedge)
            else:
                G = self.graphs[self.level]
                target = G.edge_index[edgekey][1]
                for edgekey in circuit:
                    (u, v, data) = G.edge_index[edgekey]
                    if v == target:
                        break
                else:
                    raise Exception("Couldn't find edge incoming to merged node.")
                edges.remove(edgekey)
        self.edges = edges
        H.add_nodes_from(self.G_original)
        for edgekey in edges:
            (u, v, d) = self.graphs[0].edge_index[edgekey]
            dd = {self.attr: self.trans(d[self.attr])}
            if preserve_attrs:
                for (key, value) in d.items():
                    if key not in [self.attr, self.candidate_attr]:
                        dd[key] = value
            H.add_edge(u, v, **dd)
        return H

@nx._dispatch(edge_attrs={'attr': 'default', 'partition': 0}, preserve_edge_attrs='preserve_attrs')
def maximum_branching(G, attr='weight', default=1, preserve_attrs=False, partition=None):
    if False:
        while True:
            i = 10

    def edmonds_add_edge(G, edge_index, u, v, key, **d):
        if False:
            i = 10
            return i + 15
        '\n        Adds an edge to `G` while also updating the edge index.\n\n        This algorithm requires the use of an external dictionary to track\n        the edge keys since it is possible that the source or destination\n        node of an edge will be changed and the default key-handling\n        capabilities of the MultiDiGraph class do not account for this.\n\n        Parameters\n        ----------\n        G : MultiDiGraph\n            The graph to insert an edge into.\n        edge_index : dict\n            A mapping from integers to the edges of the graph.\n        u : node\n            The source node of the new edge.\n        v : node\n            The destination node of the new edge.\n        key : int\n            The key to use from `edge_index`.\n        d : keyword arguments, optional\n            Other attributes to store on the new edge.\n        '
        if key in edge_index:
            (uu, vv, _) = edge_index[key]
            if u != uu or v != vv:
                raise Exception(f'Key {key!r} is already in use.')
        G.add_edge(u, v, key, **d)
        edge_index[key] = (u, v, G.succ[u][v][key])

    def edmonds_remove_node(G, edge_index, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove a node from the graph, updating the edge index to match.\n\n        Parameters\n        ----------\n        G : MultiDiGraph\n            The graph to remove an edge from.\n        edge_index : dict\n            A mapping from integers to the edges of the graph.\n        n : node\n            The node to remove from `G`.\n        '
        keys = set()
        for keydict in G.pred[n].values():
            keys.update(keydict)
        for keydict in G.succ[n].values():
            keys.update(keydict)
        for key in keys:
            del edge_index[key]
        G.remove_node(n)
    candidate_attr = "edmonds' secret candidate attribute"
    new_node_base_name = 'edmonds new node base name '
    G_original = G
    G = nx.MultiDiGraph()
    G_edge_index = {}
    for (key, (u, v, data)) in enumerate(G_original.edges(data=True)):
        d = {attr: data.get(attr, default)}
        if data.get(partition) is not None:
            d[partition] = data.get(partition)
        if preserve_attrs:
            for (d_k, d_v) in data.items():
                if d_k != attr:
                    d[d_k] = d_v
        edmonds_add_edge(G, G_edge_index, u, v, key, **d)
    level = 0
    B = nx.MultiDiGraph()
    B_edge_index = {}
    graphs = []
    branchings = []
    selected_nodes = set()
    uf = nx.utils.UnionFind()
    circuits = []
    minedge_circuit = []

    def edmonds_find_desired_edge(v):
        if False:
            i = 10
            return i + 15
        '\n        Find the edge directed towards v with maximal weight.\n\n        If an edge partition exists in this graph, return the included\n        edge if it exists and never return any excluded edge.\n\n        Note: There can only be one included edge for each vertex otherwise\n        the edge partition is empty.\n\n        Parameters\n        ----------\n        v : node\n            The node to search for the maximal weight incoming edge.\n        '
        edge = None
        max_weight = -INF
        for (u, _, key, data) in G.in_edges(v, data=True, keys=True):
            if data.get(partition) == nx.EdgePartition.EXCLUDED:
                continue
            new_weight = data[attr]
            if data.get(partition) == nx.EdgePartition.INCLUDED:
                max_weight = new_weight
                edge = (u, v, key, new_weight, data)
                break
            if new_weight > max_weight:
                max_weight = new_weight
                edge = (u, v, key, new_weight, data)
        return (edge, max_weight)

    def edmonds_step_I2(v, desired_edge, level):
        if False:
            print('Hello World!')
        "\n        Perform step I2 from Edmonds' paper\n\n        First, check if the last step I1 created a cycle. If it did not, do nothing.\n        If it did, store the cycle for later reference and contract it.\n\n        Parameters\n        ----------\n        v : node\n            The current node to consider\n        desired_edge : edge\n            The minimum desired edge to remove from the cycle.\n        level : int\n            The current level, i.e. the number of cycles that have already been removed.\n        "
        u = desired_edge[0]
        Q_nodes = nx.shortest_path(B, v, u)
        Q_edges = [list(B[Q_nodes[i]][vv].keys())[0] for (i, vv) in enumerate(Q_nodes[1:])]
        Q_edges.append(desired_edge[2])
        minweight = INF
        minedge = None
        Q_incoming_weight = {}
        for edge_key in Q_edges:
            (u, v, data) = B_edge_index[edge_key]
            w = data[attr]
            Q_incoming_weight[v] = w
            if data.get(partition) == nx.EdgePartition.INCLUDED:
                continue
            if w < minweight:
                minweight = w
                minedge = edge_key
        circuits.append(Q_edges)
        minedge_circuit.append(minedge)
        graphs.append((G.copy(), G_edge_index.copy()))
        branchings.append((B.copy(), B_edge_index.copy()))
        new_node = new_node_base_name + str(level)
        G.add_node(new_node)
        new_edges = []
        for (u, v, key, data) in G.edges(data=True, keys=True):
            if u in Q_incoming_weight:
                if v in Q_incoming_weight:
                    continue
                else:
                    dd = data.copy()
                    new_edges.append((new_node, v, key, dd))
            elif v in Q_incoming_weight:
                w = data[attr]
                w += minweight - Q_incoming_weight[v]
                dd = data.copy()
                dd[attr] = w
                new_edges.append((u, new_node, key, dd))
            else:
                continue
        for node in Q_nodes:
            edmonds_remove_node(G, G_edge_index, node)
            edmonds_remove_node(B, B_edge_index, node)
        selected_nodes.difference_update(set(Q_nodes))
        for (u, v, key, data) in new_edges:
            edmonds_add_edge(G, G_edge_index, u, v, key, **data)
            if candidate_attr in data:
                del data[candidate_attr]
                edmonds_add_edge(B, B_edge_index, u, v, key, **data)
                uf.union(u, v)

    def is_root(G, u, edgekeys):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns True if `u` is a root node in G.\n\n        Node `u` is a root node if its in-degree over the specified edges is zero.\n\n        Parameters\n        ----------\n        G : Graph\n            The current graph.\n        u : node\n            The node in `G` to check if it is a root.\n        edgekeys : iterable of edges\n            The edges for which to check if `u` is a root of.\n        '
        if u not in G:
            raise Exception(f'{u!r} not in G')
        for v in G.pred[u]:
            for edgekey in G.pred[u][v]:
                if edgekey in edgekeys:
                    return (False, edgekey)
        else:
            return (True, None)
    nodes = iter(list(G.nodes))
    while True:
        try:
            v = next(nodes)
        except StopIteration:
            assert len(G) == len(B)
            if len(B):
                assert is_branching(B)
            graphs.append((G.copy(), G_edge_index.copy()))
            branchings.append((B.copy(), B_edge_index.copy()))
            circuits.append([])
            minedge_circuit.append(None)
            break
        else:
            if v in selected_nodes:
                continue
        selected_nodes.add(v)
        B.add_node(v)
        (desired_edge, desired_edge_weight) = edmonds_find_desired_edge(v)
        if desired_edge is not None and desired_edge_weight > 0:
            u = desired_edge[0]
            circuit = uf[u] == uf[v]
            dd = {attr: desired_edge_weight}
            if desired_edge[4].get(partition) is not None:
                dd[partition] = desired_edge[4].get(partition)
            edmonds_add_edge(B, B_edge_index, u, v, desired_edge[2], **dd)
            G[u][v][desired_edge[2]][candidate_attr] = True
            uf.union(u, v)
            if circuit:
                edmonds_step_I2(v, desired_edge, level)
                nodes = iter(list(G.nodes()))
                level += 1
    H = G_original.__class__()
    edges = set(branchings[level][1])
    while level > 0:
        level -= 1
        merged_node = new_node_base_name + str(level)
        circuit = circuits[level]
        (isroot, edgekey) = is_root(graphs[level + 1][0], merged_node, edges)
        edges.update(circuit)
        if isroot:
            minedge = minedge_circuit[level]
            if minedge is None:
                raise Exception
            edges.remove(minedge)
        else:
            (G, G_edge_index) = graphs[level]
            target = G_edge_index[edgekey][1]
            for edgekey in circuit:
                (u, v, data) = G_edge_index[edgekey]
                if v == target:
                    break
            else:
                raise Exception("Couldn't find edge incoming to merged node.")
            edges.remove(edgekey)
    H.add_nodes_from(G_original)
    for edgekey in edges:
        (u, v, d) = graphs[0][1][edgekey]
        dd = {attr: d[attr]}
        if preserve_attrs:
            for (key, value) in d.items():
                if key not in [attr, candidate_attr]:
                    dd[key] = value
        H.add_edge(u, v, **dd)
    return H

@nx._dispatch(edge_attrs={'attr': 'default', 'partition': None}, preserve_edge_attrs='preserve_attrs')
def minimum_branching(G, attr='weight', default=1, preserve_attrs=False, partition=None):
    if False:
        return 10
    for (_, _, d) in G.edges(data=True):
        d[attr] = -d[attr]
    B = maximum_branching(G, attr, default, preserve_attrs, partition)
    for (_, _, d) in G.edges(data=True):
        d[attr] = -d[attr]
    for (_, _, d) in B.edges(data=True):
        d[attr] = -d[attr]
    return B

@nx._dispatch(edge_attrs={'attr': 'default', 'partition': None}, preserve_edge_attrs='preserve_attrs')
def minimal_branching(G, /, *, attr='weight', default=1, preserve_attrs=False, partition=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a minimal branching from `G`.\n\n    A minimal branching is a branching similar to a minimal arborescence but\n    without the requirement that the result is actually a spanning arborescence.\n    This allows minimal branchinges to be computed over graphs which may not\n    have arborescence (such as multiple components).\n\n    Parameters\n    ----------\n    G : (multi)digraph-like\n        The graph to be searched.\n    attr : str\n        The edge attribute used in determining optimality.\n    default : float\n        The value of the edge attribute used if an edge does not have\n        the attribute `attr`.\n    preserve_attrs : bool\n        If True, preserve the other attributes of the original graph (that are not\n        passed to `attr`)\n    partition : str\n        The key for the edge attribute containing the partition\n        data on the graph. Edges can be included, excluded or open using the\n        `EdgePartition` enum.\n\n    Returns\n    -------\n    B : (multi)digraph-like\n        A minimal branching.\n    '
    max_weight = -INF
    min_weight = INF
    for (_, _, w) in G.edges(data=attr):
        if w > max_weight:
            max_weight = w
        if w < min_weight:
            min_weight = w
    for (_, _, d) in G.edges(data=True):
        d[attr] = max_weight + 1 + (max_weight - min_weight) - d[attr]
    B = maximum_branching(G, attr, default, preserve_attrs, partition)
    for (_, _, d) in G.edges(data=True):
        d[attr] = max_weight + 1 + (max_weight - min_weight) - d[attr]
    for (_, _, d) in B.edges(data=True):
        d[attr] = max_weight + 1 + (max_weight - min_weight) - d[attr]
    return B

@nx._dispatch(edge_attrs={'attr': 'default', 'partition': None}, preserve_edge_attrs='preserve_attrs')
def maximum_spanning_arborescence(G, attr='weight', default=1, preserve_attrs=False, partition=None):
    if False:
        for i in range(10):
            print('nop')
    min_weight = INF
    max_weight = -INF
    for (_, _, w) in G.edges(data=attr):
        if w < min_weight:
            min_weight = w
        if w > max_weight:
            max_weight = w
    for (_, _, d) in G.edges(data=True):
        d[attr] = d[attr] - min_weight + 1 - (min_weight - max_weight)
    B = maximum_branching(G, attr, default, preserve_attrs, partition)
    for (_, _, d) in G.edges(data=True):
        d[attr] = d[attr] + min_weight - 1 + (min_weight - max_weight)
    for (_, _, d) in B.edges(data=True):
        d[attr] = d[attr] + min_weight - 1 + (min_weight - max_weight)
    if not is_arborescence(B):
        raise nx.exception.NetworkXException('No maximum spanning arborescence in G.')
    return B

@nx._dispatch(edge_attrs={'attr': 'default', 'partition': None}, preserve_edge_attrs='preserve_attrs')
def minimum_spanning_arborescence(G, attr='weight', default=1, preserve_attrs=False, partition=None):
    if False:
        for i in range(10):
            print('nop')
    B = minimal_branching(G, attr=attr, default=default, preserve_attrs=preserve_attrs, partition=partition)
    if not is_arborescence(B):
        raise nx.exception.NetworkXException('No minimum spanning arborescence in G.')
    return B
docstring_branching = '\nReturns a {kind} {style} from G.\n\nParameters\n----------\nG : (multi)digraph-like\n    The graph to be searched.\nattr : str\n    The edge attribute used to in determining optimality.\ndefault : float\n    The value of the edge attribute used if an edge does not have\n    the attribute `attr`.\npreserve_attrs : bool\n    If True, preserve the other attributes of the original graph (that are not\n    passed to `attr`)\npartition : str\n    The key for the edge attribute containing the partition\n    data on the graph. Edges can be included, excluded or open using the\n    `EdgePartition` enum.\n\nReturns\n-------\nB : (multi)digraph-like\n    A {kind} {style}.\n'
docstring_arborescence = docstring_branching + '\nRaises\n------\nNetworkXException\n    If the graph does not contain a {kind} {style}.\n\n'
maximum_branching.__doc__ = docstring_branching.format(kind='maximum', style='branching')
minimum_branching.__doc__ = docstring_branching.format(kind='minimum', style='branching') + '\nSee Also \n-------- \n    minimal_branching\n'
maximum_spanning_arborescence.__doc__ = docstring_arborescence.format(kind='maximum', style='spanning arborescence')
minimum_spanning_arborescence.__doc__ = docstring_arborescence.format(kind='minimum', style='spanning arborescence')

class ArborescenceIterator:
    """
    Iterate over all spanning arborescences of a graph in either increasing or
    decreasing cost.

    Notes
    -----
    This iterator uses the partition scheme from [1]_ (included edges,
    excluded edges and open edges). It generates minimum spanning
    arborescences using a modified Edmonds' Algorithm which respects the
    partition of edges. For arborescences with the same weight, ties are
    broken arbitrarily.

    References
    ----------
    .. [1] G.K. Janssens, K. Sörensen, An algorithm to generate all spanning
           trees in order of increasing cost, Pesquisa Operacional, 2005-08,
           Vol. 25 (2), p. 219-229,
           https://www.scielo.br/j/pope/a/XHswBwRwJyrfL88dmMwYNWp/?lang=en
    """

    @dataclass(order=True)
    class Partition:
        """
        This dataclass represents a partition and stores a dict with the edge
        data and the weight of the minimum spanning arborescence of the
        partition dict.
        """
        mst_weight: float
        partition_dict: dict = field(compare=False)

        def __copy__(self):
            if False:
                print('Hello World!')
            return ArborescenceIterator.Partition(self.mst_weight, self.partition_dict.copy())

    def __init__(self, G, weight='weight', minimum=True, init_partition=None):
        if False:
            return 10
        '\n        Initialize the iterator\n\n        Parameters\n        ----------\n        G : nx.DiGraph\n            The directed graph which we need to iterate trees over\n\n        weight : String, default = "weight"\n            The edge attribute used to store the weight of the edge\n\n        minimum : bool, default = True\n            Return the trees in increasing order while true and decreasing order\n            while false.\n\n        init_partition : tuple, default = None\n            In the case that certain edges have to be included or excluded from\n            the arborescences, `init_partition` should be in the form\n            `(included_edges, excluded_edges)` where each edges is a\n            `(u, v)`-tuple inside an iterable such as a list or set.\n\n        '
        self.G = G.copy()
        self.weight = weight
        self.minimum = minimum
        self.method = minimum_spanning_arborescence if minimum else maximum_spanning_arborescence
        self.partition_key = 'ArborescenceIterators super secret partition attribute name'
        if init_partition is not None:
            partition_dict = {}
            for e in init_partition[0]:
                partition_dict[e] = nx.EdgePartition.INCLUDED
            for e in init_partition[1]:
                partition_dict[e] = nx.EdgePartition.EXCLUDED
            self.init_partition = ArborescenceIterator.Partition(0, partition_dict)
        else:
            self.init_partition = None

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns\n        -------\n        ArborescenceIterator\n            The iterator object for this graph\n        '
        self.partition_queue = PriorityQueue()
        self._clear_partition(self.G)
        if self.init_partition is not None:
            self._write_partition(self.init_partition)
        mst_weight = self.method(self.G, self.weight, partition=self.partition_key, preserve_attrs=True).size(weight=self.weight)
        self.partition_queue.put(self.Partition(mst_weight if self.minimum else -mst_weight, {} if self.init_partition is None else self.init_partition.partition_dict))
        return self

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns\n        -------\n        (multi)Graph\n            The spanning tree of next greatest weight, which ties broken\n            arbitrarily.\n        '
        if self.partition_queue.empty():
            del self.G, self.partition_queue
            raise StopIteration
        partition = self.partition_queue.get()
        self._write_partition(partition)
        next_arborescence = self.method(self.G, self.weight, partition=self.partition_key, preserve_attrs=True)
        self._partition(partition, next_arborescence)
        self._clear_partition(next_arborescence)
        return next_arborescence

    def _partition(self, partition, partition_arborescence):
        if False:
            return 10
        '\n        Create new partitions based of the minimum spanning tree of the\n        current minimum partition.\n\n        Parameters\n        ----------\n        partition : Partition\n            The Partition instance used to generate the current minimum spanning\n            tree.\n        partition_arborescence : nx.Graph\n            The minimum spanning arborescence of the input partition.\n        '
        p1 = self.Partition(0, partition.partition_dict.copy())
        p2 = self.Partition(0, partition.partition_dict.copy())
        for e in partition_arborescence.edges:
            if e not in partition.partition_dict:
                p1.partition_dict[e] = nx.EdgePartition.EXCLUDED
                p2.partition_dict[e] = nx.EdgePartition.INCLUDED
                self._write_partition(p1)
                try:
                    p1_mst = self.method(self.G, self.weight, partition=self.partition_key, preserve_attrs=True)
                    p1_mst_weight = p1_mst.size(weight=self.weight)
                    p1.mst_weight = p1_mst_weight if self.minimum else -p1_mst_weight
                    self.partition_queue.put(p1.__copy__())
                except nx.NetworkXException:
                    pass
                p1.partition_dict = p2.partition_dict.copy()

    def _write_partition(self, partition):
        if False:
            i = 10
            return i + 15
        "\n        Writes the desired partition into the graph to calculate the minimum\n        spanning tree. Also, if one incoming edge is included, mark all others\n        as excluded so that if that vertex is merged during Edmonds' algorithm\n        we cannot still pick another of that vertex's included edges.\n\n        Parameters\n        ----------\n        partition : Partition\n            A Partition dataclass describing a partition on the edges of the\n            graph.\n        "
        for (u, v, d) in self.G.edges(data=True):
            if (u, v) in partition.partition_dict:
                d[self.partition_key] = partition.partition_dict[u, v]
            else:
                d[self.partition_key] = nx.EdgePartition.OPEN
        for n in self.G:
            included_count = 0
            excluded_count = 0
            for (u, v, d) in self.G.in_edges(nbunch=n, data=True):
                if d.get(self.partition_key) == nx.EdgePartition.INCLUDED:
                    included_count += 1
                elif d.get(self.partition_key) == nx.EdgePartition.EXCLUDED:
                    excluded_count += 1
            if included_count == 1 and excluded_count != self.G.in_degree(n) - 1:
                for (u, v, d) in self.G.in_edges(nbunch=n, data=True):
                    if d.get(self.partition_key) != nx.EdgePartition.INCLUDED:
                        d[self.partition_key] = nx.EdgePartition.EXCLUDED

    def _clear_partition(self, G):
        if False:
            print('Hello World!')
        '\n        Removes partition data from the graph\n        '
        for (u, v, d) in G.edges(data=True):
            if self.partition_key in d:
                del d[self.partition_key]