"""Generate graphs with a given joint degree and directed joint degree"""
import networkx as nx
from networkx.utils import py_random_state
__all__ = ['is_valid_joint_degree', 'is_valid_directed_joint_degree', 'joint_degree_graph', 'directed_joint_degree_graph']

@nx._dispatch(graphs=None)
def is_valid_joint_degree(joint_degrees):
    if False:
        while True:
            i = 10
    'Checks whether the given joint degree dictionary is realizable.\n\n    A *joint degree dictionary* is a dictionary of dictionaries, in\n    which entry ``joint_degrees[k][l]`` is an integer representing the\n    number of edges joining nodes of degree *k* with nodes of degree\n    *l*. Such a dictionary is realizable as a simple graph if and only\n    if the following conditions are satisfied.\n\n    - each entry must be an integer,\n    - the total number of nodes of degree *k*, computed by\n      ``sum(joint_degrees[k].values()) / k``, must be an integer,\n    - the total number of edges joining nodes of degree *k* with\n      nodes of degree *l* cannot exceed the total number of possible edges,\n    - each diagonal entry ``joint_degrees[k][k]`` must be even (this is\n      a convention assumed by the :func:`joint_degree_graph` function).\n\n\n    Parameters\n    ----------\n    joint_degrees :  dictionary of dictionary of integers\n        A joint degree dictionary in which entry ``joint_degrees[k][l]``\n        is the number of edges joining nodes of degree *k* with nodes of\n        degree *l*.\n\n    Returns\n    -------\n    bool\n        Whether the given joint degree dictionary is realizable as a\n        simple graph.\n\n    References\n    ----------\n    .. [1] M. Gjoka, M. Kurant, A. Markopoulou, "2.5K Graphs: from Sampling\n       to Generation", IEEE Infocom, 2013.\n    .. [2] I. Stanton, A. Pinar, "Constructing and sampling graphs with a\n       prescribed joint degree distribution", Journal of Experimental\n       Algorithmics, 2012.\n    '
    degree_count = {}
    for k in joint_degrees:
        if k > 0:
            k_size = sum(joint_degrees[k].values()) / k
            if not k_size.is_integer():
                return False
            degree_count[k] = k_size
    for k in joint_degrees:
        for l in joint_degrees[k]:
            if not float(joint_degrees[k][l]).is_integer():
                return False
            if k != l and joint_degrees[k][l] > degree_count[k] * degree_count[l]:
                return False
            elif k == l:
                if joint_degrees[k][k] > degree_count[k] * (degree_count[k] - 1):
                    return False
                if joint_degrees[k][k] % 2 != 0:
                    return False
    return True

def _neighbor_switch(G, w, unsat, h_node_residual, avoid_node_id=None):
    if False:
        print('Hello World!')
    'Releases one free stub for ``w``, while preserving joint degree in G.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Graph in which the neighbor switch will take place.\n    w : integer\n        Node id for which we will execute this neighbor switch.\n    unsat : set of integers\n        Set of unsaturated node ids that have the same degree as w.\n    h_node_residual: dictionary of integers\n        Keeps track of the remaining stubs  for a given node.\n    avoid_node_id: integer\n        Node id to avoid when selecting w_prime.\n\n    Notes\n    -----\n    First, it selects *w_prime*, an  unsaturated node that has the same degree\n    as ``w``. Second, it selects *switch_node*, a neighbor node of ``w`` that\n    is not  connected to *w_prime*. Then it executes an edge swap i.e. removes\n    (``w``,*switch_node*) and adds (*w_prime*,*switch_node*). Gjoka et. al. [1]\n    prove that such an edge swap is always possible.\n\n    References\n    ----------\n    .. [1] M. Gjoka, B. Tillman, A. Markopoulou, "Construction of Simple\n       Graphs with a Target Joint Degree Matrix and Beyond", IEEE Infocom, \'15\n    '
    if avoid_node_id is None or h_node_residual[avoid_node_id] > 1:
        w_prime = next(iter(unsat))
    else:
        iter_var = iter(unsat)
        while True:
            w_prime = next(iter_var)
            if w_prime != avoid_node_id:
                break
    w_prime_neighbs = G[w_prime]
    for v in G[w]:
        if v not in w_prime_neighbs and v != w_prime:
            switch_node = v
            break
    G.remove_edge(w, switch_node)
    G.add_edge(w_prime, switch_node)
    h_node_residual[w] += 1
    h_node_residual[w_prime] -= 1
    if h_node_residual[w_prime] == 0:
        unsat.remove(w_prime)

@py_random_state(1)
@nx._dispatch(graphs=None)
def joint_degree_graph(joint_degrees, seed=None):
    if False:
        for i in range(10):
            print('nop')
    'Generates a random simple graph with the given joint degree dictionary.\n\n    Parameters\n    ----------\n    joint_degrees :  dictionary of dictionary of integers\n        A joint degree dictionary in which entry ``joint_degrees[k][l]`` is the\n        number of edges joining nodes of degree *k* with nodes of degree *l*.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    G : Graph\n        A graph with the specified joint degree dictionary.\n\n    Raises\n    ------\n    NetworkXError\n        If *joint_degrees* dictionary is not realizable.\n\n    Notes\n    -----\n    In each iteration of the "while loop" the algorithm picks two disconnected\n    nodes *v* and *w*, of degree *k* and *l* correspondingly,  for which\n    ``joint_degrees[k][l]`` has not reached its target yet. It then adds\n    edge (*v*, *w*) and increases the number of edges in graph G by one.\n\n    The intelligence of the algorithm lies in the fact that  it is always\n    possible to add an edge between such disconnected nodes *v* and *w*,\n    even if one or both nodes do not have free stubs. That is made possible by\n    executing a "neighbor switch", an edge rewiring move that releases\n    a free stub while keeping the joint degree of G the same.\n\n    The algorithm continues for E (number of edges) iterations of\n    the "while loop", at the which point all entries of the given\n    ``joint_degrees[k][l]`` have reached their target values and the\n    construction is complete.\n\n    References\n    ----------\n    ..  [1] M. Gjoka, B. Tillman, A. Markopoulou, "Construction of Simple\n        Graphs with a Target Joint Degree Matrix and Beyond", IEEE Infocom, \'15\n\n    Examples\n    --------\n    >>> joint_degrees = {\n    ...     1: {4: 1},\n    ...     2: {2: 2, 3: 2, 4: 2},\n    ...     3: {2: 2, 4: 1},\n    ...     4: {1: 1, 2: 2, 3: 1},\n    ... }\n    >>> G = nx.joint_degree_graph(joint_degrees)\n    >>>\n    '
    if not is_valid_joint_degree(joint_degrees):
        msg = 'Input joint degree dict not realizable as a simple graph'
        raise nx.NetworkXError(msg)
    degree_count = {k: sum(l.values()) // k for (k, l) in joint_degrees.items() if k > 0}
    N = sum(degree_count.values())
    G = nx.empty_graph(N)
    h_degree_nodelist = {}
    h_node_residual = {}
    nodeid = 0
    for (degree, num_nodes) in degree_count.items():
        h_degree_nodelist[degree] = range(nodeid, nodeid + num_nodes)
        for v in h_degree_nodelist[degree]:
            h_node_residual[v] = degree
        nodeid += int(num_nodes)
    for k in joint_degrees:
        for l in joint_degrees[k]:
            n_edges_add = joint_degrees[k][l]
            if n_edges_add > 0 and k >= l:
                k_size = degree_count[k]
                l_size = degree_count[l]
                k_nodes = h_degree_nodelist[k]
                l_nodes = h_degree_nodelist[l]
                k_unsat = {v for v in k_nodes if h_node_residual[v] > 0}
                if k != l:
                    l_unsat = {w for w in l_nodes if h_node_residual[w] > 0}
                else:
                    l_unsat = k_unsat
                    n_edges_add = joint_degrees[k][l] // 2
                while n_edges_add > 0:
                    v = k_nodes[seed.randrange(k_size)]
                    w = l_nodes[seed.randrange(l_size)]
                    if not G.has_edge(v, w) and v != w:
                        if h_node_residual[v] == 0:
                            _neighbor_switch(G, v, k_unsat, h_node_residual)
                        if h_node_residual[w] == 0:
                            if k != l:
                                _neighbor_switch(G, w, l_unsat, h_node_residual)
                            else:
                                _neighbor_switch(G, w, l_unsat, h_node_residual, avoid_node_id=v)
                        G.add_edge(v, w)
                        h_node_residual[v] -= 1
                        h_node_residual[w] -= 1
                        n_edges_add -= 1
                        if h_node_residual[v] == 0:
                            k_unsat.discard(v)
                        if h_node_residual[w] == 0:
                            l_unsat.discard(w)
    return G

@nx._dispatch(graphs=None)
def is_valid_directed_joint_degree(in_degrees, out_degrees, nkk):
    if False:
        print('Hello World!')
    'Checks whether the given directed joint degree input is realizable\n\n    Parameters\n    ----------\n    in_degrees :  list of integers\n        in degree sequence contains the in degrees of nodes.\n    out_degrees : list of integers\n        out degree sequence contains the out degrees of nodes.\n    nkk  :  dictionary of dictionary of integers\n        directed joint degree dictionary. for nodes of out degree k (first\n        level of dict) and nodes of in degree l (second level of dict)\n        describes the number of edges.\n\n    Returns\n    -------\n    boolean\n        returns true if given input is realizable, else returns false.\n\n    Notes\n    -----\n    Here is the list of conditions that the inputs (in/out degree sequences,\n    nkk) need to satisfy for simple directed graph realizability:\n\n    - Condition 0: in_degrees and out_degrees have the same length\n    - Condition 1: nkk[k][l]  is integer for all k,l\n    - Condition 2: sum(nkk[k])/k = number of nodes with partition id k, is an\n                   integer and matching degree sequence\n    - Condition 3: number of edges and non-chords between k and l cannot exceed\n                   maximum possible number of edges\n\n\n    References\n    ----------\n    [1] B. Tillman, A. Markopoulou, C. T. Butts & M. Gjoka,\n        "Construction of Directed 2K Graphs". In Proc. of KDD 2017.\n    '
    V = {}
    forbidden = {}
    if len(in_degrees) != len(out_degrees):
        return False
    for idx in range(len(in_degrees)):
        i = in_degrees[idx]
        o = out_degrees[idx]
        V[i, 0] = V.get((i, 0), 0) + 1
        V[o, 1] = V.get((o, 1), 0) + 1
        forbidden[o, i] = forbidden.get((o, i), 0) + 1
    S = {}
    for k in nkk:
        for l in nkk[k]:
            val = nkk[k][l]
            if not float(val).is_integer():
                return False
            if val > 0:
                S[k, 1] = S.get((k, 1), 0) + val
                S[l, 0] = S.get((l, 0), 0) + val
                if val + forbidden.get((k, l), 0) > V[k, 1] * V[l, 0]:
                    return False
    return all((S[s] / s[0] == V[s] for s in S))

def _directed_neighbor_switch(G, w, unsat, h_node_residual_out, chords, h_partition_in, partition):
    if False:
        for i in range(10):
            print('nop')
    'Releases one free stub for node w, while preserving joint degree in G.\n\n    Parameters\n    ----------\n    G : networkx directed graph\n        graph within which the edge swap will take place.\n    w : integer\n        node id for which we need to perform a neighbor switch.\n    unsat: set of integers\n        set of node ids that have the same degree as w and are unsaturated.\n    h_node_residual_out: dict of integers\n        for a given node, keeps track of the remaining stubs to be added.\n    chords: set of tuples\n        keeps track of available positions to add edges.\n    h_partition_in: dict of integers\n        for a given node, keeps track of its partition id (in degree).\n    partition: integer\n        partition id to check if chords have to be updated.\n\n    Notes\n    -----\n    First, it selects node w_prime that (1) has the same degree as w and\n    (2) is unsaturated. Then, it selects node v, a neighbor of w, that is\n    not connected to w_prime and does an edge swap i.e. removes (w,v) and\n    adds (w_prime,v). If neighbor switch is not possible for w using\n    w_prime and v, then return w_prime; in [1] it\'s proven that\n    such unsaturated nodes can be used.\n\n    References\n    ----------\n    [1] B. Tillman, A. Markopoulou, C. T. Butts & M. Gjoka,\n        "Construction of Directed 2K Graphs". In Proc. of KDD 2017.\n    '
    w_prime = unsat.pop()
    unsat.add(w_prime)
    w_neighbs = list(G.successors(w))
    w_prime_neighbs = list(G.successors(w_prime))
    for v in w_neighbs:
        if v not in w_prime_neighbs and w_prime != v:
            G.remove_edge(w, v)
            G.add_edge(w_prime, v)
            if h_partition_in[v] == partition:
                chords.add((w, v))
                chords.discard((w_prime, v))
            h_node_residual_out[w] += 1
            h_node_residual_out[w_prime] -= 1
            if h_node_residual_out[w_prime] == 0:
                unsat.remove(w_prime)
            return None
    return w_prime

def _directed_neighbor_switch_rev(G, w, unsat, h_node_residual_in, chords, h_partition_out, partition):
    if False:
        print('Hello World!')
    'The reverse of directed_neighbor_switch.\n\n    Parameters\n    ----------\n    G : networkx directed graph\n        graph within which the edge swap will take place.\n    w : integer\n        node id for which we need to perform a neighbor switch.\n    unsat: set of integers\n        set of node ids that have the same degree as w and are unsaturated.\n    h_node_residual_in: dict of integers\n        for a given node, keeps track of the remaining stubs to be added.\n    chords: set of tuples\n        keeps track of available positions to add edges.\n    h_partition_out: dict of integers\n        for a given node, keeps track of its partition id (out degree).\n    partition: integer\n        partition id to check if chords have to be updated.\n\n    Notes\n    -----\n    Same operation as directed_neighbor_switch except it handles this operation\n    for incoming edges instead of outgoing.\n    '
    w_prime = unsat.pop()
    unsat.add(w_prime)
    w_neighbs = list(G.predecessors(w))
    w_prime_neighbs = list(G.predecessors(w_prime))
    for v in w_neighbs:
        if v not in w_prime_neighbs and w_prime != v:
            G.remove_edge(v, w)
            G.add_edge(v, w_prime)
            if h_partition_out[v] == partition:
                chords.add((v, w))
                chords.discard((v, w_prime))
            h_node_residual_in[w] += 1
            h_node_residual_in[w_prime] -= 1
            if h_node_residual_in[w_prime] == 0:
                unsat.remove(w_prime)
            return None
    return w_prime

@py_random_state(3)
@nx._dispatch(graphs=None)
def directed_joint_degree_graph(in_degrees, out_degrees, nkk, seed=None):
    if False:
        i = 10
        return i + 15
    'Generates a random simple directed graph with the joint degree.\n\n    Parameters\n    ----------\n    degree_seq :  list of tuples (of size 3)\n        degree sequence contains tuples of nodes with node id, in degree and\n        out degree.\n    nkk  :  dictionary of dictionary of integers\n        directed joint degree dictionary, for nodes of out degree k (first\n        level of dict) and nodes of in degree l (second level of dict)\n        describes the number of edges.\n    seed : hashable object, optional\n        Seed for random number generator.\n\n    Returns\n    -------\n    G : Graph\n        A directed graph with the specified inputs.\n\n    Raises\n    ------\n    NetworkXError\n        If degree_seq and nkk are not realizable as a simple directed graph.\n\n\n    Notes\n    -----\n    Similarly to the undirected version:\n    In each iteration of the "while loop" the algorithm picks two disconnected\n    nodes v and w, of degree k and l correspondingly,  for which nkk[k][l] has\n    not reached its target yet i.e. (for given k,l): n_edges_add < nkk[k][l].\n    It then adds edge (v,w) and always increases the number of edges in graph G\n    by one.\n\n    The intelligence of the algorithm lies in the fact that  it is always\n    possible to add an edge between disconnected nodes v and w, for which\n    nkk[degree(v)][degree(w)] has not reached its target, even if one or both\n    nodes do not have free stubs. If either node v or w does not have a free\n    stub, we perform a "neighbor switch", an edge rewiring move that releases a\n    free stub while keeping nkk the same.\n\n    The difference for the directed version lies in the fact that neighbor\n    switches might not be able to rewire, but in these cases unsaturated nodes\n    can be reassigned to use instead, see [1] for detailed description and\n    proofs.\n\n    The algorithm continues for E (number of edges in the graph) iterations of\n    the "while loop", at which point all entries of the given nkk[k][l] have\n    reached their target values and the construction is complete.\n\n    References\n    ----------\n    [1] B. Tillman, A. Markopoulou, C. T. Butts & M. Gjoka,\n        "Construction of Directed 2K Graphs". In Proc. of KDD 2017.\n\n    Examples\n    --------\n    >>> in_degrees = [0, 1, 1, 2]\n    >>> out_degrees = [1, 1, 1, 1]\n    >>> nkk = {1: {1: 2, 2: 2}}\n    >>> G = nx.directed_joint_degree_graph(in_degrees, out_degrees, nkk)\n    >>>\n    '
    if not is_valid_directed_joint_degree(in_degrees, out_degrees, nkk):
        msg = 'Input is not realizable as a simple graph'
        raise nx.NetworkXError(msg)
    G = nx.DiGraph()
    h_degree_nodelist_in = {}
    h_degree_nodelist_out = {}
    h_degree_nodelist_in_unsat = {}
    h_degree_nodelist_out_unsat = {}
    h_node_residual_out = {}
    h_node_residual_in = {}
    h_partition_out = {}
    h_partition_in = {}
    non_chords = {}
    for (idx, i) in enumerate(in_degrees):
        idx = int(idx)
        if i > 0:
            h_degree_nodelist_in.setdefault(i, [])
            h_degree_nodelist_in_unsat.setdefault(i, set())
            h_degree_nodelist_in[i].append(idx)
            h_degree_nodelist_in_unsat[i].add(idx)
            h_node_residual_in[idx] = i
            h_partition_in[idx] = i
    for (idx, o) in enumerate(out_degrees):
        o = out_degrees[idx]
        non_chords[o, in_degrees[idx]] = non_chords.get((o, in_degrees[idx]), 0) + 1
        idx = int(idx)
        if o > 0:
            h_degree_nodelist_out.setdefault(o, [])
            h_degree_nodelist_out_unsat.setdefault(o, set())
            h_degree_nodelist_out[o].append(idx)
            h_degree_nodelist_out_unsat[o].add(idx)
            h_node_residual_out[idx] = o
            h_partition_out[idx] = o
        G.add_node(idx)
    nk_in = {}
    nk_out = {}
    for p in h_degree_nodelist_in:
        nk_in[p] = len(h_degree_nodelist_in[p])
    for p in h_degree_nodelist_out:
        nk_out[p] = len(h_degree_nodelist_out[p])
    for k in nkk:
        for l in nkk[k]:
            n_edges_add = nkk[k][l]
            if n_edges_add > 0:
                chords = set()
                k_len = nk_out[k]
                l_len = nk_in[l]
                chords_sample = seed.sample(range(k_len * l_len), n_edges_add + non_chords.get((k, l), 0))
                num = 0
                while len(chords) < n_edges_add:
                    i = h_degree_nodelist_out[k][chords_sample[num] % k_len]
                    j = h_degree_nodelist_in[l][chords_sample[num] // k_len]
                    num += 1
                    if i != j:
                        chords.add((i, j))
                k_unsat = h_degree_nodelist_out_unsat[k]
                l_unsat = h_degree_nodelist_in_unsat[l]
                while n_edges_add > 0:
                    (v, w) = chords.pop()
                    chords.add((v, w))
                    if h_node_residual_out[v] == 0:
                        _v = _directed_neighbor_switch(G, v, k_unsat, h_node_residual_out, chords, h_partition_in, l)
                        if _v is not None:
                            v = _v
                    if h_node_residual_in[w] == 0:
                        _w = _directed_neighbor_switch_rev(G, w, l_unsat, h_node_residual_in, chords, h_partition_out, k)
                        if _w is not None:
                            w = _w
                    G.add_edge(v, w)
                    h_node_residual_out[v] -= 1
                    h_node_residual_in[w] -= 1
                    n_edges_add -= 1
                    chords.discard((v, w))
                    if h_node_residual_out[v] == 0:
                        k_unsat.discard(v)
                    if h_node_residual_in[w] == 0:
                        l_unsat.discard(w)
    return G