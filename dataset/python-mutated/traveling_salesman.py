"""
=================================
Travelling Salesman Problem (TSP)
=================================

Implementation of approximate algorithms
for solving and approximating the TSP problem.

Categories of algorithms which are implemented:

- Christofides (provides a 3/2-approximation of TSP)
- Greedy
- Simulated Annealing (SA)
- Threshold Accepting (TA)
- Asadpour Asymmetric Traveling Salesman Algorithm

The Travelling Salesman Problem tries to find, given the weight
(distance) between all points where a salesman has to visit, the
route so that:

- The total distance (cost) which the salesman travels is minimized.
- The salesman returns to the starting point.
- Note that for a complete graph, the salesman visits each point once.

The function `travelling_salesman_problem` allows for incomplete
graphs by finding all-pairs shortest paths, effectively converting
the problem to a complete graph problem. It calls one of the
approximate methods on that problem and then converts the result
back to the original graph using the previously found shortest paths.

TSP is an NP-hard problem in combinatorial optimization,
important in operations research and theoretical computer science.

http://en.wikipedia.org/wiki/Travelling_salesman_problem
"""
import math
import networkx as nx
from networkx.algorithms.tree.mst import random_spanning_tree
from networkx.utils import not_implemented_for, pairwise, py_random_state
__all__ = ['traveling_salesman_problem', 'christofides', 'asadpour_atsp', 'greedy_tsp', 'simulated_annealing_tsp', 'threshold_accepting_tsp']

def swap_two_nodes(soln, seed):
    if False:
        for i in range(10):
            print('nop')
    "Swap two nodes in `soln` to give a neighbor solution.\n\n    Parameters\n    ----------\n    soln : list of nodes\n        Current cycle of nodes\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    list\n        The solution after move is applied. (A neighbor solution.)\n\n    Notes\n    -----\n        This function assumes that the incoming list `soln` is a cycle\n        (that the first and last element are the same) and also that\n        we don't want any move to change the first node in the list\n        (and thus not the last node either).\n\n        The input list is changed as well as returned. Make a copy if needed.\n\n    See Also\n    --------\n        move_one_node\n    "
    (a, b) = seed.sample(range(1, len(soln) - 1), k=2)
    (soln[a], soln[b]) = (soln[b], soln[a])
    return soln

def move_one_node(soln, seed):
    if False:
        i = 10
        return i + 15
    "Move one node to another position to give a neighbor solution.\n\n    The node to move and the position to move to are chosen randomly.\n    The first and last nodes are left untouched as soln must be a cycle\n    starting at that node.\n\n    Parameters\n    ----------\n    soln : list of nodes\n        Current cycle of nodes\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    list\n        The solution after move is applied. (A neighbor solution.)\n\n    Notes\n    -----\n        This function assumes that the incoming list `soln` is a cycle\n        (that the first and last element are the same) and also that\n        we don't want any move to change the first node in the list\n        (and thus not the last node either).\n\n        The input list is changed as well as returned. Make a copy if needed.\n\n    See Also\n    --------\n        swap_two_nodes\n    "
    (a, b) = seed.sample(range(1, len(soln) - 1), k=2)
    soln.insert(b, soln.pop(a))
    return soln

@not_implemented_for('directed')
@nx._dispatch(edge_attrs='weight')
def christofides(G, weight='weight', tree=None):
    if False:
        for i in range(10):
            print('nop')
    'Approximate a solution of the traveling salesman problem\n\n    Compute a 3/2-approximation of the traveling salesman problem\n    in a complete undirected graph using Christofides [1]_ algorithm.\n\n    Parameters\n    ----------\n    G : Graph\n        `G` should be a complete weighted undirected graph.\n        The distance between all pairs of nodes should be included.\n\n    weight : string, optional (default="weight")\n        Edge data key corresponding to the edge weight.\n        If any edge does not have this attribute the weight is set to 1.\n\n    tree : NetworkX graph or None (default: None)\n        A minimum spanning tree of G. Or, if None, the minimum spanning\n        tree is computed using :func:`networkx.minimum_spanning_tree`\n\n    Returns\n    -------\n    list\n        List of nodes in `G` along a cycle with a 3/2-approximation of\n        the minimal Hamiltonian cycle.\n\n    References\n    ----------\n    .. [1] Christofides, Nicos. "Worst-case analysis of a new heuristic for\n       the travelling salesman problem." No. RR-388. Carnegie-Mellon Univ\n       Pittsburgh Pa Management Sciences Research Group, 1976.\n    '
    loop_nodes = nx.nodes_with_selfloops(G)
    try:
        node = next(loop_nodes)
    except StopIteration:
        pass
    else:
        G = G.copy()
        G.remove_edge(node, node)
        G.remove_edges_from(((n, n) for n in loop_nodes))
    N = len(G) - 1
    if any((len(nbrdict) != N for (n, nbrdict) in G.adj.items())):
        raise nx.NetworkXError('G must be a complete graph.')
    if tree is None:
        tree = nx.minimum_spanning_tree(G, weight=weight)
    L = G.copy()
    L.remove_nodes_from([v for (v, degree) in tree.degree if not degree % 2])
    MG = nx.MultiGraph()
    MG.add_edges_from(tree.edges)
    edges = nx.min_weight_matching(L, weight=weight)
    MG.add_edges_from(edges)
    return _shortcutting(nx.eulerian_circuit(MG))

def _shortcutting(circuit):
    if False:
        i = 10
        return i + 15
    'Remove duplicate nodes in the path'
    nodes = []
    for (u, v) in circuit:
        if v in nodes:
            continue
        if not nodes:
            nodes.append(u)
        nodes.append(v)
    nodes.append(nodes[0])
    return nodes

@nx._dispatch(edge_attrs='weight')
def traveling_salesman_problem(G, weight='weight', nodes=None, cycle=True, method=None):
    if False:
        i = 10
        return i + 15
    'Find the shortest path in `G` connecting specified nodes\n\n    This function allows approximate solution to the traveling salesman\n    problem on networks that are not complete graphs and/or where the\n    salesman does not need to visit all nodes.\n\n    This function proceeds in two steps. First, it creates a complete\n    graph using the all-pairs shortest_paths between nodes in `nodes`.\n    Edge weights in the new graph are the lengths of the paths\n    between each pair of nodes in the original graph.\n    Second, an algorithm (default: `christofides` for undirected and\n    `asadpour_atsp` for directed) is used to approximate the minimal Hamiltonian\n    cycle on this new graph. The available algorithms are:\n\n     - christofides\n     - greedy_tsp\n     - simulated_annealing_tsp\n     - threshold_accepting_tsp\n     - asadpour_atsp\n\n    Once the Hamiltonian Cycle is found, this function post-processes to\n    accommodate the structure of the original graph. If `cycle` is ``False``,\n    the biggest weight edge is removed to make a Hamiltonian path.\n    Then each edge on the new complete graph used for that analysis is\n    replaced by the shortest_path between those nodes on the original graph.\n    If the input graph `G` includes edges with weights that do not adhere to\n    the triangle inequality, such as when `G` is not a complete graph (i.e\n    length of non-existent edges is infinity), then the returned path may\n    contain some repeating nodes (other than the starting node).\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        A possibly weighted graph\n\n    nodes : collection of nodes (default=G.nodes)\n        collection (list, set, etc.) of nodes to visit\n\n    weight : string, optional (default="weight")\n        Edge data key corresponding to the edge weight.\n        If any edge does not have this attribute the weight is set to 1.\n\n    cycle : bool (default: True)\n        Indicates whether a cycle should be returned, or a path.\n        Note: the cycle is the approximate minimal cycle.\n        The path simply removes the biggest edge in that cycle.\n\n    method : function (default: None)\n        A function that returns a cycle on all nodes and approximates\n        the solution to the traveling salesman problem on a complete\n        graph. The returned cycle is then used to find a corresponding\n        solution on `G`. `method` should be callable; take inputs\n        `G`, and `weight`; and return a list of nodes along the cycle.\n\n        Provided options include :func:`christofides`, :func:`greedy_tsp`,\n        :func:`simulated_annealing_tsp` and :func:`threshold_accepting_tsp`.\n\n        If `method is None`: use :func:`christofides` for undirected `G` and\n        :func:`threshold_accepting_tsp` for directed `G`.\n\n        To specify parameters for these provided functions, construct lambda\n        functions that state the specific value. `method` must have 2 inputs.\n        (See examples).\n\n    Returns\n    -------\n    list\n        List of nodes in `G` along a path with an approximation of the minimal\n        path through `nodes`.\n\n    Raises\n    ------\n    NetworkXError\n        If `G` is a directed graph it has to be strongly connected or the\n        complete version cannot be generated.\n\n    Examples\n    --------\n    >>> tsp = nx.approximation.traveling_salesman_problem\n    >>> G = nx.cycle_graph(9)\n    >>> G[4][5]["weight"] = 5  # all other weights are 1\n    >>> tsp(G, nodes=[3, 6])\n    [3, 2, 1, 0, 8, 7, 6, 7, 8, 0, 1, 2, 3]\n    >>> path = tsp(G, cycle=False)\n    >>> path in ([4, 3, 2, 1, 0, 8, 7, 6, 5], [5, 6, 7, 8, 0, 1, 2, 3, 4])\n    True\n\n    Build (curry) your own function to provide parameter values to the methods.\n\n    >>> SA_tsp = nx.approximation.simulated_annealing_tsp\n    >>> method = lambda G, wt: SA_tsp(G, "greedy", weight=wt, temp=500)\n    >>> path = tsp(G, cycle=False, method=method)\n    >>> path in ([4, 3, 2, 1, 0, 8, 7, 6, 5], [5, 6, 7, 8, 0, 1, 2, 3, 4])\n    True\n\n    '
    if method is None:
        if G.is_directed():
            method = asadpour_atsp
        else:
            method = christofides
    if nodes is None:
        nodes = list(G.nodes)
    dist = {}
    path = {}
    for (n, (d, p)) in nx.all_pairs_dijkstra(G, weight=weight):
        dist[n] = d
        path[n] = p
    if G.is_directed():
        if not nx.is_strongly_connected(G):
            raise nx.NetworkXError('G is not strongly connected')
        GG = nx.DiGraph()
    else:
        GG = nx.Graph()
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            GG.add_edge(u, v, weight=dist[u][v])
    best_GG = method(GG, weight)
    if not cycle:
        (u, v) = max(pairwise(best_GG), key=lambda x: dist[x[0]][x[1]])
        pos = best_GG.index(u) + 1
        while best_GG[pos] != v:
            pos = best_GG[pos:].index(u) + 1
        best_GG = best_GG[pos:-1] + best_GG[:pos]
    best_path = []
    for (u, v) in pairwise(best_GG):
        best_path.extend(path[u][v][:-1])
    best_path.append(v)
    return best_path

@not_implemented_for('undirected')
@py_random_state(2)
@nx._dispatch(edge_attrs='weight')
def asadpour_atsp(G, weight='weight', seed=None, source=None):
    if False:
        i = 10
        return i + 15
    '\n    Returns an approximate solution to the traveling salesman problem.\n\n    This approximate solution is one of the best known approximations for the\n    asymmetric traveling salesman problem developed by Asadpour et al,\n    [1]_. The algorithm first solves the Held-Karp relaxation to find a lower\n    bound for the weight of the cycle. Next, it constructs an exponential\n    distribution of undirected spanning trees where the probability of an\n    edge being in the tree corresponds to the weight of that edge using a\n    maximum entropy rounding scheme. Next we sample that distribution\n    $2 \\lceil \\ln n \\rceil$ times and save the minimum sampled tree once the\n    direction of the arcs is added back to the edges. Finally, we augment\n    then short circuit that graph to find the approximate tour for the\n    salesman.\n\n    Parameters\n    ----------\n    G : nx.DiGraph\n        The graph should be a complete weighted directed graph. The\n        distance between all paris of nodes should be included and the triangle\n        inequality should hold. That is, the direct edge between any two nodes\n        should be the path of least cost.\n\n    weight : string, optional (default="weight")\n        Edge data key corresponding to the edge weight.\n        If any edge does not have this attribute the weight is set to 1.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    source : node label (default=`None`)\n        If given, return the cycle starting and ending at the given node.\n\n    Returns\n    -------\n    cycle : list of nodes\n        Returns the cycle (list of nodes) that a salesman can follow to minimize\n        the total weight of the trip.\n\n    Raises\n    ------\n    NetworkXError\n        If `G` is not complete or has less than two nodes, the algorithm raises\n        an exception.\n\n    NetworkXError\n        If `source` is not `None` and is not a node in `G`, the algorithm raises\n        an exception.\n\n    NetworkXNotImplemented\n        If `G` is an undirected graph.\n\n    References\n    ----------\n    .. [1] A. Asadpour, M. X. Goemans, A. Madry, S. O. Gharan, and A. Saberi,\n       An o(log n/log log n)-approximation algorithm for the asymmetric\n       traveling salesman problem, Operations research, 65 (2017),\n       pp. 1043–1061\n\n    Examples\n    --------\n    >>> import networkx as nx\n    >>> import networkx.algorithms.approximation as approx\n    >>> G = nx.complete_graph(3, create_using=nx.DiGraph)\n    >>> nx.set_edge_attributes(G, {(0, 1): 2, (1, 2): 2, (2, 0): 2, (0, 2): 1, (2, 1): 1, (1, 0): 1}, "weight")\n    >>> tour = approx.asadpour_atsp(G,source=0)\n    >>> tour\n    [0, 2, 1, 0]\n    '
    from math import ceil, exp
    from math import log as ln
    N = len(G) - 1
    if N < 2:
        raise nx.NetworkXError('G must have at least two nodes')
    if any((len(nbrdict) - (n in nbrdict) != N for (n, nbrdict) in G.adj.items())):
        raise nx.NetworkXError('G is not a complete DiGraph')
    if source is not None and source not in G.nodes:
        raise nx.NetworkXError('Given source node not in G.')
    (opt_hk, z_star) = held_karp_ascent(G, weight)
    if not isinstance(z_star, dict):
        return _shortcutting(nx.eulerian_circuit(z_star, source=source))
    z_support = nx.MultiGraph()
    for (u, v) in z_star:
        if (u, v) not in z_support.edges:
            edge_weight = min(G[u][v][weight], G[v][u][weight])
            z_support.add_edge(u, v, **{weight: edge_weight})
    gamma = spanning_tree_distribution(z_support, z_star)
    z_support = nx.Graph(z_support)
    lambda_dict = {(u, v): exp(gamma[u, v]) for (u, v) in z_support.edges()}
    nx.set_edge_attributes(z_support, lambda_dict, 'weight')
    del gamma, lambda_dict
    minimum_sampled_tree = None
    minimum_sampled_tree_weight = math.inf
    for _ in range(2 * ceil(ln(G.number_of_nodes()))):
        sampled_tree = random_spanning_tree(z_support, 'weight', seed=seed)
        sampled_tree_weight = sampled_tree.size(weight)
        if sampled_tree_weight < minimum_sampled_tree_weight:
            minimum_sampled_tree = sampled_tree.copy()
            minimum_sampled_tree_weight = sampled_tree_weight
    t_star = nx.MultiDiGraph()
    for (u, v, d) in minimum_sampled_tree.edges(data=weight):
        if d == G[u][v][weight]:
            t_star.add_edge(u, v, **{weight: d})
        else:
            t_star.add_edge(v, u, **{weight: d})
    node_demands = {n: t_star.out_degree(n) - t_star.in_degree(n) for n in t_star}
    nx.set_node_attributes(G, node_demands, 'demand')
    flow_dict = nx.min_cost_flow(G, 'demand')
    for (source, values) in flow_dict.items():
        for target in values:
            if (source, target) not in t_star.edges and values[target] > 0:
                for _ in range(values[target]):
                    t_star.add_edge(source, target)
    circuit = nx.eulerian_circuit(t_star, source=source)
    return _shortcutting(circuit)

@nx._dispatch(edge_attrs='weight')
def held_karp_ascent(G, weight='weight'):
    if False:
        print('Hello World!')
    '\n    Minimizes the Held-Karp relaxation of the TSP for `G`\n\n    Solves the Held-Karp relaxation of the input complete digraph and scales\n    the output solution for use in the Asadpour [1]_ ASTP algorithm.\n\n    The Held-Karp relaxation defines the lower bound for solutions to the\n    ATSP, although it does return a fractional solution. This is used in the\n    Asadpour algorithm as an initial solution which is later rounded to a\n    integral tree within the spanning tree polytopes. This function solves\n    the relaxation with the branch and bound method in [2]_.\n\n    Parameters\n    ----------\n    G : nx.DiGraph\n        The graph should be a complete weighted directed graph.\n        The distance between all paris of nodes should be included.\n\n    weight : string, optional (default="weight")\n        Edge data key corresponding to the edge weight.\n        If any edge does not have this attribute the weight is set to 1.\n\n    Returns\n    -------\n    OPT : float\n        The cost for the optimal solution to the Held-Karp relaxation\n    z : dict or nx.Graph\n        A symmetrized and scaled version of the optimal solution to the\n        Held-Karp relaxation for use in the Asadpour algorithm.\n\n        If an integral solution is found, then that is an optimal solution for\n        the ATSP problem and that is returned instead.\n\n    References\n    ----------\n    .. [1] A. Asadpour, M. X. Goemans, A. Madry, S. O. Gharan, and A. Saberi,\n       An o(log n/log log n)-approximation algorithm for the asymmetric\n       traveling salesman problem, Operations research, 65 (2017),\n       pp. 1043–1061\n\n    .. [2] M. Held, R. M. Karp, The traveling-salesman problem and minimum\n           spanning trees, Operations Research, 1970-11-01, Vol. 18 (6),\n           pp.1138-1162\n    '
    import numpy as np
    from scipy import optimize

    def k_pi():
        if False:
            print('Hello World!')
        '\n        Find the set of minimum 1-Arborescences for G at point pi.\n\n        Returns\n        -------\n        Set\n            The set of minimum 1-Arborescences\n        '
        G_1 = G.copy()
        minimum_1_arborescences = set()
        minimum_1_arborescence_weight = math.inf
        n = next(G.__iter__())
        G_1.remove_node(n)
        min_root = {'node': None, weight: math.inf}
        max_root = {'node': None, weight: -math.inf}
        for (u, v, d) in G.edges(n, data=True):
            if d[weight] < min_root[weight]:
                min_root = {'node': v, weight: d[weight]}
            if d[weight] > max_root[weight]:
                max_root = {'node': v, weight: d[weight]}
        min_in_edge = min(G.in_edges(n, data=True), key=lambda x: x[2][weight])
        min_root[weight] = min_root[weight] + min_in_edge[2][weight]
        max_root[weight] = max_root[weight] + min_in_edge[2][weight]
        min_arb_weight = math.inf
        for arb in nx.ArborescenceIterator(G_1):
            arb_weight = arb.size(weight)
            if min_arb_weight == math.inf:
                min_arb_weight = arb_weight
            elif arb_weight > min_arb_weight + max_root[weight] - min_root[weight]:
                break
            for (N, deg) in arb.in_degree:
                if deg == 0:
                    arb.add_edge(n, N, **{weight: G[n][N][weight]})
                    arb_weight += G[n][N][weight]
                    break
            edge_data = G[N][n]
            G.remove_edge(N, n)
            min_weight = min(G.in_edges(n, data=weight), key=lambda x: x[2])[2]
            min_edges = [(u, v, d) for (u, v, d) in G.in_edges(n, data=weight) if d == min_weight]
            for (u, v, d) in min_edges:
                new_arb = arb.copy()
                new_arb.add_edge(u, v, **{weight: d})
                new_arb_weight = arb_weight + d
                if new_arb_weight < minimum_1_arborescence_weight:
                    minimum_1_arborescences.clear()
                    minimum_1_arborescence_weight = new_arb_weight
                if new_arb_weight == minimum_1_arborescence_weight:
                    minimum_1_arborescences.add(new_arb)
            G.add_edge(N, n, **edge_data)
        return minimum_1_arborescences

    def direction_of_ascent():
        if False:
            while True:
                i = 10
        '\n        Find the direction of ascent at point pi.\n\n        See [1]_ for more information.\n\n        Returns\n        -------\n        dict\n            A mapping from the nodes of the graph which represents the direction\n            of ascent.\n\n        References\n        ----------\n        .. [1] M. Held, R. M. Karp, The traveling-salesman problem and minimum\n           spanning trees, Operations Research, 1970-11-01, Vol. 18 (6),\n           pp.1138-1162\n        '
        d = {}
        for n in G:
            d[n] = 0
        del n
        minimum_1_arborescences = k_pi()
        while True:
            min_k_d_weight = math.inf
            min_k_d = None
            for arborescence in minimum_1_arborescences:
                weighted_cost = 0
                for (n, deg) in arborescence.degree:
                    weighted_cost += d[n] * (deg - 2)
                if weighted_cost < min_k_d_weight:
                    min_k_d_weight = weighted_cost
                    min_k_d = arborescence
            if min_k_d_weight > 0:
                return (d, min_k_d)
            for (n, deg) in min_k_d.degree:
                d[n] += deg - 2
            c = np.full(len(minimum_1_arborescences), -1, dtype=int)
            a_eq = np.empty((len(G) + 1, len(minimum_1_arborescences)), dtype=int)
            b_eq = np.zeros(len(G) + 1, dtype=int)
            b_eq[len(G)] = 1
            for (arb_count, arborescence) in enumerate(minimum_1_arborescences):
                n_count = len(G) - 1
                for (n, deg) in arborescence.degree:
                    a_eq[n_count][arb_count] = deg - 2
                    n_count -= 1
                a_eq[len(G)][arb_count] = 1
            program_result = optimize.linprog(c, A_eq=a_eq, b_eq=b_eq)
            if program_result.success:
                return (None, minimum_1_arborescences)

    def find_epsilon(k, d):
        if False:
            while True:
                i = 10
        '\n        Given the direction of ascent at pi, find the maximum distance we can go\n        in that direction.\n\n        Parameters\n        ----------\n        k_xy : set\n            The set of 1-arborescences which have the minimum rate of increase\n            in the direction of ascent\n\n        d : dict\n            The direction of ascent\n\n        Returns\n        -------\n        float\n            The distance we can travel in direction `d`\n        '
        min_epsilon = math.inf
        for (e_u, e_v, e_w) in G.edges(data=weight):
            if (e_u, e_v) in k.edges:
                continue
            if len(k.in_edges(e_v, data=weight)) > 1:
                raise Exception
            (sub_u, sub_v, sub_w) = next(k.in_edges(e_v, data=weight).__iter__())
            k.add_edge(e_u, e_v, **{weight: e_w})
            k.remove_edge(sub_u, sub_v)
            if max((d for (n, d) in k.in_degree())) <= 1 and len(G) == k.number_of_edges() and nx.is_weakly_connected(k):
                if d[sub_u] == d[e_u] or sub_w == e_w:
                    k.remove_edge(e_u, e_v)
                    k.add_edge(sub_u, sub_v, **{weight: sub_w})
                    continue
                epsilon = (sub_w - e_w) / (d[e_u] - d[sub_u])
                if 0 < epsilon < min_epsilon:
                    min_epsilon = epsilon
            k.remove_edge(e_u, e_v)
            k.add_edge(sub_u, sub_v, **{weight: sub_w})
        return min_epsilon
    pi_dict = {}
    for n in G:
        pi_dict[n] = 0
    del n
    original_edge_weights = {}
    for (u, v, d) in G.edges(data=True):
        original_edge_weights[u, v] = d[weight]
    (dir_ascent, k_d) = direction_of_ascent()
    while dir_ascent is not None:
        max_distance = find_epsilon(k_d, dir_ascent)
        for (n, v) in dir_ascent.items():
            pi_dict[n] += max_distance * v
        for (u, v, d) in G.edges(data=True):
            d[weight] = original_edge_weights[u, v] + pi_dict[u]
        (dir_ascent, k_d) = direction_of_ascent()
    k_max = k_d
    for k in k_max:
        if len([n for n in k if k.degree(n) == 2]) == G.order():
            return (k.size(weight), k)
    x_star = {}
    size_k_max = len(k_max)
    for (u, v, d) in G.edges(data=True):
        edge_count = 0
        d[weight] = original_edge_weights[u, v]
        for k in k_max:
            if (u, v) in k.edges():
                edge_count += 1
                k[u][v][weight] = original_edge_weights[u, v]
        x_star[u, v] = edge_count / size_k_max
    z_star = {}
    scale_factor = (G.order() - 1) / G.order()
    for (u, v) in x_star:
        frequency = x_star[u, v] + x_star[v, u]
        if frequency > 0:
            z_star[u, v] = scale_factor * frequency
    del x_star
    return (next(k_max.__iter__()).size(weight), z_star)

@nx._dispatch
def spanning_tree_distribution(G, z):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find the asadpour exponential distribution of spanning trees.\n\n    Solves the Maximum Entropy Convex Program in the Asadpour algorithm [1]_\n    using the approach in section 7 to build an exponential distribution of\n    undirected spanning trees.\n\n    This algorithm ensures that the probability of any edge in a spanning\n    tree is proportional to the sum of the probabilities of the tress\n    containing that edge over the sum of the probabilities of all spanning\n    trees of the graph.\n\n    Parameters\n    ----------\n    G : nx.MultiGraph\n        The undirected support graph for the Held Karp relaxation\n\n    z : dict\n        The output of `held_karp_ascent()`, a scaled version of the Held-Karp\n        solution.\n\n    Returns\n    -------\n    gamma : dict\n        The probability distribution which approximately preserves the marginal\n        probabilities of `z`.\n    '
    from math import exp
    from math import log as ln

    def q(e):
        if False:
            return 10
        '\n        The value of q(e) is described in the Asadpour paper is "the\n        probability that edge e will be included in a spanning tree T that is\n        chosen with probability proportional to exp(gamma(T))" which\n        basically means that it is the total probability of the edge appearing\n        across the whole distribution.\n\n        Parameters\n        ----------\n        e : tuple\n            The `(u, v)` tuple describing the edge we are interested in\n\n        Returns\n        -------\n        float\n            The probability that a spanning tree chosen according to the\n            current values of gamma will include edge `e`.\n        '
        for (u, v, d) in G.edges(data=True):
            d[lambda_key] = exp(gamma[u, v])
        G_Kirchhoff = nx.total_spanning_tree_weight(G, lambda_key)
        G_e = nx.contracted_edge(G, e, self_loops=False)
        G_e_Kirchhoff = nx.total_spanning_tree_weight(G_e, lambda_key)
        return exp(gamma[e[0], e[1]]) * G_e_Kirchhoff / G_Kirchhoff
    gamma = {}
    for (u, v, _) in G.edges:
        gamma[u, v] = 0
    EPSILON = 0.2
    lambda_key = "spanning_tree_distribution's secret attribute name for lambda"
    while True:
        in_range_count = 0
        for (u, v) in gamma:
            e = (u, v)
            q_e = q(e)
            z_e = z[e]
            if q_e > (1 + EPSILON) * z_e:
                delta = ln(q_e * (1 - (1 + EPSILON / 2) * z_e) / ((1 - q_e) * (1 + EPSILON / 2) * z_e))
                gamma[e] -= delta
                new_q_e = q(e)
                desired_q_e = (1 + EPSILON / 2) * z_e
                if round(new_q_e, 8) != round(desired_q_e, 8):
                    raise nx.NetworkXError(f'Unable to modify probability for edge ({u}, {v})')
            else:
                in_range_count += 1
        if in_range_count == len(gamma):
            break
    for (_, _, d) in G.edges(data=True):
        if lambda_key in d:
            del d[lambda_key]
    return gamma

@nx._dispatch(edge_attrs='weight')
def greedy_tsp(G, weight='weight', source=None):
    if False:
        return 10
    'Return a low cost cycle starting at `source` and its cost.\n\n    This approximates a solution to the traveling salesman problem.\n    It finds a cycle of all the nodes that a salesman can visit in order\n    to visit many nodes while minimizing total distance.\n    It uses a simple greedy algorithm.\n    In essence, this function returns a large cycle given a source point\n    for which the total cost of the cycle is minimized.\n\n    Parameters\n    ----------\n    G : Graph\n        The Graph should be a complete weighted undirected graph.\n        The distance between all pairs of nodes should be included.\n\n    weight : string, optional (default="weight")\n        Edge data key corresponding to the edge weight.\n        If any edge does not have this attribute the weight is set to 1.\n\n    source : node, optional (default: first node in list(G))\n        Starting node.  If None, defaults to ``next(iter(G))``\n\n    Returns\n    -------\n    cycle : list of nodes\n        Returns the cycle (list of nodes) that a salesman\n        can follow to minimize total weight of the trip.\n\n    Raises\n    ------\n    NetworkXError\n        If `G` is not complete, the algorithm raises an exception.\n\n    Examples\n    --------\n    >>> from networkx.algorithms import approximation as approx\n    >>> G = nx.DiGraph()\n    >>> G.add_weighted_edges_from({\n    ...     ("A", "B", 3), ("A", "C", 17), ("A", "D", 14), ("B", "A", 3),\n    ...     ("B", "C", 12), ("B", "D", 16), ("C", "A", 13),("C", "B", 12),\n    ...     ("C", "D", 4), ("D", "A", 14), ("D", "B", 15), ("D", "C", 2)\n    ... })\n    >>> cycle = approx.greedy_tsp(G, source="D")\n    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))\n    >>> cycle\n    [\'D\', \'C\', \'B\', \'A\', \'D\']\n    >>> cost\n    31\n\n    Notes\n    -----\n    This implementation of a greedy algorithm is based on the following:\n\n    - The algorithm adds a node to the solution at every iteration.\n    - The algorithm selects a node not already in the cycle whose connection\n      to the previous node adds the least cost to the cycle.\n\n    A greedy algorithm does not always give the best solution.\n    However, it can construct a first feasible solution which can\n    be passed as a parameter to an iterative improvement algorithm such\n    as Simulated Annealing, or Threshold Accepting.\n\n    Time complexity: It has a running time $O(|V|^2)$\n    '
    N = len(G) - 1
    if any((len(nbrdict) - (n in nbrdict) != N for (n, nbrdict) in G.adj.items())):
        raise nx.NetworkXError('G must be a complete graph.')
    if source is None:
        source = nx.utils.arbitrary_element(G)
    if G.number_of_nodes() == 2:
        neighbor = next(G.neighbors(source))
        return [source, neighbor, source]
    nodeset = set(G)
    nodeset.remove(source)
    cycle = [source]
    next_node = source
    while nodeset:
        nbrdict = G[next_node]
        next_node = min(nodeset, key=lambda n: nbrdict[n].get(weight, 1))
        cycle.append(next_node)
        nodeset.remove(next_node)
    cycle.append(cycle[0])
    return cycle

@py_random_state(9)
@nx._dispatch(edge_attrs='weight')
def simulated_annealing_tsp(G, init_cycle, weight='weight', source=None, temp=100, move='1-1', max_iterations=10, N_inner=100, alpha=0.01, seed=None):
    if False:
        i = 10
        return i + 15
    'Returns an approximate solution to the traveling salesman problem.\n\n    This function uses simulated annealing to approximate the minimal cost\n    cycle through the nodes. Starting from a suboptimal solution, simulated\n    annealing perturbs that solution, occasionally accepting changes that make\n    the solution worse to escape from a locally optimal solution. The chance\n    of accepting such changes decreases over the iterations to encourage\n    an optimal result.  In summary, the function returns a cycle starting\n    at `source` for which the total cost is minimized. It also returns the cost.\n\n    The chance of accepting a proposed change is related to a parameter called\n    the temperature (annealing has a physical analogue of steel hardening\n    as it cools). As the temperature is reduced, the chance of moves that\n    increase cost goes down.\n\n    Parameters\n    ----------\n    G : Graph\n        `G` should be a complete weighted graph.\n        The distance between all pairs of nodes should be included.\n\n    init_cycle : list of all nodes or "greedy"\n        The initial solution (a cycle through all nodes returning to the start).\n        This argument has no default to make you think about it.\n        If "greedy", use `greedy_tsp(G, weight)`.\n        Other common starting cycles are `list(G) + [next(iter(G))]` or the final\n        result of `simulated_annealing_tsp` when doing `threshold_accepting_tsp`.\n\n    weight : string, optional (default="weight")\n        Edge data key corresponding to the edge weight.\n        If any edge does not have this attribute the weight is set to 1.\n\n    source : node, optional (default: first node in list(G))\n        Starting node.  If None, defaults to ``next(iter(G))``\n\n    temp : int, optional (default=100)\n        The algorithm\'s temperature parameter. It represents the initial\n        value of temperature\n\n    move : "1-1" or "1-0" or function, optional (default="1-1")\n        Indicator of what move to use when finding new trial solutions.\n        Strings indicate two special built-in moves:\n\n        - "1-1": 1-1 exchange which transposes the position\n          of two elements of the current solution.\n          The function called is :func:`swap_two_nodes`.\n          For example if we apply 1-1 exchange in the solution\n          ``A = [3, 2, 1, 4, 3]``\n          we can get the following by the transposition of 1 and 4 elements:\n          ``A\' = [3, 2, 4, 1, 3]``\n        - "1-0": 1-0 exchange which moves an node in the solution\n          to a new position.\n          The function called is :func:`move_one_node`.\n          For example if we apply 1-0 exchange in the solution\n          ``A = [3, 2, 1, 4, 3]``\n          we can transfer the fourth element to the second position:\n          ``A\' = [3, 4, 2, 1, 3]``\n\n        You may provide your own functions to enact a move from\n        one solution to a neighbor solution. The function must take\n        the solution as input along with a `seed` input to control\n        random number generation (see the `seed` input here).\n        Your function should maintain the solution as a cycle with\n        equal first and last node and all others appearing once.\n        Your function should return the new solution.\n\n    max_iterations : int, optional (default=10)\n        Declared done when this number of consecutive iterations of\n        the outer loop occurs without any change in the best cost solution.\n\n    N_inner : int, optional (default=100)\n        The number of iterations of the inner loop.\n\n    alpha : float between (0, 1), optional (default=0.01)\n        Percentage of temperature decrease in each iteration\n        of outer loop\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    cycle : list of nodes\n        Returns the cycle (list of nodes) that a salesman\n        can follow to minimize total weight of the trip.\n\n    Raises\n    ------\n    NetworkXError\n        If `G` is not complete the algorithm raises an exception.\n\n    Examples\n    --------\n    >>> from networkx.algorithms import approximation as approx\n    >>> G = nx.DiGraph()\n    >>> G.add_weighted_edges_from({\n    ...     ("A", "B", 3), ("A", "C", 17), ("A", "D", 14), ("B", "A", 3),\n    ...     ("B", "C", 12), ("B", "D", 16), ("C", "A", 13),("C", "B", 12),\n    ...     ("C", "D", 4), ("D", "A", 14), ("D", "B", 15), ("D", "C", 2)\n    ... })\n    >>> cycle = approx.simulated_annealing_tsp(G, "greedy", source="D")\n    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))\n    >>> cycle\n    [\'D\', \'C\', \'B\', \'A\', \'D\']\n    >>> cost\n    31\n    >>> incycle = ["D", "B", "A", "C", "D"]\n    >>> cycle = approx.simulated_annealing_tsp(G, incycle, source="D")\n    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))\n    >>> cycle\n    [\'D\', \'C\', \'B\', \'A\', \'D\']\n    >>> cost\n    31\n\n    Notes\n    -----\n    Simulated Annealing is a metaheuristic local search algorithm.\n    The main characteristic of this algorithm is that it accepts\n    even solutions which lead to the increase of the cost in order\n    to escape from low quality local optimal solutions.\n\n    This algorithm needs an initial solution. If not provided, it is\n    constructed by a simple greedy algorithm. At every iteration, the\n    algorithm selects thoughtfully a neighbor solution.\n    Consider $c(x)$ cost of current solution and $c(x\')$ cost of a\n    neighbor solution.\n    If $c(x\') - c(x) <= 0$ then the neighbor solution becomes the current\n    solution for the next iteration. Otherwise, the algorithm accepts\n    the neighbor solution with probability $p = exp - ([c(x\') - c(x)] / temp)$.\n    Otherwise the current solution is retained.\n\n    `temp` is a parameter of the algorithm and represents temperature.\n\n    Time complexity:\n    For $N_i$ iterations of the inner loop and $N_o$ iterations of the\n    outer loop, this algorithm has running time $O(N_i * N_o * |V|)$.\n\n    For more information and how the algorithm is inspired see:\n    http://en.wikipedia.org/wiki/Simulated_annealing\n    '
    if move == '1-1':
        move = swap_two_nodes
    elif move == '1-0':
        move = move_one_node
    if init_cycle == 'greedy':
        cycle = greedy_tsp(G, weight=weight, source=source)
        if G.number_of_nodes() == 2:
            return cycle
    else:
        cycle = list(init_cycle)
        if source is None:
            source = cycle[0]
        elif source != cycle[0]:
            raise nx.NetworkXError('source must be first node in init_cycle')
        if cycle[0] != cycle[-1]:
            raise nx.NetworkXError('init_cycle must be a cycle. (return to start)')
        if len(cycle) - 1 != len(G) or len(set(G.nbunch_iter(cycle))) != len(G):
            raise nx.NetworkXError('init_cycle should be a cycle over all nodes in G.')
        N = len(G) - 1
        if any((len(nbrdict) - (n in nbrdict) != N for (n, nbrdict) in G.adj.items())):
            raise nx.NetworkXError('G must be a complete graph.')
        if G.number_of_nodes() == 2:
            neighbor = next(G.neighbors(source))
            return [source, neighbor, source]
    cost = sum((G[u][v].get(weight, 1) for (u, v) in pairwise(cycle)))
    count = 0
    best_cycle = cycle.copy()
    best_cost = cost
    while count <= max_iterations and temp > 0:
        count += 1
        for i in range(N_inner):
            adj_sol = move(cycle, seed)
            adj_cost = sum((G[u][v].get(weight, 1) for (u, v) in pairwise(adj_sol)))
            delta = adj_cost - cost
            if delta <= 0:
                cycle = adj_sol
                cost = adj_cost
                if cost < best_cost:
                    count = 0
                    best_cycle = cycle.copy()
                    best_cost = cost
            else:
                p = math.exp(-delta / temp)
                if p >= seed.random():
                    cycle = adj_sol
                    cost = adj_cost
        temp -= temp * alpha
    return best_cycle

@py_random_state(9)
@nx._dispatch(edge_attrs='weight')
def threshold_accepting_tsp(G, init_cycle, weight='weight', source=None, threshold=1, move='1-1', max_iterations=10, N_inner=100, alpha=0.1, seed=None):
    if False:
        i = 10
        return i + 15
    'Returns an approximate solution to the traveling salesman problem.\n\n    This function uses threshold accepting methods to approximate the minimal cost\n    cycle through the nodes. Starting from a suboptimal solution, threshold\n    accepting methods perturb that solution, accepting any changes that make\n    the solution no worse than increasing by a threshold amount. Improvements\n    in cost are accepted, but so are changes leading to small increases in cost.\n    This allows the solution to leave suboptimal local minima in solution space.\n    The threshold is decreased slowly as iterations proceed helping to ensure\n    an optimum. In summary, the function returns a cycle starting at `source`\n    for which the total cost is minimized.\n\n    Parameters\n    ----------\n    G : Graph\n        `G` should be a complete weighted graph.\n        The distance between all pairs of nodes should be included.\n\n    init_cycle : list or "greedy"\n        The initial solution (a cycle through all nodes returning to the start).\n        This argument has no default to make you think about it.\n        If "greedy", use `greedy_tsp(G, weight)`.\n        Other common starting cycles are `list(G) + [next(iter(G))]` or the final\n        result of `simulated_annealing_tsp` when doing `threshold_accepting_tsp`.\n\n    weight : string, optional (default="weight")\n        Edge data key corresponding to the edge weight.\n        If any edge does not have this attribute the weight is set to 1.\n\n    source : node, optional (default: first node in list(G))\n        Starting node.  If None, defaults to ``next(iter(G))``\n\n    threshold : int, optional (default=1)\n        The algorithm\'s threshold parameter. It represents the initial\n        threshold\'s value\n\n    move : "1-1" or "1-0" or function, optional (default="1-1")\n        Indicator of what move to use when finding new trial solutions.\n        Strings indicate two special built-in moves:\n\n        - "1-1": 1-1 exchange which transposes the position\n          of two elements of the current solution.\n          The function called is :func:`swap_two_nodes`.\n          For example if we apply 1-1 exchange in the solution\n          ``A = [3, 2, 1, 4, 3]``\n          we can get the following by the transposition of 1 and 4 elements:\n          ``A\' = [3, 2, 4, 1, 3]``\n        - "1-0": 1-0 exchange which moves an node in the solution\n          to a new position.\n          The function called is :func:`move_one_node`.\n          For example if we apply 1-0 exchange in the solution\n          ``A = [3, 2, 1, 4, 3]``\n          we can transfer the fourth element to the second position:\n          ``A\' = [3, 4, 2, 1, 3]``\n\n        You may provide your own functions to enact a move from\n        one solution to a neighbor solution. The function must take\n        the solution as input along with a `seed` input to control\n        random number generation (see the `seed` input here).\n        Your function should maintain the solution as a cycle with\n        equal first and last node and all others appearing once.\n        Your function should return the new solution.\n\n    max_iterations : int, optional (default=10)\n        Declared done when this number of consecutive iterations of\n        the outer loop occurs without any change in the best cost solution.\n\n    N_inner : int, optional (default=100)\n        The number of iterations of the inner loop.\n\n    alpha : float between (0, 1), optional (default=0.1)\n        Percentage of threshold decrease when there is at\n        least one acceptance of a neighbor solution.\n        If no inner loop moves are accepted the threshold remains unchanged.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    cycle : list of nodes\n        Returns the cycle (list of nodes) that a salesman\n        can follow to minimize total weight of the trip.\n\n    Raises\n    ------\n    NetworkXError\n        If `G` is not complete the algorithm raises an exception.\n\n    Examples\n    --------\n    >>> from networkx.algorithms import approximation as approx\n    >>> G = nx.DiGraph()\n    >>> G.add_weighted_edges_from({\n    ...     ("A", "B", 3), ("A", "C", 17), ("A", "D", 14), ("B", "A", 3),\n    ...     ("B", "C", 12), ("B", "D", 16), ("C", "A", 13),("C", "B", 12),\n    ...     ("C", "D", 4), ("D", "A", 14), ("D", "B", 15), ("D", "C", 2)\n    ... })\n    >>> cycle = approx.threshold_accepting_tsp(G, "greedy", source="D")\n    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))\n    >>> cycle\n    [\'D\', \'C\', \'B\', \'A\', \'D\']\n    >>> cost\n    31\n    >>> incycle = ["D", "B", "A", "C", "D"]\n    >>> cycle = approx.threshold_accepting_tsp(G, incycle, source="D")\n    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))\n    >>> cycle\n    [\'D\', \'C\', \'B\', \'A\', \'D\']\n    >>> cost\n    31\n\n    Notes\n    -----\n    Threshold Accepting is a metaheuristic local search algorithm.\n    The main characteristic of this algorithm is that it accepts\n    even solutions which lead to the increase of the cost in order\n    to escape from low quality local optimal solutions.\n\n    This algorithm needs an initial solution. This solution can be\n    constructed by a simple greedy algorithm. At every iteration, it\n    selects thoughtfully a neighbor solution.\n    Consider $c(x)$ cost of current solution and $c(x\')$ cost of\n    neighbor solution.\n    If $c(x\') - c(x) <= threshold$ then the neighbor solution becomes the current\n    solution for the next iteration, where the threshold is named threshold.\n\n    In comparison to the Simulated Annealing algorithm, the Threshold\n    Accepting algorithm does not accept very low quality solutions\n    (due to the presence of the threshold value). In the case of\n    Simulated Annealing, even a very low quality solution can\n    be accepted with probability $p$.\n\n    Time complexity:\n    It has a running time $O(m * n * |V|)$ where $m$ and $n$ are the number\n    of times the outer and inner loop run respectively.\n\n    For more information and how algorithm is inspired see:\n    https://doi.org/10.1016/0021-9991(90)90201-B\n\n    See Also\n    --------\n    simulated_annealing_tsp\n\n    '
    if move == '1-1':
        move = swap_two_nodes
    elif move == '1-0':
        move = move_one_node
    if init_cycle == 'greedy':
        cycle = greedy_tsp(G, weight=weight, source=source)
        if G.number_of_nodes() == 2:
            return cycle
    else:
        cycle = list(init_cycle)
        if source is None:
            source = cycle[0]
        elif source != cycle[0]:
            raise nx.NetworkXError('source must be first node in init_cycle')
        if cycle[0] != cycle[-1]:
            raise nx.NetworkXError('init_cycle must be a cycle. (return to start)')
        if len(cycle) - 1 != len(G) or len(set(G.nbunch_iter(cycle))) != len(G):
            raise nx.NetworkXError('init_cycle is not all and only nodes.')
        N = len(G) - 1
        if any((len(nbrdict) - (n in nbrdict) != N for (n, nbrdict) in G.adj.items())):
            raise nx.NetworkXError('G must be a complete graph.')
        if G.number_of_nodes() == 2:
            neighbor = list(G.neighbors(source))[0]
            return [source, neighbor, source]
    cost = sum((G[u][v].get(weight, 1) for (u, v) in pairwise(cycle)))
    count = 0
    best_cycle = cycle.copy()
    best_cost = cost
    while count <= max_iterations:
        count += 1
        accepted = False
        for i in range(N_inner):
            adj_sol = move(cycle, seed)
            adj_cost = sum((G[u][v].get(weight, 1) for (u, v) in pairwise(adj_sol)))
            delta = adj_cost - cost
            if delta <= threshold:
                accepted = True
                cycle = adj_sol
                cost = adj_cost
                if cost < best_cost:
                    count = 0
                    best_cycle = cycle.copy()
                    best_cost = cost
        if accepted:
            threshold -= threshold * alpha
    return best_cycle