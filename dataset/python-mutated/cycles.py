"""
========================
Cycle finding algorithms
========================
"""
from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
__all__ = ['cycle_basis', 'simple_cycles', 'recursive_simple_cycles', 'find_cycle', 'minimum_cycle_basis', 'chordless_cycles', 'girth']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def cycle_basis(G, root=None):
    if False:
        return 10
    'Returns a list of cycles which form a basis for cycles of G.\n\n    A basis for cycles of a network is a minimal collection of\n    cycles such that any cycle in the network can be written\n    as a sum of cycles in the basis.  Here summation of cycles\n    is defined as "exclusive or" of the edges. Cycle bases are\n    useful, e.g. when deriving equations for electric circuits\n    using Kirchhoff\'s Laws.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n    root : node, optional\n       Specify starting node for basis.\n\n    Returns\n    -------\n    A list of cycle lists.  Each cycle list is a list of nodes\n    which forms a cycle (loop) in G.\n\n    Examples\n    --------\n    >>> G = nx.Graph()\n    >>> nx.add_cycle(G, [0, 1, 2, 3])\n    >>> nx.add_cycle(G, [0, 3, 4, 5])\n    >>> nx.cycle_basis(G, 0)\n    [[3, 4, 5, 0], [1, 2, 3, 0]]\n\n    Notes\n    -----\n    This is adapted from algorithm CACM 491 [1]_.\n\n    References\n    ----------\n    .. [1] Paton, K. An algorithm for finding a fundamental set of\n       cycles of a graph. Comm. ACM 12, 9 (Sept 1969), 514-518.\n\n    See Also\n    --------\n    simple_cycles\n    '
    gnodes = dict.fromkeys(G)
    cycles = []
    while gnodes:
        if root is None:
            root = gnodes.popitem()[0]
        stack = [root]
        pred = {root: root}
        used = {root: set()}
        while stack:
            z = stack.pop()
            zused = used[z]
            for nbr in G[z]:
                if nbr not in used:
                    pred[nbr] = z
                    stack.append(nbr)
                    used[nbr] = {z}
                elif nbr == z:
                    cycles.append([z])
                elif nbr not in zused:
                    pn = used[nbr]
                    cycle = [nbr, z]
                    p = pred[z]
                    while p not in pn:
                        cycle.append(p)
                        p = pred[p]
                    cycle.append(p)
                    cycles.append(cycle)
                    used[nbr].add(z)
        for node in pred:
            gnodes.pop(node, None)
        root = None
    return cycles

@nx._dispatch
def simple_cycles(G, length_bound=None):
    if False:
        while True:
            i = 10
    "Find simple cycles (elementary circuits) of a graph.\n\n    A `simple cycle`, or `elementary circuit`, is a closed path where\n    no node appears twice.  In a directed graph, two simple cycles are distinct\n    if they are not cyclic permutations of each other.  In an undirected graph,\n    two simple cycles are distinct if they are not cyclic permutations of each\n    other nor of the other's reversal.\n\n    Optionally, the cycles are bounded in length.  In the unbounded case, we use\n    a nonrecursive, iterator/generator version of Johnson's algorithm [1]_.  In\n    the bounded case, we use a version of the algorithm of Gupta and\n    Suzumura[2]_. There may be better algorithms for some cases [3]_ [4]_ [5]_.\n\n    The algorithms of Johnson, and Gupta and Suzumura, are enhanced by some\n    well-known preprocessing techniques.  When G is directed, we restrict our\n    attention to strongly connected components of G, generate all simple cycles\n    containing a certain node, remove that node, and further decompose the\n    remainder into strongly connected components.  When G is undirected, we\n    restrict our attention to biconnected components, generate all simple cycles\n    containing a particular edge, remove that edge, and further decompose the\n    remainder into biconnected components.\n\n    Note that multigraphs are supported by this function -- and in undirected\n    multigraphs, a pair of parallel edges is considered a cycle of length 2.\n    Likewise, self-loops are considered to be cycles of length 1.  We define\n    cycles as sequences of nodes; so the presence of loops and parallel edges\n    does not change the number of simple cycles in a graph.\n\n    Parameters\n    ----------\n    G : NetworkX DiGraph\n       A directed graph\n\n    length_bound : int or None, optional (default=None)\n       If length_bound is an int, generate all simple cycles of G with length at\n       most length_bound.  Otherwise, generate all simple cycles of G.\n\n    Yields\n    ------\n    list of nodes\n       Each cycle is represented by a list of nodes along the cycle.\n\n    Examples\n    --------\n    >>> edges = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]\n    >>> G = nx.DiGraph(edges)\n    >>> sorted(nx.simple_cycles(G))\n    [[0], [0, 1, 2], [0, 2], [1, 2], [2]]\n\n    To filter the cycles so that they don't include certain nodes or edges,\n    copy your graph and eliminate those nodes or edges before calling.\n    For example, to exclude self-loops from the above example:\n\n    >>> H = G.copy()\n    >>> H.remove_edges_from(nx.selfloop_edges(G))\n    >>> sorted(nx.simple_cycles(H))\n    [[0, 1, 2], [0, 2], [1, 2]]\n\n    Notes\n    -----\n    When length_bound is None, the time complexity is $O((n+e)(c+1))$ for $n$\n    nodes, $e$ edges and $c$ simple circuits.  Otherwise, when length_bound > 1,\n    the time complexity is $O((c+n)(k-1)d^k)$ where $d$ is the average degree of\n    the nodes of G and $k$ = length_bound.\n\n    Raises\n    ------\n    ValueError\n        when length_bound < 0.\n\n    References\n    ----------\n    .. [1] Finding all the elementary circuits of a directed graph.\n       D. B. Johnson, SIAM Journal on Computing 4, no. 1, 77-84, 1975.\n       https://doi.org/10.1137/0204007\n    .. [2] Finding All Bounded-Length Simple Cycles in a Directed Graph\n       A. Gupta and T. Suzumura https://arxiv.org/abs/2105.10094\n    .. [3] Enumerating the cycles of a digraph: a new preprocessing strategy.\n       G. Loizou and P. Thanish, Information Sciences, v. 27, 163-182, 1982.\n    .. [4] A search strategy for the elementary cycles of a directed graph.\n       J.L. Szwarcfiter and P.E. Lauer, BIT NUMERICAL MATHEMATICS,\n       v. 16, no. 2, 192-204, 1976.\n    .. [5] Optimal Listing of Cycles and st-Paths in Undirected Graphs\n        R. Ferreira and R. Grossi and A. Marino and N. Pisanti and R. Rizzi and\n        G. Sacomoto https://arxiv.org/abs/1205.2766\n\n    See Also\n    --------\n    cycle_basis\n    chordless_cycles\n    "
    if length_bound is not None:
        if length_bound == 0:
            return
        elif length_bound < 0:
            raise ValueError('length bound must be non-negative')
    directed = G.is_directed()
    yield from ([v] for (v, Gv) in G.adj.items() if v in Gv)
    if length_bound is not None and length_bound == 1:
        return
    if G.is_multigraph() and (not directed):
        visited = set()
        for (u, Gu) in G.adj.items():
            multiplicity = ((v, len(Guv)) for (v, Guv) in Gu.items() if v in visited)
            yield from ([u, v] for (v, m) in multiplicity if m > 1)
            visited.add(u)
    if directed:
        G = nx.DiGraph(((u, v) for (u, Gu) in G.adj.items() for v in Gu if v != u))
    else:
        G = nx.Graph(((u, v) for (u, Gu) in G.adj.items() for v in Gu if v != u))
    if length_bound is not None and length_bound == 2:
        if directed:
            visited = set()
            for (u, Gu) in G.adj.items():
                yield from ([v, u] for v in visited.intersection(Gu) if G.has_edge(v, u))
                visited.add(u)
        return
    if directed:
        yield from _directed_cycle_search(G, length_bound)
    else:
        yield from _undirected_cycle_search(G, length_bound)

def _directed_cycle_search(G, length_bound):
    if False:
        for i in range(10):
            print('nop')
    'A dispatch function for `simple_cycles` for directed graphs.\n\n    We generate all cycles of G through binary partition.\n\n        1. Pick a node v in G which belongs to at least one cycle\n            a. Generate all cycles of G which contain the node v.\n            b. Recursively generate all cycles of G \\ v.\n\n    This is accomplished through the following:\n\n        1. Compute the strongly connected components SCC of G.\n        2. Select and remove a biconnected component C from BCC.  Select a\n           non-tree edge (u, v) of a depth-first search of G[C].\n        3. For each simple cycle P containing v in G[C], yield P.\n        4. Add the biconnected components of G[C \\ v] to BCC.\n\n    If the parameter length_bound is not None, then step 3 will be limited to\n    simple cycles of length at most length_bound.\n\n    Parameters\n    ----------\n    G : NetworkX DiGraph\n       A directed graph\n\n    length_bound : int or None\n       If length_bound is an int, generate all simple cycles of G with length at most length_bound.\n       Otherwise, generate all simple cycles of G.\n\n    Yields\n    ------\n    list of nodes\n       Each cycle is represented by a list of nodes along the cycle.\n    '
    scc = nx.strongly_connected_components
    components = [c for c in scc(G) if len(c) >= 2]
    while components:
        c = components.pop()
        Gc = G.subgraph(c)
        v = next(iter(c))
        if length_bound is None:
            yield from _johnson_cycle_search(Gc, [v])
        else:
            yield from _bounded_cycle_search(Gc, [v], length_bound)
        G.remove_node(v)
        components.extend((c for c in scc(Gc) if len(c) >= 2))

def _undirected_cycle_search(G, length_bound):
    if False:
        i = 10
        return i + 15
    'A dispatch function for `simple_cycles` for undirected graphs.\n\n    We generate all cycles of G through binary partition.\n\n        1. Pick an edge (u, v) in G which belongs to at least one cycle\n            a. Generate all cycles of G which contain the edge (u, v)\n            b. Recursively generate all cycles of G \\ (u, v)\n\n    This is accomplished through the following:\n\n        1. Compute the biconnected components BCC of G.\n        2. Select and remove a biconnected component C from BCC.  Select a\n           non-tree edge (u, v) of a depth-first search of G[C].\n        3. For each (v -> u) path P remaining in G[C] \\ (u, v), yield P.\n        4. Add the biconnected components of G[C] \\ (u, v) to BCC.\n\n    If the parameter length_bound is not None, then step 3 will be limited to simple paths\n    of length at most length_bound.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n       An undirected graph\n\n    length_bound : int or None\n       If length_bound is an int, generate all simple cycles of G with length at most length_bound.\n       Otherwise, generate all simple cycles of G.\n\n    Yields\n    ------\n    list of nodes\n       Each cycle is represented by a list of nodes along the cycle.\n    '
    bcc = nx.biconnected_components
    components = [c for c in bcc(G) if len(c) >= 3]
    while components:
        c = components.pop()
        Gc = G.subgraph(c)
        uv = list(next(iter(Gc.edges)))
        G.remove_edge(*uv)
        if length_bound is None:
            yield from _johnson_cycle_search(Gc, uv)
        else:
            yield from _bounded_cycle_search(Gc, uv, length_bound)
        components.extend((c for c in bcc(Gc) if len(c) >= 3))

class _NeighborhoodCache(dict):
    """Very lightweight graph wrapper which caches neighborhoods as list.

    This dict subclass uses the __missing__ functionality to query graphs for
    their neighborhoods, and store the result as a list.  This is used to avoid
    the performance penalty incurred by subgraph views.
    """

    def __init__(self, G):
        if False:
            i = 10
            return i + 15
        self.G = G

    def __missing__(self, v):
        if False:
            print('Hello World!')
        Gv = self[v] = list(self.G[v])
        return Gv

def _johnson_cycle_search(G, path):
    if False:
        return 10
    'The main loop of the cycle-enumeration algorithm of Johnson.\n\n    Parameters\n    ----------\n    G : NetworkX Graph or DiGraph\n       A graph\n\n    path : list\n       A cycle prefix.  All cycles generated will begin with this prefix.\n\n    Yields\n    ------\n    list of nodes\n       Each cycle is represented by a list of nodes along the cycle.\n\n    References\n    ----------\n        .. [1] Finding all the elementary circuits of a directed graph.\n       D. B. Johnson, SIAM Journal on Computing 4, no. 1, 77-84, 1975.\n       https://doi.org/10.1137/0204007\n\n    '
    G = _NeighborhoodCache(G)
    blocked = set(path)
    B = defaultdict(set)
    start = path[0]
    stack = [iter(G[path[-1]])]
    closed = [False]
    while stack:
        nbrs = stack[-1]
        for w in nbrs:
            if w == start:
                yield path[:]
                closed[-1] = True
            elif w not in blocked:
                path.append(w)
                closed.append(False)
                stack.append(iter(G[w]))
                blocked.add(w)
                break
        else:
            stack.pop()
            v = path.pop()
            if closed.pop():
                if closed:
                    closed[-1] = True
                unblock_stack = {v}
                while unblock_stack:
                    u = unblock_stack.pop()
                    if u in blocked:
                        blocked.remove(u)
                        unblock_stack.update(B[u])
                        B[u].clear()
            else:
                for w in G[v]:
                    B[w].add(v)

def _bounded_cycle_search(G, path, length_bound):
    if False:
        i = 10
        return i + 15
    'The main loop of the cycle-enumeration algorithm of Gupta and Suzumura.\n\n    Parameters\n    ----------\n    G : NetworkX Graph or DiGraph\n       A graph\n\n    path : list\n       A cycle prefix.  All cycles generated will begin with this prefix.\n\n    length_bound: int\n        A length bound.  All cycles generated will have length at most length_bound.\n\n    Yields\n    ------\n    list of nodes\n       Each cycle is represented by a list of nodes along the cycle.\n\n    References\n    ----------\n    .. [1] Finding All Bounded-Length Simple Cycles in a Directed Graph\n       A. Gupta and T. Suzumura https://arxiv.org/abs/2105.10094\n\n    '
    G = _NeighborhoodCache(G)
    lock = {v: 0 for v in path}
    B = defaultdict(set)
    start = path[0]
    stack = [iter(G[path[-1]])]
    blen = [length_bound]
    while stack:
        nbrs = stack[-1]
        for w in nbrs:
            if w == start:
                yield path[:]
                blen[-1] = 1
            elif len(path) < lock.get(w, length_bound):
                path.append(w)
                blen.append(length_bound)
                lock[w] = len(path)
                stack.append(iter(G[w]))
                break
        else:
            stack.pop()
            v = path.pop()
            bl = blen.pop()
            if blen:
                blen[-1] = min(blen[-1], bl)
            if bl < length_bound:
                relax_stack = [(bl, v)]
                while relax_stack:
                    (bl, u) = relax_stack.pop()
                    if lock.get(u, length_bound) < length_bound - bl + 1:
                        lock[u] = length_bound - bl + 1
                        relax_stack.extend(((bl + 1, w) for w in B[u].difference(path)))
            else:
                for w in G[v]:
                    B[w].add(v)

@nx._dispatch
def chordless_cycles(G, length_bound=None):
    if False:
        return 10
    "Find simple chordless cycles of a graph.\n\n    A `simple cycle` is a closed path where no node appears twice.  In a simple\n    cycle, a `chord` is an additional edge between two nodes in the cycle.  A\n    `chordless cycle` is a simple cycle without chords.  Said differently, a\n    chordless cycle is a cycle C in a graph G where the number of edges in the\n    induced graph G[C] is equal to the length of `C`.\n\n    Note that some care must be taken in the case that G is not a simple graph\n    nor a simple digraph.  Some authors limit the definition of chordless cycles\n    to have a prescribed minimum length; we do not.\n\n        1. We interpret self-loops to be chordless cycles, except in multigraphs\n           with multiple loops in parallel.  Likewise, in a chordless cycle of\n           length greater than 1, there can be no nodes with self-loops.\n\n        2. We interpret directed two-cycles to be chordless cycles, except in\n           multi-digraphs when any edge in a two-cycle has a parallel copy.\n\n        3. We interpret parallel pairs of undirected edges as two-cycles, except\n           when a third (or more) parallel edge exists between the two nodes.\n\n        4. Generalizing the above, edges with parallel clones may not occur in\n           chordless cycles.\n\n    In a directed graph, two chordless cycles are distinct if they are not\n    cyclic permutations of each other.  In an undirected graph, two chordless\n    cycles are distinct if they are not cyclic permutations of each other nor of\n    the other's reversal.\n\n    Optionally, the cycles are bounded in length.\n\n    We use an algorithm strongly inspired by that of Dias et al [1]_.  It has\n    been modified in the following ways:\n\n        1. Recursion is avoided, per Python's limitations\n\n        2. The labeling function is not necessary, because the starting paths\n            are chosen (and deleted from the host graph) to prevent multiple\n            occurrences of the same path\n\n        3. The search is optionally bounded at a specified length\n\n        4. Support for directed graphs is provided by extending cycles along\n            forward edges, and blocking nodes along forward and reverse edges\n\n        5. Support for multigraphs is provided by omitting digons from the set\n            of forward edges\n\n    Parameters\n    ----------\n    G : NetworkX DiGraph\n       A directed graph\n\n    length_bound : int or None, optional (default=None)\n       If length_bound is an int, generate all simple cycles of G with length at\n       most length_bound.  Otherwise, generate all simple cycles of G.\n\n    Yields\n    ------\n    list of nodes\n       Each cycle is represented by a list of nodes along the cycle.\n\n    Examples\n    --------\n    >>> sorted(list(nx.chordless_cycles(nx.complete_graph(4))))\n    [[1, 0, 2], [1, 0, 3], [2, 0, 3], [2, 1, 3]]\n\n    Notes\n    -----\n    When length_bound is None, and the graph is simple, the time complexity is\n    $O((n+e)(c+1))$ for $n$ nodes, $e$ edges and $c$ chordless cycles.\n\n    Raises\n    ------\n    ValueError\n        when length_bound < 0.\n\n    References\n    ----------\n    .. [1] Efficient enumeration of chordless cycles\n       E. Dias and D. Castonguay and H. Longo and W.A.R. Jradi\n       https://arxiv.org/abs/1309.1051\n\n    See Also\n    --------\n    simple_cycles\n    "
    if length_bound is not None:
        if length_bound == 0:
            return
        elif length_bound < 0:
            raise ValueError('length bound must be non-negative')
    directed = G.is_directed()
    multigraph = G.is_multigraph()
    if multigraph:
        yield from ([v] for (v, Gv) in G.adj.items() if len(Gv.get(v, ())) == 1)
    else:
        yield from ([v] for (v, Gv) in G.adj.items() if v in Gv)
    if length_bound is not None and length_bound == 1:
        return
    if directed:
        F = nx.DiGraph(((u, v) for (u, Gu) in G.adj.items() if u not in Gu for v in Gu))
        B = F.to_undirected(as_view=False)
    else:
        F = nx.Graph(((u, v) for (u, Gu) in G.adj.items() if u not in Gu for v in Gu))
        B = None
    if multigraph:
        if not directed:
            B = F.copy()
            visited = set()
        for (u, Gu) in G.adj.items():
            if directed:
                multiplicity = ((v, len(Guv)) for (v, Guv) in Gu.items())
                for (v, m) in multiplicity:
                    if m > 1:
                        F.remove_edges_from(((u, v), (v, u)))
            else:
                multiplicity = ((v, len(Guv)) for (v, Guv) in Gu.items() if v in visited)
                for (v, m) in multiplicity:
                    if m == 2:
                        yield [u, v]
                    if m > 1:
                        F.remove_edge(u, v)
                visited.add(u)
    if directed:
        for (u, Fu) in F.adj.items():
            digons = [[u, v] for v in Fu if F.has_edge(v, u)]
            yield from digons
            F.remove_edges_from(digons)
            F.remove_edges_from((e[::-1] for e in digons))
    if length_bound is not None and length_bound == 2:
        return
    if directed:
        separate = nx.strongly_connected_components

        def stems(C, v):
            if False:
                for i in range(10):
                    print('nop')
            for (u, w) in product(C.pred[v], C.succ[v]):
                if not G.has_edge(u, w):
                    yield ([u, v, w], F.has_edge(w, u))
    else:
        separate = nx.biconnected_components

        def stems(C, v):
            if False:
                print('Hello World!')
            yield from (([u, v, w], F.has_edge(w, u)) for (u, w) in combinations(C[v], 2))
    components = [c for c in separate(F) if len(c) > 2]
    while components:
        c = components.pop()
        v = next(iter(c))
        Fc = F.subgraph(c)
        Fcc = Bcc = None
        for (S, is_triangle) in stems(Fc, v):
            if is_triangle:
                yield S
            else:
                if Fcc is None:
                    Fcc = _NeighborhoodCache(Fc)
                    Bcc = Fcc if B is None else _NeighborhoodCache(B.subgraph(c))
                yield from _chordless_cycle_search(Fcc, Bcc, S, length_bound)
        components.extend((c for c in separate(F.subgraph(c - {v})) if len(c) > 2))

def _chordless_cycle_search(F, B, path, length_bound):
    if False:
        return 10
    "The main loop for chordless cycle enumeration.\n\n    This algorithm is strongly inspired by that of Dias et al [1]_.  It has been\n    modified in the following ways:\n\n        1. Recursion is avoided, per Python's limitations\n\n        2. The labeling function is not necessary, because the starting paths\n            are chosen (and deleted from the host graph) to prevent multiple\n            occurrences of the same path\n\n        3. The search is optionally bounded at a specified length\n\n        4. Support for directed graphs is provided by extending cycles along\n            forward edges, and blocking nodes along forward and reverse edges\n\n        5. Support for multigraphs is provided by omitting digons from the set\n            of forward edges\n\n    Parameters\n    ----------\n    F : _NeighborhoodCache\n       A graph of forward edges to follow in constructing cycles\n\n    B : _NeighborhoodCache\n       A graph of blocking edges to prevent the production of chordless cycles\n\n    path : list\n       A cycle prefix.  All cycles generated will begin with this prefix.\n\n    length_bound : int\n       A length bound.  All cycles generated will have length at most length_bound.\n\n\n    Yields\n    ------\n    list of nodes\n       Each cycle is represented by a list of nodes along the cycle.\n\n    References\n    ----------\n    .. [1] Efficient enumeration of chordless cycles\n       E. Dias and D. Castonguay and H. Longo and W.A.R. Jradi\n       https://arxiv.org/abs/1309.1051\n\n    "
    blocked = defaultdict(int)
    target = path[0]
    blocked[path[1]] = 1
    for w in path[1:]:
        for v in B[w]:
            blocked[v] += 1
    stack = [iter(F[path[2]])]
    while stack:
        nbrs = stack[-1]
        for w in nbrs:
            if blocked[w] == 1 and (length_bound is None or len(path) < length_bound):
                Fw = F[w]
                if target in Fw:
                    yield (path + [w])
                else:
                    Bw = B[w]
                    if target in Bw:
                        continue
                    for v in Bw:
                        blocked[v] += 1
                    path.append(w)
                    stack.append(iter(Fw))
                    break
        else:
            stack.pop()
            for v in B[path.pop()]:
                blocked[v] -= 1

@not_implemented_for('undirected')
@nx._dispatch
def recursive_simple_cycles(G):
    if False:
        for i in range(10):
            print('nop')
    'Find simple cycles (elementary circuits) of a directed graph.\n\n    A `simple cycle`, or `elementary circuit`, is a closed path where\n    no node appears twice. Two elementary circuits are distinct if they\n    are not cyclic permutations of each other.\n\n    This version uses a recursive algorithm to build a list of cycles.\n    You should probably use the iterator version called simple_cycles().\n    Warning: This recursive version uses lots of RAM!\n    It appears in NetworkX for pedagogical value.\n\n    Parameters\n    ----------\n    G : NetworkX DiGraph\n       A directed graph\n\n    Returns\n    -------\n    A list of cycles, where each cycle is represented by a list of nodes\n    along the cycle.\n\n    Example:\n\n    >>> edges = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]\n    >>> G = nx.DiGraph(edges)\n    >>> nx.recursive_simple_cycles(G)\n    [[0], [2], [0, 1, 2], [0, 2], [1, 2]]\n\n    Notes\n    -----\n    The implementation follows pp. 79-80 in [1]_.\n\n    The time complexity is $O((n+e)(c+1))$ for $n$ nodes, $e$ edges and $c$\n    elementary circuits.\n\n    References\n    ----------\n    .. [1] Finding all the elementary circuits of a directed graph.\n       D. B. Johnson, SIAM Journal on Computing 4, no. 1, 77-84, 1975.\n       https://doi.org/10.1137/0204007\n\n    See Also\n    --------\n    simple_cycles, cycle_basis\n    '

    def _unblock(thisnode):
        if False:
            for i in range(10):
                print('nop')
        'Recursively unblock and remove nodes from B[thisnode].'
        if blocked[thisnode]:
            blocked[thisnode] = False
            while B[thisnode]:
                _unblock(B[thisnode].pop())

    def circuit(thisnode, startnode, component):
        if False:
            while True:
                i = 10
        closed = False
        path.append(thisnode)
        blocked[thisnode] = True
        for nextnode in component[thisnode]:
            if nextnode == startnode:
                result.append(path[:])
                closed = True
            elif not blocked[nextnode]:
                if circuit(nextnode, startnode, component):
                    closed = True
        if closed:
            _unblock(thisnode)
        else:
            for nextnode in component[thisnode]:
                if thisnode not in B[nextnode]:
                    B[nextnode].append(thisnode)
        path.pop()
        return closed
    path = []
    blocked = defaultdict(bool)
    B = defaultdict(list)
    result = []
    for v in G:
        if G.has_edge(v, v):
            result.append([v])
            G.remove_edge(v, v)
    ordering = dict(zip(G, range(len(G))))
    for s in ordering:
        subgraph = G.subgraph((node for node in G if ordering[node] >= ordering[s]))
        strongcomp = nx.strongly_connected_components(subgraph)
        mincomp = min(strongcomp, key=lambda ns: min((ordering[n] for n in ns)))
        component = G.subgraph(mincomp)
        if len(component) > 1:
            startnode = min(component, key=ordering.__getitem__)
            for node in component:
                blocked[node] = False
                B[node][:] = []
            dummy = circuit(startnode, startnode, component)
    return result

@nx._dispatch
def find_cycle(G, source=None, orientation=None):
    if False:
        print('Hello World!')
    'Returns a cycle found via depth-first traversal.\n\n    The cycle is a list of edges indicating the cyclic path.\n    Orientation of directed edges is controlled by `orientation`.\n\n    Parameters\n    ----------\n    G : graph\n        A directed/undirected graph/multigraph.\n\n    source : node, list of nodes\n        The node from which the traversal begins. If None, then a source\n        is chosen arbitrarily and repeatedly until all edges from each node in\n        the graph are searched.\n\n    orientation : None | \'original\' | \'reverse\' | \'ignore\' (default: None)\n        For directed graphs and directed multigraphs, edge traversals need not\n        respect the original orientation of the edges.\n        When set to \'reverse\' every edge is traversed in the reverse direction.\n        When set to \'ignore\', every edge is treated as undirected.\n        When set to \'original\', every edge is treated as directed.\n        In all three cases, the yielded edge tuples add a last entry to\n        indicate the direction in which that edge was traversed.\n        If orientation is None, the yielded edge has no direction indicated.\n        The direction is respected, but not reported.\n\n    Returns\n    -------\n    edges : directed edges\n        A list of directed edges indicating the path taken for the loop.\n        If no cycle is found, then an exception is raised.\n        For graphs, an edge is of the form `(u, v)` where `u` and `v`\n        are the tail and head of the edge as determined by the traversal.\n        For multigraphs, an edge is of the form `(u, v, key)`, where `key` is\n        the key of the edge. When the graph is directed, then `u` and `v`\n        are always in the order of the actual directed edge.\n        If orientation is not None then the edge tuple is extended to include\n        the direction of traversal (\'forward\' or \'reverse\') on that edge.\n\n    Raises\n    ------\n    NetworkXNoCycle\n        If no cycle was found.\n\n    Examples\n    --------\n    In this example, we construct a DAG and find, in the first call, that there\n    are no directed cycles, and so an exception is raised. In the second call,\n    we ignore edge orientations and find that there is an undirected cycle.\n    Note that the second call finds a directed cycle while effectively\n    traversing an undirected graph, and so, we found an "undirected cycle".\n    This means that this DAG structure does not form a directed tree (which\n    is also known as a polytree).\n\n    >>> G = nx.DiGraph([(0, 1), (0, 2), (1, 2)])\n    >>> nx.find_cycle(G, orientation="original")\n    Traceback (most recent call last):\n        ...\n    networkx.exception.NetworkXNoCycle: No cycle found.\n    >>> list(nx.find_cycle(G, orientation="ignore"))\n    [(0, 1, \'forward\'), (1, 2, \'forward\'), (0, 2, \'reverse\')]\n\n    See Also\n    --------\n    simple_cycles\n    '
    if not G.is_directed() or orientation in (None, 'original'):

        def tailhead(edge):
            if False:
                while True:
                    i = 10
            return edge[:2]
    elif orientation == 'reverse':

        def tailhead(edge):
            if False:
                while True:
                    i = 10
            return (edge[1], edge[0])
    elif orientation == 'ignore':

        def tailhead(edge):
            if False:
                for i in range(10):
                    print('nop')
            if edge[-1] == 'reverse':
                return (edge[1], edge[0])
            return edge[:2]
    explored = set()
    cycle = []
    final_node = None
    for start_node in G.nbunch_iter(source):
        if start_node in explored:
            continue
        edges = []
        seen = {start_node}
        active_nodes = {start_node}
        previous_head = None
        for edge in nx.edge_dfs(G, start_node, orientation):
            (tail, head) = tailhead(edge)
            if head in explored:
                continue
            if previous_head is not None and tail != previous_head:
                while True:
                    try:
                        popped_edge = edges.pop()
                    except IndexError:
                        edges = []
                        active_nodes = {tail}
                        break
                    else:
                        popped_head = tailhead(popped_edge)[1]
                        active_nodes.remove(popped_head)
                    if edges:
                        last_head = tailhead(edges[-1])[1]
                        if tail == last_head:
                            break
            edges.append(edge)
            if head in active_nodes:
                cycle.extend(edges)
                final_node = head
                break
            else:
                seen.add(head)
                active_nodes.add(head)
                previous_head = head
        if cycle:
            break
        else:
            explored.update(seen)
    else:
        assert len(cycle) == 0
        raise nx.exception.NetworkXNoCycle('No cycle found.')
    for (i, edge) in enumerate(cycle):
        (tail, head) = tailhead(edge)
        if tail == final_node:
            break
    return cycle[i:]

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch(edge_attrs='weight')
def minimum_cycle_basis(G, weight=None):
    if False:
        print('Hello World!')
    'Returns a minimum weight cycle basis for G\n\n    Minimum weight means a cycle basis for which the total weight\n    (length for unweighted graphs) of all the cycles is minimum.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n    weight: string\n        name of the edge attribute to use for edge weights\n\n    Returns\n    -------\n    A list of cycle lists.  Each cycle list is a list of nodes\n    which forms a cycle (loop) in G. Note that the nodes are not\n    necessarily returned in a order by which they appear in the cycle\n\n    Examples\n    --------\n    >>> G = nx.Graph()\n    >>> nx.add_cycle(G, [0, 1, 2, 3])\n    >>> nx.add_cycle(G, [0, 3, 4, 5])\n    >>> nx.minimum_cycle_basis(G)\n    [[5, 4, 3, 0], [3, 2, 1, 0]]\n\n    References:\n        [1] Kavitha, Telikepalli, et al. "An O(m^2n) Algorithm for\n        Minimum Cycle Basis of Graphs."\n        http://link.springer.com/article/10.1007/s00453-007-9064-z\n        [2] de Pina, J. 1995. Applications of shortest path methods.\n        Ph.D. thesis, University of Amsterdam, Netherlands\n\n    See Also\n    --------\n    simple_cycles, cycle_basis\n    '
    return sum((_min_cycle_basis(G.subgraph(c), weight) for c in nx.connected_components(G)), [])

def _min_cycle_basis(G, weight):
    if False:
        for i in range(10):
            print('nop')
    cb = []
    tree_edges = list(nx.minimum_spanning_edges(G, weight=None, data=False))
    chords = G.edges - tree_edges - {(v, u) for (u, v) in tree_edges}
    set_orth = [{edge} for edge in chords]
    while set_orth:
        base = set_orth.pop()
        cycle_edges = _min_cycle(G, base, weight)
        cb.append([v for (u, v) in cycle_edges])
        set_orth = [{e for e in orth if e not in base if e[::-1] not in base} | {e for e in base if e not in orth if e[::-1] not in orth} if sum((e in orth or e[::-1] in orth for e in cycle_edges)) % 2 else orth for orth in set_orth]
    return cb

def _min_cycle(G, orth, weight):
    if False:
        print('Hello World!')
    "\n    Computes the minimum weight cycle in G,\n    orthogonal to the vector orth as per [p. 338, 1]\n    Use (u, 1) to indicate the lifted copy of u (denoted u' in paper).\n    "
    Gi = nx.Graph()
    for (u, v, wt) in G.edges(data=weight, default=1):
        if (u, v) in orth or (v, u) in orth:
            Gi.add_edges_from([(u, (v, 1)), ((u, 1), v)], Gi_weight=wt)
        else:
            Gi.add_edges_from([(u, v), ((u, 1), (v, 1))], Gi_weight=wt)
    spl = nx.shortest_path_length
    lift = {n: spl(Gi, source=n, target=(n, 1), weight='Gi_weight') for n in G}
    start = min(lift, key=lift.get)
    end = (start, 1)
    min_path_i = nx.shortest_path(Gi, source=start, target=end, weight='Gi_weight')
    min_path = [n if n in G else n[0] for n in min_path_i]
    edgelist = list(pairwise(min_path))
    edgeset = set()
    for e in edgelist:
        if e in edgeset:
            edgeset.remove(e)
        elif e[::-1] in edgeset:
            edgeset.remove(e[::-1])
        else:
            edgeset.add(e)
    min_edgelist = []
    for e in edgelist:
        if e in edgeset:
            min_edgelist.append(e)
            edgeset.remove(e)
        elif e[::-1] in edgeset:
            min_edgelist.append(e[::-1])
            edgeset.remove(e[::-1])
    return min_edgelist

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def girth(G):
    if False:
        for i in range(10):
            print('nop')
    'Returns the girth of the graph.\n\n    The girth of a graph is the length of its shortest cycle, or infinity if\n    the graph is acyclic. The algorithm follows the description given on the\n    Wikipedia page [1]_, and runs in time O(mn) on a graph with m edges and n\n    nodes.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n\n    Returns\n    -------\n    int or math.inf\n\n    Examples\n    --------\n    All examples below (except P_5) can easily be checked using Wikipedia,\n    which has a page for each of these famous graphs.\n\n    >>> nx.girth(nx.chvatal_graph())\n    4\n    >>> nx.girth(nx.tutte_graph())\n    4\n    >>> nx.girth(nx.petersen_graph())\n    5\n    >>> nx.girth(nx.heawood_graph())\n    6\n    >>> nx.girth(nx.pappus_graph())\n    6\n    >>> nx.girth(nx.path_graph(5))\n    inf\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Girth_(graph_theory)\n\n    '
    girth = depth_limit = inf
    tree_edge = nx.algorithms.traversal.breadth_first_search.TREE_EDGE
    level_edge = nx.algorithms.traversal.breadth_first_search.LEVEL_EDGE
    for n in G:
        depth = {n: 0}
        for (u, v, label) in nx.bfs_labeled_edges(G, n):
            du = depth[u]
            if du > depth_limit:
                break
            if label is tree_edge:
                depth[v] = du + 1
            else:
                delta = label is level_edge
                length = du + du + 2 - delta
                if length < girth:
                    girth = length
                    depth_limit = du - delta
    return girth