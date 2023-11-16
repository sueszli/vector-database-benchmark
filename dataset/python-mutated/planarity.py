from collections import defaultdict
import networkx as nx
__all__ = ['check_planarity', 'is_planar', 'PlanarEmbedding']

@nx._dispatch
def is_planar(G):
    if False:
        return 10
    'Returns True if and only if `G` is planar.\n\n    A graph is *planar* iff it can be drawn in a plane without\n    any edge intersections.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    Returns\n    -------\n    bool\n       Whether the graph is planar.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (0, 2)])\n    >>> nx.is_planar(G)\n    True\n    >>> nx.is_planar(nx.complete_graph(5))\n    False\n\n    See Also\n    --------\n    check_planarity :\n        Check if graph is planar *and* return a `PlanarEmbedding` instance if True.\n    '
    return check_planarity(G, counterexample=False)[0]

@nx._dispatch
def check_planarity(G, counterexample=False):
    if False:
        i = 10
        return i + 15
    'Check if a graph is planar and return a counterexample or an embedding.\n\n    A graph is planar iff it can be drawn in a plane without\n    any edge intersections.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n    counterexample : bool\n        A Kuratowski subgraph (to proof non planarity) is only returned if set\n        to true.\n\n    Returns\n    -------\n    (is_planar, certificate) : (bool, NetworkX graph) tuple\n        is_planar is true if the graph is planar.\n        If the graph is planar `certificate` is a PlanarEmbedding\n        otherwise it is a Kuratowski subgraph.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (0, 2)])\n    >>> is_planar, P = nx.check_planarity(G)\n    >>> print(is_planar)\n    True\n\n    When `G` is planar, a `PlanarEmbedding` instance is returned:\n\n    >>> P.get_data()\n    {0: [1, 2], 1: [0], 2: [0]}\n\n    Notes\n    -----\n    A (combinatorial) embedding consists of cyclic orderings of the incident\n    edges at each vertex. Given such an embedding there are multiple approaches\n    discussed in literature to drawing the graph (subject to various\n    constraints, e.g. integer coordinates), see e.g. [2].\n\n    The planarity check algorithm and extraction of the combinatorial embedding\n    is based on the Left-Right Planarity Test [1].\n\n    A counterexample is only generated if the corresponding parameter is set,\n    because the complexity of the counterexample generation is higher.\n\n    See also\n    --------\n    is_planar :\n        Check for planarity without creating a `PlanarEmbedding` or counterexample.\n\n    References\n    ----------\n    .. [1] Ulrik Brandes:\n        The Left-Right Planarity Test\n        2009\n        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.217.9208\n    .. [2] Takao Nishizeki, Md Saidur Rahman:\n        Planar graph drawing\n        Lecture Notes Series on Computing: Volume 12\n        2004\n    '
    planarity_state = LRPlanarity(G)
    embedding = planarity_state.lr_planarity()
    if embedding is None:
        if counterexample:
            return (False, get_counterexample(G))
        else:
            return (False, None)
    else:
        return (True, embedding)

@nx._dispatch
def check_planarity_recursive(G, counterexample=False):
    if False:
        print('Hello World!')
    'Recursive version of :meth:`check_planarity`.'
    planarity_state = LRPlanarity(G)
    embedding = planarity_state.lr_planarity_recursive()
    if embedding is None:
        if counterexample:
            return (False, get_counterexample_recursive(G))
        else:
            return (False, None)
    else:
        return (True, embedding)

@nx._dispatch
def get_counterexample(G):
    if False:
        while True:
            i = 10
    'Obtains a Kuratowski subgraph.\n\n    Raises nx.NetworkXException if G is planar.\n\n    The function removes edges such that the graph is still not planar.\n    At some point the removal of any edge would make the graph planar.\n    This subgraph must be a Kuratowski subgraph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    Returns\n    -------\n    subgraph : NetworkX graph\n        A Kuratowski subgraph that proves that G is not planar.\n\n    '
    G = nx.Graph(G)
    if check_planarity(G)[0]:
        raise nx.NetworkXException('G is planar - no counter example.')
    subgraph = nx.Graph()
    for u in G:
        nbrs = list(G[u])
        for v in nbrs:
            G.remove_edge(u, v)
            if check_planarity(G)[0]:
                G.add_edge(u, v)
                subgraph.add_edge(u, v)
    return subgraph

@nx._dispatch
def get_counterexample_recursive(G):
    if False:
        print('Hello World!')
    'Recursive version of :meth:`get_counterexample`.'
    G = nx.Graph(G)
    if check_planarity_recursive(G)[0]:
        raise nx.NetworkXException('G is planar - no counter example.')
    subgraph = nx.Graph()
    for u in G:
        nbrs = list(G[u])
        for v in nbrs:
            G.remove_edge(u, v)
            if check_planarity_recursive(G)[0]:
                G.add_edge(u, v)
                subgraph.add_edge(u, v)
    return subgraph

class Interval:
    """Represents a set of return edges.

    All return edges in an interval induce a same constraint on the contained
    edges, which means that all edges must either have a left orientation or
    all edges must have a right orientation.
    """

    def __init__(self, low=None, high=None):
        if False:
            for i in range(10):
                print('nop')
        self.low = low
        self.high = high

    def empty(self):
        if False:
            for i in range(10):
                print('nop')
        'Check if the interval is empty'
        return self.low is None and self.high is None

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a copy of this interval'
        return Interval(self.low, self.high)

    def conflicting(self, b, planarity_state):
        if False:
            return 10
        'Returns True if interval I conflicts with edge b'
        return not self.empty() and planarity_state.lowpt[self.high] > planarity_state.lowpt[b]

class ConflictPair:
    """Represents a different constraint between two intervals.

    The edges in the left interval must have a different orientation than
    the one in the right interval.
    """

    def __init__(self, left=Interval(), right=Interval()):
        if False:
            i = 10
            return i + 15
        self.left = left
        self.right = right

    def swap(self):
        if False:
            print('Hello World!')
        'Swap left and right intervals'
        temp = self.left
        self.left = self.right
        self.right = temp

    def lowest(self, planarity_state):
        if False:
            i = 10
            return i + 15
        'Returns the lowest lowpoint of a conflict pair'
        if self.left.empty():
            return planarity_state.lowpt[self.right.low]
        if self.right.empty():
            return planarity_state.lowpt[self.left.low]
        return min(planarity_state.lowpt[self.left.low], planarity_state.lowpt[self.right.low])

def top_of_stack(l):
    if False:
        while True:
            i = 10
    'Returns the element on top of the stack.'
    if not l:
        return None
    return l[-1]

class LRPlanarity:
    """A class to maintain the state during planarity check."""
    __slots__ = ['G', 'roots', 'height', 'lowpt', 'lowpt2', 'nesting_depth', 'parent_edge', 'DG', 'adjs', 'ordered_adjs', 'ref', 'side', 'S', 'stack_bottom', 'lowpt_edge', 'left_ref', 'right_ref', 'embedding']

    def __init__(self, G):
        if False:
            return 10
        self.G = nx.Graph()
        self.G.add_nodes_from(G.nodes)
        for e in G.edges:
            if e[0] != e[1]:
                self.G.add_edge(e[0], e[1])
        self.roots = []
        self.height = defaultdict(lambda : None)
        self.lowpt = {}
        self.lowpt2 = {}
        self.nesting_depth = {}
        self.parent_edge = defaultdict(lambda : None)
        self.DG = nx.DiGraph()
        self.DG.add_nodes_from(G.nodes)
        self.adjs = {}
        self.ordered_adjs = {}
        self.ref = defaultdict(lambda : None)
        self.side = defaultdict(lambda : 1)
        self.S = []
        self.stack_bottom = {}
        self.lowpt_edge = {}
        self.left_ref = {}
        self.right_ref = {}
        self.embedding = PlanarEmbedding()

    def lr_planarity(self):
        if False:
            print('Hello World!')
        'Execute the LR planarity test.\n\n        Returns\n        -------\n        embedding : dict\n            If the graph is planar an embedding is returned. Otherwise None.\n        '
        if self.G.order() > 2 and self.G.size() > 3 * self.G.order() - 6:
            return None
        for v in self.G:
            self.adjs[v] = list(self.G[v])
        for v in self.G:
            if self.height[v] is None:
                self.height[v] = 0
                self.roots.append(v)
                self.dfs_orientation(v)
        self.G = None
        self.lowpt2 = None
        self.adjs = None
        for v in self.DG:
            self.ordered_adjs[v] = sorted(self.DG[v], key=lambda x: self.nesting_depth[v, x])
        for v in self.roots:
            if not self.dfs_testing(v):
                return None
        self.height = None
        self.lowpt = None
        self.S = None
        self.stack_bottom = None
        self.lowpt_edge = None
        for e in self.DG.edges:
            self.nesting_depth[e] = self.sign(e) * self.nesting_depth[e]
        self.embedding.add_nodes_from(self.DG.nodes)
        for v in self.DG:
            self.ordered_adjs[v] = sorted(self.DG[v], key=lambda x: self.nesting_depth[v, x])
            previous_node = None
            for w in self.ordered_adjs[v]:
                self.embedding.add_half_edge_cw(v, w, previous_node)
                previous_node = w
        self.DG = None
        self.nesting_depth = None
        self.ref = None
        for v in self.roots:
            self.dfs_embedding(v)
        self.roots = None
        self.parent_edge = None
        self.ordered_adjs = None
        self.left_ref = None
        self.right_ref = None
        self.side = None
        return self.embedding

    def lr_planarity_recursive(self):
        if False:
            while True:
                i = 10
        'Recursive version of :meth:`lr_planarity`.'
        if self.G.order() > 2 and self.G.size() > 3 * self.G.order() - 6:
            return None
        for v in self.G:
            if self.height[v] is None:
                self.height[v] = 0
                self.roots.append(v)
                self.dfs_orientation_recursive(v)
        self.G = None
        for v in self.DG:
            self.ordered_adjs[v] = sorted(self.DG[v], key=lambda x: self.nesting_depth[v, x])
        for v in self.roots:
            if not self.dfs_testing_recursive(v):
                return None
        for e in self.DG.edges:
            self.nesting_depth[e] = self.sign_recursive(e) * self.nesting_depth[e]
        self.embedding.add_nodes_from(self.DG.nodes)
        for v in self.DG:
            self.ordered_adjs[v] = sorted(self.DG[v], key=lambda x: self.nesting_depth[v, x])
            previous_node = None
            for w in self.ordered_adjs[v]:
                self.embedding.add_half_edge_cw(v, w, previous_node)
                previous_node = w
        for v in self.roots:
            self.dfs_embedding_recursive(v)
        return self.embedding

    def dfs_orientation(self, v):
        if False:
            print('Hello World!')
        'Orient the graph by DFS, compute lowpoints and nesting order.'
        dfs_stack = [v]
        ind = defaultdict(lambda : 0)
        skip_init = defaultdict(lambda : False)
        while dfs_stack:
            v = dfs_stack.pop()
            e = self.parent_edge[v]
            for w in self.adjs[v][ind[v]:]:
                vw = (v, w)
                if not skip_init[vw]:
                    if (v, w) in self.DG.edges or (w, v) in self.DG.edges:
                        ind[v] += 1
                        continue
                    self.DG.add_edge(v, w)
                    self.lowpt[vw] = self.height[v]
                    self.lowpt2[vw] = self.height[v]
                    if self.height[w] is None:
                        self.parent_edge[w] = vw
                        self.height[w] = self.height[v] + 1
                        dfs_stack.append(v)
                        dfs_stack.append(w)
                        skip_init[vw] = True
                        break
                    else:
                        self.lowpt[vw] = self.height[w]
                self.nesting_depth[vw] = 2 * self.lowpt[vw]
                if self.lowpt2[vw] < self.height[v]:
                    self.nesting_depth[vw] += 1
                if e is not None:
                    if self.lowpt[vw] < self.lowpt[e]:
                        self.lowpt2[e] = min(self.lowpt[e], self.lowpt2[vw])
                        self.lowpt[e] = self.lowpt[vw]
                    elif self.lowpt[vw] > self.lowpt[e]:
                        self.lowpt2[e] = min(self.lowpt2[e], self.lowpt[vw])
                    else:
                        self.lowpt2[e] = min(self.lowpt2[e], self.lowpt2[vw])
                ind[v] += 1

    def dfs_orientation_recursive(self, v):
        if False:
            return 10
        'Recursive version of :meth:`dfs_orientation`.'
        e = self.parent_edge[v]
        for w in self.G[v]:
            if (v, w) in self.DG.edges or (w, v) in self.DG.edges:
                continue
            vw = (v, w)
            self.DG.add_edge(v, w)
            self.lowpt[vw] = self.height[v]
            self.lowpt2[vw] = self.height[v]
            if self.height[w] is None:
                self.parent_edge[w] = vw
                self.height[w] = self.height[v] + 1
                self.dfs_orientation_recursive(w)
            else:
                self.lowpt[vw] = self.height[w]
            self.nesting_depth[vw] = 2 * self.lowpt[vw]
            if self.lowpt2[vw] < self.height[v]:
                self.nesting_depth[vw] += 1
            if e is not None:
                if self.lowpt[vw] < self.lowpt[e]:
                    self.lowpt2[e] = min(self.lowpt[e], self.lowpt2[vw])
                    self.lowpt[e] = self.lowpt[vw]
                elif self.lowpt[vw] > self.lowpt[e]:
                    self.lowpt2[e] = min(self.lowpt2[e], self.lowpt[vw])
                else:
                    self.lowpt2[e] = min(self.lowpt2[e], self.lowpt2[vw])

    def dfs_testing(self, v):
        if False:
            return 10
        'Test for LR partition.'
        dfs_stack = [v]
        ind = defaultdict(lambda : 0)
        skip_init = defaultdict(lambda : False)
        while dfs_stack:
            v = dfs_stack.pop()
            e = self.parent_edge[v]
            skip_final = False
            for w in self.ordered_adjs[v][ind[v]:]:
                ei = (v, w)
                if not skip_init[ei]:
                    self.stack_bottom[ei] = top_of_stack(self.S)
                    if ei == self.parent_edge[w]:
                        dfs_stack.append(v)
                        dfs_stack.append(w)
                        skip_init[ei] = True
                        skip_final = True
                        break
                    else:
                        self.lowpt_edge[ei] = ei
                        self.S.append(ConflictPair(right=Interval(ei, ei)))
                if self.lowpt[ei] < self.height[v]:
                    if w == self.ordered_adjs[v][0]:
                        self.lowpt_edge[e] = self.lowpt_edge[ei]
                    elif not self.add_constraints(ei, e):
                        return False
                ind[v] += 1
            if not skip_final:
                if e is not None:
                    self.remove_back_edges(e)
        return True

    def dfs_testing_recursive(self, v):
        if False:
            i = 10
            return i + 15
        'Recursive version of :meth:`dfs_testing`.'
        e = self.parent_edge[v]
        for w in self.ordered_adjs[v]:
            ei = (v, w)
            self.stack_bottom[ei] = top_of_stack(self.S)
            if ei == self.parent_edge[w]:
                if not self.dfs_testing_recursive(w):
                    return False
            else:
                self.lowpt_edge[ei] = ei
                self.S.append(ConflictPair(right=Interval(ei, ei)))
            if self.lowpt[ei] < self.height[v]:
                if w == self.ordered_adjs[v][0]:
                    self.lowpt_edge[e] = self.lowpt_edge[ei]
                elif not self.add_constraints(ei, e):
                    return False
        if e is not None:
            self.remove_back_edges(e)
        return True

    def add_constraints(self, ei, e):
        if False:
            print('Hello World!')
        P = ConflictPair()
        while True:
            Q = self.S.pop()
            if not Q.left.empty():
                Q.swap()
            if not Q.left.empty():
                return False
            if self.lowpt[Q.right.low] > self.lowpt[e]:
                if P.right.empty():
                    P.right = Q.right.copy()
                else:
                    self.ref[P.right.low] = Q.right.high
                P.right.low = Q.right.low
            else:
                self.ref[Q.right.low] = self.lowpt_edge[e]
            if top_of_stack(self.S) == self.stack_bottom[ei]:
                break
        while top_of_stack(self.S).left.conflicting(ei, self) or top_of_stack(self.S).right.conflicting(ei, self):
            Q = self.S.pop()
            if Q.right.conflicting(ei, self):
                Q.swap()
            if Q.right.conflicting(ei, self):
                return False
            self.ref[P.right.low] = Q.right.high
            if Q.right.low is not None:
                P.right.low = Q.right.low
            if P.left.empty():
                P.left = Q.left.copy()
            else:
                self.ref[P.left.low] = Q.left.high
            P.left.low = Q.left.low
        if not (P.left.empty() and P.right.empty()):
            self.S.append(P)
        return True

    def remove_back_edges(self, e):
        if False:
            print('Hello World!')
        u = e[0]
        while self.S and top_of_stack(self.S).lowest(self) == self.height[u]:
            P = self.S.pop()
            if P.left.low is not None:
                self.side[P.left.low] = -1
        if self.S:
            P = self.S.pop()
            while P.left.high is not None and P.left.high[1] == u:
                P.left.high = self.ref[P.left.high]
            if P.left.high is None and P.left.low is not None:
                self.ref[P.left.low] = P.right.low
                self.side[P.left.low] = -1
                P.left.low = None
            while P.right.high is not None and P.right.high[1] == u:
                P.right.high = self.ref[P.right.high]
            if P.right.high is None and P.right.low is not None:
                self.ref[P.right.low] = P.left.low
                self.side[P.right.low] = -1
                P.right.low = None
            self.S.append(P)
        if self.lowpt[e] < self.height[u]:
            hl = top_of_stack(self.S).left.high
            hr = top_of_stack(self.S).right.high
            if hl is not None and (hr is None or self.lowpt[hl] > self.lowpt[hr]):
                self.ref[e] = hl
            else:
                self.ref[e] = hr

    def dfs_embedding(self, v):
        if False:
            print('Hello World!')
        'Completes the embedding.'
        dfs_stack = [v]
        ind = defaultdict(lambda : 0)
        while dfs_stack:
            v = dfs_stack.pop()
            for w in self.ordered_adjs[v][ind[v]:]:
                ind[v] += 1
                ei = (v, w)
                if ei == self.parent_edge[w]:
                    self.embedding.add_half_edge_first(w, v)
                    self.left_ref[v] = w
                    self.right_ref[v] = w
                    dfs_stack.append(v)
                    dfs_stack.append(w)
                    break
                elif self.side[ei] == 1:
                    self.embedding.add_half_edge_cw(w, v, self.right_ref[w])
                else:
                    self.embedding.add_half_edge_ccw(w, v, self.left_ref[w])
                    self.left_ref[w] = v

    def dfs_embedding_recursive(self, v):
        if False:
            return 10
        'Recursive version of :meth:`dfs_embedding`.'
        for w in self.ordered_adjs[v]:
            ei = (v, w)
            if ei == self.parent_edge[w]:
                self.embedding.add_half_edge_first(w, v)
                self.left_ref[v] = w
                self.right_ref[v] = w
                self.dfs_embedding_recursive(w)
            elif self.side[ei] == 1:
                self.embedding.add_half_edge_cw(w, v, self.right_ref[w])
            else:
                self.embedding.add_half_edge_ccw(w, v, self.left_ref[w])
                self.left_ref[w] = v

    def sign(self, e):
        if False:
            i = 10
            return i + 15
        'Resolve the relative side of an edge to the absolute side.'
        dfs_stack = [e]
        old_ref = defaultdict(lambda : None)
        while dfs_stack:
            e = dfs_stack.pop()
            if self.ref[e] is not None:
                dfs_stack.append(e)
                dfs_stack.append(self.ref[e])
                old_ref[e] = self.ref[e]
                self.ref[e] = None
            else:
                self.side[e] *= self.side[old_ref[e]]
        return self.side[e]

    def sign_recursive(self, e):
        if False:
            print('Hello World!')
        'Recursive version of :meth:`sign`.'
        if self.ref[e] is not None:
            self.side[e] = self.side[e] * self.sign_recursive(self.ref[e])
            self.ref[e] = None
        return self.side[e]

class PlanarEmbedding(nx.DiGraph):
    """Represents a planar graph with its planar embedding.

    The planar embedding is given by a `combinatorial embedding
    <https://en.wikipedia.org/wiki/Graph_embedding#Combinatorial_embedding>`_.

    .. note:: `check_planarity` is the preferred way to check if a graph is planar.

    **Neighbor ordering:**

    In comparison to a usual graph structure, the embedding also stores the
    order of all neighbors for every vertex.
    The order of the neighbors can be given in clockwise (cw) direction or
    counterclockwise (ccw) direction. This order is stored as edge attributes
    in the underlying directed graph. For the edge (u, v) the edge attribute
    'cw' is set to the neighbor of u that follows immediately after v in
    clockwise direction.

    In order for a PlanarEmbedding to be valid it must fulfill multiple
    conditions. It is possible to check if these conditions are fulfilled with
    the method :meth:`check_structure`.
    The conditions are:

    * Edges must go in both directions (because the edge attributes differ)
    * Every edge must have a 'cw' and 'ccw' attribute which corresponds to a
      correct planar embedding.
    * A node with non zero degree must have a node attribute 'first_nbr'.

    As long as a PlanarEmbedding is invalid only the following methods should
    be called:

    * :meth:`add_half_edge_ccw`
    * :meth:`add_half_edge_cw`
    * :meth:`connect_components`
    * :meth:`add_half_edge_first`

    Even though the graph is a subclass of nx.DiGraph, it can still be used
    for algorithms that require undirected graphs, because the method
    :meth:`is_directed` is overridden. This is possible, because a valid
    PlanarGraph must have edges in both directions.

    **Half edges:**

    In methods like `add_half_edge_ccw` the term "half-edge" is used, which is
    a term that is used in `doubly connected edge lists
    <https://en.wikipedia.org/wiki/Doubly_connected_edge_list>`_. It is used
    to emphasize that the edge is only in one direction and there exists
    another half-edge in the opposite direction.
    While conventional edges always have two faces (including outer face) next
    to them, it is possible to assign each half-edge *exactly one* face.
    For a half-edge (u, v) that is orientated such that u is below v then the
    face that belongs to (u, v) is to the right of this half-edge.

    See Also
    --------
    is_planar :
        Preferred way to check if an existing graph is planar.

    check_planarity :
        A convenient way to create a `PlanarEmbedding`. If not planar,
        it returns a subgraph that shows this.

    Examples
    --------

    Create an embedding of a star graph (compare `nx.star_graph(3)`):

    >>> G = nx.PlanarEmbedding()
    >>> G.add_half_edge_cw(0, 1, None)
    >>> G.add_half_edge_cw(0, 2, 1)
    >>> G.add_half_edge_cw(0, 3, 2)
    >>> G.add_half_edge_cw(1, 0, None)
    >>> G.add_half_edge_cw(2, 0, None)
    >>> G.add_half_edge_cw(3, 0, None)

    Alternatively the same embedding can also be defined in counterclockwise
    orientation. The following results in exactly the same PlanarEmbedding:

    >>> G = nx.PlanarEmbedding()
    >>> G.add_half_edge_ccw(0, 1, None)
    >>> G.add_half_edge_ccw(0, 3, 1)
    >>> G.add_half_edge_ccw(0, 2, 3)
    >>> G.add_half_edge_ccw(1, 0, None)
    >>> G.add_half_edge_ccw(2, 0, None)
    >>> G.add_half_edge_ccw(3, 0, None)

    After creating a graph, it is possible to validate that the PlanarEmbedding
    object is correct:

    >>> G.check_structure()

    """

    def get_data(self):
        if False:
            return 10
        'Converts the adjacency structure into a better readable structure.\n\n        Returns\n        -------\n        embedding : dict\n            A dict mapping all nodes to a list of neighbors sorted in\n            clockwise order.\n\n        See Also\n        --------\n        set_data\n\n        '
        embedding = {}
        for v in self:
            embedding[v] = list(self.neighbors_cw_order(v))
        return embedding

    def set_data(self, data):
        if False:
            print('Hello World!')
        'Inserts edges according to given sorted neighbor list.\n\n        The input format is the same as the output format of get_data().\n\n        Parameters\n        ----------\n        data : dict\n            A dict mapping all nodes to a list of neighbors sorted in\n            clockwise order.\n\n        See Also\n        --------\n        get_data\n\n        '
        for v in data:
            for w in reversed(data[v]):
                self.add_half_edge_first(v, w)

    def neighbors_cw_order(self, v):
        if False:
            for i in range(10):
                print('nop')
        'Generator for the neighbors of v in clockwise order.\n\n        Parameters\n        ----------\n        v : node\n\n        Yields\n        ------\n        node\n\n        '
        if len(self[v]) == 0:
            return
        start_node = self.nodes[v]['first_nbr']
        yield start_node
        current_node = self[v][start_node]['cw']
        while start_node != current_node:
            yield current_node
            current_node = self[v][current_node]['cw']

    def check_structure(self):
        if False:
            i = 10
            return i + 15
        "Runs without exceptions if this object is valid.\n\n        Checks that the following properties are fulfilled:\n\n        * Edges go in both directions (because the edge attributes differ).\n        * Every edge has a 'cw' and 'ccw' attribute which corresponds to a\n          correct planar embedding.\n        * A node with a degree larger than 0 has a node attribute 'first_nbr'.\n\n        Running this method verifies that the underlying Graph must be planar.\n\n        Raises\n        ------\n        NetworkXException\n            This exception is raised with a short explanation if the\n            PlanarEmbedding is invalid.\n        "
        for v in self:
            try:
                sorted_nbrs = set(self.neighbors_cw_order(v))
            except KeyError as err:
                msg = f'Bad embedding. Missing orientation for a neighbor of {v}'
                raise nx.NetworkXException(msg) from err
            unsorted_nbrs = set(self[v])
            if sorted_nbrs != unsorted_nbrs:
                msg = 'Bad embedding. Edge orientations not set correctly.'
                raise nx.NetworkXException(msg)
            for w in self[v]:
                if not self.has_edge(w, v):
                    msg = 'Bad embedding. Opposite half-edge is missing.'
                    raise nx.NetworkXException(msg)
        counted_half_edges = set()
        for component in nx.connected_components(self):
            if len(component) == 1:
                continue
            num_nodes = len(component)
            num_half_edges = 0
            num_faces = 0
            for v in component:
                for w in self.neighbors_cw_order(v):
                    num_half_edges += 1
                    if (v, w) not in counted_half_edges:
                        num_faces += 1
                        self.traverse_face(v, w, counted_half_edges)
            num_edges = num_half_edges // 2
            if num_nodes - num_edges + num_faces != 2:
                msg = "Bad embedding. The graph does not match Euler's formula"
                raise nx.NetworkXException(msg)

    def add_half_edge_ccw(self, start_node, end_node, reference_neighbor):
        if False:
            i = 10
            return i + 15
        'Adds a half-edge from start_node to end_node.\n\n        The half-edge is added counter clockwise next to the existing half-edge\n        (start_node, reference_neighbor).\n\n        Parameters\n        ----------\n        start_node : node\n            Start node of inserted edge.\n        end_node : node\n            End node of inserted edge.\n        reference_neighbor: node\n            End node of reference edge.\n\n        Raises\n        ------\n        NetworkXException\n            If the reference_neighbor does not exist.\n\n        See Also\n        --------\n        add_half_edge_cw\n        connect_components\n        add_half_edge_first\n\n        '
        if reference_neighbor is None:
            self.add_edge(start_node, end_node)
            self[start_node][end_node]['cw'] = end_node
            self[start_node][end_node]['ccw'] = end_node
            self.nodes[start_node]['first_nbr'] = end_node
        else:
            ccw_reference = self[start_node][reference_neighbor]['ccw']
            self.add_half_edge_cw(start_node, end_node, ccw_reference)
            if reference_neighbor == self.nodes[start_node].get('first_nbr', None):
                self.nodes[start_node]['first_nbr'] = end_node

    def add_half_edge_cw(self, start_node, end_node, reference_neighbor):
        if False:
            i = 10
            return i + 15
        'Adds a half-edge from start_node to end_node.\n\n        The half-edge is added clockwise next to the existing half-edge\n        (start_node, reference_neighbor).\n\n        Parameters\n        ----------\n        start_node : node\n            Start node of inserted edge.\n        end_node : node\n            End node of inserted edge.\n        reference_neighbor: node\n            End node of reference edge.\n\n        Raises\n        ------\n        NetworkXException\n            If the reference_neighbor does not exist.\n\n        See Also\n        --------\n        add_half_edge_ccw\n        connect_components\n        add_half_edge_first\n        '
        self.add_edge(start_node, end_node)
        if reference_neighbor is None:
            self[start_node][end_node]['cw'] = end_node
            self[start_node][end_node]['ccw'] = end_node
            self.nodes[start_node]['first_nbr'] = end_node
            return
        if reference_neighbor not in self[start_node]:
            raise nx.NetworkXException('Cannot add edge. Reference neighbor does not exist')
        cw_reference = self[start_node][reference_neighbor]['cw']
        self[start_node][reference_neighbor]['cw'] = end_node
        self[start_node][end_node]['cw'] = cw_reference
        self[start_node][cw_reference]['ccw'] = end_node
        self[start_node][end_node]['ccw'] = reference_neighbor

    def connect_components(self, v, w):
        if False:
            print('Hello World!')
        'Adds half-edges for (v, w) and (w, v) at some position.\n\n        This method should only be called if v and w are in different\n        components, or it might break the embedding.\n        This especially means that if `connect_components(v, w)`\n        is called it is not allowed to call `connect_components(w, v)`\n        afterwards. The neighbor orientations in both directions are\n        all set correctly after the first call.\n\n        Parameters\n        ----------\n        v : node\n        w : node\n\n        See Also\n        --------\n        add_half_edge_ccw\n        add_half_edge_cw\n        add_half_edge_first\n        '
        self.add_half_edge_first(v, w)
        self.add_half_edge_first(w, v)

    def add_half_edge_first(self, start_node, end_node):
        if False:
            i = 10
            return i + 15
        'The added half-edge is inserted at the first position in the order.\n\n        Parameters\n        ----------\n        start_node : node\n        end_node : node\n\n        See Also\n        --------\n        add_half_edge_ccw\n        add_half_edge_cw\n        connect_components\n        '
        if start_node in self and 'first_nbr' in self.nodes[start_node]:
            reference = self.nodes[start_node]['first_nbr']
        else:
            reference = None
        self.add_half_edge_ccw(start_node, end_node, reference)

    def next_face_half_edge(self, v, w):
        if False:
            print('Hello World!')
        'Returns the following half-edge left of a face.\n\n        Parameters\n        ----------\n        v : node\n        w : node\n\n        Returns\n        -------\n        half-edge : tuple\n        '
        new_node = self[w][v]['ccw']
        return (w, new_node)

    def traverse_face(self, v, w, mark_half_edges=None):
        if False:
            return 10
        'Returns nodes on the face that belong to the half-edge (v, w).\n\n        The face that is traversed lies to the right of the half-edge (in an\n        orientation where v is below w).\n\n        Optionally it is possible to pass a set to which all encountered half\n        edges are added. Before calling this method, this set must not include\n        any half-edges that belong to the face.\n\n        Parameters\n        ----------\n        v : node\n            Start node of half-edge.\n        w : node\n            End node of half-edge.\n        mark_half_edges: set, optional\n            Set to which all encountered half-edges are added.\n\n        Returns\n        -------\n        face : list\n            A list of nodes that lie on this face.\n        '
        if mark_half_edges is None:
            mark_half_edges = set()
        face_nodes = [v]
        mark_half_edges.add((v, w))
        prev_node = v
        cur_node = w
        incoming_node = self[v][w]['cw']
        while cur_node != v or prev_node != incoming_node:
            face_nodes.append(cur_node)
            (prev_node, cur_node) = self.next_face_half_edge(prev_node, cur_node)
            if (prev_node, cur_node) in mark_half_edges:
                raise nx.NetworkXException('Bad planar embedding. Impossible face.')
            mark_half_edges.add((prev_node, cur_node))
        return face_nodes

    def is_directed(self):
        if False:
            while True:
                i = 10
        'A valid PlanarEmbedding is undirected.\n\n        All reverse edges are contained, i.e. for every existing\n        half-edge (v, w) the half-edge in the opposite direction (w, v) is also\n        contained.\n        '
        return False