"""
ISMAGS Algorithm
================

Provides a Python implementation of the ISMAGS algorithm. [1]_

It is capable of finding (subgraph) isomorphisms between two graphs, taking the
symmetry of the subgraph into account. In most cases the VF2 algorithm is
faster (at least on small graphs) than this implementation, but in some cases
there is an exponential number of isomorphisms that are symmetrically
equivalent. In that case, the ISMAGS algorithm will provide only one solution
per symmetry group.

>>> petersen = nx.petersen_graph()
>>> ismags = nx.isomorphism.ISMAGS(petersen, petersen)
>>> isomorphisms = list(ismags.isomorphisms_iter(symmetry=False))
>>> len(isomorphisms)
120
>>> isomorphisms = list(ismags.isomorphisms_iter(symmetry=True))
>>> answer = [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}]
>>> answer == isomorphisms
True

In addition, this implementation also provides an interface to find the
largest common induced subgraph [2]_ between any two graphs, again taking
symmetry into account. Given `graph` and `subgraph` the algorithm will remove
nodes from the `subgraph` until `subgraph` is isomorphic to a subgraph of
`graph`. Since only the symmetry of `subgraph` is taken into account it is
worth thinking about how you provide your graphs:

>>> graph1 = nx.path_graph(4)
>>> graph2 = nx.star_graph(3)
>>> ismags = nx.isomorphism.ISMAGS(graph1, graph2)
>>> ismags.is_isomorphic()
False
>>> largest_common_subgraph = list(ismags.largest_common_subgraph())
>>> answer = [{1: 0, 0: 1, 2: 2}, {2: 0, 1: 1, 3: 2}]
>>> answer == largest_common_subgraph
True
>>> ismags2 = nx.isomorphism.ISMAGS(graph2, graph1)
>>> largest_common_subgraph = list(ismags2.largest_common_subgraph())
>>> answer = [
...     {1: 0, 0: 1, 2: 2},
...     {1: 0, 0: 1, 3: 2},
...     {2: 0, 0: 1, 1: 2},
...     {2: 0, 0: 1, 3: 2},
...     {3: 0, 0: 1, 1: 2},
...     {3: 0, 0: 1, 2: 2},
... ]
>>> answer == largest_common_subgraph
True

However, when not taking symmetry into account, it doesn't matter:

>>> largest_common_subgraph = list(ismags.largest_common_subgraph(symmetry=False))
>>> answer = [
...     {1: 0, 0: 1, 2: 2},
...     {1: 0, 2: 1, 0: 2},
...     {2: 0, 1: 1, 3: 2},
...     {2: 0, 3: 1, 1: 2},
...     {1: 0, 0: 1, 2: 3},
...     {1: 0, 2: 1, 0: 3},
...     {2: 0, 1: 1, 3: 3},
...     {2: 0, 3: 1, 1: 3},
...     {1: 0, 0: 2, 2: 3},
...     {1: 0, 2: 2, 0: 3},
...     {2: 0, 1: 2, 3: 3},
...     {2: 0, 3: 2, 1: 3},
... ]
>>> answer == largest_common_subgraph
True
>>> largest_common_subgraph = list(ismags2.largest_common_subgraph(symmetry=False))
>>> answer = [
...     {1: 0, 0: 1, 2: 2},
...     {1: 0, 0: 1, 3: 2},
...     {2: 0, 0: 1, 1: 2},
...     {2: 0, 0: 1, 3: 2},
...     {3: 0, 0: 1, 1: 2},
...     {3: 0, 0: 1, 2: 2},
...     {1: 1, 0: 2, 2: 3},
...     {1: 1, 0: 2, 3: 3},
...     {2: 1, 0: 2, 1: 3},
...     {2: 1, 0: 2, 3: 3},
...     {3: 1, 0: 2, 1: 3},
...     {3: 1, 0: 2, 2: 3},
... ]
>>> answer == largest_common_subgraph
True

Notes
-----
- The current implementation works for undirected graphs only. The algorithm
  in general should work for directed graphs as well though.
- Node keys for both provided graphs need to be fully orderable as well as
  hashable.
- Node and edge equality is assumed to be transitive: if A is equal to B, and
  B is equal to C, then A is equal to C.

References
----------
.. [1] M. Houbraken, S. Demeyer, T. Michoel, P. Audenaert, D. Colle,
   M. Pickavet, "The Index-Based Subgraph Matching Algorithm with General
   Symmetries (ISMAGS): Exploiting Symmetry for Faster Subgraph
   Enumeration", PLoS One 9(5): e97896, 2014.
   https://doi.org/10.1371/journal.pone.0097896
.. [2] https://en.wikipedia.org/wiki/Maximum_common_induced_subgraph
"""
__all__ = ['ISMAGS']
import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps

def are_all_equal(iterable):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns ``True`` if and only if all elements in `iterable` are equal; and\n    ``False`` otherwise.\n\n    Parameters\n    ----------\n    iterable: collections.abc.Iterable\n        The container whose elements will be checked.\n\n    Returns\n    -------\n    bool\n        ``True`` iff all elements in `iterable` compare equal, ``False``\n        otherwise.\n    '
    try:
        shape = iterable.shape
    except AttributeError:
        pass
    else:
        if len(shape) > 1:
            message = 'The function does not works on multidimensional arrays.'
            raise NotImplementedError(message) from None
    iterator = iter(iterable)
    first = next(iterator, None)
    return all((item == first for item in iterator))

def make_partitions(items, test):
    if False:
        while True:
            i = 10
    '\n    Partitions items into sets based on the outcome of ``test(item1, item2)``.\n    Pairs of items for which `test` returns `True` end up in the same set.\n\n    Parameters\n    ----------\n    items : collections.abc.Iterable[collections.abc.Hashable]\n        Items to partition\n    test : collections.abc.Callable[collections.abc.Hashable, collections.abc.Hashable]\n        A function that will be called with 2 arguments, taken from items.\n        Should return `True` if those 2 items need to end up in the same\n        partition, and `False` otherwise.\n\n    Returns\n    -------\n    list[set]\n        A list of sets, with each set containing part of the items in `items`,\n        such that ``all(test(*pair) for pair in  itertools.combinations(set, 2))\n        == True``\n\n    Notes\n    -----\n    The function `test` is assumed to be transitive: if ``test(a, b)`` and\n    ``test(b, c)`` return ``True``, then ``test(a, c)`` must also be ``True``.\n    '
    partitions = []
    for item in items:
        for partition in partitions:
            p_item = next(iter(partition))
            if test(item, p_item):
                partition.add(item)
                break
        else:
            partitions.append({item})
    return partitions

def partition_to_color(partitions):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a dictionary that maps each item in each partition to the index of\n    the partition to which it belongs.\n\n    Parameters\n    ----------\n    partitions: collections.abc.Sequence[collections.abc.Iterable]\n        As returned by :func:`make_partitions`.\n\n    Returns\n    -------\n    dict\n    '
    colors = {}
    for (color, keys) in enumerate(partitions):
        for key in keys:
            colors[key] = color
    return colors

def intersect(collection_of_sets):
    if False:
        return 10
    '\n    Given an collection of sets, returns the intersection of those sets.\n\n    Parameters\n    ----------\n    collection_of_sets: collections.abc.Collection[set]\n        A collection of sets.\n\n    Returns\n    -------\n    set\n        An intersection of all sets in `collection_of_sets`. Will have the same\n        type as the item initially taken from `collection_of_sets`.\n    '
    collection_of_sets = list(collection_of_sets)
    first = collection_of_sets.pop()
    out = reduce(set.intersection, collection_of_sets, set(first))
    return type(first)(out)

class ISMAGS:
    """
    Implements the ISMAGS subgraph matching algorithm. [1]_ ISMAGS stands for
    "Index-based Subgraph Matching Algorithm with General Symmetries". As the
    name implies, it is symmetry aware and will only generate non-symmetric
    isomorphisms.

    Notes
    -----
    The implementation imposes additional conditions compared to the VF2
    algorithm on the graphs provided and the comparison functions
    (:attr:`node_equality` and :attr:`edge_equality`):

     - Node keys in both graphs must be orderable as well as hashable.
     - Equality must be transitive: if A is equal to B, and B is equal to C,
       then A must be equal to C.

    Attributes
    ----------
    graph: networkx.Graph
    subgraph: networkx.Graph
    node_equality: collections.abc.Callable
        The function called to see if two nodes should be considered equal.
        It's signature looks like this:
        ``f(graph1: networkx.Graph, node1, graph2: networkx.Graph, node2) -> bool``.
        `node1` is a node in `graph1`, and `node2` a node in `graph2`.
        Constructed from the argument `node_match`.
    edge_equality: collections.abc.Callable
        The function called to see if two edges should be considered equal.
        It's signature looks like this:
        ``f(graph1: networkx.Graph, edge1, graph2: networkx.Graph, edge2) -> bool``.
        `edge1` is an edge in `graph1`, and `edge2` an edge in `graph2`.
        Constructed from the argument `edge_match`.

    References
    ----------
    .. [1] M. Houbraken, S. Demeyer, T. Michoel, P. Audenaert, D. Colle,
       M. Pickavet, "The Index-Based Subgraph Matching Algorithm with General
       Symmetries (ISMAGS): Exploiting Symmetry for Faster Subgraph
       Enumeration", PLoS One 9(5): e97896, 2014.
       https://doi.org/10.1371/journal.pone.0097896
    """

    def __init__(self, graph, subgraph, node_match=None, edge_match=None, cache=None):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        graph: networkx.Graph\n        subgraph: networkx.Graph\n        node_match: collections.abc.Callable or None\n            Function used to determine whether two nodes are equivalent. Its\n            signature should look like ``f(n1: dict, n2: dict) -> bool``, with\n            `n1` and `n2` node property dicts. See also\n            :func:`~networkx.algorithms.isomorphism.categorical_node_match` and\n            friends.\n            If `None`, all nodes are considered equal.\n        edge_match: collections.abc.Callable or None\n            Function used to determine whether two edges are equivalent. Its\n            signature should look like ``f(e1: dict, e2: dict) -> bool``, with\n            `e1` and `e2` edge property dicts. See also\n            :func:`~networkx.algorithms.isomorphism.categorical_edge_match` and\n            friends.\n            If `None`, all edges are considered equal.\n        cache: collections.abc.Mapping\n            A cache used for caching graph symmetries.\n        '
        self.graph = graph
        self.subgraph = subgraph
        self._symmetry_cache = cache
        self._sgn_partitions_ = None
        self._sge_partitions_ = None
        self._sgn_colors_ = None
        self._sge_colors_ = None
        self._gn_partitions_ = None
        self._ge_partitions_ = None
        self._gn_colors_ = None
        self._ge_colors_ = None
        self._node_compat_ = None
        self._edge_compat_ = None
        if node_match is None:
            self.node_equality = self._node_match_maker(lambda n1, n2: True)
            self._sgn_partitions_ = [set(self.subgraph.nodes)]
            self._gn_partitions_ = [set(self.graph.nodes)]
            self._node_compat_ = {0: 0}
        else:
            self.node_equality = self._node_match_maker(node_match)
        if edge_match is None:
            self.edge_equality = self._edge_match_maker(lambda e1, e2: True)
            self._sge_partitions_ = [set(self.subgraph.edges)]
            self._ge_partitions_ = [set(self.graph.edges)]
            self._edge_compat_ = {0: 0}
        else:
            self.edge_equality = self._edge_match_maker(edge_match)

    @property
    def _sgn_partitions(self):
        if False:
            return 10
        if self._sgn_partitions_ is None:

            def nodematch(node1, node2):
                if False:
                    while True:
                        i = 10
                return self.node_equality(self.subgraph, node1, self.subgraph, node2)
            self._sgn_partitions_ = make_partitions(self.subgraph.nodes, nodematch)
        return self._sgn_partitions_

    @property
    def _sge_partitions(self):
        if False:
            return 10
        if self._sge_partitions_ is None:

            def edgematch(edge1, edge2):
                if False:
                    return 10
                return self.edge_equality(self.subgraph, edge1, self.subgraph, edge2)
            self._sge_partitions_ = make_partitions(self.subgraph.edges, edgematch)
        return self._sge_partitions_

    @property
    def _gn_partitions(self):
        if False:
            return 10
        if self._gn_partitions_ is None:

            def nodematch(node1, node2):
                if False:
                    i = 10
                    return i + 15
                return self.node_equality(self.graph, node1, self.graph, node2)
            self._gn_partitions_ = make_partitions(self.graph.nodes, nodematch)
        return self._gn_partitions_

    @property
    def _ge_partitions(self):
        if False:
            while True:
                i = 10
        if self._ge_partitions_ is None:

            def edgematch(edge1, edge2):
                if False:
                    print('Hello World!')
                return self.edge_equality(self.graph, edge1, self.graph, edge2)
            self._ge_partitions_ = make_partitions(self.graph.edges, edgematch)
        return self._ge_partitions_

    @property
    def _sgn_colors(self):
        if False:
            i = 10
            return i + 15
        if self._sgn_colors_ is None:
            self._sgn_colors_ = partition_to_color(self._sgn_partitions)
        return self._sgn_colors_

    @property
    def _sge_colors(self):
        if False:
            print('Hello World!')
        if self._sge_colors_ is None:
            self._sge_colors_ = partition_to_color(self._sge_partitions)
        return self._sge_colors_

    @property
    def _gn_colors(self):
        if False:
            print('Hello World!')
        if self._gn_colors_ is None:
            self._gn_colors_ = partition_to_color(self._gn_partitions)
        return self._gn_colors_

    @property
    def _ge_colors(self):
        if False:
            while True:
                i = 10
        if self._ge_colors_ is None:
            self._ge_colors_ = partition_to_color(self._ge_partitions)
        return self._ge_colors_

    @property
    def _node_compatibility(self):
        if False:
            return 10
        if self._node_compat_ is not None:
            return self._node_compat_
        self._node_compat_ = {}
        for (sgn_part_color, gn_part_color) in itertools.product(range(len(self._sgn_partitions)), range(len(self._gn_partitions))):
            sgn = next(iter(self._sgn_partitions[sgn_part_color]))
            gn = next(iter(self._gn_partitions[gn_part_color]))
            if self.node_equality(self.subgraph, sgn, self.graph, gn):
                self._node_compat_[sgn_part_color] = gn_part_color
        return self._node_compat_

    @property
    def _edge_compatibility(self):
        if False:
            i = 10
            return i + 15
        if self._edge_compat_ is not None:
            return self._edge_compat_
        self._edge_compat_ = {}
        for (sge_part_color, ge_part_color) in itertools.product(range(len(self._sge_partitions)), range(len(self._ge_partitions))):
            sge = next(iter(self._sge_partitions[sge_part_color]))
            ge = next(iter(self._ge_partitions[ge_part_color]))
            if self.edge_equality(self.subgraph, sge, self.graph, ge):
                self._edge_compat_[sge_part_color] = ge_part_color
        return self._edge_compat_

    @staticmethod
    def _node_match_maker(cmp):
        if False:
            return 10

        @wraps(cmp)
        def comparer(graph1, node1, graph2, node2):
            if False:
                while True:
                    i = 10
            return cmp(graph1.nodes[node1], graph2.nodes[node2])
        return comparer

    @staticmethod
    def _edge_match_maker(cmp):
        if False:
            while True:
                i = 10

        @wraps(cmp)
        def comparer(graph1, edge1, graph2, edge2):
            if False:
                for i in range(10):
                    print('nop')
            return cmp(graph1.edges[edge1], graph2.edges[edge2])
        return comparer

    def find_isomorphisms(self, symmetry=True):
        if False:
            i = 10
            return i + 15
        'Find all subgraph isomorphisms between subgraph and graph\n\n        Finds isomorphisms where :attr:`subgraph` <= :attr:`graph`.\n\n        Parameters\n        ----------\n        symmetry: bool\n            Whether symmetry should be taken into account. If False, found\n            isomorphisms may be symmetrically equivalent.\n\n        Yields\n        ------\n        dict\n            The found isomorphism mappings of {graph_node: subgraph_node}.\n        '
        if not self.subgraph:
            yield {}
            return
        elif not self.graph:
            return
        elif len(self.graph) < len(self.subgraph):
            return
        if symmetry:
            (_, cosets) = self.analyze_symmetry(self.subgraph, self._sgn_partitions, self._sge_colors)
            constraints = self._make_constraints(cosets)
        else:
            constraints = []
        candidates = self._find_nodecolor_candidates()
        la_candidates = self._get_lookahead_candidates()
        for sgn in self.subgraph:
            extra_candidates = la_candidates[sgn]
            if extra_candidates:
                candidates[sgn] = candidates[sgn] | {frozenset(extra_candidates)}
        if any(candidates.values()):
            start_sgn = min(candidates, key=lambda n: min(candidates[n], key=len))
            candidates[start_sgn] = (intersect(candidates[start_sgn]),)
            yield from self._map_nodes(start_sgn, candidates, constraints)
        else:
            return

    @staticmethod
    def _find_neighbor_color_count(graph, node, node_color, edge_color):
        if False:
            print('Hello World!')
        '\n        For `node` in `graph`, count the number of edges of a specific color\n        it has to nodes of a specific color.\n        '
        counts = Counter()
        neighbors = graph[node]
        for neighbor in neighbors:
            n_color = node_color[neighbor]
            if (node, neighbor) in edge_color:
                e_color = edge_color[node, neighbor]
            else:
                e_color = edge_color[neighbor, node]
            counts[e_color, n_color] += 1
        return counts

    def _get_lookahead_candidates(self):
        if False:
            while True:
                i = 10
        '\n        Returns a mapping of {subgraph node: collection of graph nodes} for\n        which the graph nodes are feasible candidates for the subgraph node, as\n        determined by looking ahead one edge.\n        '
        g_counts = {}
        for gn in self.graph:
            g_counts[gn] = self._find_neighbor_color_count(self.graph, gn, self._gn_colors, self._ge_colors)
        candidates = defaultdict(set)
        for sgn in self.subgraph:
            sg_count = self._find_neighbor_color_count(self.subgraph, sgn, self._sgn_colors, self._sge_colors)
            new_sg_count = Counter()
            for ((sge_color, sgn_color), count) in sg_count.items():
                try:
                    ge_color = self._edge_compatibility[sge_color]
                    gn_color = self._node_compatibility[sgn_color]
                except KeyError:
                    pass
                else:
                    new_sg_count[ge_color, gn_color] = count
            for (gn, g_count) in g_counts.items():
                if all((new_sg_count[x] <= g_count[x] for x in new_sg_count)):
                    candidates[sgn].add(gn)
        return candidates

    def largest_common_subgraph(self, symmetry=True):
        if False:
            i = 10
            return i + 15
        '\n        Find the largest common induced subgraphs between :attr:`subgraph` and\n        :attr:`graph`.\n\n        Parameters\n        ----------\n        symmetry: bool\n            Whether symmetry should be taken into account. If False, found\n            largest common subgraphs may be symmetrically equivalent.\n\n        Yields\n        ------\n        dict\n            The found isomorphism mappings of {graph_node: subgraph_node}.\n        '
        if not self.subgraph:
            yield {}
            return
        elif not self.graph:
            return
        if symmetry:
            (_, cosets) = self.analyze_symmetry(self.subgraph, self._sgn_partitions, self._sge_colors)
            constraints = self._make_constraints(cosets)
        else:
            constraints = []
        candidates = self._find_nodecolor_candidates()
        if any(candidates.values()):
            yield from self._largest_common_subgraph(candidates, constraints)
        else:
            return

    def analyze_symmetry(self, graph, node_partitions, edge_colors):
        if False:
            print('Hello World!')
        '\n        Find a minimal set of permutations and corresponding co-sets that\n        describe the symmetry of `graph`, given the node and edge equalities\n        given by `node_partitions` and `edge_colors`, respectively.\n\n        Parameters\n        ----------\n        graph : networkx.Graph\n            The graph whose symmetry should be analyzed.\n        node_partitions : list of sets\n            A list of sets containing node keys. Node keys in the same set\n            are considered equivalent. Every node key in `graph` should be in\n            exactly one of the sets. If all nodes are equivalent, this should\n            be ``[set(graph.nodes)]``.\n        edge_colors : dict mapping edges to their colors\n            A dict mapping every edge in `graph` to its corresponding color.\n            Edges with the same color are considered equivalent. If all edges\n            are equivalent, this should be ``{e: 0 for e in graph.edges}``.\n\n\n        Returns\n        -------\n        set[frozenset]\n            The found permutations. This is a set of frozensets of pairs of node\n            keys which can be exchanged without changing :attr:`subgraph`.\n        dict[collections.abc.Hashable, set[collections.abc.Hashable]]\n            The found co-sets. The co-sets is a dictionary of\n            ``{node key: set of node keys}``.\n            Every key-value pair describes which ``values`` can be interchanged\n            without changing nodes less than ``key``.\n        '
        if self._symmetry_cache is not None:
            key = hash((tuple(graph.nodes), tuple(graph.edges), tuple(map(tuple, node_partitions)), tuple(edge_colors.items())))
            if key in self._symmetry_cache:
                return self._symmetry_cache[key]
        node_partitions = list(self._refine_node_partitions(graph, node_partitions, edge_colors))
        assert len(node_partitions) == 1
        node_partitions = node_partitions[0]
        (permutations, cosets) = self._process_ordered_pair_partitions(graph, node_partitions, node_partitions, edge_colors)
        if self._symmetry_cache is not None:
            self._symmetry_cache[key] = (permutations, cosets)
        return (permutations, cosets)

    def is_isomorphic(self, symmetry=False):
        if False:
            i = 10
            return i + 15
        '\n        Returns True if :attr:`graph` is isomorphic to :attr:`subgraph` and\n        False otherwise.\n\n        Returns\n        -------\n        bool\n        '
        return len(self.subgraph) == len(self.graph) and self.subgraph_is_isomorphic(symmetry)

    def subgraph_is_isomorphic(self, symmetry=False):
        if False:
            i = 10
            return i + 15
        '\n        Returns True if a subgraph of :attr:`graph` is isomorphic to\n        :attr:`subgraph` and False otherwise.\n\n        Returns\n        -------\n        bool\n        '
        isom = next(self.subgraph_isomorphisms_iter(symmetry=symmetry), None)
        return isom is not None

    def isomorphisms_iter(self, symmetry=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Does the same as :meth:`find_isomorphisms` if :attr:`graph` and\n        :attr:`subgraph` have the same number of nodes.\n        '
        if len(self.graph) == len(self.subgraph):
            yield from self.subgraph_isomorphisms_iter(symmetry=symmetry)

    def subgraph_isomorphisms_iter(self, symmetry=True):
        if False:
            print('Hello World!')
        'Alternative name for :meth:`find_isomorphisms`.'
        return self.find_isomorphisms(symmetry)

    def _find_nodecolor_candidates(self):
        if False:
            i = 10
            return i + 15
        '\n        Per node in subgraph find all nodes in graph that have the same color.\n        '
        candidates = defaultdict(set)
        for sgn in self.subgraph.nodes:
            sgn_color = self._sgn_colors[sgn]
            if sgn_color in self._node_compatibility:
                gn_color = self._node_compatibility[sgn_color]
                candidates[sgn].add(frozenset(self._gn_partitions[gn_color]))
            else:
                candidates[sgn].add(frozenset())
        candidates = dict(candidates)
        for (sgn, options) in candidates.items():
            candidates[sgn] = frozenset(options)
        return candidates

    @staticmethod
    def _make_constraints(cosets):
        if False:
            print('Hello World!')
        '\n        Turn cosets into constraints.\n        '
        constraints = []
        for (node_i, node_ts) in cosets.items():
            for node_t in node_ts:
                if node_i != node_t:
                    constraints.append((node_i, node_t))
        return constraints

    @staticmethod
    def _find_node_edge_color(graph, node_colors, edge_colors):
        if False:
            while True:
                i = 10
        '\n        For every node in graph, come up with a color that combines 1) the\n        color of the node, and 2) the number of edges of a color to each type\n        of node.\n        '
        counts = defaultdict(lambda : defaultdict(int))
        for (node1, node2) in graph.edges:
            if (node1, node2) in edge_colors:
                ecolor = edge_colors[node1, node2]
            else:
                ecolor = edge_colors[node2, node1]
            counts[node1][ecolor, node_colors[node2]] += 1
            counts[node2][ecolor, node_colors[node1]] += 1
        node_edge_colors = {}
        for node in graph.nodes:
            node_edge_colors[node] = (node_colors[node], set(counts[node].items()))
        return node_edge_colors

    @staticmethod
    def _get_permutations_by_length(items):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get all permutations of items, but only permute items with the same\n        length.\n\n        >>> found = list(ISMAGS._get_permutations_by_length([[1], [2], [3, 4], [4, 5]]))\n        >>> answer = [\n        ...     (([1], [2]), ([3, 4], [4, 5])),\n        ...     (([1], [2]), ([4, 5], [3, 4])),\n        ...     (([2], [1]), ([3, 4], [4, 5])),\n        ...     (([2], [1]), ([4, 5], [3, 4])),\n        ... ]\n        >>> found == answer\n        True\n        '
        by_len = defaultdict(list)
        for item in items:
            by_len[len(item)].append(item)
        yield from itertools.product(*(itertools.permutations(by_len[l]) for l in sorted(by_len)))

    @classmethod
    def _refine_node_partitions(cls, graph, node_partitions, edge_colors, branch=False):
        if False:
            return 10
        '\n        Given a partition of nodes in graph, make the partitions smaller such\n        that all nodes in a partition have 1) the same color, and 2) the same\n        number of edges to specific other partitions.\n        '

        def equal_color(node1, node2):
            if False:
                for i in range(10):
                    print('nop')
            return node_edge_colors[node1] == node_edge_colors[node2]
        node_partitions = list(node_partitions)
        node_colors = partition_to_color(node_partitions)
        node_edge_colors = cls._find_node_edge_color(graph, node_colors, edge_colors)
        if all((are_all_equal((node_edge_colors[node] for node in partition)) for partition in node_partitions)):
            yield node_partitions
            return
        new_partitions = []
        output = [new_partitions]
        for partition in node_partitions:
            if not are_all_equal((node_edge_colors[node] for node in partition)):
                refined = make_partitions(partition, equal_color)
                if branch and len(refined) != 1 and (len({len(r) for r in refined}) != len([len(r) for r in refined])):
                    permutations = cls._get_permutations_by_length(refined)
                    new_output = []
                    for n_p in output:
                        for permutation in permutations:
                            new_output.append(n_p + list(permutation[0]))
                    output = new_output
                else:
                    for n_p in output:
                        n_p.extend(sorted(refined, key=len))
            else:
                for n_p in output:
                    n_p.append(partition)
        for n_p in output:
            yield from cls._refine_node_partitions(graph, n_p, edge_colors, branch)

    def _edges_of_same_color(self, sgn1, sgn2):
        if False:
            print('Hello World!')
        '\n        Returns all edges in :attr:`graph` that have the same colour as the\n        edge between sgn1 and sgn2 in :attr:`subgraph`.\n        '
        if (sgn1, sgn2) in self._sge_colors:
            sge_color = self._sge_colors[sgn1, sgn2]
        else:
            sge_color = self._sge_colors[sgn2, sgn1]
        if sge_color in self._edge_compatibility:
            ge_color = self._edge_compatibility[sge_color]
            g_edges = self._ge_partitions[ge_color]
        else:
            g_edges = []
        return g_edges

    def _map_nodes(self, sgn, candidates, constraints, mapping=None, to_be_mapped=None):
        if False:
            print('Hello World!')
        '\n        Find all subgraph isomorphisms honoring constraints.\n        '
        if mapping is None:
            mapping = {}
        else:
            mapping = mapping.copy()
        if to_be_mapped is None:
            to_be_mapped = set(self.subgraph.nodes)
        sgn_candidates = intersect(candidates[sgn])
        candidates[sgn] = frozenset([sgn_candidates])
        for gn in sgn_candidates:
            if gn in mapping.values() or sgn not in to_be_mapped:
                continue
            mapping[sgn] = gn
            if to_be_mapped == set(mapping.keys()):
                yield {v: k for (k, v) in mapping.items()}
                continue
            left_to_map = to_be_mapped - set(mapping.keys())
            new_candidates = candidates.copy()
            sgn_neighbours = set(self.subgraph[sgn])
            not_gn_neighbours = set(self.graph.nodes) - set(self.graph[gn])
            for sgn2 in left_to_map:
                if sgn2 not in sgn_neighbours:
                    gn2_options = not_gn_neighbours
                else:
                    g_edges = self._edges_of_same_color(sgn, sgn2)
                    gn2_options = {n for e in g_edges for n in e if gn in e}
                new_candidates[sgn2] = new_candidates[sgn2].union([frozenset(gn2_options)])
                if (sgn, sgn2) in constraints:
                    gn2_options = {gn2 for gn2 in self.graph if gn2 > gn}
                elif (sgn2, sgn) in constraints:
                    gn2_options = {gn2 for gn2 in self.graph if gn2 < gn}
                else:
                    continue
                new_candidates[sgn2] = new_candidates[sgn2].union([frozenset(gn2_options)])
            next_sgn = min(left_to_map, key=lambda n: min(new_candidates[n], key=len))
            yield from self._map_nodes(next_sgn, new_candidates, constraints, mapping=mapping, to_be_mapped=to_be_mapped)

    def _largest_common_subgraph(self, candidates, constraints, to_be_mapped=None):
        if False:
            while True:
                i = 10
        '\n        Find all largest common subgraphs honoring constraints.\n        '
        if to_be_mapped is None:
            to_be_mapped = {frozenset(self.subgraph.nodes)}
        current_size = len(next(iter(to_be_mapped), []))
        found_iso = False
        if current_size <= len(self.graph):
            for nodes in sorted(to_be_mapped, key=sorted):
                next_sgn = min(nodes, key=lambda n: min(candidates[n], key=len))
                isomorphs = self._map_nodes(next_sgn, candidates, constraints, to_be_mapped=nodes)
                try:
                    item = next(isomorphs)
                except StopIteration:
                    pass
                else:
                    yield item
                    yield from isomorphs
                    found_iso = True
        if found_iso or current_size == 1:
            return
        left_to_be_mapped = set()
        for nodes in to_be_mapped:
            for sgn in nodes:
                new_nodes = self._remove_node(sgn, nodes, constraints)
                left_to_be_mapped.add(new_nodes)
        yield from self._largest_common_subgraph(candidates, constraints, to_be_mapped=left_to_be_mapped)

    @staticmethod
    def _remove_node(node, nodes, constraints):
        if False:
            i = 10
            return i + 15
        '\n        Returns a new set where node has been removed from nodes, subject to\n        symmetry constraints. We know, that for every constraint we have\n        those subgraph nodes are equal. So whenever we would remove the\n        lower part of a constraint, remove the higher instead.\n        '
        while True:
            for (low, high) in constraints:
                if low == node and high in nodes:
                    node = high
                    break
            else:
                break
        return frozenset(nodes - {node})

    @staticmethod
    def _find_permutations(top_partitions, bottom_partitions):
        if False:
            i = 10
            return i + 15
        '\n        Return the pairs of top/bottom partitions where the partitions are\n        different. Ensures that all partitions in both top and bottom\n        partitions have size 1.\n        '
        permutations = set()
        for (top, bot) in zip(top_partitions, bottom_partitions):
            if len(top) != 1 or len(bot) != 1:
                raise IndexError(f'Not all nodes are coupled. This is impossible: {top_partitions}, {bottom_partitions}')
            if top != bot:
                permutations.add(frozenset((next(iter(top)), next(iter(bot)))))
        return permutations

    @staticmethod
    def _update_orbits(orbits, permutations):
        if False:
            print('Hello World!')
        '\n        Update orbits based on permutations. Orbits is modified in place.\n        For every pair of items in permutations their respective orbits are\n        merged.\n        '
        for permutation in permutations:
            (node, node2) = permutation
            first = second = None
            for (idx, orbit) in enumerate(orbits):
                if first is not None and second is not None:
                    break
                if node in orbit:
                    first = idx
                if node2 in orbit:
                    second = idx
            if first != second:
                orbits[first].update(orbits[second])
                del orbits[second]

    def _couple_nodes(self, top_partitions, bottom_partitions, pair_idx, t_node, b_node, graph, edge_colors):
        if False:
            return 10
        '\n        Generate new partitions from top and bottom_partitions where t_node is\n        coupled to b_node. pair_idx is the index of the partitions where t_ and\n        b_node can be found.\n        '
        t_partition = top_partitions[pair_idx]
        b_partition = bottom_partitions[pair_idx]
        assert t_node in t_partition and b_node in b_partition
        new_top_partitions = [top.copy() for top in top_partitions]
        new_bottom_partitions = [bot.copy() for bot in bottom_partitions]
        new_t_groups = ({t_node}, t_partition - {t_node})
        new_b_groups = ({b_node}, b_partition - {b_node})
        del new_top_partitions[pair_idx]
        del new_bottom_partitions[pair_idx]
        new_top_partitions[pair_idx:pair_idx] = new_t_groups
        new_bottom_partitions[pair_idx:pair_idx] = new_b_groups
        new_top_partitions = self._refine_node_partitions(graph, new_top_partitions, edge_colors)
        new_bottom_partitions = self._refine_node_partitions(graph, new_bottom_partitions, edge_colors, branch=True)
        new_top_partitions = list(new_top_partitions)
        assert len(new_top_partitions) == 1
        new_top_partitions = new_top_partitions[0]
        for bot in new_bottom_partitions:
            yield (list(new_top_partitions), bot)

    def _process_ordered_pair_partitions(self, graph, top_partitions, bottom_partitions, edge_colors, orbits=None, cosets=None):
        if False:
            return 10
        '\n        Processes ordered pair partitions as per the reference paper. Finds and\n        returns all permutations and cosets that leave the graph unchanged.\n        '
        if orbits is None:
            orbits = [{node} for node in graph.nodes]
        else:
            orbits = orbits
        if cosets is None:
            cosets = {}
        else:
            cosets = cosets.copy()
        assert all((len(t_p) == len(b_p) for (t_p, b_p) in zip(top_partitions, bottom_partitions)))
        if all((len(top) == 1 for top in top_partitions)):
            permutations = self._find_permutations(top_partitions, bottom_partitions)
            self._update_orbits(orbits, permutations)
            if permutations:
                return ([permutations], cosets)
            else:
                return ([], cosets)
        permutations = []
        unmapped_nodes = {(node, idx) for (idx, t_partition) in enumerate(top_partitions) for node in t_partition if len(t_partition) > 1}
        (node, pair_idx) = min(unmapped_nodes)
        b_partition = bottom_partitions[pair_idx]
        for node2 in sorted(b_partition):
            if len(b_partition) == 1:
                continue
            if node != node2 and any((node in orbit and node2 in orbit for orbit in orbits)):
                continue
            partitions = self._couple_nodes(top_partitions, bottom_partitions, pair_idx, node, node2, graph, edge_colors)
            for opp in partitions:
                (new_top_partitions, new_bottom_partitions) = opp
                (new_perms, new_cosets) = self._process_ordered_pair_partitions(graph, new_top_partitions, new_bottom_partitions, edge_colors, orbits, cosets)
                permutations += new_perms
                cosets.update(new_cosets)
        mapped = {k for (top, bottom) in zip(top_partitions, bottom_partitions) for k in top if len(top) == 1 and top == bottom}
        ks = {k for k in graph.nodes if k < node}
        find_coset = ks <= mapped and node not in cosets
        if find_coset:
            for orbit in orbits:
                if node in orbit:
                    cosets[node] = orbit.copy()
        return (permutations, cosets)