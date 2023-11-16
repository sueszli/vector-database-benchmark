"""
*************
VF2 Algorithm
*************

An implementation of VF2 algorithm for graph isomorphism testing.

The simplest interface to use this module is to call the
:func:`is_isomorphic <networkx.algorithms.isomorphism.is_isomorphic>`
function.

Introduction
------------

The GraphMatcher and DiGraphMatcher are responsible for matching
graphs or directed graphs in a predetermined manner.  This
usually means a check for an isomorphism, though other checks
are also possible.  For example, a subgraph of one graph
can be checked for isomorphism to a second graph.

Matching is done via syntactic feasibility. It is also possible
to check for semantic feasibility. Feasibility, then, is defined
as the logical AND of the two functions.

To include a semantic check, the (Di)GraphMatcher class should be
subclassed, and the
:meth:`semantic_feasibility <networkx.algorithms.isomorphism.GraphMatcher.semantic_feasibility>`
function should be redefined.  By default, the semantic feasibility function always
returns ``True``.  The effect of this is that semantics are not
considered in the matching of G1 and G2.

Examples
--------

Suppose G1 and G2 are isomorphic graphs. Verification is as follows:

>>> from networkx.algorithms import isomorphism
>>> G1 = nx.path_graph(4)
>>> G2 = nx.path_graph(4)
>>> GM = isomorphism.GraphMatcher(G1, G2)
>>> GM.is_isomorphic()
True

GM.mapping stores the isomorphism mapping from G1 to G2.

>>> GM.mapping
{0: 0, 1: 1, 2: 2, 3: 3}


Suppose G1 and G2 are isomorphic directed graphs.
Verification is as follows:

>>> G1 = nx.path_graph(4, create_using=nx.DiGraph())
>>> G2 = nx.path_graph(4, create_using=nx.DiGraph())
>>> DiGM = isomorphism.DiGraphMatcher(G1, G2)
>>> DiGM.is_isomorphic()
True

DiGM.mapping stores the isomorphism mapping from G1 to G2.

>>> DiGM.mapping
{0: 0, 1: 1, 2: 2, 3: 3}



Subgraph Isomorphism
--------------------
Graph theory literature can be ambiguous about the meaning of the
above statement, and we seek to clarify it now.

In the VF2 literature, a mapping `M` is said to be a graph-subgraph
isomorphism iff `M` is an isomorphism between `G2` and a subgraph of `G1`.
Thus, to say that `G1` and `G2` are graph-subgraph isomorphic is to say
that a subgraph of `G1` is isomorphic to `G2`.

Other literature uses the phrase 'subgraph isomorphic' as in '`G1` does
not have a subgraph isomorphic to `G2`'.  Another use is as an in adverb
for isomorphic.  Thus, to say that `G1` and `G2` are subgraph isomorphic
is to say that a subgraph of `G1` is isomorphic to `G2`.

Finally, the term 'subgraph' can have multiple meanings. In this
context, 'subgraph' always means a 'node-induced subgraph'. Edge-induced
subgraph isomorphisms are not directly supported, but one should be
able to perform the check by making use of
:func:`line_graph <networkx.generators.line.line_graph>`. For
subgraphs which are not induced, the term 'monomorphism' is preferred
over 'isomorphism'.

Let ``G = (N, E)`` be a graph with a set of nodes `N` and set of edges `E`.

If ``G' = (N', E')`` is a subgraph, then:
    `N'` is a subset of `N` and
    `E'` is a subset of `E`.

If ``G' = (N', E')`` is a node-induced subgraph, then:
    `N'` is a subset of `N` and
    `E'` is the subset of edges in `E` relating nodes in `N'`.

If `G' = (N', E')` is an edge-induced subgraph, then:
    `N'` is the subset of nodes in `N` related by edges in `E'` and
    `E'` is a subset of `E`.

If `G' = (N', E')` is a monomorphism, then:
    `N'` is a subset of `N` and
    `E'` is a subset of the set of edges in `E` relating nodes in `N'`.

Note that if `G'` is a node-induced subgraph of `G`, then it is always a
subgraph monomorphism of `G`, but the opposite is not always true, as a
monomorphism can have fewer edges.

References
----------
[1]   Luigi P. Cordella, Pasquale Foggia, Carlo Sansone, Mario Vento,
      "A (Sub)Graph Isomorphism Algorithm for Matching Large Graphs",
      IEEE Transactions on Pattern Analysis and Machine Intelligence,
      vol. 26,  no. 10,  pp. 1367-1372,  Oct.,  2004.
      http://ieeexplore.ieee.org/iel5/34/29305/01323804.pdf

[2]   L. P. Cordella, P. Foggia, C. Sansone, M. Vento, "An Improved
      Algorithm for Matching Large Graphs", 3rd IAPR-TC15 Workshop
      on Graph-based Representations in Pattern Recognition, Cuen,
      pp. 149-159, 2001.
      https://www.researchgate.net/publication/200034365_An_Improved_Algorithm_for_Matching_Large_Graphs

See Also
--------
:meth:`semantic_feasibility <networkx.algorithms.isomorphism.GraphMatcher.semantic_feasibility>`
:meth:`syntactic_feasibility <networkx.algorithms.isomorphism.GraphMatcher.syntactic_feasibility>`

Notes
-----

The implementation handles both directed and undirected graphs as well
as multigraphs.

In general, the subgraph isomorphism problem is NP-complete whereas the
graph isomorphism problem is most likely not NP-complete (although no
polynomial-time algorithm is known to exist).

"""
import sys
__all__ = ['GraphMatcher', 'DiGraphMatcher']

class GraphMatcher:
    """Implementation of VF2 algorithm for matching undirected graphs.

    Suitable for Graph and MultiGraph instances.
    """

    def __init__(self, G1, G2):
        if False:
            return 10
        'Initialize GraphMatcher.\n\n        Parameters\n        ----------\n        G1,G2: NetworkX Graph or MultiGraph instances.\n           The two graphs to check for isomorphism or monomorphism.\n\n        Examples\n        --------\n        To create a GraphMatcher which checks for syntactic feasibility:\n\n        >>> from networkx.algorithms import isomorphism\n        >>> G1 = nx.path_graph(4)\n        >>> G2 = nx.path_graph(4)\n        >>> GM = isomorphism.GraphMatcher(G1, G2)\n        '
        self.G1 = G1
        self.G2 = G2
        self.G1_nodes = set(G1.nodes())
        self.G2_nodes = set(G2.nodes())
        self.G2_node_order = {n: i for (i, n) in enumerate(G2)}
        self.old_recursion_limit = sys.getrecursionlimit()
        expected_max_recursion_level = len(self.G2)
        if self.old_recursion_limit < 1.5 * expected_max_recursion_level:
            sys.setrecursionlimit(int(1.5 * expected_max_recursion_level))
        self.test = 'graph'
        self.initialize()

    def reset_recursion_limit(self):
        if False:
            while True:
                i = 10
        'Restores the recursion limit.'
        sys.setrecursionlimit(self.old_recursion_limit)

    def candidate_pairs_iter(self):
        if False:
            for i in range(10):
                print('nop')
        'Iterator over candidate pairs of nodes in G1 and G2.'
        G1_nodes = self.G1_nodes
        G2_nodes = self.G2_nodes
        min_key = self.G2_node_order.__getitem__
        T1_inout = [node for node in self.inout_1 if node not in self.core_1]
        T2_inout = [node for node in self.inout_2 if node not in self.core_2]
        if T1_inout and T2_inout:
            node_2 = min(T2_inout, key=min_key)
            for node_1 in T1_inout:
                yield (node_1, node_2)
        elif 1:
            other_node = min(G2_nodes - set(self.core_2), key=min_key)
            for node in self.G1:
                if node not in self.core_1:
                    yield (node, other_node)

    def initialize(self):
        if False:
            while True:
                i = 10
        'Reinitializes the state of the algorithm.\n\n        This method should be redefined if using something other than GMState.\n        If only subclassing GraphMatcher, a redefinition is not necessary.\n\n        '
        self.core_1 = {}
        self.core_2 = {}
        self.inout_1 = {}
        self.inout_2 = {}
        self.state = GMState(self)
        self.mapping = self.core_1.copy()

    def is_isomorphic(self):
        if False:
            i = 10
            return i + 15
        'Returns True if G1 and G2 are isomorphic graphs.'
        if self.G1.order() != self.G2.order():
            return False
        d1 = sorted((d for (n, d) in self.G1.degree()))
        d2 = sorted((d for (n, d) in self.G2.degree()))
        if d1 != d2:
            return False
        try:
            x = next(self.isomorphisms_iter())
            return True
        except StopIteration:
            return False

    def isomorphisms_iter(self):
        if False:
            return 10
        'Generator over isomorphisms between G1 and G2.'
        self.test = 'graph'
        self.initialize()
        yield from self.match()

    def match(self):
        if False:
            return 10
        'Extends the isomorphism mapping.\n\n        This function is called recursively to determine if a complete\n        isomorphism can be found between G1 and G2.  It cleans up the class\n        variables after each recursive call. If an isomorphism is found,\n        we yield the mapping.\n\n        '
        if len(self.core_1) == len(self.G2):
            self.mapping = self.core_1.copy()
            yield self.mapping
        else:
            for (G1_node, G2_node) in self.candidate_pairs_iter():
                if self.syntactic_feasibility(G1_node, G2_node):
                    if self.semantic_feasibility(G1_node, G2_node):
                        newstate = self.state.__class__(self, G1_node, G2_node)
                        yield from self.match()
                        newstate.restore()

    def semantic_feasibility(self, G1_node, G2_node):
        if False:
            while True:
                i = 10
        "Returns True if adding (G1_node, G2_node) is semantically feasible.\n\n        The semantic feasibility function should return True if it is\n        acceptable to add the candidate pair (G1_node, G2_node) to the current\n        partial isomorphism mapping.   The logic should focus on semantic\n        information contained in the edge data or a formalized node class.\n\n        By acceptable, we mean that the subsequent mapping can still become a\n        complete isomorphism mapping.  Thus, if adding the candidate pair\n        definitely makes it so that the subsequent mapping cannot become a\n        complete isomorphism mapping, then this function must return False.\n\n        The default semantic feasibility function always returns True. The\n        effect is that semantics are not considered in the matching of G1\n        and G2.\n\n        The semantic checks might differ based on the what type of test is\n        being performed.  A keyword description of the test is stored in\n        self.test.  Here is a quick description of the currently implemented\n        tests::\n\n          test='graph'\n            Indicates that the graph matcher is looking for a graph-graph\n            isomorphism.\n\n          test='subgraph'\n            Indicates that the graph matcher is looking for a subgraph-graph\n            isomorphism such that a subgraph of G1 is isomorphic to G2.\n\n          test='mono'\n            Indicates that the graph matcher is looking for a subgraph-graph\n            monomorphism such that a subgraph of G1 is monomorphic to G2.\n\n        Any subclass which redefines semantic_feasibility() must maintain\n        the above form to keep the match() method functional. Implementations\n        should consider multigraphs.\n        "
        return True

    def subgraph_is_isomorphic(self):
        if False:
            print('Hello World!')
        'Returns True if a subgraph of G1 is isomorphic to G2.'
        try:
            x = next(self.subgraph_isomorphisms_iter())
            return True
        except StopIteration:
            return False

    def subgraph_is_monomorphic(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if a subgraph of G1 is monomorphic to G2.'
        try:
            x = next(self.subgraph_monomorphisms_iter())
            return True
        except StopIteration:
            return False

    def subgraph_isomorphisms_iter(self):
        if False:
            return 10
        'Generator over isomorphisms between a subgraph of G1 and G2.'
        self.test = 'subgraph'
        self.initialize()
        yield from self.match()

    def subgraph_monomorphisms_iter(self):
        if False:
            return 10
        'Generator over monomorphisms between a subgraph of G1 and G2.'
        self.test = 'mono'
        self.initialize()
        yield from self.match()

    def syntactic_feasibility(self, G1_node, G2_node):
        if False:
            return 10
        'Returns True if adding (G1_node, G2_node) is syntactically feasible.\n\n        This function returns True if it is adding the candidate pair\n        to the current partial isomorphism/monomorphism mapping is allowable.\n        The addition is allowable if the inclusion of the candidate pair does\n        not make it impossible for an isomorphism/monomorphism to be found.\n        '
        if self.test == 'mono':
            if self.G1.number_of_edges(G1_node, G1_node) < self.G2.number_of_edges(G2_node, G2_node):
                return False
        elif self.G1.number_of_edges(G1_node, G1_node) != self.G2.number_of_edges(G2_node, G2_node):
            return False
        if self.test != 'mono':
            for neighbor in self.G1[G1_node]:
                if neighbor in self.core_1:
                    if self.core_1[neighbor] not in self.G2[G2_node]:
                        return False
                    elif self.G1.number_of_edges(neighbor, G1_node) != self.G2.number_of_edges(self.core_1[neighbor], G2_node):
                        return False
        for neighbor in self.G2[G2_node]:
            if neighbor in self.core_2:
                if self.core_2[neighbor] not in self.G1[G1_node]:
                    return False
                elif self.test == 'mono':
                    if self.G1.number_of_edges(self.core_2[neighbor], G1_node) < self.G2.number_of_edges(neighbor, G2_node):
                        return False
                elif self.G1.number_of_edges(self.core_2[neighbor], G1_node) != self.G2.number_of_edges(neighbor, G2_node):
                    return False
        if self.test != 'mono':
            num1 = 0
            for neighbor in self.G1[G1_node]:
                if neighbor in self.inout_1 and neighbor not in self.core_1:
                    num1 += 1
            num2 = 0
            for neighbor in self.G2[G2_node]:
                if neighbor in self.inout_2 and neighbor not in self.core_2:
                    num2 += 1
            if self.test == 'graph':
                if num1 != num2:
                    return False
            elif not num1 >= num2:
                return False
            num1 = 0
            for neighbor in self.G1[G1_node]:
                if neighbor not in self.inout_1:
                    num1 += 1
            num2 = 0
            for neighbor in self.G2[G2_node]:
                if neighbor not in self.inout_2:
                    num2 += 1
            if self.test == 'graph':
                if num1 != num2:
                    return False
            elif not num1 >= num2:
                return False
        return True

class DiGraphMatcher(GraphMatcher):
    """Implementation of VF2 algorithm for matching directed graphs.

    Suitable for DiGraph and MultiDiGraph instances.
    """

    def __init__(self, G1, G2):
        if False:
            while True:
                i = 10
        'Initialize DiGraphMatcher.\n\n        G1 and G2 should be nx.Graph or nx.MultiGraph instances.\n\n        Examples\n        --------\n        To create a GraphMatcher which checks for syntactic feasibility:\n\n        >>> from networkx.algorithms import isomorphism\n        >>> G1 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))\n        >>> G2 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))\n        >>> DiGM = isomorphism.DiGraphMatcher(G1, G2)\n        '
        super().__init__(G1, G2)

    def candidate_pairs_iter(self):
        if False:
            for i in range(10):
                print('nop')
        'Iterator over candidate pairs of nodes in G1 and G2.'
        G1_nodes = self.G1_nodes
        G2_nodes = self.G2_nodes
        min_key = self.G2_node_order.__getitem__
        T1_out = [node for node in self.out_1 if node not in self.core_1]
        T2_out = [node for node in self.out_2 if node not in self.core_2]
        if T1_out and T2_out:
            node_2 = min(T2_out, key=min_key)
            for node_1 in T1_out:
                yield (node_1, node_2)
        else:
            T1_in = [node for node in self.in_1 if node not in self.core_1]
            T2_in = [node for node in self.in_2 if node not in self.core_2]
            if T1_in and T2_in:
                node_2 = min(T2_in, key=min_key)
                for node_1 in T1_in:
                    yield (node_1, node_2)
            else:
                node_2 = min(G2_nodes - set(self.core_2), key=min_key)
                for node_1 in G1_nodes:
                    if node_1 not in self.core_1:
                        yield (node_1, node_2)

    def initialize(self):
        if False:
            return 10
        'Reinitializes the state of the algorithm.\n\n        This method should be redefined if using something other than DiGMState.\n        If only subclassing GraphMatcher, a redefinition is not necessary.\n        '
        self.core_1 = {}
        self.core_2 = {}
        self.in_1 = {}
        self.in_2 = {}
        self.out_1 = {}
        self.out_2 = {}
        self.state = DiGMState(self)
        self.mapping = self.core_1.copy()

    def syntactic_feasibility(self, G1_node, G2_node):
        if False:
            while True:
                i = 10
        'Returns True if adding (G1_node, G2_node) is syntactically feasible.\n\n        This function returns True if it is adding the candidate pair\n        to the current partial isomorphism/monomorphism mapping is allowable.\n        The addition is allowable if the inclusion of the candidate pair does\n        not make it impossible for an isomorphism/monomorphism to be found.\n        '
        if self.test == 'mono':
            if self.G1.number_of_edges(G1_node, G1_node) < self.G2.number_of_edges(G2_node, G2_node):
                return False
        elif self.G1.number_of_edges(G1_node, G1_node) != self.G2.number_of_edges(G2_node, G2_node):
            return False
        if self.test != 'mono':
            for predecessor in self.G1.pred[G1_node]:
                if predecessor in self.core_1:
                    if self.core_1[predecessor] not in self.G2.pred[G2_node]:
                        return False
                    elif self.G1.number_of_edges(predecessor, G1_node) != self.G2.number_of_edges(self.core_1[predecessor], G2_node):
                        return False
        for predecessor in self.G2.pred[G2_node]:
            if predecessor in self.core_2:
                if self.core_2[predecessor] not in self.G1.pred[G1_node]:
                    return False
                elif self.test == 'mono':
                    if self.G1.number_of_edges(self.core_2[predecessor], G1_node) < self.G2.number_of_edges(predecessor, G2_node):
                        return False
                elif self.G1.number_of_edges(self.core_2[predecessor], G1_node) != self.G2.number_of_edges(predecessor, G2_node):
                    return False
        if self.test != 'mono':
            for successor in self.G1[G1_node]:
                if successor in self.core_1:
                    if self.core_1[successor] not in self.G2[G2_node]:
                        return False
                    elif self.G1.number_of_edges(G1_node, successor) != self.G2.number_of_edges(G2_node, self.core_1[successor]):
                        return False
        for successor in self.G2[G2_node]:
            if successor in self.core_2:
                if self.core_2[successor] not in self.G1[G1_node]:
                    return False
                elif self.test == 'mono':
                    if self.G1.number_of_edges(G1_node, self.core_2[successor]) < self.G2.number_of_edges(G2_node, successor):
                        return False
                elif self.G1.number_of_edges(G1_node, self.core_2[successor]) != self.G2.number_of_edges(G2_node, successor):
                    return False
        if self.test != 'mono':
            num1 = 0
            for predecessor in self.G1.pred[G1_node]:
                if predecessor in self.in_1 and predecessor not in self.core_1:
                    num1 += 1
            num2 = 0
            for predecessor in self.G2.pred[G2_node]:
                if predecessor in self.in_2 and predecessor not in self.core_2:
                    num2 += 1
            if self.test == 'graph':
                if num1 != num2:
                    return False
            elif not num1 >= num2:
                return False
            num1 = 0
            for successor in self.G1[G1_node]:
                if successor in self.in_1 and successor not in self.core_1:
                    num1 += 1
            num2 = 0
            for successor in self.G2[G2_node]:
                if successor in self.in_2 and successor not in self.core_2:
                    num2 += 1
            if self.test == 'graph':
                if num1 != num2:
                    return False
            elif not num1 >= num2:
                return False
            num1 = 0
            for predecessor in self.G1.pred[G1_node]:
                if predecessor in self.out_1 and predecessor not in self.core_1:
                    num1 += 1
            num2 = 0
            for predecessor in self.G2.pred[G2_node]:
                if predecessor in self.out_2 and predecessor not in self.core_2:
                    num2 += 1
            if self.test == 'graph':
                if num1 != num2:
                    return False
            elif not num1 >= num2:
                return False
            num1 = 0
            for successor in self.G1[G1_node]:
                if successor in self.out_1 and successor not in self.core_1:
                    num1 += 1
            num2 = 0
            for successor in self.G2[G2_node]:
                if successor in self.out_2 and successor not in self.core_2:
                    num2 += 1
            if self.test == 'graph':
                if num1 != num2:
                    return False
            elif not num1 >= num2:
                return False
            num1 = 0
            for predecessor in self.G1.pred[G1_node]:
                if predecessor not in self.in_1 and predecessor not in self.out_1:
                    num1 += 1
            num2 = 0
            for predecessor in self.G2.pred[G2_node]:
                if predecessor not in self.in_2 and predecessor not in self.out_2:
                    num2 += 1
            if self.test == 'graph':
                if num1 != num2:
                    return False
            elif not num1 >= num2:
                return False
            num1 = 0
            for successor in self.G1[G1_node]:
                if successor not in self.in_1 and successor not in self.out_1:
                    num1 += 1
            num2 = 0
            for successor in self.G2[G2_node]:
                if successor not in self.in_2 and successor not in self.out_2:
                    num2 += 1
            if self.test == 'graph':
                if num1 != num2:
                    return False
            elif not num1 >= num2:
                return False
        return True

class GMState:
    """Internal representation of state for the GraphMatcher class.

    This class is used internally by the GraphMatcher class.  It is used
    only to store state specific data. There will be at most G2.order() of
    these objects in memory at a time, due to the depth-first search
    strategy employed by the VF2 algorithm.
    """

    def __init__(self, GM, G1_node=None, G2_node=None):
        if False:
            return 10
        "Initializes GMState object.\n\n        Pass in the GraphMatcher to which this GMState belongs and the\n        new node pair that will be added to the GraphMatcher's current\n        isomorphism mapping.\n        "
        self.GM = GM
        self.G1_node = None
        self.G2_node = None
        self.depth = len(GM.core_1)
        if G1_node is None or G2_node is None:
            GM.core_1 = {}
            GM.core_2 = {}
            GM.inout_1 = {}
            GM.inout_2 = {}
        if G1_node is not None and G2_node is not None:
            GM.core_1[G1_node] = G2_node
            GM.core_2[G2_node] = G1_node
            self.G1_node = G1_node
            self.G2_node = G2_node
            self.depth = len(GM.core_1)
            if G1_node not in GM.inout_1:
                GM.inout_1[G1_node] = self.depth
            if G2_node not in GM.inout_2:
                GM.inout_2[G2_node] = self.depth
            new_nodes = set()
            for node in GM.core_1:
                new_nodes.update([neighbor for neighbor in GM.G1[node] if neighbor not in GM.core_1])
            for node in new_nodes:
                if node not in GM.inout_1:
                    GM.inout_1[node] = self.depth
            new_nodes = set()
            for node in GM.core_2:
                new_nodes.update([neighbor for neighbor in GM.G2[node] if neighbor not in GM.core_2])
            for node in new_nodes:
                if node not in GM.inout_2:
                    GM.inout_2[node] = self.depth

    def restore(self):
        if False:
            for i in range(10):
                print('nop')
        'Deletes the GMState object and restores the class variables.'
        if self.G1_node is not None and self.G2_node is not None:
            del self.GM.core_1[self.G1_node]
            del self.GM.core_2[self.G2_node]
        for vector in (self.GM.inout_1, self.GM.inout_2):
            for node in list(vector.keys()):
                if vector[node] == self.depth:
                    del vector[node]

class DiGMState:
    """Internal representation of state for the DiGraphMatcher class.

    This class is used internally by the DiGraphMatcher class.  It is used
    only to store state specific data. There will be at most G2.order() of
    these objects in memory at a time, due to the depth-first search
    strategy employed by the VF2 algorithm.

    """

    def __init__(self, GM, G1_node=None, G2_node=None):
        if False:
            while True:
                i = 10
        "Initializes DiGMState object.\n\n        Pass in the DiGraphMatcher to which this DiGMState belongs and the\n        new node pair that will be added to the GraphMatcher's current\n        isomorphism mapping.\n        "
        self.GM = GM
        self.G1_node = None
        self.G2_node = None
        self.depth = len(GM.core_1)
        if G1_node is None or G2_node is None:
            GM.core_1 = {}
            GM.core_2 = {}
            GM.in_1 = {}
            GM.in_2 = {}
            GM.out_1 = {}
            GM.out_2 = {}
        if G1_node is not None and G2_node is not None:
            GM.core_1[G1_node] = G2_node
            GM.core_2[G2_node] = G1_node
            self.G1_node = G1_node
            self.G2_node = G2_node
            self.depth = len(GM.core_1)
            for vector in (GM.in_1, GM.out_1):
                if G1_node not in vector:
                    vector[G1_node] = self.depth
            for vector in (GM.in_2, GM.out_2):
                if G2_node not in vector:
                    vector[G2_node] = self.depth
            new_nodes = set()
            for node in GM.core_1:
                new_nodes.update([predecessor for predecessor in GM.G1.predecessors(node) if predecessor not in GM.core_1])
            for node in new_nodes:
                if node not in GM.in_1:
                    GM.in_1[node] = self.depth
            new_nodes = set()
            for node in GM.core_2:
                new_nodes.update([predecessor for predecessor in GM.G2.predecessors(node) if predecessor not in GM.core_2])
            for node in new_nodes:
                if node not in GM.in_2:
                    GM.in_2[node] = self.depth
            new_nodes = set()
            for node in GM.core_1:
                new_nodes.update([successor for successor in GM.G1.successors(node) if successor not in GM.core_1])
            for node in new_nodes:
                if node not in GM.out_1:
                    GM.out_1[node] = self.depth
            new_nodes = set()
            for node in GM.core_2:
                new_nodes.update([successor for successor in GM.G2.successors(node) if successor not in GM.core_2])
            for node in new_nodes:
                if node not in GM.out_2:
                    GM.out_2[node] = self.depth

    def restore(self):
        if False:
            return 10
        'Deletes the DiGMState object and restores the class variables.'
        if self.G1_node is not None and self.G2_node is not None:
            del self.GM.core_1[self.G1_node]
            del self.GM.core_2[self.G2_node]
        for vector in (self.GM.in_1, self.GM.in_2, self.GM.out_1, self.GM.out_2):
            for node in list(vector.keys()):
                if vector[node] == self.depth:
                    del vector[node]