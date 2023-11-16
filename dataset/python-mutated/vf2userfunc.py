"""
    Module to simplify the specification of user-defined equality functions for
    node and edge attributes during isomorphism checks.

    During the construction of an isomorphism, the algorithm considers two
    candidate nodes n1 in G1 and n2 in G2.  The graphs G1 and G2 are then
    compared with respect to properties involving n1 and n2, and if the outcome
    is good, then the candidate nodes are considered isomorphic. NetworkX
    provides a simple mechanism for users to extend the comparisons to include
    node and edge attributes.

    Node attributes are handled by the node_match keyword. When considering
    n1 and n2, the algorithm passes their node attribute dictionaries to
    node_match, and if it returns False, then n1 and n2 cannot be
    considered to be isomorphic.

    Edge attributes are handled by the edge_match keyword. When considering
    n1 and n2, the algorithm must verify that outgoing edges from n1 are
    commensurate with the outgoing edges for n2. If the graph is directed,
    then a similar check is also performed for incoming edges.

    Focusing only on outgoing edges, we consider pairs of nodes (n1, v1) from
    G1 and (n2, v2) from G2. For graphs and digraphs, there is only one edge
    between (n1, v1) and only one edge between (n2, v2). Those edge attribute
    dictionaries are passed to edge_match, and if it returns False, then
    n1 and n2 cannot be considered isomorphic. For multigraphs and
    multidigraphs, there can be multiple edges between (n1, v1) and also
    multiple edges between (n2, v2).  Now, there must exist an isomorphism
    from "all the edges between (n1, v1)" to "all the edges between (n2, v2)".
    So, all of the edge attribute dictionaries are passed to edge_match, and
    it must determine if there is an isomorphism between the two sets of edges.
"""
from . import isomorphvf2 as vf2
__all__ = ['GraphMatcher', 'DiGraphMatcher', 'MultiGraphMatcher', 'MultiDiGraphMatcher']

def _semantic_feasibility(self, G1_node, G2_node):
    if False:
        return 10
    'Returns True if mapping G1_node to G2_node is semantically feasible.'
    if self.node_match is not None:
        nm = self.node_match(self.G1.nodes[G1_node], self.G2.nodes[G2_node])
        if not nm:
            return False
    if self.edge_match is not None:
        G1nbrs = self.G1_adj[G1_node]
        G2nbrs = self.G2_adj[G2_node]
        core_1 = self.core_1
        edge_match = self.edge_match
        for neighbor in G1nbrs:
            if neighbor == G1_node:
                if G2_node in G2nbrs and (not edge_match(G1nbrs[G1_node], G2nbrs[G2_node])):
                    return False
            elif neighbor in core_1:
                G2_nbr = core_1[neighbor]
                if G2_nbr in G2nbrs and (not edge_match(G1nbrs[neighbor], G2nbrs[G2_nbr])):
                    return False
    return True

class GraphMatcher(vf2.GraphMatcher):
    """VF2 isomorphism checker for undirected graphs."""

    def __init__(self, G1, G2, node_match=None, edge_match=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize graph matcher.\n\n        Parameters\n        ----------\n        G1, G2: graph\n            The graphs to be tested.\n\n        node_match: callable\n            A function that returns True iff node n1 in G1 and n2 in G2\n            should be considered equal during the isomorphism test. The\n            function will be called like::\n\n               node_match(G1.nodes[n1], G2.nodes[n2])\n\n            That is, the function will receive the node attribute dictionaries\n            of the nodes under consideration. If None, then no attributes are\n            considered when testing for an isomorphism.\n\n        edge_match: callable\n            A function that returns True iff the edge attribute dictionary for\n            the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should be\n            considered equal during the isomorphism test. The function will be\n            called like::\n\n               edge_match(G1[u1][v1], G2[u2][v2])\n\n            That is, the function will receive the edge attribute dictionaries\n            of the edges under consideration. If None, then no attributes are\n            considered when testing for an isomorphism.\n\n        '
        vf2.GraphMatcher.__init__(self, G1, G2)
        self.node_match = node_match
        self.edge_match = edge_match
        self.G1_adj = self.G1.adj
        self.G2_adj = self.G2.adj
    semantic_feasibility = _semantic_feasibility

class DiGraphMatcher(vf2.DiGraphMatcher):
    """VF2 isomorphism checker for directed graphs."""

    def __init__(self, G1, G2, node_match=None, edge_match=None):
        if False:
            i = 10
            return i + 15
        'Initialize graph matcher.\n\n        Parameters\n        ----------\n        G1, G2 : graph\n            The graphs to be tested.\n\n        node_match : callable\n            A function that returns True iff node n1 in G1 and n2 in G2\n            should be considered equal during the isomorphism test. The\n            function will be called like::\n\n               node_match(G1.nodes[n1], G2.nodes[n2])\n\n            That is, the function will receive the node attribute dictionaries\n            of the nodes under consideration. If None, then no attributes are\n            considered when testing for an isomorphism.\n\n        edge_match : callable\n            A function that returns True iff the edge attribute dictionary for\n            the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should be\n            considered equal during the isomorphism test. The function will be\n            called like::\n\n               edge_match(G1[u1][v1], G2[u2][v2])\n\n            That is, the function will receive the edge attribute dictionaries\n            of the edges under consideration. If None, then no attributes are\n            considered when testing for an isomorphism.\n\n        '
        vf2.DiGraphMatcher.__init__(self, G1, G2)
        self.node_match = node_match
        self.edge_match = edge_match
        self.G1_adj = self.G1.adj
        self.G2_adj = self.G2.adj

    def semantic_feasibility(self, G1_node, G2_node):
        if False:
            i = 10
            return i + 15
        'Returns True if mapping G1_node to G2_node is semantically feasible.'
        feasible = _semantic_feasibility(self, G1_node, G2_node)
        if not feasible:
            return False
        self.G1_adj = self.G1.pred
        self.G2_adj = self.G2.pred
        feasible = _semantic_feasibility(self, G1_node, G2_node)
        self.G1_adj = self.G1.adj
        self.G2_adj = self.G2.adj
        return feasible

class MultiGraphMatcher(GraphMatcher):
    """VF2 isomorphism checker for undirected multigraphs."""

class MultiDiGraphMatcher(DiGraphMatcher):
    """VF2 isomorphism checker for directed multigraphs."""