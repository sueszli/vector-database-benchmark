"""View of Graphs as SubGraph, Reverse, Directed, Undirected.

In some algorithms it is convenient to temporarily morph
a graph to exclude some nodes or edges. It should be better
to do that via a view than to remove and then re-add.
In other algorithms it is convenient to temporarily morph
a graph to reverse directed edges, or treat a directed graph
as undirected, etc. This module provides those graph views.

The resulting views are essentially read-only graphs that
report data from the original graph object. We provide an
attribute G._graph which points to the underlying graph object.

Note: Since graphviews look like graphs, one can end up with
view-of-view-of-view chains. Be careful with chains because
they become very slow with about 15 nested views.
For the common simple case of node induced subgraphs created
from the graph class, we short-cut the chain by returning a
subgraph of the original graph directly rather than a subgraph
of a subgraph. We are careful not to disrupt any edge filter in
the middle subgraph. In general, determining how to short-cut
the chain is tricky and much harder with restricted_views than
with induced subgraphs.
Often it is easiest to use .copy() to avoid chains.
"""
import networkx as nx
from networkx.classes.coreviews import FilterAdjacency, FilterAtlas, FilterMultiAdjacency, UnionAdjacency, UnionMultiAdjacency
from networkx.classes.filters import no_filter
from networkx.exception import NetworkXError
from networkx.utils import deprecate_positional_args, not_implemented_for
__all__ = ['generic_graph_view', 'subgraph_view', 'reverse_view']

def generic_graph_view(G, create_using=None):
    if False:
        for i in range(10):
            print('nop')
    "Returns a read-only view of `G`.\n\n    The graph `G` and its attributes are not copied but viewed through the new graph object\n    of the same class as `G` (or of the class specified in `create_using`).\n\n    Parameters\n    ----------\n    G : graph\n        A directed/undirected graph/multigraph.\n\n    create_using : NetworkX graph constructor, optional (default=None)\n       Graph type to create. If graph instance, then cleared before populated.\n       If `None`, then the appropriate Graph type is inferred from `G`.\n\n    Returns\n    -------\n    newG : graph\n        A view of the input graph `G` and its attributes as viewed through\n        the `create_using` class.\n\n    Raises\n    ------\n    NetworkXError\n        If `G` is a multigraph (or multidigraph) but `create_using` is not, or vice versa.\n\n    Notes\n    -----\n    The returned graph view is read-only (cannot modify the graph).\n    Yet the view reflects any changes in `G`. The intent is to mimic dict views.\n\n    Examples\n    --------\n    >>> G = nx.Graph()\n    >>> G.add_edge(1, 2, weight=0.3)\n    >>> G.add_edge(2, 3, weight=0.5)\n    >>> G.edges(data=True)\n    EdgeDataView([(1, 2, {'weight': 0.3}), (2, 3, {'weight': 0.5})])\n\n    The view exposes the attributes from the original graph.\n\n    >>> viewG = nx.graphviews.generic_graph_view(G)\n    >>> viewG.edges(data=True)\n    EdgeDataView([(1, 2, {'weight': 0.3}), (2, 3, {'weight': 0.5})])\n\n    Changes to `G` are reflected in `viewG`.\n\n    >>> G.remove_edge(2, 3)\n    >>> G.edges(data=True)\n    EdgeDataView([(1, 2, {'weight': 0.3})])\n\n    >>> viewG.edges(data=True)\n    EdgeDataView([(1, 2, {'weight': 0.3})])\n\n    We can change the graph type with the `create_using` parameter.\n\n    >>> type(G)\n    <class 'networkx.classes.graph.Graph'>\n    >>> viewDG = nx.graphviews.generic_graph_view(G, create_using=nx.DiGraph)\n    >>> type(viewDG)\n    <class 'networkx.classes.digraph.DiGraph'>\n    "
    if create_using is None:
        newG = G.__class__()
    else:
        newG = nx.empty_graph(0, create_using)
    if G.is_multigraph() != newG.is_multigraph():
        raise NetworkXError('Multigraph for G must agree with create_using')
    newG = nx.freeze(newG)
    newG._graph = G
    newG.graph = G.graph
    newG._node = G._node
    if newG.is_directed():
        if G.is_directed():
            newG._succ = G._succ
            newG._pred = G._pred
        else:
            newG._succ = G._adj
            newG._pred = G._adj
    elif G.is_directed():
        if G.is_multigraph():
            newG._adj = UnionMultiAdjacency(G._succ, G._pred)
        else:
            newG._adj = UnionAdjacency(G._succ, G._pred)
    else:
        newG._adj = G._adj
    return newG

@deprecate_positional_args(version='3.4')
def subgraph_view(G, *, filter_node=no_filter, filter_edge=no_filter):
    if False:
        i = 10
        return i + 15
    'View of `G` applying a filter on nodes and edges.\n\n    `subgraph_view` provides a read-only view of the input graph that excludes\n    nodes and edges based on the outcome of two filter functions `filter_node`\n    and `filter_edge`.\n\n    The `filter_node` function takes one argument --- the node --- and returns\n    `True` if the node should be included in the subgraph, and `False` if it\n    should not be included.\n\n    The `filter_edge` function takes two (or three arguments if `G` is a\n    multi-graph) --- the nodes describing an edge, plus the edge-key if\n    parallel edges are possible --- and returns `True` if the edge should be\n    included in the subgraph, and `False` if it should not be included.\n\n    Both node and edge filter functions are called on graph elements as they\n    are queried, meaning there is no up-front cost to creating the view.\n\n    Parameters\n    ----------\n    G : networkx.Graph\n        A directed/undirected graph/multigraph\n\n    filter_node : callable, optional\n        A function taking a node as input, which returns `True` if the node\n        should appear in the view.\n\n    filter_edge : callable, optional\n        A function taking as input the two nodes describing an edge (plus the\n        edge-key if `G` is a multi-graph), which returns `True` if the edge\n        should appear in the view.\n\n    Returns\n    -------\n    graph : networkx.Graph\n        A read-only graph view of the input graph.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(6)\n\n    Filter functions operate on the node, and return `True` if the node should\n    appear in the view:\n\n    >>> def filter_node(n1):\n    ...     return n1 != 5\n    ...\n    >>> view = nx.subgraph_view(G, filter_node=filter_node)\n    >>> view.nodes()\n    NodeView((0, 1, 2, 3, 4))\n\n    We can use a closure pattern to filter graph elements based on additional\n    data --- for example, filtering on edge data attached to the graph:\n\n    >>> G[3][4]["cross_me"] = False\n    >>> def filter_edge(n1, n2):\n    ...     return G[n1][n2].get("cross_me", True)\n    ...\n    >>> view = nx.subgraph_view(G, filter_edge=filter_edge)\n    >>> view.edges()\n    EdgeView([(0, 1), (1, 2), (2, 3), (4, 5)])\n\n    >>> view = nx.subgraph_view(G, filter_node=filter_node, filter_edge=filter_edge,)\n    >>> view.nodes()\n    NodeView((0, 1, 2, 3, 4))\n    >>> view.edges()\n    EdgeView([(0, 1), (1, 2), (2, 3)])\n    '
    newG = nx.freeze(G.__class__())
    newG._NODE_OK = filter_node
    newG._EDGE_OK = filter_edge
    newG._graph = G
    newG.graph = G.graph
    newG._node = FilterAtlas(G._node, filter_node)
    if G.is_multigraph():
        Adj = FilterMultiAdjacency

        def reverse_edge(u, v, k=None):
            if False:
                for i in range(10):
                    print('nop')
            return filter_edge(v, u, k)
    else:
        Adj = FilterAdjacency

        def reverse_edge(u, v, k=None):
            if False:
                return 10
            return filter_edge(v, u)
    if G.is_directed():
        newG._succ = Adj(G._succ, filter_node, filter_edge)
        newG._pred = Adj(G._pred, filter_node, reverse_edge)
    else:
        newG._adj = Adj(G._adj, filter_node, filter_edge)
    return newG

@not_implemented_for('undirected')
def reverse_view(G):
    if False:
        while True:
            i = 10
    'View of `G` with edge directions reversed\n\n    `reverse_view` returns a read-only view of the input graph where\n    edge directions are reversed.\n\n    Identical to digraph.reverse(copy=False)\n\n    Parameters\n    ----------\n    G : networkx.DiGraph\n\n    Returns\n    -------\n    graph : networkx.DiGraph\n\n    Examples\n    --------\n    >>> G = nx.DiGraph()\n    >>> G.add_edge(1, 2)\n    >>> G.add_edge(2, 3)\n    >>> G.edges()\n    OutEdgeView([(1, 2), (2, 3)])\n\n    >>> view = nx.reverse_view(G)\n    >>> view.edges()\n    OutEdgeView([(2, 1), (3, 2)])\n    '
    newG = generic_graph_view(G)
    (newG._succ, newG._pred) = (G._pred, G._succ)
    return newG