"""Functions for computing communities based on centrality notions."""
import networkx as nx
__all__ = ['girvan_newman']

@nx._dispatch(preserve_edge_attrs='most_valuable_edge')
def girvan_newman(G, most_valuable_edge=None):
    if False:
        i = 10
        return i + 15
    'Finds communities in a graph using the Girvan–Newman method.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    most_valuable_edge : function\n        Function that takes a graph as input and outputs an edge. The\n        edge returned by this function will be recomputed and removed at\n        each iteration of the algorithm.\n\n        If not specified, the edge with the highest\n        :func:`networkx.edge_betweenness_centrality` will be used.\n\n    Returns\n    -------\n    iterator\n        Iterator over tuples of sets of nodes in `G`. Each set of node\n        is a community, each tuple is a sequence of communities at a\n        particular level of the algorithm.\n\n    Examples\n    --------\n    To get the first pair of communities::\n\n        >>> G = nx.path_graph(10)\n        >>> comp = nx.community.girvan_newman(G)\n        >>> tuple(sorted(c) for c in next(comp))\n        ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])\n\n    To get only the first *k* tuples of communities, use\n    :func:`itertools.islice`::\n\n        >>> import itertools\n        >>> G = nx.path_graph(8)\n        >>> k = 2\n        >>> comp = nx.community.girvan_newman(G)\n        >>> for communities in itertools.islice(comp, k):\n        ...     print(tuple(sorted(c) for c in communities))\n        ...\n        ([0, 1, 2, 3], [4, 5, 6, 7])\n        ([0, 1], [2, 3], [4, 5, 6, 7])\n\n    To stop getting tuples of communities once the number of communities\n    is greater than *k*, use :func:`itertools.takewhile`::\n\n        >>> import itertools\n        >>> G = nx.path_graph(8)\n        >>> k = 4\n        >>> comp = nx.community.girvan_newman(G)\n        >>> limited = itertools.takewhile(lambda c: len(c) <= k, comp)\n        >>> for communities in limited:\n        ...     print(tuple(sorted(c) for c in communities))\n        ...\n        ([0, 1, 2, 3], [4, 5, 6, 7])\n        ([0, 1], [2, 3], [4, 5, 6, 7])\n        ([0, 1], [2, 3], [4, 5], [6, 7])\n\n    To just choose an edge to remove based on the weight::\n\n        >>> from operator import itemgetter\n        >>> G = nx.path_graph(10)\n        >>> edges = G.edges()\n        >>> nx.set_edge_attributes(G, {(u, v): v for u, v in edges}, "weight")\n        >>> def heaviest(G):\n        ...     u, v, w = max(G.edges(data="weight"), key=itemgetter(2))\n        ...     return (u, v)\n        ...\n        >>> comp = nx.community.girvan_newman(G, most_valuable_edge=heaviest)\n        >>> tuple(sorted(c) for c in next(comp))\n        ([0, 1, 2, 3, 4, 5, 6, 7, 8], [9])\n\n    To utilize edge weights when choosing an edge with, for example, the\n    highest betweenness centrality::\n\n        >>> from networkx import edge_betweenness_centrality as betweenness\n        >>> def most_central_edge(G):\n        ...     centrality = betweenness(G, weight="weight")\n        ...     return max(centrality, key=centrality.get)\n        ...\n        >>> G = nx.path_graph(10)\n        >>> comp = nx.community.girvan_newman(G, most_valuable_edge=most_central_edge)\n        >>> tuple(sorted(c) for c in next(comp))\n        ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])\n\n    To specify a different ranking algorithm for edges, use the\n    `most_valuable_edge` keyword argument::\n\n        >>> from networkx import edge_betweenness_centrality\n        >>> from random import random\n        >>> def most_central_edge(G):\n        ...     centrality = edge_betweenness_centrality(G)\n        ...     max_cent = max(centrality.values())\n        ...     # Scale the centrality values so they are between 0 and 1,\n        ...     # and add some random noise.\n        ...     centrality = {e: c / max_cent for e, c in centrality.items()}\n        ...     # Add some random noise.\n        ...     centrality = {e: c + random() for e, c in centrality.items()}\n        ...     return max(centrality, key=centrality.get)\n        ...\n        >>> G = nx.path_graph(10)\n        >>> comp = nx.community.girvan_newman(G, most_valuable_edge=most_central_edge)\n\n    Notes\n    -----\n    The Girvan–Newman algorithm detects communities by progressively\n    removing edges from the original graph. The algorithm removes the\n    "most valuable" edge, traditionally the edge with the highest\n    betweenness centrality, at each step. As the graph breaks down into\n    pieces, the tightly knit community structure is exposed and the\n    result can be depicted as a dendrogram.\n\n    '
    if G.number_of_edges() == 0:
        yield tuple(nx.connected_components(G))
        return
    if most_valuable_edge is None:

        def most_valuable_edge(G):
            if False:
                while True:
                    i = 10
            'Returns the edge with the highest betweenness centrality\n            in the graph `G`.\n\n            '
            betweenness = nx.edge_betweenness_centrality(G)
            return max(betweenness, key=betweenness.get)
    g = G.copy().to_undirected()
    g.remove_edges_from(nx.selfloop_edges(g))
    while g.number_of_edges() > 0:
        yield _without_most_central_edges(g, most_valuable_edge)

def _without_most_central_edges(G, most_valuable_edge):
    if False:
        i = 10
        return i + 15
    'Returns the connected components of the graph that results from\n    repeatedly removing the most "valuable" edge in the graph.\n\n    `G` must be a non-empty graph. This function modifies the graph `G`\n    in-place; that is, it removes edges on the graph `G`.\n\n    `most_valuable_edge` is a function that takes the graph `G` as input\n    (or a subgraph with one or more edges of `G` removed) and returns an\n    edge. That edge will be removed and this process will be repeated\n    until the number of connected components in the graph increases.\n\n    '
    original_num_components = nx.number_connected_components(G)
    num_new_components = original_num_components
    while num_new_components <= original_num_components:
        edge = most_valuable_edge(G)
        G.remove_edge(*edge)
        new_components = tuple(nx.connected_components(G))
        num_new_components = len(new_components)
    return new_components