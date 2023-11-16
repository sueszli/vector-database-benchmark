"""Functions for encoding and decoding trees.

Since a tree is a highly restricted form of graph, it can be represented
concisely in several ways. This module includes functions for encoding
and decoding trees in the form of nested tuples and Prüfer
sequences. The former requires a rooted tree, whereas the latter can be
applied to unrooted trees. Furthermore, there is a bijection from Prüfer
sequences to labeled trees.

"""
from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['from_nested_tuple', 'from_prufer_sequence', 'NotATree', 'to_nested_tuple', 'to_prufer_sequence']

class NotATree(nx.NetworkXException):
    """Raised when a function expects a tree (that is, a connected
    undirected graph with no cycles) but gets a non-tree graph as input
    instead.

    """

@not_implemented_for('directed')
@nx._dispatch(graphs='T')
def to_nested_tuple(T, root, canonical_form=False):
    if False:
        for i in range(10):
            print('nop')
    'Returns a nested tuple representation of the given tree.\n\n    The nested tuple representation of a tree is defined\n    recursively. The tree with one node and no edges is represented by\n    the empty tuple, ``()``. A tree with ``k`` subtrees is represented\n    by a tuple of length ``k`` in which each element is the nested tuple\n    representation of a subtree.\n\n    Parameters\n    ----------\n    T : NetworkX graph\n        An undirected graph object representing a tree.\n\n    root : node\n        The node in ``T`` to interpret as the root of the tree.\n\n    canonical_form : bool\n        If ``True``, each tuple is sorted so that the function returns\n        a canonical form for rooted trees. This means "lighter" subtrees\n        will appear as nested tuples before "heavier" subtrees. In this\n        way, each isomorphic rooted tree has the same nested tuple\n        representation.\n\n    Returns\n    -------\n    tuple\n        A nested tuple representation of the tree.\n\n    Notes\n    -----\n    This function is *not* the inverse of :func:`from_nested_tuple`; the\n    only guarantee is that the rooted trees are isomorphic.\n\n    See also\n    --------\n    from_nested_tuple\n    to_prufer_sequence\n\n    Examples\n    --------\n    The tree need not be a balanced binary tree::\n\n        >>> T = nx.Graph()\n        >>> T.add_edges_from([(0, 1), (0, 2), (0, 3)])\n        >>> T.add_edges_from([(1, 4), (1, 5)])\n        >>> T.add_edges_from([(3, 6), (3, 7)])\n        >>> root = 0\n        >>> nx.to_nested_tuple(T, root)\n        (((), ()), (), ((), ()))\n\n    Continuing the above example, if ``canonical_form`` is ``True``, the\n    nested tuples will be sorted::\n\n        >>> nx.to_nested_tuple(T, root, canonical_form=True)\n        ((), ((), ()), ((), ()))\n\n    Even the path graph can be interpreted as a tree::\n\n        >>> T = nx.path_graph(4)\n        >>> root = 0\n        >>> nx.to_nested_tuple(T, root)\n        ((((),),),)\n\n    '

    def _make_tuple(T, root, _parent):
        if False:
            return 10
        'Recursively compute the nested tuple representation of the\n        given rooted tree.\n\n        ``_parent`` is the parent node of ``root`` in the supertree in\n        which ``T`` is a subtree, or ``None`` if ``root`` is the root of\n        the supertree. This argument is used to determine which\n        neighbors of ``root`` are children and which is the parent.\n\n        '
        children = set(T[root]) - {_parent}
        if len(children) == 0:
            return ()
        nested = (_make_tuple(T, v, root) for v in children)
        if canonical_form:
            nested = sorted(nested)
        return tuple(nested)
    if not nx.is_tree(T):
        raise nx.NotATree('provided graph is not a tree')
    if root not in T:
        raise nx.NodeNotFound(f'Graph {T} contains no node {root}')
    return _make_tuple(T, root, None)

@nx._dispatch(graphs=None)
def from_nested_tuple(sequence, sensible_relabeling=False):
    if False:
        i = 10
        return i + 15
    'Returns the rooted tree corresponding to the given nested tuple.\n\n    The nested tuple representation of a tree is defined\n    recursively. The tree with one node and no edges is represented by\n    the empty tuple, ``()``. A tree with ``k`` subtrees is represented\n    by a tuple of length ``k`` in which each element is the nested tuple\n    representation of a subtree.\n\n    Parameters\n    ----------\n    sequence : tuple\n        A nested tuple representing a rooted tree.\n\n    sensible_relabeling : bool\n        Whether to relabel the nodes of the tree so that nodes are\n        labeled in increasing order according to their breadth-first\n        search order from the root node.\n\n    Returns\n    -------\n    NetworkX graph\n        The tree corresponding to the given nested tuple, whose root\n        node is node 0. If ``sensible_labeling`` is ``True``, nodes will\n        be labeled in breadth-first search order starting from the root\n        node.\n\n    Notes\n    -----\n    This function is *not* the inverse of :func:`to_nested_tuple`; the\n    only guarantee is that the rooted trees are isomorphic.\n\n    See also\n    --------\n    to_nested_tuple\n    from_prufer_sequence\n\n    Examples\n    --------\n    Sensible relabeling ensures that the nodes are labeled from the root\n    starting at 0::\n\n        >>> balanced = (((), ()), ((), ()))\n        >>> T = nx.from_nested_tuple(balanced, sensible_relabeling=True)\n        >>> edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]\n        >>> all((u, v) in T.edges() or (v, u) in T.edges() for (u, v) in edges)\n        True\n\n    '

    def _make_tree(sequence):
        if False:
            while True:
                i = 10
        'Recursively creates a tree from the given sequence of nested\n        tuples.\n\n        This function employs the :func:`~networkx.tree.join` function\n        to recursively join subtrees into a larger tree.\n\n        '
        if len(sequence) == 0:
            return nx.empty_graph(1)
        return nx.tree.join_trees([(_make_tree(child), 0) for child in sequence])
    T = _make_tree(sequence)
    if sensible_relabeling:
        bfs_nodes = chain([0], (v for (u, v) in nx.bfs_edges(T, 0)))
        labels = {v: i for (i, v) in enumerate(bfs_nodes)}
        T = nx.relabel_nodes(T, labels)
    return T

@not_implemented_for('directed')
@nx._dispatch(graphs='T')
def to_prufer_sequence(T):
    if False:
        return 10
    'Returns the Prüfer sequence of the given tree.\n\n    A *Prüfer sequence* is a list of *n* - 2 numbers between 0 and\n    *n* - 1, inclusive. The tree corresponding to a given Prüfer\n    sequence can be recovered by repeatedly joining a node in the\n    sequence with a node with the smallest potential degree according to\n    the sequence.\n\n    Parameters\n    ----------\n    T : NetworkX graph\n        An undirected graph object representing a tree.\n\n    Returns\n    -------\n    list\n        The Prüfer sequence of the given tree.\n\n    Raises\n    ------\n    NetworkXPointlessConcept\n        If the number of nodes in `T` is less than two.\n\n    NotATree\n        If `T` is not a tree.\n\n    KeyError\n        If the set of nodes in `T` is not {0, …, *n* - 1}.\n\n    Notes\n    -----\n    There is a bijection from labeled trees to Prüfer sequences. This\n    function is the inverse of the :func:`from_prufer_sequence`\n    function.\n\n    Sometimes Prüfer sequences use nodes labeled from 1 to *n* instead\n    of from 0 to *n* - 1. This function requires nodes to be labeled in\n    the latter form. You can use :func:`~networkx.relabel_nodes` to\n    relabel the nodes of your tree to the appropriate format.\n\n    This implementation is from [1]_ and has a running time of\n    $O(n)$.\n\n    See also\n    --------\n    to_nested_tuple\n    from_prufer_sequence\n\n    References\n    ----------\n    .. [1] Wang, Xiaodong, Lei Wang, and Yingjie Wu.\n           "An optimal algorithm for Prufer codes."\n           *Journal of Software Engineering and Applications* 2.02 (2009): 111.\n           <https://doi.org/10.4236/jsea.2009.22016>\n\n    Examples\n    --------\n    There is a bijection between Prüfer sequences and labeled trees, so\n    this function is the inverse of the :func:`from_prufer_sequence`\n    function:\n\n    >>> edges = [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5)]\n    >>> tree = nx.Graph(edges)\n    >>> sequence = nx.to_prufer_sequence(tree)\n    >>> sequence\n    [3, 3, 3, 4]\n    >>> tree2 = nx.from_prufer_sequence(sequence)\n    >>> list(tree2.edges()) == edges\n    True\n\n    '
    n = len(T)
    if n < 2:
        msg = 'Prüfer sequence undefined for trees with fewer than two nodes'
        raise nx.NetworkXPointlessConcept(msg)
    if not nx.is_tree(T):
        raise nx.NotATree('provided graph is not a tree')
    if set(T) != set(range(n)):
        raise KeyError('tree must have node labels {0, ..., n - 1}')
    degree = dict(T.degree())

    def parents(u):
        if False:
            i = 10
            return i + 15
        return next((v for v in T[u] if degree[v] > 1))
    index = u = next((k for k in range(n) if degree[k] == 1))
    result = []
    for i in range(n - 2):
        v = parents(u)
        result.append(v)
        degree[v] -= 1
        if v < index and degree[v] == 1:
            u = v
        else:
            index = u = next((k for k in range(index + 1, n) if degree[k] == 1))
    return result

@nx._dispatch(graphs=None)
def from_prufer_sequence(sequence):
    if False:
        i = 10
        return i + 15
    'Returns the tree corresponding to the given Prüfer sequence.\n\n    A *Prüfer sequence* is a list of *n* - 2 numbers between 0 and\n    *n* - 1, inclusive. The tree corresponding to a given Prüfer\n    sequence can be recovered by repeatedly joining a node in the\n    sequence with a node with the smallest potential degree according to\n    the sequence.\n\n    Parameters\n    ----------\n    sequence : list\n        A Prüfer sequence, which is a list of *n* - 2 integers between\n        zero and *n* - 1, inclusive.\n\n    Returns\n    -------\n    NetworkX graph\n        The tree corresponding to the given Prüfer sequence.\n\n    Raises\n    ------\n    NetworkXError\n        If the Prüfer sequence is not valid.\n\n    Notes\n    -----\n    There is a bijection from labeled trees to Prüfer sequences. This\n    function is the inverse of the :func:`from_prufer_sequence` function.\n\n    Sometimes Prüfer sequences use nodes labeled from 1 to *n* instead\n    of from 0 to *n* - 1. This function requires nodes to be labeled in\n    the latter form. You can use :func:`networkx.relabel_nodes` to\n    relabel the nodes of your tree to the appropriate format.\n\n    This implementation is from [1]_ and has a running time of\n    $O(n)$.\n\n    References\n    ----------\n    .. [1] Wang, Xiaodong, Lei Wang, and Yingjie Wu.\n           "An optimal algorithm for Prufer codes."\n           *Journal of Software Engineering and Applications* 2.02 (2009): 111.\n           <https://doi.org/10.4236/jsea.2009.22016>\n\n    See also\n    --------\n    from_nested_tuple\n    to_prufer_sequence\n\n    Examples\n    --------\n    There is a bijection between Prüfer sequences and labeled trees, so\n    this function is the inverse of the :func:`to_prufer_sequence`\n    function:\n\n    >>> edges = [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5)]\n    >>> tree = nx.Graph(edges)\n    >>> sequence = nx.to_prufer_sequence(tree)\n    >>> sequence\n    [3, 3, 3, 4]\n    >>> tree2 = nx.from_prufer_sequence(sequence)\n    >>> list(tree2.edges()) == edges\n    True\n\n    '
    n = len(sequence) + 2
    degree = Counter(chain(sequence, range(n)))
    T = nx.empty_graph(n)
    not_orphaned = set()
    index = u = next((k for k in range(n) if degree[k] == 1))
    for v in sequence:
        if v < 0 or v > n - 1:
            raise nx.NetworkXError(f'Invalid Prufer sequence: Values must be between 0 and {n - 1}, got {v}')
        T.add_edge(u, v)
        not_orphaned.add(u)
        degree[v] -= 1
        if v < index and degree[v] == 1:
            u = v
        else:
            index = u = next((k for k in range(index + 1, n) if degree[k] == 1))
    orphans = set(T) - not_orphaned
    (u, v) = orphans
    T.add_edge(u, v)
    return T