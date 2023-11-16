"""
Algorithm for testing d-separation in DAGs.

*d-separation* is a test for conditional independence in probability
distributions that can be factorized using DAGs.  It is a purely
graphical test that uses the underlying graph and makes no reference
to the actual distribution parameters.  See [1]_ for a formal
definition.

The implementation is based on the conceptually simple linear time
algorithm presented in [2]_.  Refer to [3]_, [4]_ for a couple of
alternative algorithms.

Here, we provide a brief overview of d-separation and related concepts that
are relevant for understanding it:

Blocking paths
--------------

Before we overview, we introduce the following terminology to describe paths:

- "open" path: A path between two nodes that can be traversed
- "blocked" path: A path between two nodes that cannot be traversed

A **collider** is a triplet of nodes along a path that is like the following:
``... u -> c <- v ...``), where 'c' is a common successor of ``u`` and ``v``. A path
through a collider is considered "blocked". When
a node that is a collider, or a descendant of a collider is included in
the d-separating set, then the path through that collider node is "open". If the
path through the collider node is open, then we will call this node an open collider.

The d-separation set blocks the paths between ``u`` and ``v``. If you include colliders,
or their descendant nodes in the d-separation set, then those colliders will open up,
enabling a path to be traversed if it is not blocked some other way.

Illustration of D-separation with examples
------------------------------------------

For a pair of two nodes, ``u`` and ``v``, all paths are considered open if
there is a path between ``u`` and ``v`` that is not blocked. That means, there is an open
path between ``u`` and ``v`` that does not encounter a collider, or a variable in the
d-separating set.

For example, if the d-separating set is the empty set, then the following paths are
unblocked between ``u`` and ``v``:

- u <- z -> v
- u -> w -> ... -> z -> v

If for example, 'z' is in the d-separating set, then 'z' blocks those paths
between ``u`` and ``v``.

Colliders block a path by default if they and their descendants are not included
in the d-separating set. An example of a path that is blocked when the d-separating
set is empty is:

- u -> w -> ... -> z <- v

because 'z' is a collider in this path and 'z' is not in the d-separating set. However,
if 'z' or a descendant of 'z' is included in the d-separating set, then the path through
the collider at 'z' (... -> z <- ...) is now "open". 

D-separation is concerned with blocking all paths between u and v. Therefore, a
d-separating set between ``u`` and ``v`` is one where all paths are blocked.

D-separation and its applications in probability
------------------------------------------------

D-separation is commonly used in probabilistic graphical models. D-separation
connects the idea of probabilistic "dependence" with separation in a graph. If
one assumes the causal Markov condition [5]_, then d-separation implies conditional
independence in probability distributions.

Examples
--------

>>>
>>> # HMM graph with five states and observation nodes
... g = nx.DiGraph()
>>> g.add_edges_from(
...     [
...         ("S1", "S2"),
...         ("S2", "S3"),
...         ("S3", "S4"),
...         ("S4", "S5"),
...         ("S1", "O1"),
...         ("S2", "O2"),
...         ("S3", "O3"),
...         ("S4", "O4"),
...         ("S5", "O5"),
...     ]
... )
>>>
>>> # states/obs before 'S3' are d-separated from states/obs after 'S3'
... nx.d_separated(g, {"S1", "S2", "O1", "O2"}, {"S4", "S5", "O4", "O5"}, {"S3"})
True


References
----------

.. [1] Pearl, J.  (2009).  Causality.  Cambridge: Cambridge University Press.

.. [2] Darwiche, A.  (2009).  Modeling and reasoning with Bayesian networks. 
   Cambridge: Cambridge University Press.

.. [3] Shachter, R.  D.  (1998).
   Bayes-ball: rational pastime (for determining irrelevance and requisite
   information in belief networks and influence diagrams).
   In , Proceedings of the Fourteenth Conference on Uncertainty in Artificial
   Intelligence (pp.  480–487).
   San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.

.. [4] Koller, D., & Friedman, N. (2009).
   Probabilistic graphical models: principles and techniques. The MIT Press.

.. [5] https://en.wikipedia.org/wiki/Causal_Markov_condition

"""
from collections import deque
import networkx as nx
from networkx.utils import UnionFind, not_implemented_for
__all__ = ['d_separated', 'minimal_d_separator', 'is_minimal_d_separator']

@not_implemented_for('undirected')
@nx._dispatch
def d_separated(G, x, y, z):
    if False:
        while True:
            i = 10
    '\n    Return whether node sets ``x`` and ``y`` are d-separated by ``z``.\n\n    Parameters\n    ----------\n    G : graph\n        A NetworkX DAG.\n\n    x : set\n        First set of nodes in ``G``.\n\n    y : set\n        Second set of nodes in ``G``.\n\n    z : set\n        Set of conditioning nodes in ``G``. Can be empty set.\n\n    Returns\n    -------\n    b : bool\n        A boolean that is true if ``x`` is d-separated from ``y`` given ``z`` in ``G``.\n\n    Raises\n    ------\n    NetworkXError\n        The *d-separation* test is commonly used with directed\n        graphical models which are acyclic.  Accordingly, the algorithm\n        raises a :exc:`NetworkXError` if the input graph is not a DAG.\n\n    NodeNotFound\n        If any of the input nodes are not found in the graph,\n        a :exc:`NodeNotFound` exception is raised.\n\n    Notes\n    -----\n    A d-separating set in a DAG is a set of nodes that\n    blocks all paths between the two sets. Nodes in `z`\n    block a path if they are part of the path and are not a collider,\n    or a descendant of a collider. A collider structure along a path\n    is ``... -> c <- ...`` where ``c`` is the collider node.\n\n    https://en.wikipedia.org/wiki/Bayesian_network#d-separation\n    '
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError('graph should be directed acyclic')
    union_xyz = x.union(y).union(z)
    if any((n not in G.nodes for n in union_xyz)):
        raise nx.NodeNotFound('one or more specified nodes not found in the graph')
    G_copy = G.copy()
    leaves = deque([n for n in G_copy.nodes if G_copy.out_degree[n] == 0])
    while len(leaves) > 0:
        leaf = leaves.popleft()
        if leaf not in union_xyz:
            for p in G_copy.predecessors(leaf):
                if G_copy.out_degree[p] == 1:
                    leaves.append(p)
            G_copy.remove_node(leaf)
    edges_to_remove = list(G_copy.out_edges(z))
    G_copy.remove_edges_from(edges_to_remove)
    disjoint_set = UnionFind(G_copy.nodes())
    for component in nx.weakly_connected_components(G_copy):
        disjoint_set.union(*component)
    disjoint_set.union(*x)
    disjoint_set.union(*y)
    if x and y and (disjoint_set[next(iter(x))] == disjoint_set[next(iter(y))]):
        return False
    else:
        return True

@not_implemented_for('undirected')
@nx._dispatch
def minimal_d_separator(G, u, v):
    if False:
        print('Hello World!')
    'Compute a minimal d-separating set between \'u\' and \'v\'.\n\n    A d-separating set in a DAG is a set of nodes that blocks all paths\n    between the two nodes, \'u\' and \'v\'. This function\n    constructs a d-separating set that is "minimal", meaning it is the smallest\n    d-separating set for \'u\' and \'v\'. This is not necessarily\n    unique. For more details, see Notes.\n\n    Parameters\n    ----------\n    G : graph\n        A networkx DAG.\n    u : node\n        A node in the graph, G.\n    v : node\n        A node in the graph, G.\n\n    Raises\n    ------\n    NetworkXError\n        Raises a :exc:`NetworkXError` if the input graph is not a DAG.\n\n    NodeNotFound\n        If any of the input nodes are not found in the graph,\n        a :exc:`NodeNotFound` exception is raised.\n\n    References\n    ----------\n    .. [1] Tian, J., & Paz, A. (1998). Finding Minimal D-separators.\n\n    Notes\n    -----\n    This function only finds ``a`` minimal d-separator. It does not guarantee\n    uniqueness, since in a DAG there may be more than one minimal d-separator\n    between two nodes. Moreover, this only checks for minimal separators\n    between two nodes, not two sets. Finding minimal d-separators between\n    two sets of nodes is not supported.\n\n    Uses the algorithm presented in [1]_. The complexity of the algorithm\n    is :math:`O(|E_{An}^m|)`, where :math:`|E_{An}^m|` stands for the\n    number of edges in the moralized graph of the sub-graph consisting\n    of only the ancestors of \'u\' and \'v\'. For full details, see [1]_.\n\n    The algorithm works by constructing the moral graph consisting of just\n    the ancestors of `u` and `v`. Then it constructs a candidate for\n    a separating set  ``Z\'`` from the predecessors of `u` and `v`.\n    Then BFS is run starting from `u` and marking nodes\n    found from ``Z\'`` and calling those nodes ``Z\'\'``.\n    Then BFS is run again starting from `v` and marking nodes if they are\n    present in ``Z\'\'``. Those marked nodes are the returned minimal\n    d-separating set.\n\n    https://en.wikipedia.org/wiki/Bayesian_network#d-separation\n    '
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError('graph should be directed acyclic')
    union_uv = {u, v}
    if any((n not in G.nodes for n in union_uv)):
        raise nx.NodeNotFound('one or more specified nodes not found in the graph')
    x_anc = nx.ancestors(G, u)
    y_anc = nx.ancestors(G, v)
    D_anc_xy = x_anc.union(y_anc)
    D_anc_xy.update((u, v))
    moral_G = nx.moral_graph(G.subgraph(D_anc_xy))
    Z_prime = set(G.predecessors(u)).union(set(G.predecessors(v)))
    Z_dprime = _bfs_with_marks(moral_G, u, Z_prime)
    Z = _bfs_with_marks(moral_G, v, Z_dprime)
    return Z

@not_implemented_for('undirected')
@nx._dispatch
def is_minimal_d_separator(G, u, v, z):
    if False:
        print('Hello World!')
    'Determine if a d-separating set is minimal.\n\n    A d-separating set, `z`, in a DAG is a set of nodes that blocks\n    all paths between the two nodes, `u` and `v`. This function\n    verifies that a set is "minimal", meaning there is no smaller\n    d-separating set between the two nodes.\n\n    Note: This function checks whether `z` is a d-separator AND is minimal.\n    One can use the function `d_separated` to only check if `z` is a d-separator.\n    See examples below.\n\n    Parameters\n    ----------\n    G : nx.DiGraph\n        The graph.\n    u : node\n        A node in the graph.\n    v : node\n        A node in the graph.\n    z : Set of nodes\n        The set of nodes to check if it is a minimal d-separating set.\n        The function :func:`d_separated` is called inside this function\n        to verify that `z` is in fact a d-separator.\n\n    Returns\n    -------\n    bool\n        Whether or not the set `z` is a d-separator and is also minimal.\n\n    Examples\n    --------\n    >>> G = nx.path_graph([0, 1, 2, 3], create_using=nx.DiGraph)\n    >>> G.add_node(4)\n    >>> nx.is_minimal_d_separator(G, 0, 2, {1})\n    True\n    >>> # since {1} is the minimal d-separator, {1, 3, 4} is not minimal\n    >>> nx.is_minimal_d_separator(G, 0, 2, {1, 3, 4})\n    False\n    >>> # alternatively, if we only want to check that {1, 3, 4} is a d-separator\n    >>> nx.d_separated(G, {0}, {4}, {1, 3, 4})\n    True\n\n    Raises\n    ------\n    NetworkXError\n        Raises a :exc:`NetworkXError` if the input graph is not a DAG.\n\n    NodeNotFound\n        If any of the input nodes are not found in the graph,\n        a :exc:`NodeNotFound` exception is raised.\n\n    References\n    ----------\n    .. [1] Tian, J., & Paz, A. (1998). Finding Minimal D-separators.\n\n    Notes\n    -----\n    This function only works on verifying a d-separating set is minimal\n    between two nodes. To verify that a d-separating set is minimal between\n    two sets of nodes is not supported.\n\n    Uses algorithm 2 presented in [1]_. The complexity of the algorithm\n    is :math:`O(|E_{An}^m|)`, where :math:`|E_{An}^m|` stands for the\n    number of edges in the moralized graph of the sub-graph consisting\n    of only the ancestors of ``u`` and ``v``.\n\n    The algorithm works by constructing the moral graph consisting of just\n    the ancestors of `u` and `v`. First, it performs BFS on the moral graph\n    starting from `u` and marking any nodes it encounters that are part of\n    the separating set, `z`. If a node is marked, then it does not continue\n    along that path. In the second stage, BFS with markings is repeated on the\n    moral graph starting from `v`. If at any stage, any node in `z` is\n    not marked, then `z` is considered not minimal. If the end of the algorithm\n    is reached, then `z` is minimal.\n\n    For full details, see [1]_.\n\n    https://en.wikipedia.org/wiki/Bayesian_network#d-separation\n    '
    if not nx.d_separated(G, {u}, {v}, z):
        return False
    x_anc = nx.ancestors(G, u)
    y_anc = nx.ancestors(G, v)
    xy_anc = x_anc.union(y_anc)
    if any((node not in xy_anc for node in z)):
        return False
    D_anc_xy = x_anc.union(y_anc)
    D_anc_xy.update((u, v))
    moral_G = nx.moral_graph(G.subgraph(D_anc_xy))
    marks = _bfs_with_marks(moral_G, u, z)
    if any((node not in marks for node in z)):
        return False
    marks = _bfs_with_marks(moral_G, v, z)
    if any((node not in marks for node in z)):
        return False
    return True

@not_implemented_for('directed')
def _bfs_with_marks(G, start_node, check_set):
    if False:
        print('Hello World!')
    'Breadth-first-search with markings.\n\n    Performs BFS starting from ``start_node`` and whenever a node\n    inside ``check_set`` is met, it is "marked". Once a node is marked,\n    BFS does not continue along that path. The resulting marked nodes\n    are returned.\n\n    Parameters\n    ----------\n    G : nx.Graph\n        An undirected graph.\n    start_node : node\n        The start of the BFS.\n    check_set : set\n        The set of nodes to check against.\n\n    Returns\n    -------\n    marked : set\n        A set of nodes that were marked.\n    '
    visited = {}
    marked = set()
    queue = []
    visited[start_node] = None
    queue.append(start_node)
    while queue:
        m = queue.pop(0)
        for nbr in G.neighbors(m):
            if nbr not in visited:
                visited[nbr] = None
                if nbr in check_set:
                    marked.add(nbr)
                else:
                    queue.append(nbr)
    return marked