"""Generators for cographs

A cograph is a graph containing no path on four vertices.
Cographs or $P_4$-free graphs can be obtained from a single vertex
by disjoint union and complementation operations.

References
----------
.. [0] D.G. Corneil, H. Lerchs, L.Stewart Burlingham,
    "Complement reducible graphs",
    Discrete Applied Mathematics, Volume 3, Issue 3, 1981, Pages 163-174,
    ISSN 0166-218X.
"""
import networkx as nx
from networkx.utils import py_random_state
__all__ = ['random_cograph']

@py_random_state(1)
@nx._dispatch(graphs=None)
def random_cograph(n, seed=None):
    if False:
        while True:
            i = 10
    'Returns a random cograph with $2 ^ n$ nodes.\n\n    A cograph is a graph containing no path on four vertices.\n    Cographs or $P_4$-free graphs can be obtained from a single vertex\n    by disjoint union and complementation operations.\n\n    This generator starts off from a single vertex and performs disjoint\n    union and full join operations on itself.\n    The decision on which operation will take place is random.\n\n    Parameters\n    ----------\n    n : int\n        The order of the cograph.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    G : A random graph containing no path on four vertices.\n\n    See Also\n    --------\n    full_join\n    union\n\n    References\n    ----------\n    .. [1] D.G. Corneil, H. Lerchs, L.Stewart Burlingham,\n       "Complement reducible graphs",\n       Discrete Applied Mathematics, Volume 3, Issue 3, 1981, Pages 163-174,\n       ISSN 0166-218X.\n    '
    R = nx.empty_graph(1)
    for i in range(n):
        RR = nx.relabel_nodes(R.copy(), lambda x: x + len(R))
        if seed.randint(0, 1) == 0:
            R = nx.full_join(R, RR)
        else:
            R = nx.disjoint_union(R, RR)
    return R