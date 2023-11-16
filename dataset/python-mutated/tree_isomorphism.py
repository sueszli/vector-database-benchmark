"""
An algorithm for finding if two undirected trees are isomorphic,
and if so returns an isomorphism between the two sets of nodes.

This algorithm uses a routine to tell if two rooted trees (trees with a
specified root node) are isomorphic, which may be independently useful.

This implements an algorithm from:
The Design and Analysis of Computer Algorithms
by Aho, Hopcroft, and Ullman
Addison-Wesley Publishing 1974
Example 3.2 pp. 84-86.

A more understandable version of this algorithm is described in:
Homework Assignment 5
McGill University SOCS 308-250B, Winter 2002
by Matthew Suderman
http://crypto.cs.mcgill.ca/~crepeau/CS250/2004/HW5+.pdf
"""
import networkx as nx
from networkx.utils.decorators import not_implemented_for
__all__ = ['rooted_tree_isomorphism', 'tree_isomorphism']

@nx._dispatch(graphs={'t1': 0, 't2': 2})
def root_trees(t1, root1, t2, root2):
    if False:
        while True:
            i = 10
    'Create a single digraph dT of free trees t1 and t2\n    #   with roots root1 and root2 respectively\n    # rename the nodes with consecutive integers\n    # so that all nodes get a unique name between both trees\n\n    # our new "fake" root node is 0\n    # t1 is numbers from 1 ... n\n    # t2 is numbered from n+1 to 2n\n    '
    dT = nx.DiGraph()
    newroot1 = 1
    newroot2 = nx.number_of_nodes(t1) + 1
    namemap1 = {root1: newroot1}
    namemap2 = {root2: newroot2}
    dT.add_edge(0, namemap1[root1])
    dT.add_edge(0, namemap2[root2])
    for (i, (v1, v2)) in enumerate(nx.bfs_edges(t1, root1)):
        namemap1[v2] = i + namemap1[root1] + 1
        dT.add_edge(namemap1[v1], namemap1[v2])
    for (i, (v1, v2)) in enumerate(nx.bfs_edges(t2, root2)):
        namemap2[v2] = i + namemap2[root2] + 1
        dT.add_edge(namemap2[v1], namemap2[v2])
    namemap = {}
    for (old, new) in namemap1.items():
        namemap[new] = old
    for (old, new) in namemap2.items():
        namemap[new] = old
    return (dT, namemap, newroot1, newroot2)

@nx._dispatch
def assign_levels(G, root):
    if False:
        i = 10
        return i + 15
    level = {}
    level[root] = 0
    for (v1, v2) in nx.bfs_edges(G, root):
        level[v2] = level[v1] + 1
    return level

def group_by_levels(levels):
    if False:
        while True:
            i = 10
    L = {}
    for (n, lev) in levels.items():
        if lev not in L:
            L[lev] = []
        L[lev].append(n)
    return L

def generate_isomorphism(v, w, M, ordered_children):
    if False:
        print('Hello World!')
    assert v < w
    M.append((v, w))
    for (i, (x, y)) in enumerate(zip(ordered_children[v], ordered_children[w])):
        generate_isomorphism(x, y, M, ordered_children)

@nx._dispatch(graphs={'t1': 0, 't2': 2})
def rooted_tree_isomorphism(t1, root1, t2, root2):
    if False:
        return 10
    '\n    Given two rooted trees `t1` and `t2`,\n    with roots `root1` and `root2` respectively\n    this routine will determine if they are isomorphic.\n\n    These trees may be either directed or undirected,\n    but if they are directed, all edges should flow from the root.\n\n    It returns the isomorphism, a mapping of the nodes of `t1` onto the nodes\n    of `t2`, such that two trees are then identical.\n\n    Note that two trees may have more than one isomorphism, and this\n    routine just returns one valid mapping.\n\n    Parameters\n    ----------\n    `t1` :  NetworkX graph\n        One of the trees being compared\n\n    `root1` : a node of `t1` which is the root of the tree\n\n    `t2` : undirected NetworkX graph\n        The other tree being compared\n\n    `root2` : a node of `t2` which is the root of the tree\n\n    This is a subroutine used to implement `tree_isomorphism`, but will\n    be somewhat faster if you already have rooted trees.\n\n    Returns\n    -------\n    isomorphism : list\n        A list of pairs in which the left element is a node in `t1`\n        and the right element is a node in `t2`.  The pairs are in\n        arbitrary order.  If the nodes in one tree is mapped to the names in\n        the other, then trees will be identical. Note that an isomorphism\n        will not necessarily be unique.\n\n        If `t1` and `t2` are not isomorphic, then it returns the empty list.\n    '
    assert nx.is_tree(t1)
    assert nx.is_tree(t2)
    (dT, namemap, newroot1, newroot2) = root_trees(t1, root1, t2, root2)
    levels = assign_levels(dT, 0)
    h = max(levels.values())
    L = group_by_levels(levels)
    label = {v: 0 for v in dT}
    ordered_labels = {v: () for v in dT}
    ordered_children = {v: () for v in dT}
    for i in range(h - 1, 0, -1):
        for v in L[i]:
            if dT.out_degree(v) > 0:
                s = sorted(((label[u], u) for u in dT.successors(v)))
                (ordered_labels[v], ordered_children[v]) = list(zip(*s))
        forlabel = sorted(((ordered_labels[v], v) for v in L[i]))
        current = 0
        for (i, (ol, v)) in enumerate(forlabel):
            if i != 0 and ol != forlabel[i - 1][0]:
                current += 1
            label[v] = current
    isomorphism = []
    if label[newroot1] == 0 and label[newroot2] == 0:
        generate_isomorphism(newroot1, newroot2, isomorphism, ordered_children)
        isomorphism = [(namemap[u], namemap[v]) for (u, v) in isomorphism]
    return isomorphism

@not_implemented_for('directed', 'multigraph')
@nx._dispatch(graphs={'t1': 0, 't2': 1})
def tree_isomorphism(t1, t2):
    if False:
        i = 10
        return i + 15
    '\n    Given two undirected (or free) trees `t1` and `t2`,\n    this routine will determine if they are isomorphic.\n    It returns the isomorphism, a mapping of the nodes of `t1` onto the nodes\n    of `t2`, such that two trees are then identical.\n\n    Note that two trees may have more than one isomorphism, and this\n    routine just returns one valid mapping.\n\n    Parameters\n    ----------\n    t1 : undirected NetworkX graph\n        One of the trees being compared\n\n    t2 : undirected NetworkX graph\n        The other tree being compared\n\n    Returns\n    -------\n    isomorphism : list\n        A list of pairs in which the left element is a node in `t1`\n        and the right element is a node in `t2`.  The pairs are in\n        arbitrary order.  If the nodes in one tree is mapped to the names in\n        the other, then trees will be identical. Note that an isomorphism\n        will not necessarily be unique.\n\n        If `t1` and `t2` are not isomorphic, then it returns the empty list.\n\n    Notes\n    -----\n    This runs in O(n*log(n)) time for trees with n nodes.\n    '
    assert nx.is_tree(t1)
    assert nx.is_tree(t2)
    if nx.number_of_nodes(t1) != nx.number_of_nodes(t2):
        return []
    degree_sequence1 = sorted((d for (n, d) in t1.degree()))
    degree_sequence2 = sorted((d for (n, d) in t2.degree()))
    if degree_sequence1 != degree_sequence2:
        return []
    center1 = nx.center(t1)
    center2 = nx.center(t2)
    if len(center1) != len(center2):
        return []
    if len(center1) == 1:
        return rooted_tree_isomorphism(t1, center1[0], t2, center2[0])
    attempts = rooted_tree_isomorphism(t1, center1[0], t2, center2[0])
    if len(attempts) > 0:
        return attempts
    return rooted_tree_isomorphism(t1, center1[0], t2, center2[1])