import random
import time
import networkx as nx
from networkx.algorithms.isomorphism.tree_isomorphism import rooted_tree_isomorphism, tree_isomorphism
from networkx.classes.function import is_directed

def check_isomorphism(t1, t2, isomorphism):
    if False:
        i = 10
        return i + 15
    mapping = {v2: v1 for (v1, v2) in isomorphism}
    d1 = is_directed(t1)
    d2 = is_directed(t2)
    assert d1 == d2
    edges_1 = []
    for (u, v) in t1.edges():
        if d1:
            edges_1.append((u, v))
        elif u < v:
            edges_1.append((u, v))
        else:
            edges_1.append((v, u))
    edges_2 = []
    for (u, v) in t2.edges():
        u = mapping[u]
        v = mapping[v]
        if d2:
            edges_2.append((u, v))
        elif u < v:
            edges_2.append((u, v))
        else:
            edges_2.append((v, u))
    return sorted(edges_1) == sorted(edges_2)

def test_hardcoded():
    if False:
        i = 10
        return i + 15
    print('hardcoded test')
    edges_1 = [('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'e'), ('b', 'f'), ('e', 'j'), ('e', 'k'), ('c', 'g'), ('c', 'h'), ('g', 'm'), ('d', 'i'), ('f', 'l')]
    edges_2 = [('v', 'y'), ('v', 'z'), ('u', 'x'), ('q', 'u'), ('q', 'v'), ('p', 't'), ('n', 'p'), ('n', 'q'), ('n', 'o'), ('o', 'r'), ('o', 's'), ('s', 'w')]
    isomorphism1 = [('a', 'n'), ('b', 'q'), ('c', 'o'), ('d', 'p'), ('e', 'v'), ('f', 'u'), ('g', 's'), ('h', 'r'), ('i', 't'), ('j', 'y'), ('k', 'z'), ('l', 'x'), ('m', 'w')]
    isomorphism2 = [('a', 'n'), ('b', 'q'), ('c', 'o'), ('d', 'p'), ('e', 'v'), ('f', 'u'), ('g', 's'), ('h', 'r'), ('i', 't'), ('j', 'z'), ('k', 'y'), ('l', 'x'), ('m', 'w')]
    t1 = nx.Graph()
    t1.add_edges_from(edges_1)
    root1 = 'a'
    t2 = nx.Graph()
    t2.add_edges_from(edges_2)
    root2 = 'n'
    isomorphism = sorted(rooted_tree_isomorphism(t1, root1, t2, root2))
    assert isomorphism in (isomorphism1, isomorphism2)
    assert check_isomorphism(t1, t2, isomorphism)
    t1 = nx.DiGraph()
    t1.add_edges_from(edges_1)
    root1 = 'a'
    t2 = nx.DiGraph()
    t2.add_edges_from(edges_2)
    root2 = 'n'
    isomorphism = sorted(rooted_tree_isomorphism(t1, root1, t2, root2))
    assert isomorphism in (isomorphism1, isomorphism2)
    assert check_isomorphism(t1, t2, isomorphism)

def random_swap(t):
    if False:
        i = 10
        return i + 15
    (a, b) = t
    if random.randint(0, 1) == 1:
        return (a, b)
    else:
        return (b, a)

def positive_single_tree(t1):
    if False:
        for i in range(10):
            print('nop')
    assert nx.is_tree(t1)
    nodes1 = list(t1.nodes())
    nodes2 = nodes1.copy()
    random.shuffle(nodes2)
    someisomorphism = list(zip(nodes1, nodes2))
    map1to2 = dict(someisomorphism)
    edges2 = [random_swap((map1to2[u], map1to2[v])) for (u, v) in t1.edges()]
    random.shuffle(edges2)
    t2 = nx.Graph()
    t2.add_edges_from(edges2)
    isomorphism = tree_isomorphism(t1, t2)
    assert len(isomorphism) > 0
    assert check_isomorphism(t1, t2, isomorphism)

def test_positive(maxk=14):
    if False:
        return 10
    print('positive test')
    for k in range(2, maxk + 1):
        start_time = time.time()
        trial = 0
        for t in nx.nonisomorphic_trees(k):
            positive_single_tree(t)
            trial += 1
        print(k, trial, time.time() - start_time)

def test_trivial():
    if False:
        for i in range(10):
            print('nop')
    print('trivial test')
    t1 = nx.Graph()
    t1.add_node('a')
    root1 = 'a'
    t2 = nx.Graph()
    t2.add_node('n')
    root2 = 'n'
    isomorphism = rooted_tree_isomorphism(t1, root1, t2, root2)
    assert isomorphism == [('a', 'n')]
    assert check_isomorphism(t1, t2, isomorphism)

def test_trivial_2():
    if False:
        print('Hello World!')
    print('trivial test 2')
    edges_1 = [('a', 'b'), ('a', 'c')]
    edges_2 = [('v', 'y')]
    t1 = nx.Graph()
    t1.add_edges_from(edges_1)
    t2 = nx.Graph()
    t2.add_edges_from(edges_2)
    isomorphism = tree_isomorphism(t1, t2)
    assert isomorphism == []

def test_negative(maxk=11):
    if False:
        print('Hello World!')
    print('negative test')
    for k in range(4, maxk + 1):
        test_trees = list(nx.nonisomorphic_trees(k))
        start_time = time.time()
        trial = 0
        for i in range(len(test_trees) - 1):
            for j in range(i + 1, len(test_trees)):
                trial += 1
                assert tree_isomorphism(test_trees[i], test_trees[j]) == []
        print(k, trial, time.time() - start_time)