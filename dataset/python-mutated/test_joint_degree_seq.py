import time
from networkx.algorithms.assortativity import degree_mixing_dict
from networkx.generators import gnm_random_graph, powerlaw_cluster_graph
from networkx.generators.joint_degree_seq import directed_joint_degree_graph, is_valid_directed_joint_degree, is_valid_joint_degree, joint_degree_graph

def test_is_valid_joint_degree():
    if False:
        i = 10
        return i + 15
    'Tests for conditions that invalidate a joint degree dict'
    joint_degrees = {1: {4: 1}, 2: {2: 2, 3: 2, 4: 2}, 3: {2: 2, 4: 1}, 4: {1: 1, 2: 2, 3: 1}}
    assert is_valid_joint_degree(joint_degrees)
    joint_degrees_1 = {1: {4: 1.5}, 2: {2: 2, 3: 2, 4: 2}, 3: {2: 2, 4: 1}, 4: {1: 1.5, 2: 2, 3: 1}}
    assert not is_valid_joint_degree(joint_degrees_1)
    joint_degrees_2 = {1: {4: 1}, 2: {2: 2, 3: 2, 4: 3}, 3: {2: 2, 4: 1}, 4: {1: 1, 2: 3, 3: 1}}
    assert not is_valid_joint_degree(joint_degrees_2)
    joint_degrees_3 = {1: {4: 2}, 2: {2: 2, 3: 2, 4: 2}, 3: {2: 2, 4: 1}, 4: {1: 2, 2: 2, 3: 1}}
    assert not is_valid_joint_degree(joint_degrees_3)
    joint_degrees_5 = {1: {1: 9}}
    assert not is_valid_joint_degree(joint_degrees_5)

def test_joint_degree_graph(ntimes=10):
    if False:
        print('Hello World!')
    for _ in range(ntimes):
        seed = int(time.time())
        (n, m, p) = (20, 10, 1)
        g = powerlaw_cluster_graph(n, m, p, seed=seed)
        joint_degrees_g = degree_mixing_dict(g, normalized=False)
        G = joint_degree_graph(joint_degrees_g)
        joint_degrees_G = degree_mixing_dict(G, normalized=False)
        assert joint_degrees_g == joint_degrees_G

def test_is_valid_directed_joint_degree():
    if False:
        while True:
            i = 10
    in_degrees = [0, 1, 1, 2]
    out_degrees = [1, 1, 1, 1]
    nkk = {1: {1: 2, 2: 2}}
    assert is_valid_directed_joint_degree(in_degrees, out_degrees, nkk)
    nkk = {1: {1: 1.5, 2: 2.5}}
    assert not is_valid_directed_joint_degree(in_degrees, out_degrees, nkk)
    nkk = {1: {1: 2, 2: 1}}
    assert not is_valid_directed_joint_degree(in_degrees, out_degrees, nkk)
    out_degrees = [1, 1, 1]
    nkk = {1: {1: 2, 2: 2}}
    assert not is_valid_directed_joint_degree(in_degrees, out_degrees, nkk)
    in_degrees = [0, 1, 2]
    assert not is_valid_directed_joint_degree(in_degrees, out_degrees, nkk)

def test_directed_joint_degree_graph(n=15, m=100, ntimes=1000):
    if False:
        return 10
    for _ in range(ntimes):
        g = gnm_random_graph(n, m, None, directed=True)
        in_degrees = list(dict(g.in_degree()).values())
        out_degrees = list(dict(g.out_degree()).values())
        nkk = degree_mixing_dict(g)
        G = directed_joint_degree_graph(in_degrees, out_degrees, nkk)
        assert in_degrees == list(dict(G.in_degree()).values())
        assert out_degrees == list(dict(G.out_degree()).values())
        assert nkk == degree_mixing_dict(G)