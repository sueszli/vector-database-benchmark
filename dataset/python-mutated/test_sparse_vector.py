from dlib import pair, make_sparse_vector, sparse_vector, sparse_vectors, sparse_vectorss
try:
    import cPickle as pickle
except ImportError:
    import pickle
from pytest import approx

def test_pair():
    if False:
        i = 10
        return i + 15
    p = pair(4, 0.9)
    assert p.first == 4
    assert p.second == 0.9
    p.first = 3
    p.second = 0.4
    assert p.first == 3
    assert p.second == 0.4
    assert str(p) == '3: 0.4'
    assert repr(p) == 'dlib.pair(3, 0.4)'
    deser = pickle.loads(pickle.dumps(p, 2))
    assert deser.first == p.first
    assert deser.second == p.second

def test_sparse_vector():
    if False:
        print('Hello World!')
    sv = sparse_vector()
    sv.append(pair(3, 0.1))
    sv.append(pair(3, 0.2))
    sv.append(pair(2, 0.3))
    sv.append(pair(1, 0.4))
    assert len(sv) == 4
    make_sparse_vector(sv)
    assert len(sv) == 3
    assert sv[0].first == 1
    assert sv[0].second == 0.4
    assert sv[1].first == 2
    assert sv[1].second == 0.3
    assert sv[2].first == 3
    assert sv[2].second == approx(0.3)
    assert str(sv) == '1: 0.4\n2: 0.3\n3: 0.3'
    assert repr(sv) == '< dlib.sparse_vector containing: \n1: 0.4\n2: 0.3\n3: 0.3 >'

def test_sparse_vectors():
    if False:
        print('Hello World!')
    svs = sparse_vectors()
    assert len(svs) == 0
    svs.resize(5)
    for sv in svs:
        assert len(sv) == 0
    svs.clear()
    assert len(svs) == 0
    svs.extend([sparse_vector([pair(1, 2), pair(3, 4)]), sparse_vector([pair(5, 6), pair(7, 8)])])
    assert len(svs) == 2
    assert svs[0][0].first == 1
    assert svs[0][0].second == 2
    assert svs[0][1].first == 3
    assert svs[0][1].second == 4
    assert svs[1][0].first == 5
    assert svs[1][0].second == 6
    assert svs[1][1].first == 7
    assert svs[1][1].second == 8
    deser = pickle.loads(pickle.dumps(svs, 2))
    assert deser == svs

def test_sparse_vectorss():
    if False:
        for i in range(10):
            print('nop')
    svss = sparse_vectorss()
    assert len(svss) == 0
    svss.resize(5)
    for svs in svss:
        assert len(svs) == 0
    svss.clear()
    assert len(svss) == 0
    svss.extend([sparse_vectors([sparse_vector([pair(1, 2), pair(3, 4)]), sparse_vector([pair(5, 6), pair(7, 8)])])])
    assert len(svss) == 1
    assert svss[0][0][0].first == 1
    assert svss[0][0][0].second == 2
    assert svss[0][0][1].first == 3
    assert svss[0][0][1].second == 4
    assert svss[0][1][0].first == 5
    assert svss[0][1][0].second == 6
    assert svss[0][1][1].first == 7
    assert svss[0][1][1].second == 8
    deser = pickle.loads(pickle.dumps(svss, 2))
    assert deser == svss