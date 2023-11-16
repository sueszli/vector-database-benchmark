from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup, DihedralGroup, AlternatingGroup, AbelianGroup, RubikGroup
from sympy.testing.pytest import raises

def test_SymmetricGroup():
    if False:
        while True:
            i = 10
    G = SymmetricGroup(5)
    elements = list(G.generate())
    assert G.generators[0].size == 5
    assert len(elements) == 120
    assert G.is_solvable is False
    assert G.is_abelian is False
    assert G.is_nilpotent is False
    assert G.is_transitive() is True
    H = SymmetricGroup(1)
    assert H.order() == 1
    L = SymmetricGroup(2)
    assert L.order() == 2

def test_CyclicGroup():
    if False:
        i = 10
        return i + 15
    G = CyclicGroup(10)
    elements = list(G.generate())
    assert len(elements) == 10
    assert G.derived_subgroup().order() == 1
    assert G.is_abelian is True
    assert G.is_solvable is True
    assert G.is_nilpotent is True
    H = CyclicGroup(1)
    assert H.order() == 1
    L = CyclicGroup(2)
    assert L.order() == 2

def test_DihedralGroup():
    if False:
        i = 10
        return i + 15
    G = DihedralGroup(6)
    elements = list(G.generate())
    assert len(elements) == 12
    assert G.is_transitive() is True
    assert G.is_abelian is False
    assert G.is_solvable is True
    assert G.is_nilpotent is False
    H = DihedralGroup(1)
    assert H.order() == 2
    L = DihedralGroup(2)
    assert L.order() == 4
    assert L.is_abelian is True
    assert L.is_nilpotent is True

def test_AlternatingGroup():
    if False:
        return 10
    G = AlternatingGroup(5)
    elements = list(G.generate())
    assert len(elements) == 60
    assert [perm.is_even for perm in elements] == [True] * 60
    H = AlternatingGroup(1)
    assert H.order() == 1
    L = AlternatingGroup(2)
    assert L.order() == 1

def test_AbelianGroup():
    if False:
        while True:
            i = 10
    A = AbelianGroup(3, 3, 3)
    assert A.order() == 27
    assert A.is_abelian is True

def test_RubikGroup():
    if False:
        i = 10
        return i + 15
    raises(ValueError, lambda : RubikGroup(1))