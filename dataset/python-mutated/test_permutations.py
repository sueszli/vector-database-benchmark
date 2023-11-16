from hypothesis import given
from hypothesis.errors import InvalidArgument
from hypothesis.strategies import permutations
from tests.common.debug import minimal
from tests.common.utils import fails_with

def test_can_find_non_trivial_permutation():
    if False:
        while True:
            i = 10
    x = minimal(permutations(list(range(5))), lambda x: x[0] != 0)
    assert x == [1, 0, 2, 3, 4]

@given(permutations(list('abcd')))
def test_permutation_values_are_permutations(perm):
    if False:
        i = 10
        return i + 15
    assert len(perm) == 4
    assert set(perm) == set('abcd')

@given(permutations([]))
def test_empty_permutations_are_empty(xs):
    if False:
        print('Hello World!')
    assert xs == []

@fails_with(InvalidArgument)
def test_cannot_permute_non_sequence_types():
    if False:
        i = 10
        return i + 15
    permutations(set()).example()