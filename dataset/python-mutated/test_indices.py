import math
import pytest
from hypothesis import assume, given, note, strategies as st
from hypothesis.extra._array_helpers import NDIM_MAX
from tests.common.debug import assert_all_examples, find_any

@pytest.mark.parametrize('condition', [lambda ix: Ellipsis in ix, lambda ix: Ellipsis not in ix, lambda ix: None in ix, lambda ix: None not in ix])
def test_generate_optional_indices(xp, xps, condition):
    if False:
        while True:
            i = 10
    'Strategy can generate indices with optional values.'
    strat = xps.array_shapes(min_dims=1, max_dims=32).flatmap(lambda s: xps.indices(s, allow_newaxis=True)).map(lambda idx: idx if isinstance(idx, tuple) else (idx,))
    find_any(strat, condition)

def test_cannot_generate_newaxis_when_disabled(xp, xps):
    if False:
        for i in range(10):
            print('nop')
    'Strategy does not generate newaxis when disabled (i.e. the default).'
    assert_all_examples(xps.indices((3, 3, 3)), lambda idx: idx == ... or None not in idx)

def test_generate_indices_for_0d_shape(xp, xps):
    if False:
        for i in range(10):
            print('nop')
    'Strategy only generates empty tuples or Ellipsis as indices for an empty\n    shape.'
    assert_all_examples(xps.indices(shape=(), allow_ellipsis=True), lambda idx: idx in [(), Ellipsis, (Ellipsis,)])

def test_generate_tuples_and_non_tuples_for_1d_shape(xp, xps):
    if False:
        return 10
    'Strategy can generate tuple and non-tuple indices with a 1-dimensional shape.'
    strat = xps.indices(shape=(1,), allow_ellipsis=True)
    find_any(strat, lambda ix: isinstance(ix, tuple))
    find_any(strat, lambda ix: not isinstance(ix, tuple))

def test_generate_long_ellipsis(xp, xps):
    if False:
        for i in range(10):
            print('nop')
    'Strategy can replace runs of slice(None) with Ellipsis.\n\n    We specifically test if [0,...,0] is generated alongside [0,:,:,:,0]\n    '
    strat = xps.indices(shape=(1, 0, 0, 0, 1), max_dims=3, allow_ellipsis=True)
    find_any(strat, lambda ix: len(ix) == 3 and ix[1] == Ellipsis)
    find_any(strat, lambda ix: len(ix) == 5 and all((isinstance(key, slice) and key == slice(None) for key in ix[1:3])))

def test_indices_replaces_whole_axis_slices_with_ellipsis(xp, xps):
    if False:
        return 10
    assert_all_examples(xps.indices(shape=(0, 0, 0, 0, 0), max_dims=5).filter(lambda idx: isinstance(idx, tuple) and Ellipsis in idx), lambda idx: slice(None) not in idx)

def test_efficiently_generate_indexers(xp, xps):
    if False:
        i = 10
        return i + 15
    'Generation is not too slow.'
    find_any(xps.indices((3, 3, 3, 3, 3)))

@given(allow_newaxis=st.booleans(), allow_ellipsis=st.booleans(), data=st.data())
def test_generate_valid_indices(xp, xps, allow_newaxis, allow_ellipsis, data):
    if False:
        for i in range(10):
            print('nop')
    'Strategy generates valid indices.'
    shape = data.draw(xps.array_shapes(min_dims=1, max_side=4) | xps.array_shapes(min_dims=1, min_side=0, max_side=10), label='shape')
    min_dims = data.draw(st.integers(0, len(shape) if not allow_newaxis else len(shape) + 2), label='min_dims')
    max_dims = data.draw(st.none() | st.integers(min_dims, len(shape) if not allow_newaxis else NDIM_MAX), label='max_dims')
    indexer = data.draw(xps.indices(shape, min_dims=min_dims, max_dims=max_dims, allow_newaxis=allow_newaxis, allow_ellipsis=allow_ellipsis), label='indexer')
    _indexer = indexer if isinstance(indexer, tuple) else (indexer,)
    if not allow_ellipsis:
        assert Ellipsis not in _indexer
    if not allow_newaxis:
        assert None not in _indexer
    for i in _indexer:
        assert isinstance(i, (int, slice)) or i is None or i == Ellipsis
    nonexpanding_indexer = [i for i in _indexer if i is not None]
    if Ellipsis in _indexer:
        assert sum((i == Ellipsis for i in _indexer)) == 1
        assert len(nonexpanding_indexer) <= len(shape) + 1
    else:
        assert len(nonexpanding_indexer) == len(shape)
    if 0 in shape:
        array = xp.zeros(shape)
        assert array.size == 0
    elif math.prod(shape) <= 10 ** 5:
        array = xp.reshape(xp.arange(math.prod(shape)), shape)
    else:
        assume(False)
    note(f'array={array!r}')
    array[indexer]