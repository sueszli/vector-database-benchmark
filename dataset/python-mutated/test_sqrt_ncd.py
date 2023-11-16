from math import isclose
import hypothesis
import pytest
import textdistance
ALG = textdistance.sqrt_ncd

@pytest.mark.parametrize('left, right, expected', [('test', 'test', 0.41421356237309503), ('test', 'nani', 1)])
def test_similarity(left, right, expected):
    if False:
        print('Hello World!')
    actual = ALG(left, right)
    assert isclose(actual, expected)

@hypothesis.given(text=hypothesis.strategies.text(min_size=1))
def test_simmetry_compressor(text):
    if False:
        for i in range(10):
            print('nop')
    rev = ''.join(reversed(text))
    assert ALG._compress(text) == ALG._compress(rev)

@hypothesis.given(text=hypothesis.strategies.text(min_size=1))
def test_idempotency_compressor(text):
    if False:
        print('Hello World!')
    assert ALG._get_size(text * 2) < ALG._get_size(text) * 2

@hypothesis.given(left=hypothesis.strategies.text(min_size=1), right=hypothesis.strategies.characters())
def test_monotonicity_compressor(left, right):
    if False:
        i = 10
        return i + 15
    if right in left:
        return
    assert ALG._get_size(left) <= ALG._get_size(left + right)

@hypothesis.given(left1=hypothesis.strategies.text(min_size=1), left2=hypothesis.strategies.text(min_size=1), right=hypothesis.strategies.text(min_size=1))
def test_distributivity_compressor(left1, left2, right):
    if False:
        for i in range(10):
            print('nop')
    actual1 = ALG._get_size(left1 + left2) + ALG._get_size(right)
    actual2 = ALG._get_size(left1 + right) + ALG._get_size(left2 + right)
    assert actual1 <= actual2

@hypothesis.given(text=hypothesis.strategies.text(min_size=1))
def test_normalization_range(text):
    if False:
        print('Hello World!')
    assert 0 <= ALG.normalized_similarity(text, text) <= 1
    assert 0 <= ALG.normalized_distance(text, text) <= 1