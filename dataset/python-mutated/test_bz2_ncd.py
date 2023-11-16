from math import isclose
import pytest
import textdistance
ALG = textdistance.bz2_ncd

@pytest.mark.parametrize('left, right, expected', [('test', 'test', 0.08), ('test', 'nani', 0.16)])
def test_similarity(left, right, expected):
    if False:
        for i in range(10):
            print('nop')
    actual = ALG(left, right)
    assert isclose(actual, expected)