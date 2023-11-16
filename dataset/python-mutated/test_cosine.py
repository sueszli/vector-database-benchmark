from math import isclose
import pytest
import textdistance
ALG = textdistance.Cosine

@pytest.mark.parametrize('left, right, expected', [('test', 'text', 3.0 / 4), ('nelson', 'neilsen', 5.0 / pow(6 * 7, 0.5))])
def test_distance(left, right, expected):
    if False:
        for i in range(10):
            print('nop')
    actual = ALG(external=False)(left, right)
    assert isclose(actual, expected)
    actual = ALG(external=True)(left, right)
    assert isclose(actual, expected)