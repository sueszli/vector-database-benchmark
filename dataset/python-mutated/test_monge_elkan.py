from math import isclose
import pytest
import textdistance
ALG = textdistance.MongeElkan

@pytest.mark.parametrize('left, right, expected', [(['Niall'], ['Neal'], 0.805), (['Niall'], ['Nigel'], 0.7866666666666667)])
def test_similarity(left, right, expected):
    if False:
        print('Hello World!')
    actual = ALG(qval=1, algorithm=textdistance.jaro_winkler).similarity(left, right)
    assert isclose(actual, expected)