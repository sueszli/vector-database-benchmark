from math import isclose
import pytest
import textdistance
ALG = textdistance.JaroWinkler

@pytest.mark.parametrize('left, right, expected', [('elephant', 'hippo', 0.44166666666666665), ('fly', 'ant', 0.0), ('frog', 'fog', 0.925), ('MARTHA', 'MARHTA', 0.9611111111111111), ('DWAYNE', 'DUANE', 0.84), ('DIXON', 'DICKSONX', 0.8133333333333332), ('duck donald', 'duck daisy', 0.867272727272)])
def test_distance(left, right, expected):
    if False:
        return 10
    actual = ALG(winklerize=True, external=False)(left, right)
    assert isclose(actual, expected)
    actual = ALG(winklerize=True, external=True)(left, right)
    assert isclose(actual, expected)