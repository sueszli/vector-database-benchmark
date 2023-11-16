from math import isclose
import hypothesis
import pytest
import textdistance
ALG = textdistance.Sorensen

@pytest.mark.parametrize('left, right, expected', [('test', 'text', 2.0 * 3 / 8)])
def test_distance(left, right, expected):
    if False:
        i = 10
        return i + 15
    actual = ALG(external=False)(left, right)
    assert isclose(actual, expected)
    actual = ALG(external=True)(left, right)
    assert isclose(actual, expected)

@hypothesis.given(left=hypothesis.strategies.text(), right=hypothesis.strategies.text())
def test_compare_with_tversky(left, right):
    if False:
        return 10
    td = textdistance.Tversky(ks=[0.5, 0.5]).distance(left, right)
    jd = ALG().distance(left, right)
    assert isclose(jd, td)

@hypothesis.given(left=hypothesis.strategies.text(), right=hypothesis.strategies.text())
def test_compare_with_tversky_as_set(left, right):
    if False:
        i = 10
        return i + 15
    td = textdistance.Tversky(ks=[0.5, 0.5], as_set=True).distance(left, right)
    jd = ALG(as_set=True).distance(left, right)
    assert isclose(jd, td)