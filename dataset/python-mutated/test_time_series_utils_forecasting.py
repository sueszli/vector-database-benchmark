"""Module to test time_series forecasting utils
"""
import pytest
from pycaret.utils.time_series.forecasting import _check_and_clean_coverage
pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

def test_check_and_clean_coverage():
    if False:
        while True:
            i = 10
    'Tests _check_and_clean_coverage'
    coverage = 0.9
    coverage = _check_and_clean_coverage(coverage=coverage)
    coverage = [round(value, 2) for value in coverage]
    assert isinstance(coverage, list)
    assert coverage == [0.05, 0.95]
    coverage_expected = [0.1, 0.9]
    coverage = _check_and_clean_coverage(coverage=coverage_expected)
    assert isinstance(coverage, list)
    assert coverage == coverage_expected
    coverage = [0.9, 0.1]
    coverage = _check_and_clean_coverage(coverage=coverage)
    assert isinstance(coverage, list)
    assert coverage == coverage_expected
    with pytest.raises(ValueError) as errmsg:
        coverage = [0.1]
        coverage = _check_and_clean_coverage(coverage=coverage)
    exceptionmsg = errmsg.value.args[0]
    assert 'When coverage is a list, it must be of length 2 corresponding to' in exceptionmsg
    with pytest.raises(ValueError) as errmsg:
        coverage = [0.1, 0.5, 0.9]
        coverage = _check_and_clean_coverage(coverage=coverage)
    exceptionmsg = errmsg.value.args[0]
    assert 'When coverage is a list, it must be of length 2 corresponding to' in exceptionmsg
    with pytest.raises(TypeError) as errmsg:
        coverage = None
        coverage = _check_and_clean_coverage(coverage=coverage)
    exceptionmsg = errmsg.value.args[0]
    assert "'coverage' must be of type float or a List of floats of length 2." in exceptionmsg