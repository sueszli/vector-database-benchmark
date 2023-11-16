"""Tests for simple tools for timing functions' execution. """
from sympy.utilities.timeutils import timed

def test_timed():
    if False:
        while True:
            i = 10
    result = timed(lambda : 1 + 1, limit=100000)
    assert result[0] == 100000 and result[3] == 'ns', str(result)
    result = timed('1 + 1', limit=100000)
    assert result[0] == 100000 and result[3] == 'ns'