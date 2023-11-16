import pytest
from pandas import Series

@pytest.mark.parametrize('data, index, expected', [([1, 2, 3], None, 3), ({'a': 1, 'b': 2, 'c': 3}, None, 3), ([1, 2, 3], ['x', 'y', 'z'], 3), ([1, 2, 3, 4, 5], ['x', 'y', 'z', 'w', 'n'], 5), ([1, 2, 3], None, 3), ([1, 2, 3], ['x', 'y', 'z'], 3), ([1, 2, 3, 4], ['x', 'y', 'z', 'w'], 4)])
def test_series(data, index, expected):
    if False:
        for i in range(10):
            print('nop')
    ser = Series(data, index=index)
    assert ser.size == expected
    assert isinstance(ser.size, int)