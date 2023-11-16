import pytest
from pandas import Series
import pandas._testing as tm

class TestStrAccessor:

    def test_str_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        methods = ['strip', 'rstrip', 'lstrip']
        ser = Series([' jack', 'jill ', ' jesse ', 'frank'])
        for method in methods:
            expected = Series([getattr(str, method)(x) for x in ser.values])
            tm.assert_series_equal(getattr(Series.str, method)(ser.str), expected)
        ser = Series(range(5))
        with pytest.raises(AttributeError, match='only use .str accessor'):
            ser.str.repeat(2)

    def test_str_accessor_updates_on_inplace(self):
        if False:
            print('Hello World!')
        ser = Series(list('abc'))
        return_value = ser.drop([0], inplace=True)
        assert return_value is None
        assert len(ser.str.lower()) == 2