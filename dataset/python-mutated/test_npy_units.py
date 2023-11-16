import numpy as np
from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit
from pandas._libs.tslibs.vectorized import is_date_array_normalized
day_arr = np.arange(10, dtype='i8').view('M8[D]')

class TestIsDateArrayNormalized:

    def test_is_date_array_normalized_day(self):
        if False:
            print('Hello World!')
        arr = day_arr
        abbrev = 'D'
        unit = abbrev_to_npy_unit(abbrev)
        result = is_date_array_normalized(arr.view('i8'), None, unit)
        assert result is True

    def test_is_date_array_normalized_seconds(self):
        if False:
            for i in range(10):
                print('nop')
        abbrev = 's'
        arr = day_arr.astype(f'M8[{abbrev}]')
        unit = abbrev_to_npy_unit(abbrev)
        result = is_date_array_normalized(arr.view('i8'), None, unit)
        assert result is True
        arr[0] += np.timedelta64(1, abbrev)
        result2 = is_date_array_normalized(arr.view('i8'), None, unit)
        assert result2 is False