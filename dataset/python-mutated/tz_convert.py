import numpy as np
from pytz import UTC
from pandas._libs.tslibs.tzconversion import tz_localize_to_utc
from .tslib import _sizes, _tzs, tzlocal_obj
try:
    old_sig = False
    from pandas._libs.tslibs import tz_convert_from_utc
except ImportError:
    try:
        old_sig = False
        from pandas._libs.tslibs.tzconversion import tz_convert_from_utc
    except ImportError:
        old_sig = True
        from pandas._libs.tslibs.tzconversion import tz_convert as tz_convert_from_utc

class TimeTZConvert:
    params = [_sizes, [x for x in _tzs if x is not None]]
    param_names = ['size', 'tz']

    def setup(self, size, tz):
        if False:
            print('Hello World!')
        if size == 10 ** 6 and tz is tzlocal_obj:
            raise NotImplementedError
        arr = np.random.randint(0, 10, size=size, dtype='i8')
        self.i8data = arr

    def time_tz_convert_from_utc(self, size, tz):
        if False:
            while True:
                i = 10
        if old_sig:
            tz_convert_from_utc(self.i8data, UTC, tz)
        else:
            tz_convert_from_utc(self.i8data, tz)

    def time_tz_localize_to_utc(self, size, tz):
        if False:
            while True:
                i = 10
        tz_localize_to_utc(self.i8data, tz, ambiguous='NaT', nonexistent='NaT')