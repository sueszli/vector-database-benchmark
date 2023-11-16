import unittest
import numpy as np
import pandas as pd
import pyspark.pandas as ps
from pyspark.testing.pandasutils import PandasOnSparkTestCase, TestUtils

class DatetimeIndexPropertyTestsMixin:

    @property
    def fixed_freqs(self):
        if False:
            return 10
        return ['D', 'H', 'T', 'S', 'L', 'U']

    @property
    def non_fixed_freqs(self):
        if False:
            i = 10
            return i + 15
        return ['W', 'Q']

    @property
    def pidxs(self):
        if False:
            print('Hello World!')
        return [pd.DatetimeIndex([0]), pd.DatetimeIndex(['2004-01-01', '2002-12-31', '2000-04-01'])] + [pd.date_range('2000-01-01', periods=3, freq=freq) for freq in self.fixed_freqs + self.non_fixed_freqs]

    @property
    def psidxs(self):
        if False:
            i = 10
            return i + 15
        return [ps.from_pandas(pidx) for pidx in self.pidxs]

    @property
    def idx_pairs(self):
        if False:
            while True:
                i = 10
        return list(zip(self.psidxs, self.pidxs))

    def test_properties(self):
        if False:
            return 10
        for (psidx, pidx) in self.idx_pairs:
            self.assert_eq(psidx.year, pidx.year)
            self.assert_eq(psidx.month, pidx.month)
            self.assert_eq(psidx.day, pidx.day)
            self.assert_eq(psidx.hour, pidx.hour)
            self.assert_eq(psidx.minute, pidx.minute)
            self.assert_eq(psidx.second, pidx.second)
            self.assert_eq(psidx.microsecond, pidx.microsecond)
            self.assert_eq(psidx.dayofweek, pidx.dayofweek)
            self.assert_eq(psidx.weekday, pidx.weekday)
            self.assert_eq(psidx.dayofyear, pidx.dayofyear)
            self.assert_eq(psidx.quarter, pidx.quarter)
            self.assert_eq(psidx.daysinmonth, pidx.daysinmonth)
            self.assert_eq(psidx.days_in_month, pidx.days_in_month)
            self.assert_eq(psidx.is_month_start, pd.Index(pidx.is_month_start))
            self.assert_eq(psidx.is_month_end, pd.Index(pidx.is_month_end))
            self.assert_eq(psidx.is_quarter_start, pd.Index(pidx.is_quarter_start))
            self.assert_eq(psidx.is_quarter_end, pd.Index(pidx.is_quarter_end))
            self.assert_eq(psidx.is_year_start, pd.Index(pidx.is_year_start))
            self.assert_eq(psidx.is_year_end, pd.Index(pidx.is_year_end))
            self.assert_eq(psidx.is_leap_year, pd.Index(pidx.is_leap_year))
            self.assert_eq(psidx.day_of_year, pidx.day_of_year)
            self.assert_eq(psidx.day_of_week, pidx.day_of_week)
            self.assert_eq(psidx.isocalendar().week, pidx.isocalendar().week.astype(np.int64))

class DatetimeIndexPropertyTests(DatetimeIndexPropertyTestsMixin, PandasOnSparkTestCase, TestUtils):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.indexes.test_datetime_property import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)