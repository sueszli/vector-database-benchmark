import unittest
import datetime
import numpy as np
import pandas as pd
from pyspark import pandas as ps
from pyspark.testing.pandasutils import PandasOnSparkTestCase, TestUtils

class FrameResampleTestsMixin:

    @property
    def pdf1(self):
        if False:
            while True:
                i = 10
        np.random.seed(11)
        dates = [pd.NaT, datetime.datetime(2011, 12, 31), datetime.datetime(2011, 12, 31, 0, 0, 1), datetime.datetime(2011, 12, 31, 23, 59, 59), datetime.datetime(2012, 1, 1), datetime.datetime(2012, 1, 1, 0, 0, 1), pd.NaT, datetime.datetime(2012, 1, 1, 23, 59, 59), datetime.datetime(2012, 1, 2), pd.NaT, datetime.datetime(2012, 1, 30, 23, 59, 59), datetime.datetime(2012, 1, 31), datetime.datetime(2012, 1, 31, 0, 0, 1), datetime.datetime(2012, 3, 31), datetime.datetime(2013, 5, 3), datetime.datetime(2022, 5, 3)]
        return pd.DataFrame(np.random.rand(len(dates), 2), index=pd.DatetimeIndex(dates), columns=list('AB'))

    @property
    def pdf2(self):
        if False:
            return 10
        np.random.seed(22)
        dates = [datetime.datetime(2022, 5, 1, 4, 5, 6), datetime.datetime(2022, 5, 3), datetime.datetime(2022, 5, 3, 23, 59, 59), datetime.datetime(2022, 5, 4), pd.NaT, datetime.datetime(2022, 5, 4, 0, 0, 1), datetime.datetime(2022, 5, 11)]
        return pd.DataFrame(np.random.rand(len(dates), 2), index=pd.DatetimeIndex(dates), columns=list('AB'))

    @property
    def pdf3(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(22)
        index = pd.date_range(start='2011-01-02', end='2022-05-01', freq='1D')
        return pd.DataFrame(np.random.rand(len(index), 2), index=index, columns=list('AB'))

    @property
    def pdf4(self):
        if False:
            print('Hello World!')
        np.random.seed(33)
        index = pd.date_range(start='2020-12-12', end='2022-05-01', freq='1H')
        return pd.DataFrame(np.random.rand(len(index), 2), index=index, columns=list('AB'))

    @property
    def pdf5(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(44)
        index = pd.date_range(start='2021-12-30 03:04:05', end='2022-01-02 06:07:08', freq='1T')
        return pd.DataFrame(np.random.rand(len(index), 2), index=index, columns=list('AB'))

    @property
    def pdf6(self):
        if False:
            return 10
        np.random.seed(55)
        index = pd.date_range(start='2022-05-02 03:04:05', end='2022-05-02 06:07:08', freq='1S')
        return pd.DataFrame(np.random.rand(len(index), 2), index=index, columns=list('AB'))

    @property
    def psdf1(self):
        if False:
            print('Hello World!')
        return ps.from_pandas(self.pdf1)

    @property
    def psdf2(self):
        if False:
            print('Hello World!')
        return ps.from_pandas(self.pdf2)

    @property
    def psdf3(self):
        if False:
            while True:
                i = 10
        return ps.from_pandas(self.pdf3)

    @property
    def psdf4(self):
        if False:
            while True:
                i = 10
        return ps.from_pandas(self.pdf4)

    @property
    def psdf5(self):
        if False:
            print('Hello World!')
        return ps.from_pandas(self.pdf5)

    @property
    def psdf6(self):
        if False:
            return 10
        return ps.from_pandas(self.pdf6)

    def _test_resample(self, pobj, psobj, rules, closed, label, func):
        if False:
            i = 10
            return i + 15
        for rule in rules:
            p_resample = pobj.resample(rule=rule, closed=closed, label=label)
            ps_resample = psobj.resample(rule=rule, closed=closed, label=label)
            self.assert_eq(getattr(p_resample, func)().sort_index(), getattr(ps_resample, func)().sort_index(), almost=True)

    def test_dataframe_resample(self):
        if False:
            print('Hello World!')
        self._test_resample(self.pdf1, self.psdf1, ['3Y', '9M', '17D'], None, None, 'min')
        self._test_resample(self.pdf2, self.psdf2, ['3A', '11M', 'D'], None, 'left', 'max')
        self._test_resample(self.pdf3, self.psdf3, ['20D', '1M'], None, 'right', 'sum')
        self._test_resample(self.pdf4, self.psdf4, ['11H', '21D'], 'left', None, 'mean')
        self._test_resample(self.pdf5, self.psdf5, ['55MIN', '2H', 'D'], 'left', 'left', 'std')
        self._test_resample(self.pdf6, self.psdf6, ['29S', '10MIN', '3H'], 'left', 'right', 'var')

class FrameResampleTests(FrameResampleTestsMixin, PandasOnSparkTestCase, TestUtils):
    pass
if __name__ == '__main__':
    from pyspark.pandas.tests.test_frame_resample import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)