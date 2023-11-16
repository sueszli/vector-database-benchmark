import datetime
import pandas as pd
from pandas.api.types import CategoricalDtype
from pyspark import pandas as ps
from pyspark.pandas.tests.data_type_ops.testing_utils import OpsTestBase

class DatetimeOpsTestsMixin:

    @property
    def pser(self):
        if False:
            print('Hello World!')
        return pd.Series(pd.date_range('1994-1-31 10:30:15', periods=3, freq='D'))

    @property
    def psser(self):
        if False:
            for i in range(10):
                print('nop')
        return ps.from_pandas(self.pser)

    @property
    def datetime_pdf(self):
        if False:
            print('Hello World!')
        psers = {'this': self.pser, 'that': pd.Series(pd.date_range('1994-2-1 10:30:15', periods=3, freq='D'))}
        return pd.concat(psers, axis=1)

    @property
    def datetime_psdf(self):
        if False:
            return 10
        return ps.from_pandas(self.datetime_pdf)

    @property
    def some_datetime(self):
        if False:
            i = 10
            return i + 15
        return datetime.datetime(1994, 1, 31, 10, 30, 0)

    def test_add(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : self.psser + 'x')
        self.assertRaises(TypeError, lambda : self.psser + 1)
        self.assertRaises(TypeError, lambda : self.psser + self.some_datetime)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser + psser)

    def test_sub(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : self.psser - 'x')
        self.assertRaises(TypeError, lambda : self.psser - 1)
        self.assert_eq((self.pser - self.some_datetime).dt.total_seconds().astype('int'), self.psser - self.some_datetime)
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.df_cols:
            if col == 'datetime':
                self.assert_eq((pdf['datetime'] - pdf[col]).dt.total_seconds().astype('int'), psdf['datetime'] - psdf[col])
            else:
                self.assertRaises(TypeError, lambda : psdf['datetime'] - psdf[col])
        (pdf, psdf) = (self.datetime_pdf, self.datetime_psdf)
        self.assert_eq((pdf['that'] - pdf['this']).dt.total_seconds().astype('int'), psdf['that'] - psdf['this'])

    def test_mul(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : self.psser * 'x')
        self.assertRaises(TypeError, lambda : self.psser * 1)
        self.assertRaises(TypeError, lambda : self.psser * self.some_datetime)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser * psser)

    def test_truediv(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : self.psser / 'x')
        self.assertRaises(TypeError, lambda : self.psser / 1)
        self.assertRaises(TypeError, lambda : self.psser / self.some_datetime)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser / psser)

    def test_floordiv(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : self.psser // 'x')
        self.assertRaises(TypeError, lambda : self.psser // 1)
        self.assertRaises(TypeError, lambda : self.psser // self.some_datetime)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser // psser)

    def test_mod(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : self.psser % 'x')
        self.assertRaises(TypeError, lambda : self.psser % 1)
        self.assertRaises(TypeError, lambda : self.psser % self.some_datetime)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser % psser)

    def test_pow(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : self.psser ** 'x')
        self.assertRaises(TypeError, lambda : self.psser ** 1)
        self.assertRaises(TypeError, lambda : self.psser ** self.some_datetime)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser ** psser)

    def test_radd(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : 'x' + self.psser)
        self.assertRaises(TypeError, lambda : 1 + self.psser)
        self.assertRaises(TypeError, lambda : self.some_datetime + self.psser)

    def test_rsub(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : 'x' - self.psser)
        self.assertRaises(TypeError, lambda : 1 - self.psser)
        self.assert_eq((self.some_datetime - self.pser).dt.total_seconds().astype('int'), self.some_datetime - self.psser)

    def test_rmul(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : 'x' * self.psser)
        self.assertRaises(TypeError, lambda : 1 * self.psser)
        self.assertRaises(TypeError, lambda : self.some_datetime * self.psser)

    def test_rtruediv(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : 'x' / self.psser)
        self.assertRaises(TypeError, lambda : 1 / self.psser)
        self.assertRaises(TypeError, lambda : self.some_datetime / self.psser)

    def test_rfloordiv(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : 'x' // self.psser)
        self.assertRaises(TypeError, lambda : 1 // self.psser)
        self.assertRaises(TypeError, lambda : self.some_datetime // self.psser)

    def test_rmod(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : 1 % self.psser)
        self.assertRaises(TypeError, lambda : self.some_datetime % self.psser)

    def test_rpow(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : 'x' ** self.psser)
        self.assertRaises(TypeError, lambda : 1 ** self.psser)
        self.assertRaises(TypeError, lambda : self.some_datetime ** self.psser)

    def test_and(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : self.psser & True)
        self.assertRaises(TypeError, lambda : self.psser & False)
        self.assertRaises(TypeError, lambda : self.psser & self.psser)

    def test_rand(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : True & self.psser)
        self.assertRaises(TypeError, lambda : False & self.psser)

    def test_or(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : self.psser | True)
        self.assertRaises(TypeError, lambda : self.psser | False)
        self.assertRaises(TypeError, lambda : self.psser | self.psser)

    def test_ror(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : True | self.psser)
        self.assertRaises(TypeError, lambda : False | self.psser)

    def test_from_to_pandas(self):
        if False:
            i = 10
            return i + 15
        data = pd.date_range('1994-1-31 10:30:15', periods=3, freq='M')
        pser = pd.Series(data)
        psser = ps.Series(data)
        self.assert_eq(pser, psser._to_pandas())
        self.assert_eq(ps.from_pandas(pser), psser)

    def test_isnull(self):
        if False:
            print('Hello World!')
        self.assert_eq(self.pser.isnull(), self.psser.isnull())

    def test_astype(self):
        if False:
            while True:
                i = 10
        pser = self.pser
        psser = self.psser
        self.assert_eq(pser.astype(str), psser.astype(str))
        self.assert_eq(pser.astype('category'), psser.astype('category'))
        cat_type = CategoricalDtype(categories=['a', 'b', 'c'])
        self.assert_eq(pser.astype(cat_type), psser.astype(cat_type))
        self.assertRaises(TypeError, lambda : psser.astype(bool))

    def test_neg(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : -self.psser)

    def test_abs(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : abs(self.psser))

    def test_invert(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : ~self.psser)

    def test_eq(self):
        if False:
            i = 10
            return i + 15
        (pdf, psdf) = (self.datetime_pdf, self.datetime_psdf)
        self.assert_eq(pdf['this'] == pdf['that'], psdf['this'] == psdf['that'])
        self.assert_eq(pdf['this'] == pdf['this'], psdf['this'] == psdf['this'])

    def test_ne(self):
        if False:
            i = 10
            return i + 15
        (pdf, psdf) = (self.datetime_pdf, self.datetime_psdf)
        self.assert_eq(pdf['this'] != pdf['that'], psdf['this'] != psdf['that'])
        self.assert_eq(pdf['this'] != pdf['this'], psdf['this'] != psdf['this'])

    def test_lt(self):
        if False:
            return 10
        (pdf, psdf) = (self.datetime_pdf, self.datetime_psdf)
        self.assert_eq(pdf['this'] < pdf['that'], psdf['this'] < psdf['that'])
        self.assert_eq(pdf['this'] < pdf['this'], psdf['this'] < psdf['this'])

    def test_le(self):
        if False:
            return 10
        (pdf, psdf) = (self.datetime_pdf, self.datetime_psdf)
        self.assert_eq(pdf['this'] <= pdf['that'], psdf['this'] <= psdf['that'])
        self.assert_eq(pdf['this'] <= pdf['this'], psdf['this'] <= psdf['this'])

    def test_gt(self):
        if False:
            print('Hello World!')
        (pdf, psdf) = (self.datetime_pdf, self.datetime_psdf)
        self.assert_eq(pdf['this'] > pdf['that'], psdf['this'] > psdf['that'])
        self.assert_eq(pdf['this'] > pdf['this'], psdf['this'] > psdf['this'])

    def test_ge(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.datetime_pdf, self.datetime_psdf)
        self.assert_eq(pdf['this'] >= pdf['that'], psdf['this'] >= psdf['that'])
        self.assert_eq(pdf['this'] >= pdf['this'], psdf['this'] >= psdf['this'])

class DatetimeOpsTests(DatetimeOpsTestsMixin, OpsTestBase):
    pass

class DatetimeNTZOpsTest(DatetimeOpsTests):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super(DatetimeOpsTests, cls).setUpClass()
        cls.spark.conf.set('spark.sql.timestampType', 'timestamp_ntz')
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.data_type_ops.test_datetime_ops import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)