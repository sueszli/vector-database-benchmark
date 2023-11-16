from datetime import timedelta
import pandas as pd
from pandas.api.types import CategoricalDtype
import pyspark.pandas as ps
from pyspark.pandas.tests.data_type_ops.testing_utils import OpsTestBase

class TimedeltaOpsTestsMixin:

    @property
    def pser(self):
        if False:
            return 10
        return pd.Series([timedelta(1), timedelta(microseconds=2), timedelta(weeks=3)])

    @property
    def psser(self):
        if False:
            return 10
        return ps.from_pandas(self.pser)

    @property
    def timedelta_pdf(self):
        if False:
            return 10
        psers = {'this': self.pser, 'that': pd.Series([timedelta(0), timedelta(microseconds=1), timedelta(seconds=2)])}
        return pd.concat(psers, axis=1)

    @property
    def timedelta_psdf(self):
        if False:
            print('Hello World!')
        return ps.from_pandas(self.timedelta_pdf)

    @property
    def some_timedelta(self):
        if False:
            for i in range(10):
                print('nop')
        return timedelta(weeks=2)

    def test_add(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : self.psser + 'x')
        self.assertRaises(TypeError, lambda : self.psser + 1)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser + psser)

    def test_sub(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : self.psser - 'x')
        self.assertRaises(TypeError, lambda : self.psser - 1)
        self.assert_eq(self.pser - self.some_timedelta, self.psser - self.some_timedelta)
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.df_cols:
            if col == 'timedelta':
                self.assert_eq(pdf['timedelta'] - pdf[col], psdf['timedelta'] - psdf[col])
            else:
                self.assertRaises(TypeError, lambda : psdf['timedelta'] - psdf[col])
        (pdf, psdf) = (self.timedelta_pdf, self.timedelta_psdf)
        self.assert_eq(pdf['that'] - pdf['this'], psdf['that'] - psdf['this'])

    def test_mul(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : self.psser * 'x')
        self.assertRaises(TypeError, lambda : self.psser * 1)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser * psser)

    def test_truediv(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : self.psser / 'x')
        self.assertRaises(TypeError, lambda : self.psser / 1)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser / psser)

    def test_floordiv(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : self.psser // 'x')
        self.assertRaises(TypeError, lambda : self.psser // 1)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser // psser)

    def test_mod(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : self.psser % 'x')
        self.assertRaises(TypeError, lambda : self.psser % 1)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser % psser)

    def test_pow(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : self.psser ** 'x')
        self.assertRaises(TypeError, lambda : self.psser ** 1)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser ** psser)

    def test_radd(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : 'x' + self.psser)
        self.assertRaises(TypeError, lambda : 1 + self.psser)

    def test_rsub(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : 'x' - self.psser)
        self.assertRaises(TypeError, lambda : 1 - self.psser)
        self.assert_eq(self.some_timedelta - self.pser, self.some_timedelta - self.psser)

    def test_rmul(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : 'x' * self.psser)
        self.assertRaises(TypeError, lambda : 2 * self.psser)

    def test_rtruediv(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : 'x' / self.psser)
        self.assertRaises(TypeError, lambda : 1 / self.psser)

    def test_rfloordiv(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : 'x' // self.psser)
        self.assertRaises(TypeError, lambda : 1 // self.psser)

    def test_rmod(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : 1 % self.psser)

    def test_rpow(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : 'x' ** self.psser)
        self.assertRaises(TypeError, lambda : 1 ** self.psser)

    def test_from_to_pandas(self):
        if False:
            return 10
        data = [timedelta(1), timedelta(microseconds=2)]
        pser = pd.Series(data)
        psser = ps.Series(data)
        self.assert_eq(pser, psser._to_pandas())
        self.assert_eq(ps.from_pandas(pser), psser)

    def test_isnull(self):
        if False:
            while True:
                i = 10
        self.assert_eq(self.pser.isnull(), self.psser.isnull())

    def test_astype(self):
        if False:
            print('Hello World!')
        pser = self.pser
        psser = self.psser
        target_psser = ps.Series(["INTERVAL '1 00:00:00' DAY TO SECOND", "INTERVAL '0 00:00:00.000002' DAY TO SECOND", "INTERVAL '21 00:00:00' DAY TO SECOND"])
        self.assert_eq(target_psser, psser.astype(str))
        self.assert_eq(pser.astype('category'), psser.astype('category'))
        cat_type = CategoricalDtype(categories=['a', 'b', 'c'])
        self.assert_eq(pser.astype(cat_type), psser.astype(cat_type))
        self.assertRaises(TypeError, lambda : psser.astype(bool))

    def test_neg(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : -self.psser)

    def test_abs(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : abs(self.psser))

    def test_invert(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : ~self.psser)

    def test_eq(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.timedelta_pdf, self.timedelta_psdf)
        self.assert_eq(pdf['this'] == pdf['this'], psdf['this'] == psdf['this'])
        self.assert_eq(pdf['this'] == pdf['that'], psdf['this'] == psdf['that'])

    def test_ne(self):
        if False:
            print('Hello World!')
        (pdf, psdf) = (self.timedelta_pdf, self.timedelta_psdf)
        self.assert_eq(pdf['this'] != pdf['this'], psdf['this'] != psdf['this'])
        self.assert_eq(pdf['this'] != pdf['that'], psdf['this'] != psdf['that'])

    def test_lt(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.timedelta_pdf, self.timedelta_psdf)
        self.assert_eq(pdf['this'] < pdf['that'], psdf['this'] < psdf['that'])
        self.assert_eq(pdf['this'] < pdf['this'], psdf['this'] < psdf['this'])

    def test_le(self):
        if False:
            i = 10
            return i + 15
        (pdf, psdf) = (self.timedelta_pdf, self.timedelta_psdf)
        self.assert_eq(pdf['this'] <= pdf['that'], psdf['this'] <= psdf['that'])
        self.assert_eq(pdf['this'] <= pdf['this'], psdf['this'] <= psdf['this'])

    def test_gt(self):
        if False:
            print('Hello World!')
        (pdf, psdf) = (self.timedelta_pdf, self.timedelta_psdf)
        self.assert_eq(pdf['this'] > pdf['that'], psdf['this'] > psdf['that'])
        self.assert_eq(pdf['this'] > pdf['this'], psdf['this'] > psdf['this'])

    def test_ge(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.timedelta_pdf, self.timedelta_psdf)
        self.assert_eq(pdf['this'] >= pdf['that'], psdf['this'] >= psdf['that'])
        self.assert_eq(pdf['this'] >= pdf['this'], psdf['this'] >= psdf['this'])

class TimedeltaOpsTests(TimedeltaOpsTestsMixin, OpsTestBase):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.data_type_ops.test_timedelta_ops import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)