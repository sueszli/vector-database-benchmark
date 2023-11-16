import datetime
import unittest
import pandas as pd
from pandas.api.types import CategoricalDtype
from pyspark import pandas as ps
from pyspark.pandas.tests.data_type_ops.testing_utils import OpsTestBase

class DateOpsTestsMixin:

    @property
    def pser(self):
        if False:
            return 10
        return pd.Series([datetime.date(1994, 1, 31), datetime.date(1994, 2, 1), datetime.date(1994, 2, 2)])

    @property
    def psser(self):
        if False:
            i = 10
            return i + 15
        return ps.from_pandas(self.pser)

    @property
    def date_pdf(self):
        if False:
            for i in range(10):
                print('nop')
        psers = {'this': self.pser, 'that': pd.Series([datetime.date(2000, 1, 31), datetime.date(1994, 3, 1), datetime.date(1990, 2, 2)])}
        return pd.concat(psers, axis=1)

    @property
    def date_psdf(self):
        if False:
            while True:
                i = 10
        return ps.from_pandas(self.date_pdf)

    @property
    def some_date(self):
        if False:
            return 10
        return datetime.date(1994, 1, 1)

    def test_add(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : self.psser + 'x')
        self.assertRaises(TypeError, lambda : self.psser + 1)
        self.assertRaises(TypeError, lambda : self.psser + self.some_date)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser + psser)

    def test_sub(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : self.psser - 'x')
        self.assertRaises(TypeError, lambda : self.psser - 1)
        self.assert_eq((self.pser - self.some_date).apply(lambda x: x.days), self.psser - self.some_date)
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.df_cols:
            if col == 'date':
                self.assert_eq((pdf['date'] - pdf[col]).apply(lambda x: x.days), psdf['date'] - psdf[col])
            else:
                self.assertRaises(TypeError, lambda : psdf['date'] - psdf[col])
        (pdf, psdf) = (self.date_pdf, self.date_psdf)
        self.assert_eq((pdf['this'] - pdf['that']).apply(lambda x: x.days), psdf['this'] - psdf['that'])

    def test_mul(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : self.psser * 'x')
        self.assertRaises(TypeError, lambda : self.psser * 1)
        self.assertRaises(TypeError, lambda : self.psser * self.some_date)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser * psser)

    def test_truediv(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : self.psser / 'x')
        self.assertRaises(TypeError, lambda : self.psser / 1)
        self.assertRaises(TypeError, lambda : self.psser / self.some_date)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser / psser)

    def test_floordiv(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : self.psser // 'x')
        self.assertRaises(TypeError, lambda : self.psser // 1)
        self.assertRaises(TypeError, lambda : self.psser // self.some_date)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser // psser)

    def test_mod(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : self.psser % 'x')
        self.assertRaises(TypeError, lambda : self.psser % 1)
        self.assertRaises(TypeError, lambda : self.psser % self.some_date)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser % psser)

    def test_pow(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : self.psser ** 'x')
        self.assertRaises(TypeError, lambda : self.psser ** 1)
        self.assertRaises(TypeError, lambda : self.psser ** self.some_date)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser ** psser)

    def test_radd(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : 'x' + self.psser)
        self.assertRaises(TypeError, lambda : 1 + self.psser)
        self.assertRaises(TypeError, lambda : self.some_date + self.psser)

    def test_rsub(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : 'x' - self.psser)
        self.assertRaises(TypeError, lambda : 1 - self.psser)
        self.assert_eq((self.some_date - self.pser).apply(lambda x: x.days), self.some_date - self.psser)

    def test_rmul(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : 'x' * self.psser)
        self.assertRaises(TypeError, lambda : 1 * self.psser)
        self.assertRaises(TypeError, lambda : self.some_date * self.psser)

    def test_rtruediv(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : 'x' / self.psser)
        self.assertRaises(TypeError, lambda : 1 / self.psser)
        self.assertRaises(TypeError, lambda : self.some_date / self.psser)

    def test_rfloordiv(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : 'x' // self.psser)
        self.assertRaises(TypeError, lambda : 1 // self.psser)
        self.assertRaises(TypeError, lambda : self.some_date // self.psser)

    def test_rmod(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : 1 % self.psser)
        self.assertRaises(TypeError, lambda : self.some_date % self.psser)

    def test_rpow(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : 'x' ** self.psser)
        self.assertRaises(TypeError, lambda : 1 ** self.psser)
        self.assertRaises(TypeError, lambda : self.some_date ** self.psser)

    def test_and(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : self.psser & True)
        self.assertRaises(TypeError, lambda : self.psser & False)
        self.assertRaises(TypeError, lambda : self.psser & self.psser)

    def test_rand(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : True & self.psser)
        self.assertRaises(TypeError, lambda : False & self.psser)

    def test_or(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : self.psser | True)
        self.assertRaises(TypeError, lambda : self.psser | False)
        self.assertRaises(TypeError, lambda : self.psser | self.psser)

    def test_ror(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : True | self.psser)
        self.assertRaises(TypeError, lambda : False | self.psser)

    def test_from_to_pandas(self):
        if False:
            for i in range(10):
                print('nop')
        data = [datetime.date(1994, 1, 31), datetime.date(1994, 2, 1), datetime.date(1994, 2, 2)]
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
            i = 10
            return i + 15
        pser = self.pser
        psser = self.psser
        self.assert_eq(pser.astype(str), psser.astype(str))
        self.assert_eq(pser.astype(bool), psser.astype(bool))
        cat_type = CategoricalDtype(categories=['a', 'b', 'c'])
        self.assert_eq(pser.astype(cat_type), psser.astype(cat_type))

    def test_neg(self):
        if False:
            while True:
                i = 10
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
            i = 10
            return i + 15
        (pdf, psdf) = (self.date_pdf, self.date_psdf)
        self.assert_eq(pdf['this'] == pdf['that'], psdf['this'] == psdf['that'])
        self.assert_eq(pdf['this'] == pdf['this'], psdf['this'] == psdf['this'])

    def test_ne(self):
        if False:
            return 10
        (pdf, psdf) = (self.date_pdf, self.date_psdf)
        self.assert_eq(pdf['this'] != pdf['that'], psdf['this'] != psdf['that'])
        self.assert_eq(pdf['this'] != pdf['this'], psdf['this'] != psdf['this'])

    def test_lt(self):
        if False:
            print('Hello World!')
        (pdf, psdf) = (self.date_pdf, self.date_psdf)
        self.assert_eq(pdf['this'] < pdf['that'], psdf['this'] < psdf['that'])
        self.assert_eq(pdf['this'] < pdf['this'], psdf['this'] < psdf['this'])

    def test_le(self):
        if False:
            while True:
                i = 10
        (pdf, psdf) = (self.date_pdf, self.date_psdf)
        self.assert_eq(pdf['this'] <= pdf['that'], psdf['this'] <= psdf['that'])
        self.assert_eq(pdf['this'] <= pdf['this'], psdf['this'] <= psdf['this'])

    def test_gt(self):
        if False:
            return 10
        (pdf, psdf) = (self.date_pdf, self.date_psdf)
        self.assert_eq(pdf['this'] > pdf['that'], psdf['this'] > psdf['that'])
        self.assert_eq(pdf['this'] > pdf['this'], psdf['this'] > psdf['this'])

    def test_ge(self):
        if False:
            i = 10
            return i + 15
        (pdf, psdf) = (self.date_pdf, self.date_psdf)
        self.assert_eq(pdf['this'] >= pdf['that'], psdf['this'] >= psdf['that'])
        self.assert_eq(pdf['this'] >= pdf['this'], psdf['this'] >= psdf['this'])

class DateOpsTests(DateOpsTestsMixin, OpsTestBase):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.data_type_ops.test_date_ops import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)