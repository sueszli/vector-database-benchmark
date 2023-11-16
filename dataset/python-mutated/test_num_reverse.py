import datetime
import unittest
import pandas as pd
from pyspark import pandas as ps
from pyspark.pandas.tests.data_type_ops.testing_utils import OpsTestBase

class ReverseTestsMixin:
    """Unit tests for arithmetic operations of numeric data types.

    A few test cases are disabled because pandas-on-Spark returns float64 whereas pandas
    returns float32.
    The underlying reason is the respective Spark operations return DoubleType always.
    """

    @property
    def float_pser(self):
        if False:
            for i in range(10):
                print('nop')
        return pd.Series([1, 2, 3], dtype=float)

    @property
    def float_psser(self):
        if False:
            while True:
                i = 10
        return ps.from_pandas(self.float_pser)

    def test_radd(self):
        if False:
            while True:
                i = 10
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            self.assert_eq(1 + pser, 1 + psser)
            self.assertRaises(TypeError, lambda : 'x' + psser)
            self.assert_eq(True + pser, True + psser)
            self.assert_eq(False + pser, False + psser)
            self.assertRaises(TypeError, lambda : datetime.date(1994, 1, 1) + psser)
            self.assertRaises(TypeError, lambda : datetime.datetime(1994, 1, 1) + psser)

    def test_rsub(self):
        if False:
            while True:
                i = 10
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            self.assert_eq(1 - pser, 1 - psser)
            self.assertRaises(TypeError, lambda : 'x' - psser)
            self.assert_eq(True - pser, True - psser)
            self.assert_eq(False - pser, False - psser)
            self.assertRaises(TypeError, lambda : datetime.date(1994, 1, 1) - psser)
            self.assertRaises(TypeError, lambda : datetime.datetime(1994, 1, 1) - psser)

    def test_rmul(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            self.assert_eq(1 * pser, 1 * psser)
            self.assertRaises(TypeError, lambda : 'x' * psser)
            self.assert_eq(True * pser, True * psser)
            self.assert_eq(False * pser, False * psser)
            self.assertRaises(TypeError, lambda : datetime.date(1994, 1, 1) * psser)
            self.assertRaises(TypeError, lambda : datetime.datetime(1994, 1, 1) * psser)

    def test_rtruediv(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            self.assertRaises(TypeError, lambda : 'x' / psser)
            self.assert_eq((True / pser).astype(float), True / psser, check_exact=False)
            self.assert_eq((False / pser).astype(float), False / psser)
            self.assertRaises(TypeError, lambda : datetime.date(1994, 1, 1) / psser)
            self.assertRaises(TypeError, lambda : datetime.datetime(1994, 1, 1) / psser)

    def test_rfloordiv(self):
        if False:
            print('Hello World!')
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            self.assertRaises(TypeError, lambda : 'x' // psser)
            self.assert_eq((True // pser).astype(float), True // psser)
            self.assert_eq((False // pser).astype(float), False // psser)
            self.assertRaises(TypeError, lambda : datetime.date(1994, 1, 1) // psser)
            self.assertRaises(TypeError, lambda : datetime.datetime(1994, 1, 1) // psser)

    def test_rpow(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            self.assertRaises(TypeError, lambda : 'x' ** psser)
            self.assert_eq((True ** pser).astype(float), True ** psser)
            self.assert_eq((False ** pser).astype(float), False ** psser)
            self.assertRaises(TypeError, lambda : datetime.date(1994, 1, 1) ** psser)
            self.assertRaises(TypeError, lambda : datetime.datetime(1994, 1, 1) ** psser)

    def test_rmod(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            self.assert_eq(1 % pser, 1 % psser)
            self.assert_eq(True % pser, True % psser)
            self.assert_eq(False % pser, False % psser)
            self.assertRaises(TypeError, lambda : datetime.date(1994, 1, 1) % psser)
            self.assertRaises(TypeError, lambda : datetime.datetime(1994, 1, 1) % psser)

class ReverseTests(ReverseTestsMixin, OpsTestBase):
    pass
if __name__ == '__main__':
    from pyspark.pandas.tests.data_type_ops.test_num_reverse import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)