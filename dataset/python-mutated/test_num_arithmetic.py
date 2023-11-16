import unittest
import pandas as pd
import numpy as np
from pyspark import pandas as ps
from pyspark.pandas.tests.data_type_ops.testing_utils import OpsTestBase

class ArithmeticTestsMixin:
    """Unit tests for arithmetic operations of numeric data types.

    A few test cases are disabled because pandas-on-Spark returns float64 whereas pandas
    returns float32.
    The underlying reason is the respective Spark operations return DoubleType always.
    """

    @property
    def float_pser(self):
        if False:
            print('Hello World!')
        return pd.Series([1, 2, 3], dtype=float)

    @property
    def float_psser(self):
        if False:
            while True:
                i = 10
        return ps.from_pandas(self.float_pser)

    def test_add(self):
        if False:
            print('Hello World!')
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            self.assert_eq(pser + pser, psser + psser)
            self.assert_eq(pser + 1, psser + 1)
            self.assert_eq(pser + pser.astype(bool), psser + psser.astype(bool))
            self.assert_eq(pser + True, psser + True)
            self.assert_eq(pser + False, psser + False)
            for n_col in self.non_numeric_df_cols:
                if n_col == 'bool':
                    self.assert_eq(pser + pdf[n_col], psser + psdf[n_col])
                else:
                    self.assertRaises(TypeError, lambda : psser + psdf[n_col])

    def test_sub(self):
        if False:
            print('Hello World!')
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            self.assert_eq(pser - pser, psser - psser)
            self.assert_eq(pser - 1, psser - 1)
            self.assert_eq(pser - pser.astype(bool), psser - psser.astype(bool))
            self.assert_eq(pser - True, psser - True)
            self.assert_eq(pser - False, psser - False)
            for n_col in self.non_numeric_df_cols:
                if n_col == 'bool':
                    self.assert_eq(pser - pdf[n_col], psser - psdf[n_col])
                else:
                    self.assertRaises(TypeError, lambda : psser - psdf[n_col])

    def test_mul(self):
        if False:
            return 10
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            self.assert_eq(pser * pser, psser * psser)
            self.assert_eq(pser * pser.astype(bool), psser * psser.astype(bool))
            self.assert_eq(pser * True, psser * True)
            self.assert_eq(pser * False, psser * False)
            if psser.dtype in [int, np.int32]:
                self.assert_eq(pser * pdf['string'], psser * psdf['string'])
            else:
                self.assertRaises(TypeError, lambda : psser * psdf['string'])
            self.assert_eq(pser * pdf['bool'], psser * psdf['bool'])
            self.assertRaises(TypeError, lambda : psser * psdf['datetime'])
            self.assertRaises(TypeError, lambda : psser * psdf['date'])
            self.assertRaises(TypeError, lambda : psser * psdf['categorical'])

    def test_truediv(self):
        if False:
            i = 10
            return i + 15
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            if psser.dtype in [float, int, np.int32]:
                self.assert_eq(pser / pser, psser / psser)
                self.assert_eq(pser / pser.astype(bool), psser / psser.astype(bool))
                self.assert_eq(pser / True, psser / True)
                self.assert_eq(pser / False, psser / False)
            for n_col in self.non_numeric_df_cols:
                if n_col == 'bool':
                    self.assert_eq(pdf['float'] / pdf[n_col], psdf['float'] / psdf[n_col])
                else:
                    self.assertRaises(TypeError, lambda : psser / psdf[n_col])

    def test_floordiv(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.pdf, self.psdf)
        (pser, psser) = (pdf['float'], psdf['float'])
        self.assert_eq(pser // pser, psser // psser)
        self.assert_eq(pser // pser.astype(bool), psser // psser.astype(bool))
        self.assert_eq(pser // True, psser // True)
        self.assert_eq(pser // False, psser // False)
        for n_col in self.non_numeric_df_cols:
            if n_col == 'bool':
                self.assert_eq(pdf['float'] // pdf['bool'], psdf['float'] // psdf['bool'])
            else:
                for col in self.numeric_df_cols:
                    psser = psdf[col]
                    self.assertRaises(TypeError, lambda : psser // psdf[n_col])

    def test_mod(self):
        if False:
            print('Hello World!')
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            self.assert_eq(pser % pser, psser % psser)
            self.assert_eq(pser % pser.astype(bool), psser % psser.astype(bool))
            self.assert_eq(pser % True, psser % True)
            if col in ['int', 'int32']:
                self.assert_eq(pd.Series([np.nan, np.nan, np.nan], dtype=float, name=col), psser % False)
            else:
                self.assert_eq(pd.Series([np.nan, np.nan, np.nan], dtype=pser.dtype, name=col), psser % False)
            for n_col in self.non_numeric_df_cols:
                if n_col == 'bool':
                    self.assert_eq(pdf['float'] % pdf[n_col], psdf['float'] % psdf[n_col])
                else:
                    self.assertRaises(TypeError, lambda : psser % psdf[n_col])

    def test_pow(self):
        if False:
            while True:
                i = 10
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            if col in ['float', 'float_w_nan']:
                self.assert_eq(pser ** pser, psser ** psser)
                self.assert_eq(pser ** pser.astype(bool), psser ** psser.astype(bool))
                self.assert_eq(pser ** True, psser ** True)
                self.assert_eq(pser ** False, psser ** False)
                self.assert_eq(pser ** 1, psser ** 1)
                self.assert_eq(pser ** 0, psser ** 0)
            for n_col in self.non_numeric_df_cols:
                if n_col == 'bool':
                    self.assert_eq(pdf['float'] ** pdf[n_col], psdf['float'] ** psdf[n_col])
                else:
                    self.assertRaises(TypeError, lambda : psser ** psdf[n_col])

class ArithmeticTests(ArithmeticTestsMixin, OpsTestBase):
    pass
if __name__ == '__main__':
    from pyspark.pandas.tests.data_type_ops.test_num_arithmetic import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)