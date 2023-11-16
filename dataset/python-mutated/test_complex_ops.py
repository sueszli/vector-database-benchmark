import decimal
import datetime
import pandas as pd
from pyspark import pandas as ps
from pyspark.pandas.tests.data_type_ops.testing_utils import OpsTestBase

class ComplexOpsTestsMixin:

    @property
    def pser(self):
        if False:
            print('Hello World!')
        return pd.Series([[1, 2, 3]])

    @property
    def psser(self):
        if False:
            print('Hello World!')
        return ps.from_pandas(self.pser)

    @property
    def numeric_array_pdf(self):
        if False:
            for i in range(10):
                print('nop')
        psers = {'int': pd.Series([[1, 2, 3]]), 'float': pd.Series([[0.1, 0.2, 0.3]]), 'decimal': pd.Series([[decimal.Decimal(1), decimal.Decimal(2), decimal.Decimal(3)]])}
        return pd.concat(psers, axis=1)

    @property
    def numeric_array_psdf(self):
        if False:
            print('Hello World!')
        return ps.from_pandas(self.numeric_array_pdf)

    @property
    def numeric_array_df_cols(self):
        if False:
            for i in range(10):
                print('nop')
        return self.numeric_array_pdf.columns

    @property
    def non_numeric_array_pdf(self):
        if False:
            print('Hello World!')
        psers = {'string': pd.Series([['x', 'y', 'z']]), 'date': pd.Series([[datetime.date(1994, 1, 1), datetime.date(1994, 1, 2), datetime.date(1994, 1, 3)]]), 'bool': pd.Series([[True, True, False]])}
        return pd.concat(psers, axis=1)

    @property
    def non_numeric_array_psdf(self):
        if False:
            i = 10
            return i + 15
        return ps.from_pandas(self.non_numeric_array_pdf)

    @property
    def non_numeric_array_df_cols(self):
        if False:
            print('Hello World!')
        return self.non_numeric_array_pdf.columns

    @property
    def array_pdf(self):
        if False:
            i = 10
            return i + 15
        return pd.concat([self.numeric_array_pdf, self.non_numeric_array_pdf], axis=1)

    @property
    def array_psdf(self):
        if False:
            print('Hello World!')
        return ps.from_pandas(self.array_pdf)

    @property
    def array_df_cols(self):
        if False:
            i = 10
            return i + 15
        return self.array_pdf.columns

    @property
    def complex_pdf(self):
        if False:
            return 10
        psers = {'this_array': self.pser, 'that_array': pd.Series([[2, 3, 4]]), 'this_struct': pd.Series([('x', 1)]), 'that_struct': pd.Series([('a', 2)])}
        return pd.concat(psers, axis=1)

    @property
    def complex_psdf(self):
        if False:
            print('Hello World!')
        pssers = {'this_array': self.psser, 'that_array': ps.Series([[2, 3, 4]]), 'this_struct': ps.Index([('x', 1)]).to_series().reset_index(drop=True), 'that_struct': ps.Index([('a', 2)]).to_series().reset_index(drop=True)}
        return ps.concat(pssers, axis=1)

    def test_add(self):
        if False:
            i = 10
            return i + 15
        (pdf, psdf) = (self.array_pdf, self.array_psdf)
        for col in self.array_df_cols:
            self.assert_eq(pdf[col] + pdf[col], psdf[col] + psdf[col])
        for col in self.numeric_array_df_cols:
            (pser1, psser1) = (pdf[col], psdf[col])
            for other_col in self.numeric_array_df_cols:
                (pser2, psser2) = (pdf[other_col], psdf[other_col])
                self.assert_eq((pser1 + pser2).sort_values(), (psser1 + psser2).sort_values())
        self.assertRaises(TypeError, lambda : psdf['string'] + psdf['bool'])
        self.assertRaises(TypeError, lambda : psdf['string'] + psdf['date'])
        self.assertRaises(TypeError, lambda : psdf['bool'] + psdf['date'])
        for col in self.non_numeric_array_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            self.assert_eq(pser + pser, psser + psser)
        for numeric_col in self.numeric_array_df_cols:
            for non_numeric_col in self.non_numeric_array_df_cols:
                self.assertRaises(TypeError, lambda : psdf[numeric_col] + psdf[non_numeric_col])

    def test_sub(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : self.psser - 'x')
        self.assertRaises(TypeError, lambda : self.psser - 1)
        psdf = self.array_psdf
        for col in self.array_df_cols:
            for other_col in self.array_df_cols:
                self.assertRaises(TypeError, lambda : psdf[col] - psdf[other_col])

    def test_mul(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : self.psser * 'x')
        self.assertRaises(TypeError, lambda : self.psser * 1)
        psdf = self.array_psdf
        for col in self.array_df_cols:
            for other_col in self.array_df_cols:
                self.assertRaises(TypeError, lambda : psdf[col] * psdf[other_col])

    def test_truediv(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : self.psser / 'x')
        self.assertRaises(TypeError, lambda : self.psser / 1)
        psdf = self.array_psdf
        for col in self.array_df_cols:
            for other_col in self.array_df_cols:
                self.assertRaises(TypeError, lambda : psdf[col] / psdf[other_col])

    def test_floordiv(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : self.psser // 'x')
        self.assertRaises(TypeError, lambda : self.psser // 1)
        psdf = self.array_psdf
        for col in self.array_df_cols:
            for other_col in self.array_df_cols:
                self.assertRaises(TypeError, lambda : psdf[col] // psdf[other_col])

    def test_mod(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : self.psser % 'x')
        self.assertRaises(TypeError, lambda : self.psser % 1)
        psdf = self.array_psdf
        for col in self.array_df_cols:
            for other_col in self.array_df_cols:
                self.assertRaises(TypeError, lambda : psdf[col] % psdf[other_col])

    def test_pow(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : self.psser ** 'x')
        self.assertRaises(TypeError, lambda : self.psser ** 1)
        psdf = self.array_psdf
        for col in self.array_df_cols:
            for other_col in self.array_df_cols:
                self.assertRaises(TypeError, lambda : psdf[col] ** psdf[other_col])

    def test_radd(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : 'x' + self.psser)
        self.assertRaises(TypeError, lambda : 1 + self.psser)

    def test_rsub(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : 'x' - self.psser)
        self.assertRaises(TypeError, lambda : 1 - self.psser)

    def test_rmul(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : 'x' * self.psser)
        self.assertRaises(TypeError, lambda : 2 * self.psser)

    def test_rtruediv(self):
        if False:
            for i in range(10):
                print('nop')
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
            return 10
        self.assertRaises(TypeError, lambda : 1 % self.psser)

    def test_rpow(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : 'x' ** self.psser)
        self.assertRaises(TypeError, lambda : 1 ** self.psser)

    def test_and(self):
        if False:
            for i in range(10):
                print('nop')
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
            print('Hello World!')
        self.assertRaises(TypeError, lambda : self.psser | True)
        self.assertRaises(TypeError, lambda : self.psser | False)
        self.assertRaises(TypeError, lambda : self.psser | self.psser)

    def test_ror(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : True | self.psser)
        self.assertRaises(TypeError, lambda : False | self.psser)

    def test_from_to_pandas(self):
        if False:
            while True:
                i = 10
        (pdf, psdf) = (self.array_pdf, self.array_psdf)
        for col in self.array_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            self.assert_eq(pser, psser._to_pandas())
            self.assert_eq(ps.from_pandas(pser), psser)

    def test_isnull(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.array_pdf, self.array_psdf)
        for col in self.array_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            self.assert_eq(pser.isnull(), psser.isnull())

    def test_astype(self):
        if False:
            print('Hello World!')
        self.assert_eq(self.pser.astype(str), self.psser.astype(str))

    def test_neg(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : -self.psser)

    def test_abs(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : abs(self.psser))

    def test_invert(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : ~self.psser)

    def test_eq(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.complex_pdf, self.complex_pdf)
        self.assert_eq(pdf['this_array'] == pdf['that_array'], psdf['this_array'] == psdf['that_array'])
        self.assert_eq(pdf['this_struct'] == pdf['that_struct'], psdf['this_struct'] == psdf['that_struct'])
        self.assert_eq(pdf['this_array'] == pdf['this_array'], psdf['this_array'] == psdf['this_array'])
        self.assert_eq(pdf['this_struct'] == pdf['this_struct'], psdf['this_struct'] == psdf['this_struct'])

    def test_ne(self):
        if False:
            return 10
        (pdf, psdf) = (self.complex_pdf, self.complex_pdf)
        self.assert_eq(pdf['this_array'] != pdf['that_array'], psdf['this_array'] != psdf['that_array'])
        self.assert_eq(pdf['this_struct'] != pdf['that_struct'], psdf['this_struct'] != psdf['that_struct'])
        self.assert_eq(pdf['this_array'] != pdf['this_array'], psdf['this_array'] != psdf['this_array'])
        self.assert_eq(pdf['this_struct'] != pdf['this_struct'], psdf['this_struct'] != psdf['this_struct'])

    def test_lt(self):
        if False:
            return 10
        (pdf, psdf) = (self.complex_pdf, self.complex_pdf)
        self.assert_eq(pdf['this_array'] < pdf['that_array'], psdf['this_array'] < psdf['that_array'])
        self.assert_eq(pdf['this_struct'] < pdf['that_struct'], psdf['this_struct'] < psdf['that_struct'])
        self.assert_eq(pdf['this_array'] < pdf['this_array'], psdf['this_array'] < psdf['this_array'])
        self.assert_eq(pdf['this_struct'] < pdf['this_struct'], psdf['this_struct'] < psdf['this_struct'])

    def test_le(self):
        if False:
            while True:
                i = 10
        (pdf, psdf) = (self.complex_pdf, self.complex_pdf)
        self.assert_eq(pdf['this_array'] <= pdf['that_array'], psdf['this_array'] <= psdf['that_array'])
        self.assert_eq(pdf['this_struct'] <= pdf['that_struct'], psdf['this_struct'] <= psdf['that_struct'])
        self.assert_eq(pdf['this_array'] <= pdf['this_array'], psdf['this_array'] <= psdf['this_array'])
        self.assert_eq(pdf['this_struct'] <= pdf['this_struct'], psdf['this_struct'] <= psdf['this_struct'])

    def test_gt(self):
        if False:
            i = 10
            return i + 15
        (pdf, psdf) = (self.complex_pdf, self.complex_pdf)
        self.assert_eq(pdf['this_array'] > pdf['that_array'], psdf['this_array'] > psdf['that_array'])
        self.assert_eq(pdf['this_struct'] > pdf['that_struct'], psdf['this_struct'] > psdf['that_struct'])
        self.assert_eq(pdf['this_array'] > pdf['this_array'], psdf['this_array'] > psdf['this_array'])
        self.assert_eq(pdf['this_struct'] > pdf['this_struct'], psdf['this_struct'] > psdf['this_struct'])

    def test_ge(self):
        if False:
            print('Hello World!')
        (pdf, psdf) = (self.complex_pdf, self.complex_pdf)
        self.assert_eq(pdf['this_array'] >= pdf['that_array'], psdf['this_array'] >= psdf['that_array'])
        self.assert_eq(pdf['this_struct'] >= pdf['that_struct'], psdf['this_struct'] >= psdf['that_struct'])
        self.assert_eq(pdf['this_array'] >= pdf['this_array'], psdf['this_array'] >= psdf['this_array'])
        self.assert_eq(pdf['this_struct'] >= pdf['this_struct'], psdf['this_struct'] >= psdf['this_struct'])

class ComplexOpsTests(ComplexOpsTestsMixin, OpsTestBase):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.data_type_ops.test_complex_ops import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)