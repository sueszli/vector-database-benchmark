import unittest
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pyspark import pandas as ps
from pyspark.pandas.config import option_context
from pyspark.pandas.tests.data_type_ops.testing_utils import OpsTestBase
from pyspark.pandas.typedef.typehints import extension_object_dtypes_available
if extension_object_dtypes_available:
    from pandas import StringDtype

class StringOpsTestsMixin:

    @property
    def bool_pdf(self):
        if False:
            for i in range(10):
                print('nop')
        return pd.DataFrame({'this': ['x', 'y', 'z'], 'that': ['z', 'y', 'x']})

    @property
    def bool_psdf(self):
        if False:
            return 10
        return ps.from_pandas(self.bool_pdf)

    @property
    def bool_non_numeric_pdf(self):
        if False:
            while True:
                i = 10
        return pd.concat([self.bool_pdf, self.non_numeric_pdf], axis=1)

    @property
    def bool_non_numeric_psdf(self):
        if False:
            i = 10
            return i + 15
        return ps.from_pandas(self.bool_non_numeric_pdf)

    def test_add(self):
        if False:
            i = 10
            return i + 15
        (pdf, psdf) = (self.bool_non_numeric_pdf, self.bool_non_numeric_psdf)
        (pser, psser) = (pdf['this'], psdf['this'])
        (other_pser, other_psser) = (pdf['that'], psdf['that'])
        self.assert_eq(pser + 'x', psser + 'x')
        self.assertRaises(TypeError, lambda : psser + 1)
        self.assert_eq(pser + other_pser, psser + other_psser)
        for col in self.non_numeric_df_cols:
            if col != 'string':
                self.assertRaises(TypeError, lambda : psser + psdf[col])

    def test_sub(self):
        if False:
            for i in range(10):
                print('nop')
        psdf = self.psdf
        self.assertRaises(TypeError, lambda : psdf['string'] - 'x')
        self.assertRaises(TypeError, lambda : psdf['string'] - 1)
        for col in self.df_cols:
            self.assertRaises(TypeError, lambda : psdf['string'] - psdf[col])

    def test_mul(self):
        if False:
            print('Hello World!')
        (pdf, psdf) = (self.pdf, self.psdf)
        self.assertRaises(TypeError, lambda : psdf['string'] * 'x')
        self.assert_eq(pdf['string'] * 1, psdf['string'] * 1)
        for col in self.df_cols:
            if col in ['int', 'int32']:
                self.assert_eq(pdf['string'] * pdf[col], psdf['string'] * psdf[col])
            else:
                self.assertRaises(TypeError, lambda : psdf['string'] * psdf[col])

    def test_truediv(self):
        if False:
            return 10
        psdf = self.psdf
        self.assertRaises(TypeError, lambda : psdf['string'] / 'x')
        self.assertRaises(TypeError, lambda : psdf['string'] / 1)
        for col in self.df_cols:
            self.assertRaises(TypeError, lambda : psdf['string'] / psdf[col])

    def test_floordiv(self):
        if False:
            for i in range(10):
                print('nop')
        psdf = self.psdf
        self.assertRaises(TypeError, lambda : psdf['string'] // 'x')
        self.assertRaises(TypeError, lambda : psdf['string'] // 1)
        for col in self.df_cols:
            self.assertRaises(TypeError, lambda : psdf['string'] // psdf[col])

    def test_mod(self):
        if False:
            print('Hello World!')
        psdf = self.psdf
        self.assertRaises(TypeError, lambda : psdf['string'] % 'x')
        self.assertRaises(TypeError, lambda : psdf['string'] % 1)
        for col in self.df_cols:
            self.assertRaises(TypeError, lambda : psdf['string'] % psdf[col])

    def test_pow(self):
        if False:
            print('Hello World!')
        psdf = self.psdf
        self.assertRaises(TypeError, lambda : psdf['string'] ** 'x')
        self.assertRaises(TypeError, lambda : psdf['string'] ** 1)
        for col in self.df_cols:
            self.assertRaises(TypeError, lambda : psdf['string'] ** psdf[col])

    def test_radd(self):
        if False:
            i = 10
            return i + 15
        self.assert_eq('x' + self.pdf['string'], 'x' + self.psdf['string'])
        self.assertRaises(TypeError, lambda : 1 + self.psdf['string'])

    def test_rsub(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : 'x' - self.psdf['string'])
        self.assertRaises(TypeError, lambda : 1 - self.psdf['string'])

    def test_rmul(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : 'x' * self.psdf['string'])
        self.assert_eq(1 * self.pdf['string'], 1 * self.psdf['string'])

    def test_rtruediv(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : 'x' / self.psdf['string'])
        self.assertRaises(TypeError, lambda : 1 / self.psdf['string'])

    def test_rfloordiv(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : 'x' // self.psdf['string'])
        self.assertRaises(TypeError, lambda : 1 // self.psdf['string'])

    def test_rmod(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : 1 % self.psdf['string'])

    def test_rpow(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : 'x' ** self.psdf['string'])
        self.assertRaises(TypeError, lambda : 1 ** self.psdf['string'])

    def test_and(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : self.psdf['string'] & True)
        self.assertRaises(TypeError, lambda : self.psdf['string'] & False)
        self.assertRaises(TypeError, lambda : self.psdf['string'] & self.psdf['string'])

    def test_rand(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : True & self.psdf['string'])
        self.assertRaises(TypeError, lambda : False & self.psdf['string'])

    def test_or(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : self.psdf['string'] | True)
        self.assertRaises(TypeError, lambda : self.psdf['string'] | False)
        self.assertRaises(TypeError, lambda : self.psdf['string'] | self.psdf['string'])

    def test_ror(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : True | self.psdf['string'])
        self.assertRaises(TypeError, lambda : False | self.psdf['string'])

    def test_from_to_pandas(self):
        if False:
            while True:
                i = 10
        data = ['x', 'y', 'z']
        pser = pd.Series(data)
        psser = ps.Series(data)
        self.assert_eq(pser, psser._to_pandas())
        self.assert_eq(ps.from_pandas(pser), psser)

    def test_isnull(self):
        if False:
            return 10
        self.assert_eq(self.pdf['string'].isnull(), self.psdf['string'].isnull())

    def test_astype(self):
        if False:
            for i in range(10):
                print('nop')
        pser = pd.Series(['1', '2', '3'])
        psser = ps.from_pandas(pser)
        self.assert_eq(pser.astype(int), psser.astype(int))
        self.assert_eq(pser.astype(float), psser.astype(float))
        self.assert_eq(pser.astype(np.float32), psser.astype(np.float32))
        self.assert_eq(pser.astype(np.int32), psser.astype(np.int32))
        self.assert_eq(pser.astype(np.int16), psser.astype(np.int16))
        self.assert_eq(pser.astype(np.int8), psser.astype(np.int8))
        self.assert_eq(pser.astype(str), psser.astype(str))
        self.assert_eq(pser.astype(bool), psser.astype(bool))
        self.assert_eq(pser.astype('category'), psser.astype('category'))
        cat_type = CategoricalDtype(categories=['3', '1', '2'])
        self.assert_eq(pser.astype(cat_type), psser.astype(cat_type))

    def test_neg(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : -self.psdf['string'])

    def test_abs(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : abs(self.psdf['string']))

    def test_invert(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : ~self.psdf['string'])

    def test_eq(self):
        if False:
            return 10
        (pdf, psdf) = (self.bool_non_numeric_pdf, self.bool_non_numeric_psdf)
        (pser, psser) = (pdf['this'], psdf['this'])
        (other_pser, other_psser) = (pdf['that'], psdf['that'])
        self.assert_eq(pser == other_pser, psser == other_psser)
        self.assert_eq(pser == pser, psser == psser)

    def test_ne(self):
        if False:
            i = 10
            return i + 15
        (pdf, psdf) = (self.bool_non_numeric_pdf, self.bool_non_numeric_psdf)
        (pser, psser) = (pdf['this'], psdf['this'])
        (other_pser, other_psser) = (pdf['that'], psdf['that'])
        self.assert_eq(pser != other_pser, psser != other_psser)
        self.assert_eq(pser != pser, psser != psser)

    def test_lt(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.bool_non_numeric_pdf, self.bool_non_numeric_psdf)
        (pser, psser) = (pdf['this'], psdf['this'])
        (other_pser, other_psser) = (pdf['that'], psdf['that'])
        self.assert_eq(pser < other_pser, psser < other_psser)
        self.assert_eq(pser < pser, psser < psser)

    def test_le(self):
        if False:
            while True:
                i = 10
        (pdf, psdf) = (self.bool_non_numeric_pdf, self.bool_non_numeric_psdf)
        (pser, psser) = (pdf['this'], psdf['this'])
        (other_pser, other_psser) = (pdf['that'], psdf['that'])
        self.assert_eq(pser <= other_pser, psser <= other_psser)
        self.assert_eq(pser <= pser, psser <= psser)

    def test_gt(self):
        if False:
            while True:
                i = 10
        (pdf, psdf) = (self.bool_non_numeric_pdf, self.bool_non_numeric_psdf)
        (pser, psser) = (pdf['this'], psdf['this'])
        (other_pser, other_psser) = (pdf['that'], psdf['that'])
        self.assert_eq(pser > other_pser, psser > other_psser)
        self.assert_eq(pser > pser, psser > psser)

    def test_ge(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.bool_non_numeric_pdf, self.bool_non_numeric_psdf)
        (pser, psser) = (pdf['this'], psdf['this'])
        (other_pser, other_psser) = (pdf['that'], psdf['that'])
        self.assert_eq(pser >= other_pser, psser >= other_psser)
        self.assert_eq(pser >= pser, psser >= psser)

class StringOpsTests(StringOpsTestsMixin, OpsTestBase):
    pass

@unittest.skipIf(not extension_object_dtypes_available, 'pandas extension object dtypes are not available')
class StringExtensionOpsTest(StringOpsTests):

    @property
    def pser(self):
        if False:
            print('Hello World!')
        return pd.Series(['x', 'y', 'z', None], dtype='string')

    @property
    def psser(self):
        if False:
            return 10
        return ps.from_pandas(self.pser)

    @property
    def other_pser(self):
        if False:
            return 10
        return pd.Series([None, 'z', 'y', 'x'], dtype='string')

    @property
    def other_psser(self):
        if False:
            for i in range(10):
                print('nop')
        return ps.from_pandas(self.other_pser)

    def test_radd(self):
        if False:
            i = 10
            return i + 15
        self.assert_eq('x' + self.pser, ('x' + self.psser).astype('string'))
        self.assertRaises(TypeError, lambda : 1 + self.psser)

    def test_mul(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : self.psser * 'x')
        self.assert_eq(self.pser * 1, self.psser * 1)
        with option_context('compute.ops_on_diff_frames', True):
            for (pser, psser) in self.pser_psser_pairs:
                if psser.dtype in [np.int32, np.int64]:
                    self.assert_eq(ps.Series(['x', 'yy', 'zzz', None]).astype('string'), (self.psser * psser).sort_index())
                else:
                    self.assertRaises(TypeError, lambda : self.psser * psser)

    def test_from_to_pandas(self):
        if False:
            i = 10
            return i + 15
        data = ['x', 'y', 'z', None]
        pser = pd.Series(data, dtype='string')
        psser = ps.Series(data, dtype='string')
        self.assert_eq(pser, psser._to_pandas())
        self.assert_eq(ps.from_pandas(pser), psser)

    def test_isnull(self):
        if False:
            print('Hello World!')
        self.assert_eq(self.pser.isnull(), self.psser.isnull())

    def test_astype(self):
        if False:
            print('Hello World!')
        pser = self.pser
        psser = self.psser
        self.assert_eq(pser.astype(str).tolist(), psser.astype(str).tolist())
        self.assert_eq(pser.astype('category'), psser.astype('category'))
        cat_type = CategoricalDtype(categories=['x', 'y'])
        self.assert_eq(pser.astype(cat_type), psser.astype(cat_type))
        for dtype in self.object_extension_dtypes:
            if dtype in ['string', StringDtype()]:
                self.check_extension(pser.astype(dtype), psser.astype(dtype))

    def test_eq(self):
        if False:
            return 10
        with option_context('compute.ops_on_diff_frames', True):
            self.check_extension(self.pser == self.other_pser, (self.psser == self.other_psser).sort_index())
            self.check_extension(self.pser == self.pser, (self.psser == self.psser).sort_index())

    def test_ne(self):
        if False:
            return 10
        with option_context('compute.ops_on_diff_frames', True):
            self.check_extension(self.pser != self.other_pser, (self.psser != self.other_psser).sort_index())
            self.check_extension(self.pser != self.pser, (self.psser != self.psser).sort_index())

    def test_lt(self):
        if False:
            while True:
                i = 10
        with option_context('compute.ops_on_diff_frames', True):
            self.check_extension(self.pser < self.other_pser, (self.psser < self.other_psser).sort_index())
            self.check_extension(self.pser < self.pser, (self.psser < self.psser).sort_index())

    def test_le(self):
        if False:
            return 10
        with option_context('compute.ops_on_diff_frames', True):
            self.check_extension(self.pser <= self.other_pser, (self.psser <= self.other_psser).sort_index())
            self.check_extension(self.pser <= self.pser, (self.psser <= self.psser).sort_index())

    def test_gt(self):
        if False:
            while True:
                i = 10
        with option_context('compute.ops_on_diff_frames', True):
            self.check_extension(self.pser > self.other_pser, (self.psser > self.other_psser).sort_index())
            self.check_extension(self.pser > self.pser, (self.psser > self.psser).sort_index())

    def test_ge(self):
        if False:
            return 10
        with option_context('compute.ops_on_diff_frames', True):
            self.check_extension(self.pser >= self.other_pser, (self.psser >= self.other_psser).sort_index())
            self.check_extension(self.pser >= self.pser, (self.psser >= self.psser).sort_index())
if __name__ == '__main__':
    from pyspark.pandas.tests.data_type_ops.test_string_ops import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)