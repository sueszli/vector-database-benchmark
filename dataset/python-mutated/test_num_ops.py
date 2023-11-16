import unittest
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from pyspark import pandas as ps
from pyspark.pandas.config import option_context
from pyspark.pandas.tests.data_type_ops.testing_utils import OpsTestBase
from pyspark.pandas.typedef.typehints import extension_dtypes_available, extension_float_dtypes_available, extension_object_dtypes_available
from pyspark.sql.types import DecimalType, IntegralType

class NumOpsTestsMixin:
    """Unit tests for arithmetic operations of numeric data types.

    A few test cases are disabled because pandas-on-Spark returns float64 whereas pandas
    returns float32.
    The underlying reason is the respective Spark operations return DoubleType always.
    """

    @property
    def float_pser(self):
        if False:
            i = 10
            return i + 15
        return pd.Series([1, 2, 3], dtype=float)

    @property
    def float_psser(self):
        if False:
            for i in range(10):
                print('nop')
        return ps.from_pandas(self.float_pser)

    def test_and(self):
        if False:
            while True:
                i = 10
        psdf = self.psdf
        for col in self.numeric_df_cols:
            psser = psdf[col]
            self.assertRaises(TypeError, lambda : psser & True)
            self.assertRaises(TypeError, lambda : psser & False)
            self.assertRaises(TypeError, lambda : psser & psser)

    def test_rand(self):
        if False:
            while True:
                i = 10
        psdf = self.psdf
        for col in self.numeric_df_cols:
            psser = psdf[col]
            self.assertRaises(TypeError, lambda : True & psser)
            self.assertRaises(TypeError, lambda : False & psser)

    def test_or(self):
        if False:
            for i in range(10):
                print('nop')
        psdf = self.psdf
        for col in self.numeric_df_cols:
            psser = psdf[col]
            self.assertRaises(TypeError, lambda : psser | True)
            self.assertRaises(TypeError, lambda : psser | False)
            self.assertRaises(TypeError, lambda : psser | psser)

    def test_ror(self):
        if False:
            return 10
        psdf = self.psdf
        for col in self.numeric_df_cols:
            psser = psdf[col]
            self.assertRaises(TypeError, lambda : True | psser)
            self.assertRaises(TypeError, lambda : False | psser)

    def test_xor(self):
        if False:
            return 10
        (pdf, psdf) = (self.integral_pdf, self.integral_psdf)
        (pser, other_pser) = (pdf['this'], pdf['that'])
        (psser, other_psser) = (psdf['this'], psdf['that'])
        self.assert_eq(pser ^ other_pser, psser ^ other_psser)
        self.assert_eq(pser ^ 2, psser ^ 2)
        self.assert_eq(pser ^ 3, psser ^ 3)
        self.assert_eq(pser ^ False, psser ^ False)
        self.assert_eq(pser ^ True, psser ^ True)
        with self.assertRaisesRegex(TypeError, 'XOR can not be applied to given types.'):
            psser ^ 'a'
            psser ^ None
        with option_context('compute.ops_on_diff_frames', True):
            (pser, other_pser) = (self.integral_pdf['this'], self.pdf['bool'])
            (psser, other_psser) = (self.integral_psdf['this'], self.psdf['bool'])
            self.assert_eq(pser ^ other_pser, psser ^ other_psser)

    def test_rxor(self):
        if False:
            return 10
        (pser, psser) = (self.pdf['int'], self.psdf['int'])
        self.assert_eq(True ^ pser, True ^ psser)
        self.assert_eq(False ^ pser, False ^ psser)
        self.assert_eq(1 ^ pser, 1 ^ psser)

    def test_from_to_pandas(self):
        if False:
            i = 10
            return i + 15
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            self.assert_eq(pser, psser._to_pandas())
            self.assert_eq(ps.from_pandas(pser), psser)

    def test_isnull(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            self.assert_eq(pdf[col].isnull(), psdf[col].isnull())

    def test_astype(self):
        if False:
            print('Hello World!')
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            for int_type in [int, np.int32, np.int16, np.int8]:
                if not pser.hasnans:
                    self.assert_eq(pser.astype(int_type), psser.astype(int_type))
                else:
                    self.assertRaisesRegex(ValueError, 'Cannot convert %s with missing values to integer' % psser._dtype_op.pretty_name, lambda : psser.astype(int_type))
            if not pser.hasnans:
                self.assert_eq(pser.astype(bool), psser.astype(bool))
            self.assert_eq(pser.astype(float), psser.astype(float))
            self.assert_eq(pser.astype(np.float32), psser.astype(np.float32))
            self.assert_eq(pser.astype(str), psser.astype(str))
            self.assert_eq(pser.astype('category'), psser.astype('category'))
            cat_type = CategoricalDtype(categories=[2, 1, 3])
            self.assert_eq(pser.astype(cat_type), psser.astype(cat_type))
        if extension_object_dtypes_available and extension_float_dtypes_available:
            pser = pd.Series(pd.Categorical([1.0, 2.0, 3.0]), dtype=pd.Float64Dtype())
            psser = ps.from_pandas(pser)
            self.assert_eq(pser.astype(pd.BooleanDtype()), psser.astype(pd.BooleanDtype()))

    def test_astype_eager_check(self):
        if False:
            i = 10
            return i + 15
        psser = self.psdf['float_nan']
        with ps.option_context('compute.eager_check', True), self.assertRaisesRegex(ValueError, 'Cannot convert'):
            psser.astype(int)
        with ps.option_context('compute.eager_check', False):
            psser.astype(int)
        psser = self.psdf['decimal_nan']
        with ps.option_context('compute.eager_check', True), self.assertRaisesRegex(ValueError, 'Cannot convert'):
            psser.astype(int)
        with ps.option_context('compute.eager_check', False):
            psser.astype(int)

    def test_neg(self):
        if False:
            i = 10
            return i + 15
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            self.assert_eq(-pdf[col], -psdf[col])

    def test_abs(self):
        if False:
            print('Hello World!')
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            self.assert_eq(abs(pdf[col]), abs(psdf[col]))

    def test_invert(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            if isinstance(psser.spark.data_type, IntegralType):
                self.assert_eq(~pser, ~psser)
            else:
                self.assertRaises(TypeError, lambda : ~psser)

    def test_eq(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            self.assert_eq(pdf[col] == pdf[col], psdf[col] == psdf[col])

    def test_ne(self):
        if False:
            while True:
                i = 10
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            self.assert_eq(pdf[col] != pdf[col], psdf[col] != psdf[col])

    def test_lt(self):
        if False:
            return 10
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            if isinstance(psser.spark.data_type, DecimalType):
                self.assertRaisesRegex(TypeError, '< can not be applied to', lambda : psser < psser)
            else:
                self.assert_eq(pser < pser, psser < psser)

    def test_le(self):
        if False:
            i = 10
            return i + 15
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            if isinstance(psser.spark.data_type, DecimalType):
                self.assertRaisesRegex(TypeError, '<= can not be applied to', lambda : psser <= psser)
            else:
                self.assert_eq(pser <= pser, psser <= psser)

    def test_gt(self):
        if False:
            return 10
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            if isinstance(psser.spark.data_type, DecimalType):
                self.assertRaisesRegex(TypeError, '> can not be applied to', lambda : psser > psser)
            else:
                self.assert_eq(pser > pser, psser > psser)

    def test_ge(self):
        if False:
            i = 10
            return i + 15
        (pdf, psdf) = (self.pdf, self.psdf)
        for col in self.numeric_df_cols:
            (pser, psser) = (pdf[col], psdf[col])
            if isinstance(psser.spark.data_type, DecimalType):
                self.assertRaisesRegex(TypeError, '>= can not be applied to', lambda : psser >= psser)
            else:
                self.assert_eq(pser >= pser, psser >= psser)

@unittest.skipIf(not extension_dtypes_available, 'pandas extension dtypes are not available')
class IntegralExtensionOpsTest(OpsTestBase):

    @property
    def intergral_extension_psers(self):
        if False:
            for i in range(10):
                print('nop')
        return [pd.Series([1, 2, 3, None], dtype=dtype) for dtype in self.integral_extension_dtypes]

    @property
    def intergral_extension_pssers(self):
        if False:
            print('Hello World!')
        return [ps.from_pandas(pser) for pser in self.intergral_extension_psers]

    @property
    def intergral_extension_pser_psser_pairs(self):
        if False:
            print('Hello World!')
        return zip(self.intergral_extension_psers, self.intergral_extension_pssers)

    def test_from_to_pandas(self):
        if False:
            while True:
                i = 10
        for (pser, psser) in self.intergral_extension_pser_psser_pairs:
            self.check_extension(pser, psser._to_pandas())
            self.check_extension(ps.from_pandas(pser), psser)

    def test_isnull(self):
        if False:
            for i in range(10):
                print('nop')
        for (pser, psser) in self.intergral_extension_pser_psser_pairs:
            self.assert_eq(pser.isnull(), psser.isnull())

    def test_astype(self):
        if False:
            while True:
                i = 10
        for (pser, psser) in self.intergral_extension_pser_psser_pairs:
            for dtype in self.extension_dtypes:
                if dtype in self.string_extension_dtype:
                    self.check_extension(pser.astype(dtype), psser.astype(dtype))
                else:
                    self.check_extension(pser.astype(dtype), psser.astype(dtype))
        for (pser, psser) in self.intergral_extension_pser_psser_pairs:
            self.assert_eq(pser.astype(float), psser.astype(float))
            self.assert_eq(pser.astype(np.float32), psser.astype(np.float32))
            with ps.option_context('compute.eager_check', True):
                self.assertRaisesRegex(ValueError, 'Cannot convert integrals with missing values to bool', lambda : psser.astype(bool))
                self.assertRaisesRegex(ValueError, 'Cannot convert integrals with missing values to integer', lambda : psser.astype(int))
                self.assertRaisesRegex(ValueError, 'Cannot convert integrals with missing values to integer', lambda : psser.astype(np.int32))
            with ps.option_context('compute.eager_check', False):
                psser.astype(bool)
                psser.astype(int)
                psser.astype(np.int32)

    def test_neg(self):
        if False:
            i = 10
            return i + 15
        for (pser, psser) in self.intergral_extension_pser_psser_pairs:
            self.check_extension(-pser, -psser)

    def test_abs(self):
        if False:
            return 10
        for (pser, psser) in self.intergral_extension_pser_psser_pairs:
            self.check_extension(abs(pser), abs(psser))

    def test_invert(self):
        if False:
            i = 10
            return i + 15
        for (pser, psser) in self.intergral_extension_pser_psser_pairs:
            self.check_extension(~pser, ~psser)

    def test_eq(self):
        if False:
            i = 10
            return i + 15
        with option_context('compute.ops_on_diff_frames', True):
            for (pser, psser) in self.intergral_extension_pser_psser_pairs:
                self.check_extension(pser == pser, (psser == psser).sort_index())

    def test_ne(self):
        if False:
            return 10
        with option_context('compute.ops_on_diff_frames', True):
            for (pser, psser) in self.intergral_extension_pser_psser_pairs:
                self.check_extension(pser != pser, (psser != psser).sort_index())

    def test_lt(self):
        if False:
            for i in range(10):
                print('nop')
        with option_context('compute.ops_on_diff_frames', True):
            for (pser, psser) in self.intergral_extension_pser_psser_pairs:
                self.check_extension(pser < pser, (psser < psser).sort_index())

    def test_le(self):
        if False:
            for i in range(10):
                print('nop')
        with option_context('compute.ops_on_diff_frames', True):
            for (pser, psser) in self.intergral_extension_pser_psser_pairs:
                self.check_extension(pser <= pser, (psser <= psser).sort_index())

    def test_gt(self):
        if False:
            while True:
                i = 10
        with option_context('compute.ops_on_diff_frames', True):
            for (pser, psser) in self.intergral_extension_pser_psser_pairs:
                self.check_extension(pser > pser, (psser > psser).sort_index())

    def test_ge(self):
        if False:
            return 10
        with option_context('compute.ops_on_diff_frames', True):
            for (pser, psser) in self.intergral_extension_pser_psser_pairs:
                self.check_extension(pser >= pser, (psser >= psser).sort_index())

    def test_xor(self):
        if False:
            print('Hello World!')
        for psser in self.intergral_extension_pssers:
            self.assertRaisesRegex(TypeError, 'XOR can not be applied to given types.', lambda : psser ^ 1)
            self.assertRaisesRegex(TypeError, 'XOR can not be applied to given types.', lambda : psser ^ psser)
            self.assertRaisesRegex(TypeError, 'XOR can not be applied to given types.', lambda : psser ^ False)

    def test_rxor(self):
        if False:
            return 10
        for psser in self.intergral_extension_pssers:
            self.assertRaisesRegex(TypeError, 'XOR can not be applied to given types.', lambda : 1 ^ psser)
            self.assertRaisesRegex(TypeError, 'XOR can not be applied to given types.', lambda : False ^ psser)

@unittest.skipIf(not extension_float_dtypes_available, 'pandas extension float dtypes are not available')
class FractionalExtensionOpsTest(OpsTestBase):

    @property
    def fractional_extension_psers(self):
        if False:
            i = 10
            return i + 15
        return [pd.Series([0.1, 0.2, 0.3, None], dtype=dtype) for dtype in self.fractional_extension_dtypes]

    @property
    def fractional_extension_pssers(self):
        if False:
            print('Hello World!')
        return [ps.from_pandas(pser) for pser in self.fractional_extension_psers]

    @property
    def fractional_extension_pser_psser_pairs(self):
        if False:
            for i in range(10):
                print('nop')
        return zip(self.fractional_extension_psers, self.fractional_extension_pssers)

    def test_from_to_pandas(self):
        if False:
            return 10
        for (pser, psser) in self.fractional_extension_pser_psser_pairs:
            self.check_extension(pser, psser._to_pandas())
            self.check_extension(ps.from_pandas(pser), psser)

    def test_isnull(self):
        if False:
            print('Hello World!')
        for (pser, psser) in self.fractional_extension_pser_psser_pairs:
            self.assert_eq(pser.isnull(), psser.isnull())

    def test_astype(self):
        if False:
            while True:
                i = 10
        for (pser, psser) in self.fractional_extension_pser_psser_pairs:
            for dtype in self.extension_dtypes:
                self.check_extension(pser.astype(dtype), psser.astype(dtype))
        for (pser, psser) in self.fractional_extension_pser_psser_pairs:
            self.assert_eq(pser.astype(float), psser.astype(float))
            self.assert_eq(pser.astype('category'), psser.astype('category'))
            self.assert_eq(pser.astype(np.float32), psser.astype(np.float32))
            with ps.option_context('compute.eager_check', True):
                self.assertRaisesRegex(ValueError, 'Cannot convert fractions with missing values to bool', lambda : psser.astype(bool))
                self.assertRaisesRegex(ValueError, 'Cannot convert fractions with missing values to integer', lambda : psser.astype(int))
                self.assertRaisesRegex(ValueError, 'Cannot convert fractions with missing values to integer', lambda : psser.astype(np.int32))
            with ps.option_context('compute.eager_check', False):
                psser.astype(bool)
                psser.astype(int)
                psser.astype(np.int32)

    def test_neg(self):
        if False:
            return 10
        for dtype in self.fractional_extension_dtypes:
            self.assert_eq(ps.Series([-0.1, -0.2, -0.3, None], dtype=dtype), -ps.Series([0.1, 0.2, 0.3, None], dtype=dtype))

    def test_abs(self):
        if False:
            i = 10
            return i + 15
        for (pser, psser) in self.fractional_extension_pser_psser_pairs:
            self.check_extension(abs(pser), abs(psser))

    def test_invert(self):
        if False:
            print('Hello World!')
        for psser in self.fractional_extension_pssers:
            self.assertRaises(TypeError, lambda : ~psser)

    def test_eq(self):
        if False:
            return 10
        with option_context('compute.ops_on_diff_frames', True):
            for (pser, psser) in self.fractional_extension_pser_psser_pairs:
                self.check_extension(pser == pser, (psser == psser).sort_index())

    def test_ne(self):
        if False:
            for i in range(10):
                print('nop')
        with option_context('compute.ops_on_diff_frames', True):
            for (pser, psser) in self.fractional_extension_pser_psser_pairs:
                self.check_extension(pser != pser, (psser != psser).sort_index())

    def test_lt(self):
        if False:
            for i in range(10):
                print('nop')
        with option_context('compute.ops_on_diff_frames', True):
            for (pser, psser) in self.fractional_extension_pser_psser_pairs:
                self.check_extension(pser < pser, (psser < psser).sort_index())

    def test_le(self):
        if False:
            print('Hello World!')
        with option_context('compute.ops_on_diff_frames', True):
            for (pser, psser) in self.fractional_extension_pser_psser_pairs:
                self.check_extension(pser <= pser, (psser <= psser).sort_index())

    def test_gt(self):
        if False:
            print('Hello World!')
        with option_context('compute.ops_on_diff_frames', True):
            for (pser, psser) in self.fractional_extension_pser_psser_pairs:
                self.check_extension(pser > pser, (psser > psser).sort_index())

    def test_ge(self):
        if False:
            while True:
                i = 10
        with option_context('compute.ops_on_diff_frames', True):
            for (pser, psser) in self.fractional_extension_pser_psser_pairs:
                self.check_extension(pser >= pser, (psser >= psser).sort_index())

class NumOpsTests(NumOpsTestsMixin, OpsTestBase):
    pass
if __name__ == '__main__':
    from pyspark.pandas.tests.data_type_ops.test_num_ops import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)