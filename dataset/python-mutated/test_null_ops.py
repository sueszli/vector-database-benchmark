import pandas as pd
from pandas.api.types import CategoricalDtype
import pyspark.pandas as ps
from pyspark.pandas.tests.data_type_ops.testing_utils import OpsTestBase

class NullOpsTestsMixin:

    @property
    def pser(self):
        if False:
            print('Hello World!')
        return pd.Series([None, None, None])

    @property
    def psser(self):
        if False:
            print('Hello World!')
        return ps.from_pandas(self.pser)

    def test_add(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : self.psser + 'x')
        self.assertRaises(TypeError, lambda : self.psser + 1)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser + psser)

    def test_sub(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : self.psser - 'x')
        self.assertRaises(TypeError, lambda : self.psser - 1)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser - psser)

    def test_mul(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : self.psser * 'x')
        self.assertRaises(TypeError, lambda : self.psser * 1)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser * psser)

    def test_truediv(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, lambda : self.psser / 'x')
        self.assertRaises(TypeError, lambda : self.psser / 1)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser / psser)

    def test_floordiv(self):
        if False:
            i = 10
            return i + 15
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
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : self.psser ** 'x')
        self.assertRaises(TypeError, lambda : self.psser ** 1)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser ** psser)

    def test_radd(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : 'x' + self.psser)
        self.assertRaises(TypeError, lambda : 1 + self.psser)

    def test_rsub(self):
        if False:
            print('Hello World!')
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
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : 'x' / self.psser)
        self.assertRaises(TypeError, lambda : 1 / self.psser)

    def test_rfloordiv(self):
        if False:
            while True:
                i = 10
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

    def test_from_to_pandas(self):
        if False:
            for i in range(10):
                print('nop')
        data = [None, None, None]
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
            while True:
                i = 10
        pser = self.pser
        psser = self.psser
        self.assert_eq(pser.astype(str), psser.astype(str))
        self.assert_eq(pser.astype(bool), psser.astype(bool))
        self.assert_eq(pser.astype('category'), psser.astype('category'))
        cat_type = CategoricalDtype(categories=[1, 2, 3])
        self.assert_eq(pser.astype(cat_type), psser.astype(cat_type))

    def test_neg(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : -self.psser)

    def test_abs(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : abs(self.psser))

    def test_invert(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : ~self.psser)

    def test_eq(self):
        if False:
            i = 10
            return i + 15
        (pser, psser) = (self.pser, self.psser)
        self.assert_eq(pser == pser, psser == psser)
        self.assert_eq(pser == [None, 1, None], psser == [None, 1, None])

    def test_ne(self):
        if False:
            print('Hello World!')
        (pser, psser) = (self.pser, self.psser)
        self.assert_eq(pser != pser, psser != psser)

    def test_lt(self):
        if False:
            return 10
        (pser, psser) = (self.pser, self.psser)
        self.assert_eq(pser < pser, psser < psser)

    def test_le(self):
        if False:
            while True:
                i = 10
        (pser, psser) = (self.pser, self.psser)
        self.assert_eq(pser <= pser, psser <= psser)

    def test_gt(self):
        if False:
            i = 10
            return i + 15
        (pser, psser) = (self.pser, self.psser)
        self.assert_eq(pser > pser, psser > psser)

    def test_ge(self):
        if False:
            return 10
        (pser, psser) = (self.pser, self.psser)
        self.assert_eq(pser >= pser, psser >= psser)

class NullOpsTests(NullOpsTestsMixin, OpsTestBase):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.data_type_ops.test_null_ops import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)