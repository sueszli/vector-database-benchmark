import pandas as pd
import pyspark.pandas as ps
from pyspark.ml.linalg import SparseVector
from pyspark.pandas.tests.data_type_ops.testing_utils import OpsTestBase

class UDTOpsTestsMixin:

    @property
    def pser(self):
        if False:
            while True:
                i = 10
        sparse_values = {0: 0.1, 1: 1.1}
        return pd.Series([SparseVector(len(sparse_values), sparse_values)])

    @property
    def psser(self):
        if False:
            i = 10
            return i + 15
        return ps.from_pandas(self.pser)

    @property
    def udt_pdf(self):
        if False:
            while True:
                i = 10
        sparse_values = {0: 0.2, 1: 1.0}
        psers = {'this': self.pser, 'that': pd.Series([SparseVector(len(sparse_values), sparse_values)])}
        return pd.concat(psers, axis=1)

    @property
    def udt_psdf(self):
        if False:
            for i in range(10):
                print('nop')
        return ps.from_pandas(self.udt_pdf)

    def test_add(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, lambda : self.psser + 'x')
        self.assertRaises(TypeError, lambda : self.psser + 1)
        for psser in self.pssers:
            self.assertRaises(TypeError, lambda : self.psser + psser)

    def test_sub(self):
        if False:
            i = 10
            return i + 15
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
            while True:
                i = 10
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
            print('Hello World!')
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
            return 10
        self.assertRaises(TypeError, lambda : 'x' - self.psser)
        self.assertRaises(TypeError, lambda : 1 - self.psser)

    def test_rmul(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : 'x' * self.psser)
        self.assertRaises(TypeError, lambda : 2 * self.psser)

    def test_rtruediv(self):
        if False:
            print('Hello World!')
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
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : 'x' ** self.psser)
        self.assertRaises(TypeError, lambda : 1 ** self.psser)

    def test_from_to_pandas(self):
        if False:
            while True:
                i = 10
        sparse_values = {0: 0.1, 1: 1.1}
        sparse_vector = SparseVector(len(sparse_values), sparse_values)
        pser = pd.Series([sparse_vector])
        psser = ps.Series([sparse_vector])
        self.assert_eq(pser, psser._to_pandas())
        self.assert_eq(ps.from_pandas(pser), psser)

    def test_isnull(self):
        if False:
            return 10
        self.assert_eq(self.pser.isnull(), self.psser.isnull())

    def test_astype(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, lambda : self.psser.astype(str))

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
            return 10
        self.assertRaises(TypeError, lambda : ~self.psser)

    def test_eq(self):
        if False:
            return 10
        (pdf, psdf) = (self.udt_pdf, self.udt_psdf)
        self.assert_eq(pdf['this'] == pdf['this'], psdf['this'] == psdf['this'])
        self.assert_eq(pdf['this'] == pdf['that'], psdf['this'] == psdf['that'])

    def test_ne(self):
        if False:
            for i in range(10):
                print('nop')
        (pdf, psdf) = (self.udt_pdf, self.udt_psdf)
        self.assert_eq(pdf['this'] != pdf['this'], psdf['this'] != psdf['this'])
        self.assert_eq(pdf['this'] != pdf['that'], psdf['this'] != psdf['that'])

    def test_lt(self):
        if False:
            while True:
                i = 10
        self.assertRaisesRegex(TypeError, '< can not be applied to', lambda : self.psser < self.psser)

    def test_le(self):
        if False:
            while True:
                i = 10
        self.assertRaisesRegex(TypeError, '<= can not be applied to', lambda : self.psser <= self.psser)

    def test_gt(self):
        if False:
            while True:
                i = 10
        self.assertRaisesRegex(TypeError, '> can not be applied to', lambda : self.psser > self.psser)

    def test_ge(self):
        if False:
            i = 10
            return i + 15
        self.assertRaisesRegex(TypeError, '>= can not be applied to', lambda : self.psser >= self.psser)

class UDTOpsTests(UDTOpsTestsMixin, OpsTestBase):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.data_type_ops.test_udt_ops import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)