import functools
import itertools
import sys
from unittest import skipIf as skipif
from pytest import raises as assert_raises
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TEST_WITH_TORCHDYNAMO, TestCase, xpassIfTorchDynamo
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_
skip = functools.partial(skipif, True)

@xpassIfTorchDynamo
class TestCommonType(TestCase):

    def test_scalar_loses1(self):
        if False:
            return 10
        res = np.find_common_type(['f4', 'f4', 'i2'], ['f8'])
        assert_(res == 'f4')

    def test_scalar_loses2(self):
        if False:
            for i in range(10):
                print('nop')
        res = np.find_common_type(['f4', 'f4'], ['i8'])
        assert_(res == 'f4')

    def test_scalar_wins(self):
        if False:
            return 10
        res = np.find_common_type(['f4', 'f4', 'i2'], ['c8'])
        assert_(res == 'c8')

    def test_scalar_wins2(self):
        if False:
            i = 10
            return i + 15
        res = np.find_common_type(['u4', 'i4', 'i4'], ['f4'])
        assert_(res == 'f8')

    def test_scalar_wins3(self):
        if False:
            print('Hello World!')
        res = np.find_common_type(['u8', 'i8', 'i8'], ['f8'])
        assert_(res == 'f8')

class TestIsSubDType(TestCase):
    wrappers = [np.dtype, lambda x: x]

    def test_both_abstract(self):
        if False:
            for i in range(10):
                print('nop')
        assert_(np.issubdtype(np.floating, np.inexact))
        assert_(not np.issubdtype(np.inexact, np.floating))

    def test_same(self):
        if False:
            return 10
        for cls in (np.float32, np.int32):
            for (w1, w2) in itertools.product(self.wrappers, repeat=2):
                assert_(np.issubdtype(w1(cls), w2(cls)))

    def test_subclass(self):
        if False:
            print('Hello World!')
        for w in self.wrappers:
            assert_(np.issubdtype(w(np.float32), np.floating))
            assert_(np.issubdtype(w(np.float64), np.floating))

    def test_subclass_backwards(self):
        if False:
            for i in range(10):
                print('nop')
        for w in self.wrappers:
            assert_(not np.issubdtype(np.floating, w(np.float32)))
            assert_(not np.issubdtype(np.floating, w(np.float64)))

    def test_sibling_class(self):
        if False:
            while True:
                i = 10
        for (w1, w2) in itertools.product(self.wrappers, repeat=2):
            assert_(not np.issubdtype(w1(np.float32), w2(np.float64)))
            assert_(not np.issubdtype(w1(np.float64), w2(np.float32)))

    def test_nondtype_nonscalartype(self):
        if False:
            i = 10
            return i + 15
        assert not np.issubdtype(np.float32, 'float64')
        assert not np.issubdtype(np.float32, 'f8')
        assert not np.issubdtype(np.int32, 'int64')
        assert not np.issubdtype(np.int8, int)
        assert not np.issubdtype(np.float32, float)
        assert not np.issubdtype(np.complex64, complex)
        assert not np.issubdtype(np.float32, 'float')
        assert not np.issubdtype(np.float64, 'f')
        assert np.issubdtype(np.float64, 'float64')
        assert np.issubdtype(np.float64, 'f8')
        assert np.issubdtype(np.int64, 'int64')
        assert np.issubdtype(np.int8, np.integer)
        assert np.issubdtype(np.float32, np.floating)
        assert np.issubdtype(np.complex64, np.complexfloating)
        assert np.issubdtype(np.float64, 'float')
        assert np.issubdtype(np.float32, 'f')

@xpassIfTorchDynamo
class TestBitName(TestCase):

    def test_abstract(self):
        if False:
            print('Hello World!')
        assert_raises(ValueError, np.core.numerictypes.bitname, np.floating)

@skip(reason='Docstrings for scalar types, not yet.')
@skipif(sys.flags.optimize > 1, reason='no docstrings present to inspect when PYTHONOPTIMIZE/Py_OptimizeFlag > 1')
class TestDocStrings(TestCase):

    def test_platform_dependent_aliases(self):
        if False:
            for i in range(10):
                print('nop')
        if np.int64 is np.int_:
            assert_('int64' in np.int_.__doc__)
        elif np.int64 is np.longlong:
            assert_('int64' in np.longlong.__doc__)

@instantiate_parametrized_tests
class TestScalarTypeNames(TestCase):
    numeric_types = [np.byte, np.short, np.intc, np.int_, np.ubyte, np.half, np.single, np.double, np.csingle, np.cdouble]

    def test_names_are_unique(self):
        if False:
            print('Hello World!')
        assert len(set(self.numeric_types)) == len(self.numeric_types)
        names = [t.__name__ for t in self.numeric_types]
        assert len(set(names)) == len(names)

    @parametrize('t', numeric_types)
    def test_names_reflect_attributes(self, t):
        if False:
            for i in range(10):
                print('nop')
        'Test that names correspond to where the type is under ``np.``'
        assert getattr(np, t.__name__) is t

    @parametrize('t', numeric_types)
    def test_names_are_undersood_by_dtype(self, t):
        if False:
            print('Hello World!')
        'Test the dtype constructor maps names back to the type'
        assert np.dtype(t.__name__).type is t
if __name__ == '__main__':
    run_tests()