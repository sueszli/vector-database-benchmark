"""
Unit-tests for `typedobjectutils.py`
"""
import warnings
from numba.core import types
from numba.tests.support import TestCase
from numba.typed.typedobjectutils import _sentry_safe_cast

class TestTypedObjectUtils(TestCase):

    def test_sentry_safe_cast_warnings(self):
        if False:
            while True:
                i = 10
        warn_cases = []
        warn_cases += [(types.int32, types.int16), (types.int32, types.uint32), (types.int64, types.uint32), (types.float64, types.float32), (types.complex128, types.complex64), (types.int32, types.float32), (types.int64, types.float32), (types.Tuple([types.int32]), types.Tuple([types.float32]))]
        for (fromty, toty) in warn_cases:
            with self.subTest(fromty=fromty, toty=toty):
                with warnings.catch_warnings(record=True) as w:
                    _sentry_safe_cast(fromty, toty)
                self.assertEqual(len(w), 1)
                self.assertIn('unsafe cast from {} to {}'.format(fromty, toty), str(w[0]))

    def test_sentry_safe_cast_no_warn(self):
        if False:
            i = 10
            return i + 15
        ok_cases = []
        ok_cases += [(types.int32, types.int64), (types.uint8, types.int32), (types.float32, types.float64), (types.complex64, types.complex128), (types.int32, types.float64), (types.uint8, types.float32), (types.float32, types.complex128), (types.float64, types.complex128), (types.Tuple([types.int32]), types.Tuple([types.int64]))]
        for (fromty, toty) in ok_cases:
            with self.subTest(fromty=fromty, toty=toty):
                with warnings.catch_warnings(record=True) as w:
                    _sentry_safe_cast(fromty, toty)
                self.assertEqual(len(w), 0)