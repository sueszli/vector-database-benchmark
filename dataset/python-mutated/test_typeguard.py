"""
Tests to ensure that typeguard is working as expected.
This mostly contains negative tests as proof that typeguard can catch errors.
"""
import unittest
from numba.tests.support import TestCase, skip_unless_typeguard

def guard_args(val: int):
    if False:
        while True:
            i = 10
    return

def guard_ret(val) -> int:
    if False:
        while True:
            i = 10
    return val

@skip_unless_typeguard
class TestTypeGuard(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        import typeguard
        self._exception_type = getattr(typeguard, 'TypeCheckError', TypeError)

    def test_check_args(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(self._exception_type):
            guard_args(float(1.2))

    def test_check_ret(self):
        if False:
            return 10
        with self.assertRaises(self._exception_type):
            guard_ret(float(1.2))

    def test_check_does_not_work_with_inner_func(self):
        if False:
            return 10

        def guard(val: int) -> int:
            if False:
                return 10
            return
        guard(float(1.2))
if __name__ == '__main__':
    unittest.main()