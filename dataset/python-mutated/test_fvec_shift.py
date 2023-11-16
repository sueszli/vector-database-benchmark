import numpy as np
from numpy.testing import TestCase, assert_equal
import aubio

class aubio_shift_test_case(TestCase):

    def run_shift_ishift(self, n):
        if False:
            print('Hello World!')
        ramp = np.arange(n, dtype=aubio.float_type)
        half = n - n // 2
        expected = np.concatenate([np.arange(half, n), np.arange(half)])
        assert_equal(aubio.shift(ramp), expected)
        assert_equal(ramp, expected)
        expected = np.arange(n)
        assert_equal(aubio.ishift(ramp), expected)
        assert_equal(ramp, expected)

    def test_can_shift_fvec(self):
        if False:
            i = 10
            return i + 15
        self.run_shift_ishift(10)

    def test_can_shift_fvec_odd(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_shift_ishift(7)
if __name__ == '__main__':
    from unittest import main
    main()