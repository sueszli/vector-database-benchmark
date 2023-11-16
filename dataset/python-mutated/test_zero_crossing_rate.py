from numpy.testing import TestCase
from aubio import fvec, zero_crossing_rate
buf_size = 2048

class zero_crossing_rate_test_case(TestCase):

    def setUp(self):
        if False:
            return 10
        self.vector = fvec(buf_size)

    def test_zeroes(self):
        if False:
            while True:
                i = 10
        ' check zero crossing rate on a buffer of 0. '
        self.assertEqual(0.0, zero_crossing_rate(self.vector))

    def test_ones(self):
        if False:
            while True:
                i = 10
        ' check zero crossing rate on a buffer of 1. '
        self.vector[:] = 1.0
        self.assertEqual(0.0, zero_crossing_rate(self.vector))

    def test_impulse(self):
        if False:
            i = 10
            return i + 15
        ' check zero crossing rate on a buffer with an impulse '
        self.vector[int(buf_size / 2)] = 1.0
        self.assertEqual(0.0, zero_crossing_rate(self.vector))

    def test_negative_impulse(self):
        if False:
            i = 10
            return i + 15
        ' check zero crossing rate on a buffer with a negative impulse '
        self.vector[int(buf_size / 2)] = -1.0
        self.assertEqual(2.0 / buf_size, zero_crossing_rate(self.vector))

    def test_single(self):
        if False:
            while True:
                i = 10
        ' check zero crossing rate on single crossing '
        self.vector[int(buf_size / 2) - 1] = 1.0
        self.vector[int(buf_size / 2)] = -1.0
        self.assertEqual(2.0 / buf_size, zero_crossing_rate(self.vector))

    def test_single_with_gap(self):
        if False:
            print('Hello World!')
        ' check zero crossing rate on single crossing with a gap'
        self.vector[int(buf_size / 2) - 2] = 1.0
        self.vector[int(buf_size / 2)] = -1.0
        self.assertEqual(2.0 / buf_size, zero_crossing_rate(self.vector))
if __name__ == '__main__':
    from unittest import main
    main()