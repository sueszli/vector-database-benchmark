import numpy as np
from numpy.testing import TestCase
from numpy.testing import assert_equal, assert_almost_equal
from aubio import window, level_lin, db_spl, silence_detection, level_detection
from aubio import fvec, float_type

class aubio_window(TestCase):

    def test_accept_name_and_size(self):
        if False:
            i = 10
            return i + 15
        window('default', 1024)

    def test_fail_name_not_string(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            window(10, 1024)

    def test_fail_size_not_int(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            window('default', 'default')

    def test_compute_hanning_1024(self):
        if False:
            i = 10
            return i + 15
        size = 1024
        aubio_window = window('hanning', size)
        numpy_window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(size) / size)
        assert_almost_equal(aubio_window, numpy_window)

class aubio_level_lin(TestCase):

    def test_accept_fvec(self):
        if False:
            print('Hello World!')
        level_lin(fvec(1024))

    def test_fail_not_fvec(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            level_lin('default')

    def test_zeros_is_zeros(self):
        if False:
            print('Hello World!')
        assert_equal(level_lin(fvec(1024)), 0.0)

    def test_minus_ones_is_one(self):
        if False:
            print('Hello World!')
        assert_equal(level_lin(-np.ones(1024, dtype=float_type)), 1.0)

class aubio_db_spl(TestCase):

    def test_accept_fvec(self):
        if False:
            i = 10
            return i + 15
        db_spl(fvec(1024))

    def test_fail_not_fvec(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            db_spl('default')

    def test_zeros_is_inf(self):
        if False:
            print('Hello World!')
        assert np.isinf(db_spl(fvec(1024)))

    def test_minus_ones_is_zero(self):
        if False:
            for i in range(10):
                print('nop')
        assert_equal(db_spl(-np.ones(1024, dtype=float_type)), 0.0)

class aubio_silence_detection(TestCase):

    def test_accept_fvec(self):
        if False:
            i = 10
            return i + 15
        silence_detection(fvec(1024), -70.0)

    def test_fail_not_fvec(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            silence_detection('default', -70)

    def test_zeros_is_one(self):
        if False:
            i = 10
            return i + 15
        assert silence_detection(fvec(1024), -70) == 1

    def test_minus_ones_is_zero(self):
        if False:
            return 10
        from numpy import ones
        assert silence_detection(ones(1024, dtype=float_type), -70) == 0

class aubio_level_detection(TestCase):

    def test_accept_fvec(self):
        if False:
            i = 10
            return i + 15
        level_detection(fvec(1024), -70.0)

    def test_fail_not_fvec(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            level_detection('default', -70)

    def test_zeros_is_one(self):
        if False:
            while True:
                i = 10
        assert level_detection(fvec(1024), -70) == 1

    def test_minus_ones_is_zero(self):
        if False:
            print('Hello World!')
        from numpy import ones
        assert level_detection(ones(1024, dtype=float_type), -70) == 0
if __name__ == '__main__':
    from unittest import main
    main()