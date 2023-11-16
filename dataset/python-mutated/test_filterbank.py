import numpy as np
from numpy.testing import TestCase, assert_equal, assert_almost_equal
from aubio import cvec, filterbank, float_type
from utils import array_from_text_file

class aubio_filterbank_test_case(TestCase):

    def test_members(self):
        if False:
            while True:
                i = 10
        f = filterbank(40, 512)
        assert_equal([f.n_filters, f.win_s], [40, 512])

    def test_set_coeffs(self):
        if False:
            print('Hello World!')
        f = filterbank(40, 512)
        r = np.random.random([40, int(512 / 2) + 1]).astype(float_type)
        f.set_coeffs(r)
        assert_equal(r, f.get_coeffs())

    def test_phase(self):
        if False:
            print('Hello World!')
        f = filterbank(40, 512)
        c = cvec(512)
        c.phas[:] = np.pi
        assert_equal(f(c), 0)

    def test_norm(self):
        if False:
            for i in range(10):
                print('nop')
        f = filterbank(40, 512)
        c = cvec(512)
        c.norm[:] = 1
        assert_equal(f(c), 0)

    def test_random_norm(self):
        if False:
            i = 10
            return i + 15
        f = filterbank(40, 512)
        c = cvec(512)
        c.norm[:] = np.random.random((int(512 / 2) + 1,)).astype(float_type)
        assert_equal(f(c), 0)

    def test_random_coeffs(self):
        if False:
            i = 10
            return i + 15
        win_s = 128
        f = filterbank(40, win_s)
        c = cvec(win_s)
        r = np.random.random([40, int(win_s / 2) + 1]).astype(float_type)
        r /= r.sum()
        f.set_coeffs(r)
        c.norm[:] = np.random.random((int(win_s / 2) + 1,)).astype(float_type)
        assert_equal(f(c) < 1.0, True)
        assert_equal(f(c) > 0.0, True)

    def test_mfcc_coeffs(self):
        if False:
            while True:
                i = 10
        f = filterbank(40, 512)
        c = cvec(512)
        f.set_mel_coeffs_slaney(44100)
        c.norm[:] = np.random.random((int(512 / 2) + 1,)).astype(float_type)
        assert_equal(f(c) < 1.0, True)
        assert_equal(f(c) > 0.0, True)

    def test_mfcc_coeffs_16000(self):
        if False:
            i = 10
            return i + 15
        expected = array_from_text_file('filterbank_mfcc_16000_512.expected')
        f = filterbank(40, 512)
        f.set_mel_coeffs_slaney(16000)
        assert_almost_equal(expected, f.get_coeffs())

    def test_mfcc_coeffs_get_coeffs(self):
        if False:
            i = 10
            return i + 15
        f = filterbank(40, 512)
        coeffs = f.get_coeffs()
        self.assertIsInstance(coeffs, np.ndarray)
        assert_equal(coeffs, 0)
        assert_equal(np.shape(coeffs), (40, 512 / 2 + 1))

class aubio_filterbank_wrong_values(TestCase):

    def test_negative_window(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ValueError, filterbank, 40, -20)

    def test_negative_filters(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, filterbank, -40, 1024)

    def test_filterbank_long_cvec(self):
        if False:
            print('Hello World!')
        f = filterbank(40, 512)
        with self.assertRaises(ValueError):
            f(cvec(1024))

    def test_filterbank_short_cvec(self):
        if False:
            for i in range(10):
                print('nop')
        f = filterbank(40, 512)
        with self.assertRaises(ValueError):
            f(cvec(256))
if __name__ == '__main__':
    from unittest import main
    main()