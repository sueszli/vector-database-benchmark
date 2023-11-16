from numpy.testing import TestCase
from _tools import parametrize, skipTest
import numpy as np
import aubio

class aubio_pitchshift(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        try:
            self.o = aubio.pitchshift(hop_size=128)
        except RuntimeError as e:
            self.skipTest('creating aubio.pitchshift {}'.format(e))

    def test_default_creation(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.o.get_pitchscale(), 1)
        self.assertEqual(self.o.get_transpose(), 0)

    def test_on_zeros(self):
        if False:
            i = 10
            return i + 15
        test_length = self.o.hop_size * 100
        read = 0
        vec = aubio.fvec(self.o.hop_size)
        transpose_range = 24
        while read < test_length:
            out = self.o(vec)
            self.assertTrue((out == 0).all())
            percent_read = read / float(test_length)
            transpose = 2 * transpose_range * percent_read - transpose_range
            self.o.set_transpose(transpose)
            read += len(vec)

    def test_on_ones(self):
        if False:
            print('Hello World!')
        test_length = self.o.hop_size * 100
        read = 0
        vec = aubio.fvec(self.o.hop_size) + 1
        transpose_range = 1.24
        while read < test_length:
            out = self.o(vec)
            percent_read = read / float(test_length)
            transpose = 2 * transpose_range * percent_read - transpose_range
            self.o.set_transpose(transpose)
            read += len(vec)

    def test_transpose_too_high(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            self.o.set_transpose(24.3)

    def test_transpose_too_low(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            self.o.set_transpose(-24.3)

class aubio_pitchshift_wrong_params(TestCase):

    def test_wrong_transpose(self):
        if False:
            print('Hello World!')
        with self.assertRaises(RuntimeError):
            aubio.pitchshift('default', -123)

class Test_aubio_pitchshift_testruns(object):
    run_args = ['mode', 'pitchscale', 'hop_size', 'samplerate']
    run_values = [('default', 1.2, 128, 44100), ('crispness:0', 0.43, 64, 8000), ('crispness:3', 0.53, 256, 8000), ('crispness:3', 1.53, 512, 8000), ('crispness:6', 2.3, 4096, 192000)]

    @parametrize(run_args, run_values)
    def test_run_with_params(self, mode, pitchscale, hop_size, samplerate):
        if False:
            i = 10
            return i + 15
        try:
            self.o = aubio.pitchshift(mode, pitchscale, hop_size, samplerate)
        except RuntimeError as e:
            skipTest('failed creating pitchshift ({})'.format(e))
        test_length = self.o.hop_size * 50
        read = 0
        vec = np.random.rand(self.o.hop_size).astype(aubio.float_type)
        transpose_range = self.o.get_transpose()
        while read < test_length:
            out = self.o(vec)
            percent_read = read / float(test_length)
            transpose = transpose_range - 2 * transpose_range * percent_read
            self.o.set_transpose(transpose)
            read += len(vec)
if __name__ == '__main__':
    from _tools import run_module_suite
    run_module_suite()