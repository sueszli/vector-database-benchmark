"""
Unit tests for fft.window
"""
import numpy
from gnuradio import gr_unittest
from gnuradio import fft

class test_window(gr_unittest.TestCase):
    """
    Unit tests for fft.window
    """

    def setUp(self):
        if False:
            return 10
        pass

    def tearDown(self):
        if False:
            print('Hello World!')
        pass

    def test_normwin(self):
        if False:
            while True:
                i = 10
        '\n        Verify window normalization\n        '
        win = fft.window.build(fft.window.WIN_BLACKMAN_hARRIS, 21, normalize=True)
        power = numpy.sum([x * x for x in win]) / len(win)
        self.assertAlmostEqual(power, 1.0, places=6)
if __name__ == '__main__':
    gr_unittest.run(test_window)