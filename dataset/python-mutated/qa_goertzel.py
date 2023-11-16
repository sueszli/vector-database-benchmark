from math import pi, cos
from gnuradio import gr, gr_unittest, fft, blocks

class test_goertzel(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tb = None

    def make_tone_data(self, rate, freq):
        if False:
            while True:
                i = 10
        return [cos(2 * pi * x * freq / rate) for x in range(rate)]

    def transform(self, src_data, rate, freq):
        if False:
            while True:
                i = 10
        src = blocks.vector_source_f(src_data, False)
        dft = fft.goertzel_fc(rate, rate, freq)
        dst = blocks.vector_sink_c()
        self.tb.connect(src, dft, dst)
        self.tb.run()
        return dst.data()

    def test_001(self):
        if False:
            while True:
                i = 10
        rate = 8000
        freq = 100
        bin = freq
        src_data = self.make_tone_data(rate, freq)
        expected_result = 0.5
        actual_result = abs(self.transform(src_data, rate, bin)[0])
        self.assertAlmostEqual(expected_result, actual_result, places=4)

    def test_002(self):
        if False:
            while True:
                i = 10
        rate = 8000
        freq = 100
        bin = freq / 2
        src_data = self.make_tone_data(rate, freq)
        expected_result = 0.0
        actual_result = abs(self.transform(src_data, rate, bin)[0])
        self.assertAlmostEqual(expected_result, actual_result, places=4)
if __name__ == '__main__':
    gr_unittest.run(test_goertzel)