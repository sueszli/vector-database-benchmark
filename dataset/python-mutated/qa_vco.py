from gnuradio import gr, gr_unittest, blocks
import math

def sig_source_f(samp_rate, freq, amp, N):
    if False:
        return 10
    t = [float(x) / samp_rate for x in range(N)]
    y = [amp * math.cos(2.0 * math.pi * freq * x) for x in t]
    return y

def sig_source_c(samp_rate, freq, amp, N):
    if False:
        i = 10
        return i + 15
    t = [float(x) / samp_rate for x in range(N)]
    y = [math.cos(2.0 * math.pi * freq * x) + 1j * math.sin(2.0 * math.pi * freq * x) for x in t]
    return y

class test_vco(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def test_001(self):
        if False:
            return 10
        src_data = 200 * [0] + 200 * [0.5] + 200 * [1]
        expected_result = 200 * [1] + sig_source_f(1, 0.125, 1, 200) + sig_source_f(1, 0.25, 1, 200)
        src = blocks.vector_source_f(src_data)
        op = blocks.vco_f(1, math.pi / 2.0, 1)
        dst = blocks.vector_sink_f()
        self.tb.connect(src, op, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, result_data, 5)

    def test_002(self):
        if False:
            return 10
        src_data = 200 * [0] + 200 * [0.5] + 200 * [1]
        expected_result = 200 * [1] + sig_source_c(1, 0.125, 1, 200) + sig_source_c(1, 0.25, 1, 200)
        src = blocks.vector_source_f(src_data)
        op = blocks.vco_c(1, math.pi / 2.0, 1)
        dst = blocks.vector_sink_c()
        self.tb.connect(src, op, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertComplexTuplesAlmostEqual(expected_result, result_data, 5)
if __name__ == '__main__':
    gr_unittest.run(test_vco)