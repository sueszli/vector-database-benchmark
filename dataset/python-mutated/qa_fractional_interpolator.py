from gnuradio import gr, gr_unittest, filter, blocks
import math

def sig_source_f(samp_rate, freq, amp, N):
    if False:
        for i in range(10):
            print('nop')
    t = [float(x) / samp_rate for x in range(N)]
    y = [math.sin(2.0 * math.pi * freq * x) for x in t]
    return y

def sig_source_c(samp_rate, freq, amp, N):
    if False:
        return 10
    t = [float(x) / samp_rate for x in range(N)]
    y = [math.cos(2.0 * math.pi * freq * x) + 1j * math.sin(2.0 * math.pi * freq * x) for x in t]
    return y

class test_mmse_resampler(gr_unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tb = None

    def test_001_ff(self):
        if False:
            i = 10
            return i + 15
        N = 10000
        fs = 1000
        rrate = 1.123
        freq = 10
        data = sig_source_f(fs, freq, 1, N)
        signal = blocks.vector_source_f(data)
        op = filter.mmse_resampler_ff(0, rrate)
        snk = blocks.vector_sink_f()
        self.tb.connect(signal, op, snk)
        self.tb.run()
        Ntest = 5000
        L = len(snk.data())
        t = [float(x) / (fs / rrate) for x in range(L)]
        phase = 0.1884
        expected_data = [math.sin(2.0 * math.pi * freq * x + phase) for x in t]
        dst_data = snk.data()
        self.assertFloatTuplesAlmostEqual(expected_data[-Ntest:], dst_data[-Ntest:], 3)

    def test_002_cc(self):
        if False:
            for i in range(10):
                print('nop')
        N = 10000
        fs = 1000
        rrate = 1.123
        freq = 10
        data = sig_source_c(fs, freq, 1, N)
        signal = blocks.vector_source_c(data)
        op = filter.mmse_resampler_cc(0.0, rrate)
        snk = blocks.vector_sink_c()
        self.tb.connect(signal, op, snk)
        self.tb.run()
        Ntest = 5000
        L = len(snk.data())
        t = [float(x) / (fs / rrate) for x in range(L)]
        phase = 0.1884
        expected_data = [math.cos(2.0 * math.pi * freq * x + phase) + 1j * math.sin(2.0 * math.pi * freq * x + phase) for x in t]
        dst_data = snk.data()
        self.assertComplexTuplesAlmostEqual(expected_data[-Ntest:], dst_data[-Ntest:], 3)
if __name__ == '__main__':
    gr_unittest.run(test_mmse_resampler)