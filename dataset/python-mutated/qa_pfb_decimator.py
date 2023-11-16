from gnuradio import gr, gr_unittest, fft, filter, blocks
import math

def sig_source_c(samp_rate, freq, amp, N):
    if False:
        return 10
    t = [float(x) / samp_rate for x in range(N)]
    y = [math.cos(2.0 * math.pi * freq * x) + 1j * math.sin(2.0 * math.pi * freq * x) for x in t]
    return y

def run_test(tb, channel, fft_rotate, fft_filter):
    if False:
        i = 10
        return i + 15
    N = 1000
    M = 5
    fs = 5000.0
    ifs = M * fs
    taps = filter.firdes.low_pass_2(1, ifs, fs / 2, fs / 10, attenuation_dB=80, window=fft.window.WIN_BLACKMAN_hARRIS)
    signals = list()
    add = blocks.add_cc()
    freqs = [-230.0, 121.0, 110.0, -513.0, 203.0]
    Mch = ((len(freqs) - 1) // 2 + channel) % len(freqs)
    for i in range(len(freqs)):
        f = freqs[i] + (M // 2 - M + i + 1) * fs
        data = sig_source_c(ifs, f, 1, N)
        signals.append(blocks.vector_source_c(data))
        tb.connect(signals[i], (add, i))
    s2ss = blocks.stream_to_streams(gr.sizeof_gr_complex, M)
    pfb = filter.pfb_decimator_ccf(M, taps, channel, fft_rotate, fft_filter)
    snk = blocks.vector_sink_c()
    tb.connect(add, s2ss)
    for i in range(M):
        tb.connect((s2ss, i), (pfb, i))
    tb.connect(pfb, snk)
    tb.run()
    L = len(snk.data())
    phase = [0.11058476216852586, 4.510824657140169, 3.9739891674564594, 2.2820531095511924, 1.378279746739787]
    phase = phase[channel]
    tpf = math.ceil(len(taps) / float(M))
    delay = -(tpf - 1.0) / 2.0
    delay = int(delay)
    t = [float(x) / fs for x in range(delay, L + delay)]
    expected_data = [math.cos(2.0 * math.pi * freqs[Mch] * x + phase) + 1j * math.sin(2.0 * math.pi * freqs[Mch] * x + phase) for x in t]
    dst_data = snk.data()
    return (dst_data, expected_data)

class test_pfb_decimator(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tb = None

    def test_000(self):
        if False:
            i = 10
            return i + 15
        Ntest = 50
        (dst_data0, expected_data0) = run_test(self.tb, 0, False, False)
        (dst_data1, expected_data1) = run_test(self.tb, 0, False, True)
        (dst_data2, expected_data2) = run_test(self.tb, 0, True, False)
        (dst_data3, expected_data3) = run_test(self.tb, 0, True, True)
        self.assertComplexTuplesAlmostEqual(expected_data0[-Ntest:], dst_data0[-Ntest:], 4)
        self.assertComplexTuplesAlmostEqual(expected_data1[-Ntest:], dst_data1[-Ntest:], 4)
        self.assertComplexTuplesAlmostEqual(expected_data2[-Ntest:], dst_data2[-Ntest:], 4)
        self.assertComplexTuplesAlmostEqual(expected_data3[-Ntest:], dst_data3[-Ntest:], 4)

    def test_001(self):
        if False:
            for i in range(10):
                print('nop')
        Ntest = 50
        (dst_data0, expected_data0) = run_test(self.tb, 1, False, False)
        (dst_data1, expected_data1) = run_test(self.tb, 1, False, True)
        (dst_data2, expected_data2) = run_test(self.tb, 1, True, False)
        (dst_data3, expected_data3) = run_test(self.tb, 1, True, True)
        self.assertComplexTuplesAlmostEqual(expected_data0[-Ntest:], dst_data0[-Ntest:], 4)
        self.assertComplexTuplesAlmostEqual(expected_data1[-Ntest:], dst_data1[-Ntest:], 4)
        self.assertComplexTuplesAlmostEqual(expected_data2[-Ntest:], dst_data2[-Ntest:], 4)
        self.assertComplexTuplesAlmostEqual(expected_data3[-Ntest:], dst_data3[-Ntest:], 4)

    def test_002(self):
        if False:
            i = 10
            return i + 15
        Ntest = 50
        (dst_data0, expected_data0) = run_test(self.tb, 2, False, False)
        (dst_data1, expected_data1) = run_test(self.tb, 2, False, True)
        (dst_data2, expected_data2) = run_test(self.tb, 2, True, False)
        (dst_data3, expected_data3) = run_test(self.tb, 2, True, True)
        self.assertComplexTuplesAlmostEqual(expected_data0[-Ntest:], dst_data0[-Ntest:], 4)
        self.assertComplexTuplesAlmostEqual(expected_data1[-Ntest:], dst_data1[-Ntest:], 4)
        self.assertComplexTuplesAlmostEqual(expected_data2[-Ntest:], dst_data2[-Ntest:], 4)
        self.assertComplexTuplesAlmostEqual(expected_data3[-Ntest:], dst_data3[-Ntest:], 4)

    def test_003(self):
        if False:
            print('Hello World!')
        Ntest = 50
        (dst_data0, expected_data0) = run_test(self.tb, 3, False, False)
        (dst_data1, expected_data1) = run_test(self.tb, 3, False, True)
        (dst_data2, expected_data2) = run_test(self.tb, 3, True, False)
        (dst_data3, expected_data3) = run_test(self.tb, 3, True, True)
        self.assertComplexTuplesAlmostEqual(expected_data0[-Ntest:], dst_data0[-Ntest:], 4)
        self.assertComplexTuplesAlmostEqual(expected_data1[-Ntest:], dst_data1[-Ntest:], 4)
        self.assertComplexTuplesAlmostEqual(expected_data2[-Ntest:], dst_data2[-Ntest:], 4)
        self.assertComplexTuplesAlmostEqual(expected_data3[-Ntest:], dst_data3[-Ntest:], 4)

    def test_004(self):
        if False:
            i = 10
            return i + 15
        Ntest = 50
        (dst_data0, expected_data0) = run_test(self.tb, 4, False, False)
        (dst_data1, expected_data1) = run_test(self.tb, 4, False, True)
        (dst_data2, expected_data2) = run_test(self.tb, 4, True, False)
        (dst_data3, expected_data3) = run_test(self.tb, 4, True, True)
        self.assertComplexTuplesAlmostEqual(expected_data0[-Ntest:], dst_data0[-Ntest:], 4)
        self.assertComplexTuplesAlmostEqual(expected_data1[-Ntest:], dst_data1[-Ntest:], 4)
        self.assertComplexTuplesAlmostEqual(expected_data2[-Ntest:], dst_data2[-Ntest:], 4)
        self.assertComplexTuplesAlmostEqual(expected_data3[-Ntest:], dst_data3[-Ntest:], 4)
if __name__ == '__main__':
    gr_unittest.run(test_pfb_decimator)