from gnuradio import gr, gr_unittest, fft, filter, blocks
import math

def sig_source_c(samp_rate, freq, amp, N):
    if False:
        print('Hello World!')
    t = [float(x) / samp_rate for x in range(N)]
    y = [math.cos(2.0 * math.pi * freq * x) + 1j * math.sin(2.0 * math.pi * freq * x) for x in t]
    return y

def sig_source_f(samp_rate, freq, amp, N):
    if False:
        i = 10
        return i + 15
    t = [float(x) / samp_rate for x in range(N)]
    y = [math.sin(2.0 * math.pi * freq * x) for x in t]
    return y

class test_pfb_arb_resampler(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def test_fff_000(self):
        if False:
            print('Hello World!')
        N = 500
        fs = 5000.0
        rrate = 2.3421
        nfilts = 32
        taps = filter.firdes.low_pass_2(nfilts, nfilts * fs, fs / 2, fs / 10, attenuation_dB=80, window=fft.window.WIN_BLACKMAN_hARRIS)
        freq = 121.213
        data = sig_source_f(fs, freq, 1, N)
        signal = blocks.vector_source_f(data)
        pfb = filter.pfb_arb_resampler_fff(rrate, taps, nfilts)
        snk = blocks.vector_sink_f()
        self.tb.connect(signal, pfb, snk)
        self.tb.run()
        Ntest = 50
        L = len(snk.data())
        delay = pfb.group_delay()
        phase = pfb.phase_offset(freq, fs)
        t = [float(x) / (fs * rrate) for x in range(-delay, L - delay)]
        expected_data = [math.sin(2.0 * math.pi * freq * x + phase) for x in t]
        dst_data = snk.data()
        self.assertFloatTuplesAlmostEqual(expected_data[-Ntest:], dst_data[-Ntest:], 2)

    def test_ccf_000(self):
        if False:
            while True:
                i = 10
        N = 5000
        fs = 5000.0
        rrate = 2.4321
        nfilts = 32
        taps = filter.firdes.low_pass_2(nfilts, nfilts * fs, fs / 2, fs / 10, attenuation_dB=80, window=fft.window.WIN_BLACKMAN_hARRIS)
        freq = 211.123
        data = sig_source_c(fs, freq, 1, N)
        signal = blocks.vector_source_c(data)
        pfb = filter.pfb_arb_resampler_ccf(rrate, taps, nfilts)
        snk = blocks.vector_sink_c()
        self.tb.connect(signal, pfb, snk)
        self.tb.run()
        Ntest = 50
        L = len(snk.data())
        delay = pfb.group_delay()
        phase = pfb.phase_offset(freq, fs)
        t = [float(x) / (fs * rrate) for x in range(-delay, L - delay)]
        expected_data = [math.cos(2.0 * math.pi * freq * x + phase) + 1j * math.sin(2.0 * math.pi * freq * x + phase) for x in t]
        dst_data = snk.data()
        self.assertComplexTuplesAlmostEqual(expected_data[-Ntest:], dst_data[-Ntest:], 2)

    def test_ccf_001(self):
        if False:
            return 10
        N = 50000
        fs = 5000.0
        rrate = 0.75
        nfilts = 32
        taps = filter.firdes.low_pass_2(nfilts, nfilts * fs, fs / 4, fs / 10, attenuation_dB=80, window=fft.window.WIN_BLACKMAN_hARRIS)
        freq = 211.123
        data = sig_source_c(fs, freq, 1, N)
        signal = blocks.vector_source_c(data)
        pfb = filter.pfb_arb_resampler_ccf(rrate, taps, nfilts)
        snk = blocks.vector_sink_c()
        self.tb.connect(signal, pfb, snk)
        self.tb.run()
        Ntest = 50
        L = len(snk.data())
        delay = pfb.group_delay()
        phase = pfb.phase_offset(freq, fs)
        t = [float(x) / (fs * rrate) for x in range(-delay, L - delay)]
        expected_data = [math.cos(2.0 * math.pi * freq * x + phase) + 1j * math.sin(2.0 * math.pi * freq * x + phase) for x in t]
        dst_data = snk.data()
        self.assertComplexTuplesAlmostEqual(expected_data[-Ntest:], dst_data[-Ntest:], 2)

    def test_ccc_000(self):
        if False:
            print('Hello World!')
        N = 5000
        fs = 5000.0
        rrate = 3.4321
        nfilts = 32
        taps = filter.firdes.complex_band_pass_2(nfilts, nfilts * fs, 50, 400, fs / 10, attenuation_dB=80, window=fft.window.WIN_BLACKMAN_hARRIS)
        freq = 211.123
        data = sig_source_c(fs, freq, 1, N)
        signal = blocks.vector_source_c(data)
        pfb = filter.pfb_arb_resampler_ccc(rrate, taps, nfilts)
        snk = blocks.vector_sink_c()
        self.tb.connect(signal, pfb, snk)
        self.tb.run()
        Ntest = 50
        L = len(snk.data())
        delay = pfb.group_delay()
        phase = pfb.phase_offset(freq, fs)
        t = [float(x) / (fs * rrate) for x in range(-delay, L - delay)]
        expected_data = [math.cos(2.0 * math.pi * freq * x + phase) + 1j * math.sin(2.0 * math.pi * freq * x + phase) for x in t]
        dst_data = snk.data()
        self.assertComplexTuplesAlmostEqual(expected_data[-Ntest:], dst_data[-Ntest:], 2)

    def test_ccc_001(self):
        if False:
            for i in range(10):
                print('nop')
        N = 50000
        fs = 5000.0
        rrate = 0.715
        nfilts = 32
        taps = filter.firdes.complex_band_pass_2(nfilts, nfilts * fs, 50, 400, fs / 10, attenuation_dB=80, window=fft.window.WIN_BLACKMAN_hARRIS)
        freq = 211.123
        data = sig_source_c(fs, freq, 1, N)
        signal = blocks.vector_source_c(data)
        pfb = filter.pfb_arb_resampler_ccc(rrate, taps, nfilts)
        snk = blocks.vector_sink_c()
        self.tb.connect(signal, pfb, snk)
        self.tb.run()
        Ntest = 50
        L = len(snk.data())
        delay = pfb.group_delay()
        phase = pfb.phase_offset(freq, fs)
        t = [float(x) / (fs * rrate) for x in range(-delay, L - delay)]
        expected_data = [math.cos(2.0 * math.pi * freq * x + phase) + 1j * math.sin(2.0 * math.pi * freq * x + phase) for x in t]
        dst_data = snk.data()
        self.assertComplexTuplesAlmostEqual(expected_data[-Ntest:], dst_data[-Ntest:], 2)
if __name__ == '__main__':
    gr_unittest.run(test_pfb_arb_resampler)