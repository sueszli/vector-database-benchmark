from gnuradio import gr, gr_unittest, fft, filter, blocks
import math

def sig_source_c(samp_rate, freq, amp, N):
    if False:
        print('Hello World!')
    t = [float(x) / samp_rate for x in range(N)]
    y = [math.cos(2.0 * math.pi * freq * x) + 1j * math.sin(2.0 * math.pi * freq * x) for x in t]
    return y

class test_pfb_interpolator(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def test_000(self):
        if False:
            i = 10
            return i + 15
        N = 1000
        M = 5
        fs = 1000
        ofs = M * fs
        taps = filter.firdes.low_pass_2(M, ofs, fs / 4, fs / 10, attenuation_dB=80, window=fft.window.WIN_BLACKMAN_hARRIS)
        freq = 123.456
        data = sig_source_c(fs, freq, 1, N)
        signal = blocks.vector_source_c(data)
        pfb = filter.pfb_interpolator_ccf(M, taps)
        snk = blocks.vector_sink_c()
        self.tb.connect(signal, pfb)
        self.tb.connect(pfb, snk)
        self.tb.run()
        Ntest = 50
        L = len(snk.data())
        phase = 4.887011296997899
        t = [float(x) / ofs for x in range(0, L)]
        expected_data = [math.cos(2.0 * math.pi * freq * x + phase) + 1j * math.sin(2.0 * math.pi * freq * x + phase) for x in t]
        dst_data = snk.data()
        self.assertComplexTuplesAlmostEqual(expected_data[-Ntest:], dst_data[-Ntest:], 4)
if __name__ == '__main__':
    gr_unittest.run(test_pfb_interpolator)