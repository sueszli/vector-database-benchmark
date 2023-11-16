from gnuradio import gr, gr_unittest, fft, filter, blocks
import math

def sin_source_f(samp_rate, freq, amp, N):
    if False:
        i = 10
        return i + 15
    t = [float(x) / samp_rate for x in range(N)]
    y = [math.sin(2.0 * math.pi * freq * x) for x in t]
    return y

def cos_source_f(samp_rate, freq, amp, N):
    if False:
        print('Hello World!')
    t = [float(x) / samp_rate for x in range(N)]
    y = [math.cos(2.0 * math.pi * freq * x) for x in t]
    return y

def fir_filter(x, taps, delay):
    if False:
        print('Hello World!')
    y = []
    x2 = (len(taps) - 1) * [0] + x
    for i in range(len(x)):
        yi = 0
        for j in range(len(taps)):
            yi += taps[len(taps) - 1 - j] * x2[i + j]
        y.append(complex(x2[i + delay], yi))
    return y

def fir_filter2(x1, x2, taps, delay):
    if False:
        while True:
            i = 10
    y = []
    x1_2 = (len(taps) - 1) * [0] + x1
    x2_2 = (len(taps) - 1) * [0] + x2
    for i in range(len(x2)):
        yi = 0
        for j in range(len(taps)):
            yi += taps[len(taps) - 1 - j] * x2_2[i + j]
        y.append(complex(x1_2[i + delay], yi))
    return y

class test_filter_delay_fc(gr_unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            return 10
        self.tb = None

    def test_001_filter_delay_one_input(self):
        if False:
            for i in range(10):
                print('nop')
        tb = self.tb
        sampling_freq = 100
        ntaps = 51
        N = int(ntaps + sampling_freq * 0.1)
        data = sin_source_f(sampling_freq, sampling_freq * 0.1, 1.0, N)
        src1 = blocks.vector_source_f(data)
        dst2 = blocks.vector_sink_c()
        taps = filter.firdes.hilbert(ntaps, fft.window.WIN_HAMMING)
        hd = filter.filter_delay_fc(taps)
        expected_result = fir_filter(data, taps, (ntaps - 1) // 2)
        tb.connect(src1, hd)
        tb.connect(hd, dst2)
        tb.run()
        result_data = dst2.data()
        self.assertComplexTuplesAlmostEqual(expected_result, result_data, 5)

    def test_002_filter_delay_two_inputs(self):
        if False:
            while True:
                i = 10
        tb = self.tb
        sampling_freq = 100
        ntaps = 51
        N = int(ntaps + sampling_freq * 0.1)
        data = sin_source_f(sampling_freq, sampling_freq * 0.1, 1.0, N)
        src1 = blocks.vector_source_f(data)
        dst2 = blocks.vector_sink_c()
        taps = filter.firdes.hilbert(ntaps, fft.window.WIN_HAMMING)
        hd = filter.filter_delay_fc(taps)
        expected_result = fir_filter2(data, data, taps, (ntaps - 1) // 2)
        tb.connect(src1, (hd, 0))
        tb.connect(src1, (hd, 1))
        tb.connect(hd, dst2)
        tb.run()
        result_data = dst2.data()
        self.assertComplexTuplesAlmostEqual(expected_result, result_data, 5)

    def test_003_filter_delay_two_inputs(self):
        if False:
            while True:
                i = 10
        tb = self.tb
        sampling_freq = 100
        ntaps = 51
        N = int(ntaps + sampling_freq * 0.1)
        data1 = sin_source_f(sampling_freq, sampling_freq * 0.1, 1.0, N)
        data2 = cos_source_f(sampling_freq, sampling_freq * 0.1, 1.0, N)
        src1 = blocks.vector_source_f(data1)
        src2 = blocks.vector_source_f(data2)
        taps = filter.firdes.hilbert(ntaps, fft.window.WIN_HAMMING)
        hd = filter.filter_delay_fc(taps)
        expected_result = fir_filter2(data1, data2, taps, (ntaps - 1) // 2)
        dst2 = blocks.vector_sink_c()
        tb.connect(src1, (hd, 0))
        tb.connect(src2, (hd, 1))
        tb.connect(hd, dst2)
        tb.run()
        result_data = dst2.data()
        self.assertComplexTuplesAlmostEqual(expected_result, result_data, 5)
if __name__ == '__main__':
    gr_unittest.run(test_filter_delay_fc)