from gnuradio import gr, gr_unittest, analog, blocks, channels
import math

class test_channel_model(gr_unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            return 10
        self.tb = None

    def test_000(self):
        if False:
            print('Hello World!')
        N = 1000
        fs = 1000
        freq = 100
        signal = analog.sig_source_c(fs, analog.GR_SIN_WAVE, freq, 1)
        head = blocks.head(gr.sizeof_gr_complex, N)
        op = channels.channel_model(0.0, 0.0, 1.0, [1], 0)
        snk = blocks.vector_sink_c()
        snk1 = blocks.vector_sink_c()
        op.set_noise_voltage(0.0)
        op.set_frequency_offset(0.0)
        op.set_taps([1])
        op.set_timing_offset(1.0)
        self.tb.connect(signal, head, op, snk)
        self.tb.connect(op, snk1)
        self.tb.run()
        dst_data = snk.data()
        exp_data = snk1.data()
        self.assertComplexTuplesAlmostEqual(exp_data, dst_data, 5)
if __name__ == '__main__':
    gr_unittest.run(test_channel_model)