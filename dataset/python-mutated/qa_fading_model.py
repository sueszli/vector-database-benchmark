from gnuradio import gr, gr_unittest, analog, blocks, channels
import math

class test_fading_model(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tb = None

    def test_000(self):
        if False:
            while True:
                i = 10
        N = 1000
        fs = 1000
        freq = 100
        fDTs = 0.01
        K = 4
        signal = analog.sig_source_c(fs, analog.GR_SIN_WAVE, freq, 1)
        head = blocks.head(gr.sizeof_gr_complex, N)
        op = channels.fading_model(8, fDTs=fDTs, LOS=True, K=K, seed=0)
        snk = blocks.vector_sink_c()
        snk1 = blocks.vector_sink_c()
        self.assertAlmostEqual(K, op.K(), 4)
        self.assertAlmostEqual(fDTs, op.fDTs(), 4)
if __name__ == '__main__':
    gr_unittest.run(test_fading_model)