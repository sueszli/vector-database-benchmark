from gnuradio import gr, gr_unittest, blocks

class qa_plateau_detector_fb(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            return 10
        self.tb = None

    def test_001_t(self):
        if False:
            for i in range(10):
                print('nop')
        test_signal = [0, 1, 0.2, 0.4, 0.6, 0.8, 1, 1, 1, 1, 1, 0.8, 0.6, 0.4, 1, 0]
        expected_sig = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        sink = blocks.vector_sink_b()
        self.tb.connect(blocks.vector_source_f(test_signal), blocks.plateau_detector_fb(5), sink)
        self.tb.run()
        self.assertEqual(expected_sig, sink.data())
if __name__ == '__main__':
    gr_unittest.run(qa_plateau_detector_fb)