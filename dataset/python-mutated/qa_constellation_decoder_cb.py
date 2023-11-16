from gnuradio import gr, gr_unittest, digital, blocks

class test_constellation_decoder(gr_unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tb = None

    def test_constellation_decoder_cb_bpsk(self):
        if False:
            print('Hello World!')
        cnst = digital.constellation_bpsk()
        src_data = (0.5 + 0.5j, 0.1 - 1.2j, -0.8 - 0.1j, -0.45 + 0.8j, 0.8 + 1j, -0.5 + 0.1j, 0.1 - 1.2j)
        expected_result = (1, 1, 0, 0, 1, 0, 1)
        src = blocks.vector_source_c(src_data)
        op = digital.constellation_decoder_cb(cnst.base())
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()
        actual_result = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, actual_result)

    def _test_constellation_decoder_cb_qpsk(self):
        if False:
            for i in range(10):
                print('nop')
        cnst = digital.constellation_qpsk()
        src_data = (0.5 + 0.5j, 0.1 - 1.2j, -0.8 - 0.1j, -0.45 + 0.8j, 0.8 + 1j, -0.5 + 0.1j, 0.1 - 1.2j)
        expected_result = (3, 1, 0, 2, 3, 2, 1)
        src = blocks.vector_source_c(src_data)
        op = digital.constellation_decoder_cb(cnst.base())
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()
        actual_result = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, actual_result)

    def test_constellation_decoder_cb_bpsk_to_qpsk(self):
        if False:
            return 10
        cnst = digital.constellation_bpsk()
        src_data = (0.5 + 0.5j, 0.1 - 1.2j, -0.8 - 0.1j, -0.45 + 0.8j, 0.8 + 1j, -0.5 + 0.1j, 0.1 - 1.2j)
        src = blocks.vector_source_c(src_data)
        op = digital.constellation_decoder_cb(cnst.base())
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()
        cnst = digital.constellation_qpsk()
        src_data = (0.5 + 0.5j, 0.1 - 1.2j, -0.8 - 0.1j, -0.45 + 0.8j, 0.8 + 1j, -0.5 + 0.1j, 0.1 - 1.2j)
        expected_result = (1, 1, 0, 0, 1, 0, 1, 3, 1, 0, 2, 3, 2, 1)
        src.set_data(src_data)
        op.set_constellation(cnst.base())
        self.tb.run()
        actual_result = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, actual_result)
if __name__ == '__main__':
    gr_unittest.run(test_constellation_decoder)