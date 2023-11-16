from gnuradio import gr, gr_unittest, blocks

class test_conjugate(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def test_000(self):
        if False:
            for i in range(10):
                print('nop')
        src_data = [-2 - 2j, -1 - 1j, -2 + 2j, -1 + 1j, 2 - 2j, 1 - 1j, 2 + 2j, 1 + 1j, 0 + 0j]
        exp_data = [-2 + 2j, -1 + 1j, -2 - 2j, -1 - 1j, 2 + 2j, 1 + 1j, 2 - 2j, 1 - 1j, 0 - 0j]
        src = blocks.vector_source_c(src_data)
        op = blocks.conjugate_cc()
        dst = blocks.vector_sink_c()
        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(exp_data, result_data)
if __name__ == '__main__':
    gr_unittest.run(test_conjugate)