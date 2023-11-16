from gnuradio import gr, gr_unittest, blocks

class test_multiply_conjugate(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def test_000(self):
        if False:
            print('Hello World!')
        src_data0 = [-2 - 2j, -1 - 1j, -2 + 2j, -1 + 1j, 2 - 2j, 1 - 1j, 2 + 2j, 1 + 1j, 0 + 0j]
        src_data1 = [-3 - 3j, -4 - 4j, -3 + 3j, -4 + 4j, 3 - 3j, 4 - 4j, 3 + 3j, 4 + 4j, 0 + 0j]
        exp_data = [12 + 0j, 8 + 0j, 12 + 0j, 8 + 0j, 12 + 0j, 8 + 0j, 12 + 0j, 8 + 0j, 0 + 0j]
        src0 = blocks.vector_source_c(src_data0)
        src1 = blocks.vector_source_c(src_data1)
        op = blocks.multiply_conjugate_cc()
        dst = blocks.vector_sink_c()
        self.tb.connect(src0, (op, 0))
        self.tb.connect(src1, (op, 1))
        self.tb.connect(op, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(exp_data, result_data)
if __name__ == '__main__':
    gr_unittest.run(test_multiply_conjugate)