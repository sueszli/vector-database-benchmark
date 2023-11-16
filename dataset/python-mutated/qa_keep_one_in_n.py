from gnuradio import gr, gr_unittest, blocks

class test_keep_one_in_n(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tb = None

    def test_001(self):
        if False:
            for i in range(10):
                print('nop')
        src_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected_data = [5, 10]
        src = blocks.vector_source_b(src_data)
        op = blocks.keep_one_in_n(gr.sizeof_char, 5)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op, dst)
        self.tb.run()
        self.assertEqual(dst.data(), expected_data)
if __name__ == '__main__':
    gr_unittest.run(test_keep_one_in_n)