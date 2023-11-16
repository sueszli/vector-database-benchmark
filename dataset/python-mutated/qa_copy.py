from gnuradio import gr, gr_unittest, blocks

class test_copy(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            return 10
        self.tb = None

    def test_copy(self):
        if False:
            i = 10
            return i + 15
        src_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected_result = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        src = blocks.vector_source_b(src_data)
        op = blocks.copy(gr.sizeof_char)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op, dst)
        self.tb.run()
        dst_data = dst.data()
        self.assertEqual(expected_result, dst_data)

    def test_copy_drop(self):
        if False:
            while True:
                i = 10
        src_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected_result = []
        src = blocks.vector_source_b(src_data)
        op = blocks.copy(gr.sizeof_char)
        op.set_enabled(False)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op, dst)
        self.tb.run()
        dst_data = dst.data()
        self.assertEqual(expected_result, dst_data)
if __name__ == '__main__':
    gr_unittest.run(test_copy)