from gnuradio import gr, gr_unittest, blocks

class test_regenerate(gr_unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def test_regen1(self):
        if False:
            return 10
        tb = self.tb
        data = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        expected_result = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        src = blocks.vector_source_b(data, False)
        regen = blocks.regenerate_bb(5, 2)
        dst = blocks.vector_sink_b()
        tb.connect(src, regen)
        tb.connect(regen, dst)
        tb.run()
        dst_data = dst.data()
        self.assertEqual(expected_result, dst_data)

    def test_regen2(self):
        if False:
            for i in range(10):
                print('nop')
        tb = self.tb
        data = 200 * [0]
        data[9] = 1
        data[99] = 1
        expected_result = 200 * [0]
        expected_result[9] = 1
        expected_result[19] = 1
        expected_result[29] = 1
        expected_result[39] = 1
        expected_result[99] = 1
        expected_result[109] = 1
        expected_result[119] = 1
        expected_result[129] = 1
        src = blocks.vector_source_b(data, False)
        regen = blocks.regenerate_bb(10, 3)
        dst = blocks.vector_sink_b()
        tb.connect(src, regen)
        tb.connect(regen, dst)
        tb.run()
        dst_data = dst.data()
        self.assertEqual(expected_result, dst_data)
if __name__ == '__main__':
    gr_unittest.run(test_regenerate)