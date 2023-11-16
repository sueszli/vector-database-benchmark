import random
from gnuradio import gr, gr_unittest, digital, blocks

class test_binary_slicer_fb(gr_unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(0)
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            return 10
        self.tb = None

    def test_binary_slicer_fb(self):
        if False:
            for i in range(10):
                print('nop')
        expected_result = (0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1)
        src_data = (-1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1)
        src_data = [s + (1 - random.random()) for s in src_data]
        src = blocks.vector_source_f(src_data)
        op = digital.binary_slicer_fb()
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()
        actual_result = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, actual_result)
if __name__ == '__main__':
    gr_unittest.run(test_binary_slicer_fb)