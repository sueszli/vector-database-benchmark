from gnuradio import gr, gr_unittest, blocks
import math

class test_transcendental(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            return 10
        self.tb = None

    def test_01(self):
        if False:
            return 10
        tb = self.tb
        data = 100 * [0]
        expected_result = 100 * [1]
        src = blocks.vector_source_f(data, False)
        op = blocks.transcendental('cos', 'float')
        dst = blocks.vector_sink_f()
        tb.connect(src, op)
        tb.connect(op, dst)
        tb.run()
        dst_data = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, dst_data, 5)

    def test_02(self):
        if False:
            return 10
        tb = self.tb
        data = 100 * [3]
        expected_result = 100 * [math.log10(3)]
        src = blocks.vector_source_f(data, False)
        op = blocks.transcendental('log10', 'float')
        dst = blocks.vector_sink_f()
        tb.connect(src, op)
        tb.connect(op, dst)
        tb.run()
        dst_data = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, dst_data, 5)

    def test_03(self):
        if False:
            for i in range(10):
                print('nop')
        tb = self.tb
        data = 100 * [3]
        expected_result = 100 * [math.tanh(3)]
        src = blocks.vector_source_f(data, False)
        op = blocks.transcendental('tanh', 'float')
        dst = blocks.vector_sink_f()
        tb.connect(src, op)
        tb.connect(op, dst)
        tb.run()
        dst_data = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, dst_data, 5)
if __name__ == '__main__':
    gr_unittest.run(test_transcendental)