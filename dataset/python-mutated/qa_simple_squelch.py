from gnuradio import gr, gr_unittest, analog, blocks

class test_simple_squelch(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def test_simple_squelch_001(self):
        if False:
            return 10
        alpha = 0.0001
        thr1 = 10
        thr2 = 20
        op = analog.simple_squelch_cc(thr1, alpha)
        op.set_threshold(thr2)
        t = op.threshold()
        self.assertEqual(thr2, t)

    def test_simple_squelch_002(self):
        if False:
            print('Hello World!')
        alpha = 0.0001
        thr = -25
        src_data = [float(x) / 10.0 for x in range(1, 40)]
        src = blocks.vector_source_c(src_data)
        op = analog.simple_squelch_cc(thr, alpha)
        dst = blocks.vector_sink_c()
        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()
        expected_result = src_data
        expected_result[0:20] = 20 * [0]
        result_data = dst.data()
        self.assertComplexTuplesAlmostEqual(expected_result, result_data, 4)
if __name__ == '__main__':
    gr_unittest.run(test_simple_squelch)