from gnuradio import gr, gr_unittest, analog, blocks

class test_dpll_bb(gr_unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def test_dpll_bb_001(self):
        if False:
            return 10
        period = 1.0
        gain = 0.1
        op = analog.dpll_bb(period, gain)
        op.set_gain(0.2)
        g = op.gain()
        self.assertAlmostEqual(g, 0.2)
        f = op.freq()
        self.assertEqual(1 / period, f)
        d0 = 1.0 - 0.5 * f
        d1 = op.decision_threshold()
        self.assertAlmostEqual(d0, d1)
        p = op.phase()
        self.assertEqual(0, p)

    def test_dpll_bb_002(self):
        if False:
            return 10
        period = 4
        gain = 0.1
        src_data = 10 * ((period - 1) * [0] + [1])
        expected_result = src_data
        src = blocks.vector_source_b(src_data)
        op = analog.dpll_bb(period, gain)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertComplexTuplesAlmostEqual(expected_result, result_data, 4)
if __name__ == '__main__':
    gr_unittest.run(test_dpll_bb)