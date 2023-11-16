from gnuradio import gr, gr_unittest, filter, blocks

class test_dc_blocker(gr_unittest.TestCase):

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

    def test_001(self):
        if False:
            while True:
                i = 10
        ' Test impulse response - long form, cc '
        src_data = [1] + 100 * [0]
        expected_result = (-0.02072429656982422 + 0j, -0.02081298828125 + 0j, 0.979156494140625 + 0j, -0.02081298828125 + 0j, -0.02072429656982422 + 0j)
        src = blocks.vector_source_c(src_data)
        op = filter.dc_blocker_cc(32, True)
        dst = blocks.vector_sink_c()
        self.tb.connect(src, op, dst)
        self.tb.run()
        result_data = dst.data()[60:65]
        self.assertComplexTuplesAlmostEqual(expected_result, result_data)

    def test_002(self):
        if False:
            while True:
                i = 10
        ' Test impulse response - short form, cc '
        src_data = [1] + 100 * [0]
        expected_result = (-0.029296875 + 0j, -0.0302734375 + 0j, 0.96875 + 0j, -0.0302734375 + 0j, -0.029296875 + 0j)
        src = blocks.vector_source_c(src_data)
        op = filter.dc_blocker_cc(32, False)
        dst = blocks.vector_sink_c()
        self.tb.connect(src, op, dst)
        self.tb.run()
        result_data = dst.data()[29:34]
        self.assertComplexTuplesAlmostEqual(expected_result, result_data)

    def test_003(self):
        if False:
            return 10
        ' Test impulse response - long form, ff '
        src_data = [1] + 100 * [0]
        expected_result = (-0.02072429656982422, -0.02081298828125, 0.979156494140625, -0.02081298828125, -0.02072429656982422)
        src = blocks.vector_source_f(src_data)
        op = filter.dc_blocker_ff(32, True)
        dst = blocks.vector_sink_f()
        self.tb.connect(src, op, dst)
        self.tb.run()
        result_data = dst.data()[60:65]
        self.assertFloatTuplesAlmostEqual(expected_result, result_data)

    def test_004(self):
        if False:
            i = 10
            return i + 15
        ' Test impulse response - short form, ff '
        src_data = [1] + 100 * [0]
        expected_result = (-0.029296875, -0.0302734375, 0.96875, -0.0302734375, -0.029296875)
        src = blocks.vector_source_f(src_data)
        op = filter.dc_blocker_ff(32, False)
        dst = blocks.vector_sink_f()
        self.tb.connect(src, op, dst)
        self.tb.run()
        result_data = dst.data()[29:34]
        self.assertFloatTuplesAlmostEqual(expected_result, result_data)
if __name__ == '__main__':
    gr_unittest.run(test_dc_blocker)