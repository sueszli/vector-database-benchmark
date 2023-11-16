from gnuradio import gr, gr_unittest, blocks

class test_peak_detector2(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            return 10
        self.tb = None

    def test_peak1(self):
        if False:
            print('Hello World!')
        tb = self.tb
        n = 10
        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] + n * [0]
        expected_result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + n * [0]
        src = blocks.vector_source_f(data, False)
        regen = blocks.peak_detector2_fb(7.0, 25, 0.001)
        dst = blocks.vector_sink_b()
        tb.connect(src, regen)
        tb.connect(regen, dst)
        tb.run()
        dst_data = dst.data()
        self.assertEqual(expected_result, dst_data)

    def test_peak2(self):
        if False:
            return 10
        tb = self.tb
        n = 10
        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] + n * [0]
        expected_result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + n * [0]
        src = blocks.vector_source_f(data, False)
        regen = blocks.peak_detector2_fb(7.0, 1000, 0.001)
        dst = blocks.vector_sink_b()
        tb.connect(src, regen)
        tb.connect(regen, dst)
        tb.run()
        dst_data = dst.data()
        self.assertEqual(expected_result[0:len(dst_data)], dst_data)

    def test_peak3(self):
        if False:
            while True:
                i = 10
        tb = self.tb
        l = 8100
        m = 100
        n = 10
        data = l * [0] + [10] + m * [0] + [100] + n * [0]
        expected_result = l * [0] + [0] + m * [0] + [1] + n * [0]
        src = blocks.vector_source_f(data, False)
        regen = blocks.peak_detector2_fb(7.0, 105, 0.001)
        dst = blocks.vector_sink_b()
        tb.connect(src, regen)
        tb.connect(regen, dst)
        tb.run()
        dst_data = dst.data()
        self.assertEqual(expected_result, dst_data)

    def test_peak4(self):
        if False:
            print('Hello World!')
        tb = self.tb
        l = 8100
        m = 100
        n = 10
        data = l * [0] + [10] + m * [0] + [100] + n * [0]
        expected_result = l * [0] + [0] + m * [0] + [1] + n * [0]
        src = blocks.vector_source_f(data, False)
        regen = blocks.peak_detector2_fb(7.0, 150, 0.001)
        dst = blocks.vector_sink_b()
        tb.connect(src, regen)
        tb.connect(regen, dst)
        tb.run()
        dst_data = dst.data()
        self.assertEqual(expected_result[0:len(dst_data)], dst_data)

    def test_peak5(self):
        if False:
            i = 10
            return i + 15
        tb = self.tb
        data = [0, 0, 0, 10, 0, 0, 0, 0]
        alpha = 0.25
        expected_result_peak = [0, 0, 0, 1, 0, 0, 0, 0]
        expected_result_average = [0]
        for i in data:
            expected_result_average.append(expected_result_average[-1] * (1 - alpha) + i * alpha)
        src = blocks.vector_source_f(data, False)
        regen = blocks.peak_detector2_fb(2.0, 2, alpha)
        dst = blocks.vector_sink_b()
        avg = blocks.vector_sink_f()
        tb.connect(src, regen)
        tb.connect((regen, 0), dst)
        tb.connect((regen, 1), avg)
        tb.run()
        dst_data = dst.data()
        dst_avg = avg.data()
        self.assertEqual(expected_result_peak, dst_data)
        self.assertFloatTuplesAlmostEqual(expected_result_average[1:], dst_avg)
if __name__ == '__main__':
    gr_unittest.run(test_peak_detector2)