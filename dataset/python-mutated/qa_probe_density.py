from gnuradio import gr, gr_unittest, digital, blocks

class test_probe_density(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def test_001(self):
        if False:
            i = 10
            return i + 15
        src_data = [0, 1, 0, 1]
        expected_data = 1
        src = blocks.vector_source_b(src_data)
        op = digital.probe_density_b(1)
        self.tb.connect(src, op)
        self.tb.run()
        result_data = op.density()
        self.assertEqual(expected_data, result_data)

    def test_002(self):
        if False:
            i = 10
            return i + 15
        src_data = [1, 1, 1, 1]
        expected_data = 1
        src = blocks.vector_source_b(src_data)
        op = digital.probe_density_b(0.01)
        self.tb.connect(src, op)
        self.tb.run()
        result_data = op.density()
        self.assertEqual(expected_data, result_data)

    def test_003(self):
        if False:
            while True:
                i = 10
        src_data = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        expected_data = 0.95243
        src = blocks.vector_source_b(src_data)
        op = digital.probe_density_b(0.01)
        self.tb.connect(src, op)
        self.tb.run()
        result_data = op.density()
        print(result_data)
        self.assertAlmostEqual(expected_data, result_data, 5)
if __name__ == '__main__':
    gr_unittest.run(test_probe_density)