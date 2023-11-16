from gnuradio import gr, gr_unittest, digital, blocks

class test_glfsr_source(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def test_000_make_b(self):
        if False:
            return 10
        src = digital.glfsr_source_b(16)
        self.assertEqual(src.mask(), 32790)
        self.assertEqual(src.period(), 2 ** 16 - 1)

    def test_001_degree_b(self):
        if False:
            print('Hello World!')
        self.assertRaises(RuntimeError, lambda : digital.glfsr_source_b(0))
        self.assertRaises(RuntimeError, lambda : digital.glfsr_source_b(65))

    def test_002_correlation_b(self):
        if False:
            return 10
        for degree in range(1, 11):
            src = digital.glfsr_source_b(degree, False)
            b2f = digital.chunks_to_symbols_bf((-1.0, 1.0), 1)
            dst = blocks.vector_sink_f()
            del self.tb
            self.tb = gr.top_block()
            self.tb.connect(src, b2f, dst)
            self.tb.run()
            self.tb.disconnect_all()
            actual_result = dst.data()
            R = auto_correlate(actual_result)
            self.assertEqual(R[0], float(len(R)))
            for i in range(len(R) - 1):
                self.assertEqual(R[i + 1], -1.0)

    def test_003_make_f(self):
        if False:
            for i in range(10):
                print('nop')
        src = digital.glfsr_source_f(16)
        self.assertEqual(src.mask(), 32790)
        self.assertEqual(src.period(), 2 ** 16 - 1)

    def test_004_degree_f(self):
        if False:
            while True:
                i = 10
        self.assertRaises(RuntimeError, lambda : digital.glfsr_source_f(0))
        self.assertRaises(RuntimeError, lambda : digital.glfsr_source_f(65))

    def test_005_correlation_f(self):
        if False:
            for i in range(10):
                print('nop')
        for degree in range(1, 11):
            src = digital.glfsr_source_f(degree, False)
            dst = blocks.vector_sink_f()
            del self.tb
            self.tb = gr.top_block()
            self.tb.connect(src, dst)
            self.tb.run()
            actual_result = dst.data()
            R = auto_correlate(actual_result)
            self.assertEqual(R[0], float(len(R)))
            for i in range(len(R) - 1):
                self.assertEqual(R[i + 1], -1.0)

def auto_correlate(data):
    if False:
        print('Hello World!')
    l = len(data)
    R = [0] * l
    for lag in range(l):
        for i in range(l):
            R[lag] += data[i] * data[i - lag]
    return R
if __name__ == '__main__':
    gr_unittest.run(test_glfsr_source)