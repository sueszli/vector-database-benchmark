import math
from gnuradio import gr, gr_unittest, analog, blocks

class test_cpfsk_bc(gr_unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            return 10
        self.tb = None

    def test_cpfsk_bc_001(self):
        if False:
            for i in range(10):
                print('nop')
        op = analog.cpfsk_bc(2, 1, 2)
        op.set_amplitude(2)
        a = op.amplitude()
        self.assertEqual(2, a)
        freq = 2 * math.pi / 2.0
        f = op.freq()
        self.assertAlmostEqual(freq, f, 5)
        p = op.phase()
        self.assertEqual(0, p)

    def test_cpfsk_bc_002(self):
        if False:
            print('Hello World!')
        src_data = 10 * [0, 1]
        expected_result = [complex(2 * x - 1, 0) for x in src_data]
        src = blocks.vector_source_b(src_data)
        op = analog.cpfsk_bc(2, 1, 2)
        dst = blocks.vector_sink_c()
        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()
        result_data = dst.data()[0:len(expected_result)]
        self.assertComplexTuplesAlmostEqual(expected_result, result_data, 4)
if __name__ == '__main__':
    gr_unittest.run(test_cpfsk_bc)