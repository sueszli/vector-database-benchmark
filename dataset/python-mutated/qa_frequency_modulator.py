import math
from gnuradio import gr, gr_unittest, analog, blocks

def sincos(x):
    if False:
        for i in range(10):
            print('nop')
    return math.cos(x) + math.sin(x) * 1j

class test_frequency_modulator(gr_unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def test_fm_001(self):
        if False:
            print('Hello World!')
        pi = math.pi
        sensitivity = pi / 4
        src_data = (1.0 / 4, 1.0 / 2, 1.0 / 4, -1.0 / 4, -1.0 / 2, -1 / 4.0)
        running_sum = (pi / 16, 3 * pi / 16, pi / 4, 3 * pi / 16, pi / 16, 0)
        expected_result = tuple([sincos(x) for x in running_sum])
        src = blocks.vector_source_f(src_data)
        op = analog.frequency_modulator_fc(sensitivity)
        dst = blocks.vector_sink_c()
        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertComplexTuplesAlmostEqual(expected_result, result_data, 5)
if __name__ == '__main__':
    gr_unittest.run(test_frequency_modulator)