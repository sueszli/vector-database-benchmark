import math
from gnuradio import gr, gr_unittest, analog, blocks

def avg_mag_sqrd_c(x, alpha):
    if False:
        print('Hello World!')
    y = [0]
    for xi in x:
        tmp = alpha * (xi.real * xi.real + xi.imag * xi.imag) + (1 - alpha) * y[-1]
        y.append(tmp)
    return y

def avg_mag_sqrd_f(x, alpha):
    if False:
        return 10
    y = [0]
    for xi in x:
        tmp = alpha * (xi * xi) + (1 - alpha) * y[-1]
        y.append(tmp)
    return y

class test_probe_avg_mag_sqrd(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def test_c_001(self):
        if False:
            print('Hello World!')
        alpha = 0.0001
        src_data = [1.0 + 1j, 2.0 + 2j, 3.0 + 3j, 4.0 + 4j, 5.0 + 5j, 6.0 + 6j, 7.0 + 7j, 8.0 + 8j, 9.0 + 9j, 10.0 + 10j]
        expected_result = avg_mag_sqrd_c(src_data, alpha)[-1]
        src = blocks.vector_source_c(src_data)
        op = analog.probe_avg_mag_sqrd_c(0, alpha)
        self.tb.connect(src, op)
        self.tb.run()
        result_data = op.level()
        self.assertAlmostEqual(expected_result, result_data, 5)

    def test_cf_002(self):
        if False:
            print('Hello World!')
        alpha = 0.0001
        src_data = [1.0 + 1j, 2.0 + 2j, 3.0 + 3j, 4.0 + 4j, 5.0 + 5j, 6.0 + 6j, 7.0 + 7j, 8.0 + 8j, 9.0 + 9j, 10.0 + 10j]
        expected_result = avg_mag_sqrd_c(src_data, alpha)[0:-1]
        src = blocks.vector_source_c(src_data)
        op = analog.probe_avg_mag_sqrd_cf(0, alpha)
        dst = blocks.vector_sink_f()
        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertComplexTuplesAlmostEqual(expected_result, result_data, 5)

    def test_f_003(self):
        if False:
            while True:
                i = 10
        alpha = 0.0001
        src_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        expected_result = avg_mag_sqrd_f(src_data, alpha)[-1]
        src = blocks.vector_source_f(src_data)
        op = analog.probe_avg_mag_sqrd_f(0, alpha)
        self.tb.connect(src, op)
        self.tb.run()
        result_data = op.level()
        self.assertAlmostEqual(expected_result, result_data, 5)
if __name__ == '__main__':
    gr_unittest.run(test_probe_avg_mag_sqrd)