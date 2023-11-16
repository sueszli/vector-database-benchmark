from gnuradio import gr, gr_unittest, blocks
import math
import random

def make_random_complex_tuple(L, scale=1):
    if False:
        for i in range(10):
            print('nop')
    result = []
    for x in range(L):
        result.append(scale * complex(2 * random.random() - 1, 2 * random.random() - 1))
    return tuple(result)

def make_random_float_tuple(L, scale=1):
    if False:
        for i in range(10):
            print('nop')
    result = []
    for x in range(L):
        result.append(scale * (2 * random.random() - 1))
    return tuple(result)

class test_moving_average(gr_unittest.TestCase):

    def assertListAlmostEqual(self, list1, list2, tol):
        if False:
            return 10
        self.assertEqual(len(list1), len(list2))
        for (a, b) in zip(list1, list2):
            self.assertAlmostEqual(a, b, tol)

    def setUp(self):
        if False:
            while True:
                i = 10
        random.seed(0)
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tb = None

    def test_01(self):
        if False:
            return 10
        tb = self.tb
        N = 10000
        data = make_random_float_tuple(N, 1)
        expected_result = N * [0]
        src = blocks.vector_source_f(data, False)
        op = blocks.moving_average_ff(100, 0.001)
        dst = blocks.vector_sink_f()
        tb.connect(src, op)
        tb.connect(op, dst)
        tb.run()
        dst_data = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, dst_data, 1)

    def test_02(self):
        if False:
            i = 10
            return i + 15
        tb = self.tb
        N = 10000
        data = make_random_complex_tuple(N, 1)
        expected_result = N * [0]
        src = blocks.vector_source_c(data, False)
        op = blocks.moving_average_cc(100, 0.001)
        dst = blocks.vector_sink_c()
        tb.connect(src, op)
        tb.connect(op, dst)
        tb.run()
        dst_data = dst.data()
        self.assertComplexTuplesAlmostEqual(expected_result, dst_data, 1)

    def test_vector_int(self):
        if False:
            print('Hello World!')
        tb = self.tb
        vlen = 5
        N = 10 * vlen
        data = make_random_float_tuple(N, 2 ** 10)
        data = [int(d * 1000) for d in data]
        src = blocks.vector_source_i(data, False)
        one_to_many = blocks.stream_to_streams(gr.sizeof_int, vlen)
        one_to_vector = blocks.stream_to_vector(gr.sizeof_int, vlen)
        many_to_vector = blocks.streams_to_vector(gr.sizeof_int, vlen)
        isolated = [blocks.moving_average_ii(100, 1) for i in range(vlen)]
        dut = blocks.moving_average_ii(100, 1, vlen=vlen)
        dut_dst = blocks.vector_sink_i(vlen=vlen)
        ref_dst = blocks.vector_sink_i(vlen=vlen)
        tb.connect(src, one_to_many)
        tb.connect(src, one_to_vector, dut, dut_dst)
        tb.connect(many_to_vector, ref_dst)
        for (idx, single) in enumerate(isolated):
            tb.connect((one_to_many, idx), single, (many_to_vector, idx))
        tb.run()
        dut_data = dut_dst.data()
        ref_data = ref_dst.data()
        self.assertEqual(dut_data, ref_data)

    def test_vector_complex(self):
        if False:
            for i in range(10):
                print('nop')
        tb = self.tb
        vlen = 5
        N = 10 * vlen
        data = make_random_complex_tuple(N, 2 ** 10)
        src = blocks.vector_source_c(data, False)
        one_to_many = blocks.stream_to_streams(gr.sizeof_gr_complex, vlen)
        one_to_vector = blocks.stream_to_vector(gr.sizeof_gr_complex, vlen)
        many_to_vector = blocks.streams_to_vector(gr.sizeof_gr_complex, vlen)
        isolated = [blocks.moving_average_cc(100, 1) for i in range(vlen)]
        dut = blocks.moving_average_cc(100, 1, vlen=vlen)
        dut_dst = blocks.vector_sink_c(vlen=vlen)
        ref_dst = blocks.vector_sink_c(vlen=vlen)
        tb.connect(src, one_to_many)
        tb.connect(src, one_to_vector, dut, dut_dst)
        tb.connect(many_to_vector, ref_dst)
        for (idx, single) in enumerate(isolated):
            tb.connect((one_to_many, idx), single, (many_to_vector, idx))
        tb.run()
        dut_data = dut_dst.data()
        ref_data = ref_dst.data()
        self.assertListAlmostEqual(dut_data, ref_data, tol=3)

    def test_complex_scalar(self):
        if False:
            while True:
                i = 10
        tb = self.tb
        N = 10000
        history = 100
        data = make_random_complex_tuple(N, 1)
        data_padded = (history - 1) * [complex(0.0, 0.0)] + list(data)
        expected_result = []
        moving_sum = sum(data_padded[:history - 1])
        for i in range(N):
            moving_sum += data_padded[i + history - 1]
            expected_result.append(moving_sum)
            moving_sum -= data_padded[i]
        src = blocks.vector_source_c(data, False)
        op = blocks.moving_average_cc(history, 1)
        dst = blocks.vector_sink_c()
        tb.connect(src, op)
        tb.connect(op, dst)
        tb.run()
        dst_data = dst.data()
        self.assertListAlmostEqual(expected_result, dst_data, 4)
if __name__ == '__main__':
    gr_unittest.run(test_moving_average)