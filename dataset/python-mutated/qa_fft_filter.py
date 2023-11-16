from gnuradio import gr, gr_unittest, filter, blocks
import sys
import random

def make_random_complex_tuple(L):
    if False:
        return 10
    return [complex(2 * random.random() - 1, 2 * random.random() - 1) for _ in range(L)]

def make_random_float_tuple(L):
    if False:
        for i in range(10):
            print('nop')
    return [2 * random.random() - 1 for _ in range(L)]

def reference_filter_ccc(dec, taps, input):
    if False:
        for i in range(10):
            print('nop')
    '\n    compute result using conventional fir filter\n    '
    tb = gr.top_block()
    src = blocks.vector_source_c(input)
    op = filter.fir_filter_ccc(dec, taps)
    dst = blocks.vector_sink_c()
    tb.connect(src, op, dst)
    tb.run()
    return dst.data()

def reference_filter_fff(dec, taps, input):
    if False:
        i = 10
        return i + 15
    '\n    compute result using conventional fir filter\n    '
    tb = gr.top_block()
    src = blocks.vector_source_f(input)
    op = filter.fir_filter_fff(dec, taps)
    dst = blocks.vector_sink_f()
    tb.connect(src, op, dst)
    tb.run()
    return dst.data()

def reference_filter_ccf(dec, taps, input):
    if False:
        while True:
            i = 10
    '\n    compute result using conventional fir filter\n    '
    tb = gr.top_block()
    src = blocks.vector_source_c(input)
    op = filter.fir_filter_ccf(dec, taps)
    dst = blocks.vector_sink_c()
    tb.connect(src, op, dst)
    tb.run()
    return dst.data()

def print_complex(x):
    if False:
        print('Hello World!')
    for i in x:
        i = complex(i)
        sys.stdout.write('(%6.3f,%6.3fj), ' % (i.real, i.imag))
    sys.stdout.write('\n')

class test_fft_filter(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        random.seed(0)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def assert_fft_ok2(self, expected_result, result_data):
        if False:
            print('Hello World!')
        expected_result = expected_result[:len(result_data)]
        self.assertComplexTuplesAlmostEqual2(expected_result, result_data, abs_eps=1e-09, rel_eps=0.0004)

    def assert_fft_float_ok2(self, expected_result, result_data, abs_eps=1e-09, rel_eps=0.0004):
        if False:
            while True:
                i = 10
        expected_result = expected_result[:len(result_data)]
        self.assertFloatTuplesAlmostEqual2(expected_result, result_data, abs_eps, rel_eps)

    def test_ccc_001(self):
        if False:
            return 10
        tb = gr.top_block()
        src_data = (0, 1, 2, 3, 4, 5, 6, 7)
        taps = (1,)
        expected_result = tuple([complex(x) for x in (0, 1, 2, 3, 4, 5, 6, 7)])
        src = blocks.vector_source_c(src_data)
        op = filter.fft_filter_ccc(1, taps)
        dst = blocks.vector_sink_c()
        tb.connect(src, op, dst)
        tb.run()
        result_data = dst.data()
        self.assertComplexTuplesAlmostEqual(expected_result, result_data, 5)

    def test_ccc_002(self):
        if False:
            return 10
        tb = gr.top_block()
        src_data = (0, 1, 2, 3, 4, 5, 6, 7)
        taps = (2,)
        nthreads = 2
        expected_result = tuple([2 * complex(x) for x in (0, 1, 2, 3, 4, 5, 6, 7)])
        src = blocks.vector_source_c(src_data)
        op = filter.fft_filter_ccc(1, taps, nthreads)
        dst = blocks.vector_sink_c()
        tb.connect(src, op, dst)
        tb.run()
        result_data = dst.data()
        self.assertComplexTuplesAlmostEqual(expected_result, result_data, 5)

    def test_ccc_003(self):
        if False:
            i = 10
            return i + 15
        tb = gr.top_block()
        src_data = (0, 1, 2, 3, 4, 5, 6, 7)
        taps = (2,)
        expected_result = tuple([2 * complex(x) for x in (0, 1, 2, 3, 4, 5, 6, 7)])
        src = blocks.vector_source_c(src_data)
        op = filter.fft_filter_ccc(1, taps)
        dst = blocks.vector_sink_c()
        tb.connect(src, op, dst)
        tb.run()
        result_data = dst.data()
        self.assertComplexTuplesAlmostEqual(expected_result, result_data, 5)

    def test_ccc_004(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(0)
        for i in range(25):
            src_len = 4 * 1024
            src_data = make_random_complex_tuple(src_len)
            ntaps = int(random.uniform(2, 1000))
            taps = make_random_complex_tuple(ntaps)
            expected_result = reference_filter_ccc(1, taps, src_data)
            src = blocks.vector_source_c(src_data)
            op = filter.fft_filter_ccc(1, taps)
            dst = blocks.vector_sink_c()
            tb = gr.top_block()
            tb.connect(src, op, dst)
            tb.run()
            result_data = dst.data()
            del tb
            self.assert_fft_ok2(expected_result, result_data)

    def test_ccc_005(self):
        if False:
            print('Hello World!')
        random.seed(0)
        for i in range(25):
            dec = i + 1
            src_len = 4 * 1024
            src_data = make_random_complex_tuple(src_len)
            ntaps = int(random.uniform(2, 100))
            taps = make_random_complex_tuple(ntaps)
            expected_result = reference_filter_ccc(dec, taps, src_data)
            src = blocks.vector_source_c(src_data)
            op = filter.fft_filter_ccc(dec, taps)
            dst = blocks.vector_sink_c()
            tb = gr.top_block()
            tb.connect(src, op, dst)
            tb.run()
            del tb
            result_data = dst.data()
            self.assert_fft_ok2(expected_result, result_data)

    def test_ccc_006(self):
        if False:
            return 10
        random.seed(0)
        nthreads = 2
        for i in range(25):
            dec = i + 1
            src_len = 4 * 1024
            src_data = make_random_complex_tuple(src_len)
            ntaps = int(random.uniform(2, 100))
            taps = make_random_complex_tuple(ntaps)
            expected_result = reference_filter_ccc(dec, taps, src_data)
            src = blocks.vector_source_c(src_data)
            op = filter.fft_filter_ccc(dec, taps, nthreads)
            dst = blocks.vector_sink_c()
            tb = gr.top_block()
            tb.connect(src, op, dst)
            tb.run()
            del tb
            result_data = dst.data()
            self.assert_fft_ok2(expected_result, result_data)

    def test_ccf_001(self):
        if False:
            print('Hello World!')
        tb = gr.top_block()
        src_data = (0, 1, 2, 3, 4, 5, 6, 7)
        taps = (1,)
        expected_result = tuple([complex(x) for x in (0, 1, 2, 3, 4, 5, 6, 7)])
        src = blocks.vector_source_c(src_data)
        op = filter.fft_filter_ccf(1, taps)
        dst = blocks.vector_sink_c()
        tb.connect(src, op, dst)
        tb.run()
        result_data = dst.data()
        self.assertComplexTuplesAlmostEqual(expected_result, result_data, 5)

    def test_ccf_002(self):
        if False:
            i = 10
            return i + 15
        tb = gr.top_block()
        src_data = (0, 1, 2, 3, 4, 5, 6, 7)
        taps = (2,)
        nthreads = 2
        expected_result = tuple([2 * complex(x) for x in (0, 1, 2, 3, 4, 5, 6, 7)])
        src = blocks.vector_source_c(src_data)
        op = filter.fft_filter_ccf(1, taps, nthreads)
        dst = blocks.vector_sink_c()
        tb.connect(src, op, dst)
        tb.run()
        result_data = dst.data()
        self.assertComplexTuplesAlmostEqual(expected_result, result_data, 5)

    def test_ccf_003(self):
        if False:
            i = 10
            return i + 15
        tb = gr.top_block()
        src_data = (0, 1, 2, 3, 4, 5, 6, 7)
        taps = (2,)
        expected_result = tuple([2 * complex(x) for x in (0, 1, 2, 3, 4, 5, 6, 7)])
        src = blocks.vector_source_c(src_data)
        op = filter.fft_filter_ccf(1, taps)
        dst = blocks.vector_sink_c()
        tb.connect(src, op, dst)
        tb.run()
        result_data = dst.data()
        self.assertComplexTuplesAlmostEqual(expected_result, result_data, 5)

    def test_ccf_004(self):
        if False:
            return 10
        random.seed(0)
        for i in range(25):
            src_len = 4 * 1024
            src_data = make_random_complex_tuple(src_len)
            ntaps = int(random.uniform(2, 1000))
            taps = make_random_float_tuple(ntaps)
            expected_result = reference_filter_ccf(1, taps, src_data)
            src = blocks.vector_source_c(src_data)
            op = filter.fft_filter_ccf(1, taps)
            dst = blocks.vector_sink_c()
            tb = gr.top_block()
            tb.connect(src, op, dst)
            tb.run()
            result_data = dst.data()
            del tb
            self.assert_fft_ok2(expected_result, result_data)

    def test_ccf_005(self):
        if False:
            return 10
        random.seed(0)
        for i in range(25):
            dec = i + 1
            src_len = 4 * 1024
            src_data = make_random_complex_tuple(src_len)
            ntaps = int(random.uniform(2, 100))
            taps = make_random_float_tuple(ntaps)
            expected_result = reference_filter_ccf(dec, taps, src_data)
            src = blocks.vector_source_c(src_data)
            op = filter.fft_filter_ccf(dec, taps)
            dst = blocks.vector_sink_c()
            tb = gr.top_block()
            tb.connect(src, op, dst)
            tb.run()
            del tb
            result_data = dst.data()
            self.assert_fft_ok2(expected_result, result_data)

    def test_ccf_006(self):
        if False:
            return 10
        random.seed(0)
        nthreads = 2
        for i in range(25):
            dec = i + 1
            src_len = 4 * 1024
            src_data = make_random_complex_tuple(src_len)
            ntaps = int(random.uniform(2, 100))
            taps = make_random_float_tuple(ntaps)
            expected_result = reference_filter_ccf(dec, taps, src_data)
            src = blocks.vector_source_c(src_data)
            op = filter.fft_filter_ccc(dec, taps, nthreads)
            dst = blocks.vector_sink_c()
            tb = gr.top_block()
            tb.connect(src, op, dst)
            tb.run()
            del tb
            result_data = dst.data()
            self.assert_fft_ok2(expected_result, result_data)

    def test_fff_001(self):
        if False:
            for i in range(10):
                print('nop')
        tb = gr.top_block()
        src_data = (0, 1, 2, 3, 4, 5, 6, 7)
        taps = (1,)
        expected_result = tuple([float(x) for x in (0, 1, 2, 3, 4, 5, 6, 7)])
        src = blocks.vector_source_f(src_data)
        op = filter.fft_filter_fff(1, taps)
        dst = blocks.vector_sink_f()
        tb.connect(src, op, dst)
        tb.run()
        result_data = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, result_data, 5)

    def test_fff_002(self):
        if False:
            for i in range(10):
                print('nop')
        tb = gr.top_block()
        src_data = (0, 1, 2, 3, 4, 5, 6, 7)
        taps = (2,)
        expected_result = tuple([2 * float(x) for x in (0, 1, 2, 3, 4, 5, 6, 7)])
        src = blocks.vector_source_f(src_data)
        op = filter.fft_filter_fff(1, taps)
        dst = blocks.vector_sink_f()
        tb.connect(src, op, dst)
        tb.run()
        result_data = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, result_data, 5)

    def test_fff_003(self):
        if False:
            i = 10
            return i + 15
        tb = gr.top_block()
        src_data = (0, 1, 2, 3, 4, 5, 6, 7)
        taps = (2,)
        nthreads = 2
        expected_result = tuple([2 * float(x) for x in (0, 1, 2, 3, 4, 5, 6, 7)])
        src = blocks.vector_source_f(src_data)
        op = filter.fft_filter_fff(1, taps, nthreads)
        dst = blocks.vector_sink_f()
        tb.connect(src, op, dst)
        tb.run()
        result_data = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, result_data, 5)

    def xtest_fff_004(self):
        if False:
            while True:
                i = 10
        random.seed(0)
        for i in range(25):
            sys.stderr.write('\n>>> Loop = %d\n' % (i,))
            src_len = 4096
            src_data = make_random_float_tuple(src_len)
            ntaps = int(random.uniform(2, 1000))
            taps = make_random_float_tuple(ntaps)
            expected_result = reference_filter_fff(1, taps, src_data)
            src = blocks.vector_source_f(src_data)
            op = filter.fft_filter_fff(1, taps)
            dst = blocks.vector_sink_f()
            tb = gr.top_block()
            tb.connect(src, op, dst)
            tb.run()
            result_data = dst.data()
            try:
                self.assert_fft_float_ok2(expected_result, result_data, abs_eps=1.0)
            except AssertionError:
                expected = open('expected', 'w')
                for x in expected_result:
                    expected.write(repr(x) + '\n')
                actual = open('actual', 'w')
                for x in result_data:
                    actual.write(repr(x) + '\n')
                raise

    def xtest_fff_005(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(0)
        for i in range(25):
            sys.stderr.write('\n>>> Loop = %d\n' % (i,))
            src_len = 4 * 1024
            src_data = make_random_float_tuple(src_len)
            ntaps = int(random.uniform(2, 1000))
            taps = make_random_float_tuple(ntaps)
            expected_result = reference_filter_fff(1, taps, src_data)
            src = blocks.vector_source_f(src_data)
            op = filter.fft_filter_fff(1, taps)
            dst = blocks.vector_sink_f()
            tb = gr.top_block()
            tb.connect(src, op, dst)
            tb.run()
            result_data = dst.data()
            self.assert_fft_float_ok2(expected_result, result_data, abs_eps=2.0)

    def xtest_fff_006(self):
        if False:
            print('Hello World!')
        random.seed(0)
        for i in range(25):
            sys.stderr.write('\n>>> Loop = %d\n' % (i,))
            dec = i + 1
            src_len = 4 * 1024
            src_data = make_random_float_tuple(src_len)
            ntaps = int(random.uniform(2, 100))
            taps = make_random_float_tuple(ntaps)
            expected_result = reference_filter_fff(dec, taps, src_data)
            src = blocks.vector_source_f(src_data)
            op = filter.fft_filter_fff(dec, taps)
            dst = blocks.vector_sink_f()
            tb = gr.top_block()
            tb.connect(src, op, dst)
            tb.run()
            result_data = dst.data()
            self.assert_fft_float_ok2(expected_result, result_data)

    def xtest_fff_007(self):
        if False:
            return 10
        random.seed(0)
        nthreads = 2
        for i in range(25):
            sys.stderr.write('\n>>> Loop = %d\n' % (i,))
            dec = i + 1
            src_len = 4 * 1024
            src_data = make_random_float_tuple(src_len)
            ntaps = int(random.uniform(2, 100))
            taps = make_random_float_tuple(ntaps)
            expected_result = reference_filter_fff(dec, taps, src_data)
            src = blocks.vector_source_f(src_data)
            op = filter.fft_filter_fff(dec, taps, nthreads)
            dst = blocks.vector_sink_f()
            tb = gr.top_block()
            tb.connect(src, op, dst)
            tb.run()
            result_data = dst.data()
            self.assert_fft_float_ok2(expected_result, result_data)

    def test_fff_get0(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(0)
        for i in range(25):
            ntaps = int(random.uniform(2, 100))
            taps = list(make_random_float_tuple(ntaps))
            op = filter.fft_filter_fff(1, taps)
            result_data = op.taps()
            self.assertFloatTuplesAlmostEqual(taps, result_data, 4)

    def test_ccc_get0(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(0)
        for i in range(25):
            ntaps = int(random.uniform(2, 100))
            taps = make_random_complex_tuple(ntaps)
            op = filter.fft_filter_ccc(1, taps)
            result_data = op.taps()
            self.assertComplexTuplesAlmostEqual(taps, result_data, 4)
if __name__ == '__main__':
    gr_unittest.run(test_fft_filter)