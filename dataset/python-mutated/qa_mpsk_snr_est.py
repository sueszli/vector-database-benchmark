""" Test digital.mpsk_snr_est_cc """
import random
from gnuradio import gr, gr_unittest, digital, blocks

def random_bit():
    if False:
        for i in range(10):
            print('nop')
    'Create random bits using random() rather than randint(). The latter\n    changed for Python 3.2.'
    return random.random() > 0.5

def get_cplx():
    if False:
        for i in range(10):
            print('nop')
    'Return a BPSK symbol (complex)'
    return complex(2 * random_bit() - 1, 0)

def get_n_cplx():
    if False:
        for i in range(10):
            print('nop')
    'Return random, normal-distributed complex number'
    return complex(random.random() - 0.5, random.random() - 0.5)

class test_mpsk_snr_est(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()
        random.seed(0)
        N = 10000
        self._noise = [get_n_cplx() for _ in range(N)]
        self._bits = [get_cplx() for _ in range(N)]

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tb = None

    def mpsk_snr_est_setup(self, op):
        if False:
            print('Hello World!')
        result = []
        for i in range(1, 6):
            src_data = [b + i * n for (b, n) in zip(self._bits, self._noise)]
            src = blocks.vector_source_c(src_data)
            dst = blocks.null_sink(gr.sizeof_gr_complex)
            tb = gr.top_block()
            tb.connect(src, op)
            tb.connect(op, dst)
            tb.run()
            result.append(op.snr())
        return result

    def test_mpsk_snr_est_simple(self):
        if False:
            while True:
                i = 10
        expected_result = [8.2, 4.99, 3.23, 2.01, 1.03]
        N = 10000
        alpha = 0.001
        op = digital.mpsk_snr_est_cc(digital.SNR_EST_SIMPLE, N, alpha)
        actual_result = self.mpsk_snr_est_setup(op)
        self.assertFloatTuplesAlmostEqual(expected_result, actual_result, 2)

    def test_mpsk_snr_est_skew(self):
        if False:
            while True:
                i = 10
        expected_result = [8.31, 1.83, -1.68, -3.56, -4.68]
        N = 10000
        alpha = 0.001
        op = digital.mpsk_snr_est_cc(digital.SNR_EST_SKEW, N, alpha)
        actual_result = self.mpsk_snr_est_setup(op)
        self.assertFloatTuplesAlmostEqual(expected_result, actual_result, 2)

    def test_mpsk_snr_est_m2m4(self):
        if False:
            i = 10
            return i + 15
        expected_result = [8.01, 3.19, 1.97, 2.15, 2.65]
        N = 10000
        alpha = 0.001
        op = digital.mpsk_snr_est_cc(digital.SNR_EST_M2M4, N, alpha)
        actual_result = self.mpsk_snr_est_setup(op)
        self.assertFloatTuplesAlmostEqual(expected_result, actual_result, 2)

    def test_mpsk_snr_est_svn(self):
        if False:
            print('Hello World!')
        expected_result = [7.91, 3.01, 1.77, 1.97, 2.49]
        N = 10000
        alpha = 0.001
        op = digital.mpsk_snr_est_cc(digital.SNR_EST_SVR, N, alpha)
        actual_result = self.mpsk_snr_est_setup(op)
        self.assertFloatTuplesAlmostEqual(expected_result, actual_result, 2)

    def test_probe_mpsk_snr_est_m2m4(self):
        if False:
            i = 10
            return i + 15
        expected_result = [8.01, 3.19, 1.97, 2.15, 2.65]
        actual_result = []
        for i in range(1, 6):
            src_data = [b + i * n for (b, n) in zip(self._bits, self._noise)]
            src = blocks.vector_source_c(src_data)
            N = 10000
            alpha = 0.001
            op = digital.probe_mpsk_snr_est_c(digital.SNR_EST_M2M4, N, alpha)
            tb = gr.top_block()
            tb.connect(src, op)
            tb.run()
            actual_result.append(op.snr())
        self.assertFloatTuplesAlmostEqual(expected_result, actual_result, 2)
if __name__ == '__main__':
    gr_unittest.run(test_mpsk_snr_est)