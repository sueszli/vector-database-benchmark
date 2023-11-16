import random
import cmath
from gnuradio import gr, gr_unittest, digital, blocks

class test_clock_recovery_mm(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        random.seed(0)
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tb = None

    def test01(self):
        if False:
            print('Hello World!')
        omega = 2
        gain_omega = 0.001
        mu = 0.5
        gain_mu = 0.01
        omega_rel_lim = 0.001
        self.test = digital.clock_recovery_mm_cc(omega, gain_omega, mu, gain_mu, omega_rel_lim)
        data = 100 * [complex(1, 1)]
        self.src = blocks.vector_source_c(data, False)
        self.snk = blocks.vector_sink_c()
        self.tb.connect(self.src, self.test, self.snk)
        self.tb.run()
        expected_result = 100 * [complex(0.99972, 0.99972)]
        dst_data = self.snk.data()
        Ncmp = 30
        len_e = len(expected_result)
        len_d = len(dst_data)
        expected_result = expected_result[len_e - Ncmp:]
        dst_data = dst_data[len_d - Ncmp:]
        self.assertComplexTuplesAlmostEqual(expected_result, dst_data, 5)

    def test02(self):
        if False:
            print('Hello World!')
        omega = 2
        gain_omega = 0.01
        mu = 0.5
        gain_mu = 0.01
        omega_rel_lim = 0.001
        self.test = digital.clock_recovery_mm_ff(omega, gain_omega, mu, gain_mu, omega_rel_lim)
        data = 100 * [1]
        self.src = blocks.vector_source_f(data, False)
        self.snk = blocks.vector_sink_f()
        self.tb.connect(self.src, self.test, self.snk)
        self.tb.run()
        expected_result = 100 * [0.9997]
        dst_data = self.snk.data()
        Ncmp = 30
        len_e = len(expected_result)
        len_d = len(dst_data)
        expected_result = expected_result[len_e - Ncmp:]
        dst_data = dst_data[len_d - Ncmp:]
        self.assertFloatTuplesAlmostEqual(expected_result, dst_data, 4)

    def test03(self):
        if False:
            return 10
        omega = 2
        gain_omega = 0.01
        mu = 0.25
        gain_mu = 0.01
        omega_rel_lim = 0.0001
        self.test = digital.clock_recovery_mm_cc(omega, gain_omega, mu, gain_mu, omega_rel_lim)
        data = 1000 * [complex(1, 1), complex(1, 1), complex(-1, -1), complex(-1, -1)]
        self.src = blocks.vector_source_c(data, False)
        self.snk = blocks.vector_sink_c()
        self.tb.connect(self.src, self.test, self.snk)
        self.tb.run()
        expected_result = 1000 * [complex(-1.2, -1.2), complex(1.2, 1.2)]
        dst_data = self.snk.data()
        Ncmp = 100
        len_e = len(expected_result)
        len_d = len(dst_data)
        expected_result = expected_result[len_e - Ncmp:]
        dst_data = dst_data[len_d - Ncmp:]
        self.assertComplexTuplesAlmostEqual(expected_result, dst_data, 1)

    def test04(self):
        if False:
            while True:
                i = 10
        omega = 2
        gain_omega = 0.01
        mu = 0.25
        gain_mu = 0.1
        omega_rel_lim = 0.001
        self.test = digital.clock_recovery_mm_ff(omega, gain_omega, mu, gain_mu, omega_rel_lim)
        data = 1000 * [1, 1, -1, -1]
        self.src = blocks.vector_source_f(data, False)
        self.snk = blocks.vector_sink_f()
        self.tb.connect(self.src, self.test, self.snk)
        self.tb.run()
        expected_result = 1000 * [-1.2, 1.2]
        dst_data = self.snk.data()
        Ncmp = 100
        len_e = len(expected_result)
        len_d = len(dst_data)
        expected_result = expected_result[len_e - Ncmp:]
        dst_data = dst_data[len_d - Ncmp:]
        self.assertFloatTuplesAlmostEqual(expected_result, dst_data, 1)
if __name__ == '__main__':
    gr_unittest.run(test_clock_recovery_mm)