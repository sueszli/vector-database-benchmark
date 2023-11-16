from gnuradio import gr, gr_unittest, digital, blocks
from math import sqrt
from numpy import random, vectorize

class test_constellation_soft_decoder(gr_unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        random.seed(0)
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def helper_with_lut(self, prec, src_data, const_gen, const_sd_gen, decimals=5):
        if False:
            return 10
        (cnst_pts, code) = const_gen()
        Es = 1.0
        lut = digital.soft_dec_table(cnst_pts, code, prec, Es)
        constel = digital.const_normalization(cnst_pts, 'POWER')
        maxamp = digital.min_max_axes(constel)
        expected_result = list()
        for s in src_data:
            res = digital.calc_soft_dec_from_table(s, lut, prec, maxamp)
            expected_result += res
        cnst = digital.constellation_calcdist(cnst_pts, code, 4, 1, digital.constellation.POWER_NORMALIZATION)
        cnst.set_soft_dec_lut(lut, int(prec))
        cnst.normalize(digital.constellation.POWER_NORMALIZATION)
        src = blocks.vector_source_c(src_data)
        op = digital.constellation_soft_decoder_cf(cnst.base())
        dst = blocks.vector_sink_f()
        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()
        actual_result = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, actual_result, decimals)

    def helper_no_lut(self, prec, src_data, const_gen, const_sd_gen):
        if False:
            while True:
                i = 10
        (cnst_pts, code) = const_gen()
        cnst = digital.constellation_calcdist(cnst_pts, code, 2, 1)
        expected_result = list()
        for s in src_data:
            res = digital.calc_soft_dec(s, cnst.points(), code)
            expected_result += res
        src = blocks.vector_source_c(src_data)
        op = digital.constellation_soft_decoder_cf(cnst.base())
        dst = blocks.vector_sink_f()
        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()
        actual_result = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, actual_result, 4)

    def test_constellation_soft_decoder_cf_bpsk_3(self):
        if False:
            i = 10
            return i + 15
        prec = 3
        src_data = (-1.0 - 1j, 1.0 - 1j, -1.0 + 1j, 1.0 + 1j, -2.0 - 2j, 2.0 - 2j, -2.0 + 2j, 2.0 + 2j, -0.2 - 0.2j, 0.2 - 0.2j, -0.2 + 0.2j, 0.2 + 0.2j, 0.3 + 0.4j, 0.1 - 1.2j, -0.8 - 0.1j, -0.4 + 0.8j, 0.8 + 1j, -0.5 + 0.1j, 0.1 + 1.2j, -1.7 - 0.9j)
        self.helper_with_lut(prec, src_data, digital.psk_2_0x0, digital.sd_psk_2_0x0)

    def test_constellation_soft_decoder_cf_bpsk_8(self):
        if False:
            i = 10
            return i + 15
        prec = 8
        src_data = (-1.0 - 1j, 1.0 - 1j, -1.0 + 1j, 1.0 + 1j, -2.0 - 2j, 2.0 - 2j, -2.0 + 2j, 2.0 + 2j, -0.2 - 0.2j, 0.2 - 0.2j, -0.2 + 0.2j, 0.2 + 0.2j, 0.3 + 0.4j, 0.1 - 1.2j, -0.8 - 0.1j, -0.4 + 0.8j, 0.8 + 1j, -0.5 + 0.1j, 0.1 + 1.2j, -1.7 - 0.9j)
        self.helper_with_lut(prec, src_data, digital.psk_2_0x0, digital.sd_psk_2_0x0)

    def test_constellation_soft_decoder_cf_bpsk_8_rand(self):
        if False:
            while True:
                i = 10
        prec = 8
        src_data = vectorize(complex)(2 * random.randn(100), 2 * random.randn(100))
        self.helper_with_lut(prec, src_data, digital.psk_2_0x0, digital.sd_psk_2_0x0)

    def test_constellation_soft_decoder_cf_bpsk_8_rand2(self):
        if False:
            i = 10
            return i + 15
        prec = 8
        src_data = vectorize(complex)(2 * random.randn(100), 2 * random.randn(100))
        self.helper_no_lut(prec, src_data, digital.psk_2_0x0, digital.sd_psk_2_0x0)

    def test_constellation_soft_decoder_cf_qpsk_3(self):
        if False:
            i = 10
            return i + 15
        prec = 3
        src_data = (-1.0 - 1j, 1.0 - 1j, -1.0 + 1j, 1.0 + 1j, -2.0 - 2j, 2.0 - 2j, -2.0 + 2j, 2.0 + 2j, -0.2 - 0.2j, 0.2 - 0.2j, -0.2 + 0.2j, 0.2 + 0.2j, 0.3 + 0.4j, 0.1 - 1.2j, -0.8 - 0.1j, -0.4 + 0.8j, 0.8 + 1j, -0.5 + 0.1j, 0.1 + 1.2j, -1.7 - 0.9j)
        self.helper_with_lut(prec, src_data, digital.psk_4_0x0_0_1, digital.sd_psk_4_0x0_0_1)

    def test_constellation_soft_decoder_cf_qpsk_8(self):
        if False:
            print('Hello World!')
        prec = 8
        src_data = (-1.0 - 1j, 1.0 - 1j, -1.0 + 1j, 1.0 + 1j, -2.0 - 2j, 2.0 - 2j, -2.0 + 2j, 2.0 + 2j, -0.2 - 0.2j, 0.2 - 0.2j, -0.2 + 0.2j, 0.2 + 0.2j, 0.3 + 0.4j, 0.1 - 1.2j, -0.8 - 0.1j, -0.4 + 0.8j, 0.8 + 1j, -0.5 + 0.1j, 0.1 + 1.2j, -1.7 - 0.9j)
        self.helper_with_lut(prec, src_data, digital.psk_4_0x0_0_1, digital.sd_psk_4_0x0_0_1)

    def test_constellation_soft_decoder_cf_qpsk_8_rand(self):
        if False:
            return 10
        prec = 8
        src_data = vectorize(complex)(2 * random.randn(100), 2 * random.randn(100))
        self.helper_with_lut(prec, src_data, digital.psk_4_0x0_0_1, digital.sd_psk_4_0x0_0_1, 3)

    def test_constellation_soft_decoder_cf_qpsk_8_rand2(self):
        if False:
            return 10
        prec = 8
        src_data = vectorize(complex)(2 * random.randn(100), 2 * random.randn(100))
        self.helper_no_lut(prec, src_data, digital.psk_4_0x0_0_1, digital.sd_psk_4_0x0_0_1)

    def test_constellation_soft_decoder_cf_qam16_3(self):
        if False:
            while True:
                i = 10
        prec = 3
        src_data = (-1.0 - 1j, 1.0 - 1j, -1.0 + 1j, 1.0 + 1j, -2.0 - 2j, 2.0 - 2j, -2.0 + 2j, 2.0 + 2j, -0.2 - 0.2j, 0.2 - 0.2j, -0.2 + 0.2j, 0.2 + 0.2j, 0.3 + 0.4j, 0.1 - 1.2j, -0.8 - 0.1j, -0.4 + 0.8j, 0.8 + 1j, -0.5 + 0.1j, 0.1 + 1.2j, -1.7 - 0.9j)
        self.helper_with_lut(prec, src_data, digital.qam_16_0x0_0_1_2_3, digital.sd_qam_16_0x0_0_1_2_3)

    def test_constellation_soft_decoder_cf_qam16_8(self):
        if False:
            print('Hello World!')
        prec = 8
        src_data = (-1.0 - 1j, 1.0 - 1j, -1.0 + 1j, 1.0 + 1j, -2.0 - 2j, 2.0 - 2j, -2.0 + 2j, 2.0 + 2j, -0.2 - 0.2j, 0.2 - 0.2j, -0.2 + 0.2j, 0.2 + 0.2j, 0.3 + 0.4j, 0.1 - 1.2j, -0.8 - 0.1j, -0.4 + 0.8j, 0.8 + 1j, -0.5 + 0.1j, 0.1 + 1.2j, -1.7 - 0.9j)
        self.helper_with_lut(prec, src_data, digital.qam_16_0x0_0_1_2_3, digital.sd_qam_16_0x0_0_1_2_3)

    def test_constellation_soft_decoder_cf_qam16_8_rand(self):
        if False:
            i = 10
            return i + 15
        prec = 8
        src_data = vectorize(complex)(2 * random.randn(100), 2 * random.randn(100))
        self.helper_with_lut(prec, src_data, digital.qam_16_0x0_0_1_2_3, digital.sd_qam_16_0x0_0_1_2_3, 3)

    def test_constellation_soft_decoder_cf_qam16_8_rand2(self):
        if False:
            print('Hello World!')
        prec = 8
        src_data = vectorize(complex)(2 * random.randn(2), 2 * random.randn(2))
        self.helper_no_lut(prec, src_data, digital.qam_16_0x0_0_1_2_3, digital.sd_qam_16_0x0_0_1_2_3)
if __name__ == '__main__':
    gr_unittest.run(test_constellation_soft_decoder)