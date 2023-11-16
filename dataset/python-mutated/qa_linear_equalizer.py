from gnuradio import gr, gr_unittest
import random
import numpy
from gnuradio import digital, blocks, channels

class qa_linear_equalizer(gr_unittest.TestCase):

    def unpack_values(self, values_in, bits_per_value, bits_per_symbol):
        if False:
            while True:
                i = 10
        m = bits_per_value / bits_per_symbol
        mask = 2 ** bits_per_symbol - 1
        if bits_per_value != m * bits_per_symbol:
            print('error - bits per symbols must fit nicely into bits_per_value bit values')
            return []
        num_values = len(values_in)
        num_symbols = int(num_values * m)
        cur_byte = 0
        cur_bit = 0
        out = []
        for i in range(num_symbols):
            s = values_in[cur_byte] >> bits_per_value - bits_per_symbol - cur_bit & mask
            out.append(s)
            cur_bit += bits_per_symbol
            if cur_bit >= bits_per_value:
                cur_bit = 0
                cur_byte += 1
        return out

    def map_symbols_to_constellation(self, symbols, cons):
        if False:
            i = 10
            return i + 15
        l = list(map(lambda x: cons.points()[x], symbols))
        return l

    def setUp(self):
        if False:
            i = 10
            return i + 15
        random.seed(987654)
        self.tb = gr.top_block()
        self.num_data = num_data = 10000
        self.sps = sps = 4
        self.eb = eb = 0.35
        self.preamble = preamble = [39, 47, 24, 93, 91, 42, 63, 113, 99, 60, 23, 12, 10, 65, 214, 31, 76, 35, 101, 104, 237, 28, 119, 167, 14, 10, 158, 71, 130, 164, 87, 36]
        self.payload_size = payload_size = 300
        self.data = data = [0] * 4 + [random.getrandbits(8) for i in range(payload_size)]
        self.gain = gain = 0.001
        self.corr_thresh = corr_thresh = 3000000.0
        self.num_taps = num_taps = 16

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def transform(self, src_data, const, alg):
        if False:
            while True:
                i = 10
        src = blocks.vector_source_c(src_data, False)
        leq = digital.linear_equalizer(4, 1, alg, True, [], '')
        dst = blocks.vector_sink_c()
        self.tb.connect(src, leq, dst)
        self.tb.run()
        return dst.data()

    def test_001_identity_lms(self):
        if False:
            i = 10
            return i + 15
        const = digital.constellation_qpsk()
        src_data = const.points() * 1000
        alg = digital.adaptive_algorithm_lms(const, 0.1).base()
        N = 100
        expected_data = src_data[N:]
        result = self.transform(src_data, const, alg)[N:]
        N = -500
        self.assertComplexTuplesAlmostEqual(expected_data[N:], result[N:], 5)

    def test_002_identity_cma(self):
        if False:
            while True:
                i = 10
        const = digital.constellation_qpsk()
        src_data = const.points() * 1000
        alg = digital.adaptive_algorithm_cma(const, 0.001, 4).base()
        N = 100
        expected_data = src_data[N:]
        result = self.transform(src_data, const, alg)[N:]
        N = -500
        self.assertComplexTuplesAlmostEqual(expected_data[N:], result[N:], 5)

    def test_qpsk_3tap_lms_training(self):
        if False:
            i = 10
            return i + 15
        gain = 0.01
        num_taps = 16
        num_samp = 2000
        num_test = 500
        cons = digital.constellation_qpsk().base()
        rxmod = digital.generic_mod(cons, False, self.sps, True, self.eb, False, False)
        modulated_sync_word_pre = digital.modulate_vector_bc(rxmod.to_basic_block(), self.preamble + self.preamble, [1])
        modulated_sync_word = modulated_sync_word_pre[86:512 + 86]
        corr_max = numpy.abs(numpy.dot(modulated_sync_word, numpy.conj(modulated_sync_word)))
        corr_calc = self.corr_thresh / (corr_max * corr_max)
        preamble_symbols = self.map_symbols_to_constellation(self.unpack_values(self.preamble, 8, 2), cons)
        alg = digital.adaptive_algorithm_lms(cons, gain).base()
        evm = digital.meas_evm_cc(cons, digital.evm_measurement_t.EVM_PERCENT)
        leq = digital.linear_equalizer(num_taps, self.sps, alg, False, preamble_symbols, 'corr_est')
        correst = digital.corr_est_cc(modulated_sync_word, self.sps, 12, corr_calc, digital.THRESHOLD_ABSOLUTE)
        constmod = digital.generic_mod(constellation=cons, differential=False, samples_per_symbol=4, pre_diff_code=True, excess_bw=0.35, verbose=False, log=False)
        chan = channels.channel_model(noise_voltage=0.0, frequency_offset=0.0, epsilon=1.0, taps=(1.0 + 1j, 0.63 - 0.22j, -0.1 + 0.07j), noise_seed=0, block_tags=False)
        vso = blocks.vector_source_b(self.preamble + self.data, True, 1, [])
        head = blocks.head(gr.sizeof_float * 1, num_samp)
        vsi = blocks.vector_sink_f()
        self.tb.connect(vso, constmod, chan, correst, leq, evm, head, vsi)
        self.tb.run()
        upper_bound = list(20.0 * numpy.ones((num_test,)))
        lower_bound = list(0.0 * numpy.zeros((num_test,)))
        output_data = vsi.data()
        output_data = output_data[-num_test:]
        self.assertLess(output_data, upper_bound)
        self.assertGreater(output_data, lower_bound)
if __name__ == '__main__':
    gr_unittest.run(qa_linear_equalizer)