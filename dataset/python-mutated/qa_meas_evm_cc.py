from gnuradio import gr, gr_unittest
from gnuradio import blocks
import random
import numpy
from gnuradio import digital
from gnuradio import channels

class qa_meas_evm_cc(gr_unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        random.seed(987654)
        self.tb = gr.top_block()
        self.num_data = num_data = 1000

    def tearDown(self):
        if False:
            return 10
        self.tb = None

    def test_qpsk(self):
        if False:
            i = 10
            return i + 15
        expected_result = list(numpy.zeros((self.num_data,)))
        self.cons = cons = digital.constellation_qpsk().base()
        self.data = data = [random.randrange(len(cons.points())) for x in range(self.num_data)]
        self.symbols = symbols = numpy.squeeze([cons.map_to_points_v(i) for i in data])
        evm = digital.meas_evm_cc(cons, digital.evm_measurement_t.EVM_PERCENT)
        vso = blocks.vector_source_c(symbols, False, 1, [])
        vsi = blocks.vector_sink_f()
        self.tb.connect(vso, evm, vsi)
        self.tb.run()
        output_data = vsi.data()
        self.assertEqual(expected_result, output_data)

    def test_qpsk_nonzeroevm(self):
        if False:
            for i in range(10):
                print('nop')
        expected_result = list(numpy.zeros((self.num_data,)))
        self.cons = cons = digital.constellation_qpsk().base()
        self.data = data = [random.randrange(len(cons.points())) for x in range(self.num_data)]
        self.symbols = symbols = numpy.squeeze([cons.map_to_points_v(i) for i in data])
        evm = digital.meas_evm_cc(cons, digital.evm_measurement_t.EVM_PERCENT)
        vso = blocks.vector_source_c(symbols, False, 1, [])
        mc = blocks.multiply_const_cc(3.0 + 2j)
        vsi = blocks.vector_sink_f()
        self.tb.connect(vso, mc, evm, vsi)
        self.tb.run()
        output_data = vsi.data()
        self.assertNotEqual(expected_result, output_data)

    def test_qpsk_channel(self):
        if False:
            while True:
                i = 10
        upper_bound = list(50.0 * numpy.ones((self.num_data,)))
        lower_bound = list(0.0 * numpy.zeros((self.num_data,)))
        self.cons = cons = digital.constellation_qpsk().base()
        self.data = data = [random.randrange(len(cons.points())) for x in range(self.num_data)]
        self.symbols = symbols = numpy.squeeze([cons.map_to_points_v(i) for i in data])
        chan = channels.channel_model(noise_voltage=0.1, frequency_offset=0.0, epsilon=1.0, taps=[1.0 + 0j], noise_seed=0, block_tags=False)
        evm = digital.meas_evm_cc(cons, digital.evm_measurement_t.EVM_PERCENT)
        vso = blocks.vector_source_c(symbols, False, 1, [])
        mc = blocks.multiply_const_cc(3.0 + 2j)
        vsi = blocks.vector_sink_f()
        self.tb.connect(vso, chan, evm, vsi)
        self.tb.run()
        output_data = vsi.data()
        self.assertLess(output_data, upper_bound)
        self.assertGreater(output_data, lower_bound)

    def test_qam16_channel(self):
        if False:
            while True:
                i = 10
        upper_bound = list(50.0 * numpy.ones((self.num_data,)))
        lower_bound = list(0.0 * numpy.zeros((self.num_data,)))
        self.cons = cons = digital.constellation_16qam().base()
        self.data = data = [random.randrange(len(cons.points())) for x in range(self.num_data)]
        self.symbols = symbols = numpy.squeeze([cons.map_to_points_v(i) for i in data])
        chan = channels.channel_model(noise_voltage=0.1, frequency_offset=0.0, epsilon=1.0, taps=[1.0 + 0j], noise_seed=0, block_tags=False)
        evm = digital.meas_evm_cc(cons, digital.evm_measurement_t.EVM_PERCENT)
        vso = blocks.vector_source_c(symbols, False, 1, [])
        mc = blocks.multiply_const_cc(3.0 + 2j)
        vsi = blocks.vector_sink_f()
        self.tb.connect(vso, chan, evm, vsi)
        self.tb.run()
        output_data = vsi.data()
        self.assertLess(output_data, upper_bound)
        self.assertGreater(output_data, lower_bound)
if __name__ == '__main__':
    gr_unittest.run(qa_meas_evm_cc)