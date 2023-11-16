import array
import unittest
import numpy as np
from tests.utils_testing import get_path_for_data_file
from urh.cythonext.signal_functions import modulate_c
from urh.signalprocessing.IQArray import IQArray
from urh.signalprocessing.ProtocolAnalyzer import ProtocolAnalyzer
from urh.signalprocessing.Signal import Signal

class TestDemodulations(unittest.TestCase):

    def test_ask(self):
        if False:
            return 10
        signal = Signal(get_path_for_data_file('ask.complex'), 'ASK-Test')
        signal.modulation_type = 'ASK'
        signal.samples_per_symbol = 295
        signal.center = 0.0219
        self.assertEqual(signal.num_samples, 13710)
        proto_analyzer = ProtocolAnalyzer(signal)
        proto_analyzer.get_protocol_from_signal()
        self.assertTrue(proto_analyzer.plain_bits_str[0].startswith('1011001001011011011011011011011011001000000'))

    def test_ask_two(self):
        if False:
            for i in range(10):
                print('nop')
        signal = Signal(get_path_for_data_file('ask_short.complex'), 'ASK-Test2')
        signal.modulation_type = 'ASK'
        signal.noise_threshold = 0.0299
        signal.samples_per_symbol = 16
        signal.center = 0.13
        signal.tolerance = 0
        self.assertEqual(signal.num_samples, 131)
        proto_analyzer = ProtocolAnalyzer(signal)
        proto_analyzer.get_protocol_from_signal()
        self.assertEqual(proto_analyzer.plain_bits_str[0], '10101010')

    def test_fsk(self):
        if False:
            for i in range(10):
                print('nop')
        signal = Signal(get_path_for_data_file('fsk.complex'), 'FSK-Test')
        signal.modulation_type = 'FSK'
        signal.samples_per_symbol = 100
        signal.center = 0
        proto_analyzer = ProtocolAnalyzer(signal)
        proto_analyzer.get_protocol_from_signal()
        self.assertEqual(proto_analyzer.plain_bits_str[0], '101010101010101010101010101010101100011000100110110001100010011011110100110111000001110110011000111011101111011110100100001001111001100110011100110100100011100111010011111100011')

    def test_fsk_short_bit_length(self):
        if False:
            while True:
                i = 10
        bits_str = '101010'
        bits = array.array('B', list(map(int, bits_str)))
        parameters = array.array('f', [-10000.0, 10000.0])
        result = modulate_c(bits, 8, 'FSK', parameters, 1, 1, 40000.0, 0, 1000000.0, 1000, 0)
        signal = Signal('')
        signal.iq_array = IQArray(result)
        self.assertLess(np.max(signal.qad), 1)
        signal.qad_center = 0
        signal.samples_per_symbol = 8
        proto_analyzer = ProtocolAnalyzer(signal)
        proto_analyzer.get_protocol_from_signal()
        self.assertEqual(proto_analyzer.plain_bits_str[0], bits_str)

    def test_psk(self):
        if False:
            print('Hello World!')
        signal = Signal(get_path_for_data_file('psk_gen_noisy.complex'), 'PSK-Test')
        signal.modulation_type = 'PSK'
        signal.samples_per_symbol = 300
        signal.center = 0
        signal.noise_threshold = 0
        signal.tolerance = 10
        proto_analyzer = ProtocolAnalyzer(signal)
        proto_analyzer.get_protocol_from_signal()
        self.assertTrue(proto_analyzer.plain_bits_str[0].startswith('1011'), msg=proto_analyzer.plain_bits_str[0])

    def test_4_psk(self):
        if False:
            while True:
                i = 10
        bits = array.array('B', [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1])
        angles_degree = [-135, -45, 45, 135]
        parameters = array.array('f', [np.pi * a / 180 for a in angles_degree])
        result = modulate_c(bits, 100, 'PSK', parameters, 2, 1, 40000.0, 0, 1000000.0, 1000, 0)
        signal = Signal('')
        signal.iq_array = IQArray(result)
        signal.bits_per_symbol = 2
        signal.center = 0
        signal.center_spacing = 1
        signal.modulation_type = 'PSK'
        proto_analyzer = ProtocolAnalyzer(signal)
        proto_analyzer.get_protocol_from_signal()
        demod_bits = proto_analyzer.plain_bits_str[0]
        self.assertEqual(len(demod_bits), len(bits))
        self.assertTrue(demod_bits.startswith('10101010'))
        np.random.seed(42)
        noised = result + 0.1 * np.random.normal(loc=0, scale=np.sqrt(2) / 2, size=(len(result), 2))
        signal.iq_array = IQArray(noised.astype(np.float32))
        signal.center_spacing = 1.5
        signal.noise_threshold = 0.2
        signal._qad = None
        proto_analyzer.get_protocol_from_signal()
        demod_bits = proto_analyzer.plain_bits_str[0]
        self.assertEqual(len(demod_bits), len(bits))
        self.assertTrue(demod_bits.startswith('10101010'))

    def test_4_fsk(self):
        if False:
            print('Hello World!')
        bits = array.array('B', [1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
        parameters = array.array('f', [-20000.0, -10000.0, 10000.0, 20000.0])
        result = modulate_c(bits, 100, 'FSK', parameters, 2, 1, 40000.0, 0, 1000000.0, 1000, 0)
        signal = Signal('')
        signal.iq_array = IQArray(result)
        signal.bits_per_symbol = 2
        signal.center = 0
        signal.center_spacing = 0.1
        proto_analyzer = ProtocolAnalyzer(signal)
        proto_analyzer.get_protocol_from_signal()
        self.assertEqual(proto_analyzer.plain_bits_str[0], '1010110001')