import unittest
from impacket.dot11 import RadioTap
from impacket.ImpactPacket import Data

class TestRadioTap(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.frame_0 = b''
        self.frame_0 += b'\x00'
        self.frame_0 += b'\x00'
        self.frame_0 += b'\x18\x00'
        self.frame_0 += b'\x0eX\x00\x00'
        self.frame_0 += b'\x10'
        self.frame_0 += b'l'
        self.frame_0 += b'l\t'
        self.frame_0 += b'\x80\x04'
        self.frame_0 += b'\x00'
        self.frame_0 += b'\x1e'
        self.frame_0 += b'\x00\x00'
        self.frame_0 += b'\x00\x00\x00\x00\x00\x00\x08\x02,\x00\x00\x1f\xe1\x19\xe4\xe4\x00\x1b\x9e\xceT\t\x00\x1b\x9e\xceT\t\xe0\xac\xaa\xaa\x03\x00\x00\x00\x08\x06\x00\x01\x08\x00\x06\x04\x00\x02\x00\x1b\x9e\xceT\t\xc0\xa8\x01\x01\x00\x1f\xe1\x19\xe4\xe4\xc0\xa8\x01p\x01p\xe0\x00\x00\xfb\x94\x04\x00\x00\x16\x00\x00\x00\xe0\x00\x00\xfb\x17\\\xa6\xca'
        self.rt0 = RadioTap(self.frame_0)
        self.frame_1 = b''
        self.frame_1 += b'\x00'
        self.frame_1 += b'\x00'
        self.frame_1 += b' \x00'
        self.frame_1 += b'g\x08\x04\x00'
        self.frame_1 += b'0\x03\x1a%\x00\x00\x00\x00'
        self.frame_1 += b'"'
        self.frame_1 += b'\x0c'
        self.frame_1 += b'\xd9'
        self.frame_1 += b'\xa0'
        self.frame_1 += b'\x02'
        self.frame_1 += b'\x00\x00\x00'
        self.frame_1 += b'@\x01\x00\x00'
        self.frame_1 += b'<\x14'
        self.frame_1 += b'$'
        self.frame_1 += b'\x11'
        self.frame_1 += b'\x08\x02\x00\x00\xff\xff\xff\xff\xff\xff\x06\x03\x7f\x07\xa0\x16\x00\x19\xe3\xd3SR\x90\x7f\xaa\xaa\x03\x00\x00\x00\x08\x06\x00\x01\x08\x00\x06\x04\x00\x01\x00\x19\xe3\xd3SR\xa9\xfe\xf7\x00\x00\x00\x00\x00\x00\x00C\x08\x0e6'
        self.rt1 = RadioTap(self.frame_1)
        self.frame_2 = b''
        self.frame_2 += b'\x00'
        self.frame_2 += b'\x00'
        self.frame_2 += b'$\x00'
        self.frame_2 += b'/@\x00\xa0'
        self.frame_2 += b' \x08\x00\x00'
        self.frame_2 += b'\x00\x00\x00\x00'
        self.frame_2 += b'\x97\xd3&D\x06\x00\x00\x00'
        self.frame_2 += b'\x10'
        self.frame_2 += b'\x02'
        self.frame_2 += b'l\t'
        self.frame_2 += b'\xc0\x00'
        self.frame_2 += b'\xa6'
        self.frame_2 += b'\x00'
        self.frame_2 += b'\x00\x00'
        self.frame_2 += b'\xa6'
        self.frame_2 += b'\x00'
        self.frame_2 += b'\xd4\x00\x00\x00\x9c\x04\xebM\xdbS\x8d\xf3\xc6\xc3'
        self.rt2 = RadioTap(self.frame_2)
        self.frame_3 = b''
        self.frame_3 += b'\x00'
        self.frame_3 += b'\x00'
        self.frame_3 += b'$\x00'
        self.frame_3 += b'/@\x00\xa0'
        self.frame_3 += b' \x08\x00\x80'
        self.frame_3 += b'\x00\x00\x00\x00'
        self.frame_3 += b'\x97\xd3&D\x06\x00\x00\x00'
        self.frame_3 += b'\x10'
        self.frame_3 += b'\x02'
        self.frame_3 += b'l\t'
        self.frame_3 += b'\xc0\x00'
        self.frame_3 += b'\xa6'
        self.frame_3 += b'\x00'
        self.frame_3 += b'\x00\x00'
        self.frame_3 += b'\xa6'
        self.frame_3 += b'\x00'
        self.frame_3 += b'\xd4\x00\x00\x00\x9c\x04\xebM\xdbS\x8d\xf3\xc6\xc3'
        self.rt3 = RadioTap(self.frame_3)

    def test_01_sizes(self):
        if False:
            print('Hello World!')
        'Test RadioTap frame sizes'
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.assertEqual(self.rt0.get_body_size(), len(self.frame_0) - 24)
        self.assertEqual(self.rt0.get_tail_size(), 0)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_body_size(), len(self.frame_1) - 32)
        self.assertEqual(self.rt1.get_tail_size(), 0)

    def test_02_version(self):
        if False:
            return 10
        'Test RadioTap version getter/setter'
        self.assertEqual(self.rt0.get_version(), 0)
        self.rt0.set_version(1)
        self.assertEqual(self.rt0.get_version(), 1)
        self.assertEqual(self.rt1.get_version(), 0)
        self.rt1.set_version(1)
        self.assertEqual(self.rt1.get_version(), 1)

    def test_03_present(self):
        if False:
            while True:
                i = 10
        'Test RadioTap present getter'
        self.assertEqual(self.rt0.get_present(), 22542)
        self.assertEqual(self.rt1.get_present(), 264295)

    def test_04_present_bits(self):
        if False:
            print('Hello World!')
        'Test RadioTap present bits tester'
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_TSFT), False)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_FLAGS), True)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_RATE), True)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_CHANNEL), True)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_FHSS), False)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_DBM_ANTSIGNAL), False)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_DBM_ANTNOISE), False)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_LOCK_QUALITY), False)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_TX_ATTENUATION), False)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_DB_TX_ATTENUATION), False)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_DBM_TX_POWER), False)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_ANTENNA), True)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_DB_ANTSIGNAL), True)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_DB_ANTNOISE), False)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_FCS_IN_HEADER), True)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_TX_FLAGS), False)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_RTS_RETRIES), False)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_DATA_RETRIES), False)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_XCHANNEL), False)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_EXT), False)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_TSFT), True)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_FLAGS), True)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_RATE), True)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_CHANNEL), False)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_FHSS), False)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_DBM_ANTSIGNAL), True)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_DBM_ANTNOISE), True)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_LOCK_QUALITY), False)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_TX_ATTENUATION), False)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_DB_TX_ATTENUATION), False)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_DBM_TX_POWER), False)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_ANTENNA), True)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_DB_ANTSIGNAL), False)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_DB_ANTNOISE), False)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_FCS_IN_HEADER), False)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_TX_FLAGS), False)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_RTS_RETRIES), False)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_DATA_RETRIES), False)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_XCHANNEL), True)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_EXT), False)

    def test_05_tsft(self):
        if False:
            for i in range(10):
                print('nop')
        'Test RadioTap tstf getter'
        self.assertEqual(self.rt0.get_tsft(), None)
        self.assertEqual(self.rt1.get_tsft(), 622461744)

    def test_06_tsft(self):
        if False:
            while True:
                i = 10
        'Test RadioTap tstf getter/setter'
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.rt0.set_tsft(72623859790382856)
        self.assertEqual(self.rt0.get_tsft(), 72623859790382856)
        self.assertEqual(self.rt0.get_header_size(), 24 + 8)
        self.rt0.set_tsft(578437695752307201)
        self.assertEqual(self.rt0.get_tsft(), 578437695752307201)
        self.assertEqual(self.rt0.get_header_size(), 24 + 8)

    def test_07_unset_fields(self):
        if False:
            while True:
                i = 10
        'Test RadioTap unset field'
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_FLAGS), True)
        self.rt0.unset_field(RadioTap.RTF_FLAGS)
        self.assertEqual(self.rt0.get_size(), len(self.frame_0) - 1)
        self.assertEqual(self.rt0.get_header_size(), 24 - 1)
        self.assertEqual(self.rt0.get_present_bit(RadioTap.RTF_FLAGS), False)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_TSFT), True)
        self.rt1.unset_field(RadioTap.RTF_TSFT)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1) - 8)
        self.assertEqual(self.rt1.get_header_size(), 32 - 8)
        self.assertEqual(self.rt1.get_present_bit(RadioTap.RTF_TSFT), False)

    def test_08_flags_field(self):
        if False:
            for i in range(10):
                print('nop')
        'Test RadioTap flags getter/setter'
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.assertEqual(self.rt0.get_flags(), 16)
        self.rt0.set_flags(171)
        self.assertEqual(self.rt0.get_flags(), 171)
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_flags(), 34)
        self.rt1.set_flags(171)
        self.assertEqual(self.rt1.get_flags(), 171)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)

    def test_09_rate_field(self):
        if False:
            print('Hello World!')
        'Test RadioTap rate getter/setter'
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.assertEqual(self.rt0.get_rate(), 108)
        self.rt0.set_rate(171)
        self.assertEqual(self.rt0.get_rate(), 171)
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_rate(), 12)
        self.rt1.set_rate(171)
        self.assertEqual(self.rt1.get_rate(), 171)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)

    def test_10_channel_field(self):
        if False:
            i = 10
            return i + 15
        'Test RadioTap channel getter/setter'
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.assertEqual(self.rt0.get_channel(), (2412, 1152))
        self.rt0.set_channel(freq=1234, flags=22136)
        self.assertEqual(self.rt0.get_channel(), (1234, 22136))
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_channel(), None)
        self.rt1.set_channel(freq=1234, flags=22136)
        self.assertEqual(self.rt1.get_channel(), (1234, 22136))
        self.assertEqual(self.rt1.get_size(), len(self.frame_1) + 4)
        self.assertEqual(self.rt1.get_header_size(), 32 + 4)

    def test_11_fhss_field(self):
        if False:
            return 10
        'Test RadioTap FHSS getter/setter'
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_FHSS(), None)
        self.rt1.set_FHSS(hop_set=171, hop_pattern=205)
        self.assertEqual(self.rt1.get_FHSS(), (171, 205))
        self.assertEqual(self.rt1.get_size(), len(self.frame_1) + 2)
        self.assertEqual(self.rt1.get_header_size(), 32 + 2)

    def test_12_dbm_ant_signal_field(self):
        if False:
            i = 10
            return i + 15
        'Test RadioTap dBm Antenna Signal getter/setter'
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_dBm_ant_signal(), 217)
        self.rt1.set_dBm_ant_signal(signal=241)
        self.assertEqual(self.rt1.get_dBm_ant_signal(), 241)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.assertEqual(self.rt0.get_dBm_ant_signal(), None)
        self.rt0.set_dBm_ant_signal(signal=241)
        self.assertEqual(self.rt0.get_dBm_ant_signal(), 241)
        self.assertEqual(self.rt0.get_size(), len(self.frame_0) + 1)
        self.assertEqual(self.rt0.get_header_size(), 24 + 1)

    def test_13_dbm_ant_noise_field(self):
        if False:
            return 10
        'Test RadioTap dBm Antenna Noise getter/setter'
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_dBm_ant_noise(), 160)
        self.rt1.set_dBm_ant_noise(signal=241)
        self.assertEqual(self.rt1.get_dBm_ant_noise(), 241)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.assertEqual(self.rt0.get_dBm_ant_noise(), None)
        self.rt0.set_dBm_ant_noise(signal=241)
        self.assertEqual(self.rt0.get_dBm_ant_noise(), 241)
        self.assertEqual(self.rt0.get_size(), len(self.frame_0) + 1)
        self.assertEqual(self.rt0.get_header_size(), 24 + 1)

    def test_14_lock_quality_field(self):
        if False:
            print('Hello World!')
        'Test RadioTap Lock Quality getter/setter'
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_lock_quality(), None)
        self.rt1.set_lock_quality(quality=43962)
        self.assertEqual(self.rt1.get_lock_quality(), 43962)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1) + 2)
        self.assertEqual(self.rt1.get_header_size(), 32 + 2)

    def test_15_tx_attenuation_field(self):
        if False:
            for i in range(10):
                print('nop')
        'Test RadioTap Tx Attenuation getter/setter'
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_tx_attenuation(), None)
        self.rt1.set_tx_attenuation(power=43962)
        self.assertEqual(self.rt1.get_tx_attenuation(), 43962)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1) + 2)
        self.assertEqual(self.rt1.get_header_size(), 32 + 2)

    def test_16_dB_tx_attenuation_field(self):
        if False:
            print('Hello World!')
        'Test RadioTap dB Tx Attenuation getter/setter'
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_dB_tx_attenuation(), None)
        self.rt1.set_dB_tx_attenuation(power=43962)
        self.assertEqual(self.rt1.get_dB_tx_attenuation(), 43962)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1) + 2)
        self.assertEqual(self.rt1.get_header_size(), 32 + 2)

    def test_17_dbm_tx_power_field(self):
        if False:
            i = 10
            return i + 15
        'Test RadioTap dBm Tx Power getter/setter'
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_dBm_tx_power(), None)
        self.rt1.set_dBm_tx_power(power=-8)
        self.assertEqual(self.rt1.get_dBm_tx_power(), -8)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1) + 1)
        self.assertEqual(self.rt1.get_header_size(), 32 + 1)

    def test_18_antenna_field(self):
        if False:
            i = 10
            return i + 15
        'Test RadioTap Antenna getter/setter'
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_antenna(), 2)
        self.rt1.set_antenna(antenna_index=241)
        self.assertEqual(self.rt1.get_antenna(), 241)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.assertEqual(self.rt0.get_antenna(), 0)
        self.rt0.set_antenna(antenna_index=241)
        self.assertEqual(self.rt0.get_antenna(), 241)
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)

    def test_19_db_ant_signal_field(self):
        if False:
            return 10
        'Test RadioTap dB Antenna Signal getter/setter'
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.assertEqual(self.rt0.get_dB_ant_signal(), 30)
        self.rt0.set_dB_ant_signal(signal=241)
        self.assertEqual(self.rt0.get_dB_ant_signal(), 241)
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_dB_ant_signal(), None)
        self.rt1.set_dB_ant_signal(signal=241)
        self.assertEqual(self.rt1.get_dB_ant_signal(), 241)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1) + 1)
        self.assertEqual(self.rt1.get_header_size(), 32 + 1)

    def test_20_db_ant_noise_field(self):
        if False:
            i = 10
            return i + 15
        'Test RadioTap dB Antenna Noise getter/setter'
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_dB_ant_noise(), None)
        self.rt1.set_dB_ant_noise(signal=241)
        self.assertEqual(self.rt1.get_dB_ant_noise(), 241)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1) + 1)
        self.assertEqual(self.rt1.get_header_size(), 32 + 1)
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.assertEqual(self.rt0.get_dB_ant_noise(), None)
        self.rt0.set_dB_ant_noise(signal=241)
        self.assertEqual(self.rt0.get_dB_ant_noise(), 241)
        self.assertEqual(self.rt0.get_size(), len(self.frame_0) + 1)
        self.assertEqual(self.rt0.get_header_size(), 24 + 1)

    def test_22_fcs_in_header_field(self):
        if False:
            for i in range(10):
                print('nop')
        'Test RadioTap FCS in header getter/setter'
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.assertEqual(self.rt0.get_FCS_in_header(), 0)
        self.rt0.set_FCS_in_header(fcs=2309737967)
        self.assertEqual(self.rt0.get_FCS_in_header(), 2309737967)
        self.assertEqual(self.rt0.get_size(), len(self.frame_0))
        self.assertEqual(self.rt0.get_header_size(), 24)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_FCS_in_header(), None)
        self.rt1.set_FCS_in_header(fcs=2309737967)
        self.assertEqual(self.rt1.get_FCS_in_header(), 2309737967)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1) + 4)
        self.assertEqual(self.rt1.get_header_size(), 32 + 4)

    def test_24_rts_retries_field(self):
        if False:
            return 10
        'Test RadioTap RTS retries getter/setter'
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_RTS_retries(), None)
        self.rt1.set_RTS_retries(retries=186)
        self.assertEqual(self.rt1.get_RTS_retries(), 186)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1) + 1)
        self.assertEqual(self.rt1.get_header_size(), 32 + 1)

    def test_25_tx_flags_field(self):
        if False:
            for i in range(10):
                print('nop')
        'Test RadioTap TX flags getter/setter'
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_tx_flags(), None)
        self.rt1.set_tx_flags(flags=43962)
        self.assertEqual(self.rt1.get_tx_flags(), 43962)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1) + 2)
        self.assertEqual(self.rt1.get_header_size(), 32 + 2)

    def test_26_xchannel_field(self):
        if False:
            i = 10
            return i + 15
        'Test RadioTap xchannel getter/setter'
        (ch_type, ch_freq, ch_num, ch_maxpower) = self.rt1.get_xchannel()
        self.assertEqual(ch_type, 320)
        self.assertEqual(ch_freq, 5180)
        self.assertEqual(ch_num, 36)
        self.assertEqual(ch_maxpower, 17)
        (ch_type, ch_freq, ch_num, ch_maxpower) = (305419896, 1234, 12, 34)
        self.rt1.set_xchannel(flags=ch_type, freq=ch_freq, channel=ch_num, maxpower=ch_maxpower)
        (nch_type, nch_freq, nch_num, nch_maxpower) = self.rt1.get_xchannel()
        self.assertEqual(ch_type, nch_type)
        self.assertEqual(ch_freq, nch_freq)
        self.assertEqual(ch_num, nch_num)
        self.assertEqual(ch_maxpower, nch_maxpower)

    def test_27_data_retries_field(self):
        if False:
            for i in range(10):
                print('nop')
        'Test RadioTap Data retries getter/setter'
        self.assertEqual(self.rt1.get_size(), len(self.frame_1))
        self.assertEqual(self.rt1.get_header_size(), 32)
        self.assertEqual(self.rt1.get_data_retries(), None)
        self.rt1.set_data_retries(retries=171)
        self.assertEqual(self.rt1.get_data_retries(), 171)
        self.assertEqual(self.rt1.get_size(), len(self.frame_1) + 1)
        self.assertEqual(self.rt1.get_header_size(), 32 + 1)

    def test_29_radiotap_length_field(self):
        if False:
            print('Hello World!')
        'Test RadioTap header length field'
        rt = RadioTap()
        self.assertEqual(rt.get_header_length(), 8)
        raw_packet = rt.get_packet()
        self.assertEqual(raw_packet, b'\x00\x00\x08\x00\x00\x00\x00\x00')
        raw_packet = RadioTap().get_packet()
        self.assertEqual(raw_packet, b'\x00\x00\x08\x00\x00\x00\x00\x00')

    def test_30_radiotap_length_filed_with_payload(self):
        if False:
            i = 10
            return i + 15
        'Test RadioTap header length field with payload'
        rt = RadioTap()
        self.assertEqual(rt.get_header_length(), 8)
        data = Data(b'aa')
        rt.contains(data)
        self.assertEqual(rt.get_header_length(), 8)
        raw_packet = rt.get_packet()
        self.assertEqual(raw_packet, b'\x00\x00\x08\x00\x00\x00\x00\x00aa')

    def test_31_radiotap_present_flags_extended(self):
        if False:
            while True:
                i = 10
        'Test RadioTap extended present flags'
        self.assertEqual(self.rt2.get_present_bit(RadioTap.RTF_EXT), True)
        self.assertEqual(self.rt2.get_present_bit(RadioTap.RTF_RATE), True)
        self.assertEqual(self.rt2.get_present_bit(RadioTap.RTF_CHANNEL), True)
        self.assertEqual(self.rt2.get_channel(), (2412, 192))
        self.assertEqual(self.rt2.get_rate(), 2)
        self.assertEqual(self.rt2.get_dBm_ant_signal(), 166)
        self.assertEqual(self.rt3.get_present_bit(RadioTap.RTF_EXT), True)
        self.assertEqual(self.rt3.get_present_bit(RadioTap.RTF_RATE), True)
        self.assertEqual(self.rt3.get_present_bit(RadioTap.RTF_CHANNEL), True)
        self.assertEqual(self.rt3.get_channel(), (2412, 192))
        self.assertEqual(self.rt3.get_rate(), 2)
        self.assertEqual(self.rt3.get_dBm_ant_signal(), 166)
if __name__ == '__main__':
    unittest.main(verbosity=1)