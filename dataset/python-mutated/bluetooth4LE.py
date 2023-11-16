"""Bluetooth 4LE layer"""
import struct
from scapy.compat import orb, chb
from scapy.config import conf
from scapy.data import DLT_BLUETOOTH_LE_LL, DLT_BLUETOOTH_LE_LL_WITH_PHDR, PPI_BTLE
from scapy.packet import Packet, bind_layers
from scapy.fields import BitEnumField, BitField, ByteEnumField, ByteField, Field, FlagsField, LEIntField, LEShortEnumField, LEShortField, MACField, PacketListField, SignedByteField, X3BytesField, XByteField, XIntField, XLEIntField, XLELongField, XLEShortField, XShortField
from scapy.contrib.ethercat import LEBitEnumField, LEBitField
from scapy.layers.bluetooth import EIR_Hdr, L2CAP_Hdr
from scapy.layers.ppi import PPI_Element, PPI_Hdr
from scapy.utils import mac2str, str2mac

class BTLE_PPI(PPI_Element):
    """Cooked BTLE PPI header

    See ``ppi_btle_t`` in
    https://github.com/greatscottgadgets/libbtbb/blob/master/lib/src/pcap.c
    """
    name = 'BTLE PPI header'
    fields_desc = [ByteField('btle_version', 0), LEShortField('btle_channel', None), ByteField('btle_clkn_high', None), LEIntField('btle_clk_100ns', None), SignedByteField('rssi_max', None), SignedByteField('rssi_min', None), SignedByteField('rssi_avg', None), ByteField('rssi_count', None)]

class BTLE_RF(Packet):
    """Cooked BTLE link-layer pseudoheader.

    https://www.tcpdump.org/linktypes/LINKTYPE_BLUETOOTH_LE_LL_WITH_PHDR.html
    """
    name = 'BTLE RF info header'
    _TYPES = {0: 'ADV_OR_DATA_UNKNOWN_DIR', 1: 'AUX_ADV', 2: 'DATA_M_TO_S', 3: 'DATA_S_TO_M', 4: 'CONN_ISO_M_TO_S', 5: 'CONN_ISO_S_TO_M', 6: 'BROADCAST_ISO', 7: 'RFU'}
    _PHY = {0: '1M', 1: '2M', 2: 'Coded', 3: 'RFU'}
    fields_desc = [ByteField('rf_channel', 0), SignedByteField('signal', -128), SignedByteField('noise', -128), ByteField('access_address_offenses', 0), XLEIntField('reference_access_address', 0), LEBitField('dewhitened', 0, 1), LEBitField('sig_power_valid', 0, 1), LEBitField('noise_power_valid', 0, 1), LEBitField('decrypted', 0, 1), LEBitField('reference_access_address_valid', 0, 1), LEBitField('access_address_offenses_valid', 0, 1), LEBitField('channel_aliased', 0, 1), LEBitEnumField('type', 0, 3, _TYPES), LEBitField('crc_checked', 0, 1), LEBitField('crc_valid', 0, 1), LEBitField('mic_checked', 0, 1), LEBitField('mic_valid', 0, 1), LEBitEnumField('phy', 0, 2, _PHY)]

class BDAddrField(MACField):

    def __init__(self, name, default, resolve=False):
        if False:
            print('Hello World!')
        MACField.__init__(self, name, default)
        if resolve:
            conf.resolve.add(self)

    def i2m(self, pkt, x):
        if False:
            for i in range(10):
                print('nop')
        if x is None:
            return b'\x00\x00\x00\x00\x00\x00'
        return mac2str(':'.join(x.split(':')[::-1]))

    def m2i(self, pkt, x):
        if False:
            print('Hello World!')
        return str2mac(x[::-1])

class BTLEChanMapField(XByteField):

    def __init__(self, name, default):
        if False:
            i = 10
            return i + 15
        Field.__init__(self, name, default, '<Q')

    def addfield(self, pkt, s, val):
        if False:
            return 10
        return s + struct.pack(self.fmt, self.i2m(pkt, val))[:5]

    def getfield(self, pkt, s):
        if False:
            for i in range(10):
                print('nop')
        return (s[5:], self.m2i(pkt, struct.unpack(self.fmt, s[:5] + b'\x00\x00\x00')[0]))

class BTLEFeatureField(FlagsField):

    def __init__(self, name, default):
        if False:
            while True:
                i = 10
        super(BTLEFeatureField, self).__init__(name, default, -64, ['le_encryption', 'conn_par_req_proc', 'ext_reject_ind', 'slave_init_feat_exch', 'le_ping', 'le_data_len_ext', 'll_privacy', 'ext_scan_filter', 'le_2m_phy', 'tx_mod_idx', 'rx_mod_idx', 'le_coded_phy', 'le_ext_adv', 'le_periodic_adv', 'ch_sel_alg', 'le_pwr_classmin_used_channels', 'conn_cte_req', 'conn_cte_rsp', 'connless_cte_tx', 'connless_cte_rx', 'antenna_switching_cte_aod_tx', 'antenna_switching_cte_aoa_rx', 'cte_rx', 'periodic_adv_sync_transfer_tx', 'periodic_adv_sync_transfer_rx', 'sleep_clock_accuracy_updates', 'remote_public_key_validation', 'cis_central', 'cis_peripheral', 'iso_broadcaster', 'synchronized_receiver', 'connected_iso_host_support', 'le_power_control_request', 'le_power_control_request', 'le_path_loss_monitoring', 'periodic_adv_adi_support', 'connection_subrating', 'connection_subrating_host_support', 'channel_classification'])

class BTLEPhysField(FlagsField):

    def __init__(self, name, default):
        if False:
            return 10
        super(BTLEPhysField, self).__init__(name, default, -8, ['phy_1m', 'phy_2m', 'phy_coded'])

class BTLE(Packet):
    name = 'BT4LE'
    fields_desc = [XLEIntField('access_addr', 2391391958), X3BytesField('crc', None)]

    @staticmethod
    def compute_crc(pdu, init=5592405):
        if False:
            for i in range(10):
                print('nop')

        def swapbits(a):
            if False:
                for i in range(10):
                    print('nop')
            v = 0
            if a & 128 != 0:
                v |= 1
            if a & 64 != 0:
                v |= 2
            if a & 32 != 0:
                v |= 4
            if a & 16 != 0:
                v |= 8
            if a & 8 != 0:
                v |= 16
            if a & 4 != 0:
                v |= 32
            if a & 2 != 0:
                v |= 64
            if a & 1 != 0:
                v |= 128
            return v
        state = swapbits(init & 255) + (swapbits(init >> 8 & 255) << 8) + (swapbits(init >> 16 & 255) << 16)
        lfsr_mask = 5922816
        for i in (orb(x) for x in pdu):
            for j in range(8):
                next_bit = (state ^ i) & 1
                i >>= 1
                state >>= 1
                if next_bit:
                    state |= 1 << 23
                    state ^= lfsr_mask
        return struct.pack('<L', state)[:-1]

    def post_build(self, p, pay):
        if False:
            return 10
        crc = p[-3:]
        p = p[:-3] + pay
        p += crc if self.crc is not None else self.compute_crc(p[4:])
        return p

    def post_dissect(self, s):
        if False:
            return 10
        self.raw_packet_cache = None
        return s

    def pre_dissect(self, s):
        if False:
            print('Hello World!')
        return s[:4] + s[-3:] + s[4:-3]

    def hashret(self):
        if False:
            i = 10
            return i + 15
        return struct.pack('!L', self.access_addr)

class BTLE_ADV(Packet):
    name = 'BTLE advertising header'
    fields_desc = [BitEnumField('RxAdd', 0, 1, {0: 'public', 1: 'random'}), BitEnumField('TxAdd', 0, 1, {0: 'public', 1: 'random'}), BitEnumField('ChSel', 0, 1, {1: '#2'}), BitField('RFU', 0, 1), BitEnumField('PDU_type', 0, 4, {0: 'ADV_IND', 1: 'ADV_DIRECT_IND', 2: 'ADV_NONCONN_IND', 3: 'SCAN_REQ', 4: 'SCAN_RSP', 5: 'CONNECT_REQ', 6: 'ADV_SCAN_IND'}), XByteField('Length', None)]

    def post_build(self, p, pay):
        if False:
            for i in range(10):
                print('nop')
        p += pay
        if self.Length is None:
            if len(pay) > 2:
                l_pay = len(pay)
            else:
                l_pay = 0
            p = p[:1] + chb(l_pay & 255) + p[2:]
        if not isinstance(self.underlayer, BTLE):
            self.add_underlayer(BTLE)
        return p

class BTLE_DATA(Packet):
    name = 'BTLE data header'
    fields_desc = [BitField('RFU', 0, 3), BitField('MD', 0, 1), BitField('SN', 0, 1), BitField('NESN', 0, 1), BitEnumField('LLID', 0, 2, {1: 'continue', 2: 'start', 3: 'control'}), ByteField('len', None)]

    def post_build(self, p, pay):
        if False:
            while True:
                i = 10
        if self.len is None:
            p = p[:-1] + chb(len(pay))
        return p + pay

class BTLE_ADV_IND(Packet):
    name = 'BTLE ADV_IND'
    fields_desc = [BDAddrField('AdvA', None), PacketListField('data', None, EIR_Hdr)]

class BTLE_ADV_DIRECT_IND(Packet):
    name = 'BTLE ADV_DIRECT_IND'
    fields_desc = [BDAddrField('AdvA', None), BDAddrField('InitA', None)]

class BTLE_ADV_NONCONN_IND(BTLE_ADV_IND):
    name = 'BTLE ADV_NONCONN_IND'

class BTLE_ADV_SCAN_IND(BTLE_ADV_IND):
    name = 'BTLE ADV_SCAN_IND'

class BTLE_SCAN_REQ(Packet):
    name = 'BTLE scan request'
    fields_desc = [BDAddrField('ScanA', None), BDAddrField('AdvA', None)]

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        return BTLE_SCAN_RSP in other and self.AdvA == other.AdvA

class BTLE_SCAN_RSP(Packet):
    name = 'BTLE scan response'
    fields_desc = [BDAddrField('AdvA', None), PacketListField('data', None, EIR_Hdr)]

    def answers(self, other):
        if False:
            while True:
                i = 10
        return BTLE_SCAN_REQ in other and self.AdvA == other.AdvA

class BTLE_CONNECT_REQ(Packet):
    name = 'BTLE connect request'
    fields_desc = [BDAddrField('InitA', None), BDAddrField('AdvA', None), XIntField('AA', 0), X3BytesField('crc_init', 0), XByteField('win_size', 0), XLEShortField('win_offset', 0), XLEShortField('interval', 0), XLEShortField('latency', 0), XLEShortField('timeout', 0), BTLEChanMapField('chM', 0), BitField('SCA', 0, 3), BitField('hop', 0, 5)]
BTLE_Versions = {6: '4.0', 7: '4.1', 8: '4.2', 9: '5.0', 10: '5.1', 11: '5.2'}
BTLE_Corp_IDs = {15: 'Broadcom Corporation', 89: 'Nordic Semiconductor ASA'}
BTLE_BTLE_CTRL_opcode = {0: 'LL_CONNECTION_UPDATE_REQ', 1: 'LL_CHANNEL_MAP_REQ', 2: 'LL_TERMINATE_IND', 3: 'LL_ENC_REQ', 4: 'LL_ENC_RSP', 5: 'LL_START_ENC_REQ', 6: 'LL_START_ENC_RSP', 7: 'LL_UNKNOWN_RSP', 8: 'LL_FEATURE_REQ', 9: 'LL_FEATURE_RSP', 10: 'LL_PAUSE_ENC_REQ', 11: 'LL_PAUSE_ENC_RSP', 12: 'LL_VERSION_IND', 13: 'LL_REJECT_IND', 14: 'LL_SLAVE_FEATURE_REQ', 15: 'LL_CONNECTION_PARAM_REQ', 16: 'LL_CONNECTION_PARAM_RSP', 20: 'LL_LENGTH_REQ', 21: 'LL_LENGTH_RSP', 22: 'LL_PHY_REQ', 23: 'LL_PHY_RSP', 24: 'LL_PHY_UPDATE_IND', 25: 'LL_MIN_USED_CHANNELS', 26: 'LL_CTE_REQ', 27: 'LL_CTE_RSP', 28: 'LL_PERIODIC_SYNC_IND', 29: 'LL_CLOCK_ACCURACY_REQ', 30: 'LL_CLOCK_ACCURACY_RSP', 31: 'LL_CIS_REQ', 32: 'LL_CIS_RSP', 33: 'LL_CIS_IND', 34: 'LL_CIS_TERMINATE_IND', 35: 'LL_POWER_CONTROL_REQ', 36: 'LL_POWER_CONTROL_RSP', 37: 'LL_POWER_CHANGE_IND', 38: 'LL_SUBRATE_REQ', 39: 'LL_SUBRATE_IND', 40: 'LL_CHANNEL_REPORTING_IND', 41: 'LL_CHANNEL_STATUS_IND'}

class BTLE_EMPTY_PDU(Packet):
    name = 'Empty data PDU'

class BTLE_CTRL(Packet):
    name = 'BTLE_CTRL'
    fields_desc = [ByteEnumField('opcode', 0, BTLE_BTLE_CTRL_opcode)]

class LL_CONNECTION_UPDATE_IND(Packet):
    name = 'LL_CONNECTION_UPDATE_IND'
    fields_desc = [XByteField('win_size', 0), XLEShortField('win_offset', 0), XLEShortField('interval', 6), XLEShortField('latency', 0), XLEShortField('timeout', 50), XLEShortField('instant', 6)]

class LL_CHANNEL_MAP_IND(Packet):
    name = 'LL_CHANNEL_MAP_IND'
    fields_desc = [BTLEChanMapField('chM', 1099511627774), XLEShortField('instant', 0)]

class LL_TERMINATE_IND(Packet):
    name = 'LL_TERMINATE_IND'
    fields_desc = [XByteField('code', 0)]

class LL_ENC_REQ(Packet):
    name = 'LL_ENC_REQ'
    fields_desc = [XLELongField('rand', 0), XLEShortField('ediv', 0), XLELongField('skdm', 0), XLEIntField('ivm', 0)]

class LL_ENC_RSP(Packet):
    name = 'LL_ENC_RSP'
    fields_desc = [XLELongField('skds', 0), XLEIntField('ivs', 0)]

class LL_START_ENC_REQ(Packet):
    name = 'LL_START_ENC_REQ'
    fields_desc = []

class LL_START_ENC_RSP(Packet):
    name = 'LL_START_ENC_RSP'

class LL_UNKNOWN_RSP(Packet):
    name = 'LL_UNKNOWN_RSP'
    fields_desc = [XByteField('code', 0)]

class LL_FEATURE_REQ(Packet):
    name = 'LL_FEATURE_REQ'
    fields_desc = [BTLEFeatureField('feature_set', 0)]

class LL_FEATURE_RSP(Packet):
    name = 'LL_FEATURE_RSP'
    fields_desc = [BTLEFeatureField('feature_set', 0)]

class LL_PAUSE_ENC_REQ(Packet):
    name = 'LL_PAUSE_ENC_REQ'

class LL_PAUSE_ENC_RSP(Packet):
    name = 'LL_PAUSE_ENC_RSP'

class LL_VERSION_IND(Packet):
    name = 'LL_VERSION_IND'
    fields_desc = [ByteEnumField('version', 8, BTLE_Versions), LEShortEnumField('company', 0, BTLE_Corp_IDs), XShortField('subversion', 0)]

class LL_REJECT_IND(Packet):
    name = 'LL_REJECT_IND'
    fields_desc = [XByteField('code', 0)]

class LL_SLAVE_FEATURE_REQ(Packet):
    name = 'LL_SLAVE_FEATURE_REQ'
    fields_desc = [BTLEFeatureField('feature_set', 0)]

class LL_CONNECTION_PARAM_REQ(Packet):
    name = 'LL_CONNECTION_PARAM_REQ'
    fields_desc = [XShortField('interval_min', 6), XShortField('interval_max', 6), XShortField('latency', 0), XShortField('timeout', 0), XByteField('preferred_periodicity', 0), XShortField('reference_conn_evt_count', 0), XShortField('offset0', 0), XShortField('offset1', 0), XShortField('offset2', 0), XShortField('offset3', 0), XShortField('offset4', 0), XShortField('offset5', 0)]

class LL_CONNECTION_PARAM_RSP(Packet):
    name = 'LL_CONNECTION_PARAM_RSP'
    fields_desc = [XShortField('interval_min', 6), XShortField('interval_max', 6), XShortField('latency', 0), XShortField('timeout', 0), XByteField('preferred_periodicity', 0), XShortField('reference_conn_evt_count', 0), XShortField('offset0', 0), XShortField('offset1', 0), XShortField('offset2', 0), XShortField('offset3', 0), XShortField('offset4', 0), XShortField('offset5', 0)]

class LL_REJECT_EXT_IND(Packet):
    name = 'LL_REJECT_EXT_IND'
    fields_desc = [XByteField('reject_opcode', 0), XByteField('error_code', 0)]

class LL_PING_REQ(Packet):
    name = 'LL_PING_REQ'

class LL_PING_RSP(Packet):
    name = 'LL_PING_RSP'

class LL_LENGTH_REQ(Packet):
    name = ' LL_LENGTH_REQ'
    fields_desc = [XLEShortField('max_rx_bytes', 251), XLEShortField('max_rx_time', 2120), XLEShortField('max_tx_bytes', 251), XLEShortField('max_tx_time', 2120)]

class LL_LENGTH_RSP(Packet):
    name = ' LL_LENGTH_RSP'
    fields_desc = [XLEShortField('max_rx_bytes', 251), XLEShortField('max_rx_time', 2120), XLEShortField('max_tx_bytes', 251), XLEShortField('max_tx_time', 2120)]

class LL_PHY_REQ(Packet):
    name = 'LL_PHY_REQ'
    fields_desc = [BTLEPhysField('tx_phys', 0), BTLEPhysField('rx_phys', 0)]

class LL_PHY_RSP(Packet):
    name = 'LL_PHY_RSP'
    fields_desc = [BTLEPhysField('tx_phys', 0), BTLEPhysField('rx_phys', 0)]

class LL_PHY_UPDATE_IND(Packet):
    name = 'LL_PHY_UPDATE_IND'
    fields_desc = [BTLEPhysField('tx_phy', 0), BTLEPhysField('rx_phy', 0), XShortField('instant', 0)]

class LL_MIN_USED_CHANNELS_IND(Packet):
    name = 'LL_MIN_USED_CHANNELS_IND'
    fields_desc = [BTLEPhysField('phys', 0), ByteField('min_used_channels', 2)]

class LL_CTE_REQ(Packet):
    name = 'LL_CTE_REQ'
    fields_desc = [LEBitField('min_cte_len_req', 0, 5), LEBitField('rfu', 0, 1), LEBitField('cte_type_req', 0, 2)]

class LL_CTE_RSP(Packet):
    name = 'LL_CTE_RSP'
    fields_desc = []

class LL_PERIODIC_SYNC_IND(Packet):
    name = 'LL_PERIODIC_SYNC_IND'
    fields_desc = [XLEShortField('id', 251), LEBitField('sync_info', 0, 18 * 8), XLEShortField('conn_event_count', 0), XLEShortField('last_pa_event_counter', 0), LEBitField('sid', 0, 4), LEBitField('a_type', 0, 1), LEBitField('sca', 0, 3), BTLEPhysField('phy', 0), BDAddrField('AdvA', None), XLEShortField('sync_conn_event_count', 0)]

class LL_CLOCK_ACCURACY_REQ(Packet):
    name = 'LL_CLOCK_ACCURACY_REQ'
    fields_desc = [XByteField('sca', 0)]

class LL_CLOCK_ACCURACY_RSP(Packet):
    name = 'LL_CLOCK_ACCURACY_RSP'
    fields_desc = [XByteField('sca', 0)]

class LL_CIS_REQ(Packet):
    name = 'LL_CIS_REQ'
    fields_desc = [XByteField('cig_id', 0), XByteField('cis_id', 0), BTLEPhysField('phy_c_to_p', 0), BTLEPhysField('phy_p_to_c', 0), LEBitField('max_sdu_c_to_p', 0, 12), LEBitField('rfu1', 0, 3), LEBitField('framed', 0, 1), LEBitField('max_sdu_p_to_c', 0, 12), LEBitField('rfu2', 0, 4), LEBitField('sdu_interval_c_to_p', 0, 20), LEBitField('rfu3', 0, 4), LEBitField('sdu_interval_p_to_c', 0, 20), LEBitField('rfu4', 0, 4), XLEShortField('max_pdu_c_to_p', 0), XLEShortField('max_pdu_p_to_c', 0), XByteField('nse', 0), X3BytesField('subinterval', 0), LEBitField('bn_c_to_p', 0, 4), LEBitField('bn_p_to_c', 0, 4), ByteField('ft_c_to_p', 0), ByteField('ft_p_to_c', 0), XLEShortField('iso_interval', 0), X3BytesField('cis_offset_min', 0), X3BytesField('cis_offset_max', 0), XLEShortField('conn_event_count', 0)]

class LL_CIS_RSP(Packet):
    name = 'LL_CIS_RSP'
    fields_desc = [X3BytesField('cis_offset_min', 0), X3BytesField('cis_offset_max', 0), XLEShortField('conn_event_count', 0)]

class LL_CIS_IND(Packet):
    name = 'LL_CIS_IND'
    fields_desc = [XIntField('AA', 0), X3BytesField('cis_offset', 0), X3BytesField('cig_sync_delay', 0), X3BytesField('cis_sync_delay', 0), XLEShortField('conn_event_count', 0)]

class LL_CIS_TERMINATE_IND(Packet):
    name = 'LL_CIS_TERMINATE_IND'
    fields_desc = [ByteField('cig_id', 0), ByteField('cis_id', 0), ByteField('error_code', 0)]

class LL_POWER_CONTROL_REQ(Packet):
    name = 'LL_POWER_CONTROL_REQ'
    fields_desc = [ByteField('phy', 0), SignedByteField('delta', 0), SignedByteField('tx_power', 0)]

class LL_POWER_CONTROL_RSP(Packet):
    name = 'LL_POWER_CONTROL_RSP'
    fields_desc = [LEBitField('min', 0, 1), LEBitField('max', 0, 1), LEBitField('rfu', 0, 6), SignedByteField('delta', 0), SignedByteField('tx_power', 0), ByteField('apr', 0)]

class LL_POWER_CHANGE_IND(Packet):
    name = 'LL_POWER_CHANGE_IND'
    fields_desc = [ByteField('phy', 0), LEBitField('min', 0, 1), LEBitField('max', 0, 1), LEBitField('rfu', 0, 6), SignedByteField('delta', 0), ByteField('tx_power', 0)]

class LL_SUBRATE_REQ(Packet):
    name = 'LL_SUBRATE_REQ'
    fields_desc = [LEShortField('subrate_factor_min', 0), LEShortField('subrate_factor_max', 0), LEShortField('max_latency', 0), LEShortField('continuation_number', 0), LEShortField('timeout', 0)]

class LL_SUBRATE_IND(Packet):
    name = 'LL_SUBRATE_IND'
    fields_desc = [LEShortField('subrate_factor', 0), LEShortField('subrate_base_event', 0), LEShortField('latency', 0), LEShortField('continuation_number', 0), LEShortField('timeout', 0)]

class LL_CHANNEL_REPORTING_IND(Packet):
    name = 'LL_SUBRATE_IND'
    fields_desc = [ByteField('enable', 0), ByteField('min_spacing', 0), ByteField('max_delay', 0)]

class LL_CHANNEL_STATUS_IND(Packet):
    name = 'LL_CHANNEL_STATUS_IND'
    fields_desc = [LEBitField('channel_classification', 0, 10 * 8)]
bind_layers(BTLE, BTLE_ADV, access_addr=2391391958)
bind_layers(BTLE, BTLE_DATA)
bind_layers(BTLE_ADV, BTLE_ADV_IND, PDU_type=0)
bind_layers(BTLE_ADV, BTLE_ADV_DIRECT_IND, PDU_type=1)
bind_layers(BTLE_ADV, BTLE_ADV_NONCONN_IND, PDU_type=2)
bind_layers(BTLE_ADV, BTLE_SCAN_REQ, PDU_type=3)
bind_layers(BTLE_ADV, BTLE_SCAN_RSP, PDU_type=4)
bind_layers(BTLE_ADV, BTLE_CONNECT_REQ, PDU_type=5)
bind_layers(BTLE_ADV, BTLE_ADV_SCAN_IND, PDU_type=6)
bind_layers(BTLE_DATA, L2CAP_Hdr, LLID=2)
bind_layers(BTLE_DATA, BTLE_CTRL, LLID=3)
bind_layers(BTLE_DATA, BTLE_EMPTY_PDU, {'len': 0, 'LLID': 1})
bind_layers(BTLE_CTRL, LL_CONNECTION_UPDATE_IND, opcode=0)
bind_layers(BTLE_CTRL, LL_CHANNEL_MAP_IND, opcode=1)
bind_layers(BTLE_CTRL, LL_TERMINATE_IND, opcode=2)
bind_layers(BTLE_CTRL, LL_ENC_REQ, opcode=3)
bind_layers(BTLE_CTRL, LL_ENC_RSP, opcode=4)
bind_layers(BTLE_CTRL, LL_START_ENC_REQ, opcode=5)
bind_layers(BTLE_CTRL, LL_START_ENC_RSP, opcode=6)
bind_layers(BTLE_CTRL, LL_UNKNOWN_RSP, opcode=7)
bind_layers(BTLE_CTRL, LL_FEATURE_REQ, opcode=8)
bind_layers(BTLE_CTRL, LL_FEATURE_RSP, opcode=9)
bind_layers(BTLE_CTRL, LL_PAUSE_ENC_REQ, opcode=10)
bind_layers(BTLE_CTRL, LL_PAUSE_ENC_RSP, opcode=11)
bind_layers(BTLE_CTRL, LL_VERSION_IND, opcode=12)
bind_layers(BTLE_CTRL, LL_REJECT_IND, opcode=13)
bind_layers(BTLE_CTRL, LL_SLAVE_FEATURE_REQ, opcode=14)
bind_layers(BTLE_CTRL, LL_CONNECTION_PARAM_REQ, opcode=15)
bind_layers(BTLE_CTRL, LL_CONNECTION_PARAM_RSP, opcode=16)
bind_layers(BTLE_CTRL, LL_REJECT_EXT_IND, opcode=17)
bind_layers(BTLE_CTRL, LL_PING_REQ, opcode=18)
bind_layers(BTLE_CTRL, LL_PING_RSP, opcode=19)
bind_layers(BTLE_CTRL, LL_LENGTH_REQ, opcode=20)
bind_layers(BTLE_CTRL, LL_LENGTH_RSP, opcode=21)
bind_layers(BTLE_CTRL, LL_PHY_REQ, opcode=22)
bind_layers(BTLE_CTRL, LL_PHY_RSP, opcode=23)
bind_layers(BTLE_CTRL, LL_PHY_UPDATE_IND, opcode=24)
bind_layers(BTLE_CTRL, LL_MIN_USED_CHANNELS_IND, opcode=25)
bind_layers(BTLE_CTRL, LL_CTE_REQ, opcode=26)
bind_layers(BTLE_CTRL, LL_CTE_RSP, opcode=27)
bind_layers(BTLE_CTRL, LL_PERIODIC_SYNC_IND, opcode=28)
bind_layers(BTLE_CTRL, LL_CLOCK_ACCURACY_REQ, opcode=29)
bind_layers(BTLE_CTRL, LL_CLOCK_ACCURACY_RSP, opcode=30)
bind_layers(BTLE_CTRL, LL_CIS_REQ, opcode=31)
bind_layers(BTLE_CTRL, LL_CIS_RSP, opcode=32)
bind_layers(BTLE_CTRL, LL_CIS_IND, opcode=33)
bind_layers(BTLE_CTRL, LL_CIS_TERMINATE_IND, opcode=34)
bind_layers(BTLE_CTRL, LL_POWER_CONTROL_REQ, opcode=35)
bind_layers(BTLE_CTRL, LL_POWER_CONTROL_RSP, opcode=36)
bind_layers(BTLE_CTRL, LL_POWER_CHANGE_IND, opcode=37)
bind_layers(BTLE_CTRL, LL_SUBRATE_REQ, opcode=38)
bind_layers(BTLE_CTRL, LL_SUBRATE_IND, opcode=39)
bind_layers(BTLE_CTRL, LL_CHANNEL_REPORTING_IND, opcode=40)
bind_layers(BTLE_CTRL, LL_CHANNEL_STATUS_IND, opcode=41)
conf.l2types.register(DLT_BLUETOOTH_LE_LL, BTLE)
conf.l2types.register(DLT_BLUETOOTH_LE_LL_WITH_PHDR, BTLE_RF)
bind_layers(BTLE_RF, BTLE)
bind_layers(PPI_Hdr, BTLE_PPI, pfh_type=PPI_BTLE)