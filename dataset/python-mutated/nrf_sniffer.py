"""
nRF sniffer

Firmware and documentation related to this module is available at:
https://www.nordicsemi.com/Software-and-Tools/Development-Tools/nRF-Sniffer
https://github.com/adafruit/Adafruit_BLESniffer_Python
https://github.com/wireshark/wireshark/blob/master/epan/dissectors/packet-nordic_ble.c
"""
import struct
from scapy.config import conf
from scapy.data import DLT_NORDIC_BLE
from scapy.fields import BitEnumField, BitField, ByteEnumField, ByteField, LEIntField, LEShortField, LenField, ScalingField
from scapy.layers.bluetooth4LE import BTLE
from scapy.packet import Packet, bind_layers

class NRFS2_Packet(Packet):
    """
    nRF Sniffer v2 Packet
    """
    fields_desc = [LenField('len', None, fmt='<H', adjust=lambda x: x + 6), ByteField('version', 2), LEShortField('counter', None), ByteEnumField('type', None, {0: 'req_follow', 1: 'event_follow', 2: 'event_device', 3: 'req_single_packet', 4: 'resp_single_packet', 5: 'event_connect', 6: 'event_packet', 7: 'req_scan_cont', 9: 'event_disconnect', 10: 'event_error', 11: 'event_empty_data_packet', 12: 'set_temporary_key', 13: 'ping_req', 14: 'ping_resp', 15: 'test_command_id', 16: 'test_result_id', 17: 'uart_test_start', 18: 'uart_dummy_packet', 19: 'switch_baud_rate_req', 20: 'switch_baud_rate_resp', 21: 'uart_out_start', 22: 'uart_out_stop', 23: 'set_adv_channel_hop_seq', 254: 'go_idle'})]

    def answer(self, other):
        if False:
            return 10
        if not isinstance(other, NRFS2_Packet):
            return False
        return self.type == 1 and other.type == 0 or (self.type == 14 and other.type == 13) or (self.type == 20 and other.type == 19)

    def post_build(self, p, pay):
        if False:
            for i in range(10):
                print('nop')
        if self.hdr_len is None:
            p = p[:1] + struct.pack('!B', len(p)) + p[2:]
        return p + pay

class NRF2_Ping_Request(Packet):
    name = 'Ping request'

class NRF2_Ping_Response(Packet):
    name = 'Ping response'
    fields_desc = [LEShortField('version', None)]

class NRF2_Packet_Event(Packet):
    name = 'Packet event (device variant)'
    fields_desc = [ByteField('header_len', 10), BitField('reserved', 0, 1), BitEnumField('phy', None, 3, {0: 'le-1m', 1: 'le-2m', 2: 'le-coded'}), BitField('mic', None, 1), BitField('encrypted', None, 1), BitField('direction', None, 1), BitField('crc_ok', 1, 1), ByteField('rf_channel', 0), ScalingField('rssi', -256, unit='dBm', fmt='b'), LEShortField('event_counter', 0), LEIntField('delta_time', 0)]
bind_layers(NRFS2_Packet, NRF2_Ping_Request, type=13)
bind_layers(NRFS2_Packet, NRF2_Ping_Response, type=14)
bind_layers(NRFS2_Packet, NRF2_Packet_Event, type=6)
bind_layers(NRF2_Packet_Event, BTLE)

class NRFS2_PCAP(Packet):
    """
    PCAP headers for DLT_NORDIC_BLE.

    Nordic's capture scripts either stick the COM port number (yep!) or a
    random number at the start of every packet.

    https://github.com/wireshark/wireshark/blob/master/epan/dissectors/packet-nordic_ble.c

    The only "rule" is that we can't start packets with ``BE EF``, otherwise
    it becomes a "0.9.7" packet. So we just set "0" here.
    """
    name = 'nRF Sniffer PCAP header'
    fields_desc = [ByteField('board_id', 0)]
bind_layers(NRFS2_PCAP, NRFS2_Packet)
conf.l2types.register(DLT_NORDIC_BLE, NRFS2_PCAP)