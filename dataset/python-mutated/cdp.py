"""
Cisco Discovery Protocol (CDP) extension for Scapy
"""
import struct
from scapy.packet import Packet, bind_layers
from scapy.fields import ByteEnumField, ByteField, FieldLenField, FlagsField, IP6Field, IPField, OUIField, PacketListField, ShortField, StrLenField, XByteField, XShortEnumField, XShortField
from scapy.layers.inet import checksum
from scapy.layers.l2 import SNAP
from scapy.compat import orb, chb
from scapy.config import conf
_cdp_tlv_cls = {1: 'CDPMsgDeviceID', 2: 'CDPMsgAddr', 3: 'CDPMsgPortID', 4: 'CDPMsgCapabilities', 5: 'CDPMsgSoftwareVersion', 6: 'CDPMsgPlatform', 8: 'CDPMsgProtoHello', 9: 'CDPMsgVTPMgmtDomain', 10: 'CDPMsgNativeVLAN', 11: 'CDPMsgDuplex', 14: 'CDPMsgVoIPVLANReply', 15: 'CDPMsgVoIPVLANQuery', 16: 'CDPMsgPower', 17: 'CDPMsgMTU', 18: 'CDPMsgTrustBitmap', 19: 'CDPMsgUntrustedPortCoS', 22: 'CDPMsgMgmtAddr', 25: 'CDPMsgUnknown19'}
_cdp_tlv_types = {1: 'Device ID', 2: 'Addresses', 3: 'Port ID', 4: 'Capabilities', 5: 'Software Version', 6: 'Platform', 7: 'IP Prefix', 8: 'Protocol Hello', 9: 'VTP Management Domain', 10: 'Native VLAN', 11: 'Duplex', 12: 'CDP Unknown command (send us a pcap file)', 13: 'CDP Unknown command (send us a pcap file)', 14: 'VoIP VLAN Reply', 15: 'VoIP VLAN Query', 16: 'Power', 17: 'MTU', 18: 'Trust Bitmap', 19: 'Untrusted Port CoS', 20: 'System Name', 21: 'System OID', 22: 'Management Address', 23: 'Location', 24: 'CDP Unknown command (send us a pcap file)', 25: 'CDP Unknown command (send us a pcap file)', 26: 'Power Available'}

def _CDPGuessPayloadClass(p, **kargs):
    if False:
        return 10
    cls = conf.raw_layer
    if len(p) >= 2:
        t = struct.unpack('!H', p[:2])[0]
        if t == 7 and len(p) > 4:
            tmp_len = struct.unpack('!H', p[2:4])[0]
            if tmp_len == 8:
                clsname = 'CDPMsgIPGateway'
            else:
                clsname = 'CDPMsgIPPrefix'
        else:
            clsname = _cdp_tlv_cls.get(t, 'CDPMsgGeneric')
        cls = globals()[clsname]
    return cls(p, **kargs)

class CDPMsgGeneric(Packet):
    name = 'CDP Generic Message'
    fields_desc = [XShortEnumField('type', None, _cdp_tlv_types), FieldLenField('len', None, 'val', '!H', adjust=lambda pkt, x: x + 4), StrLenField('val', '', length_from=lambda x: x.len - 4, max_length=65531)]

    def guess_payload_class(self, p):
        if False:
            return 10
        return conf.padding_layer

class CDPMsgDeviceID(CDPMsgGeneric):
    name = 'Device ID'
    type = 1
_cdp_addr_record_ptype = {1: 'NLPID', 2: '802.2'}
_cdp_addrrecord_proto_ip = b'\xcc'
_cdp_addrrecord_proto_ipv6 = b'\xaa\xaa\x03\x00\x00\x00\x86\xdd'

class CDPAddrRecord(Packet):
    name = 'CDP Address'
    fields_desc = [ByteEnumField('ptype', 1, _cdp_addr_record_ptype), FieldLenField('plen', None, 'proto', 'B'), StrLenField('proto', None, length_from=lambda x: x.plen, max_length=255), FieldLenField('addrlen', None, length_of=lambda x: x.addr), StrLenField('addr', None, length_from=lambda x: x.addrlen, max_length=65535)]

    def guess_payload_class(self, p):
        if False:
            print('Hello World!')
        return conf.padding_layer

class CDPAddrRecordIPv4(CDPAddrRecord):
    name = 'CDP Address IPv4'
    fields_desc = [ByteEnumField('ptype', 1, _cdp_addr_record_ptype), FieldLenField('plen', 1, 'proto', 'B'), StrLenField('proto', _cdp_addrrecord_proto_ip, length_from=lambda x: x.plen, max_length=255), ShortField('addrlen', 4), IPField('addr', '0.0.0.0')]

class CDPAddrRecordIPv6(CDPAddrRecord):
    name = 'CDP Address IPv6'
    fields_desc = [ByteEnumField('ptype', 2, _cdp_addr_record_ptype), FieldLenField('plen', 8, 'proto', 'B'), StrLenField('proto', _cdp_addrrecord_proto_ipv6, length_from=lambda x: x.plen, max_length=255), ShortField('addrlen', 16), IP6Field('addr', '::1')]

def _CDPGuessAddrRecord(p, **kargs):
    if False:
        print('Hello World!')
    cls = conf.raw_layer
    if len(p) >= 2:
        plen = orb(p[1])
        proto = p[2:plen + 2]
        if proto == _cdp_addrrecord_proto_ip:
            clsname = 'CDPAddrRecordIPv4'
        elif proto == _cdp_addrrecord_proto_ipv6:
            clsname = 'CDPAddrRecordIPv6'
        else:
            clsname = 'CDPAddrRecord'
        cls = globals()[clsname]
    return cls(p, **kargs)

class CDPMsgAddr(CDPMsgGeneric):
    name = 'Addresses'
    fields_desc = [XShortEnumField('type', 2, _cdp_tlv_types), ShortField('len', None), FieldLenField('naddr', None, fmt='!I', count_of='addr'), PacketListField('addr', [], _CDPGuessAddrRecord, length_from=lambda x: x.len - 8)]

    def post_build(self, pkt, pay):
        if False:
            for i in range(10):
                print('nop')
        if self.len is None:
            pkt = pkt[:2] + struct.pack('!H', len(pkt)) + pkt[4:]
        p = pkt + pay
        return p

class CDPMsgPortID(CDPMsgGeneric):
    name = 'Port ID'
    fields_desc = [XShortEnumField('type', 3, _cdp_tlv_types), FieldLenField('len', None, 'iface', '!H', adjust=lambda pkt, x: x + 4), StrLenField('iface', 'Port 1', length_from=lambda x: x.len - 4)]
_cdp_capabilities = ['Router', 'TransparentBridge', 'SourceRouteBridge', 'Switch', 'Host', 'IGMPCapable', 'Repeater'] + ['Bit%d' % x for x in range(25, 0, -1)]

class CDPMsgCapabilities(CDPMsgGeneric):
    name = 'Capabilities'
    fields_desc = [XShortEnumField('type', 4, _cdp_tlv_types), ShortField('len', 8), FlagsField('cap', 0, 32, _cdp_capabilities)]

class CDPMsgSoftwareVersion(CDPMsgGeneric):
    name = 'Software Version'
    type = 5

class CDPMsgPlatform(CDPMsgGeneric):
    name = 'Platform'
    type = 6
_cdp_duplex = {0: 'Half', 1: 'Full'}

class CDPMsgIPGateway(CDPMsgGeneric):
    name = 'IP Gateway'
    type = 7
    fields_desc = [XShortEnumField('type', 7, _cdp_tlv_types), ShortField('len', 8), IPField('defaultgw', '192.168.0.1')]

class CDPMsgIPPrefix(CDPMsgGeneric):
    name = 'IP Prefix'
    type = 7
    fields_desc = [XShortEnumField('type', 7, _cdp_tlv_types), ShortField('len', 9), IPField('prefix', '192.168.0.1'), ByteField('plen', 24)]

class CDPMsgProtoHello(CDPMsgGeneric):
    name = 'Protocol Hello'
    type = 8
    fields_desc = [XShortEnumField('type', 8, _cdp_tlv_types), ShortField('len', 32), OUIField('oui', 12), XShortField('protocol_id', 0), StrLenField('data', '', length_from=lambda p: p.len - 9)]

class CDPMsgVTPMgmtDomain(CDPMsgGeneric):
    name = 'VTP Management Domain'
    type = 9

class CDPMsgNativeVLAN(CDPMsgGeneric):
    name = 'Native VLAN'
    fields_desc = [XShortEnumField('type', 10, _cdp_tlv_types), ShortField('len', 6), ShortField('vlan', 1)]

class CDPMsgDuplex(CDPMsgGeneric):
    name = 'Duplex'
    fields_desc = [XShortEnumField('type', 11, _cdp_tlv_types), ShortField('len', 5), ByteEnumField('duplex', 0, _cdp_duplex)]

class CDPMsgVoIPVLANReply(CDPMsgGeneric):
    name = 'VoIP VLAN Reply'
    fields_desc = [XShortEnumField('type', 14, _cdp_tlv_types), ShortField('len', 7), ByteField('status', 1), ShortField('vlan', 1)]

class CDPMsgVoIPVLANQuery(CDPMsgGeneric):
    name = 'VoIP VLAN Query'
    type = 15
    fields_desc = [XShortEnumField('type', 15, _cdp_tlv_types), FieldLenField('len', None, 'unknown2', fmt='!H', adjust=lambda pkt, x: x + 7), XByteField('unknown1', 0), ShortField('vlan', 1), StrLenField('unknown2', '', length_from=lambda p: p.len - 7, max_length=65528)]

class _CDPPowerField(ShortField):

    def i2repr(self, pkt, x):
        if False:
            print('Hello World!')
        if x is None:
            x = 0
        return '%d mW' % x

class CDPMsgPower(CDPMsgGeneric):
    name = 'Power'
    fields_desc = [XShortEnumField('type', 16, _cdp_tlv_types), ShortField('len', 6), _CDPPowerField('power', 1337)]

class CDPMsgMTU(CDPMsgGeneric):
    name = 'MTU'
    fields_desc = [XShortEnumField('type', 17, _cdp_tlv_types), ShortField('len', 6), ShortField('mtu', 1500)]

class CDPMsgTrustBitmap(CDPMsgGeneric):
    name = 'Trust Bitmap'
    fields_desc = [XShortEnumField('type', 18, _cdp_tlv_types), ShortField('len', 5), XByteField('trust_bitmap', 0)]

class CDPMsgUntrustedPortCoS(CDPMsgGeneric):
    name = 'Untrusted Port CoS'
    fields_desc = [XShortEnumField('type', 19, _cdp_tlv_types), ShortField('len', 5), XByteField('untrusted_port_cos', 0)]

class CDPMsgMgmtAddr(CDPMsgAddr):
    name = 'Management Address'
    type = 22

class CDPMsgUnknown19(CDPMsgGeneric):
    name = 'Unknown CDP Message'
    type = 25

class CDPMsg(CDPMsgGeneric):
    name = 'CDP '
    fields_desc = [XShortEnumField('type', None, _cdp_tlv_types), FieldLenField('len', None, 'val', fmt='!H', adjust=lambda pkt, x: x + 4), StrLenField('val', '', length_from=lambda x: x.len - 4, max_length=65531)]

class _CDPChecksum:

    def _check_len(self, pkt):
        if False:
            return 10
        'Check for odd packet length and pad according to Cisco spec.\n        This padding is only used for checksum computation.  The original\n        packet should not be altered.'
        if len(pkt) % 2:
            last_chr = orb(pkt[-1])
            if last_chr <= 128:
                return pkt[:-1] + b'\x00' + chb(last_chr)
            else:
                return pkt[:-1] + b'\xff' + chb(orb(last_chr) - 1)
        else:
            return pkt

    def post_build(self, pkt, pay):
        if False:
            while True:
                i = 10
        p = pkt + pay
        if self.cksum is None:
            cksum = checksum(self._check_len(p))
            p = p[:2] + struct.pack('!H', cksum) + p[4:]
        return p

class CDPv2_HDR(_CDPChecksum, CDPMsgGeneric):
    name = 'Cisco Discovery Protocol version 2'
    fields_desc = [ByteField('vers', 2), ByteField('ttl', 180), XShortField('cksum', None), PacketListField('msg', [], _CDPGuessPayloadClass)]
bind_layers(SNAP, CDPv2_HDR, {'code': 8192, 'OUI': 12})