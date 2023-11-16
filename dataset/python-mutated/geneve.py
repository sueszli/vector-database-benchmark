"""
Geneve: Generic Network Virtualization Encapsulation

draft-ietf-nvo3-geneve-16
"""
import struct
from scapy.fields import BitField, XByteField, XShortEnumField, X3BytesField, StrLenField, PacketListField
from scapy.packet import Packet, bind_layers
from scapy.layers.inet import IP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import Ether, ETHER_TYPES
from scapy.compat import chb, orb
CLASS_IDS = {256: 'Linux', 257: 'Open vSwitch', 258: 'Open Virtual Networking (OVN)', 259: 'In-band Network Telemetry (INT)', 260: 'VMware', 261: 'Amazon.com, Inc.', 262: 'Cisco Systems, Inc.', 263: 'Oracle Corporation', 272: 'Amazon.com, Inc.', 280: 'IBM', 296: 'Ericsson', 65279: 'Unassigned', 65535: 'Experimental'}

class GeneveOptions(Packet):
    name = 'Geneve Options'
    fields_desc = [XShortEnumField('classid', 0, CLASS_IDS), XByteField('type', 0), BitField('reserved', 0, 3), BitField('length', None, 5), StrLenField('data', '', length_from=lambda x: x.length * 4)]

    def post_build(self, p, pay):
        if False:
            return 10
        if self.length is None:
            tmp_len = len(self.data) // 4
            p = p[:3] + struct.pack('!B', tmp_len) + p[4:]
        return p + pay

class GENEVE(Packet):
    name = 'GENEVE'
    fields_desc = [BitField('version', 0, 2), BitField('optionlen', None, 6), BitField('oam', 0, 1), BitField('critical', 0, 1), BitField('reserved', 0, 6), XShortEnumField('proto', 0, ETHER_TYPES), X3BytesField('vni', 0), XByteField('reserved2', 0), PacketListField('options', [], GeneveOptions, length_from=lambda pkt: pkt.optionlen * 4)]

    def post_build(self, p, pay):
        if False:
            while True:
                i = 10
        if self.optionlen is None:
            tmp_len = (len(p) - 8) // 4
            p = chb(tmp_len & 47 | orb(p[0]) & 192) + p[1:]
        return p + pay

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, GENEVE):
            if self.proto == other.proto and self.vni == other.vni:
                return self.payload.answers(other.payload)
        else:
            return self.payload.answers(other)
        return 0

    def mysummary(self):
        if False:
            for i in range(10):
                print('nop')
        return self.sprintf('GENEVE (vni=%GENEVE.vni%,optionlen=%GENEVE.optionlen%,proto=%GENEVE.proto%)')
bind_layers(UDP, GENEVE, dport=6081)
bind_layers(GENEVE, Ether, proto=25944)
bind_layers(GENEVE, IP, proto=2048)
bind_layers(GENEVE, IPv6, proto=34525)