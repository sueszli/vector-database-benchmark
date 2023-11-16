"""
RoCE: RDMA over Converged Ethernet
"""
from scapy.packet import Packet, bind_layers, Raw
from scapy.fields import ByteEnumField, ByteField, XByteField, ShortField, XShortField, XLongField, BitField, XBitField, FCSField
from scapy.layers.inet import IP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import Ether
from scapy.compat import raw
from scapy.error import warning
from zlib import crc32
import struct
from typing import Tuple
_transports = {'RC': 0, 'UC': 32, 'RD': 64, 'UD': 96}
_ops = {'SEND_FIRST': 0, 'SEND_MIDDLE': 1, 'SEND_LAST': 2, 'SEND_LAST_WITH_IMMEDIATE': 3, 'SEND_ONLY': 4, 'SEND_ONLY_WITH_IMMEDIATE': 5, 'RDMA_WRITE_FIRST': 6, 'RDMA_WRITE_MIDDLE': 7, 'RDMA_WRITE_LAST': 8, 'RDMA_WRITE_LAST_WITH_IMMEDIATE': 9, 'RDMA_WRITE_ONLY': 10, 'RDMA_WRITE_ONLY_WITH_IMMEDIATE': 11, 'RDMA_READ_REQUEST': 12, 'RDMA_READ_RESPONSE_FIRST': 13, 'RDMA_READ_RESPONSE_MIDDLE': 14, 'RDMA_READ_RESPONSE_LAST': 15, 'RDMA_READ_RESPONSE_ONLY': 16, 'ACKNOWLEDGE': 17, 'ATOMIC_ACKNOWLEDGE': 18, 'COMPARE_SWAP': 19, 'FETCH_ADD': 20}
CNP_OPCODE = 129

def opcode(transport, op):
    if False:
        i = 10
        return i + 15
    return (_transports[transport] + _ops[op], '{}_{}'.format(transport, op))
_bth_opcodes = dict([opcode('RC', 'SEND_FIRST'), opcode('RC', 'SEND_MIDDLE'), opcode('RC', 'SEND_LAST'), opcode('RC', 'SEND_LAST_WITH_IMMEDIATE'), opcode('RC', 'SEND_ONLY'), opcode('RC', 'SEND_ONLY_WITH_IMMEDIATE'), opcode('RC', 'RDMA_WRITE_FIRST'), opcode('RC', 'RDMA_WRITE_MIDDLE'), opcode('RC', 'RDMA_WRITE_LAST'), opcode('RC', 'RDMA_WRITE_LAST_WITH_IMMEDIATE'), opcode('RC', 'RDMA_WRITE_ONLY'), opcode('RC', 'RDMA_WRITE_ONLY_WITH_IMMEDIATE'), opcode('RC', 'RDMA_READ_REQUEST'), opcode('RC', 'RDMA_READ_RESPONSE_FIRST'), opcode('RC', 'RDMA_READ_RESPONSE_MIDDLE'), opcode('RC', 'RDMA_READ_RESPONSE_LAST'), opcode('RC', 'RDMA_READ_RESPONSE_ONLY'), opcode('RC', 'ACKNOWLEDGE'), opcode('RC', 'ATOMIC_ACKNOWLEDGE'), opcode('RC', 'COMPARE_SWAP'), opcode('RC', 'FETCH_ADD'), opcode('UC', 'SEND_FIRST'), opcode('UC', 'SEND_MIDDLE'), opcode('UC', 'SEND_LAST'), opcode('UC', 'SEND_LAST_WITH_IMMEDIATE'), opcode('UC', 'SEND_ONLY'), opcode('UC', 'SEND_ONLY_WITH_IMMEDIATE'), opcode('UC', 'RDMA_WRITE_FIRST'), opcode('UC', 'RDMA_WRITE_MIDDLE'), opcode('UC', 'RDMA_WRITE_LAST'), opcode('UC', 'RDMA_WRITE_LAST_WITH_IMMEDIATE'), opcode('UC', 'RDMA_WRITE_ONLY'), opcode('UC', 'RDMA_WRITE_ONLY_WITH_IMMEDIATE'), opcode('RD', 'SEND_FIRST'), opcode('RD', 'SEND_MIDDLE'), opcode('RD', 'SEND_LAST'), opcode('RD', 'SEND_LAST_WITH_IMMEDIATE'), opcode('RD', 'SEND_ONLY'), opcode('RD', 'SEND_ONLY_WITH_IMMEDIATE'), opcode('RD', 'RDMA_WRITE_FIRST'), opcode('RD', 'RDMA_WRITE_MIDDLE'), opcode('RD', 'RDMA_WRITE_LAST'), opcode('RD', 'RDMA_WRITE_LAST_WITH_IMMEDIATE'), opcode('RD', 'RDMA_WRITE_ONLY'), opcode('RD', 'RDMA_WRITE_ONLY_WITH_IMMEDIATE'), opcode('RD', 'RDMA_READ_REQUEST'), opcode('RD', 'RDMA_READ_RESPONSE_FIRST'), opcode('RD', 'RDMA_READ_RESPONSE_MIDDLE'), opcode('RD', 'RDMA_READ_RESPONSE_LAST'), opcode('RD', 'RDMA_READ_RESPONSE_ONLY'), opcode('RD', 'ACKNOWLEDGE'), opcode('RD', 'ATOMIC_ACKNOWLEDGE'), opcode('RD', 'COMPARE_SWAP'), opcode('RD', 'FETCH_ADD'), opcode('UD', 'SEND_ONLY'), opcode('UD', 'SEND_ONLY_WITH_IMMEDIATE'), (CNP_OPCODE, 'CNP')])

class BTH(Packet):
    name = 'BTH'
    fields_desc = [ByteEnumField('opcode', 0, _bth_opcodes), BitField('solicited', 0, 1), BitField('migreq', 0, 1), BitField('padcount', 0, 2), BitField('version', 0, 4), XShortField('pkey', 65535), BitField('fecn', 0, 1), BitField('becn', 0, 1), BitField('resv6', 0, 6), BitField('dqpn', 0, 24), BitField('ackreq', 0, 1), BitField('resv7', 0, 7), BitField('psn', 0, 24), FCSField('icrc', None, fmt='!I')]

    @staticmethod
    def pack_icrc(icrc):
        if False:
            while True:
                i = 10
        return struct.pack('!I', icrc & 4294967295)[::-1]

    def compute_icrc(self, p):
        if False:
            while True:
                i = 10
        udp = self.underlayer
        if udp is None or not isinstance(udp, UDP):
            warning('Expecting UDP underlayer to compute checksum. Got %s.', udp and udp.name)
            return self.pack_icrc(0)
        ip = udp.underlayer
        if isinstance(ip, IP):
            pshdr = Raw(b'\xff' * 8) / ip.copy()
            pshdr.chksum = 65535
            pshdr.ttl = 255
            pshdr.tos = 255
            pshdr[UDP].chksum = 65535
            pshdr[BTH].fecn = 1
            pshdr[BTH].becn = 1
            pshdr[BTH].resv6 = 255
            bth = pshdr[BTH].self_build()
            payload = raw(pshdr[BTH].payload)
            icrc_placeholder = b'\xff\xff\xff\xff'
            pshdr[UDP].payload = Raw(bth + payload + icrc_placeholder)
            icrc = crc32(raw(pshdr)[:-4]) & 4294967295
            return self.pack_icrc(icrc)
        elif isinstance(ip, IPv6):
            pshdr = Raw(b'\xff' * 8) / ip.copy()
            pshdr.hlim = 255
            pshdr.fl = 1048575
            pshdr.tc = 255
            pshdr[UDP].chksum = 65535
            pshdr[BTH].fecn = 1
            pshdr[BTH].becn = 1
            pshdr[BTH].resv6 = 255
            bth = pshdr[BTH].self_build()
            payload = raw(pshdr[BTH].payload)
            icrc_placeholder = b'\xff\xff\xff\xff'
            pshdr[UDP].payload = Raw(bth + payload + icrc_placeholder)
            icrc = crc32(raw(pshdr)[:-4]) & 4294967295
            return self.pack_icrc(icrc)
        else:
            warning('The underlayer protocol %s is not supported.', ip and ip.name)
            return self.pack_icrc(0)

    def post_build(self, p, pay):
        if False:
            return 10
        p += pay
        if self.icrc is None:
            p = p[:-4] + self.compute_icrc(p)
        return p

class CNPPadding(Packet):
    name = 'CNPPadding'
    fields_desc = [XLongField('reserved1', 0), XLongField('reserved2', 0)]

def cnp(dqpn):
    if False:
        return 10
    return BTH(opcode=CNP_OPCODE, becn=1, dqpn=dqpn) / CNPPadding()

class GRH(Packet):
    name = 'GRH'
    fields_desc = [BitField('ipver', 6, 4), BitField('tclass', 0, 8), BitField('flowlabel', 6, 20), ShortField('paylen', 0), ByteField('nexthdr', 0), ByteField('hoplmt', 0), XBitField('sgid', 0, 128), XBitField('dgid', 0, 128)]

class AETH(Packet):
    name = 'AETH'
    fields_desc = [XByteField('syndrome', 0), XBitField('msn', 0, 24)]
bind_layers(BTH, CNPPadding, opcode=CNP_OPCODE)
bind_layers(Ether, GRH, type=35093)
bind_layers(GRH, BTH)
bind_layers(BTH, AETH, opcode=opcode('RC', 'ACKNOWLEDGE')[0])
bind_layers(BTH, AETH, opcode=opcode('RD', 'ACKNOWLEDGE')[0])
bind_layers(UDP, BTH, dport=4791)