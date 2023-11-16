"""
IPv4 (Internet Protocol v4).
"""
import time
import struct
import re
import random
import select
import socket
from collections import defaultdict
from scapy.utils import checksum, do_graph, incremental_label, linehexdump, strxor, whois, colgen
from scapy.ansmachine import AnsweringMachine
from scapy.base_classes import Gen, Net
from scapy.data import ETH_P_IP, ETH_P_ALL, DLT_RAW, DLT_RAW_ALT, DLT_IPV4, IP_PROTOS, TCP_SERVICES, UDP_SERVICES
from scapy.layers.l2 import CookedLinux, Dot3, Ether, GRE, Loopback, SNAP, arpcachepoison, getmacbyip
from scapy.compat import raw, chb, orb, bytes_encode, Optional
from scapy.config import conf
from scapy.fields import BitEnumField, BitField, ByteEnumField, ByteField, ConditionalField, DestField, Emph, FieldLenField, FieldListField, FlagsField, IPField, IntField, MultiEnumField, MultipleTypeField, PacketListField, ShortEnumField, ShortField, SourceIPField, StrField, StrFixedLenField, StrLenField, XByteField, XShortField
from scapy.packet import Packet, bind_layers, bind_bottom_up, NoPayload
from scapy.volatile import RandShort, RandInt, RandBin, RandNum, VolatileValue
from scapy.sendrecv import sr, sr1
from scapy.plist import _PacketList, PacketList, SndRcvList
from scapy.automaton import Automaton, ATMT
from scapy.error import log_runtime, warning
from scapy.pton_ntop import inet_pton
import scapy.as_resolvers

class IPTools(object):
    """Add more powers to a class with an "src" attribute."""
    __slots__ = []

    def whois(self):
        if False:
            print('Hello World!')
        'whois the source and print the output'
        print(whois(self.src).decode('utf8', 'ignore'))

    def _ttl(self):
        if False:
            i = 10
            return i + 15
        'Returns ttl or hlim, depending on the IP version'
        return self.hlim if isinstance(self, scapy.layers.inet6.IPv6) else self.ttl

    def ottl(self):
        if False:
            i = 10
            return i + 15
        t = sorted([32, 64, 128, 255] + [self._ttl()])
        return t[t.index(self._ttl()) + 1]

    def hops(self):
        if False:
            i = 10
            return i + 15
        return self.ottl() - self._ttl()
_ip_options_names = {0: 'end_of_list', 1: 'nop', 2: 'security', 3: 'loose_source_route', 4: 'timestamp', 5: 'extended_security', 6: 'commercial_security', 7: 'record_route', 8: 'stream_id', 9: 'strict_source_route', 10: 'experimental_measurement', 11: 'mtu_probe', 12: 'mtu_reply', 13: 'flow_control', 14: 'access_control', 15: 'encode', 16: 'imi_traffic_descriptor', 17: 'extended_IP', 18: 'traceroute', 19: 'address_extension', 20: 'router_alert', 21: 'selective_directed_broadcast_mode', 23: 'dynamic_packet_state', 24: 'upstream_multicast_packet', 25: 'quick_start', 30: 'rfc4727_experiment'}

class _IPOption_HDR(Packet):
    fields_desc = [BitField('copy_flag', 0, 1), BitEnumField('optclass', 0, 2, {0: 'control', 2: 'debug'}), BitEnumField('option', 0, 5, _ip_options_names)]

class IPOption(Packet):
    name = 'IP Option'
    fields_desc = [_IPOption_HDR, FieldLenField('length', None, fmt='B', length_of='value', adjust=lambda pkt, l: l + 2), StrLenField('value', '', length_from=lambda pkt: pkt.length - 2)]

    def extract_padding(self, p):
        if False:
            print('Hello World!')
        return (b'', p)
    registered_ip_options = {}

    @classmethod
    def register_variant(cls):
        if False:
            i = 10
            return i + 15
        cls.registered_ip_options[cls.option.default] = cls

    @classmethod
    def dispatch_hook(cls, pkt=None, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        if pkt:
            opt = orb(pkt[0]) & 31
            if opt in cls.registered_ip_options:
                return cls.registered_ip_options[opt]
        return cls

class IPOption_EOL(IPOption):
    name = 'IP Option End of Options List'
    option = 0
    fields_desc = [_IPOption_HDR]

class IPOption_NOP(IPOption):
    name = 'IP Option No Operation'
    option = 1
    fields_desc = [_IPOption_HDR]

class IPOption_Security(IPOption):
    name = 'IP Option Security'
    copy_flag = 1
    option = 2
    fields_desc = [_IPOption_HDR, ByteField('length', 11), ShortField('security', 0), ShortField('compartment', 0), ShortField('handling_restrictions', 0), StrFixedLenField('transmission_control_code', 'xxx', 3)]

class IPOption_RR(IPOption):
    name = 'IP Option Record Route'
    option = 7
    fields_desc = [_IPOption_HDR, FieldLenField('length', None, fmt='B', length_of='routers', adjust=lambda pkt, l: l + 3), ByteField('pointer', 4), FieldListField('routers', [], IPField('', '0.0.0.0'), length_from=lambda pkt: pkt.length - 3)]

    def get_current_router(self):
        if False:
            return 10
        return self.routers[self.pointer // 4 - 1]

class IPOption_LSRR(IPOption_RR):
    name = 'IP Option Loose Source and Record Route'
    copy_flag = 1
    option = 3

class IPOption_SSRR(IPOption_RR):
    name = 'IP Option Strict Source and Record Route'
    copy_flag = 1
    option = 9

class IPOption_Stream_Id(IPOption):
    name = 'IP Option Stream ID'
    copy_flag = 1
    option = 8
    fields_desc = [_IPOption_HDR, ByteField('length', 4), ShortField('security', 0)]

class IPOption_MTU_Probe(IPOption):
    name = 'IP Option MTU Probe'
    option = 11
    fields_desc = [_IPOption_HDR, ByteField('length', 4), ShortField('mtu', 0)]

class IPOption_MTU_Reply(IPOption_MTU_Probe):
    name = 'IP Option MTU Reply'
    option = 12

class IPOption_Traceroute(IPOption):
    name = 'IP Option Traceroute'
    option = 18
    fields_desc = [_IPOption_HDR, ByteField('length', 12), ShortField('id', 0), ShortField('outbound_hops', 0), ShortField('return_hops', 0), IPField('originator_ip', '0.0.0.0')]

class IPOption_Timestamp(IPOption):
    name = 'IP Option Timestamp'
    optclass = 2
    option = 4
    fields_desc = [_IPOption_HDR, ByteField('length', None), ByteField('pointer', 9), BitField('oflw', 0, 4), BitEnumField('flg', 1, 4, {0: 'timestamp_only', 1: 'timestamp_and_ip_addr', 3: 'prespecified_ip_addr'}), ConditionalField(IPField('internet_address', '0.0.0.0'), lambda pkt: pkt.flg != 0), IntField('timestamp', 0)]

    def post_build(self, p, pay):
        if False:
            i = 10
            return i + 15
        if self.length is None:
            p = p[:1] + struct.pack('!B', len(p)) + p[2:]
        return p + pay

class IPOption_Address_Extension(IPOption):
    name = 'IP Option Address Extension'
    copy_flag = 1
    option = 19
    fields_desc = [_IPOption_HDR, ByteField('length', 10), IPField('src_ext', '0.0.0.0'), IPField('dst_ext', '0.0.0.0')]

class IPOption_Router_Alert(IPOption):
    name = 'IP Option Router Alert'
    copy_flag = 1
    option = 20
    fields_desc = [_IPOption_HDR, ByteField('length', 4), ShortEnumField('alert', 0, {0: 'router_shall_examine_packet'})]

class IPOption_SDBM(IPOption):
    name = 'IP Option Selective Directed Broadcast Mode'
    copy_flag = 1
    option = 21
    fields_desc = [_IPOption_HDR, FieldLenField('length', None, fmt='B', length_of='addresses', adjust=lambda pkt, l: l + 2), FieldListField('addresses', [], IPField('', '0.0.0.0'), length_from=lambda pkt: pkt.length - 2)]
TCPOptions = ({0: ('EOL', None), 1: ('NOP', None), 2: ('MSS', '!H'), 3: ('WScale', '!B'), 4: ('SAckOK', None), 5: ('SAck', '!'), 8: ('Timestamp', '!II'), 14: ('AltChkSum', '!BH'), 15: ('AltChkSumOpt', None), 19: ('MD5', '16s'), 25: ('Mood', '!p'), 28: ('UTO', '!H'), 29: ('AO', None), 34: ('TFO', '!II')}, {'EOL': 0, 'NOP': 1, 'MSS': 2, 'WScale': 3, 'SAckOK': 4, 'SAck': 5, 'Timestamp': 8, 'AltChkSum': 14, 'AltChkSumOpt': 15, 'MD5': 19, 'Mood': 25, 'UTO': 28, 'AO': 29, 'TFO': 34})

class TCPAOValue(Packet):
    """Value of TCP-AO option"""
    fields_desc = [ByteField('keyid', None), ByteField('rnextkeyid', None), StrLenField('mac', '', length_from=lambda p: len(p.original) - 2)]

def get_tcpao(tcphdr):
    if False:
        print('Hello World!')
    'Get the TCP-AO option from the header'
    for (optid, optval) in tcphdr.options:
        if optid == 'AO':
            return optval
    return None

class RandTCPOptions(VolatileValue):

    def __init__(self, size=None):
        if False:
            i = 10
            return i + 15
        if size is None:
            size = RandNum(1, 5)
        self.size = size

    def _fix(self):
        if False:
            print('Hello World!')
        rand_patterns = [random.choice(list(((opt, fmt) for (opt, fmt) in TCPOptions[0].values() if opt != 'EOL'))) for _ in range(self.size)]
        rand_vals = []
        for (oname, fmt) in rand_patterns:
            if fmt is None:
                rand_vals.append((oname, b''))
            else:
                structs = re.findall('!?([bBhHiIlLqQfdpP]|\\d+[spx])', fmt)
                rval = []
                for stru in structs:
                    stru = '!' + stru
                    if 's' in stru or 'p' in stru:
                        v = bytes(RandBin(struct.calcsize(stru)))
                    else:
                        _size = struct.calcsize(stru)
                        v = random.randint(0, 2 ** (8 * _size) - 1)
                    rval.append(v)
                rand_vals.append((oname, tuple(rval)))
        return rand_vals

    def __bytes__(self):
        if False:
            while True:
                i = 10
        return TCPOptionsField.i2m(None, None, self._fix())

class TCPOptionsField(StrField):
    islist = 1

    def getfield(self, pkt, s):
        if False:
            for i in range(10):
                print('nop')
        opsz = (pkt.dataofs - 5) * 4
        if opsz < 0:
            log_runtime.info('bad dataofs (%i). Assuming dataofs=5', pkt.dataofs)
            opsz = 0
        return (s[opsz:], self.m2i(pkt, s[:opsz]))

    def m2i(self, pkt, x):
        if False:
            print('Hello World!')
        opt = []
        while x:
            onum = orb(x[0])
            if onum == 0:
                opt.append(('EOL', None))
                break
            if onum == 1:
                opt.append(('NOP', None))
                x = x[1:]
                continue
            try:
                olen = orb(x[1])
            except IndexError:
                olen = 0
            if olen < 2:
                log_runtime.info('Malformed TCP option (announced length is %i)', olen)
                olen = 2
            oval = x[2:olen]
            if onum in TCPOptions[0]:
                (oname, ofmt) = TCPOptions[0][onum]
                if onum == 5:
                    ofmt += '%iI' % (len(oval) // 4)
                if onum == 29:
                    oval = TCPAOValue(oval)
                if ofmt and struct.calcsize(ofmt) == len(oval):
                    oval = struct.unpack(ofmt, oval)
                    if len(oval) == 1:
                        oval = oval[0]
                opt.append((oname, oval))
            else:
                opt.append((onum, oval))
            x = x[olen:]
        return opt

    def i2h(self, pkt, x):
        if False:
            print('Hello World!')
        if not x:
            return []
        return x

    def i2m(self, pkt, x):
        if False:
            print('Hello World!')
        opt = b''
        for (oname, oval) in x:
            oname = {0: 'EOL', 1: 'NOP'}.get(oname, oname)
            if isinstance(oname, str):
                if oname == 'NOP':
                    opt += b'\x01'
                    continue
                elif oname == 'EOL':
                    opt += b'\x00'
                    continue
                elif oname in TCPOptions[1]:
                    onum = TCPOptions[1][oname]
                    ofmt = TCPOptions[0][onum][1]
                    if onum == 5:
                        ofmt += '%iI' % len(oval)
                    _test_isinstance = not isinstance(oval, (bytes, str))
                    if ofmt is not None and (_test_isinstance or 's' in ofmt):
                        if not isinstance(oval, tuple):
                            oval = (oval,)
                        oval = struct.pack(ofmt, *oval)
                    if onum == 29:
                        oval = bytes(oval)
                else:
                    warning('Option [%s] unknown. Skipped.', oname)
                    continue
            else:
                onum = oname
                if not isinstance(onum, int):
                    warning('Invalid option number [%i]' % onum)
                    continue
                if not isinstance(oval, (bytes, str)):
                    warning('Option [%i] is not bytes.' % onum)
                    continue
            if isinstance(oval, str):
                oval = bytes_encode(oval)
            opt += chb(onum) + chb(2 + len(oval)) + oval
        return opt + b'\x00' * (3 - (len(opt) + 3) % 4)

    def randval(self):
        if False:
            print('Hello World!')
        return RandTCPOptions()

class ICMPTimeStampField(IntField):
    re_hmsm = re.compile('([0-2]?[0-9])[Hh:](([0-5]?[0-9])([Mm:]([0-5]?[0-9])([sS:.]([0-9]{0,3}))?)?)?$')

    def i2repr(self, pkt, val):
        if False:
            i = 10
            return i + 15
        if val is None:
            return '--'
        else:
            (sec, milli) = divmod(val, 1000)
            (min, sec) = divmod(sec, 60)
            (hour, min) = divmod(min, 60)
            return '%d:%d:%d.%d' % (hour, min, sec, int(milli))

    def any2i(self, pkt, val):
        if False:
            print('Hello World!')
        if isinstance(val, str):
            hmsms = self.re_hmsm.match(val)
            if hmsms:
                (h, _, m, _, s, _, ms) = hmsms.groups()
                ms = int(((ms or '') + '000')[:3])
                val = ((int(h) * 60 + int(m or 0)) * 60 + int(s or 0)) * 1000 + ms
            else:
                val = 0
        elif val is None:
            val = int(time.time() % (24 * 60 * 60) * 1000)
        return val

class DestIPField(IPField, DestField):
    bindings = {}

    def __init__(self, name, default):
        if False:
            for i in range(10):
                print('nop')
        IPField.__init__(self, name, None)
        DestField.__init__(self, name, default)

    def i2m(self, pkt, x):
        if False:
            i = 10
            return i + 15
        if x is None:
            x = self.dst_from_pkt(pkt)
        return IPField.i2m(self, pkt, x)

    def i2h(self, pkt, x):
        if False:
            print('Hello World!')
        if x is None:
            x = self.dst_from_pkt(pkt)
        return IPField.i2h(self, pkt, x)

class IP(Packet, IPTools):
    name = 'IP'
    fields_desc = [BitField('version', 4, 4), BitField('ihl', None, 4), XByteField('tos', 0), ShortField('len', None), ShortField('id', 1), FlagsField('flags', 0, 3, ['MF', 'DF', 'evil']), BitField('frag', 0, 13), ByteField('ttl', 64), ByteEnumField('proto', 0, IP_PROTOS), XShortField('chksum', None), Emph(SourceIPField('src', 'dst')), Emph(DestIPField('dst', '127.0.0.1')), PacketListField('options', [], IPOption, length_from=lambda p: p.ihl * 4 - 20)]

    def post_build(self, p, pay):
        if False:
            i = 10
            return i + 15
        ihl = self.ihl
        p += b'\x00' * (-len(p) % 4)
        if ihl is None:
            ihl = len(p) // 4
            p = chb((self.version & 15) << 4 | ihl & 15) + p[1:]
        if self.len is None:
            tmp_len = len(p) + len(pay)
            p = p[:2] + struct.pack('!H', tmp_len) + p[4:]
        if self.chksum is None:
            ck = checksum(p)
            p = p[:10] + chb(ck >> 8) + chb(ck & 255) + p[12:]
        return p + pay

    def extract_padding(self, s):
        if False:
            for i in range(10):
                print('nop')
        tmp_len = self.len - (self.ihl << 2)
        if tmp_len < 0:
            return (s, b'')
        return (s[:tmp_len], s[tmp_len:])

    def route(self):
        if False:
            print('Hello World!')
        dst = self.dst
        if isinstance(dst, Gen):
            dst = next(iter(dst))
        if conf.route is None:
            import scapy.route
        return conf.route.route(dst)

    def hashret(self):
        if False:
            for i in range(10):
                print('nop')
        if self.proto == socket.IPPROTO_ICMP and isinstance(self.payload, ICMP) and (self.payload.type in [3, 4, 5, 11, 12]):
            return self.payload.payload.hashret()
        if not conf.checkIPinIP and self.proto in [4, 41]:
            return self.payload.hashret()
        if self.dst == '224.0.0.251':
            return struct.pack('B', self.proto) + self.payload.hashret()
        if conf.checkIPsrc and conf.checkIPaddr:
            return strxor(inet_pton(socket.AF_INET, self.src), inet_pton(socket.AF_INET, self.dst)) + struct.pack('B', self.proto) + self.payload.hashret()
        return struct.pack('B', self.proto) + self.payload.hashret()

    def answers(self, other):
        if False:
            return 10
        if not conf.checkIPinIP:
            if self.proto in [4, 41]:
                return self.payload.answers(other)
            if isinstance(other, IP) and other.proto in [4, 41]:
                return self.answers(other.payload)
            if conf.ipv6_enabled and isinstance(other, scapy.layers.inet6.IPv6) and (other.nh in [4, 41]):
                return self.answers(other.payload)
        if not isinstance(other, IP):
            return 0
        if conf.checkIPaddr:
            if other.dst == '224.0.0.251' and self.dst == '224.0.0.251':
                return self.payload.answers(other.payload)
            elif self.dst != other.src:
                return 0
        if self.proto == socket.IPPROTO_ICMP and isinstance(self.payload, ICMP) and (self.payload.type in [3, 4, 5, 11, 12]):
            return self.payload.payload.answers(other)
        else:
            if conf.checkIPaddr and self.src != other.dst or self.proto != other.proto:
                return 0
            return self.payload.answers(other.payload)

    def mysummary(self):
        if False:
            while True:
                i = 10
        s = self.sprintf('%IP.src% > %IP.dst% %IP.proto%')
        if self.frag:
            s += ' frag:%i' % self.frag
        return s

    def fragment(self, fragsize=1480):
        if False:
            return 10
        'Fragment IP datagrams'
        return fragment(self, fragsize=fragsize)

def in4_pseudoheader(proto, u, plen):
    if False:
        print('Hello World!')
    'IPv4 Pseudo Header as defined in RFC793 as bytes\n\n    :param proto: value of upper layer protocol\n    :param u: IP layer instance\n    :param plen: the length of the upper layer and payload\n    '
    if u.len is not None:
        if u.ihl is None:
            olen = sum((len(x) for x in u.options))
            ihl = 5 + olen // 4 + (1 if olen % 4 else 0)
        else:
            ihl = u.ihl
        ln = max(u.len - 4 * ihl, 0)
    else:
        ln = plen
    sr_options = [opt for opt in u.options if isinstance(opt, IPOption_LSRR) or isinstance(opt, IPOption_SSRR)]
    len_sr_options = len(sr_options)
    if len_sr_options == 1 and len(sr_options[0].routers):
        u.dst = sr_options[0].routers[-1]
    elif len_sr_options > 1:
        message = 'Found %d Source Routing Options! '
        message += 'Falling back to IP.dst for checksum computation.'
        warning(message, len_sr_options)
    return struct.pack('!4s4sHH', inet_pton(socket.AF_INET, u.src), inet_pton(socket.AF_INET, u.dst), proto, ln)

def in4_chksum(proto, u, p):
    if False:
        print('Hello World!')
    'IPv4 Pseudo Header checksum as defined in RFC793\n\n    :param proto: value of upper layer protocol\n    :param u: upper layer instance\n    :param p: the payload of the upper layer provided as a string\n    '
    if not isinstance(u, IP):
        warning('No IP underlayer to compute checksum. Leaving null.')
        return 0
    psdhdr = in4_pseudoheader(proto, u, len(p))
    return checksum(psdhdr + p)

def _is_ipv6_layer(p):
    if False:
        for i in range(10):
            print('nop')
    return isinstance(p, scapy.layers.inet6.IPv6) or isinstance(p, scapy.layers.inet6._IPv6ExtHdr)

def tcp_pseudoheader(tcp):
    if False:
        for i in range(10):
            print('nop')
    'Pseudoheader of a TCP packet as bytes\n\n    Requires underlayer to be either IP or IPv6\n    '
    if isinstance(tcp.underlayer, IP):
        plen = len(bytes(tcp))
        return in4_pseudoheader(socket.IPPROTO_TCP, tcp.underlayer, plen)
    elif conf.ipv6_enabled and _is_ipv6_layer(tcp.underlayer):
        plen = len(bytes(tcp))
        return raw(scapy.layers.inet6.in6_pseudoheader(socket.IPPROTO_TCP, tcp.underlayer, plen))
    else:
        raise ValueError('TCP packet does not have IP or IPv6 underlayer')

def calc_tcp_md5_hash(tcp, key):
    if False:
        return 10
    'Calculate TCP-MD5 hash from packet and return a 16-byte string'
    import hashlib
    h = hashlib.md5()
    tcp_bytes = bytes(tcp)
    h.update(tcp_pseudoheader(tcp))
    h.update(tcp_bytes[:16])
    h.update(b'\x00\x00')
    h.update(tcp_bytes[18:])
    h.update(key)
    return h.digest()

def sign_tcp_md5(tcp, key):
    if False:
        i = 10
        return i + 15
    'Append TCP-MD5 signature to tcp packet'
    sig = calc_tcp_md5_hash(tcp, key)
    tcp.options = tcp.options + [('MD5', sig)]

class TCP(Packet):
    name = 'TCP'
    fields_desc = [ShortEnumField('sport', 20, TCP_SERVICES), ShortEnumField('dport', 80, TCP_SERVICES), IntField('seq', 0), IntField('ack', 0), BitField('dataofs', None, 4), BitField('reserved', 0, 3), FlagsField('flags', 2, 9, 'FSRPAUECN'), ShortField('window', 8192), XShortField('chksum', None), ShortField('urgptr', 0), TCPOptionsField('options', '')]

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        p += pay
        dataofs = self.dataofs
        if dataofs is None:
            opt_len = len(self.get_field('options').i2m(self, self.options))
            dataofs = 5 + (opt_len + 3) // 4
            dataofs = dataofs << 4 | orb(p[12]) & 15
            p = p[:12] + chb(dataofs & 255) + p[13:]
        if self.chksum is None:
            if isinstance(self.underlayer, IP):
                ck = in4_chksum(socket.IPPROTO_TCP, self.underlayer, p)
                p = p[:16] + struct.pack('!H', ck) + p[18:]
            elif conf.ipv6_enabled and isinstance(self.underlayer, scapy.layers.inet6.IPv6) or isinstance(self.underlayer, scapy.layers.inet6._IPv6ExtHdr):
                ck = scapy.layers.inet6.in6_chksum(socket.IPPROTO_TCP, self.underlayer, p)
                p = p[:16] + struct.pack('!H', ck) + p[18:]
            else:
                log_runtime.info('No IP underlayer to compute checksum. Leaving null.')
        return p

    def hashret(self):
        if False:
            print('Hello World!')
        if conf.checkIPsrc:
            return struct.pack('H', self.sport ^ self.dport) + self.payload.hashret()
        else:
            return self.payload.hashret()

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, TCP):
            return 0
        if other.flags.R:
            return 0
        if self.flags.S:
            if not self.flags.A:
                return 0
            if not other.flags.S:
                return 0
        if conf.checkIPsrc:
            if not (self.sport == other.dport and self.dport == other.sport):
                return 0
        if not (other.flags.S and (not other.flags.A)) and abs(other.ack - self.seq) > 2:
            return 0
        if self.flags.R and (not self.flags.A):
            return 1
        if abs(other.seq - self.ack) > 2 + len(other.payload):
            return 0
        return 1

    def mysummary(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self.underlayer, IP):
            return self.underlayer.sprintf('TCP %IP.src%:%TCP.sport% > %IP.dst%:%TCP.dport% %TCP.flags%')
        elif conf.ipv6_enabled and isinstance(self.underlayer, scapy.layers.inet6.IPv6):
            return self.underlayer.sprintf('TCP %IPv6.src%:%TCP.sport% > %IPv6.dst%:%TCP.dport% %TCP.flags%')
        else:
            return self.sprintf('TCP %TCP.sport% > %TCP.dport% %TCP.flags%')

class UDP(Packet):
    name = 'UDP'
    fields_desc = [ShortEnumField('sport', 53, UDP_SERVICES), ShortEnumField('dport', 53, UDP_SERVICES), ShortField('len', None), XShortField('chksum', None)]

    def post_build(self, p, pay):
        if False:
            return 10
        p += pay
        tmp_len = self.len
        if tmp_len is None:
            tmp_len = len(p)
            p = p[:4] + struct.pack('!H', tmp_len) + p[6:]
        if self.chksum is None:
            if isinstance(self.underlayer, IP):
                ck = in4_chksum(socket.IPPROTO_UDP, self.underlayer, p)
                if ck == 0:
                    ck = 65535
                p = p[:6] + struct.pack('!H', ck) + p[8:]
            elif isinstance(self.underlayer, scapy.layers.inet6.IPv6) or isinstance(self.underlayer, scapy.layers.inet6._IPv6ExtHdr):
                ck = scapy.layers.inet6.in6_chksum(socket.IPPROTO_UDP, self.underlayer, p)
                if ck == 0:
                    ck = 65535
                p = p[:6] + struct.pack('!H', ck) + p[8:]
            else:
                log_runtime.info('No IP underlayer to compute checksum. Leaving null.')
        return p

    def extract_padding(self, s):
        if False:
            print('Hello World!')
        tmp_len = self.len - 8
        return (s[:tmp_len], s[tmp_len:])

    def hashret(self):
        if False:
            i = 10
            return i + 15
        return self.payload.hashret()

    def answers(self, other):
        if False:
            return 10
        if not isinstance(other, UDP):
            return 0
        if conf.checkIPsrc:
            if self.dport != other.sport:
                return 0
        return self.payload.answers(other.payload)

    def mysummary(self):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.underlayer, IP):
            return self.underlayer.sprintf('UDP %IP.src%:%UDP.sport% > %IP.dst%:%UDP.dport%')
        elif isinstance(self.underlayer, scapy.layers.inet6.IPv6):
            return self.underlayer.sprintf('UDP %IPv6.src%:%UDP.sport% > %IPv6.dst%:%UDP.dport%')
        else:
            return self.sprintf('UDP %UDP.sport% > %UDP.dport%')
icmptypes = {0: 'echo-reply', 3: 'dest-unreach', 4: 'source-quench', 5: 'redirect', 8: 'echo-request', 9: 'router-advertisement', 10: 'router-solicitation', 11: 'time-exceeded', 12: 'parameter-problem', 13: 'timestamp-request', 14: 'timestamp-reply', 15: 'information-request', 16: 'information-response', 17: 'address-mask-request', 18: 'address-mask-reply', 30: 'traceroute', 31: 'datagram-conversion-error', 32: 'mobile-host-redirect', 33: 'ipv6-where-are-you', 34: 'ipv6-i-am-here', 35: 'mobile-registration-request', 36: 'mobile-registration-reply', 37: 'domain-name-request', 38: 'domain-name-reply', 39: 'skip', 40: 'photuris'}
icmpcodes = {3: {0: 'network-unreachable', 1: 'host-unreachable', 2: 'protocol-unreachable', 3: 'port-unreachable', 4: 'fragmentation-needed', 5: 'source-route-failed', 6: 'network-unknown', 7: 'host-unknown', 9: 'network-prohibited', 10: 'host-prohibited', 11: 'TOS-network-unreachable', 12: 'TOS-host-unreachable', 13: 'communication-prohibited', 14: 'host-precedence-violation', 15: 'precedence-cutoff'}, 5: {0: 'network-redirect', 1: 'host-redirect', 2: 'TOS-network-redirect', 3: 'TOS-host-redirect'}, 11: {0: 'ttl-zero-during-transit', 1: 'ttl-zero-during-reassembly'}, 12: {0: 'ip-header-bad', 1: 'required-option-missing'}, 40: {0: 'bad-spi', 1: 'authentication-failed', 2: 'decompression-failed', 3: 'decryption-failed', 4: 'need-authentification', 5: 'need-authorization'}}
icmp_id_seq_types = [0, 8, 13, 14, 15, 16, 17, 18, 37, 38]

class ICMP(Packet):
    name = 'ICMP'
    fields_desc = [ByteEnumField('type', 8, icmptypes), MultiEnumField('code', 0, icmpcodes, depends_on=lambda pkt: pkt.type, fmt='B'), XShortField('chksum', None), ConditionalField(XShortField('id', 0), lambda pkt: pkt.type in icmp_id_seq_types), ConditionalField(XShortField('seq', 0), lambda pkt: pkt.type in icmp_id_seq_types), ConditionalField(ICMPTimeStampField('ts_ori', None), lambda pkt: pkt.type in [13, 14]), ConditionalField(ICMPTimeStampField('ts_rx', None), lambda pkt: pkt.type in [13, 14]), ConditionalField(ICMPTimeStampField('ts_tx', None), lambda pkt: pkt.type in [13, 14]), ConditionalField(IPField('gw', '0.0.0.0'), lambda pkt: pkt.type == 5), ConditionalField(ByteField('ptr', 0), lambda pkt: pkt.type == 12), ConditionalField(ByteField('reserved', 0), lambda pkt: pkt.type in [3, 11]), ConditionalField(ByteField('length', 0), lambda pkt: pkt.type in [3, 11, 12]), ConditionalField(IPField('addr_mask', '0.0.0.0'), lambda pkt: pkt.type in [17, 18]), ConditionalField(ShortField('nexthopmtu', 0), lambda pkt: pkt.type == 3), MultipleTypeField([(ShortField('unused', 0), lambda pkt: pkt.type in [11, 12]), (IntField('unused', 0), lambda pkt: pkt.type not in [0, 3, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18])], StrFixedLenField('unused', '', length=0))]

    def post_build(self, p, pay):
        if False:
            while True:
                i = 10
        p += pay
        if self.chksum is None:
            ck = checksum(p)
            p = p[:2] + chb(ck >> 8) + chb(ck & 255) + p[4:]
        return p

    def hashret(self):
        if False:
            while True:
                i = 10
        if self.type in icmp_id_seq_types:
            return struct.pack('HH', self.id, self.seq) + self.payload.hashret()
        return self.payload.hashret()

    def answers(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, ICMP):
            return 0
        if (other.type, self.type) in [(8, 0), (13, 14), (15, 16), (17, 18), (33, 34), (35, 36), (37, 38)] and self.id == other.id and (self.seq == other.seq):
            return 1
        return 0

    def guess_payload_class(self, payload):
        if False:
            while True:
                i = 10
        if self.type in [3, 4, 5, 11, 12]:
            return IPerror
        else:
            return None

    def mysummary(self):
        if False:
            while True:
                i = 10
        if isinstance(self.underlayer, IP):
            return self.underlayer.sprintf('ICMP %IP.src% > %IP.dst% %ICMP.type% %ICMP.code%')
        else:
            return self.sprintf('ICMP %ICMP.type% %ICMP.code%')

class IPerror(IP):
    name = 'IP in ICMP'

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, IP):
            return 0
        test_IPsrc = not conf.checkIPsrc or self.src == other.src
        test_IPdst = self.dst == other.dst
        test_IPid = not conf.checkIPID or self.id == other.id
        test_IPid |= conf.checkIPID and self.id == socket.htons(other.id)
        test_IPproto = self.proto == other.proto
        if not (test_IPsrc and test_IPdst and test_IPid and test_IPproto):
            return 0
        return self.payload.answers(other.payload)

    def mysummary(self):
        if False:
            for i in range(10):
                print('nop')
        return Packet.mysummary(self)

class TCPerror(TCP):
    name = 'TCP in ICMP'

    def answers(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, TCP):
            return 0
        if conf.checkIPsrc:
            if not (self.sport == other.sport and self.dport == other.dport):
                return 0
        if conf.check_TCPerror_seqack:
            if self.seq is not None:
                if self.seq != other.seq:
                    return 0
            if self.ack is not None:
                if self.ack != other.ack:
                    return 0
        return 1

    def mysummary(self):
        if False:
            i = 10
            return i + 15
        return Packet.mysummary(self)

class UDPerror(UDP):
    name = 'UDP in ICMP'

    def answers(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, UDP):
            return 0
        if conf.checkIPsrc:
            if not (self.sport == other.sport and self.dport == other.dport):
                return 0
        return 1

    def mysummary(self):
        if False:
            while True:
                i = 10
        return Packet.mysummary(self)

class ICMPerror(ICMP):
    name = 'ICMP in ICMP'

    def answers(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, ICMP):
            return 0
        if not (self.type == other.type and self.code == other.code):
            return 0
        if self.code in [0, 8, 13, 14, 17, 18]:
            if self.id == other.id and self.seq == other.seq:
                return 1
            else:
                return 0
        else:
            return 1

    def mysummary(self):
        if False:
            while True:
                i = 10
        return Packet.mysummary(self)
bind_layers(Ether, IP, type=2048)
bind_layers(CookedLinux, IP, proto=2048)
bind_layers(GRE, IP, proto=2048)
bind_layers(SNAP, IP, code=2048)
bind_bottom_up(Loopback, IP, type=0)
bind_layers(Loopback, IP, type=socket.AF_INET)
bind_layers(IPerror, IPerror, frag=0, proto=4)
bind_layers(IPerror, ICMPerror, frag=0, proto=1)
bind_layers(IPerror, TCPerror, frag=0, proto=6)
bind_layers(IPerror, UDPerror, frag=0, proto=17)
bind_layers(IP, IP, frag=0, proto=4)
bind_layers(IP, ICMP, frag=0, proto=1)
bind_layers(IP, TCP, frag=0, proto=6)
bind_layers(IP, UDP, frag=0, proto=17)
bind_layers(IP, GRE, frag=0, proto=47)
conf.l2types.register(DLT_RAW, IP)
conf.l2types.register_num2layer(DLT_RAW_ALT, IP)
conf.l2types.register(DLT_IPV4, IP)
conf.l3types.register(ETH_P_IP, IP)
conf.l3types.register_num2layer(ETH_P_ALL, IP)

def inet_register_l3(l2, l3):
    if False:
        return 10
    '\n    Resolves the default L2 destination address when IP is used.\n    '
    return getmacbyip(l3.dst)
conf.neighbor.register_l3(Ether, IP, inet_register_l3)
conf.neighbor.register_l3(Dot3, IP, inet_register_l3)

@conf.commands.register
def fragment(pkt, fragsize=1480):
    if False:
        print('Hello World!')
    'Fragment a big IP datagram'
    if fragsize < 8:
        warning('fragsize cannot be lower than 8')
        fragsize = max(fragsize, 8)
    lastfragsz = fragsize
    fragsize -= fragsize % 8
    lst = []
    for p in pkt:
        s = raw(p[IP].payload)
        nb = (len(s) - lastfragsz + fragsize - 1) // fragsize + 1
        for i in range(nb):
            q = p.copy()
            del q[IP].payload
            del q[IP].chksum
            del q[IP].len
            if i != nb - 1:
                q[IP].flags |= 1
                fragend = (i + 1) * fragsize
            else:
                fragend = i * fragsize + lastfragsz
            q[IP].frag += i * fragsize // 8
            r = conf.raw_layer(load=s[i * fragsize:fragend])
            r.overload_fields = p[IP].payload.overload_fields.copy()
            q.add_payload(r)
            lst.append(q)
    return lst

@conf.commands.register
def overlap_frag(p, overlap, fragsize=8, overlap_fragsize=None):
    if False:
        i = 10
        return i + 15
    'Build overlapping fragments to bypass NIPS\n\np:                the original packet\noverlap:          the overlapping data\nfragsize:         the fragment size of the packet\noverlap_fragsize: the fragment size of the overlapping packet'
    if overlap_fragsize is None:
        overlap_fragsize = fragsize
    q = p.copy()
    del q[IP].payload
    q[IP].add_payload(overlap)
    qfrag = fragment(q, overlap_fragsize)
    qfrag[-1][IP].flags |= 1
    return qfrag + fragment(p, fragsize)

class BadFragments(ValueError):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.frags = kwargs.pop('frags', None)
        super(BadFragments, self).__init__(*args, **kwargs)

def _defrag_iter_and_check_offsets(frags):
    if False:
        while True:
            i = 10
    '\n    Internal generator used in _defrag_ip_pkt\n    '
    offset = 0
    for (pkt, o, length) in frags:
        if offset != o:
            if offset > o:
                op = '>'
            else:
                op = '<'
            warning('Fragment overlap (%i %s %i) on %r' % (offset, op, o, pkt))
            raise BadFragments
        offset += length
        yield bytes(pkt[IP].payload)

def _defrag_ip_pkt(pkt, frags):
    if False:
        print('Hello World!')
    '\n    Defragment a single IP packet.\n\n    :param pkt: the new pkt\n    :param frags: a defaultdict(list) used for storage\n    :return: a tuple (fragmented, defragmented_value)\n    '
    ip = pkt[IP]
    if pkt.frag != 0 or ip.flags.MF:
        uid = (ip.id, ip.src, ip.dst, ip.proto)
        if ip.len is None or ip.ihl is None:
            fraglen = len(ip.payload)
        else:
            fraglen = ip.len - (ip.ihl << 2)
        frags[uid].append((pkt, ip.frag << 3, fraglen))
        if not ip.flags.MF:
            curfrags = sorted(frags[uid], key=lambda x: x[1])
            try:
                data = b''.join(_defrag_iter_and_check_offsets(curfrags))
            except ValueError:
                badfrags = frags[uid]
                del frags[uid]
                raise BadFragments(frags=badfrags)
            p = curfrags[0][0].copy()
            pay_class = p[IP].payload.__class__
            p[IP].flags.MF = False
            p[IP].remove_payload()
            p[IP].len = None
            p[IP].chksum = None
            p /= pay_class(data)
            del frags[uid]
            return (True, p)
        return (True, None)
    return (False, pkt)

def _defrag_logic(plist, complete=False):
    if False:
        print('Hello World!')
    '\n    Internal function used to defragment a list of packets.\n    It contains the logic behind the defrag() and defragment() functions\n    '
    frags = defaultdict(list)
    final = []
    notfrag = []
    badfrag = []
    for (i, pkt) in enumerate(plist):
        if IP not in pkt:
            if complete:
                final.append(pkt)
            continue
        try:
            (fragmented, defragmented_value) = _defrag_ip_pkt(pkt, frags)
        except BadFragments as ex:
            if complete:
                final.extend(ex.frags)
            else:
                badfrag.extend(ex.frags)
            continue
        if complete and defragmented_value:
            final.append(defragmented_value)
        elif defragmented_value:
            if fragmented:
                final.append(defragmented_value)
            else:
                notfrag.append(defragmented_value)
    if complete:
        if hasattr(plist, 'listname'):
            name = 'Defragmented %s' % plist.listname
        else:
            name = 'Defragmented'
        return PacketList(final, name=name)
    else:
        return (PacketList(notfrag), PacketList(final), PacketList(badfrag))

@conf.commands.register
def defrag(plist):
    if False:
        for i in range(10):
            print('nop')
    'defrag(plist) -> ([not fragmented], [defragmented],\n                  [ [bad fragments], [bad fragments], ... ])'
    return _defrag_logic(plist, complete=False)

@conf.commands.register
def defragment(plist):
    if False:
        print('Hello World!')
    'defragment(plist) -> plist defragmented as much as possible '
    return _defrag_logic(plist, complete=True)

def _packetlist_timeskew_graph(self, ip, **kargs):
    if False:
        while True:
            i = 10
    'Tries to graph the timeskew between the timestamps and real time for a given ip'
    from scapy.libs.matplot import plt, MATPLOTLIB_INLINED, MATPLOTLIB_DEFAULT_PLOT_KARGS
    tmp = (self._elt2pkt(x) for x in self.res)
    b = (x for x in tmp if IP in x and x[IP].src == ip and (TCP in x))
    c = []
    tsf = ICMPTimeStampField('', None)
    for p in b:
        opts = p.getlayer(TCP).options
        for o in opts:
            if o[0] == 'Timestamp':
                c.append((p.time, tsf.any2i('', o[1][0])))
    if not c:
        warning('No timestamps found in packet list')
        return []
    first_creation_time = c[0][0]
    first_replied_timestamp = c[0][1]

    def _wrap_data(ts_tuple, wrap_seconds=2000):
        if False:
            i = 10
            return i + 15
        'Wrap the list of tuples.'
        (ct, rt) = ts_tuple
        X = ct % wrap_seconds
        Y = ct - first_creation_time - (rt - first_replied_timestamp) / 1000.0
        return (X, Y)
    data = [_wrap_data(e) for e in c]
    if kargs == {}:
        kargs = MATPLOTLIB_DEFAULT_PLOT_KARGS
    lines = plt.plot(data, **kargs)
    if not MATPLOTLIB_INLINED:
        plt.show()
    return lines
_PacketList.timeskew_graph = _packetlist_timeskew_graph

class TracerouteResult(SndRcvList):
    __slots__ = ['graphdef', 'graphpadding', 'graphASres', 'padding', 'hloc', 'nloc']

    def __init__(self, res=None, name='Traceroute', stats=None):
        if False:
            print('Hello World!')
        SndRcvList.__init__(self, res, name, stats)
        self.graphdef = None
        self.graphASres = None
        self.padding = 0
        self.hloc = None
        self.nloc = None

    def show(self):
        if False:
            i = 10
            return i + 15
        return self.make_table(lambda s, r: (s.sprintf('%IP.dst%:{TCP:tcp%ir,TCP.dport%}{UDP:udp%ir,UDP.dport%}{ICMP:ICMP}'), s.ttl, r.sprintf('%-15s,IP.src% {TCP:%TCP.flags%}{ICMP:%ir,ICMP.type%}')))

    def get_trace(self):
        if False:
            while True:
                i = 10
        trace = {}
        for (s, r) in self.res:
            if IP not in s:
                continue
            d = s[IP].dst
            if d not in trace:
                trace[d] = {}
            trace[d][s[IP].ttl] = (r[IP].src, ICMP not in r)
        for k in trace.values():
            try:
                m = min((x for (x, y) in k.items() if y[1]))
            except ValueError:
                continue
            for li in list(k):
                if li > m:
                    del k[li]
        return trace

    def trace3D(self, join=True):
        if False:
            return 10
        'Give a 3D representation of the traceroute.\n        right button: rotate the scene\n        middle button: zoom\n        shift-left button: move the scene\n        left button on a ball: toggle IP displaying\n        double-click button on a ball: scan ports 21,22,23,25,80 and 443 and display the result'
        import multiprocessing
        p = multiprocessing.Process(target=self.trace3D_notebook)
        p.start()
        if join:
            p.join()

    def trace3D_notebook(self):
        if False:
            i = 10
            return i + 15
        'Same than trace3D, used when ran from Jupyter notebooks'
        trace = self.get_trace()
        import vpython

        class IPsphere(vpython.sphere):

            def __init__(self, ip, **kargs):
                if False:
                    while True:
                        i = 10
                vpython.sphere.__init__(self, **kargs)
                self.ip = ip
                self.label = None
                self.setlabel(self.ip)
                self.last_clicked = None
                self.full = False
                self.savcolor = vpython.vec(*self.color.value)

            def fullinfos(self):
                if False:
                    print('Hello World!')
                self.full = True
                self.color = vpython.vec(1, 0, 0)
                (a, b) = sr(IP(dst=self.ip) / TCP(dport=[21, 22, 23, 25, 80, 443], flags='S'), timeout=2, verbose=0)
                if len(a) == 0:
                    txt = '%s:\nno results' % self.ip
                else:
                    txt = '%s:\n' % self.ip
                    for (s, r) in a:
                        txt += r.sprintf('{TCP:%IP.src%:%TCP.sport% %TCP.flags%}{TCPerror:%IPerror.dst%:%TCPerror.dport% %IP.src% %ir,ICMP.type%}\n')
                self.setlabel(txt, visible=1)

            def unfull(self):
                if False:
                    return 10
                self.color = self.savcolor
                self.full = False
                self.setlabel(self.ip)

            def setlabel(self, txt, visible=None):
                if False:
                    print('Hello World!')
                if self.label is not None:
                    if visible is None:
                        visible = self.label.visible
                    self.label.visible = 0
                elif visible is None:
                    visible = 0
                self.label = vpython.label(text=txt, pos=self.pos, space=self.radius, xoffset=10, yoffset=20, visible=visible)

            def check_double_click(self):
                if False:
                    for i in range(10):
                        print('nop')
                try:
                    if self.full or not self.label.visible:
                        return False
                    if self.last_clicked is not None:
                        return time.time() - self.last_clicked < 0.5
                    return False
                finally:
                    self.last_clicked = time.time()

            def action(self):
                if False:
                    return 10
                self.label.visible ^= 1
                if self.full:
                    self.unfull()
        vpython.scene = vpython.canvas()
        vpython.scene.title = '<center><u><b>%s</b></u></center>' % self.listname
        vpython.scene.append_to_caption(re.sub('\\%(.*)\\%', '<span style="color: red">\\1</span>', re.sub('\\`(.*)\\`', '<span style="color: #3399ff">\\1</span>', '<u><b>Commands:</b></u>\n%Click% to toggle information about a node.\n%Double click% to perform a quick web scan on this node.\n<u><b>Camera usage:</b></u>\n`Right button drag or Ctrl-drag` to rotate "camera" to view scene.\n`Shift-drag` to move the object around.\n`Middle button or Alt-drag` to drag up or down to zoom in or out.\n  On a two-button mouse, `middle is wheel or left + right`.\nTouch screen: pinch/extend to zoom, swipe or two-finger rotate.')))
        vpython.scene.exit = True
        rings = {}
        tr3d = {}
        for i in trace:
            tr = trace[i]
            tr3d[i] = []
            for t in range(1, max(tr) + 1):
                if t not in rings:
                    rings[t] = []
                if t in tr:
                    if tr[t] not in rings[t]:
                        rings[t].append(tr[t])
                    tr3d[i].append(rings[t].index(tr[t]))
                else:
                    rings[t].append(('unk', -1))
                    tr3d[i].append(len(rings[t]) - 1)
        for t in rings:
            r = rings[t]
            tmp_len = len(r)
            for i in range(tmp_len):
                if r[i][1] == -1:
                    col = vpython.vec(0.75, 0.75, 0.75)
                elif r[i][1]:
                    col = vpython.color.green
                else:
                    col = vpython.color.blue
                s = IPsphere(pos=vpython.vec((tmp_len - 1) * vpython.cos(2 * i * vpython.pi / tmp_len), (tmp_len - 1) * vpython.sin(2 * i * vpython.pi / tmp_len), 2 * t), ip=r[i][0], color=col)
                for trlst in tr3d.values():
                    if t <= len(trlst):
                        if trlst[t - 1] == i:
                            trlst[t - 1] = s
        forecol = colgen(0.625, 0.4375, 0.25, 0.125)
        for trlst in tr3d.values():
            col = vpython.vec(*next(forecol))
            start = vpython.vec(0, 0, 0)
            for ip in trlst:
                vpython.cylinder(pos=start, axis=ip.pos - start, color=col, radius=0.2)
                start = ip.pos
        vpython.rate(50)

        def mouse_click(ev):
            if False:
                return 10
            if ev.press == 'left':
                o = vpython.scene.mouse.pick
                if o and isinstance(o, IPsphere):
                    if o.check_double_click():
                        if o.ip == 'unk':
                            return
                        o.fullinfos()
                    else:
                        o.action()
        vpython.scene.bind('mousedown', mouse_click)

    def world_trace(self):
        if False:
            return 10
        'Display traceroute results on a world map.'
        from scapy.libs.matplot import plt, MATPLOTLIB, MATPLOTLIB_INLINED
        try:
            import geoip2.database
            import geoip2.errors
        except ImportError:
            log_runtime.error("Cannot import geoip2. Won't be able to plot the world.")
            return []
        if not conf.geoip_city:
            log_runtime.error('Cannot import the geolite2 CITY database.\nDownload it from http://dev.maxmind.com/geoip/geoip2/geolite2/ then set its path to conf.geoip_city')
            return []
        try:
            import cartopy.crs as ccrs
        except ImportError:
            log_runtime.error('Cannot import cartopy.\nMore infos on http://scitools.org.uk/cartopy/docs/latest/installing.html')
            return []
        if not MATPLOTLIB:
            log_runtime.error("Matplotlib is not installed. Won't be able to plot the world.")
            return []
        try:
            db = geoip2.database.Reader(conf.geoip_city)
        except Exception:
            log_runtime.error('Cannot open geoip2 database at %s', conf.geoip_city)
            return []
        ips = {}
        rt = {}
        ports_done = {}
        for (s, r) in self.res:
            ips[r.src] = None
            if s.haslayer(TCP) or s.haslayer(UDP):
                trace_id = (s.src, s.dst, s.proto, s.dport)
            elif s.haslayer(ICMP):
                trace_id = (s.src, s.dst, s.proto, s.type)
            else:
                trace_id = (s.src, s.dst, s.proto, 0)
            trace = rt.get(trace_id, {})
            if not r.haslayer(ICMP) or r.type != 11:
                if trace_id in ports_done:
                    continue
                ports_done[trace_id] = None
            trace[s.ttl] = r.src
            rt[trace_id] = trace
        trt = {}
        for trace_id in rt:
            trace = rt[trace_id]
            loctrace = []
            for i in range(max(trace)):
                ip = trace.get(i, None)
                if ip is None:
                    continue
                try:
                    sresult = db.city(ip)
                except geoip2.errors.AddressNotFoundError:
                    continue
                loctrace.append((sresult.location.longitude, sresult.location.latitude))
            if loctrace:
                trt[trace_id] = loctrace
        plt.figure(num='Scapy')
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.stock_img()
        ax.set_global()
        plt.title('Scapy traceroute results')
        from matplotlib.collections import LineCollection
        from matplotlib import colors as mcolors
        colors_cycle = iter(mcolors.BASE_COLORS)
        lines = []
        for (key, trc) in trt.items():
            color = next(colors_cycle)
            data_lines = [(trc[i], trc[i + 1]) for i in range(len(trc) - 1)]
            line_col = LineCollection(data_lines, linewidths=2, label=key[1], color=color)
            lines.append(line_col)
            ax.add_collection(line_col)
            lines.extend([ax.plot(*x, marker='.', color=color) for x in trc])
        ax.legend()
        if not MATPLOTLIB_INLINED:
            plt.show()
        ax.remove()
        return lines

    def make_graph(self, ASres=None, padding=0):
        if False:
            i = 10
            return i + 15
        self.graphASres = ASres
        self.graphpadding = padding
        ips = {}
        rt = {}
        ports = {}
        ports_done = {}
        for (s, r) in self.res:
            r = r.getlayer(IP) or (conf.ipv6_enabled and r[scapy.layers.inet6.IPv6]) or r
            s = s.getlayer(IP) or (conf.ipv6_enabled and s[scapy.layers.inet6.IPv6]) or s
            ips[r.src] = None
            if TCP in s:
                trace_id = (s.src, s.dst, 6, s.dport)
            elif UDP in s:
                trace_id = (s.src, s.dst, 17, s.dport)
            elif ICMP in s:
                trace_id = (s.src, s.dst, 1, s.type)
            else:
                trace_id = (s.src, s.dst, s.proto, 0)
            trace = rt.get(trace_id, {})
            ttl = conf.ipv6_enabled and scapy.layers.inet6.IPv6 in s and s.hlim or s.ttl
            if not (ICMP in r and r[ICMP].type == 11) and (not (conf.ipv6_enabled and scapy.layers.inet6.IPv6 in r and (scapy.layers.inet6.ICMPv6TimeExceeded in r))):
                if trace_id in ports_done:
                    continue
                ports_done[trace_id] = None
                p = ports.get(r.src, [])
                if TCP in r:
                    p.append(r.sprintf('<T%ir,TCP.sport%> %TCP.sport% %TCP.flags%'))
                    trace[ttl] = r.sprintf('"%r,src%":T%ir,TCP.sport%')
                elif UDP in r:
                    p.append(r.sprintf('<U%ir,UDP.sport%> %UDP.sport%'))
                    trace[ttl] = r.sprintf('"%r,src%":U%ir,UDP.sport%')
                elif ICMP in r:
                    p.append(r.sprintf('<I%ir,ICMP.type%> ICMP %ICMP.type%'))
                    trace[ttl] = r.sprintf('"%r,src%":I%ir,ICMP.type%')
                else:
                    p.append(r.sprintf('{IP:<P%ir,proto%> IP %proto%}{IPv6:<P%ir,nh%> IPv6 %nh%}'))
                    trace[ttl] = r.sprintf('"%r,src%":{IP:P%ir,proto%}{IPv6:P%ir,nh%}')
                ports[r.src] = p
            else:
                trace[ttl] = r.sprintf('"%r,src%"')
            rt[trace_id] = trace
        unknown_label = incremental_label('unk%i')
        blackholes = []
        bhip = {}
        for rtk in rt:
            trace = rt[rtk]
            max_trace = max(trace)
            for n in range(min(trace), max_trace):
                if n not in trace:
                    trace[n] = next(unknown_label)
            if rtk not in ports_done:
                if rtk[2] == 1:
                    bh = '%s %i/icmp' % (rtk[1], rtk[3])
                elif rtk[2] == 6:
                    bh = '%s %i/tcp' % (rtk[1], rtk[3])
                elif rtk[2] == 17:
                    bh = '%s %i/udp' % (rtk[1], rtk[3])
                else:
                    bh = '%s %i/proto' % (rtk[1], rtk[2])
                ips[bh] = None
                bhip[rtk[1]] = bh
                bh = '"%s"' % bh
                trace[max_trace + 1] = bh
                blackholes.append(bh)
        ASN_query_list = set((x.rsplit(' ', 1)[0] for x in ips))
        if ASres is None:
            ASNlist = []
        else:
            ASNlist = ASres.resolve(*ASN_query_list)
        ASNs = {}
        ASDs = {}
        for (ip, asn, desc) in ASNlist:
            if asn is None:
                continue
            iplist = ASNs.get(asn, [])
            if ip in bhip:
                if ip in ports:
                    iplist.append(ip)
                iplist.append(bhip[ip])
            else:
                iplist.append(ip)
            ASNs[asn] = iplist
            ASDs[asn] = desc
        backcolorlist = colgen('60', '86', 'ba', 'ff')
        forecolorlist = colgen('a0', '70', '40', '20')
        s = 'digraph trace {\n'
        s += '\n\tnode [shape=ellipse,color=black,style=solid];\n\n'
        s += '\n#ASN clustering\n'
        for asn in ASNs:
            s += '\tsubgraph cluster_%s {\n' % asn
            col = next(backcolorlist)
            s += '\t\tcolor="#%s%s%s";' % col
            s += '\t\tnode [fillcolor="#%s%s%s",style=filled];' % col
            s += '\t\tfontsize = 10;'
            s += '\t\tlabel = "%s\\n[%s]"\n' % (asn, ASDs[asn])
            for ip in ASNs[asn]:
                s += '\t\t"%s";\n' % ip
            s += '\t}\n'
        s += '#endpoints\n'
        for p in ports:
            s += '\t"%s" [shape=record,color=black,fillcolor=green,style=filled,label="%s|%s"];\n' % (p, p, '|'.join(ports[p]))
        s += '\n#Blackholes\n'
        for bh in blackholes:
            s += '\t%s [shape=octagon,color=black,fillcolor=red,style=filled];\n' % bh
        if padding:
            s += '\n#Padding\n'
            pad = {}
            for (snd, rcv) in self.res:
                if rcv.src not in ports and rcv.haslayer(conf.padding_layer):
                    p = rcv.getlayer(conf.padding_layer).load
                    if p != b'\x00' * len(p):
                        pad[rcv.src] = None
            for rcv in pad:
                s += '\t"%s" [shape=triangle,color=black,fillcolor=red,style=filled];\n' % rcv
        s += '\n\tnode [shape=ellipse,color=black,style=solid];\n\n'
        for rtk in rt:
            s += '#---[%s\n' % repr(rtk)
            s += '\t\tedge [color="#%s%s%s"];\n' % next(forecolorlist)
            trace = rt[rtk]
            maxtrace = max(trace)
            for n in range(min(trace), maxtrace):
                s += '\t%s ->\n' % trace[n]
            s += '\t%s;\n' % trace[maxtrace]
        s += '}\n'
        self.graphdef = s

    def graph(self, ASres=conf.AS_resolver, padding=0, **kargs):
        if False:
            return 10
        'x.graph(ASres=conf.AS_resolver, other args):\n        ASres=None          : no AS resolver => no clustering\n        ASres=AS_resolver() : default whois AS resolver (riswhois.ripe.net)\n        ASres=AS_resolver_cymru(): use whois.cymru.com whois database\n        ASres=AS_resolver(server="whois.ra.net")\n        type: output type (svg, ps, gif, jpg, etc.), passed to dot\'s "-T" option  # noqa: E501\n        target: filename or redirect. Defaults pipe to Imagemagick\'s display program  # noqa: E501\n        prog: which graphviz program to use'
        if self.graphdef is None or self.graphASres != ASres or self.graphpadding != padding:
            self.make_graph(ASres, padding)
        return do_graph(self.graphdef, **kargs)

@conf.commands.register
def traceroute(target, dport=80, minttl=1, maxttl=30, sport=RandShort(), l4=None, filter=None, timeout=2, verbose=None, **kargs):
    if False:
        for i in range(10):
            print('nop')
    'Instant TCP traceroute\n\n       :param target:  hostnames or IP addresses\n       :param dport:   TCP destination port (default is 80)\n       :param minttl:  minimum TTL (default is 1)\n       :param maxttl:  maximum TTL (default is 30)\n       :param sport:   TCP source port (default is random)\n       :param l4:      use a Scapy packet instead of TCP\n       :param filter:  BPF filter applied to received packets\n       :param timeout: time to wait for answers (default is 2s)\n       :param verbose: detailed output\n       :return: an TracerouteResult, and a list of unanswered packets'
    if verbose is None:
        verbose = conf.verb
    if filter is None:
        filter = '(icmp and (icmp[0]=3 or icmp[0]=4 or icmp[0]=5 or icmp[0]=11 or icmp[0]=12)) or (tcp and (tcp[13] & 0x16 > 0x10))'
    if l4 is None:
        (a, b) = sr(IP(dst=target, id=RandShort(), ttl=(minttl, maxttl)) / TCP(seq=RandInt(), sport=sport, dport=dport), timeout=timeout, filter=filter, verbose=verbose, **kargs)
    else:
        filter = 'ip'
        (a, b) = sr(IP(dst=target, id=RandShort(), ttl=(minttl, maxttl)) / l4, timeout=timeout, filter=filter, verbose=verbose, **kargs)
    a = TracerouteResult(a.res)
    if verbose:
        a.show()
    return (a, b)

@conf.commands.register
def traceroute_map(ips, **kargs):
    if False:
        return 10
    'Util function to call traceroute on multiple targets, then\n    show the different paths on a map.\n\n    :param ips: a list of IPs on which traceroute will be called\n    :param kargs: (optional) kwargs, passed to traceroute\n    '
    kargs.setdefault('verbose', 0)
    return traceroute(ips)[0].world_trace()

class TCP_client(Automaton):
    """
    Creates a TCP Client Automaton.
    This automaton will handle TCP 3-way handshake.

    Usage: the easiest usage is to use it as a SuperSocket.
        >>> a = TCP_client.tcplink(HTTP, "www.google.com", 80)
        >>> a.send(HTTPRequest())
        >>> a.recv()

    :param ip: the ip to connect to
    :param port:
    :param src: (optional) use another source IP
    """

    def parse_args(self, ip, port, srcip=None, **kargs):
        if False:
            for i in range(10):
                print('nop')
        from scapy.sessions import TCPSession
        self.dst = str(Net(ip))
        self.dport = port
        self.sport = random.randrange(0, 2 ** 16)
        self.l4 = IP(dst=ip, src=srcip) / TCP(sport=self.sport, dport=self.dport, flags=0, seq=random.randrange(0, 2 ** 32))
        self.src = self.l4.src
        self.sack = self.l4[TCP].ack
        self.rel_seq = None
        self.rcvbuf = TCPSession()
        bpf = 'host %s  and host %s and port %i and port %i' % (self.src, self.dst, self.sport, self.dport)
        Automaton.parse_args(self, filter=bpf, **kargs)

    def _transmit_packet(self, pkt):
        if False:
            return 10
        'Transmits a packet from TCPSession to the SuperSocket'
        self.oi.tcp.send(raw(pkt[TCP].payload))

    def master_filter(self, pkt):
        if False:
            print('Hello World!')
        return IP in pkt and pkt[IP].src == self.dst and (pkt[IP].dst == self.src) and (TCP in pkt) and (pkt[TCP].sport == self.dport) and (pkt[TCP].dport == self.sport) and (self.l4[TCP].seq >= pkt[TCP].ack) and (self.l4[TCP].ack == 0 or self.sack <= pkt[TCP].seq <= self.l4[TCP].ack + pkt[TCP].window)

    @ATMT.state(initial=1)
    def START(self):
        if False:
            while True:
                i = 10
        pass

    @ATMT.state()
    def SYN_SENT(self):
        if False:
            print('Hello World!')
        pass

    @ATMT.state()
    def ESTABLISHED(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @ATMT.state()
    def LAST_ACK(self):
        if False:
            while True:
                i = 10
        pass

    @ATMT.state(final=1)
    def CLOSED(self):
        if False:
            return 10
        pass

    @ATMT.state(stop=1)
    def STOP(self):
        if False:
            while True:
                i = 10
        pass

    @ATMT.state()
    def STOP_SENT_FIN_ACK(self):
        if False:
            while True:
                i = 10
        pass

    @ATMT.condition(START)
    def connect(self):
        if False:
            i = 10
            return i + 15
        raise self.SYN_SENT()

    @ATMT.action(connect)
    def send_syn(self):
        if False:
            print('Hello World!')
        self.l4[TCP].flags = 'S'
        self.send(self.l4)
        self.l4[TCP].seq += 1

    @ATMT.receive_condition(SYN_SENT)
    def synack_received(self, pkt):
        if False:
            for i in range(10):
                print('nop')
        if pkt[TCP].flags.SA:
            raise self.ESTABLISHED().action_parameters(pkt)

    @ATMT.action(synack_received)
    def send_ack_of_synack(self, pkt):
        if False:
            return 10
        self.l4[TCP].ack = pkt[TCP].seq + 1
        self.l4[TCP].flags = 'A'
        self.send(self.l4)

    @ATMT.receive_condition(ESTABLISHED)
    def incoming_data_received(self, pkt):
        if False:
            while True:
                i = 10
        if not isinstance(pkt[TCP].payload, (NoPayload, conf.padding_layer)):
            raise self.ESTABLISHED().action_parameters(pkt)

    @ATMT.action(incoming_data_received)
    def receive_data(self, pkt):
        if False:
            return 10
        data = raw(pkt[TCP].payload)
        if data and self.l4[TCP].ack == pkt[TCP].seq:
            self.sack = self.l4[TCP].ack
            self.l4[TCP].ack += len(data)
            self.l4[TCP].flags = 'A'
            self.send(self.l4)
            self._transmit_packet(self.rcvbuf.process(pkt))

    @ATMT.ioevent(ESTABLISHED, name='tcp', as_supersocket='tcplink')
    def outgoing_data_received(self, fd):
        if False:
            for i in range(10):
                print('nop')
        raise self.ESTABLISHED().action_parameters(fd.recv())

    @ATMT.action(outgoing_data_received)
    def send_data(self, d):
        if False:
            print('Hello World!')
        self.l4[TCP].flags = 'PA'
        self.send(self.l4 / d)
        self.l4[TCP].seq += len(d)

    @ATMT.receive_condition(ESTABLISHED)
    def reset_received(self, pkt):
        if False:
            return 10
        if pkt[TCP].flags.R:
            raise self.CLOSED()

    @ATMT.receive_condition(ESTABLISHED)
    def fin_received(self, pkt):
        if False:
            while True:
                i = 10
        if pkt[TCP].flags.F:
            raise self.LAST_ACK().action_parameters(pkt)

    @ATMT.action(fin_received)
    def send_finack(self, pkt):
        if False:
            i = 10
            return i + 15
        self.l4[TCP].flags = 'FA'
        self.l4[TCP].ack = pkt[TCP].seq + 1
        self.send(self.l4)
        self.l4[TCP].seq += 1

    @ATMT.receive_condition(LAST_ACK)
    def ack_of_fin_received(self, pkt):
        if False:
            print('Hello World!')
        if pkt[TCP].flags.A:
            raise self.CLOSED()

    @ATMT.condition(STOP)
    def stop_requested(self):
        if False:
            return 10
        raise self.STOP_SENT_FIN_ACK()

    @ATMT.action(stop_requested)
    def stop_send_finack(self):
        if False:
            for i in range(10):
                print('nop')
        self.l4[TCP].flags = 'FA'
        self.send(self.l4)
        self.l4[TCP].seq += 1

    @ATMT.receive_condition(STOP_SENT_FIN_ACK)
    def stop_fin_received(self, pkt):
        if False:
            for i in range(10):
                print('nop')
        if pkt[TCP].flags.F:
            raise self.CLOSED().action_parameters(pkt)

    @ATMT.action(stop_fin_received)
    def stop_send_ack(self, pkt):
        if False:
            for i in range(10):
                print('nop')
        self.l4[TCP].flags = 'A'
        self.l4[TCP].ack = pkt[TCP].seq + 1
        self.send(self.l4)

    @ATMT.timeout(SYN_SENT, 1)
    def syn_ack_timeout(self):
        if False:
            while True:
                i = 10
        raise self.CLOSED()

    @ATMT.timeout(STOP_SENT_FIN_ACK, 1)
    def stop_ack_timeout(self):
        if False:
            while True:
                i = 10
        raise self.CLOSED()

@conf.commands.register
def report_ports(target, ports):
    if False:
        i = 10
        return i + 15
    'portscan a target and output a LaTeX table\nreport_ports(target, ports) -> string'
    (ans, unans) = sr(IP(dst=target) / TCP(dport=ports), timeout=5)
    rep = '\\begin{tabular}{|r|l|l|}\n\\hline\n'
    for (s, r) in ans:
        if not r.haslayer(ICMP):
            if r.payload.flags == 18:
                rep += r.sprintf('%TCP.sport% & open & SA \\\\\n')
    rep += '\\hline\n'
    for (s, r) in ans:
        if r.haslayer(ICMP):
            rep += r.sprintf('%TCPerror.dport% & closed & ICMP type %ICMP.type%/%ICMP.code% from %IP.src% \\\\\n')
        elif r.payload.flags != 18:
            rep += r.sprintf('%TCP.sport% & closed & TCP %TCP.flags% \\\\\n')
    rep += '\\hline\n'
    for i in unans:
        rep += i.sprintf('%TCP.dport% & ? & unanswered \\\\\n')
    rep += '\\hline\n\\end{tabular}\n'
    return rep

@conf.commands.register
def IPID_count(lst, funcID=lambda x: x[1].id, funcpres=lambda x: x[1].summary()):
    if False:
        i = 10
        return i + 15
    'Identify IP id values classes in a list of packets\n\nlst:      a list of packets\nfuncID:   a function that returns IP id values\nfuncpres: a function used to summarize packets'
    idlst = [funcID(e) for e in lst]
    idlst.sort()
    classes = [idlst[0]]
    classes += [t[1] for t in zip(idlst[:-1], idlst[1:]) if abs(t[0] - t[1]) > 50]
    lst = [(funcID(x), funcpres(x)) for x in lst]
    lst.sort()
    print('Probably %i classes: %s' % (len(classes), classes))
    for (id, pr) in lst:
        print('%5i' % id, pr)

@conf.commands.register
def fragleak(target, sport=123, dport=123, timeout=0.2, onlyasc=0, count=None):
    if False:
        while True:
            i = 10
    load = 'XXXXYYYYYYYYYY'
    pkt = IP(dst=target, id=RandShort(), options=b'\x00' * 40, flags=1)
    pkt /= UDP(sport=sport, dport=sport) / load
    s = conf.L3socket()
    intr = 0
    found = {}
    try:
        while count is None or count:
            if count is not None and isinstance(count, int):
                count -= 1
            try:
                if not intr:
                    s.send(pkt)
                sin = select.select([s], [], [], timeout)[0]
                if not sin:
                    continue
                ans = s.recv(1600)
                if not isinstance(ans, IP):
                    continue
                if not isinstance(ans.payload, ICMP):
                    continue
                if not isinstance(ans.payload.payload, IPerror):
                    continue
                if ans.payload.payload.dst != target:
                    continue
                if ans.src != target:
                    print('leak from', ans.src)
                if not ans.haslayer(conf.padding_layer):
                    continue
                leak = ans.getlayer(conf.padding_layer).load
                if leak not in found:
                    found[leak] = None
                    linehexdump(leak, onlyasc=onlyasc)
            except KeyboardInterrupt:
                if intr:
                    raise
                intr = 1
    except KeyboardInterrupt:
        pass

@conf.commands.register
def fragleak2(target, timeout=0.4, onlyasc=0, count=None):
    if False:
        for i in range(10):
            print('nop')
    found = {}
    try:
        while count is None or count:
            if count is not None and isinstance(count, int):
                count -= 1
            pkt = IP(dst=target, options=b'\x00' * 40, proto=200)
            pkt /= 'XXXXYYYYYYYYYYYY'
            p = sr1(pkt, timeout=timeout, verbose=0)
            if not p:
                continue
            if conf.padding_layer in p:
                leak = p[conf.padding_layer].load
                if leak not in found:
                    found[leak] = None
                    linehexdump(leak, onlyasc=onlyasc)
    except Exception:
        pass

@conf.commands.register
class connect_from_ip:
    """
    Open a TCP socket to a host:port while spoofing another IP.

    :param host: the host to connect to
    :param port: the port to connect to
    :param srcip: the IP to spoof. the cache of the gateway will
                  be poisonned with this IP.
    :param poison: (optional, default True) ARP poison the gateway (or next hop),
                   so that it answers us.
    :param timeout: (optional) the socket timeout.

    Example - Connect to 192.168.0.1:80 spoofing 192.168.0.2::

        from scapy.layers.http import HTTP, HTTPRequest
        client = connect_from_ip("192.168.0.1", 80, "192.168.0.2")
        sock = SSLStreamSocket(client.sock, HTTP)
        resp = sock.sr1(HTTP() / HTTPRequest(Path="/"))

    Example - Connect to 192.168.0.1:443 with TLS wrapping spoofing 192.168.0.2::

        import ssl
        from scapy.layers.http import HTTP, HTTPRequest
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        client = connect_from_ip("192.168.0.1", 443, "192.168.0.2")
        sock = context.wrap_socket(client.sock)
        sock = SSLStreamSocket(client.sock, HTTP)
        resp = sock.sr1(HTTP() / HTTPRequest(Path="/"))
    """

    def __init__(self, host, port, srcip, poison=True, timeout=1):
        if False:
            i = 10
            return i + 15
        host = str(Net(host))
        if poison:
            gateway = conf.route.route(host)[2]
            if gateway == '0.0.0.0':
                gateway = host
            arpcachepoison(gateway, srcip, count=1, interval=0, verbose=0)
        (self._sock, self.sock) = socket.socketpair()
        self.sock.settimeout(timeout)
        self.client = TCP_client(host, port, srcip=srcip, external_fd={'tcp': self._sock})
        self.client.runbg()

    def close(self):
        if False:
            i = 10
            return i + 15
        self.client.stop()
        self.client.destroy()
        self.sock.close()
        self._sock.close()

class ICMPEcho_am(AnsweringMachine):
    """Responds to ICMP Echo-Requests (ping)"""
    function_name = 'icmpechod'

    def is_request(self, req):
        if False:
            print('Hello World!')
        if req.haslayer(ICMP):
            icmp_req = req.getlayer(ICMP)
            if icmp_req.type == 8:
                return True
        return False

    def print_reply(self, req, reply):
        if False:
            print('Hello World!')
        print('Replying %s to %s' % (reply.getlayer(IP).dst, req.dst))

    def make_reply(self, req):
        if False:
            i = 10
            return i + 15
        reply = IP(dst=req[IP].src) / ICMP()
        reply[ICMP].type = 0
        reply[ICMP].seq = req[ICMP].seq
        reply[ICMP].id = req[ICMP].id
        reply[ICMP].chksum = None
        return reply
conf.stats_classic_protocols += [TCP, UDP, ICMP]
conf.stats_dot11_protocols += [TCP, UDP, ICMP]
if conf.ipv6_enabled:
    import scapy.layers.inet6