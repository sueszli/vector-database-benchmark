"""
    EIGRP Scapy Extension
    ~~~~~~~~~~~~~~~~~~~~~

    :version:   2009-08-13
    :e-mail:    lobo@c3a.de / jochen.bartl@gmail.com

    :TODO

    - Replace TLV code with a more generic solution
        * http://trac.secdev.org/scapy/ticket/90
    - Write function for calculating authentication data

    :Thanks:

    - TLV code derived from the CDP implementation of scapy. (Thanks to Nicolas Bareil and Arnaud Ebalard)
        http://trac.secdev.org/scapy/ticket/18
    - IOS / EIGRP Version Representation FIX by Dirk Loss
"""
import socket
import struct
from scapy.packet import Packet
from scapy.fields import StrField, IPField, XShortField, FieldLenField, StrLenField, IntField, ByteEnumField, ByteField, ConditionalField, FlagsField, IP6Field, PacketListField, ShortEnumField, ShortField, StrFixedLenField, ThreeBytesField
from scapy.layers.inet import IP, checksum, bind_layers
from scapy.layers.inet6 import IPv6
from scapy.compat import chb
from scapy.config import conf
from scapy.utils import inet_aton, inet_ntoa
from scapy.pton_ntop import inet_ntop, inet_pton
from scapy.error import warning, Scapy_Exception
from scapy.volatile import RandShort, RandString

class EigrpIPField(StrField, IPField):
    """
    This is a special field type for handling ip addresses of destination networks in internal and
    external route updates.

    EIGRP removes zeros from the host portion of the ip address if the netmask is 8, 16 or 24 bits.
    """
    __slots__ = ['length_from']

    def __init__(self, name, default, length=None, length_from=None):
        if False:
            return 10
        StrField.__init__(self, name, default)
        self.length_from = length_from
        if length is not None:
            self.length_from = lambda pkt, length=length: length

    def h2i(self, pkt, x):
        if False:
            print('Hello World!')
        return IPField.h2i(self, pkt, x)

    def i2m(self, pkt, x):
        if False:
            while True:
                i = 10
        x = inet_aton(x)
        tmp_len = self.length_from(pkt)
        if tmp_len <= 8:
            return x[:1]
        elif tmp_len <= 16:
            return x[:2]
        elif tmp_len <= 24:
            return x[:3]
        else:
            return x

    def m2i(self, pkt, x):
        if False:
            print('Hello World!')
        tmp_len = self.length_from(pkt)
        if tmp_len <= 8:
            x += b'\x00\x00\x00'
        elif tmp_len <= 16:
            x += b'\x00\x00'
        elif tmp_len <= 24:
            x += b'\x00'
        return inet_ntoa(x)

    def prefixlen_to_bytelen(self, tmp_len):
        if False:
            for i in range(10):
                print('nop')
        if tmp_len <= 8:
            tmp_len = 1
        elif tmp_len <= 16:
            tmp_len = 2
        elif tmp_len <= 24:
            tmp_len = 3
        else:
            tmp_len = 4
        return tmp_len

    def i2len(self, pkt, x):
        if False:
            for i in range(10):
                print('nop')
        tmp_len = self.length_from(pkt)
        tmp_len = self.prefixlen_to_bytelen(tmp_len)
        return tmp_len

    def getfield(self, pkt, s):
        if False:
            while True:
                i = 10
        tmp_len = self.length_from(pkt)
        tmp_len = self.prefixlen_to_bytelen(tmp_len)
        return (s[tmp_len:], self.m2i(pkt, s[:tmp_len]))

    def randval(self):
        if False:
            i = 10
            return i + 15
        return IPField.randval(self)

class EigrpIP6Field(StrField, IP6Field):
    """
    This is a special field type for handling ip addresses of destination networks in internal and
    external route updates.

    """
    __slots__ = ['length_from']

    def __init__(self, name, default, length=None, length_from=None):
        if False:
            print('Hello World!')
        StrField.__init__(self, name, default)
        self.length_from = length_from
        if length is not None:
            self.length_from = lambda pkt, length=length: length

    def any2i(self, pkt, x):
        if False:
            i = 10
            return i + 15
        return IP6Field.any2i(self, pkt, x)

    def i2repr(self, pkt, x):
        if False:
            while True:
                i = 10
        return IP6Field.i2repr(self, pkt, x)

    def h2i(self, pkt, x):
        if False:
            for i in range(10):
                print('nop')
        return IP6Field.h2i(self, pkt, x)

    def i2m(self, pkt, x):
        if False:
            i = 10
            return i + 15
        x = inet_pton(socket.AF_INET6, x)
        tmp_len = self.length_from(pkt)
        tmp_len = self.prefixlen_to_bytelen(tmp_len)
        return x[:tmp_len]

    def m2i(self, pkt, x):
        if False:
            for i in range(10):
                print('nop')
        tmp_len = self.length_from(pkt)
        prefixlen = self.prefixlen_to_bytelen(tmp_len)
        if tmp_len > 128:
            warning('EigrpIP6Field: Prefix length is > 128. Dissection of this packet will fail')
        else:
            pad = b'\x00' * (16 - prefixlen)
            x += pad
        return inet_ntop(socket.AF_INET6, x)

    def prefixlen_to_bytelen(self, plen):
        if False:
            i = 10
            return i + 15
        plen = plen // 8
        if plen < 16:
            plen += 1
        return plen

    def i2len(self, pkt, x):
        if False:
            print('Hello World!')
        tmp_len = self.length_from(pkt)
        tmp_len = self.prefixlen_to_bytelen(tmp_len)
        return tmp_len

    def getfield(self, pkt, s):
        if False:
            print('Hello World!')
        tmp_len = self.length_from(pkt)
        tmp_len = self.prefixlen_to_bytelen(tmp_len)
        return (s[tmp_len:], self.m2i(pkt, s[:tmp_len]))

    def randval(self):
        if False:
            for i in range(10):
                print('nop')
        return IP6Field.randval(self)

class EIGRPGeneric(Packet):
    name = 'EIGRP Generic TLV'
    fields_desc = [XShortField('type', 0), FieldLenField('len', None, 'value', '!H', adjust=lambda pkt, x: x + 4), StrLenField('value', b'\x00', length_from=lambda pkt: pkt.len - 4)]

    def guess_payload_class(self, p):
        if False:
            i = 10
            return i + 15
        return conf.padding_layer

class EIGRPParam(EIGRPGeneric):
    name = 'EIGRP Parameters'
    fields_desc = [XShortField('type', 1), ShortField('len', 12), ByteField('k1', 1), ByteField('k2', 0), ByteField('k3', 1), ByteField('k4', 0), ByteField('k5', 0), ByteField('reserved', 0), ShortField('holdtime', 15)]

class EIGRPAuthData(EIGRPGeneric):
    name = 'EIGRP Authentication Data'
    fields_desc = [XShortField('type', 2), FieldLenField('len', None, 'authdata', '!H', adjust=lambda pkt, x: x + 24), ShortEnumField('authtype', 2, {2: 'MD5'}), ShortField('keysize', None), IntField('keyid', 1), StrFixedLenField('nullpad', b'\x00' * 12, 12), StrLenField('authdata', RandString(16), length_from=lambda pkt: pkt.keysize)]

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        p += pay
        if self.keysize is None:
            keysize = len(self.authdata)
            p = p[:6] + chb(keysize >> 8 & 255) + chb(keysize & 255) + p[8:]
        return p

class EIGRPSeq(EIGRPGeneric):
    name = 'EIGRP Sequence'
    fields_desc = [XShortField('type', 3), ShortField('len', None), ByteField('addrlen', 4), ConditionalField(IPField('ipaddr', '192.168.0.1'), lambda pkt: pkt.addrlen == 4), ConditionalField(IP6Field('ip6addr', '2001::'), lambda pkt: pkt.addrlen == 16)]

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        p += pay
        if self.len is None:
            tmp_len = len(p)
            tmp_p = p[:2] + chb(tmp_len >> 8 & 255)
            p = tmp_p + chb(tmp_len & 255) + p[4:]
        return p

class ShortVersionField(ShortField):

    def i2repr(self, pkt, x):
        if False:
            for i in range(10):
                print('nop')
        try:
            minor = x & 255
            major = x >> 8 & 255
        except TypeError:
            return 'unknown'
        else:
            return 'v%s.%s' % (major, minor)

    def h2i(self, pkt, x):
        if False:
            while True:
                i = 10
        'The field accepts string values like v12.1, v1.1 or integer values.\n           String values have to start with a "v" followed by a\n           floating point number. Valid numbers are between 0 and 255.\n\n        '
        if isinstance(x, str) and x.startswith('v') and (len(x) <= 8):
            major = int(x.split('.')[0][1:])
            minor = int(x.split('.')[1])
            return major << 8 | minor
        elif isinstance(x, int) and 0 <= x <= 65535:
            return x
        else:
            if not hasattr(self, 'default'):
                return x
            if self.default is not None:
                warning('set value to default. Format of %r is invalid', x)
                return self.default
            else:
                raise Scapy_Exception('Format of value is invalid')

    def randval(self):
        if False:
            while True:
                i = 10
        return RandShort()

class EIGRPSwVer(EIGRPGeneric):
    name = 'EIGRP Software Version'
    fields_desc = [XShortField('type', 4), ShortField('len', 8), ShortVersionField('ios', 'v12.0'), ShortVersionField('eigrp', 'v1.2')]

class EIGRPNms(EIGRPGeneric):
    name = 'EIGRP Next Multicast Sequence'
    fields_desc = [XShortField('type', 5), ShortField('len', 8), IntField('nms', 2)]
_EIGRP_STUB_FLAGS = ['connected', 'static', 'summary', 'receive-only', 'redistributed', 'leak-map']

class EIGRPStub(EIGRPGeneric):
    name = 'EIGRP Stub Router'
    fields_desc = [XShortField('type', 6), ShortField('len', 6), FlagsField('flags', 13, 16, _EIGRP_STUB_FLAGS)]

class EIGRPIntRoute(EIGRPGeneric):
    name = 'EIGRP Internal Route'
    fields_desc = [XShortField('type', 258), FieldLenField('len', None, 'dst', '!H', adjust=lambda pkt, x: x + 25), IPField('nexthop', '192.168.0.0'), IntField('delay', 128000), IntField('bandwidth', 256), ThreeBytesField('mtu', 1500), ByteField('hopcount', 0), ByteField('reliability', 255), ByteField('load', 0), XShortField('reserved', 0), ByteField('prefixlen', 24), EigrpIPField('dst', '192.168.1.0', length_from=lambda pkt: pkt.prefixlen)]
_EIGRP_EXTERNAL_PROTOCOL_ID = {1: 'IGRP', 2: 'EIGRP', 3: 'Static Route', 4: 'RIP', 5: 'Hello', 6: 'OSPF', 7: 'IS-IS', 8: 'EGP', 9: 'BGP', 10: 'IDRP', 11: 'Connected Link'}
_EIGRP_EXTROUTE_FLAGS = ['external', 'candidate-default']

class EIGRPExtRoute(EIGRPGeneric):
    name = 'EIGRP External Route'
    fields_desc = [XShortField('type', 259), FieldLenField('len', None, 'dst', '!H', adjust=lambda pkt, x: x + 45), IPField('nexthop', '192.168.0.0'), IPField('originrouter', '192.168.0.1'), IntField('originasn', 0), IntField('tag', 0), IntField('externalmetric', 0), ShortField('reserved', 0), ByteEnumField('extprotocolid', 3, _EIGRP_EXTERNAL_PROTOCOL_ID), FlagsField('flags', 0, 8, _EIGRP_EXTROUTE_FLAGS), IntField('delay', 0), IntField('bandwidth', 256), ThreeBytesField('mtu', 1500), ByteField('hopcount', 0), ByteField('reliability', 255), ByteField('load', 0), XShortField('reserved2', 0), ByteField('prefixlen', 24), EigrpIPField('dst', '192.168.1.0', length_from=lambda pkt: pkt.prefixlen)]

class EIGRPv6IntRoute(EIGRPGeneric):
    name = 'EIGRP for IPv6 Internal Route'
    fields_desc = [XShortField('type', 1026), FieldLenField('len', None, 'dst', '!H', adjust=lambda pkt, x: x + 37), IP6Field('nexthop', '::'), IntField('delay', 128000), IntField('bandwidth', 256000), ThreeBytesField('mtu', 1500), ByteField('hopcount', 1), ByteField('reliability', 255), ByteField('load', 0), XShortField('reserved', 0), ByteField('prefixlen', 16), EigrpIP6Field('dst', '2001::', length_from=lambda pkt: pkt.prefixlen)]

class EIGRPv6ExtRoute(EIGRPGeneric):
    name = 'EIGRP for IPv6 External Route'
    fields_desc = [XShortField('type', 1027), FieldLenField('len', None, 'dst', '!H', adjust=lambda pkt, x: x + 57), IP6Field('nexthop', '::'), IPField('originrouter', '192.168.0.1'), IntField('originasn', 0), IntField('tag', 0), IntField('externalmetric', 0), ShortField('reserved', 0), ByteEnumField('extprotocolid', 3, _EIGRP_EXTERNAL_PROTOCOL_ID), FlagsField('flags', 0, 8, _EIGRP_EXTROUTE_FLAGS), IntField('delay', 0), IntField('bandwidth', 256000), ThreeBytesField('mtu', 1500), ByteField('hopcount', 1), ByteField('reliability', 0), ByteField('load', 1), XShortField('reserved2', 0), ByteField('prefixlen', 8), EigrpIP6Field('dst', '::', length_from=lambda pkt: pkt.prefixlen)]
_eigrp_tlv_cls = {1: 'EIGRPParam', 2: 'EIGRPAuthData', 3: 'EIGRPSeq', 4: 'EIGRPSwVer', 5: 'EIGRPNms', 6: 'EIGRPStub', 258: 'EIGRPIntRoute', 259: 'EIGRPExtRoute', 1026: 'EIGRPv6IntRoute', 1027: 'EIGRPv6ExtRoute'}

def _EIGRPGuessPayloadClass(p, **kargs):
    if False:
        return 10
    cls = conf.raw_layer
    if len(p) >= 2:
        t = struct.unpack('!H', p[:2])[0]
        clsname = _eigrp_tlv_cls.get(t, 'EIGRPGeneric')
        cls = globals()[clsname]
    return cls(p, **kargs)
_EIGRP_OPCODES = {1: 'Update', 2: 'Request', 3: 'Query', 4: 'Replay', 5: 'Hello', 6: 'IPX SAP', 10: 'SIA Query', 11: 'SIA Reply'}
_EIGRP_FLAGS = ['init', 'cond-recv', 'unknown', 'update']

class EIGRP(Packet):
    name = 'EIGRP'
    fields_desc = [ByteField('ver', 2), ByteEnumField('opcode', 5, _EIGRP_OPCODES), XShortField('chksum', None), FlagsField('flags', 0, 32, _EIGRP_FLAGS), IntField('seq', 0), IntField('ack', 0), IntField('asn', 100), PacketListField('tlvlist', [], _EIGRPGuessPayloadClass)]

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        p += pay
        if self.chksum is None:
            c = checksum(p)
            p = p[:2] + chb(c >> 8 & 255) + chb(c & 255) + p[4:]
        return p

    def mysummary(self):
        if False:
            i = 10
            return i + 15
        summarystr = 'EIGRP (AS=%EIGRP.asn% Opcode=%EIGRP.opcode%'
        if self.opcode == 5 and self.ack != 0:
            summarystr += ' (ACK)'
        if self.flags != 0:
            summarystr += ' Flags=%EIGRP.flags%'
        return self.sprintf(summarystr + ')')
bind_layers(IP, EIGRP, proto=88)
bind_layers(IPv6, EIGRP, nh=88)