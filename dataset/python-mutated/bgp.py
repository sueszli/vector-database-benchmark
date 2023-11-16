"""
BGP (Border Gateway Protocol).
"""
import struct
import re
import socket
from scapy import pton_ntop
from scapy.packet import Packet, Packet_metaclass, bind_layers
from scapy.fields import Field, BitField, BitEnumField, XBitField, ByteField, ByteEnumField, ShortField, ShortEnumField, IntField, IntEnumField, LongField, IEEEFloatField, StrField, StrLenField, StrFixedLenField, FieldLenField, FieldListField, PacketField, PacketListField, IPField, FlagsField, ConditionalField, MultiEnumField
from scapy.layers.inet import TCP
from scapy.layers.inet6 import IP6Field
from scapy.config import conf, ConfClass
from scapy.compat import orb, chb
from scapy.error import log_runtime

class BGPConf(ConfClass):
    """
    BGP module configuration.
    """
    use_2_bytes_asn = True
bgp_module_conf = BGPConf()
BGP_MAXIMUM_MESSAGE_SIZE = 4096
_BGP_HEADER_SIZE = 19
_BGP_HEADER_MARKER = b'\xff' * 16
_BGP_PA_EXTENDED_LENGTH = 16
_BGP_CAPABILITY_MIN_SIZE = 2
_BGP_PATH_ATTRIBUTE_MIN_SIZE = 3

def _bits_to_bytes_len(length_in_bits):
    if False:
        return 10
    '\n    Helper function that returns the numbers of bytes necessary to store the\n    given number of bits.\n    '
    return (length_in_bits + 7) // 8

class BGPFieldIPv4(Field):
    """
    IPv4 Field (CIDR)
    """

    def mask2iplen(self, mask):
        if False:
            while True:
                i = 10
        'Get the IP field mask length (in bytes).'
        return (mask + 7) // 8

    def h2i(self, pkt, h):
        if False:
            i = 10
            return i + 15
        'x.x.x.x/y to "internal" representation.'
        (ip, mask) = re.split('/', h)
        return (int(mask), ip)

    def i2h(self, pkt, i):
        if False:
            for i in range(10):
                print('nop')
        '"Internal" representation to "human" representation\n        (x.x.x.x/y).'
        (mask, ip) = i
        return ip + '/' + str(mask)

    def i2repr(self, pkt, i):
        if False:
            print('Hello World!')
        return self.i2h(pkt, i)

    def i2len(self, pkt, i):
        if False:
            i = 10
            return i + 15
        (mask, _) = i
        return self.mask2iplen(mask) + 1

    def i2m(self, pkt, i):
        if False:
            for i in range(10):
                print('nop')
        '"Internal" (IP as bytes, mask as int) to "machine"\n        representation.'
        (mask, ip) = i
        ip = socket.inet_aton(ip)
        return struct.pack('>B', mask) + ip[:self.mask2iplen(mask)]

    def addfield(self, pkt, s, val):
        if False:
            print('Hello World!')
        return s + self.i2m(pkt, val)

    def getfield(self, pkt, s):
        if False:
            return 10
        length = self.mask2iplen(orb(s[0])) + 1
        return (s[length:], self.m2i(pkt, s[:length]))

    def m2i(self, pkt, m):
        if False:
            return 10
        mask = orb(m[0])
        mask2iplen_res = self.mask2iplen(mask)
        ip = b''.join((m[i + 1:i + 2] if i < mask2iplen_res else b'\x00' for i in range(4)))
        return (mask, socket.inet_ntoa(ip))

class BGPFieldIPv6(Field):
    """IPv6 Field (CIDR)"""

    def mask2iplen(self, mask):
        if False:
            print('Hello World!')
        'Get the IP field mask length (in bytes).'
        return (mask + 7) // 8

    def h2i(self, pkt, h):
        if False:
            return 10
        'x.x.x.x/y to internal representation.'
        (ip, mask) = re.split('/', h)
        return (int(mask), ip)

    def i2h(self, pkt, i):
        if False:
            print('Hello World!')
        '"Internal" representation to "human" representation.'
        (mask, ip) = i
        return ip + '/' + str(mask)

    def i2repr(self, pkt, i):
        if False:
            print('Hello World!')
        return self.i2h(pkt, i)

    def i2len(self, pkt, i):
        if False:
            for i in range(10):
                print('nop')
        (mask, ip) = i
        return self.mask2iplen(mask) + 1

    def i2m(self, pkt, i):
        if False:
            while True:
                i = 10
        '"Internal" (IP as bytes, mask as int) to "machine" representation.'
        (mask, ip) = i
        ip = pton_ntop.inet_pton(socket.AF_INET6, ip)
        return struct.pack('>B', mask) + ip[:self.mask2iplen(mask)]

    def addfield(self, pkt, s, val):
        if False:
            i = 10
            return i + 15
        return s + self.i2m(pkt, val)

    def getfield(self, pkt, s):
        if False:
            i = 10
            return i + 15
        length = self.mask2iplen(orb(s[0])) + 1
        return (s[length:], self.m2i(pkt, s[:length]))

    def m2i(self, pkt, m):
        if False:
            print('Hello World!')
        mask = orb(m[0])
        ip = b''.join((m[i + 1:i + 2] if i < self.mask2iplen(mask) else b'\x00' for i in range(16)))
        return (mask, pton_ntop.inet_ntop(socket.AF_INET6, ip))

def has_extended_length(flags):
    if False:
        print('Hello World!')
    '\n    Used in BGPPathAttr to check if the extended-length flag is\n    set.\n    '
    return flags & _BGP_PA_EXTENDED_LENGTH == _BGP_PA_EXTENDED_LENGTH

def detect_add_path_prefix46(s, max_bit_length):
    if False:
        print('Hello World!')
    "\n    Detect IPv4/IPv6 prefixes conform to BGP Additional Path but NOT conform\n    to standard BGP..\n\n    This is an adapted version of wireshark's detect_add_path_prefix46\n    https://github.com/wireshark/wireshark/blob/ed9e958a2ed506220fdab320738f1f96a3c2ffbb/epan/dissectors/packet-bgp.c#L2905\n    Kudos to them !\n    "
    i = 0
    while i + 4 < len(s):
        i += 4
        prefix_len = orb(s[i])
        if prefix_len > max_bit_length:
            return False
        addr_len = (prefix_len + 7) // 8
        i += 1 + addr_len
        if i > len(s):
            return False
        if prefix_len % 8:
            if orb(s[i - 1]) & 255 >> prefix_len % 8:
                return False
    i = 0
    while i + 4 < len(s):
        prefix_len = orb(s[i])
        if prefix_len == 0 and len(s) > 1:
            return True
        if prefix_len > max_bit_length:
            return True
        addr_len = (prefix_len + 7) // 8
        i += 1 + addr_len
        if i > len(s):
            return True
        if prefix_len % 8:
            if orb(s[i - 1]) & 255 >> prefix_len % 8:
                return True
    return False

class BGPNLRI_IPv4(Packet):
    """
    Packet handling IPv4 NLRI fields.
    """
    name = 'IPv4 NLRI'
    fields_desc = [BGPFieldIPv4('prefix', '0.0.0.0/0')]

    def default_payload_class(self, payload):
        if False:
            while True:
                i = 10
        return conf.padding_layer

class BGPNLRI_IPv6(Packet):
    """
    Packet handling IPv6 NLRI fields.
    """
    name = 'IPv6 NLRI'
    fields_desc = [BGPFieldIPv6('prefix', '::/0')]

    def default_payload_class(self, payload):
        if False:
            while True:
                i = 10
        return conf.padding_layer

class BGPNLRI_IPv4_AP(BGPNLRI_IPv4):
    """
    Packet handling IPv4 NLRI fields WITH BGP ADDITIONAL PATH
    """
    name = 'IPv4 NLRI (Additional Path)'
    fields_desc = [IntField('nlri_path_id', 0), BGPFieldIPv4('prefix', '0.0.0.0/0')]

class BGPNLRI_IPv6_AP(BGPNLRI_IPv6):
    """
    Packet handling IPv6 NLRI fields WITH BGP ADDITIONAL PATH
    """
    name = 'IPv6 NLRI (Additional Path)'
    fields_desc = [IntField('nlri_path_id', 0), BGPFieldIPv6('prefix', '::/0')]

class BGPNLRIPacketListField(PacketListField):
    """
    PacketListField handling NLRI fields.
    """
    __slots__ = ['max_bit_length', 'cls_group', 'no_length']

    def __init__(self, name, default, ip_mode, **kwargs):
        if False:
            print('Hello World!')
        super(BGPNLRIPacketListField, self).__init__(name, default, Packet, **kwargs)
        (self.max_bit_length, self.cls_group) = {'IPv4': (32, [BGPNLRI_IPv4, BGPNLRI_IPv4_AP]), 'IPv6': (128, [BGPNLRI_IPv6, BGPNLRI_IPv6_AP])}[ip_mode]
        self.no_length = 'length_from' not in kwargs

    def getfield(self, pkt, s):
        if False:
            print('Hello World!')
        if self.no_length:
            index = s.find(_BGP_HEADER_MARKER)
            if index == 0:
                return (s, [])
            if index != -1:
                self.length_from = lambda pkt: index
        remain = s[:self.length_from(pkt)] if self.length_from else s
        cls = self.cls_group[detect_add_path_prefix46(remain, self.max_bit_length)]
        self.next_cls_cb = lambda *args: cls
        res = super(BGPNLRIPacketListField, self).getfield(pkt, s)
        if self.no_length:
            self.length_from = None
        return res

class _BGPInvalidDataException(Exception):
    """
    Raised when it is not possible to instantiate a BGP packet with the given
    data.
    """

    def __init__(self, details):
        if False:
            print('Hello World!')
        Exception.__init__(self, 'Impossible to build packet from the given data' + details)

def _get_cls(name, fallback_cls=conf.raw_layer):
    if False:
        while True:
            i = 10
    '\n    Returns class named "name" if it exists, fallback_cls otherwise.\n    '
    return globals().get(name, fallback_cls)
_bgp_message_types = {0: 'NONE', 1: 'OPEN', 2: 'UPDATE', 3: 'NOTIFICATION', 4: 'KEEPALIVE', 5: 'ROUTE-REFRESH'}
address_family_identifiers = {0: 'Reserved', 1: 'IP (IP version 4)', 2: 'IP6 (IP version 6)', 3: 'NSAP', 4: 'HDLC (8-bit multidrop)', 5: 'BBN 1822', 6: '802 (includes all 802 media plus Ethernet "canonical format")', 7: 'E.163', 8: 'E.164 (SMDS, Frame Relay, ATM)', 9: 'F.69 (Telex)', 10: 'X.121 (X.25, Frame Relay)', 11: 'IPX', 12: 'Appletalk', 13: 'Decnet IV', 14: 'Banyan Vines', 15: 'E.164 with NSAP format subaddress', 16: 'DNS (Domain Name System)', 17: 'Distinguished Name', 18: 'AS Number', 19: 'XTP over IP version 4', 20: 'XTP over IP version 6', 21: 'XTP native mode XTP', 22: 'Fibre Channel World-Wide Port Name', 23: 'Fibre Channel World-Wide Node Name', 24: 'GWID', 25: 'AFI for L2VPN information', 26: 'MPLS-TP Section Endpoint Identifier', 27: 'MPLS-TP LSP Endpoint Identifier', 28: 'MPLS-TP Pseudowire Endpoint Identifier', 29: 'MT IP: Multi-Topology IP version 4', 30: 'MT IPv6: Multi-Topology IP version 6', 16384: 'EIGRP Common Service Family', 16385: 'EIGRP IPv4 Service Family', 16386: 'EIGRP IPv6 Service Family', 16387: 'LISP Canonical Address Format (LCAF)', 16388: 'BGP-LS', 16389: '48-bit MAC', 16390: '64-bit MAC', 16391: 'OUI', 16392: 'MAC/24', 16393: 'MAC/40', 16394: 'IPv6/64', 16395: 'RBridge Port ID', 16396: 'TRILL Nickname', 65535: 'Reserved'}
subsequent_afis = {0: 'Reserved', 1: 'Network Layer Reachability Information used for unicast forwarding', 2: 'Network Layer Reachability Information used for multicast forwarding', 3: 'Reserved', 4: 'Network Layer Reachability Information (NLRI) with MPLS Labels', 5: 'MCAST-VPN', 6: 'Network Layer Reachability Information used for Dynamic Placement of        Multi-Segment Pseudowires', 7: 'Encapsulation SAFI', 8: 'MCAST-VPLS', 64: 'Tunnel SAFI', 65: 'Virtual Private LAN Service (VPLS)', 66: 'BGP MDT SAFI', 67: 'BGP 4over6 SAFI', 68: 'BGP 6over4 SAFI', 69: 'Layer-1 VPN auto-discovery information', 70: 'BGP EVPNs', 71: 'BGP-LS', 72: 'BGP-LS-VPN', 128: 'MPLS-labeled VPN address', 129: 'Multicast for BGP/MPLS IP Virtual Private Networks (VPNs)', 132: 'Route Target constraint', 133: 'IPv4 dissemination of flow specification rules', 134: 'VPNv4 dissemination of flow specification rules', 140: 'VPN auto-discovery', 255: 'Reserved'}
_bgp_cls_by_type = {1: 'BGPOpen', 2: 'BGPUpdate', 3: 'BGPNotification', 4: 'BGPKeepAlive', 5: 'BGPRouteRefresh'}

class BGPHeader(Packet):
    """
    The header of any BGP message.
    References: RFC 4271
    """
    name = 'HEADER'
    fields_desc = [XBitField('marker', 340282366920938463463374607431768211455, 128), ShortField('len', None), ByteEnumField('type', 4, _bgp_message_types)]

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            return 10
        '\n        Returns the right class for the given data.\n        '
        return _bgp_dispatcher(_pkt)

    def post_build(self, p, pay):
        if False:
            while True:
                i = 10
        if self.len is None:
            length = len(p)
            if pay:
                length = length + len(pay)
            p = p[:16] + struct.pack('!H', length) + p[18:]
        return p + pay

    def guess_payload_class(self, payload):
        if False:
            while True:
                i = 10
        return _get_cls(_bgp_cls_by_type.get(self.type, conf.raw_layer), conf.raw_layer)

def _bgp_dispatcher(payload):
    if False:
        i = 10
        return i + 15
    '\n    Returns the right class for a given BGP message.\n    '
    cls = conf.raw_layer
    if payload is None:
        cls = _get_cls('BGPHeader', conf.raw_layer)
    elif len(payload) >= _BGP_HEADER_SIZE and payload[:16] == _BGP_HEADER_MARKER:
        message_type = orb(payload[18])
        if message_type == 4:
            cls = _get_cls('BGPKeepAlive')
        else:
            cls = _get_cls('BGPHeader')
    return cls

class BGP(Packet):
    """
    Every BGP message inherits from this class.
    """
    OPEN_TYPE = 1
    UPDATE_TYPE = 2
    NOTIFICATION_TYPE = 3
    KEEPALIVE_TYPE = 4
    ROUTEREFRESH_TYPE = 5

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            print('Hello World!')
        '\n        Returns the right class for the given data.\n        '
        return _bgp_dispatcher(_pkt)

    def guess_payload_class(self, p):
        if False:
            for i in range(10):
                print('nop')
        cls = None
        if len(p) > 15 and p[:16] == _BGP_HEADER_MARKER:
            cls = BGPHeader
        return cls

class BGPKeepAlive(BGP, BGPHeader):
    """
    KEEPALIVE message.
    """
    name = 'KEEPALIVE'
optional_parameter_codes = {0: 'Reserved', 1: 'Authentication (deprecated)', 2: 'Capabilities'}
_capabilities = {0: 'Reserved', 1: 'Multiprotocol Extensions for BGP-4', 2: 'Route Refresh Capability for BGP-4', 3: 'Outbound Route Filtering Capability', 4: 'Multiple routes to a destination capability', 5: 'Extended Next Hop Encoding', 6: 'BGP-Extended Message', 64: 'Graceful Restart Capability', 65: 'Support for 4-octet AS number capability', 66: 'Deprecated (2003-03-06)', 67: 'Support for Dynamic Capability (capability specific)', 68: 'Multisession BGP Capability', 69: 'ADD-PATH Capability', 70: 'Enhanced Route Refresh Capability', 71: 'Long-Lived Graceful Restart (LLGR) Capability', 73: 'FQDN Capability', 128: 'Route Refresh Capability for BGP-4 (Cisco)', 130: 'Outbound Route Filtering Capability (Cisco)'}
_capabilities_objects = {1: 'BGPCapMultiprotocol', 2: 'BGPCapGeneric', 3: 'BGPCapORF', 64: 'BGPCapGracefulRestart', 65: 'BGPCapFourBytesASN', 70: 'BGPCapGeneric', 130: 'BGPCapORF'}

def _register_cls(registry, cls):
    if False:
        print('Hello World!')
    registry[cls.__name__] = cls
    return cls
_capabilities_registry = {}

def _bgp_capability_dispatcher(payload):
    if False:
        i = 10
        return i + 15
    '\n    Returns the right class for a given BGP capability.\n    '
    cls = _capabilities_registry['BGPCapGeneric']
    if payload is None:
        cls = _capabilities_registry['BGPCapGeneric']
    else:
        length = len(payload)
        if length >= _BGP_CAPABILITY_MIN_SIZE:
            code = orb(payload[0])
            cls = _get_cls(_capabilities_objects.get(code, 'BGPCapGeneric'))
    return cls

class _BGPCap_metaclass(type):

    def __new__(cls, clsname, bases, attrs):
        if False:
            print('Hello World!')
        newclass = super(_BGPCap_metaclass, cls).__new__(cls, clsname, bases, attrs)
        _register_cls(_capabilities_registry, newclass)
        return newclass

class _BGPCapability_metaclass(_BGPCap_metaclass, Packet_metaclass):
    pass

class BGPCapability(Packet, metaclass=_BGPCapability_metaclass):
    """
    Generic BGP capability.
    """

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            print('Hello World!')
        '\n        Returns the right class for the given data.\n        '
        return _bgp_capability_dispatcher(_pkt)

    def pre_dissect(self, s):
        if False:
            while True:
                i = 10
        '\n        Check that the payload is long enough (at least 2 bytes).\n        '
        length = len(s)
        if length < _BGP_CAPABILITY_MIN_SIZE:
            err = ' ({}'.format(length) + ' is < _BGP_CAPABILITY_MIN_SIZE '
            err += '({})).'.format(_BGP_CAPABILITY_MIN_SIZE)
            raise _BGPInvalidDataException(err)
        return s

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        length = 0
        if self.length is None:
            length = len(p) - 2
            p = p[:1] + chb(length) + p[2:]
        return p + pay

class BGPCapGeneric(BGPCapability):
    """
    This class provides an implementation of a generic capability.
    """
    name = 'BGP Capability'
    match_subclass = True
    fields_desc = [ByteEnumField('code', 0, _capabilities), FieldLenField('length', None, fmt='B', length_of='cap_data'), StrLenField('cap_data', '', length_from=lambda p: p.length, max_length=255)]

class BGPCapMultiprotocol(BGPCapability):
    """
    This class provides an implementation of the Multiprotocol
    capability.
    References: RFC 4760
    """
    name = 'Multiprotocol Extensions for BGP-4'
    match_subclass = True
    fields_desc = [ByteEnumField('code', 1, _capabilities), ByteField('length', 4), ShortEnumField('afi', 0, address_family_identifiers), ByteField('reserved', 0), ByteEnumField('safi', 0, subsequent_afis)]
_orf_types = {0: 'Reserved', 64: 'Address Prefix ORF', 65: 'CP-ORF'}
send_receive_values = {1: 'receive', 2: 'send', 3: 'receive + send'}

class BGPCapORFBlock(Packet):
    """
    The "ORFBlock" is made of <AFI, rsvd, SAFI, Number of ORFs, and
    <ORF Type, Send/Receive> entries.
    """

    class ORFTuple(Packet):
        """
        Packet handling <ORF Types, Send/Receive> tuples.
        """
        name = 'ORF Type'
        fields_desc = [ByteEnumField('orf_type', 0, _orf_types), ByteEnumField('send_receive', 0, send_receive_values)]
    name = 'ORF Capability Entry'
    fields_desc = [ShortEnumField('afi', 0, address_family_identifiers), ByteField('reserved', 0), ByteEnumField('safi', 0, subsequent_afis), FieldLenField('orf_number', None, count_of='entries', fmt='!B'), PacketListField('entries', [], ORFTuple, count_from=lambda p: p.orf_number)]

    def post_build(self, p, pay):
        if False:
            return 10
        count = None
        if self.orf_number is None:
            count = len(self.entries)
            p = p[:4] + struct.pack('!B', count) + p[5:]
        return p + pay

class BGPCapORFBlockPacketListField(PacketListField):
    """
    Handles lists of BGPCapORFBlocks.
    """

    def getfield(self, pkt, s):
        if False:
            while True:
                i = 10
        lst = []
        length = None
        if self.length_from is not None:
            length = self.length_from(pkt)
        remain = s
        if length is not None:
            remain = s[:length]
        while remain:
            orf_number = orb(remain[4])
            entries_length = orf_number * 2
            current = remain[:5 + entries_length]
            remain = remain[5 + entries_length:]
            packet = self.m2i(pkt, current)
            lst.append(packet)
        return (remain, lst)

class BGPCapORF(BGPCapability):
    """
    This class provides an implementation of the Outbound Route Filtering
    capability.
    References: RFC 5291
    """
    name = 'Outbound Route Filtering Capability'
    match_subclass = True
    fields_desc = [ByteEnumField('code', 3, _capabilities), ByteField('length', None), BGPCapORFBlockPacketListField('orf', [], BGPCapORFBlock, length_from=lambda p: p.length)]
gr_address_family_flags = {128: 'Forwarding state preserved (0x80: F bit set)'}

class BGPCapGracefulRestart(BGPCapability):
    """
    This class provides an implementation of the Graceful Restart
    capability.
    References: RFC 4724
    """

    class GRTuple(Packet):
        """Tuple <AFI, SAFI, Flags for address family>"""
        name = '<AFI, SAFI, Flags for address family>'
        fields_desc = [ShortEnumField('afi', 0, address_family_identifiers), ByteEnumField('safi', 0, subsequent_afis), ByteEnumField('flags', 0, gr_address_family_flags)]
    name = 'Graceful Restart Capability'
    match_subclass = True
    fields_desc = [ByteEnumField('code', 64, _capabilities), ByteField('length', None), BitField('restart_flags', 0, 4), BitField('restart_time', 0, 12), PacketListField('entries', [], GRTuple)]

class BGPCapFourBytesASN(BGPCapability):
    """
    This class provides an implementation of the 4-octet AS number
    capability.
    References: RFC 4893
    """
    name = 'Support for 4-octet AS number capability'
    match_subclass = True
    fields_desc = [ByteEnumField('code', 65, _capabilities), ByteField('length', 4), IntField('asn', 0)]

class BGPAuthenticationInformation(Packet):
    """
    Provides an implementation of the Authentication Information optional
    parameter, which is now obsolete.
    References: RFC 1771, RFC 1654, RFC 4271
    """
    name = 'BGP Authentication Data'
    fields_desc = [ByteField('authentication_code', 0), StrField('authentication_data', None)]

class BGPOptParamPacketListField(PacketListField):
    """
    PacketListField handling the optional parameters (OPEN message).
    """

    def getfield(self, pkt, s):
        if False:
            i = 10
            return i + 15
        lst = []
        length = 0
        if self.length_from is not None:
            length = self.length_from(pkt)
        remain = s
        if length is not None:
            (remain, ret) = (s[:length], s[length:])
        while remain:
            param_len = orb(remain[1])
            current = remain[:2 + param_len]
            remain = remain[2 + param_len:]
            packet = self.m2i(pkt, current)
            lst.append(packet)
        return (remain + ret, lst)

class BGPOptParam(Packet):
    """
    Provides an implementation the OPEN message optional parameters.
    References: RFC 4271
    """
    name = 'Optional parameter'
    fields_desc = [ByteEnumField('param_type', 2, optional_parameter_codes), ByteField('param_length', None), ConditionalField(PacketField('param_value', None, BGPCapability), lambda p: p.param_type == 2), ConditionalField(PacketField('authentication_data', None, BGPAuthenticationInformation), lambda p: p.param_type == 1)]

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        length = None
        packet = p
        if self.param_length is None:
            if self.param_value is None and self.authentication_data is None:
                length = 0
            else:
                length = len(p) - 2
            packet = p[:1] + chb(length)
            if self.param_type == 2 and self.param_value is not None or (self.param_type == 1 and self.authentication_data is not None):
                packet = packet + p[2:]
        return packet + pay

class BGPOpen(BGP):
    """
    OPEN messages are exchanged in order to open a new BGP session.
    References: RFC 4271
    """
    name = 'OPEN'
    fields_desc = [ByteField('version', 4), ShortField('my_as', 0), ShortField('hold_time', 0), IPField('bgp_id', '0.0.0.0'), FieldLenField('opt_param_len', None, length_of='opt_params', fmt='!B'), BGPOptParamPacketListField('opt_params', [], BGPOptParam, length_from=lambda p: p.opt_param_len)]

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        if self.opt_param_len is None:
            length = len(p) - 10
            p = p[:9] + struct.pack('!B', length) + p[10:]
        return p + pay
path_attributes = {0: 'Reserved', 1: 'ORIGIN', 2: 'AS_PATH', 3: 'NEXT_HOP', 4: 'MULTI_EXIT_DISC', 5: 'LOCAL_PREF', 6: 'ATOMIC_AGGREGATE', 7: 'AGGREGATOR', 8: 'COMMUNITY', 9: 'ORIGINATOR_ID', 10: 'CLUSTER_LIST', 11: 'DPA (deprecated)', 12: 'ADVERTISER  (Historic) (deprecated)', 13: 'RCID_PATH / CLUSTER_ID (Historic) (deprecated)', 14: 'MP_REACH_NLRI', 15: 'MP_UNREACH_NLRI', 16: 'EXTENDED COMMUNITIES', 17: 'AS4_PATH', 18: 'AS4_AGGREGATOR', 19: 'SAFI Specific Attribute (SSA) (deprecated)', 20: 'Connector Attribute (deprecated)', 21: 'AS_PATHLIMIT (deprecated)', 22: 'PMSI_TUNNEL', 23: 'Tunnel Encapsulation Attribute', 24: 'Traffic Engineering', 25: 'IPv6 Address Specific Extended Community', 26: 'AIGP', 27: 'PE Distinguisher Labels', 28: 'BGP Entropy Label Capability Attribute (deprecated)', 29: 'BGP-LS Attribute', 40: 'BGP Prefix-SID', 128: 'ATTR_SET', 255: 'Reserved for development'}
attributes_flags = {1: 64, 2: 64, 3: 64, 4: 128, 5: 64, 6: 64, 7: 192, 8: 192, 9: 128, 10: 128, 11: 192, 12: 128, 13: 128, 14: 128, 15: 128, 16: 192, 17: 192, 18: 192, 19: 192, 20: 192, 21: 192, 22: 192, 23: 192, 24: 128, 25: 192, 26: 128, 27: 192, 28: 192, 29: 128, 40: 192, 128: 192}

class BGPPathAttrPacketListField(PacketListField):
    """
    PacketListField handling the path attributes (UPDATE message).
    """

    def getfield(self, pkt, s):
        if False:
            i = 10
            return i + 15
        lst = []
        length = 0
        if self.length_from is not None:
            length = self.length_from(pkt)
        ret = ''
        remain = s
        if length is not None:
            (remain, ret) = (s[:length], s[length:])
        while remain:
            flags = orb(remain[0])
            attr_len = 0
            if has_extended_length(flags):
                attr_len = struct.unpack('!H', remain[2:4])[0]
                current = remain[:4 + attr_len]
                remain = remain[4 + attr_len:]
            else:
                attr_len = orb(remain[2])
                current = remain[:3 + attr_len]
                remain = remain[3 + attr_len:]
            packet = self.m2i(pkt, current)
            lst.append(packet)
        return (remain + ret, lst)

class BGPPAOrigin(Packet):
    """
    Packet handling the ORIGIN attribute value.
    References: RFC 4271
    """
    name = 'ORIGIN'
    fields_desc = [ByteEnumField('origin', 0, {0: 'IGP', 1: 'EGP', 2: 'INCOMPLETE'})]
as_path_segment_types = {1: 'AS_SET', 2: 'AS_SEQUENCE', 3: 'AS_CONFED_SEQUENCE', 4: 'AS_CONFED_SET'}

class ASPathSegmentPacketListField(PacketListField):
    """
    PacketListField handling AS_PATH segments.
    """

    def getfield(self, pkt, s):
        if False:
            while True:
                i = 10
        lst = []
        remain = s
        while remain:
            segment_length = orb(remain[1])
            if bgp_module_conf.use_2_bytes_asn:
                current = remain[:2 + segment_length * 2]
                remain = remain[2 + segment_length * 2:]
            else:
                current = remain[:2 + segment_length * 4]
                remain = remain[2 + segment_length * 4:]
            packet = self.m2i(pkt, current)
            lst.append(packet)
        return (remain, lst)

class BGPPAASPath(Packet):
    """
    Packet handling the AS_PATH attribute value (2 bytes ASNs, for old
    speakers).
    References: RFC 4271, RFC 5065
    """
    AS_TRANS = 23456

    class ASPathSegment(Packet):
        """
        Provides an implementation for AS_PATH segments with 2 bytes ASNs.
        """
        fields_desc = [ByteEnumField('segment_type', 2, as_path_segment_types), ByteField('segment_length', None), FieldListField('segment_value', [], ShortField('asn', 0))]

        def post_build(self, p, pay):
            if False:
                print('Hello World!')
            segment_len = self.segment_length
            if segment_len is None:
                segment_len = len(self.segment_value)
                p = p[:1] + chb(segment_len) + p[2:]
            return p + pay
    name = 'AS_PATH (RFC 4271)'
    fields_desc = [ASPathSegmentPacketListField('segments', [], ASPathSegment)]

class BGPPAAS4BytesPath(Packet):
    """
    Packet handling the AS_PATH attribute value (4 bytes ASNs, for new
    speakers -> ASNs are encoded as IntFields).
    References: RFC 4893
    """

    class ASPathSegment(Packet):
        """
        Provides an implementation for AS_PATH segments with 4 bytes ASNs.
        """
        fields_desc = [ByteEnumField('segment_type', 2, as_path_segment_types), ByteField('segment_length', None), FieldListField('segment_value', [], IntField('asn', 0))]

        def post_build(self, p, pay):
            if False:
                return 10
            segment_len = self.segment_length
            if segment_len is None:
                segment_len = len(self.segment_value)
                p = p[:1] + chb(segment_len) + p[2:]
            return p + pay
    name = 'AS_PATH (RFC 4893)'
    fields_desc = [ASPathSegmentPacketListField('segments', [], ASPathSegment)]

class BGPPANextHop(Packet):
    """
    Packet handling the NEXT_HOP attribute value.
    References: RFC 4271
    """
    name = 'NEXT_HOP'
    fields_desc = [IPField('next_hop', '0.0.0.0')]

class BGPPAMultiExitDisc(Packet):
    """
    Packet handling the MULTI_EXIT_DISC attribute value.
    References: RFC 4271
    """
    name = 'MULTI_EXIT_DISC'
    fields_desc = [IntField('med', 0)]

class BGPPALocalPref(Packet):
    """
    Packet handling the LOCAL_PREF attribute value.
    References: RFC 4271
    """
    name = 'LOCAL_PREF'
    fields_desc = [IntField('local_pref', 0)]

class BGPPAAtomicAggregate(Packet):
    """
    Packet handling the ATOMIC_AGGREGATE attribute value.
    References: RFC 4271
    """
    name = 'ATOMIC_AGGREGATE'

class BGPPAAggregator(Packet):
    """
    Packet handling the AGGREGATOR attribute value.
    References: RFC 4271
    """
    name = 'AGGREGATOR'
    fields_desc = [ShortField('aggregator_asn', 0), IPField('speaker_address', '0.0.0.0')]
well_known_communities = {4294967041: 'NO_EXPORT', 4294967042: 'NO_ADVERTISE', 4294967043: 'NO_EXPORT_SUBCONFED', 4294967044: 'NOPEER', 4294901760: 'planned-shut', 4294901761: 'ACCEPT-OWN', 4294901762: 'ROUTE_FILTER_TRANSLATED_v4', 4294901763: 'ROUTE_FILTER_v4', 4294901764: 'ROUTE_FILTER_TRANSLATED_v6', 4294901765: 'ROUTE_FILTER_v6', 4294901766: 'LLGR_STALE', 4294901767: 'NO_LLGR', 4294901768: 'accept-own-nexthop'}

class BGPPACommunity(Packet):
    """
    Packet handling the COMMUNITIES attribute value.
    References: RFC 1997
    """
    name = 'COMMUNITIES'
    fields_desc = [IntEnumField('community', 0, well_known_communities)]

class BGPPAOriginatorID(Packet):
    """
    Packet handling the ORIGINATOR_ID attribute value.
    References: RFC 4456
    """
    name = 'ORIGINATOR_ID'
    fields_desc = [IPField('originator_id', '0.0.0.0')]

class BGPPAClusterList(Packet):
    """
    Packet handling the CLUSTER_LIST attribute value.
    References: RFC 4456
    """
    name = 'CLUSTER_LIST'
    fields_desc = [FieldListField('cluster_list', [], IntField('cluster_id', 0))]
_ext_comm_types = {0: 'Transitive Two-Octet AS-Specific Extended Community', 1: 'Transitive IPv4-Address-Specific Extended Community', 2: 'Transitive Four-Octet AS-Specific Extended Community', 3: 'Transitive Opaque Extended Community', 4: 'QoS Marking', 5: 'CoS Capability', 6: 'EVPN', 7: 'Unassigned', 8: 'Flow spec redirect/mirror to IP next-hop', 64: 'Non-Transitive Two-Octet AS-Specific Extended Community', 65: 'Non-Transitive IPv4-Address-Specific Extended Community', 66: 'Non-Transitive Four-Octet AS-Specific Extended Community', 67: 'Non-Transitive Opaque Extended Community', 68: 'QoS Marking', 128: 'Generic Transitive Experimental Use Extended Community', 129: 'Generic Transitive Experimental Use Extended Community Part 2', 130: 'Generic Transitive Experimental Use Extended Community Part 3'}
_ext_comm_evpn_subtypes = {0: 'MAC Mobility', 1: 'ESI Label', 2: 'ES-Import Route Target', 3: 'EVPN Router"s MAC Extended Community', 4: 'Layer 2 Extended Community', 5: 'E-TREE Extended Community', 6: 'DF Election Extended Community', 7: 'I-SID Extended Community'}
_ext_comm_trans_two_octets_as_specific_subtypes = {2: 'Route Target', 3: 'Route Origin', 4: 'Unassigned', 5: 'OSPF Domain Identifier', 8: 'BGP Data Collection', 9: 'Source AS', 10: 'L2VPN Identifier', 16: 'Cisco VPN-Distinguisher'}
_ext_comm_non_trans_two_octets_as_specific_subtypes = {4: 'Link Bandwidth Extended Community', 128: 'Virtual-Network Identifier Extended Community'}
_ext_comm_trans_four_octets_as_specific_subtypes = {2: 'Route Target', 3: 'Route Origin', 4: 'Generic', 5: 'OSPF Domain Identifier', 8: 'BGP Data Collection', 9: 'Source AS', 16: 'Cisco VPN Identifier'}
_ext_comm_non_trans_four_octets_as_specific_subtypes = {4: 'Generic'}
_ext_comm_trans_ipv4_addr_specific_subtypes = {2: 'Route Target', 3: 'Route Origin', 5: 'OSPF Domain Identifier', 7: 'OSPF Route ID', 10: 'L2VPN Identifier', 11: 'VRF Route Import', 12: 'Flow-spec Redirect to IPv4', 16: 'Cisco VPN-Distinguisher', 18: 'Inter-Area P2MP Segmented Next-Hop'}
_ext_comm_non_trans_ipv4_addr_specific_subtypes = {}
_ext_comm_trans_opaque_subtypes = {1: 'Cost Community', 3: 'CP-ORF', 4: 'Extranet Source Extended Community', 5: 'Extranet Separation Extended Community', 6: 'OSPF Route Type', 7: 'Additional PMSI Tunnel Attribute Flags', 11: 'Color Extended Community', 12: 'Encapsulation Extended Community', 13: 'Default Gateway', 14: 'Point-to-Point-to-Multipoint (PPMP) Label', 19: 'Route-Target Record', 20: 'Consistent Hash Sort Order'}
_ext_comm_non_trans_opaque_subtypes = {0: 'BGP Origin Validation State', 1: 'Cost Community'}
_ext_comm_generic_transitive_exp_subtypes = {0: 'OSPF Route Type (deprecated)', 1: 'OSPF Router ID (deprecated)', 5: 'OSPF Domain Identifier (deprecated)', 6: 'Flow spec traffic-rate', 7: 'Flow spec traffic-action', 8: 'Flow spec redirect AS-2byte format', 9: 'Flow spec traffic-remarking', 10: 'Layer2 Info Extended Community', 11: 'E-Tree Info'}
_ext_comm_generic_transitive_exp_part2_subtypes = {8: 'Flow spec redirect IPv4 format'}
_ext_comm_generic_transitive_exp_part3_subtypes = {8: 'Flow spec redirect AS-4byte format'}
_ext_comm_traffic_action_fields = {47: 'Terminal Action', 46: 'Sample'}
_ext_comm_trans_ipv6_addr_specific_types = {2: 'Route Target', 3: 'Route Origin', 4: 'OSPFv3 Route Attributes (DEPRECATED)', 11: 'VRF Route Import', 12: 'Flow-spec Redirect to IPv6', 16: 'Cisco VPN-Distinguisher', 17: 'UUID-based Route Target', 18: 'Inter-Area P2MP Segmented Next-Hop'}
_ext_comm_non_trans_ipv6_addr_specific_types = {}
_ext_comm_subtypes_classes = {0: _ext_comm_trans_two_octets_as_specific_subtypes, 1: _ext_comm_trans_ipv4_addr_specific_subtypes, 2: _ext_comm_trans_four_octets_as_specific_subtypes, 3: _ext_comm_trans_opaque_subtypes, 6: _ext_comm_evpn_subtypes, 64: _ext_comm_non_trans_two_octets_as_specific_subtypes, 65: _ext_comm_non_trans_ipv4_addr_specific_subtypes, 66: _ext_comm_non_trans_four_octets_as_specific_subtypes, 67: _ext_comm_non_trans_opaque_subtypes, 128: _ext_comm_generic_transitive_exp_subtypes, 129: _ext_comm_generic_transitive_exp_part2_subtypes, 130: _ext_comm_generic_transitive_exp_part3_subtypes}

class BGPPAExtCommTwoOctetASSpecific(Packet):
    """
    Packet handling the Two-Octet AS Specific Extended Community attribute
    value.
    References: RFC 4360
    """
    name = 'Two-Octet AS Specific Extended Community'
    fields_desc = [ShortField('global_administrator', 0), IntField('local_administrator', 0)]

class BGPPAExtCommFourOctetASSpecific(Packet):
    """
    Packet handling the Four-Octet AS Specific Extended Community
    attribute value.
    References: RFC 5668
    """
    name = 'Four-Octet AS Specific Extended Community'
    fields_desc = [IntField('global_administrator', 0), ShortField('local_administrator', 0)]

class BGPPAExtCommIPv4AddressSpecific(Packet):
    """
    Packet handling the IPv4 Address Specific Extended Community attribute
    value.
    References: RFC 4360
    """
    name = 'IPv4 Address Specific Extended Community'
    fields_desc = [IntField('global_administrator', 0), ShortField('local_administrator', 0)]

class BGPPAExtCommOpaque(Packet):
    """
    Packet handling the Opaque Extended Community attribute value.
    References: RFC 4360
    """
    name = 'Opaque Extended Community'
    fields_desc = [StrFixedLenField('value', '', length=6)]

class BGPPAExtCommTrafficRate(Packet):
    """
    Packet handling the (FlowSpec) "traffic-rate" extended community.
    References: RFC 5575
    """
    name = 'FlowSpec traffic-rate extended community'
    fields_desc = [ShortField('id', 0), IEEEFloatField('rate', 0)]

class BGPPAExtCommTrafficAction(Packet):
    """
    Packet handling the (FlowSpec) "traffic-action" extended community.
    References: RFC 5575
    """
    name = 'FlowSpec traffic-action extended community'
    fields_desc = [BitField('reserved', 0, 46), BitField('sample', 0, 1), BitField('terminal_action', 0, 1)]

class BGPPAExtCommRedirectAS2Byte(Packet):
    """
    Packet handling the (FlowSpec) "redirect AS-2byte" extended community
    (RFC 7674).
    References: RFC 7674
    """
    name = 'FlowSpec redirect AS-2byte extended community'
    fields_desc = [ShortField('asn', 0), IntField('value', 0)]

class BGPPAExtCommRedirectIPv4(Packet):
    """
    Packet handling the (FlowSpec) "redirect IPv4" extended community.
    (RFC 7674).
    References: RFC 7674
    """
    name = 'FlowSpec redirect IPv4 extended community'
    fields_desc = [IntField('ip_addr', 0), ShortField('value', 0)]

class BGPPAExtCommRedirectAS4Byte(Packet):
    """
    Packet handling the (FlowSpec) "redirect AS-4byte" extended community.
    (RFC 7674).
    References: RFC 7674
    """
    name = 'FlowSpec redirect AS-4byte extended community'
    fields_desc = [IntField('asn', 0), ShortField('value', 0)]

class BGPPAExtCommTrafficMarking(Packet):
    """
    Packet handling the (FlowSpec) "traffic-marking" extended community.
    References: RFC 5575
    """
    name = 'FlowSpec traffic-marking extended community'
    fields_desc = [BitEnumField('dscp', 48, 48, _ext_comm_traffic_action_fields)]
_ext_high_low_dict = {BGPPAExtCommTwoOctetASSpecific: (0, 0), BGPPAExtCommIPv4AddressSpecific: (1, 0), BGPPAExtCommFourOctetASSpecific: (2, 0), BGPPAExtCommOpaque: (3, 0), BGPPAExtCommTrafficRate: (128, 6), BGPPAExtCommTrafficAction: (128, 7), BGPPAExtCommRedirectAS2Byte: (128, 8), BGPPAExtCommTrafficMarking: (128, 9), BGPPAExtCommRedirectIPv4: (129, 8), BGPPAExtCommRedirectAS4Byte: (130, 8)}

class _ExtCommValuePacketField(PacketField):
    """
    PacketField handling Extended Communities "value parts".
    """
    __slots__ = ['type_from']

    def __init__(self, name, default, cls, type_from=(0, 0)):
        if False:
            print('Hello World!')
        PacketField.__init__(self, name, default, cls)
        self.type_from = type_from

    def m2i(self, pkt, m):
        if False:
            return 10
        ret = None
        (type_high, type_low) = self.type_from(pkt)
        if type_high == 0 or type_high == 64:
            ret = BGPPAExtCommTwoOctetASSpecific(m)
        elif type_high == 1 or type_high == 65:
            ret = BGPPAExtCommIPv4AddressSpecific(m)
        elif type_high == 2 or type_high == 66:
            ret = BGPPAExtCommFourOctetASSpecific(m)
        elif type_high == 3 or type_high == 67:
            ret = BGPPAExtCommOpaque(m)
        elif type_high == 128:
            if type_low == 6:
                ret = BGPPAExtCommTrafficRate(m)
            elif type_low == 7:
                ret = BGPPAExtCommTrafficAction(m)
            elif type_low == 8:
                ret = BGPPAExtCommRedirectAS2Byte(m)
            elif type_low == 9:
                ret = BGPPAExtCommTrafficMarking(m)
        elif type_high == 129:
            if type_low == 8:
                ret = BGPPAExtCommRedirectIPv4(m)
        elif type_high == 130:
            if type_low == 8:
                ret = BGPPAExtCommRedirectAS4Byte(m)
        else:
            ret = conf.raw_layer(m)
        return ret

class BGPPAIPv6AddressSpecificExtComm(Packet):
    """
    Provides an implementation of the IPv6 Address Specific Extended
    Community attribute. This attribute is not defined using the existing
    BGP Extended Community attribute (see the RFC 5701 excerpt below).
    References: RFC 5701
    """
    name = 'IPv6 Address Specific Extended Community'
    fields_desc = [IP6Field('global_administrator', '::'), ShortField('local_administrator', 0)]

def _get_ext_comm_subtype(type_high):
    if False:
        print('Hello World!')
    '\n    Returns a ByteEnumField with the right sub-types dict for a given community.  # noqa: E501\n    http://www.iana.org/assignments/bgp-extended-communities/bgp-extended-communities.xhtml\n    '
    return _ext_comm_subtypes_classes.get(type_high, {})

class _TypeLowField(ByteField):
    """
    Field used to retrieve "dynamically" the right sub-type dict.
    """
    __slots__ = ['enum_from']

    def __init__(self, name, default, enum_from=None):
        if False:
            while True:
                i = 10
        ByteField.__init__(self, name=name, default=default)
        self.enum_from = enum_from

    def i2repr(self, pkt, i):
        if False:
            while True:
                i = 10
        enum = self.enum_from(pkt)
        return enum.get(i, i)

class BGPPAExtCommunity(Packet):
    """
    Provides an implementation of the Extended Communities attribute.
    References: RFC 4360
    """
    name = 'EXTENDED_COMMUNITY'
    fields_desc = [ByteEnumField('type_high', 0, _ext_comm_types), _TypeLowField('type_low', None, enum_from=lambda x: _get_ext_comm_subtype(x.type_high)), _ExtCommValuePacketField('value', None, Packet, type_from=lambda x: (x.type_high, x.type_low))]

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        if self.value is None:
            p = p[:2]
        if self.type_low is None and self.value is not None:
            (high, low) = _ext_high_low_dict.get(self.value.__class__, (0, 0))
            p = chb(high) + chb(low) + p[2:]
        return p + pay

class _ExtCommsPacketListField(PacketListField):
    """
    PacketListField handling a list of extended communities.
    """

    def getfield(self, pkt, s):
        if False:
            print('Hello World!')
        lst = []
        length = len(s)
        remain = s[:length]
        while remain:
            current = remain[:8]
            remain = remain[8:]
            packet = self.m2i(pkt, current)
            lst.append(packet)
        return (remain, lst)

class BGPPAExtComms(Packet):
    """
    Packet handling the multiple extended communities.
    """
    name = 'EXTENDED_COMMUNITIES'
    fields_desc = [_ExtCommsPacketListField('extended_communities', [], BGPPAExtCommunity)]

class MPReachNLRIPacketListField(PacketListField):
    """
    PacketListField handling the AFI specific part (except for the length of
    Next Hop Network Address field, which is not AFI specific) of the
    MP_REACH_NLRI attribute.
    """

    def getfield(self, pkt, s):
        if False:
            i = 10
            return i + 15
        lst = []
        remain = s
        if pkt.afi == 2:
            if pkt.safi == 1:
                while remain:
                    mask = orb(remain[0])
                    length_in_bytes = (mask + 7) // 8
                    current = remain[:length_in_bytes + 1]
                    remain = remain[length_in_bytes + 1:]
                    prefix = self.m2i(pkt, current)
                    lst.append(prefix)
        return (remain, lst)

class BGPPAMPReachNLRI(Packet):
    """
    Packet handling the MP_REACH_NLRI attribute value, for non IPv6
    AFI.
    References: RFC 4760
    """
    name = 'MP_REACH_NLRI'
    fields_desc = [ShortEnumField('afi', 0, address_family_identifiers), ByteEnumField('safi', 0, subsequent_afis), ByteField('nh_addr_len', 0), ConditionalField(IPField('nh_v4_addr', '0.0.0.0'), lambda x: x.afi == 1 and x.nh_addr_len == 4), ConditionalField(IP6Field('nh_v6_addr', '::'), lambda x: x.afi == 2 and x.nh_addr_len == 16), ConditionalField(IP6Field('nh_v6_global', '::'), lambda x: x.afi == 2 and x.nh_addr_len == 32), ConditionalField(IP6Field('nh_v6_link_local', '::'), lambda x: x.afi == 2 and x.nh_addr_len == 32), ByteField('reserved', 0), MPReachNLRIPacketListField('nlri', [], BGPNLRI_IPv6)]

    def post_build(self, p, pay):
        if False:
            i = 10
            return i + 15
        if self.nlri is None:
            p = p[:3]
        return p + pay

class BGPPAMPUnreachNLRI_IPv6(Packet):
    """
    Packet handling the MP_UNREACH_NLRI attribute value, for IPv6 AFI.
    """
    name = 'MP_UNREACH_NLRI (IPv6 NLRI)'
    fields_desc = [BGPNLRIPacketListField('withdrawn_routes', [], 'IPv6')]

class MPUnreachNLRIPacketField(PacketField):
    """
    PacketField handling the AFI specific part of the MP_UNREACH_NLRI
    attribute.
    """

    def m2i(self, pkt, m):
        if False:
            print('Hello World!')
        ret = None
        if pkt.afi == 2:
            ret = BGPPAMPUnreachNLRI_IPv6(m)
        else:
            ret = conf.raw_layer(m)
        return ret

class BGPPAMPUnreachNLRI(Packet):
    """
    Packet handling the MP_UNREACH_NLRI attribute value, for non IPv6
    AFI.
    References: RFC 4760
    """
    name = 'MP_UNREACH_NLRI'
    fields_desc = [ShortEnumField('afi', 0, address_family_identifiers), ByteEnumField('safi', 0, subsequent_afis), MPUnreachNLRIPacketField('afi_safi_specific', None, Packet)]

    def post_build(self, p, pay):
        if False:
            for i in range(10):
                print('nop')
        if self.afi_safi_specific is None:
            p = p[:3]
        return p + pay

class BGPPAAS4Path(Packet):
    """
    Provides an implementation of the AS4_PATH attribute "value part".
    References: RFC 4893
    """
    name = 'AS4_PATH'
    fields_desc = [ByteEnumField('segment_type', 2, {1: 'AS_SET', 2: 'AS_SEQUENCE'}), ByteField('segment_length', None), FieldListField('segment_value', [], IntField('asn', 0))]

    def post_build(self, p, pay):
        if False:
            return 10
        if self.segment_length is None:
            segment_len = len(self.segment_value)
            p = p[:1] + chb(segment_len) + p[2:]
        return p + pay

class BGPPAAS4Aggregator(Packet):
    """
    Provides an implementation of the AS4_AGGREGATOR attribute
    "value part".
    References: RFC 4893
    """
    name = 'AS4_AGGREGATOR '
    fields_desc = [IntField('aggregator_asn', 0), IPField('speaker_address', '0.0.0.0')]
_path_attr_objects = {1: 'BGPPAOrigin', 2: 'BGPPAASPath', 3: 'BGPPANextHop', 4: 'BGPPAMultiExitDisc', 5: 'BGPPALocalPref', 6: 'BGPPAAtomicAggregate', 7: 'BGPPAAggregator', 8: 'BGPPACommunity', 9: 'BGPPAOriginatorID', 10: 'BGPPAClusterList', 14: 'BGPPAMPReachNLRI', 15: 'BGPPAMPUnreachNLRI', 16: 'BGPPAExtComms', 17: 'BGPPAAS4Path', 25: 'BGPPAIPv6AddressSpecificExtComm'}

class _PathAttrPacketField(PacketField):
    """
    PacketField handling path attribute value parts.
    """

    def m2i(self, pkt, m):
        if False:
            while True:
                i = 10
        ret = None
        type_code = pkt.type_code
        if type_code == 0 or type_code == 255:
            ret = conf.raw_layer(m)
        elif type_code >= 30 and type_code <= 39 or (type_code >= 41 and type_code <= 127) or (type_code >= 129 and type_code <= 254):
            ret = conf.raw_layer(m)
        elif type_code == 2 and (not bgp_module_conf.use_2_bytes_asn):
            ret = BGPPAAS4BytesPath(m)
        else:
            ret = _get_cls(_path_attr_objects.get(type_code, conf.raw_layer))(m)
        return ret

class BGPPathAttr(Packet):
    """
    Provides an implementation of the path attributes.
    References: RFC 4271
    """
    name = 'BGPPathAttr'
    fields_desc = [FlagsField('type_flags', 128, 8, ['NA0', 'NA1', 'NA2', 'NA3', 'Extended-Length', 'Partial', 'Transitive', 'Optional']), ByteEnumField('type_code', 0, path_attributes), ConditionalField(ShortField('attr_ext_len', None), lambda x: x.type_flags is not None and has_extended_length(x.type_flags)), ConditionalField(ByteField('attr_len', None), lambda x: x.type_flags is not None and (not has_extended_length(x.type_flags))), _PathAttrPacketField('attribute', None, Packet)]

    def post_build(self, p, pay):
        if False:
            i = 10
            return i + 15
        flags_value = None
        length = None
        packet = None
        extended_length = False
        if self.type_flags is None:
            if self.type_code in attributes_flags:
                flags_value = attributes_flags.get(self.type_code)
            else:
                flags_value = 128
            extended_length = has_extended_length(flags_value)
        else:
            extended_length = has_extended_length(self.type_flags)
        if flags_value is None:
            packet = p[:2]
        else:
            packet = struct.pack('!B', flags_value) + p[1]
        if self.attr_len is None:
            if self.attribute is None:
                length = 0
            elif extended_length:
                length = len(p) - 4
            else:
                length = len(p) - 3
        if length is None:
            if extended_length:
                packet = packet + p[2:4]
            else:
                packet = packet + p[2]
        elif extended_length:
            packet = packet + struct.pack('!H', length)
        else:
            packet = packet + struct.pack('!B', length)
        if extended_length:
            if self.attribute is not None:
                packet = packet + p[4:]
        elif self.attribute is not None:
            packet = packet + p[3:]
        return packet + pay

class BGPUpdate(BGP):
    """
    UPDATE messages allow peers to exchange routes.
    References: RFC 4271
    """
    name = 'UPDATE'
    fields_desc = [FieldLenField('withdrawn_routes_len', None, length_of='withdrawn_routes', fmt='!H'), BGPNLRIPacketListField('withdrawn_routes', [], 'IPv4', length_from=lambda p: p.withdrawn_routes_len), FieldLenField('path_attr_len', None, length_of='path_attr', fmt='!H'), BGPPathAttrPacketListField('path_attr', [], BGPPathAttr, length_from=lambda p: p.path_attr_len), BGPNLRIPacketListField('nlri', [], 'IPv4')]

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        subpacklen = lambda p: len(p)
        packet = ''
        if self.withdrawn_routes_len is None:
            wl = sum(map(subpacklen, self.withdrawn_routes))
            packet = p[:0] + struct.pack('!H', wl) + p[2:]
        else:
            wl = self.withdrawn_routes_len
        if self.path_attr_len is None:
            length = sum(map(subpacklen, self.path_attr))
            packet = p[:2 + wl] + struct.pack('!H', length) + p[4 + wl:]
        return packet + pay
_error_codes = {1: 'Message Header Error', 2: 'OPEN Message Error', 3: 'UPDATE Message Error', 4: 'Hold Timer Expired', 5: 'Finite State Machine Error', 6: 'Cease', 7: 'ROUTE-REFRESH Message Error'}
_error_subcodes = {0: {}, 1: {0: 'Unspecific', 1: 'Connection Not Synchronized', 2: 'Bad Message Length', 3: 'Bad Message Type'}, 2: {0: 'Reserved', 1: 'Unsupported Version Number', 2: 'Bad Peer AS', 3: 'Bad BGP Identifier', 4: 'Unsupported Optional Parameter', 5: 'Authentication Failure - Deprecated (RFC 4271)', 6: 'Unacceptable Hold Time', 7: 'Unsupported Capability'}, 3: {0: 'Reserved', 1: 'Malformed Attribute List', 2: 'Unrecognized Well-known Attribute', 3: 'Missing Well-known Attribute', 4: 'Attribute Flags Error', 5: 'Attribute Length Error', 6: 'Invalid ORIGIN Attribute', 7: 'AS Routing Loop - Deprecated (RFC 4271)', 8: 'Invalid NEXT_HOP Attribute', 9: 'Optional Attribute Error', 10: 'Invalid Network Field', 11: 'Malformed AS_PATH'}, 4: {}, 5: {0: 'Unspecified Error', 1: 'Receive Unexpected Message in OpenSent State', 2: 'Receive Unexpected Message in OpenConfirm State', 3: 'Receive Unexpected Message in Established State'}, 6: {0: 'Unspecified Error', 1: 'Maximum Number of Prefixes Reached', 2: 'Administrative Shutdown', 3: 'Peer De-configured', 4: 'Administrative Reset', 5: 'Connection Rejected', 6: 'Other Configuration Change', 7: 'Connection Collision Resolution', 8: 'Out of Resources'}, 7: {0: 'Reserved', 1: 'Invalid Message Length'}}

class BGPNotification(BGP):
    """
    NOTIFICATION messages end a BGP session.
    References: RFC 4271
    """
    name = 'NOTIFICATION'
    fields_desc = [ByteEnumField('error_code', 0, _error_codes), MultiEnumField('error_subcode', 0, _error_subcodes, depends_on=lambda p: p.error_code, fmt='B'), StrField(name='data', default=None)]
_orf_when_to_refresh = {1: 'IMMEDIATE', 2: 'DEFER'}
_orf_actions = {0: 'ADD', 1: 'REMOVE', 2: 'REMOVE-ALL'}
_orf_match = {0: 'PERMIT', 1: 'DENY'}
_orf_entry_afi = 1
_orf_entry_safi = 1

def _update_orf_afi_safi(afi, safi):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function that sets the afi / safi values\n    of ORP entries.\n    '
    global _orf_entry_afi
    global _orf_entry_safi
    _orf_entry_afi = afi
    _orf_entry_safi = safi

class BGPORFEntry(Packet):
    """
    Provides an implementation of an ORF entry.
    References: RFC 5291
    """
    __slots__ = ['afi', 'safi']
    name = 'ORF entry'
    fields_desc = [BitEnumField('action', 0, 2, _orf_actions), BitEnumField('match', 0, 1, _orf_match), BitField('reserved', 0, 5), StrField('value', '')]

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.afi = kwargs.pop('afi', 1)
        self.safi = kwargs.pop('safi', 1)
        super(BGPORFEntry, self).__init__(*args, **kwargs)

class _ORFNLRIPacketField(PacketField):
    """
    PacketField handling the ORF NLRI.
    """

    def m2i(self, pkt, m):
        if False:
            print('Hello World!')
        ret = None
        if pkt.afi == 1:
            ret = BGPNLRI_IPv4(m)
        elif pkt.afi == 2:
            ret = BGPNLRI_IPv6(m)
        else:
            ret = conf.raw_layer(m)
        return ret

class BGPORFAddressPrefix(BGPORFEntry):
    """
    Provides an implementation of the Address Prefix ORF (RFC 5292).
    """
    name = 'Address Prefix ORF'
    fields_desc = [BitEnumField('action', 0, 2, _orf_actions), BitEnumField('match', 0, 1, _orf_match), BitField('reserved', 0, 5), IntField('sequence', 0), ByteField('min_len', 0), ByteField('max_len', 0), _ORFNLRIPacketField('prefix', '', Packet)]

class BGPORFCoveringPrefix(BGPORFEntry):
    """
    Provides an implementation of the CP-ORF (RFC 7543).
    """
    name = 'CP-ORF'
    fields_desc = [BitEnumField('action', 0, 2, _orf_actions), BitEnumField('match', 0, 1, _orf_match), BitField('reserved', 0, 5), IntField('sequence', 0), ByteField('min_len', 0), ByteField('max_len', 0), LongField('rt', 0), LongField('import_rt', 0), ByteField('route_type', 0), PacketField('host_addr', None, Packet)]

class BGPORFEntryPacketListField(PacketListField):
    """
    PacketListField handling the ORF entries.
    """

    def m2i(self, pkt, m):
        if False:
            print('Hello World!')
        ret = None
        if isinstance(pkt.underlayer, BGPRouteRefresh):
            afi = pkt.underlayer.afi
            safi = pkt.underlayer.safi
        else:
            afi = 1
            safi = 1
        if pkt.orf_type == 64 or pkt.orf_type == 128:
            ret = BGPORFAddressPrefix(m, afi=afi, safi=safi)
        elif pkt.orf_type == 65:
            ret = BGPORFCoveringPrefix(m, afi=afi, safi=safi)
        else:
            ret = conf.raw_layer(m)
        return ret

    def getfield(self, pkt, s):
        if False:
            for i in range(10):
                print('nop')
        lst = []
        length = 0
        ret = b''
        if self.length_from is not None:
            length = self.length_from(pkt)
        remain = s
        if length <= 0:
            return (s, [])
        if length is not None:
            (remain, ret) = (s[:length], s[length:])
        while remain:
            orf_len = length
            if pkt.orf_type == 64 or pkt.orf_type == 128:
                prefix_len = _bits_to_bytes_len(orb(remain[6]))
                orf_len = 8 + prefix_len
            elif pkt.orf_type == 65:
                if pkt.afi == 1:
                    orf_len = 23 + 4
                elif pkt.afi == 2:
                    orf_len = 23 + 16
                elif pkt.afi == 25:
                    route_type = orb(remain[22])
                    if route_type == 2:
                        orf_len = 23 + 6
                    else:
                        orf_len = 23
            current = remain[:orf_len]
            remain = remain[orf_len:]
            packet = self.m2i(pkt, current)
            lst.append(packet)
        return (remain + ret, lst)

class BGPORF(Packet):
    """
    Provides an implementation of ORFs carried in the RR message.
    References: RFC 5291
    """
    name = 'ORF'
    fields_desc = [ByteEnumField('when_to_refresh', 0, _orf_when_to_refresh), ByteEnumField('orf_type', 0, _orf_types), FieldLenField('orf_len', None, length_of='entries', fmt='!H'), BGPORFEntryPacketListField('entries', [], Packet, length_from=lambda p: p.orf_len)]
rr_message_subtypes = {0: 'Route-Refresh', 1: 'BoRR', 2: 'EoRR', 255: 'Reserved'}

class BGPRouteRefresh(BGP):
    """
    Provides an implementation of the ROUTE-REFRESH message.
    References: RFC 2918, RFC 7313
    """
    name = 'ROUTE-REFRESH'
    fields_desc = [ShortEnumField('afi', 1, address_family_identifiers), ByteEnumField('subtype', 0, rr_message_subtypes), ByteEnumField('safi', 1, subsequent_afis), ConditionalField(PacketField('orf_data', '', BGPORF), lambda p: (p.underlayer and p.underlayer.len or 24) > 23)]
bind_layers(TCP, BGP, dport=179)
bind_layers(TCP, BGP, sport=179)
bind_layers(BGPHeader, BGPOpen, {'type': 1})
bind_layers(BGPHeader, BGPUpdate, {'type': 2})
bind_layers(BGPHeader, BGPNotification, {'type': 3})
bind_layers(BGPHeader, BGPKeepAlive, {'type': 4})
bind_layers(BGPHeader, BGPRouteRefresh, {'type': 5})
log_runtime.warning('[bgp.py] use_2_bytes_asn: %s', bgp_module_conf.use_2_bytes_asn)