"""
    STUN (RFC 8489)

    TLV code derived from the DTP implementation:
      Thanks to Nicolas Bareil,
                Arnaud Ebalard,
                Jochen Bartl.
"""
import struct
import itertools
from scapy.layers.inet import UDP, TCP
from scapy.config import conf
from scapy.packet import Packet, bind_layers
from scapy.utils import inet_ntoa, inet_aton
from scapy.fields import BitField, BitEnumField, LenField, IntField, PadField, StrLenField, PacketListField, XShortField, FieldLenField, ShortField, ByteEnumField, ByteField, XNBytesField, XLongField, XIntField, XBitField, IPField
MAGIC_COOKIE = 554869826
_stun_class = {'request': 0, 'indication': 1, 'success response': 2, 'error response': 3}
_stun_method = {'Binding': 1}
_stun_message_type = {'{} {}'.format(method, class_): method_code & 15 | (class_code & 1) << 4 | (method_code & 112) << 5 | (class_code & 2) << 7 | (method_code & 3968) << 9 for ((method, method_code), (class_, class_code)) in itertools.product(_stun_method.items(), _stun_class.items())}

class STUNGenericTlv(Packet):
    name = 'STUN Generic TLV'
    fields_desc = [XShortField('type', 0), FieldLenField('length', None, length_of='value'), PadField(StrLenField('value', '', length_from=lambda pkt: pkt.length), align=4)]

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if _pkt and len(_pkt) >= 2:
            t = struct.unpack('!H', _pkt[:2])[0]
            return _stun_tlv_class.get(t, cls)
        return cls

    def guess_payload_class(self, payload):
        if False:
            for i in range(10):
                print('nop')
        return conf.padding_layer

class STUNUsername(STUNGenericTlv):
    name = 'STUN Username'
    fields_desc = [XShortField('type', 6), FieldLenField('length', None, length_of='username'), PadField(StrLenField('username', '', length_from=lambda pkt: pkt.length), align=4, padwith=b' ')]

class STUNMessageIntegrity(STUNGenericTlv):
    name = 'STUN Message Integrity'
    fields_desc = [XShortField('type', 8), ShortField('length', 20), XNBytesField('hmac_sha1', 0, 20)]

    def post_build(self, pkt, pay):
        if False:
            i = 10
            return i + 15
        pkt += pay
        return pkt

class STUNPriority(STUNGenericTlv):
    name = 'STUN Priority'
    fields_desc = [XShortField('type', 36), ShortField('length', 4), IntField('priority', 0)]
_xor_mapped_address_family = {'IPv4': 1, 'IPv6': 2}

class XorPort(ShortField):

    def m2i(self, pkt, x):
        if False:
            i = 10
            return i + 15
        return x ^ MAGIC_COOKIE >> 16

    def i2m(self, pkt, x):
        if False:
            print('Hello World!')
        return x ^ MAGIC_COOKIE >> 16

class XorIp(IPField):

    def m2i(self, pkt, x):
        if False:
            i = 10
            return i + 15
        return inet_ntoa(struct.pack('>i', struct.unpack('>i', x)[0] ^ MAGIC_COOKIE))

    def i2m(self, pkt, x):
        if False:
            print('Hello World!')
        if x is None:
            return b'\x00\x00\x00\x00'
        return struct.pack('>i', struct.unpack('>i', inet_aton(x)) ^ MAGIC_COOKIE)

class STUNXorMappedAddress(STUNGenericTlv):
    name = 'STUN XOR Mapped Address'
    fields_desc = [XShortField('type', 32), ShortField('length', 8), ByteField('RESERVED', 0), ByteEnumField('address_family', 1, _xor_mapped_address_family), XorPort('xport', 0), XorIp('xip', 0)]

class STUNUseCandidate(STUNGenericTlv):
    name = 'STUN Use Candidate'
    fields_desc = [XShortField('type', 37), FieldLenField('length', 0, length_of='value'), PadField(StrLenField('value', '', length_from=lambda pkt: pkt.length), align=4)]

class STUNFingerprint(STUNGenericTlv):
    name = 'STUN Fingerprint'
    fields_desc = [XShortField('type', 32808), ShortField('length', 4), XIntField('crc_32', None)]

class STUNIceControlling(STUNGenericTlv):
    name = 'STUN ICE-controlling'
    fields_desc = [XShortField('type', 32810), ShortField('length', 8), XLongField('tie_breaker', None)]

class STUNGoogNetworkInfo(STUNGenericTlv):
    name = 'STUN Google Network Info'
    fields_desc = [XShortField('type', 49239), ShortField('length', 4), ShortField('network_id', 0), ShortField('network_cost', 999)]
_stun_tlv_class = {6: STUNUsername, 8: STUNMessageIntegrity, 32: STUNXorMappedAddress, 37: STUNUseCandidate, 36: STUNPriority, 32808: STUNFingerprint, 32810: STUNIceControlling, 49239: STUNGoogNetworkInfo}
_stun_tlv_attribute_types = {'MAPPED-ADDRESS': 1, 'USERNAME': 6, 'MESSAGE-INTEGRITY': 8, 'ERROR-CODE': 9, 'UNKNOWN-ATTRIBUTES': 10, 'REALM': 20, 'NONCE': 21, 'XOR-MAPPED-ADDRESS': 32, 'PRIORITY': 36, 'USE-CANDIDATE': 37, 'SOFTWARE': 32802, 'ALTERNATE-SERVER': 32803, 'FINGERPRINT': 32808, 'ICE-CONTROLLED': 32809, 'ICE-CONTROLLING': 32810, 'GOOG-NETWORK-INFO': 49239}

class STUN(Packet):
    description = ''
    fields_desc = [BitField('RESERVED', 0, size=2), BitEnumField('stun_message_type', None, 14, _stun_message_type), LenField('length', None, fmt='!h'), XIntField('magic_cookie', MAGIC_COOKIE), XBitField('transaction_id', None, 96), PacketListField('attributes', [], STUNGenericTlv)]

    def post_build(self, pkt, pay):
        if False:
            i = 10
            return i + 15
        pkt += pay
        if self.length is None:
            pkt = pkt[:2] + struct.pack('!h', len(pkt) - 20) + pkt[4:]
        for attr in self.tlvlist:
            if isinstance(attr, STUNMessageIntegrity):
                pass
        return pkt
bind_layers(UDP, STUN, dport=3478)
bind_layers(TCP, STUN, dport=3478)