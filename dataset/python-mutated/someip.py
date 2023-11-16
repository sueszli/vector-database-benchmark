import ctypes
import collections
import struct
from scapy.layers.inet import TCP, UDP
from scapy.layers.inet6 import IP6Field
from scapy.compat import raw, orb
from scapy.config import conf
from scapy.packet import Packet, Raw, bind_top_down, bind_bottom_up
from scapy.fields import XShortField, BitEnumField, ConditionalField, BitField, XBitField, IntField, XByteField, ByteEnumField, ShortField, X3BytesField, StrLenField, IPField, FieldLenField, PacketListField, XIntField

class SOMEIP(Packet):
    """ SOME/IP Packet."""
    PROTOCOL_VERSION = 1
    INTERFACE_VERSION = 1
    LEN_OFFSET = 8
    LEN_OFFSET_TP = 12
    TYPE_REQUEST = 0
    TYPE_REQUEST_NO_RET = 1
    TYPE_NOTIFICATION = 2
    TYPE_REQUEST_ACK = 64
    TYPE_REQUEST_NORET_ACK = 65
    TYPE_NOTIFICATION_ACK = 66
    TYPE_RESPONSE = 128
    TYPE_ERROR = 129
    TYPE_RESPONSE_ACK = 192
    TYPE_ERROR_ACK = 193
    TYPE_TP_REQUEST = 32
    TYPE_TP_REQUEST_NO_RET = 33
    TYPE_TP_NOTIFICATION = 34
    TYPE_TP_RESPONSE = 160
    TYPE_TP_ERROR = 161
    RET_E_OK = 0
    RET_E_NOT_OK = 1
    RET_E_UNKNOWN_SERVICE = 2
    RET_E_UNKNOWN_METHOD = 3
    RET_E_NOT_READY = 4
    RET_E_NOT_REACHABLE = 5
    RET_E_TIMEOUT = 6
    RET_E_WRONG_PROTOCOL_V = 7
    RET_E_WRONG_INTERFACE_V = 8
    RET_E_MALFORMED_MSG = 9
    RET_E_WRONG_MESSAGE_TYPE = 10
    _OVERALL_LEN_NOPAYLOAD = 16
    name = 'SOME/IP'
    fields_desc = [XShortField('srv_id', 0), BitEnumField('sub_id', 0, 1, {0: 'METHOD_ID', 1: 'EVENT_ID'}), ConditionalField(XBitField('method_id', 0, 15), lambda pkt: pkt.sub_id == 0), ConditionalField(XBitField('event_id', 0, 15), lambda pkt: pkt.sub_id == 1), IntField('len', None), XShortField('client_id', 0), XShortField('session_id', 0), XByteField('proto_ver', PROTOCOL_VERSION), XByteField('iface_ver', INTERFACE_VERSION), ByteEnumField('msg_type', TYPE_REQUEST, {TYPE_REQUEST: 'REQUEST', TYPE_REQUEST_NO_RET: 'REQUEST_NO_RETURN', TYPE_NOTIFICATION: 'NOTIFICATION', TYPE_REQUEST_ACK: 'REQUEST_ACK', TYPE_REQUEST_NORET_ACK: 'REQUEST_NO_RETURN_ACK', TYPE_NOTIFICATION_ACK: 'NOTIFICATION_ACK', TYPE_RESPONSE: 'RESPONSE', TYPE_ERROR: 'ERROR', TYPE_RESPONSE_ACK: 'RESPONSE_ACK', TYPE_ERROR_ACK: 'ERROR_ACK', TYPE_TP_REQUEST: 'TP_REQUEST', TYPE_TP_REQUEST_NO_RET: 'TP_REQUEST_NO_RETURN', TYPE_TP_NOTIFICATION: 'TP_NOTIFICATION', TYPE_TP_RESPONSE: 'TP_RESPONSE', TYPE_TP_ERROR: 'TP_ERROR'}), ByteEnumField('retcode', 0, {RET_E_OK: 'E_OK', RET_E_NOT_OK: 'E_NOT_OK', RET_E_UNKNOWN_SERVICE: 'E_UNKNOWN_SERVICE', RET_E_UNKNOWN_METHOD: 'E_UNKNOWN_METHOD', RET_E_NOT_READY: 'E_NOT_READY', RET_E_NOT_REACHABLE: 'E_NOT_REACHABLE', RET_E_TIMEOUT: 'E_TIMEOUT', RET_E_WRONG_PROTOCOL_V: 'E_WRONG_PROTOCOL_VERSION', RET_E_WRONG_INTERFACE_V: 'E_WRONG_INTERFACE_VERSION', RET_E_MALFORMED_MSG: 'E_MALFORMED_MESSAGE', RET_E_WRONG_MESSAGE_TYPE: 'E_WRONG_MESSAGE_TYPE'}), ConditionalField(BitField('offset', 0, 28), lambda pkt: SOMEIP._is_tp(pkt)), ConditionalField(BitField('res', 0, 3), lambda pkt: SOMEIP._is_tp(pkt)), ConditionalField(BitField('more_seg', 0, 1), lambda pkt: SOMEIP._is_tp(pkt))]

    def post_build(self, pkt, pay):
        if False:
            while True:
                i = 10
        length = self.len
        if length is None:
            if SOMEIP._is_tp(self):
                length = SOMEIP.LEN_OFFSET_TP + len(pay)
            else:
                length = SOMEIP.LEN_OFFSET + len(pay)
            pkt = pkt[:4] + struct.pack('!I', length) + pkt[8:]
        return pkt + pay

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, type(self)):
            if self.msg_type in [SOMEIP.TYPE_REQUEST_NO_RET, SOMEIP.TYPE_REQUEST_NORET_ACK, SOMEIP.TYPE_NOTIFICATION, SOMEIP.TYPE_TP_REQUEST_NO_RET, SOMEIP.TYPE_TP_NOTIFICATION]:
                return 0
            return self.payload.answers(other.payload)
        return 0

    @staticmethod
    def _is_tp(pkt):
        if False:
            i = 10
            return i + 15
        'Returns true if pkt is using SOMEIP-TP, else returns false.'
        tp = [SOMEIP.TYPE_TP_REQUEST, SOMEIP.TYPE_TP_REQUEST_NO_RET, SOMEIP.TYPE_TP_NOTIFICATION, SOMEIP.TYPE_TP_RESPONSE, SOMEIP.TYPE_TP_ERROR]
        if isinstance(pkt, Packet):
            return pkt.msg_type in tp
        else:
            return pkt[15] in tp

    def fragment(self, fragsize=1392):
        if False:
            while True:
                i = 10
        'Fragment SOME/IP-TP'
        fnb = 0
        fl = self
        lst = list()
        while fl.underlayer is not None:
            fnb += 1
            fl = fl.underlayer
        for p in fl:
            s = raw(p[fnb].payload)
            nb = (len(s) + fragsize) // fragsize
            for i in range(nb):
                q = p.copy()
                del q[fnb].payload
                q[fnb].len = SOMEIP.LEN_OFFSET_TP + len(s[i * fragsize:(i + 1) * fragsize])
                q[fnb].more_seg = 1
                if i == nb - 1:
                    q[fnb].more_seg = 0
                q[fnb].offset += i * fragsize // 16
                r = conf.raw_layer(load=s[i * fragsize:(i + 1) * fragsize])
                r.overload_fields = p[fnb].payload.overload_fields.copy()
                q.add_payload(r)
                lst.append(q)
        return lst

def _bind_someip_layers():
    if False:
        i = 10
        return i + 15
    bind_top_down(UDP, SOMEIP, sport=30490, dport=30490)
    for i in range(15):
        bind_bottom_up(UDP, SOMEIP, sport=30490 + i)
        bind_bottom_up(TCP, SOMEIP, sport=30490 + i)
        bind_bottom_up(UDP, SOMEIP, dport=30490 + i)
        bind_bottom_up(TCP, SOMEIP, dport=30490 + i)
_bind_someip_layers()

class _SDPacketBase(Packet):
    """ base class to be used among all SD Packet definitions."""

    def extract_padding(self, s):
        if False:
            i = 10
            return i + 15
        return ('', s)
SDENTRY_TYPE_SRV_FINDSERVICE = 0
SDENTRY_TYPE_SRV_OFFERSERVICE = 1
SDENTRY_TYPE_SRV = (SDENTRY_TYPE_SRV_FINDSERVICE, SDENTRY_TYPE_SRV_OFFERSERVICE)
SDENTRY_TYPE_EVTGRP_SUBSCRIBE = 6
SDENTRY_TYPE_EVTGRP_SUBSCRIBE_ACK = 7
SDENTRY_TYPE_EVTGRP = (SDENTRY_TYPE_EVTGRP_SUBSCRIBE, SDENTRY_TYPE_EVTGRP_SUBSCRIBE_ACK)
SDENTRY_OVERALL_LEN = 16

def _MAKE_SDENTRY_COMMON_FIELDS_DESC(type):
    if False:
        return 10
    return [XByteField('type', type), XByteField('index_1', 0), XByteField('index_2', 0), XBitField('n_opt_1', 0, 4), XBitField('n_opt_2', 0, 4), XShortField('srv_id', 0), XShortField('inst_id', 0), XByteField('major_ver', 0), X3BytesField('ttl', 0)]

class SDEntry_Service(_SDPacketBase):
    name = 'Service Entry'
    fields_desc = _MAKE_SDENTRY_COMMON_FIELDS_DESC(SDENTRY_TYPE_SRV_FINDSERVICE)
    fields_desc += [XIntField('minor_ver', 0)]

class SDEntry_EventGroup(_SDPacketBase):
    name = 'Eventgroup Entry'
    fields_desc = _MAKE_SDENTRY_COMMON_FIELDS_DESC(SDENTRY_TYPE_EVTGRP_SUBSCRIBE)
    fields_desc += [XBitField('res', 0, 12), XBitField('cnt', 0, 4), XShortField('eventgroup_id', 0)]

def _sdentry_class(payload, **kargs):
    if False:
        i = 10
        return i + 15
    TYPE_PAYLOAD_I = 0
    pl_type = orb(payload[TYPE_PAYLOAD_I])
    cls = None
    if pl_type in SDENTRY_TYPE_SRV:
        cls = SDEntry_Service
    elif pl_type in SDENTRY_TYPE_EVTGRP:
        cls = SDEntry_EventGroup
    return cls(payload, **kargs)

def _sdoption_class(payload, **kargs):
    if False:
        i = 10
        return i + 15
    pl_type = orb(payload[2])
    cls = {SDOPTION_CFG_TYPE: SDOption_Config, SDOPTION_LOADBALANCE_TYPE: SDOption_LoadBalance, SDOPTION_IP4_ENDPOINT_TYPE: SDOption_IP4_EndPoint, SDOPTION_IP4_MCAST_TYPE: SDOption_IP4_Multicast, SDOPTION_IP4_SDENDPOINT_TYPE: SDOption_IP4_SD_EndPoint, SDOPTION_IP6_ENDPOINT_TYPE: SDOption_IP6_EndPoint, SDOPTION_IP6_MCAST_TYPE: SDOption_IP6_Multicast, SDOPTION_IP6_SDENDPOINT_TYPE: SDOption_IP6_SD_EndPoint}.get(pl_type, Raw)
    return cls(payload, **kargs)
SDOPTION_CFG_TYPE = 1
SDOPTION_LOADBALANCE_TYPE = 2
SDOPTION_LOADBALANCE_LEN = 5
SDOPTION_IP4_ENDPOINT_TYPE = 4
SDOPTION_IP4_ENDPOINT_LEN = 9
SDOPTION_IP4_MCAST_TYPE = 20
SDOPTION_IP4_MCAST_LEN = 9
SDOPTION_IP4_SDENDPOINT_TYPE = 36
SDOPTION_IP4_SDENDPOINT_LEN = 9
SDOPTION_IP6_ENDPOINT_TYPE = 6
SDOPTION_IP6_ENDPOINT_LEN = 21
SDOPTION_IP6_MCAST_TYPE = 22
SDOPTION_IP6_MCAST_LEN = 21
SDOPTION_IP6_SDENDPOINT_TYPE = 38
SDOPTION_IP6_SDENDPOINT_LEN = 21

def _MAKE_COMMON_SDOPTION_FIELDS_DESC(type, length=None):
    if False:
        return 10
    return [ShortField('len', length), XByteField('type', type), XByteField('res_hdr', 0)]

def _MAKE_COMMON_IP_SDOPTION_FIELDS_DESC():
    if False:
        for i in range(10):
            print('nop')
    return [XByteField('res_tail', 0), ByteEnumField('l4_proto', 17, {6: 'TCP', 17: 'UDP'}), ShortField('port', 0)]

class SDOption_Config(_SDPacketBase):
    name = 'Config Option'
    fields_desc = _MAKE_COMMON_SDOPTION_FIELDS_DESC(SDOPTION_CFG_TYPE) + [StrLenField('cfg_str', '\x00', length_from=lambda pkt: pkt.len - 1)]

    def post_build(self, pkt, pay):
        if False:
            while True:
                i = 10
        if self.len is None:
            length = len(self.cfg_str) + 1
            pkt = struct.pack('!H', length) + pkt[2:]
        return pkt + pay

    @staticmethod
    def make_string(data):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(data, dict):
            data = data.items()
        data = ('{}={}'.format(k, v) for (k, v) in data)
        data = ('{}{}'.format(chr(len(v)), v) for v in data)
        data = ''.join(data)
        data += '\x00'
        return data.encode('utf8')

class SDOption_LoadBalance(_SDPacketBase):
    name = 'LoadBalance Option'
    fields_desc = _MAKE_COMMON_SDOPTION_FIELDS_DESC(SDOPTION_LOADBALANCE_TYPE, SDOPTION_LOADBALANCE_LEN)
    fields_desc += [ShortField('priority', 0), ShortField('weight', 0)]

class SDOption_IP4_EndPoint(_SDPacketBase):
    name = 'IP4 EndPoint Option'
    fields_desc = _MAKE_COMMON_SDOPTION_FIELDS_DESC(SDOPTION_IP4_ENDPOINT_TYPE, SDOPTION_IP4_ENDPOINT_LEN)
    fields_desc += [IPField('addr', '0.0.0.0')] + _MAKE_COMMON_IP_SDOPTION_FIELDS_DESC()

class SDOption_IP4_Multicast(_SDPacketBase):
    name = 'IP4 Multicast Option'
    fields_desc = _MAKE_COMMON_SDOPTION_FIELDS_DESC(SDOPTION_IP4_MCAST_TYPE, SDOPTION_IP4_MCAST_LEN)
    fields_desc += [IPField('addr', '0.0.0.0')] + _MAKE_COMMON_IP_SDOPTION_FIELDS_DESC()

class SDOption_IP4_SD_EndPoint(_SDPacketBase):
    name = 'IP4 SDEndPoint Option'
    fields_desc = _MAKE_COMMON_SDOPTION_FIELDS_DESC(SDOPTION_IP4_SDENDPOINT_TYPE, SDOPTION_IP4_SDENDPOINT_LEN)
    fields_desc += [IPField('addr', '0.0.0.0')] + _MAKE_COMMON_IP_SDOPTION_FIELDS_DESC()

class SDOption_IP6_EndPoint(_SDPacketBase):
    name = 'IP6 EndPoint Option'
    fields_desc = _MAKE_COMMON_SDOPTION_FIELDS_DESC(SDOPTION_IP6_ENDPOINT_TYPE, SDOPTION_IP6_ENDPOINT_LEN)
    fields_desc += [IP6Field('addr', '::')] + _MAKE_COMMON_IP_SDOPTION_FIELDS_DESC()

class SDOption_IP6_Multicast(_SDPacketBase):
    name = 'IP6 Multicast Option'
    fields_desc = _MAKE_COMMON_SDOPTION_FIELDS_DESC(SDOPTION_IP6_MCAST_TYPE, SDOPTION_IP6_MCAST_LEN)
    fields_desc += [IP6Field('addr', '::')] + _MAKE_COMMON_IP_SDOPTION_FIELDS_DESC()

class SDOption_IP6_SD_EndPoint(_SDPacketBase):
    name = 'IP6 SDEndPoint Option'
    fields_desc = _MAKE_COMMON_SDOPTION_FIELDS_DESC(SDOPTION_IP6_SDENDPOINT_TYPE, SDOPTION_IP6_SDENDPOINT_LEN)
    fields_desc += [IP6Field('addr', '::')] + _MAKE_COMMON_IP_SDOPTION_FIELDS_DESC()

class SD(_SDPacketBase):
    """
    SD Packet

    NOTE :   when adding 'entries' or 'options', do not use list.append()
        method but create a new list
    e.g. :  p = SD()
            p.option_array = [SDOption_Config(),SDOption_IP6_EndPoint()]
    """
    SOMEIP_MSGID_SRVID = 65535
    SOMEIP_MSGID_SUBID = 1
    SOMEIP_MSGID_EVENTID = 256
    SOMEIP_CLIENT_ID = 0
    SOMEIP_MINIMUM_SESSION_ID = 1
    SOMEIP_PROTO_VER = 1
    SOMEIP_IFACE_VER = 1
    SOMEIP_MSG_TYPE = SOMEIP.TYPE_NOTIFICATION
    SOMEIP_RETCODE = SOMEIP.RET_E_OK
    _sdFlag = collections.namedtuple('Flag', 'mask offset')
    FLAGSDEF = {'REBOOT': _sdFlag(mask=128, offset=7), 'UNICAST': _sdFlag(mask=64, offset=6), 'EXPLICIT_INITIAL_DATA_CONTROL': _sdFlag(mask=32, offset=5)}
    name = 'SD'
    fields_desc = [XByteField('flags', 0), X3BytesField('res', 0), FieldLenField('len_entry_array', None, length_of='entry_array', fmt='!I'), PacketListField('entry_array', None, _sdentry_class, length_from=lambda pkt: pkt.len_entry_array), FieldLenField('len_option_array', None, length_of='option_array', fmt='!I'), PacketListField('option_array', None, _sdoption_class, length_from=lambda pkt: pkt.len_option_array)]

    def get_flag(self, name):
        if False:
            for i in range(10):
                print('nop')
        name = name.upper()
        if name in self.FLAGSDEF:
            return (self.flags & self.FLAGSDEF[name].mask) >> self.FLAGSDEF[name].offset
        else:
            return None

    def set_flag(self, name, value):
        if False:
            while True:
                i = 10
        name = name.upper()
        if name in self.FLAGSDEF:
            self.flags = self.flags & ctypes.c_ubyte(~self.FLAGSDEF[name].mask).value | (value & 1) << self.FLAGSDEF[name].offset

    def set_entryArray(self, entry_list):
        if False:
            print('Hello World!')
        if isinstance(entry_list, list):
            self.entry_array = entry_list
        else:
            self.entry_array = [entry_list]

    def set_optionArray(self, option_list):
        if False:
            while True:
                i = 10
        if isinstance(option_list, list):
            self.option_array = option_list
        else:
            self.option_array = [option_list]
bind_top_down(SOMEIP, SD, srv_id=SD.SOMEIP_MSGID_SRVID, sub_id=SD.SOMEIP_MSGID_SUBID, client_id=SD.SOMEIP_CLIENT_ID, session_id=SD.SOMEIP_MINIMUM_SESSION_ID, event_id=SD.SOMEIP_MSGID_EVENTID, proto_ver=SD.SOMEIP_PROTO_VER, iface_ver=SD.SOMEIP_IFACE_VER, msg_type=SD.SOMEIP_MSG_TYPE, retcode=SD.SOMEIP_RETCODE)
bind_bottom_up(SOMEIP, SD, srv_id=SD.SOMEIP_MSGID_SRVID, sub_id=SD.SOMEIP_MSGID_SUBID, event_id=SD.SOMEIP_MSGID_EVENTID, proto_ver=SD.SOMEIP_PROTO_VER, iface_ver=SD.SOMEIP_IFACE_VER, msg_type=SD.SOMEIP_MSG_TYPE, retcode=SD.SOMEIP_RETCODE)