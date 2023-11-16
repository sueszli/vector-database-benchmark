"""
RSVP layer
"""
from scapy.compat import chb
from scapy.packet import Packet, bind_layers
from scapy.fields import BitField, ByteEnumField, ByteField, FieldLenField, IPField, ShortField, StrLenField, XByteField, XShortField
from scapy.layers.inet import IP, checksum
rsvpmsgtypes = {1: 'Path', 2: 'Reservation request', 3: 'Path error', 4: 'Reservation request error', 5: 'Path teardown', 6: 'Reservation teardown', 7: 'Reservation request acknowledgment'}

class RSVP(Packet):
    name = 'RSVP'
    fields_desc = [BitField('Version', 1, 4), BitField('Flags', 1, 4), ByteEnumField('Class', 1, rsvpmsgtypes), XShortField('chksum', None), ByteField('TTL', 1), XByteField('dataofs', 0), ShortField('Length', None)]

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        p += pay
        if self.Length is None:
            tmp_len = len(p)
            tmp_p = p[:6] + chb(tmp_len >> 8 & 255) + chb(tmp_len & 255)
            p = tmp_p + p[8:]
        if self.chksum is None:
            ck = checksum(p)
            p = p[:2] + chb(ck >> 8) + chb(ck & 255) + p[4:]
        return p
rsvptypes = {1: 'Session', 3: 'HOP', 4: 'INTEGRITY', 5: 'TIME_VALUES', 6: 'ERROR_SPEC', 7: 'SCOPE', 8: 'STYLE', 9: 'FLOWSPEC', 10: 'FILTER_SPEC', 11: 'SENDER_TEMPLATE', 12: 'SENDER_TSPEC', 13: 'ADSPEC', 14: 'POLICY_DATA', 15: 'RESV_CONFIRM', 16: 'RSVP_LABEL', 17: 'HOP_COUNT', 18: 'STRICT_SOURCE_ROUTE', 19: 'LABEL_REQUEST', 20: 'EXPLICIT_ROUTE', 21: 'ROUTE_RECORD', 22: 'HELLO', 23: 'MESSAGE_ID', 24: 'MESSAGE_ID_ACK', 25: 'MESSAGE_ID_LIST', 30: 'DIAGNOSTIC', 31: 'ROUTE', 32: 'DIAG_RESPONSE', 33: 'DIAG_SELECT', 34: 'RECOVERY_LABEL', 35: 'UPSTREAM_LABEL', 36: 'LABEL_SET', 37: 'PROTECTION', 38: 'PRIMARY PATH ROUTE', 42: 'DSBM IP ADDRESS', 43: 'SBM_PRIORITY', 44: 'DSBM TIMER INTERVALS', 45: 'SBM_INFO', 50: 'S2L_SUB_LSP', 63: 'DETOUR', 64: 'CHALLENGE', 65: 'DIFF-SERV', 66: 'CLASSTYPE', 67: 'LSP_REQUIRED_ATTRIBUTES', 128: 'NODE_CHAR', 129: 'SUGGESTED_LABEL', 130: 'ACCEPTABLE_LABEL_SET', 131: 'RESTART_CA', 132: 'SESSION-OF-INTEREST', 133: 'LINK_CAPABILITY', 134: 'Capability Object', 161: 'RSVP_HOP_L2', 162: 'LAN_NHOP_L2', 163: 'LAN_NHOP_L3', 164: 'LAN_LOOPBACK', 165: 'TCLASS', 192: 'TUNNEL', 193: 'LSP_TUNNEL_INTERFACE_ID', 194: 'USER_ERROR_SPEC', 195: 'NOTIFY_REQUEST', 196: 'ADMIN-STATUS', 197: 'LSP_ATTRIBUTES', 198: 'ALARM_SPEC', 199: 'ASSOCIATION', 200: 'SECONDARY_EXPLICIT_ROUTE', 201: 'SECONDARY_RECORD_ROUTE', 205: 'FAST_REROUTE', 207: 'SESSION_ATTRIBUTE', 225: 'DCLASS', 226: 'PACKETCABLE EXTENSIONS', 227: 'ATM_SERVICECLASS', 228: 'CALL_OPS (ASON)', 229: 'GENERALIZED_UNI', 230: 'CALL_ID', 231: '3GPP2_Object', 232: 'EXCLUDE_ROUTE'}

class RSVP_Object(Packet):
    name = 'RSVP_Object'
    fields_desc = [ShortField('Length', 4), ByteEnumField('Class', 1, rsvptypes), ByteField('C_Type', 1)]

    def guess_payload_class(self, payload):
        if False:
            print('Hello World!')
        if self.Class == 3:
            return RSVP_HOP
        elif self.Class == 5:
            return RSVP_Time
        elif self.Class == 12:
            return RSVP_SenderTSPEC
        elif self.Class == 19:
            return RSVP_LabelReq
        elif self.Class == 207:
            return RSVP_SessionAttrb
        else:
            return RSVP_Data

class RSVP_Data(Packet):
    name = 'Data'
    overload_fields = {RSVP_Object: {'Class': 1}}
    fields_desc = [StrLenField('Data', '', length_from=lambda pkt: pkt.underlayer.Length - 4)]

    def default_payload_class(self, payload):
        if False:
            for i in range(10):
                print('nop')
        return RSVP_Object

class RSVP_HOP(Packet):
    name = 'HOP'
    overload_fields = {RSVP_Object: {'Class': 3}}
    fields_desc = [IPField('neighbor', '0.0.0.0'), BitField('inface', 1, 32)]

    def default_payload_class(self, payload):
        if False:
            return 10
        return RSVP_Object

class RSVP_Time(Packet):
    name = 'Time Val'
    overload_fields = {RSVP_Object: {'Class': 5}}
    fields_desc = [BitField('refresh', 1, 32)]

    def default_payload_class(self, payload):
        if False:
            for i in range(10):
                print('nop')
        return RSVP_Object

class RSVP_SenderTSPEC(Packet):
    name = 'Sender_TSPEC'
    overload_fields = {RSVP_Object: {'Class': 12}}
    fields_desc = [ByteField('Msg_Format', 0), ByteField('reserve', 0), ShortField('Data_Length', 4), ByteField('Srv_hdr', 1), ByteField('reserve2', 0), ShortField('Srv_Length', 4), StrLenField('Tokens', '', length_from=lambda pkt: pkt.underlayer.Length - 12)]

    def default_payload_class(self, payload):
        if False:
            print('Hello World!')
        return RSVP_Object

class RSVP_LabelReq(Packet):
    name = 'Label Req'
    overload_fields = {RSVP_Object: {'Class': 19}}
    fields_desc = [ShortField('reserve', 1), ShortField('L3PID', 1)]

    def default_payload_class(self, payload):
        if False:
            while True:
                i = 10
        return RSVP_Object

class RSVP_SessionAttrb(Packet):
    name = 'Session_Attribute'
    overload_fields = {RSVP_Object: {'Class': 207}}
    fields_desc = [ByteField('Setup_priority', 1), ByteField('Hold_priority', 1), ByteField('flags', 1), FieldLenField('Name_length', None, length_of='Name'), StrLenField('Name', '', length_from=lambda pkt: pkt.Name_length)]

    def default_payload_class(self, payload):
        if False:
            return 10
        return RSVP_Object
bind_layers(IP, RSVP, {'proto': 46})
bind_layers(RSVP, RSVP_Object)