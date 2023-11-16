"""
SCTP (Stream Control Transmission Protocol).
"""
import struct
from scapy.compat import orb, raw
from scapy.volatile import RandBin
from scapy.config import conf
from scapy.packet import Packet, bind_layers
from scapy.fields import BitField, ByteEnumField, Field, FieldLenField, FieldListField, IPField, IntEnumField, IntField, MultipleTypeField, PacketListField, PadField, ShortEnumField, ShortField, StrFixedLenField, StrLenField, XByteField, XIntField, XShortField
from scapy.data import SCTP_SERVICES
from scapy.layers.inet import IP, IPerror
from scapy.layers.inet6 import IP6Field, IPv6, IPerror6
IPPROTO_SCTP = 132
crc32c_table = [0, 4067132163, 3778769143, 324072436, 3348797215, 904991772, 648144872, 3570033899, 2329499855, 2024987596, 1809983544, 2575936315, 1296289744, 3207089363, 2893594407, 1578318884, 274646895, 3795141740, 4049975192, 51262619, 3619967088, 632279923, 922689671, 3298075524, 2592579488, 1760304291, 2075979607, 2312596564, 1562183871, 2943781820, 3156637768, 1313733451, 549293790, 3537243613, 3246849577, 871202090, 3878099393, 357341890, 102525238, 4101499445, 2858735121, 1477399826, 1264559846, 3107202533, 1845379342, 2677391885, 2361733625, 2125378298, 820201905, 3263744690, 3520608582, 598981189, 4151959214, 85089709, 373468761, 3827903834, 3124367742, 1213305469, 1526817161, 2842354314, 2107672161, 2412447074, 2627466902, 1861252501, 1098587580, 3004210879, 2688576843, 1378610760, 2262928035, 1955203488, 1742404180, 2511436119, 3416409459, 969524848, 714683780, 3639785095, 205050476, 4266873199, 3976438427, 526918040, 1361435347, 2739821008, 2954799652, 1114974503, 2529119692, 1691668175, 2005155131, 2247081528, 3690758684, 697762079, 986182379, 3366744552, 476452099, 3993867776, 4250756596, 255256311, 1640403810, 2477592673, 2164122517, 1922457750, 2791048317, 1412925310, 1197962378, 3037525897, 3944729517, 427051182, 170179418, 4165941337, 746937522, 3740196785, 3451792453, 1070968646, 1905808397, 2213795598, 2426610938, 1657317369, 3053634322, 1147748369, 1463399397, 2773627110, 4215344322, 153784257, 444234805, 3893493558, 1021025245, 3467647198, 3722505002, 797665321, 2197175160, 1889384571, 1674398607, 2443626636, 1164749927, 3070701412, 2757221520, 1446797203, 137323447, 4198817972, 3910406976, 461344835, 3484808360, 1037989803, 781091935, 3705997148, 2460548119, 1623424788, 1939049696, 2180517859, 1429367560, 2807687179, 3020495871, 1180866812, 410100952, 3927582683, 4182430767, 186734380, 3756733383, 763408580, 1053836080, 3434856499, 2722870694, 1344288421, 1131464017, 2971354706, 1708204729, 2545590714, 2229949006, 1988219213, 680717673, 3673779818, 3383336350, 1002577565, 4010310262, 493091189, 238226049, 4233660802, 2987750089, 1082061258, 1395524158, 2705686845, 1972364758, 2279892693, 2494862625, 1725896226, 952904198, 3399985413, 3656866545, 731699698, 4283874585, 222117402, 510512622, 3959836397, 3280807620, 837199303, 582374963, 3504198960, 68661723, 4135334616, 3844915500, 390545967, 1230274059, 3141532936, 2825850620, 1510247935, 2395924756, 2091215383, 1878366691, 2644384480, 3553878443, 565732008, 854102364, 3229815391, 340358836, 3861050807, 4117890627, 119113024, 1493875044, 2875275879, 3090270611, 1247431312, 2660249211, 1828433272, 2141937292, 2378227087, 3811616794, 291187481, 34330861, 4032846830, 615137029, 3603020806, 3314634738, 939183345, 1776939221, 2609017814, 2295496738, 2058945313, 2926798794, 1545135305, 1330124605, 3173225534, 4084100981, 17165430, 307568514, 3762199681, 888469610, 3332340585, 3587147933, 665062302, 2042050490, 2346497209, 2559330125, 1793573966, 3190661285, 1279665062, 1595330642, 2910671697]

def crc32c(buf):
    if False:
        return 10
    crc = 4294967295
    for c in buf:
        crc = crc >> 8 ^ crc32c_table[(crc ^ orb(c)) & 255]
    crc = ~crc & 4294967295
    return struct.unpack('>I', struct.pack('<I', crc))[0]
'\nBASE = 65521 # largest prime smaller than 65536\ndef update_adler32(adler, buf):\n    s1 = adler & 0xffff\n    s2 = (adler >> 16) & 0xffff\n    print s1,s2\n\n    for c in buf:\n        print orb(c)\n        s1 = (s1 + orb(c)) % BASE\n        s2 = (s2 + s1) % BASE\n        print s1,s2\n    return (s2 << 16) + s1\n\ndef sctp_checksum(buf):\n    return update_adler32(1, buf)\n'
hmactypes = {0: 'Reserved1', 1: 'SHA-1', 2: 'Reserved2', 3: 'SHA-256'}
sctpchunktypescls = {0: 'SCTPChunkData', 1: 'SCTPChunkInit', 2: 'SCTPChunkInitAck', 3: 'SCTPChunkSACK', 4: 'SCTPChunkHeartbeatReq', 5: 'SCTPChunkHeartbeatAck', 6: 'SCTPChunkAbort', 7: 'SCTPChunkShutdown', 8: 'SCTPChunkShutdownAck', 9: 'SCTPChunkError', 10: 'SCTPChunkCookieEcho', 11: 'SCTPChunkCookieAck', 14: 'SCTPChunkShutdownComplete', 15: 'SCTPChunkAuthentication', 64: 'SCTPChunkIData', 130: 'SCTPChunkReConfig', 132: 'SCTPChunkPad', 128: 'SCTPChunkAddressConfAck', 192: 'SCTPChunkForwardTSN', 193: 'SCTPChunkAddressConf', 194: 'SCTPChunkIForwardTSN'}
sctpchunktypes = {0: 'data', 1: 'init', 2: 'init-ack', 3: 'sack', 4: 'heartbeat-req', 5: 'heartbeat-ack', 6: 'abort', 7: 'shutdown', 8: 'shutdown-ack', 9: 'error', 10: 'cookie-echo', 11: 'cookie-ack', 14: 'shutdown-complete', 15: 'authentication', 64: 'i-data', 130: 're-config', 132: 'pad', 128: 'address-configuration-ack', 192: 'forward-tsn', 193: 'address-configuration', 194: 'i-forward-tsn'}
sctpchunkparamtypescls = {1: 'SCTPChunkParamHeartbeatInfo', 5: 'SCTPChunkParamIPv4Addr', 6: 'SCTPChunkParamIPv6Addr', 7: 'SCTPChunkParamStateCookie', 8: 'SCTPChunkParamUnrocognizedParam', 9: 'SCTPChunkParamCookiePreservative', 11: 'SCTPChunkParamHostname', 12: 'SCTPChunkParamSupportedAddrTypes', 13: 'SCTPChunkParamOutgoingSSNResetRequest', 14: 'SCTPChunkParamIncomingSSNResetRequest', 15: 'SCTPChunkParamSSNTSNResetRequest', 16: 'SCTPChunkParamReConfigurationResponse', 17: 'SCTPChunkParamAddOutgoingStreamRequest', 18: 'SCTPChunkParamAddIncomingStreamRequest', 32768: 'SCTPChunkParamECNCapable', 32770: 'SCTPChunkParamRandom', 32771: 'SCTPChunkParamChunkList', 32772: 'SCTPChunkParamRequestedHMACFunctions', 32776: 'SCTPChunkParamSupportedExtensions', 49152: 'SCTPChunkParamFwdTSN', 49153: 'SCTPChunkParamAddIPAddr', 49154: 'SCTPChunkParamDelIPAddr', 49155: 'SCTPChunkParamErrorIndication', 49156: 'SCTPChunkParamSetPrimaryAddr', 49157: 'SCTPChunkParamSuccessIndication', 49158: 'SCTPChunkParamAdaptationLayer'}
sctpchunkparamtypes = {1: 'heartbeat-info', 5: 'IPv4', 6: 'IPv6', 7: 'state-cookie', 8: 'unrecognized-param', 9: 'cookie-preservative', 11: 'hostname', 12: 'addrtypes', 13: 'out-ssn-reset-req', 14: 'in-ssn-reset-req', 15: 'ssn-tsn-reset-req', 16: 're-configuration-response', 17: 'add-outgoing-stream-req', 18: 'add-incoming-stream-req', 32768: 'ecn-capable', 32770: 'random', 32771: 'chunk-list', 32772: 'requested-HMAC-functions', 32776: 'supported-extensions', 49152: 'fwd-tsn-supported', 49153: 'add-IP', 49154: 'del-IP', 49155: 'error-indication', 49156: 'set-primary-addr', 49157: 'success-indication', 49158: 'adaptation-layer'}

class _SCTPChunkGuessPayload:

    def default_payload_class(self, p):
        if False:
            while True:
                i = 10
        if len(p) < 4:
            return conf.padding_layer
        else:
            t = orb(p[0])
            return globals().get(sctpchunktypescls.get(t, 'Raw'), conf.raw_layer)

class SCTP(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ShortEnumField('sport', 0, SCTP_SERVICES), ShortEnumField('dport', 0, SCTP_SERVICES), XIntField('tag', 0), XIntField('chksum', None)]

    def answers(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, SCTP):
            return 0
        if conf.checkIPsrc:
            if not (self.sport == other.dport and self.dport == other.sport):
                return 0
        return 1

    def post_build(self, p, pay):
        if False:
            while True:
                i = 10
        p += pay
        if self.chksum is None:
            crc = crc32c(raw(p))
            p = p[:8] + struct.pack('>I', crc) + p[12:]
        return p

class SCTPerror(SCTP):
    name = 'SCTP in ICMP'

    def answers(self, other):
        if False:
            return 10
        if not isinstance(other, SCTP):
            return 0
        if conf.checkIPsrc:
            if not (self.sport == other.sport and self.dport == other.dport):
                return 0
        return 1

    def mysummary(self):
        if False:
            i = 10
            return i + 15
        return Packet.mysummary(self)
resultcode = {0: 'Success - Nothing to do', 1: 'Success - Performed', 2: 'Denied', 3: 'Error - Wrong SSN', 4: 'Error - Request already in progress', 5: 'Error - Bad Sequence Number', 6: 'In Progress'}

class ChunkParamField(PacketListField):

    def __init__(self, name, default, count_from=None, length_from=None):
        if False:
            return 10
        PacketListField.__init__(self, name, default, conf.raw_layer, count_from=count_from, length_from=length_from)

    def m2i(self, p, m):
        if False:
            return 10
        cls = conf.raw_layer
        if len(m) >= 4:
            t = orb(m[0]) * 256 + orb(m[1])
            cls = globals().get(sctpchunkparamtypescls.get(t, 'Raw'), conf.raw_layer)
        return cls(m)

class _SCTPChunkParam:

    def extract_padding(self, s):
        if False:
            while True:
                i = 10
        return (b'', s[:])

class SCTPChunkParamHeartbeatInfo(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 1, sctpchunkparamtypes), FieldLenField('len', None, length_of='data', adjust=lambda pkt, x: x + 4), PadField(StrLenField('data', '', length_from=lambda pkt: pkt.len - 4), 4, padwith=b'\x00')]

class SCTPChunkParamIPv4Addr(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 5, sctpchunkparamtypes), ShortField('len', 8), IPField('addr', '127.0.0.1')]

class SCTPChunkParamIPv6Addr(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 6, sctpchunkparamtypes), ShortField('len', 20), IP6Field('addr', '::1')]

class SCTPChunkParamStateCookie(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 7, sctpchunkparamtypes), FieldLenField('len', None, length_of='cookie', adjust=lambda pkt, x: x + 4), PadField(StrLenField('cookie', '', length_from=lambda pkt: pkt.len - 4), 4, padwith=b'\x00')]

class SCTPChunkParamUnrocognizedParam(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 8, sctpchunkparamtypes), FieldLenField('len', None, length_of='param', adjust=lambda pkt, x: x + 4), PadField(StrLenField('param', '', length_from=lambda pkt: pkt.len - 4), 4, padwith=b'\x00')]

class SCTPChunkParamCookiePreservative(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 9, sctpchunkparamtypes), ShortField('len', 8), XIntField('sug_cookie_inc', None)]

class SCTPChunkParamHostname(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 11, sctpchunkparamtypes), FieldLenField('len', None, length_of='hostname', adjust=lambda pkt, x: x + 4), PadField(StrLenField('hostname', '', length_from=lambda pkt: pkt.len - 4), 4, padwith=b'\x00')]

class SCTPChunkParamSupportedAddrTypes(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 12, sctpchunkparamtypes), FieldLenField('len', None, length_of='addr_type_list', adjust=lambda pkt, x: x + 4), PadField(FieldListField('addr_type_list', ['IPv4'], ShortEnumField('addr_type', 5, sctpchunkparamtypes), length_from=lambda pkt: pkt.len - 4), 4, padwith=b'\x00')]

class SCTPChunkParamOutSSNResetReq(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 13, sctpchunkparamtypes), FieldLenField('len', None, length_of='stream_num_list', adjust=lambda pkt, x: x + 16), XIntField('re_conf_req_seq_num', None), XIntField('re_conf_res_seq_num', None), XIntField('tsn', None), PadField(FieldListField('stream_num_list', [], XShortField('stream_num', None), length_from=lambda pkt: pkt.len - 16), 4, padwith=b'\x00')]

class SCTPChunkParamInSSNResetReq(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 14, sctpchunkparamtypes), FieldLenField('len', None, length_of='stream_num_list', adjust=lambda pkt, x: x + 8), XIntField('re_conf_req_seq_num', None), PadField(FieldListField('stream_num_list', [], XShortField('stream_num', None), length_from=lambda pkt: pkt.len - 8), 4, padwith=b'\x00')]

class SCTPChunkParamSSNTSNResetReq(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 15, sctpchunkparamtypes), XShortField('len', 8), XIntField('re_conf_req_seq_num', None)]

class SCTPChunkParamReConfigRes(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 16, sctpchunkparamtypes), XShortField('len', 12), XIntField('re_conf_res_seq_num', None), IntEnumField('result', None, resultcode), XIntField('sender_next_tsn', None), XIntField('receiver_next_tsn', None)]

class SCTPChunkParamAddOutgoingStreamReq(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 17, sctpchunkparamtypes), XShortField('len', 12), XIntField('re_conf_req_seq_num', None), XShortField('num_new_stream', None), XShortField('reserved', None)]

class SCTPChunkParamAddIncomingStreamReq(SCTPChunkParamAddOutgoingStreamReq):
    type = 18

class SCTPChunkParamECNCapable(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 32768, sctpchunkparamtypes), ShortField('len', 4)]

class SCTPChunkParamRandom(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 32770, sctpchunkparamtypes), FieldLenField('len', None, length_of='random', adjust=lambda pkt, x: x + 4), PadField(StrLenField('random', RandBin(32), length_from=lambda pkt: pkt.len - 4), 4, padwith=b'\x00')]

class SCTPChunkParamChunkList(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 32771, sctpchunkparamtypes), FieldLenField('len', None, length_of='chunk_list', adjust=lambda pkt, x: x + 4), PadField(FieldListField('chunk_list', None, ByteEnumField('chunk', None, sctpchunktypes), length_from=lambda pkt: pkt.len - 4), 4, padwith=b'\x00')]

class SCTPChunkParamRequestedHMACFunctions(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 32772, sctpchunkparamtypes), FieldLenField('len', None, length_of='HMAC_functions_list', adjust=lambda pkt, x: x + 4), PadField(FieldListField('HMAC_functions_list', ['SHA-1'], ShortEnumField('HMAC_function', 1, hmactypes), length_from=lambda pkt: pkt.len - 4), 4, padwith=b'\x00')]

class SCTPChunkParamSupportedExtensions(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 32776, sctpchunkparamtypes), FieldLenField('len', None, length_of='supported_extensions', adjust=lambda pkt, x: x + 4), PadField(FieldListField('supported_extensions', ['authentication', 'address-configuration', 'address-configuration-ack'], ByteEnumField('supported_extensions', None, sctpchunktypes), length_from=lambda pkt: pkt.len - 4), 4, padwith=b'\x00')]

class SCTPChunkParamFwdTSN(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 49152, sctpchunkparamtypes), ShortField('len', 4)]

class SCTPChunkParamAddIPAddr(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 49153, sctpchunkparamtypes), FieldLenField('len', None, length_of='addr', adjust=lambda pkt, x: x + 12), XIntField('correlation_id', None), ShortEnumField('addr_type', 5, sctpchunkparamtypes), FieldLenField('addr_len', None, length_of='addr', adjust=lambda pkt, x: x + 4), MultipleTypeField([(IPField('addr', '127.0.0.1'), lambda p: p.addr_type == 5), (IP6Field('addr', '::1'), lambda p: p.addr_type == 6)], StrFixedLenField('addr', '', length_from=lambda pkt: pkt.addr_len))]

class SCTPChunkParamDelIPAddr(SCTPChunkParamAddIPAddr):
    type = 49154

class SCTPChunkParamErrorIndication(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 49155, sctpchunkparamtypes), FieldLenField('len', None, length_of='error_causes', adjust=lambda pkt, x: x + 8), XIntField('correlation_id', None), PadField(StrLenField('error_causes', '', length_from=lambda pkt: pkt.len - 4), 4, padwith=b'\x00')]

class SCTPChunkParamSetPrimaryAddr(SCTPChunkParamAddIPAddr):
    type = 49156

class SCTPChunkParamSuccessIndication(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 49157, sctpchunkparamtypes), ShortField('len', 8), XIntField('correlation_id', None)]

class SCTPChunkParamAdaptationLayer(_SCTPChunkParam, Packet):
    fields_desc = [ShortEnumField('type', 49158, sctpchunkparamtypes), ShortField('len', 8), XIntField('indication', None)]
SCTP_PAYLOAD_PROTOCOL_INDENTIFIERS = {0: 'Reserved', 1: 'IUA', 2: 'M2UA', 3: 'M3UA', 4: 'SUA', 5: 'M2PA', 6: 'V5UA', 7: 'H.248', 8: 'BICC/Q.2150.3', 9: 'TALI', 10: 'DUA', 11: 'ASAP', 12: 'ENRP', 13: 'H.323', 14: 'Q.IPC/Q.2150.3', 15: 'SIMCO', 16: 'DDP Segment Chunk', 17: 'DDP Stream Session Control', 18: 'S1AP', 19: 'RUA', 20: 'HNBAP', 21: 'ForCES-HP', 22: 'ForCES-MP', 23: 'ForCES-LP', 24: 'SBc-AP', 25: 'NBAP', 26: 'Unassigned', 27: 'X2AP', 28: 'IRCP', 29: 'LCS-AP', 30: 'MPICH2', 31: 'SABP', 32: 'FGP', 33: 'PPP', 34: 'CALCAPP', 35: 'SSP', 36: 'NPMP-CONTROL', 37: 'NPMP-DATA', 38: 'ECHO', 39: 'DISCARD', 40: 'DAYTIME', 41: 'CHARGEN', 42: '3GPP RNA', 43: '3GPP M2AP', 44: '3GPP M3AP', 45: 'SSH/SCTP', 46: 'Diameter/SCTP', 47: 'Diameter/DTLS/SCTP', 48: 'R14P', 49: 'Unassigned', 50: 'WebRTC DCEP', 51: 'WebRTC String', 52: 'WebRTC Binary Partial', 53: 'WebRTC Binary', 54: 'WebRTC String Partial', 55: '3GPP PUA', 56: 'WebRTC String Empty', 57: 'WebRTC Binary Empty'}

class SCTPChunkData(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ByteEnumField('type', 0, sctpchunktypes), BitField('reserved', None, 4), BitField('delay_sack', 0, 1), BitField('unordered', 0, 1), BitField('beginning', 0, 1), BitField('ending', 0, 1), FieldLenField('len', None, length_of='data', adjust=lambda pkt, x: x + 16), XIntField('tsn', None), XShortField('stream_id', None), XShortField('stream_seq', None), IntEnumField('proto_id', None, SCTP_PAYLOAD_PROTOCOL_INDENTIFIERS), PadField(StrLenField('data', None, length_from=lambda pkt: pkt.len - 16), 4, padwith=b'\x00')]

class SCTPChunkIData(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ByteEnumField('type', 64, sctpchunktypes), BitField('reserved', None, 4), BitField('delay_sack', 0, 1), BitField('unordered', 0, 1), BitField('beginning', 0, 1), BitField('ending', 0, 1), FieldLenField('len', None, length_of='data', adjust=lambda pkt, x: x + 20), XIntField('tsn', None), XShortField('stream_id', None), XShortField('reserved_16', None), XIntField('message_id', None), MultipleTypeField([(IntEnumField('ppid_fsn', None, SCTP_PAYLOAD_PROTOCOL_INDENTIFIERS), lambda pkt: pkt.beginning == 1), (XIntField('ppid_fsn', None), lambda pkt: pkt.beginning == 0)], XIntField('ppid_fsn', None)), PadField(StrLenField('data', None, length_from=lambda pkt: pkt.len - 20), 4, padwith=b'\x00')]

class SCTPForwardSkip(_SCTPChunkParam, Packet):
    fields_desc = [ShortField('stream_id', None), ShortField('stream_seq', None)]

class SCTPChunkForwardTSN(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ByteEnumField('type', 192, sctpchunktypes), XByteField('flags', None), FieldLenField('len', None, length_of='skips', adjust=lambda pkt, x: x + 8), IntField('new_tsn', None), ChunkParamField('skips', None, length_from=lambda pkt: pkt.len - 8)]

class SCTPIForwardSkip(_SCTPChunkParam, Packet):
    fields_desc = [ShortField('stream_id', None), BitField('reserved', None, 15), BitField('unordered', None, 1), IntField('message_id', None)]

class SCTPChunkIForwardTSN(SCTPChunkForwardTSN):
    type = 194

class SCTPChunkInit(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ByteEnumField('type', 1, sctpchunktypes), XByteField('flags', None), FieldLenField('len', None, length_of='params', adjust=lambda pkt, x: x + 20), XIntField('init_tag', None), IntField('a_rwnd', None), ShortField('n_out_streams', None), ShortField('n_in_streams', None), XIntField('init_tsn', None), ChunkParamField('params', None, length_from=lambda pkt: pkt.len - 20)]

class SCTPChunkInitAck(SCTPChunkInit):
    type = 2

class GapAckField(Field):

    def __init__(self, name, default):
        if False:
            for i in range(10):
                print('nop')
        Field.__init__(self, name, default, '4s')

    def i2m(self, pkt, x):
        if False:
            i = 10
            return i + 15
        if x is None:
            return b'\x00\x00\x00\x00'
        (sta, end) = [int(e) for e in x.split(':')]
        args = tuple(['>HH', sta, end])
        return struct.pack(*args)

    def m2i(self, pkt, x):
        if False:
            i = 10
            return i + 15
        return '%d:%d' % struct.unpack('>HH', x)

    def any2i(self, pkt, x):
        if False:
            return 10
        if isinstance(x, tuple) and len(x) == 2:
            return '%d:%d' % x
        return x

class SCTPChunkSACK(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ByteEnumField('type', 3, sctpchunktypes), XByteField('flags', None), ShortField('len', None), XIntField('cumul_tsn_ack', None), IntField('a_rwnd', None), FieldLenField('n_gap_ack', None, count_of='gap_ack_list'), FieldLenField('n_dup_tsn', None, count_of='dup_tsn_list'), FieldListField('gap_ack_list', [], GapAckField('gap_ack', None), count_from=lambda pkt: pkt.n_gap_ack), FieldListField('dup_tsn_list', [], XIntField('dup_tsn', None), count_from=lambda pkt: pkt.n_dup_tsn)]

    def post_build(self, p, pay):
        if False:
            i = 10
            return i + 15
        if self.len is None:
            p = p[:2] + struct.pack('>H', len(p)) + p[4:]
        return p + pay

class SCTPChunkHeartbeatReq(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ByteEnumField('type', 4, sctpchunktypes), XByteField('flags', None), FieldLenField('len', None, length_of='params', adjust=lambda pkt, x: x + 4), ChunkParamField('params', None, length_from=lambda pkt: pkt.len - 4)]

class SCTPChunkHeartbeatAck(SCTPChunkHeartbeatReq):
    type = 5

class SCTPChunkAbort(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ByteEnumField('type', 6, sctpchunktypes), BitField('reserved', None, 7), BitField('TCB', 0, 1), FieldLenField('len', None, length_of='error_causes', adjust=lambda pkt, x: x + 4), PadField(StrLenField('error_causes', '', length_from=lambda pkt: pkt.len - 4), 4, padwith=b'\x00')]

class SCTPChunkShutdown(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ByteEnumField('type', 7, sctpchunktypes), XByteField('flags', None), ShortField('len', 8), XIntField('cumul_tsn_ack', None)]

class SCTPChunkShutdownAck(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ByteEnumField('type', 8, sctpchunktypes), XByteField('flags', None), ShortField('len', 4)]

class SCTPChunkError(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ByteEnumField('type', 9, sctpchunktypes), XByteField('flags', None), FieldLenField('len', None, length_of='error_causes', adjust=lambda pkt, x: x + 4), PadField(StrLenField('error_causes', '', length_from=lambda pkt: pkt.len - 4), 4, padwith=b'\x00')]

class SCTPChunkCookieEcho(SCTPChunkError):
    fields_desc = [ByteEnumField('type', 10, sctpchunktypes), XByteField('flags', None), FieldLenField('len', None, length_of='cookie', adjust=lambda pkt, x: x + 4), PadField(StrLenField('cookie', '', length_from=lambda pkt: pkt.len - 4), 4, padwith=b'\x00')]

class SCTPChunkCookieAck(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ByteEnumField('type', 11, sctpchunktypes), XByteField('flags', None), ShortField('len', 4)]

class SCTPChunkShutdownComplete(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ByteEnumField('type', 14, sctpchunktypes), BitField('reserved', None, 7), BitField('TCB', 0, 1), ShortField('len', 4)]

class SCTPChunkAuthentication(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ByteEnumField('type', 15, sctpchunktypes), XByteField('flags', None), FieldLenField('len', None, length_of='HMAC', adjust=lambda pkt, x: x + 8), ShortField('shared_key_id', None), ShortField('HMAC_function', None), PadField(StrLenField('HMAC', '', length_from=lambda pkt: pkt.len - 8), 4, padwith=b'\x00')]

class SCTPChunkAddressConf(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ByteEnumField('type', 193, sctpchunktypes), XByteField('flags', None), FieldLenField('len', None, length_of='params', adjust=lambda pkt, x: x + 8), IntField('seq', 0), ChunkParamField('params', None, length_from=lambda pkt: pkt.len - 8)]

class SCTPChunkReConfig(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ByteEnumField('type', 130, sctpchunktypes), XByteField('flags', None), FieldLenField('len', None, length_of='params', adjust=lambda pkt, x: x + 4), ChunkParamField('params', None, length_from=lambda pkt: pkt.len - 4)]

class SCTPChunkPad(_SCTPChunkGuessPayload, Packet):
    fields_desc = [ByteEnumField('type', 132, sctpchunktypes), XByteField('flags', None), FieldLenField('len', None, length_of='padding', adjust=lambda pkt, x: x + 8), PadField(StrLenField('padding', None, length_from=lambda pkt: pkt.len - 8), 4, padwith=b'\x00')]

class SCTPChunkAddressConfAck(SCTPChunkAddressConf):
    type = 128
bind_layers(IP, SCTP, proto=IPPROTO_SCTP)
bind_layers(IPerror, SCTPerror, proto=IPPROTO_SCTP)
bind_layers(IPv6, SCTP, nh=IPPROTO_SCTP)
bind_layers(IPerror6, SCTPerror, proto=IPPROTO_SCTP)