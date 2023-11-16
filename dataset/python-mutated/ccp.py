import struct
from scapy.packet import Packet, bind_layers, bind_bottom_up
from scapy.fields import XIntField, FlagsField, ByteEnumField, ThreeBytesField, XBitField, ShortField, IntField, XShortField, ByteField, XByteField, StrFixedLenField, LEShortField
from scapy.layers.can import CAN

class CCP(CAN):
    name = 'CAN Calibration Protocol'
    fields_desc = [FlagsField('flags', 0, 3, ['error', 'remote_transmission_request', 'extended']), XBitField('identifier', 0, 29), ByteField('length', 8), ThreeBytesField('reserved', 0)]

    def extract_padding(self, p):
        if False:
            return 10
        return (p, None)

class CRO(Packet):
    commands = {1: 'CONNECT', 27: 'GET_CCP_VERSION', 23: 'EXCHANGE_ID', 18: 'GET_SEED', 19: 'UNLOCK', 2: 'SET_MTA', 3: 'DNLOAD', 35: 'DNLOAD_6', 4: 'UPLOAD', 15: 'SHORT_UP', 17: 'SELECT_CAL_PAGE', 20: 'GET_DAQ_SIZE', 21: 'SET_DAQ_PTR', 22: 'WRITE_DAQ', 6: 'START_STOP', 7: 'DISCONNECT', 12: 'SET_S_STATUS', 13: 'GET_S_STATUS', 14: 'BUILD_CHKSUM', 16: 'CLEAR_MEMORY', 24: 'PROGRAM', 34: 'PROGRAM_6', 25: 'MOVE', 5: 'TEST', 9: 'GET_ACTIVE_CAL_PAGE', 8: 'START_STOP_ALL', 32: 'DIAG_SERVICE', 33: 'ACTION_SERVICE'}
    name = 'Command Receive Object'
    fields_desc = [ByteEnumField('cmd', 1, commands), ByteField('ctr', 0)]

    def hashret(self):
        if False:
            return 10
        return struct.pack('B', self.ctr)

class CONNECT(Packet):
    fields_desc = [LEShortField('station_address', 0), StrFixedLenField('ccp_reserved', b'\xff' * 4, length=4)]
bind_layers(CRO, CONNECT, cmd=1)

class GET_CCP_VERSION(Packet):
    fields_desc = [XByteField('main_protocol_version', 0), XByteField('release_version', 0), StrFixedLenField('ccp_reserved', b'\xff' * 4, length=4)]
bind_layers(CRO, GET_CCP_VERSION, cmd=27)

class EXCHANGE_ID(Packet):
    fields_desc = [StrFixedLenField('ccp_master_device_id', b'\x00' * 6, length=6)]
bind_layers(CRO, EXCHANGE_ID, cmd=23)

class GET_SEED(Packet):
    fields_desc = [XByteField('resource', 0), StrFixedLenField('ccp_reserved', b'\xff' * 5, length=5)]
bind_layers(CRO, GET_SEED, cmd=18)

class UNLOCK(Packet):
    fields_desc = [StrFixedLenField('key', b'\x00' * 6, length=6)]
bind_layers(CRO, UNLOCK, cmd=19)

class SET_MTA(Packet):
    fields_desc = [XByteField('mta_num', 0), XByteField('address_extension', 0), XIntField('address', 0)]
bind_layers(CRO, SET_MTA, cmd=2)

class DNLOAD(Packet):
    fields_desc = [XByteField('size', 0), StrFixedLenField('data', b'\x00' * 5, length=5)]
bind_layers(CRO, DNLOAD, cmd=3)

class DNLOAD_6(Packet):
    fields_desc = [StrFixedLenField('data', b'\x00' * 6, length=6)]
bind_layers(CRO, DNLOAD_6, cmd=35)

class UPLOAD(Packet):
    fields_desc = [XByteField('size', 0), StrFixedLenField('ccp_reserved', b'\xff' * 5, length=5)]
bind_layers(CRO, UPLOAD, cmd=4)

class SHORT_UP(Packet):
    fields_desc = [XByteField('size', 0), XByteField('address_extension', 0), XIntField('address', 0)]
bind_layers(CRO, SHORT_UP, cmd=15)

class SELECT_CAL_PAGE(Packet):
    fields_desc = [StrFixedLenField('ccp_reserved', b'\xff' * 6, length=6)]
bind_layers(CRO, SELECT_CAL_PAGE, cmd=17)

class GET_DAQ_SIZE(Packet):
    fields_desc = [XByteField('DAQ_num', 0), XByteField('ccp_reserved', 0), XIntField('DTO_identifier', 0)]
bind_layers(CRO, GET_DAQ_SIZE, cmd=20)

class SET_DAQ_PTR(Packet):
    fields_desc = [XByteField('DAQ_num', 0), XByteField('ODT_num', 0), XByteField('ODT_element', 0), StrFixedLenField('ccp_reserved', b'\xff' * 3, length=3)]
bind_layers(CRO, SET_DAQ_PTR, cmd=21)

class WRITE_DAQ(Packet):
    fields_desc = [XByteField('DAQ_size', 0), XByteField('address_extension', 0), XIntField('address', 0)]
bind_layers(CRO, WRITE_DAQ, cmd=22)

class START_STOP(Packet):
    fields_desc = [XByteField('mode', 0), XByteField('DAQ_num', 0), XByteField('ODT_num', 0), XByteField('event_channel', 0), XShortField('transmission_rate', 0)]
bind_layers(CRO, START_STOP, cmd=6)

class DISCONNECT(Packet):
    fields_desc = [ByteEnumField('type', 0, {0: 'temporary', 1: 'end_of_session'}), StrFixedLenField('ccp_reserved0', b'\xff' * 1, length=1), LEShortField('station_address', 0), StrFixedLenField('ccp_reserved', b'\xff' * 2, length=2)]
bind_layers(CRO, DISCONNECT, cmd=7)

class SET_S_STATUS(Packet):
    name = 'Set Session Status'
    fields_desc = [FlagsField('session_status', 0, 8, ['CAL', 'DAQ', 'RESUME', 'RES0', 'RES1', 'RES2', 'STORE', 'RUN']), StrFixedLenField('ccp_reserved', b'\xff' * 5, length=5)]
bind_layers(CRO, SET_S_STATUS, cmd=12)

class GET_S_STATUS(Packet):
    fields_desc = [StrFixedLenField('ccp_reserved', b'\xff' * 6, length=6)]
bind_layers(CRO, GET_S_STATUS, cmd=13)

class BUILD_CHKSUM(Packet):
    fields_desc = [IntField('size', 0), StrFixedLenField('ccp_reserved', b'\xff' * 2, length=2)]
bind_layers(CRO, BUILD_CHKSUM, cmd=14)

class CLEAR_MEMORY(Packet):
    fields_desc = [IntField('size', 0), StrFixedLenField('ccp_reserved', b'\xff' * 2, length=2)]
bind_layers(CRO, CLEAR_MEMORY, cmd=16)

class PROGRAM(Packet):
    fields_desc = [XByteField('size', 0), StrFixedLenField('data', b'\x00' * 0, length_from=lambda pkt: pkt.size), StrFixedLenField('ccp_reserved', b'\xff' * 5, length_from=lambda pkt: 5 - pkt.size)]
bind_layers(CRO, PROGRAM, cmd=24)

class PROGRAM_6(Packet):
    fields_desc = [StrFixedLenField('data', b'\x00' * 6, length=6)]
bind_layers(CRO, PROGRAM_6, cmd=34)

class MOVE(Packet):
    fields_desc = [IntField('size', 0), StrFixedLenField('ccp_reserved', b'\xff' * 2, length=2)]
bind_layers(CRO, MOVE, cmd=25)

class TEST(Packet):
    fields_desc = [LEShortField('station_address', 0), StrFixedLenField('ccp_reserved', b'\xff' * 4, length=4)]
bind_layers(CRO, TEST, cmd=5)

class GET_ACTIVE_CAL_PAGE(Packet):
    fields_desc = [StrFixedLenField('ccp_reserved', b'\xff' * 6, length=6)]
bind_layers(CRO, GET_ACTIVE_CAL_PAGE, cmd=9)

class START_STOP_ALL(Packet):
    fields_desc = [ByteEnumField('type', 0, {0: 'stop', 1: 'start'}), StrFixedLenField('ccp_reserved', b'\xff' * 5, length=5)]
bind_layers(CRO, START_STOP_ALL, cmd=8)

class DIAG_SERVICE(Packet):
    fields_desc = [ShortField('diag_service', 0), StrFixedLenField('ccp_reserved', b'\xff' * 4, length=4)]
bind_layers(CRO, DIAG_SERVICE, cmd=32)

class ACTION_SERVICE(Packet):
    fields_desc = [ShortField('action_service', 0), StrFixedLenField('ccp_reserved', b'\xff' * 4, length=4)]
bind_layers(CRO, ACTION_SERVICE, cmd=33)

class DEFAULT_DTO(Packet):
    fields_desc = [StrFixedLenField('load', b'\xff' * 5, length=5)]

class GET_CCP_VERSION_DTO(Packet):
    fields_desc = [XByteField('main_protocol_version', 0), XByteField('release_version', 0), StrFixedLenField('ccp_reserved', b'\x00' * 3, length=3)]

class EXCHANGE_ID_DTO(Packet):
    fields_desc = [ByteField('slave_device_ID_length', 0), ByteField('data_type_qualifier', 0), ByteField('resource_availability_mask', 0), ByteField('resource_protection_mask', 0), StrFixedLenField('ccp_reserved', b'\xff' * 1, length=1)]

class GET_SEED_DTO(Packet):
    fields_desc = [XByteField('protection_status', 0), StrFixedLenField('seed', b'\x00' * 4, length=4)]

class UNLOCK_DTO(Packet):
    fields_desc = [ByteField('privilege_status', 0), StrFixedLenField('ccp_reserved', b'\xff' * 4, length=4)]

class DNLOAD_DTO(Packet):
    fields_desc = [XByteField('MTA0_extension', 0), XIntField('MTA0_address', 0)]

class DNLOAD_6_DTO(Packet):
    fields_desc = [XByteField('MTA0_extension', 0), XIntField('MTA0_address', 0)]

class UPLOAD_DTO(Packet):
    fields_desc = [StrFixedLenField('data', b'\x00' * 5, length=5)]

class SHORT_UP_DTO(Packet):
    fields_desc = [StrFixedLenField('data', b'\x00' * 5, length=5)]

class GET_DAQ_SIZE_DTO(Packet):
    fields_desc = [XByteField('DAQ_list_size', 0), XByteField('first_pid', 0), StrFixedLenField('ccp_reserved', b'\xff' * 3, length=3)]

class GET_S_STATUS_DTO(Packet):
    fields_desc = [FlagsField('session_status', 0, 8, ['CAL', 'DAQ', 'RESUME', 'RES0', 'RES1', 'RES2', 'STORE', 'RUN']), ByteField('information_qualifier', 0), StrFixedLenField('information', b'\x00' * 3, length=3)]

class BUILD_CHKSUM_DTO(Packet):
    fields_desc = [ByteField('checksum_size', 0), StrFixedLenField('checksum_data', b'\x00' * 4, length_from=lambda pkt: pkt.checksum_size), StrFixedLenField('ccp_reserved', b'\xff' * 0, length_from=lambda pkt: 4 - pkt.checksum_size)]

class PROGRAM_DTO(Packet):
    fields_desc = [ByteField('MTA0_extension', 0), XIntField('MTA0_address', 0)]

class PROGRAM_6_DTO(Packet):
    fields_desc = [ByteField('MTA0_extension', 0), XIntField('MTA0_address', 0)]

class GET_ACTIVE_CAL_PAGE_DTO(Packet):
    fields_desc = [XByteField('address_extension', 0), XIntField('address', 0)]

class DIAG_SERVICE_DTO(Packet):
    fields_desc = [ByteField('data_length', 0), ByteField('data_type', 0), StrFixedLenField('ccp_reserved', b'\xff' * 3, length=3)]

class ACTION_SERVICE_DTO(Packet):
    fields_desc = [ByteField('data_length', 0), ByteField('data_type', 0), StrFixedLenField('ccp_reserved', b'\xff' * 3, length=3)]

class DTO(Packet):
    __slots__ = Packet.__slots__ + ['payload_cls']
    return_codes = {0: 'acknowledge / no error', 1: 'DAQ processor overload', 16: 'command processor busy', 17: 'DAQ processor busy', 18: 'internal timeout', 24: 'key request', 25: 'session status request', 32: 'cold start request', 33: 'cal. data init. request', 34: 'DAQ list init. request', 35: 'code update request', 48: 'unknown command', 49: 'command syntax', 50: 'parameter(s) out of range', 51: 'access denied', 52: 'overload', 53: 'access locked', 54: 'resource/function not available'}
    fields_desc = [XByteField('packet_id', 255), ByteEnumField('return_code', 0, return_codes), ByteField('ctr', 0)]

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        self.payload_cls = DEFAULT_DTO
        if 'payload_cls' in kwargs:
            self.payload_cls = kwargs['payload_cls']
            del kwargs['payload_cls']
        Packet.__init__(self, *args, **kwargs)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return super(DTO, self).__eq__(other) and self.payload_cls == other.payload_cls

    def guess_payload_class(self, payload):
        if False:
            print('Hello World!')
        return self.payload_cls

    @staticmethod
    def get_dto_cls(cmd):
        if False:
            i = 10
            return i + 15
        try:
            return {3: DNLOAD_DTO, 4: UPLOAD_DTO, 9: GET_ACTIVE_CAL_PAGE_DTO, 13: GET_S_STATUS_DTO, 14: BUILD_CHKSUM_DTO, 15: SHORT_UP_DTO, 18: GET_SEED_DTO, 19: UNLOCK_DTO, 20: GET_DAQ_SIZE_DTO, 23: EXCHANGE_ID_DTO, 24: PROGRAM_DTO, 27: GET_CCP_VERSION_DTO, 32: DIAG_SERVICE_DTO, 33: ACTION_SERVICE_DTO, 34: PROGRAM_6_DTO, 35: DNLOAD_6_DTO}[cmd]
        except KeyError:
            return DEFAULT_DTO

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        'In CCP, the payload of a DTO packet is dependent on the cmd field\n        of a corresponding CRO packet. Two packets correspond, if there\n        ctr field is equal. If answers detect the corresponding CRO, it will\n        interpret the payload of a DTO with the correct class. In CCP, there is\n        no other way, to determine the class of a DTO payload. Since answers is\n        called on sr and sr1, this modification of the original answers\n        implementation will give a better user experience. '
        if not hasattr(other, 'ctr'):
            return 0
        if self.ctr != other.ctr:
            return 0
        if not hasattr(other, 'cmd'):
            return 0
        new_pl_cls = self.get_dto_cls(other.cmd)
        if self.payload_cls != new_pl_cls and self.payload_cls == DEFAULT_DTO:
            data = bytes(self.load)
            self.remove_payload()
            self.add_payload(new_pl_cls(data))
            self.payload_cls = new_pl_cls
        return 1

    def hashret(self):
        if False:
            print('Hello World!')
        return struct.pack('B', self.ctr)
bind_bottom_up(CCP, DTO)