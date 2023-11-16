import struct
from scapy.packet import Packet, bind_layers
from scapy.fields import XByteField, XShortField, StrLenField, ByteEnumField, BitFieldLenField, ByteField, ConditionalField, EnumField, FieldListField, ShortField, StrFixedLenField, XShortEnumField
from scapy.layers.inet import TCP
from scapy.utils import orb
from scapy.config import conf
from scapy.volatile import VolatileValue
_modbus_exceptions = {1: 'Illegal Function Code', 2: 'Illegal Data Address', 3: 'Illegal Data Value', 4: 'Server Device Failure', 5: 'Acknowledge', 6: 'Server Device Busy', 8: 'Memory Parity Error', 10: 'Gateway Path Unavailable', 11: 'Gateway Target Device Failed to Respond'}

class _ModbusPDUNoPayload(Packet):

    def extract_padding(self, s):
        if False:
            i = 10
            return i + 15
        return (b'', None)

class ModbusPDU01ReadCoilsRequest(_ModbusPDUNoPayload):
    name = 'Read Coils Request'
    fields_desc = [XByteField('funcCode', 1), XShortField('startAddr', 0), XShortField('quantity', 1)]

class ModbusPDU01ReadCoilsResponse(_ModbusPDUNoPayload):
    name = 'Read Coils Response'
    fields_desc = [XByteField('funcCode', 1), BitFieldLenField('byteCount', None, 8, count_of='coilStatus'), FieldListField('coilStatus', [0], ByteField('', 0), count_from=lambda pkt: pkt.byteCount)]

class ModbusPDU01ReadCoilsError(_ModbusPDUNoPayload):
    name = 'Read Coils Exception'
    fields_desc = [XByteField('funcCode', 129), ByteEnumField('exceptCode', 1, _modbus_exceptions)]

class ModbusPDU02ReadDiscreteInputsRequest(_ModbusPDUNoPayload):
    name = 'Read Discrete Inputs'
    fields_desc = [XByteField('funcCode', 2), XShortField('startAddr', 0), XShortField('quantity', 1)]

class ModbusPDU02ReadDiscreteInputsResponse(Packet):
    """ inputStatus: result is represented as bytes, padded with 0 to have a
        integer number of bytes. The field does not parse this result and
        present the bytes directly
    """
    name = 'Read Discrete Inputs Response'
    fields_desc = [XByteField('funcCode', 2), BitFieldLenField('byteCount', None, 8, count_of='inputStatus'), FieldListField('inputStatus', [0], ByteField('', 0), count_from=lambda pkt: pkt.byteCount)]

class ModbusPDU02ReadDiscreteInputsError(Packet):
    name = 'Read Discrete Inputs Exception'
    fields_desc = [XByteField('funcCode', 130), ByteEnumField('exceptCode', 1, _modbus_exceptions)]

class ModbusPDU03ReadHoldingRegistersRequest(_ModbusPDUNoPayload):
    name = 'Read Holding Registers'
    fields_desc = [XByteField('funcCode', 3), XShortField('startAddr', 0), XShortField('quantity', 1)]

class ModbusPDU03ReadHoldingRegistersResponse(Packet):
    name = 'Read Holding Registers Response'
    fields_desc = [XByteField('funcCode', 3), BitFieldLenField('byteCount', None, 8, count_of='registerVal', adjust=lambda pkt, x: x * 2), FieldListField('registerVal', [0], ShortField('', 0), count_from=lambda pkt: pkt.byteCount)]

class ModbusPDU03ReadHoldingRegistersError(Packet):
    name = 'Read Holding Registers Exception'
    fields_desc = [XByteField('funcCode', 131), ByteEnumField('exceptCode', 1, _modbus_exceptions)]

class ModbusPDU04ReadInputRegistersRequest(_ModbusPDUNoPayload):
    name = 'Read Input Registers'
    fields_desc = [XByteField('funcCode', 4), XShortField('startAddr', 0), XShortField('quantity', 1)]

class ModbusPDU04ReadInputRegistersResponse(Packet):
    name = 'Read Input Registers Response'
    fields_desc = [XByteField('funcCode', 4), BitFieldLenField('byteCount', None, 8, count_of='registerVal', adjust=lambda pkt, x: x * 2), FieldListField('registerVal', [0], ShortField('', 0), count_from=lambda pkt: pkt.byteCount)]

class ModbusPDU04ReadInputRegistersError(Packet):
    name = 'Read Input Registers Exception'
    fields_desc = [XByteField('funcCode', 132), ByteEnumField('exceptCode', 1, _modbus_exceptions)]

class ModbusPDU05WriteSingleCoilRequest(Packet):
    name = 'Write Single Coil'
    fields_desc = [XByteField('funcCode', 5), XShortField('outputAddr', 0), XShortField('outputValue', 0)]

class ModbusPDU05WriteSingleCoilResponse(Packet):
    name = 'Write Single Coil'
    fields_desc = [XByteField('funcCode', 5), XShortField('outputAddr', 0), XShortField('outputValue', 0)]

class ModbusPDU05WriteSingleCoilError(Packet):
    name = 'Write Single Coil Exception'
    fields_desc = [XByteField('funcCode', 133), ByteEnumField('exceptCode', 1, _modbus_exceptions)]

class ModbusPDU06WriteSingleRegisterRequest(_ModbusPDUNoPayload):
    name = 'Write Single Register'
    fields_desc = [XByteField('funcCode', 6), XShortField('registerAddr', 0), XShortField('registerValue', 0)]

class ModbusPDU06WriteSingleRegisterResponse(Packet):
    name = 'Write Single Register Response'
    fields_desc = [XByteField('funcCode', 6), XShortField('registerAddr', 0), XShortField('registerValue', 0)]

class ModbusPDU06WriteSingleRegisterError(Packet):
    name = 'Write Single Register Exception'
    fields_desc = [XByteField('funcCode', 134), ByteEnumField('exceptCode', 1, _modbus_exceptions)]

class ModbusPDU07ReadExceptionStatusRequest(_ModbusPDUNoPayload):
    name = 'Read Exception Status'
    fields_desc = [XByteField('funcCode', 7)]

class ModbusPDU07ReadExceptionStatusResponse(Packet):
    name = 'Read Exception Status Response'
    fields_desc = [XByteField('funcCode', 7), XByteField('startAddr', 0)]

class ModbusPDU07ReadExceptionStatusError(Packet):
    name = 'Read Exception Status Exception'
    fields_desc = [XByteField('funcCode', 135), ByteEnumField('exceptCode', 1, _modbus_exceptions)]
_diagnostics_sub_function = {0: 'Return Query Data', 1: 'Restart Communications Option', 2: 'Return Diagnostic Register', 3: 'Change ASCII Input Delimiter', 4: 'Force Listen Only Mode', 10: 'Clear Counters and Diagnostic Register', 11: 'Return Bus Message Count', 12: 'Return Bus Communication Error Count', 13: 'Return Bus Exception Error Count', 14: 'Return Slave Message Count', 15: 'Return Slave No Response Count', 16: 'Return Slave NAK Count', 17: 'Return Slave Busy Count', 18: 'Return Bus Character Overrun Count', 20: 'Clear Overrun Counter and Flag'}

class ModbusPDU08DiagnosticsRequest(_ModbusPDUNoPayload):
    name = 'Diagnostics'
    fields_desc = [XByteField('funcCode', 8), XShortEnumField('subFunc', 0, _diagnostics_sub_function), FieldListField('data', [0], XShortField('', 0))]

class ModbusPDU08DiagnosticsResponse(_ModbusPDUNoPayload):
    name = 'Diagnostics Response'
    fields_desc = [XByteField('funcCode', 8), XShortEnumField('subFunc', 0, _diagnostics_sub_function), FieldListField('data', [0], XShortField('', 0))]

class ModbusPDU08DiagnosticsError(_ModbusPDUNoPayload):
    name = 'Diagnostics Exception'
    fields_desc = [XByteField('funcCode', 136), ByteEnumField('exceptionCode', 1, _modbus_exceptions)]

class ModbusPDU0BGetCommEventCounterRequest(_ModbusPDUNoPayload):
    name = 'Get Comm Event Counter'
    fields_desc = [XByteField('funcCode', 11)]

class ModbusPDU0BGetCommEventCounterResponse(_ModbusPDUNoPayload):
    name = 'Get Comm Event Counter Response'
    fields_desc = [XByteField('funcCode', 11), XShortField('status', 0), XShortField('eventCount', 65535)]

class ModbusPDU0BGetCommEventCounterError(_ModbusPDUNoPayload):
    name = 'Get Comm Event Counter Exception'
    fields_desc = [XByteField('funcCode', 139), ByteEnumField('exceptionCode', 1, _modbus_exceptions)]

class ModbusPDU0CGetCommEventLogRequest(_ModbusPDUNoPayload):
    name = 'Get Comm Event Log'
    fields_desc = [XByteField('funcCode', 12)]

class ModbusPDU0CGetCommEventLogResponse(_ModbusPDUNoPayload):
    name = 'Get Comm Event Log Response'
    fields_desc = [XByteField('funcCode', 12), ByteField('byteCount', 8), XShortField('status', 0), XShortField('eventCount', 264), XShortField('messageCount', 289), FieldListField('event', [32, 0], XByteField('', 0))]

class ModbusPDU0CGetCommEventLogError(_ModbusPDUNoPayload):
    name = 'Get Comm Event Log Exception'
    fields_desc = [XByteField('funcCode', 140), XByteField('exceptionCode', 1)]

class ModbusPDU0FWriteMultipleCoilsRequest(Packet):
    name = 'Write Multiple Coils'
    fields_desc = [XByteField('funcCode', 15), XShortField('startAddr', 0), XShortField('quantityOutput', 1), BitFieldLenField('byteCount', None, 8, count_of='outputsValue'), FieldListField('outputsValue', [0], XByteField('', 0), count_from=lambda pkt: pkt.byteCount)]

class ModbusPDU0FWriteMultipleCoilsResponse(Packet):
    name = 'Write Multiple Coils Response'
    fields_desc = [XByteField('funcCode', 15), XShortField('startAddr', 0), XShortField('quantityOutput', 1)]

class ModbusPDU0FWriteMultipleCoilsError(Packet):
    name = 'Write Multiple Coils Exception'
    fields_desc = [XByteField('funcCode', 143), ByteEnumField('exceptCode', 1, _modbus_exceptions)]

class ModbusPDU10WriteMultipleRegistersRequest(Packet):
    name = 'Write Multiple Registers'
    fields_desc = [XByteField('funcCode', 16), XShortField('startAddr', 0), BitFieldLenField('quantityRegisters', None, 16, count_of='outputsValue'), BitFieldLenField('byteCount', None, 8, count_of='outputsValue', adjust=lambda pkt, x: x * 2), FieldListField('outputsValue', [0], XShortField('', 0), count_from=lambda pkt: pkt.byteCount)]

class ModbusPDU10WriteMultipleRegistersResponse(Packet):
    name = 'Write Multiple Registers Response'
    fields_desc = [XByteField('funcCode', 16), XShortField('startAddr', 0), XShortField('quantityRegisters', 1)]

class ModbusPDU10WriteMultipleRegistersError(Packet):
    name = 'Write Multiple Registers Exception'
    fields_desc = [XByteField('funcCode', 144), ByteEnumField('exceptCode', 1, _modbus_exceptions)]

class ModbusPDU11ReportSlaveIdRequest(_ModbusPDUNoPayload):
    name = 'Report Slave Id'
    fields_desc = [XByteField('funcCode', 17)]

class ModbusPDU11ReportSlaveIdResponse(Packet):
    name = 'Report Slave Id Response'
    fields_desc = [XByteField('funcCode', 17), BitFieldLenField('byteCount', None, 8, length_of='slaveId'), ConditionalField(StrLenField('slaveId', '', length_from=lambda pkt: pkt.byteCount), lambda pkt: pkt.byteCount > 0), ConditionalField(XByteField('runIdicatorStatus', 0), lambda pkt: pkt.byteCount > 0)]

class ModbusPDU11ReportSlaveIdError(Packet):
    name = 'Report Slave Id Exception'
    fields_desc = [XByteField('funcCode', 145), ByteEnumField('exceptCode', 1, _modbus_exceptions)]

class ModbusReadFileSubRequest(Packet):
    name = 'Sub-request of Read File Record'
    fields_desc = [ByteField('refType', 6), ShortField('fileNumber', 1), ShortField('recordNumber', 0), ShortField('recordLength', 1)]

    def guess_payload_class(self, payload):
        if False:
            while True:
                i = 10
        return ModbusReadFileSubRequest

class ModbusPDU14ReadFileRecordRequest(Packet):
    name = 'Read File Record'
    fields_desc = [XByteField('funcCode', 20), ByteField('byteCount', None)]

    def guess_payload_class(self, payload):
        if False:
            for i in range(10):
                print('nop')
        if self.byteCount > 0:
            return ModbusReadFileSubRequest
        else:
            return Packet.guess_payload_class(self, payload)

    def post_build(self, p, pay):
        if False:
            i = 10
            return i + 15
        if self.byteCount is None:
            tmp_len = len(pay)
            p = p[:1] + struct.pack('!B', tmp_len) + p[3:]
        return p + pay

class ModbusReadFileSubResponse(Packet):
    name = 'Sub-response'
    fields_desc = [BitFieldLenField('respLength', None, 8, count_of='recData', adjust=lambda pkt, p: p * 2 + 1), ByteField('refType', 6), FieldListField('recData', [0], XShortField('', 0), count_from=lambda pkt: (pkt.respLength - 1) // 2)]

    def guess_payload_class(self, payload):
        if False:
            print('Hello World!')
        return ModbusReadFileSubResponse

class ModbusPDU14ReadFileRecordResponse(Packet):
    name = 'Read File Record Response'
    fields_desc = [XByteField('funcCode', 20), ByteField('dataLength', None)]

    def post_build(self, p, pay):
        if False:
            i = 10
            return i + 15
        if self.dataLength is None:
            tmp_len = len(pay)
            p = p[:1] + struct.pack('!B', tmp_len) + p[3:]
        return p + pay

    def guess_payload_class(self, payload):
        if False:
            print('Hello World!')
        if self.dataLength > 0:
            return ModbusReadFileSubResponse
        else:
            return Packet.guess_payload_class(self, payload)

class ModbusPDU14ReadFileRecordError(Packet):
    name = 'Read File Record Exception'
    fields_desc = [XByteField('funcCode', 148), ByteEnumField('exceptCode', 1, _modbus_exceptions)]

class ModbusWriteFileSubRequest(Packet):
    name = 'Sub request of Write File Record'
    fields_desc = [ByteField('refType', 6), ShortField('fileNumber', 1), ShortField('recordNumber', 0), BitFieldLenField('recordLength', None, 16, length_of='recordData', adjust=lambda pkt, p: p // 2), FieldListField('recordData', [0], ShortField('', 0), length_from=lambda pkt: pkt.recordLength * 2)]

    def guess_payload_class(self, payload):
        if False:
            return 10
        if payload:
            return ModbusWriteFileSubRequest

class ModbusPDU15WriteFileRecordRequest(Packet):
    name = 'Write File Record'
    fields_desc = [XByteField('funcCode', 21), ByteField('dataLength', None)]

    def post_build(self, p, pay):
        if False:
            while True:
                i = 10
        if self.dataLength is None:
            tmp_len = len(pay)
            p = p[:1] + struct.pack('!B', tmp_len) + p[3:]
            return p + pay

    def guess_payload_class(self, payload):
        if False:
            print('Hello World!')
        if self.dataLength > 0:
            return ModbusWriteFileSubRequest
        else:
            return Packet.guess_payload_class(self, payload)

class ModbusWriteFileSubResponse(ModbusWriteFileSubRequest):
    name = 'Sub response of Write File Record'

    def guess_payload_class(self, payload):
        if False:
            return 10
        if payload:
            return ModbusWriteFileSubResponse

class ModbusPDU15WriteFileRecordResponse(ModbusPDU15WriteFileRecordRequest):
    name = 'Write File Record Response'

    def guess_payload_class(self, payload):
        if False:
            i = 10
            return i + 15
        if self.dataLength > 0:
            return ModbusWriteFileSubResponse
        else:
            return Packet.guess_payload_class(self, payload)

class ModbusPDU15WriteFileRecordError(Packet):
    name = 'Write File Record Exception'
    fields_desc = [XByteField('funcCode', 149), ByteEnumField('exceptCode', 1, _modbus_exceptions)]

class ModbusPDU16MaskWriteRegisterRequest(Packet):
    name = 'Mask Write Register'
    fields_desc = [XByteField('funcCode', 22), XShortField('refAddr', 0), XShortField('andMask', 65535), XShortField('orMask', 0)]

class ModbusPDU16MaskWriteRegisterResponse(Packet):
    name = 'Mask Write Register Response'
    fields_desc = [XByteField('funcCode', 22), XShortField('refAddr', 0), XShortField('andMask', 65535), XShortField('orMask', 0)]

class ModbusPDU16MaskWriteRegisterError(Packet):
    name = 'Mask Write Register Exception'
    fields_desc = [XByteField('funcCode', 150), ByteEnumField('exceptCode', 1, _modbus_exceptions)]

class ModbusPDU17ReadWriteMultipleRegistersRequest(Packet):
    name = 'Read Write Multiple Registers'
    fields_desc = [XByteField('funcCode', 23), XShortField('readStartingAddr', 0), XShortField('readQuantityRegisters', 1), XShortField('writeStartingAddr', 0), BitFieldLenField('writeQuantityRegisters', None, 16, count_of='writeRegistersValue'), BitFieldLenField('byteCount', None, 8, count_of='writeRegistersValue', adjust=lambda pkt, x: x * 2), FieldListField('writeRegistersValue', [0], XShortField('', 0), count_from=lambda pkt: pkt.byteCount)]

class ModbusPDU17ReadWriteMultipleRegistersResponse(Packet):
    name = 'Read Write Multiple Registers Response'
    fields_desc = [XByteField('funcCode', 23), BitFieldLenField('byteCount', None, 8, count_of='registerVal', adjust=lambda pkt, x: x * 2), FieldListField('registerVal', [0], ShortField('', 0), count_from=lambda pkt: pkt.byteCount)]

class ModbusPDU17ReadWriteMultipleRegistersError(Packet):
    name = 'Read Write Multiple Exception'
    fields_desc = [XByteField('funcCode', 151), ByteEnumField('exceptCode', 1, _modbus_exceptions)]

class ModbusPDU18ReadFIFOQueueRequest(Packet):
    name = 'Read FIFO Queue'
    fields_desc = [XByteField('funcCode', 24), XShortField('FIFOPointerAddr', 0)]

class ModbusPDU18ReadFIFOQueueResponse(Packet):
    name = 'Read FIFO Queue Response'
    fields_desc = [XByteField('funcCode', 24), BitFieldLenField('byteCount', None, 16, count_of='FIFOVal', adjust=lambda pkt, p: p * 2 + 2), BitFieldLenField('FIFOCount', None, 16, count_of='FIFOVal'), FieldListField('FIFOVal', [], ShortField('', 0), count_from=lambda pkt: pkt.byteCount)]

class ModbusPDU18ReadFIFOQueueError(Packet):
    name = 'Read FIFO Queue Exception'
    fields_desc = [XByteField('funcCode', 152), ByteEnumField('exceptCode', 1, _modbus_exceptions)]
_read_device_id_codes = {1: 'Basic', 2: 'Regular', 3: 'Extended', 4: 'Specific'}
_read_device_id_object_id = {0: 'VendorName', 1: 'ProductCode', 2: 'MajorMinorRevision', 3: 'VendorUrl', 4: 'ProductName', 5: 'ModelName', 6: 'UserApplicationName'}
_read_device_id_conformity_lvl = {1: 'Basic Identification (stream only)', 2: 'Regular Identification (stream only)', 3: 'Extended Identification (stream only)', 129: 'Basic Identification (stream and individual access)', 130: 'Regular Identification (stream and individual access)', 131: 'Extended Identification (stream and individual access)'}
_read_device_id_more_follow = {0: 'No', 1: 'Yes'}

class ModbusPDU2B0EReadDeviceIdentificationRequest(Packet):
    name = 'Read Device Identification'
    fields_desc = [XByteField('funcCode', 43), XByteField('MEIType', 14), ByteEnumField('readCode', 1, _read_device_id_codes), ByteEnumField('objectId', 0, _read_device_id_object_id)]

class ModbusPDU2B0EReadDeviceIdentificationResponse(Packet):
    name = 'Read Device Identification'
    fields_desc = [XByteField('funcCode', 43), XByteField('MEIType', 14), ByteEnumField('readCode', 4, _read_device_id_codes), ByteEnumField('conformityLevel', 1, _read_device_id_conformity_lvl), ByteEnumField('more', 0, _read_device_id_more_follow), ByteEnumField('nextObjId', 0, _read_device_id_object_id), ByteField('objCount', 0)]

    def guess_payload_class(self, payload):
        if False:
            while True:
                i = 10
        if self.objCount > 0:
            return ModbusObjectId
        else:
            return Packet.guess_payload_class(self, payload)

class ModbusPDU2B0EReadDeviceIdentificationError(Packet):
    name = 'Read Exception Status Exception'
    fields_desc = [XByteField('funcCode', 171), ByteEnumField('exceptCode', 1, _modbus_exceptions)]
_reserved_funccode_request = {9: '0x09 Unknown Reserved Request', 10: '0x0a Unknown Reserved Request', 13: '0x0d Unknown Reserved Request', 14: '0x0e Unknown Reserved Request', 41: '0x29 Unknown Reserved Request', 42: '0x2a Unknown Reserved Request', 90: 'Specific Schneider Electric Request', 91: '0x5b Unknown Reserved Request', 125: '0x7d Unknown Reserved Request', 126: '0x7e Unknown Reserved Request', 127: '0x7f Unknown Reserved Request'}
_reserved_funccode_response = {9: '0x09 Unknown Reserved Response', 10: '0x0a Unknown Reserved Response', 13: '0x0d Unknown Reserved Response', 14: '0x0e Unknown Reserved Response', 41: '0x29 Unknown Reserved Response', 42: '0x2a Unknown Reserved Response', 90: 'Specific Schneider Electric Response', 91: '0x5b Unknown Reserved Response', 125: '0x7d Unknown Reserved Response', 126: '0x7e Unknown Reserved Response', 127: '0x7f Unknown Reserved Response'}
_reserved_funccode_error = {137: '0x89 Unknown Reserved Error', 138: '0x8a Unknown Reserved Error', 141: '0x8d Unknown Reserved Error', 142: '0x8e Unknown Reserved Error', 169: '0x88 Unknown Reserved Error', 170: '0x88 Unknown Reserved Error', 218: 'Specific Schneider Electric Error', 219: '0xdb Unknown Reserved Error', 220: '0xdc Unknown Reserved Error', 253: '0xfd Unknown Reserved Error', 254: '0xfe Unknown Reserved Error', 255: '0xff Unknown Reserved Error'}

class ModbusPDUReservedFunctionCodeRequest(_ModbusPDUNoPayload):
    name = 'Reserved Function Code Request'
    fields_desc = [ByteEnumField('funcCode', 0, _reserved_funccode_request), StrFixedLenField('payload', '', 255)]

    def mysummary(self):
        if False:
            while True:
                i = 10
        return self.sprintf('Modbus Reserved Request %funcCode%')

class ModbusPDUReservedFunctionCodeResponse(_ModbusPDUNoPayload):
    name = 'Reserved Function Code Response'
    fields_desc = [ByteEnumField('funcCode', 0, _reserved_funccode_response), StrFixedLenField('payload', '', 255)]

    def mysummary(self):
        if False:
            while True:
                i = 10
        return self.sprintf('Modbus Reserved Response %funcCode%')

class ModbusPDUReservedFunctionCodeError(_ModbusPDUNoPayload):
    name = 'Reserved Function Code Error'
    fields_desc = [ByteEnumField('funcCode', 0, _reserved_funccode_error), StrFixedLenField('payload', '', 255)]

    def mysummary(self):
        if False:
            print('Hello World!')
        return self.sprintf('Modbus Reserved Error %funcCode%')
_userdefined_funccode_request = {}
_userdefined_funccode_response = {}
_userdefined_funccode_error = {}

class ModbusByteEnumField(EnumField):
    __slots__ = 'defEnum'

    def __init__(self, name, default, enum, defEnum):
        if False:
            print('Hello World!')
        EnumField.__init__(self, name, default, enum, 'B')
        self.defEnum = defEnum

    def i2repr_one(self, pkt, x):
        if False:
            for i in range(10):
                print('nop')
        if self not in conf.noenum and (not isinstance(x, VolatileValue)) and (x in self.i2s):
            return self.i2s[x]
        if self.defEnum:
            return self.defEnum
        return repr(x)

class ModbusPDUUserDefinedFunctionCodeRequest(_ModbusPDUNoPayload):
    name = 'User-Defined Function Code Request'
    fields_desc = [ModbusByteEnumField('funcCode', 0, _userdefined_funccode_request, 'Unknown user-defined request function Code'), StrFixedLenField('payload', '', 255)]

    def mysummary(self):
        if False:
            while True:
                i = 10
        return self.sprintf('Modbus User-Defined Request %funcCode%')

class ModbusPDUUserDefinedFunctionCodeResponse(_ModbusPDUNoPayload):
    name = 'User-Defined Function Code Response'
    fields_desc = [ModbusByteEnumField('funcCode', 0, _userdefined_funccode_response, 'Unknown user-defined response function Code'), StrFixedLenField('payload', '', 255)]

    def mysummary(self):
        if False:
            i = 10
            return i + 15
        return self.sprintf('Modbus User-Defined Response %funcCode%')

class ModbusPDUUserDefinedFunctionCodeError(_ModbusPDUNoPayload):
    name = 'User-Defined Function Code Error'
    fields_desc = [ModbusByteEnumField('funcCode', 0, _userdefined_funccode_error, 'Unknown user-defined error function Code'), StrFixedLenField('payload', '', 255)]

    def mysummary(self):
        if False:
            i = 10
            return i + 15
        return self.sprintf('Modbus User-Defined Error %funcCode%')

class ModbusObjectId(Packet):
    name = 'Object'
    fields_desc = [ByteEnumField('id', 0, _read_device_id_object_id), BitFieldLenField('length', None, 8, length_of='value'), StrLenField('value', '', length_from=lambda pkt: pkt.length)]

    def guess_payload_class(self, payload):
        if False:
            return 10
        return ModbusObjectId
_modbus_request_classes = {1: ModbusPDU01ReadCoilsRequest, 2: ModbusPDU02ReadDiscreteInputsRequest, 3: ModbusPDU03ReadHoldingRegistersRequest, 4: ModbusPDU04ReadInputRegistersRequest, 5: ModbusPDU05WriteSingleCoilRequest, 6: ModbusPDU06WriteSingleRegisterRequest, 7: ModbusPDU07ReadExceptionStatusRequest, 8: ModbusPDU08DiagnosticsRequest, 11: ModbusPDU0BGetCommEventCounterRequest, 12: ModbusPDU0CGetCommEventLogRequest, 15: ModbusPDU0FWriteMultipleCoilsRequest, 16: ModbusPDU10WriteMultipleRegistersRequest, 17: ModbusPDU11ReportSlaveIdRequest, 20: ModbusPDU14ReadFileRecordRequest, 21: ModbusPDU15WriteFileRecordRequest, 22: ModbusPDU16MaskWriteRegisterRequest, 23: ModbusPDU17ReadWriteMultipleRegistersRequest, 24: ModbusPDU18ReadFIFOQueueRequest}
_modbus_error_classes = {129: ModbusPDU01ReadCoilsError, 130: ModbusPDU02ReadDiscreteInputsError, 131: ModbusPDU03ReadHoldingRegistersError, 132: ModbusPDU04ReadInputRegistersError, 133: ModbusPDU05WriteSingleCoilError, 134: ModbusPDU06WriteSingleRegisterError, 135: ModbusPDU07ReadExceptionStatusError, 136: ModbusPDU08DiagnosticsError, 139: ModbusPDU0BGetCommEventCounterError, 140: ModbusPDU0CGetCommEventLogError, 143: ModbusPDU0FWriteMultipleCoilsError, 144: ModbusPDU10WriteMultipleRegistersError, 145: ModbusPDU11ReportSlaveIdError, 148: ModbusPDU14ReadFileRecordError, 149: ModbusPDU15WriteFileRecordError, 150: ModbusPDU16MaskWriteRegisterError, 151: ModbusPDU17ReadWriteMultipleRegistersError, 152: ModbusPDU18ReadFIFOQueueError, 171: ModbusPDU2B0EReadDeviceIdentificationError}
_modbus_response_classes = {1: ModbusPDU01ReadCoilsResponse, 2: ModbusPDU02ReadDiscreteInputsResponse, 3: ModbusPDU03ReadHoldingRegistersResponse, 4: ModbusPDU04ReadInputRegistersResponse, 5: ModbusPDU05WriteSingleCoilResponse, 6: ModbusPDU06WriteSingleRegisterResponse, 7: ModbusPDU07ReadExceptionStatusResponse, 8: ModbusPDU08DiagnosticsResponse, 11: ModbusPDU0BGetCommEventCounterResponse, 12: ModbusPDU0CGetCommEventLogResponse, 15: ModbusPDU0FWriteMultipleCoilsResponse, 16: ModbusPDU10WriteMultipleRegistersResponse, 17: ModbusPDU11ReportSlaveIdResponse, 20: ModbusPDU14ReadFileRecordResponse, 21: ModbusPDU15WriteFileRecordResponse, 22: ModbusPDU16MaskWriteRegisterResponse, 23: ModbusPDU17ReadWriteMultipleRegistersResponse, 24: ModbusPDU18ReadFIFOQueueResponse}
_mei_types_request = {14: ModbusPDU2B0EReadDeviceIdentificationRequest}
_mei_types_response = {14: ModbusPDU2B0EReadDeviceIdentificationResponse}

class ModbusADURequest(Packet):
    name = 'ModbusADU'
    fields_desc = [XShortField('transId', 0), XShortField('protoId', 0), ShortField('len', None), XByteField('unitId', 255)]

    def guess_payload_class(self, payload):
        if False:
            return 10
        function_code = orb(payload[0])
        if function_code == 43:
            sub_code = orb(payload[1])
            try:
                return _mei_types_request[sub_code]
            except KeyError:
                pass
        try:
            return _modbus_request_classes[function_code]
        except KeyError:
            pass
        if function_code in _reserved_funccode_request:
            return ModbusPDUReservedFunctionCodeRequest
        return ModbusPDUUserDefinedFunctionCodeRequest

    def post_build(self, p, pay):
        if False:
            for i in range(10):
                print('nop')
        if self.len is None:
            tmp_len = len(pay) + 1
            p = p[:4] + struct.pack('!H', tmp_len) + p[6:]
        return p + pay

class ModbusADUResponse(Packet):
    name = 'ModbusADU'
    fields_desc = [XShortField('transId', 0), XShortField('protoId', 0), ShortField('len', None), XByteField('unitId', 255)]

    def guess_payload_class(self, payload):
        if False:
            return 10
        function_code = orb(payload[0])
        if function_code == 43:
            sub_code = orb(payload[1])
            try:
                return _mei_types_response[sub_code]
            except KeyError:
                pass
        try:
            return _modbus_response_classes[function_code]
        except KeyError:
            pass
        try:
            return _modbus_error_classes[function_code]
        except KeyError:
            pass
        if function_code in _reserved_funccode_response:
            return ModbusPDUReservedFunctionCodeResponse
        elif function_code in _reserved_funccode_error:
            return ModbusPDUReservedFunctionCodeError
        if function_code < 128:
            return ModbusPDUUserDefinedFunctionCodeResponse
        return ModbusPDUUserDefinedFunctionCodeError

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        if self.len is None:
            tmp_len = len(pay) + 1
            p = p[:4] + struct.pack('!H', tmp_len) + p[6:]
        return p + pay
bind_layers(TCP, ModbusADURequest, dport=502)
bind_layers(TCP, ModbusADUResponse, sport=502)