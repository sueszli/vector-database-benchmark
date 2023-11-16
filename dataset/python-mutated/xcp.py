import struct
from scapy.config import conf
from scapy.contrib.automotive.xcp.cto_commands_master import Connect, Disconnect, GetStatus, Synch, GetCommModeInfo, GetId, SetRequest, GetSeed, Unlock, SetMta, Upload, ShortUpload, BuildChecksum, TransportLayerCmd, TransportLayerCmdGetSlaveId, TransportLayerCmdGetDAQId, TransportLayerCmdSetDAQId, UserCmd, Download, DownloadNext, DownloadMax, ShortDownload, ModifyBits, SetCalPage, GetCalPage, GetPagProcessorInfo, GetSegmentInfo, GetPageInfo, SetSegmentMode, GetSegmentMode, CopyCalPage, SetDaqPtr, WriteDaq, SetDaqListMode, GetDaqListMode, StartStopDaqList, StartStopSynch, ReadDaq, GetDaqClock, GetDaqProcessorInfo, GetDaqResolutionInfo, GetDaqListInfo, GetDaqEventInfo, ClearDaqList, FreeDaq, AllocDaq, AllocOdt, AllocOdtEntry, ProgramStart, ProgramClear, Program, ProgramReset, GetPgmProcessorInfo, GetSectorInfo, ProgramPrepare, ProgramFormat, ProgramNext, ProgramMax, ProgramVerify
from scapy.contrib.automotive.xcp.cto_commands_slave import GenericResponse, NegativeResponse, EvPacket, ServPacket, TransportLayerCmdGetSlaveIdResponse, TransportLayerCmdGetDAQIdResponse, SegmentInfoMode0PositiveResponse, SegmentInfoMode1PositiveResponse, SegmentInfoMode2PositiveResponse, ConnectPositiveResponse, StatusPositiveResponse, CommonModeInfoPositiveResponse, IdPositiveResponse, SeedPositiveResponse, UnlockPositiveResponse, UploadPositiveResponse, ShortUploadPositiveResponse, ChecksumPositiveResponse, CalPagePositiveResponse, PagProcessorInfoPositiveResponse, PageInfoPositiveResponse, SegmentModePositiveResponse, DAQListModePositiveResponse, StartStopDAQListPositiveResponse, DAQClockListPositiveResponse, ReadDAQPositiveResponse, DAQProcessorInfoPositiveResponse, DAQResolutionInfoPositiveResponse, DAQListInfoPositiveResponse, DAQEventInfoPositiveResponse, ProgramStartPositiveResponse, PgmProcessorPositiveResponse, SectorInfoPositiveResponse
from scapy.contrib.automotive.xcp.utils import get_timestamp_length, identification_field_needs_alignment, get_daq_length, get_daq_data_field_length
from scapy.fields import ByteEnumField, ShortField, XBitField, FlagsField, ByteField, ThreeBytesField, StrField, ConditionalField, XByteField, StrLenField
from scapy.layers.can import CAN
from scapy.layers.inet import UDP, TCP
from scapy.packet import Packet, bind_layers, bind_bottom_up, bind_top_down
conf.contribs.setdefault('XCP', {})
conf.contribs['XCP'].setdefault('byte_order', 1)
conf.contribs['XCP'].setdefault('allow_byte_order_change', True)
conf.contribs['XCP'].setdefault('Address_Granularity_Byte', None)
conf.contribs['XCP'].setdefault('allow_ag_change', True)
conf.contribs['XCP'].setdefault('MAX_CTO', None)
conf.contribs['XCP'].setdefault('MAX_DTO', None)
conf.contribs['XCP'].setdefault('allow_cto_and_dto_change', True)
conf.contribs['XCP'].setdefault('add_padding_for_can', False)
conf.contribs['XCP'].setdefault('timestamp_size', 0)

class XCPOnCAN(CAN):
    name = 'Universal calibration and measurement protocol on CAN'
    fields_desc = [FlagsField('flags', 0, 3, ['error', 'remote_transmission_request', 'extended']), XBitField('identifier', 0, 29), ByteField('length', None), ThreeBytesField('reserved', 0)]

    def post_build(self, pkt, pay):
        if False:
            while True:
                i = 10
        if self.length is None or (len(pay) < 8 and conf.contribs['XCP']['add_padding_for_can']):
            tmp_len = 8 if conf.contribs['XCP']['add_padding_for_can'] else len(pay)
            pkt = pkt[:4] + struct.pack('B', tmp_len) + pkt[5:]
            pay += b'\xcc' * (tmp_len - len(pay))
        return super(XCPOnCAN, self).post_build(pkt, pay)

    def extract_padding(self, p):
        if False:
            print('Hello World!')
        return (p[:self.length], None)

class XCPOnUDP(UDP):
    name = 'Universal calibration and measurement protocol on Ethernet'
    fields_desc = UDP.fields_desc + [ShortField('length', None), ShortField('ctr', 0)]

    def post_build(self, pkt, pay):
        if False:
            i = 10
            return i + 15
        if self.length is None:
            tmp_len = len(pay)
            pkt = pkt[:8] + struct.pack('!H', tmp_len) + pkt[10:]
        return super(XCPOnUDP, self).post_build(pkt, pay)

class XCPOnTCP(TCP):
    name = 'Universal calibration and measurement protocol on Ethernet'
    fields_desc = TCP.fields_desc + [ShortField('length', None), ShortField('ctr', 0)]

    def answers(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, XCPOnTCP):
            return 0
        if isinstance(other.payload, CTORequest) and isinstance(self.payload, CTOResponse):
            return self.payload.answers(other.payload)

    def post_build(self, pkt, pay):
        if False:
            for i in range(10):
                print('nop')
        if self.length is None:
            len_offset = 20 + len(self.options)
            tmp_len = len(pay)
            tmp_len = struct.pack('!H', tmp_len)
            pkt = pkt[:len_offset] + tmp_len + pkt[len_offset + 2:]
        return super(XCPOnTCP, self).post_build(pkt, pay)

class XCPOnCANTail(Packet):
    name = 'XCP Tail on CAN'
    fields_desc = [StrField('control_field', '')]

class CTORequest(Packet):
    pids = {255: 'CONNECT', 254: 'DISCONNECT', 253: 'GET_STATUS', 252: 'SYNCH', 251: 'GET_COMM_MODE_INFO', 250: 'GET_ID', 249: 'SET_REQUEST', 248: 'GET_SEED', 247: 'UNLOCK', 246: 'SET_MTA', 245: 'UPLOAD', 244: 'SHORT_UPLOAD', 243: 'BUILD_CHECKSUM', 242: 'TRANSPORT_LAYER_CMD', 241: 'USER_CMD', 240: 'DOWNLOAD', 239: 'DOWNLOAD_NEXT', 238: 'DOWNLOAD_MAX', 237: 'SHORT_DOWNLOAD', 236: 'MODIFY_BITS', 235: 'SET_CAL_PAGE', 234: 'GET_CAL_PAGE', 233: 'GET_PAG_PROCESSOR_INFO', 232: 'GET_SEGMENT_INFO', 231: 'GET_PAGE_INFO', 230: 'SET_SEGMENT_MODE', 229: 'GET_SEGMENT_MODE', 228: 'COPY_CAL_PAGE', 226: 'SET_DAQ_PTR', 225: 'WRITE_DAQ', 224: 'SET_DAQ_LIST_MODE', 223: 'GET_DAQ_LIST_MODE', 222: 'START_STOP_DAQ_LIST', 221: 'START_STOP_SYNCH', 199: 'WRITE_DAQ_MULTIPLE', 219: 'READ_DAQ', 220: 'GET_DAQ_CLOCK', 218: 'GET_DAQ_PROCESSOR_INFO', 217: 'GET_DAQ_RESOLUTION_INFO', 216: 'GET_DAQ_LIST_INFO', 215: 'GET_DAQ_EVENT_INFO', 227: 'CLEAR_DAQ_LIST', 214: 'FREE_DAQ', 213: 'ALLOC_DAQ', 212: 'ALLOC_ODT', 211: 'ALLOC_ODT_ENTRY', 210: 'PROGRAM_START', 209: 'PROGRAM_CLEAR', 208: 'PROGRAM', 207: 'PROGRAM_RESET', 206: 'GET_PGM_PROCESSOR_INFO', 205: 'GET_SECTOR_INFO', 204: 'PROGRAM_PREPARE', 203: 'PROGRAM_FORMAT', 202: 'PROGRAM_NEXT', 201: 'PROGRAM_MAX', 200: 'PROGRAM_VERIFY'}
    for pid in range(0, 192):
        pids[pid] = 'STIM'
    name = 'Command Transfer Object Request'
    fields_desc = [ByteEnumField('pid', 255, pids)]
bind_layers(CTORequest, Connect, pid=255)
bind_layers(CTORequest, Disconnect, pid=254)
bind_layers(CTORequest, GetStatus, pid=253)
bind_layers(CTORequest, Synch, pid=252)
bind_layers(CTORequest, GetCommModeInfo, pid=251)
bind_layers(CTORequest, GetId, pid=250)
bind_layers(CTORequest, SetRequest, pid=249)
bind_layers(CTORequest, GetSeed, pid=248)
bind_layers(CTORequest, Unlock, pid=247)
bind_layers(CTORequest, SetMta, pid=246)
bind_layers(CTORequest, Upload, pid=245)
bind_layers(CTORequest, ShortUpload, pid=244)
bind_layers(CTORequest, BuildChecksum, pid=243)
bind_layers(CTORequest, TransportLayerCmd, pid=242)
bind_layers(CTORequest, TransportLayerCmdGetSlaveId, pid=242, sub_command_code=255)
bind_layers(CTORequest, TransportLayerCmdGetDAQId, pid=242, sub_command_code=254)
bind_layers(CTORequest, TransportLayerCmdSetDAQId, pid=242, sub_command_code=253)
bind_layers(CTORequest, UserCmd, pid=241)
bind_layers(CTORequest, Download, pid=240)
bind_layers(CTORequest, DownloadNext, pid=239)
bind_layers(CTORequest, DownloadMax, pid=238)
bind_layers(CTORequest, ShortDownload, pid=237)
bind_layers(CTORequest, ModifyBits, pid=236)
bind_layers(CTORequest, SetCalPage, pid=235)
bind_layers(CTORequest, GetCalPage, pid=234)
bind_layers(CTORequest, GetPagProcessorInfo, pid=233)
bind_layers(CTORequest, GetSegmentInfo, pid=232)
bind_layers(CTORequest, GetPageInfo, pid=231)
bind_layers(CTORequest, SetSegmentMode, pid=230)
bind_layers(CTORequest, GetSegmentMode, pid=229)
bind_layers(CTORequest, CopyCalPage, pid=228)
bind_layers(CTORequest, SetDaqPtr, pid=226)
bind_layers(CTORequest, WriteDaq, pid=225)
bind_layers(CTORequest, SetDaqListMode, pid=224)
bind_layers(CTORequest, GetDaqListMode, pid=223)
bind_layers(CTORequest, StartStopDaqList, pid=222)
bind_layers(CTORequest, StartStopSynch, pid=221)
bind_layers(CTORequest, ReadDaq, pid=219)
bind_layers(CTORequest, GetDaqClock, pid=220)
bind_layers(CTORequest, GetDaqProcessorInfo, pid=218)
bind_layers(CTORequest, GetDaqResolutionInfo, pid=217)
bind_layers(CTORequest, GetDaqListInfo, pid=216)
bind_layers(CTORequest, GetDaqEventInfo, pid=215)
bind_layers(CTORequest, ClearDaqList, pid=227)
bind_layers(CTORequest, FreeDaq, pid=214)
bind_layers(CTORequest, AllocDaq, pid=213)
bind_layers(CTORequest, AllocOdt, pid=212)
bind_layers(CTORequest, AllocOdtEntry, pid=211)
bind_layers(CTORequest, ProgramStart, pid=210)
bind_layers(CTORequest, ProgramClear, pid=209)
bind_layers(CTORequest, Program, pid=208)
bind_layers(CTORequest, ProgramReset, pid=207)
bind_layers(CTORequest, GetPgmProcessorInfo, pid=206)
bind_layers(CTORequest, GetSectorInfo, pid=205)
bind_layers(CTORequest, ProgramPrepare, pid=204)
bind_layers(CTORequest, ProgramFormat, pid=203)
bind_layers(CTORequest, ProgramNext, pid=202)
bind_layers(CTORequest, ProgramMax, pid=201)
bind_layers(CTORequest, ProgramVerify, pid=200)

class DTO(Packet):
    name = 'Data transfer object'
    fields_desc = [ConditionalField(XByteField('fill', 0), lambda _: identification_field_needs_alignment()), ConditionalField(StrLenField('daq', b'', length_from=lambda _: get_daq_length()), lambda _: get_daq_length() > 0), ConditionalField(StrLenField('timestamp', b'', length_from=lambda _: get_timestamp_length()), lambda _: get_timestamp_length() > 0), ConditionalField(StrLenField('data', b'', length_from=lambda _: get_daq_data_field_length()), lambda _: get_daq_data_field_length() > 0)]
for pid in range(0, 191 + 1):
    bind_layers(CTORequest, DTO, pid=pid)

class CTOResponse(Packet):
    packet_codes = {255: 'RES', 254: 'ERR', 253: 'EV', 252: 'SERV'}
    name = 'Command Transfer Object Response'
    fields_desc = [ByteEnumField('packet_code', 255, packet_codes)]

    @staticmethod
    def get_positive_response_cls(request):
        if False:
            while True:
                i = 10
        request_pid = request.pid
        if request_pid == 242:
            if request.sub_command_code == 255:
                return TransportLayerCmdGetSlaveIdResponse
            if request.sub_command_code == 254:
                return TransportLayerCmdGetDAQIdResponse
        if request_pid == 232:
            if request.mode == 'get_basic_address_info':
                return SegmentInfoMode0PositiveResponse
            if request.mode == 'get_standard_info':
                return SegmentInfoMode1PositiveResponse
            if request.mode == 'get_address_mapping_info':
                return SegmentInfoMode2PositiveResponse
        return {255: ConnectPositiveResponse, 253: StatusPositiveResponse, 251: CommonModeInfoPositiveResponse, 250: IdPositiveResponse, 248: SeedPositiveResponse, 247: UnlockPositiveResponse, 245: UploadPositiveResponse, 244: ShortUploadPositiveResponse, 243: ChecksumPositiveResponse, 234: CalPagePositiveResponse, 233: PagProcessorInfoPositiveResponse, 231: PageInfoPositiveResponse, 229: SegmentModePositiveResponse, 223: DAQListModePositiveResponse, 222: StartStopDAQListPositiveResponse, 220: DAQClockListPositiveResponse, 219: ReadDAQPositiveResponse, 218: DAQProcessorInfoPositiveResponse, 217: DAQResolutionInfoPositiveResponse, 216: DAQListInfoPositiveResponse, 215: DAQEventInfoPositiveResponse, 210: ProgramStartPositiveResponse, 206: PgmProcessorPositiveResponse, 205: SectorInfoPositiveResponse}.get(request_pid, GenericResponse)

    def answers(self, request):
        if False:
            for i in range(10):
                print('nop')
        'In XCP, the payload of a response packet is dependent on the pid\n        field of the corresponding request.\n        This method changes the class of the payload to the class\n        which is expected for the given request.'
        if not isinstance(request, CTORequest):
            return False
        if self.packet_code in [254, 253, 252]:
            return True
        if self.packet_code != 255:
            return False
        payload_cls = self.get_positive_response_cls(request)
        minimum_expected_byte_count = len(payload_cls())
        given_byte_count = len(self.payload)
        if given_byte_count < minimum_expected_byte_count:
            return False
        try:
            data = bytes(self.payload)
            self.remove_payload()
            self.add_payload(payload_cls(data))
        except struct.error:
            return False
        return True
for pid in range(0, 251 + 1):
    bind_layers(CTOResponse, DTO, pid=pid)
positive_response_classes = [ConnectPositiveResponse, StatusPositiveResponse, CommonModeInfoPositiveResponse, IdPositiveResponse, SeedPositiveResponse, UnlockPositiveResponse, UploadPositiveResponse, ShortUploadPositiveResponse, ChecksumPositiveResponse, CalPagePositiveResponse, PagProcessorInfoPositiveResponse, PageInfoPositiveResponse, SegmentModePositiveResponse, DAQListModePositiveResponse, StartStopDAQListPositiveResponse, DAQClockListPositiveResponse, ReadDAQPositiveResponse, DAQProcessorInfoPositiveResponse, DAQResolutionInfoPositiveResponse, DAQListInfoPositiveResponse, DAQEventInfoPositiveResponse, ProgramStartPositiveResponse, PgmProcessorPositiveResponse, SectorInfoPositiveResponse]
for cls in positive_response_classes:
    bind_top_down(CTOResponse, cls, packet_code=255)
bind_layers(CTOResponse, NegativeResponse, packet_code=254)
bind_layers(CTOResponse, EvPacket, packet_code=253)
bind_layers(CTOResponse, ServPacket, packet_code=252)
bind_bottom_up(XCPOnCAN, CTOResponse)
bind_bottom_up(XCPOnUDP, CTOResponse)
bind_bottom_up(XCPOnTCP, CTOResponse)