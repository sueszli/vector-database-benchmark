"""
PNIO RPC endpoints
"""
import struct
from uuid import UUID
from scapy.packet import Packet, Raw, bind_layers
from scapy.config import conf
from scapy.fields import BitField, ByteField, BitEnumField, ByteEnumField, ConditionalField, FieldLenField, FieldListField, IntField, IntEnumField, LenField, MACField, PadField, PacketField, PacketListField, ShortEnumField, ShortField, StrFixedLenField, StrLenField, UUIDField, XByteField, XIntField, XShortEnumField, XShortField
from scapy.layers.dcerpc import DceRpc4, DceRpc4Payload
from scapy.contrib.rtps.common_types import EField
from scapy.compat import bytes_hex
from scapy.volatile import RandUUID
BLOCK_TYPES_ENUM = {1: 'AlarmNotification_High', 2: 'AlarmNotification_Low', 8: 'IODWriteReqHeader', 9: 'IODReadReqHeader', 16: 'DiagnosisData', 18: 'ExpectedIdentificationData', 19: 'RealIdentificationData', 20: 'SubsituteValue', 21: 'RecordInputDataObjectElement', 22: 'RecordOutputDataObjectElement', 24: 'ARData', 25: 'LogBookData', 26: 'APIData', 27: 'SRLData', 32: 'I&M0', 33: 'I&M1', 34: 'I&M2', 35: 'I&M3', 36: 'I&M4', 48: 'I&M0FilterDataSubmodule', 49: 'I&M0FilterDataModule', 50: 'I&M0FilterDataDevice', 257: 'ARBlockReq', 258: 'IOCRBlockReq', 259: 'AlarmCRBlockReq', 260: 'ExpectedSubmoduleBlockReq', 261: 'PrmServerBlockReq', 262: 'MCRBlockReq', 263: 'ARRPCBlockReq', 264: 'ARVendorBlockReq', 265: 'IRInfoBlock', 266: 'SRInfoBlock', 267: 'ARFSUBlock', 272: 'IODBlockReq_connect_end', 273: 'IODBlockReq_plug', 274: 'IOXBlockReq_connect', 275: 'IOXBlockReq_plug', 276: 'ReleaseBlockReq', 278: 'IOXBlockReq_companion', 279: 'IOXBlockReq_rt_class_3', 280: 'IODBlockReq_connect_begin', 281: 'SubmoduleListBlock', 512: 'PDPortDataCheck', 513: 'PdevData', 514: 'PDPortDataAdjust', 515: 'PDSyncData', 516: 'IsochronousModeData', 517: 'PDIRData', 518: 'PDIRGlobalData', 519: 'PDIRFrameData', 520: 'PDIRBeginEndData', 521: 'AdjustDomainBoundary', 522: 'SubBlock_check_Peers', 523: 'SubBlock_check_LineDelay', 524: 'SubBlock_check_MAUType', 526: 'AdjustMAUType', 527: 'PDPortDataReal', 528: 'AdjustMulticastBoundary', 529: 'PDInterfaceMrpDataAdjust', 530: 'PDInterfaceMrpDataReal', 531: 'PDInterfaceMrpDataCheck', 532: 'PDPortMrpDataAdjust', 533: 'PDPortMrpDataReal', 534: 'MrpManagerParams', 535: 'MrpClientParams', 537: 'MrpRingStateData', 539: 'AdjustLinkState', 540: 'CheckLinkState', 542: 'CheckSyncDifference', 543: 'CheckMAUTypeDifference', 544: 'PDPortFODataReal', 545: 'FiberOpticManufacturerSpecific', 546: 'PDPortFODataAdjust', 547: 'PDPortFODataCheck', 548: 'AdjustPeerToPeerBoundary', 549: 'AdjustDCPBoundary', 550: 'AdjustPreambleLength', 552: 'FiberOpticDiagnosisInfo', 554: 'PDIRSubframeData', 555: 'SubframeBlock', 557: 'PDTimeData', 560: 'PDNCDataCheck', 561: 'MrpInstanceDataAdjustBlock', 562: 'MrpInstanceDataRealBlock', 563: 'MrpInstanceDataCheckBlock', 576: 'PDInterfaceDataReal', 592: 'PDInterfaceAdjust', 593: 'PDPortStatistic', 1024: 'MultipleBlockHeader', 1025: 'COContainerContent', 1280: 'RecordDataReadQuery', 1536: 'FSHelloBlock', 1537: 'FSParameterBlock', 1544: 'PDInterfaceFSUDataAdjust', 1545: 'ARFSUDataAdjust', 1792: 'AutoConfiguration', 1793: 'AutoConfigurationCommunication', 1794: 'AutoConfigurationConfiguration', 1795: 'AutoConfigurationIsochronous', 2560: 'UploadBLOBQuery', 2561: 'UploadBLOB', 2562: 'NestedDiagnosisInfo', 3840: 'MaintenanceItem', 3841: 'UploadRecord', 3842: 'iParameterItem', 3843: 'RetrieveRecord', 3844: 'RetrieveAllRecord', 32769: 'AlarmAckHigh', 32770: 'AlarmAckLow', 32776: 'IODWriteResHeader', 32777: 'IODReadResHeader', 33025: 'ARBlockRes', 33026: 'IOCRBlockRes', 33027: 'AlarmCRBlockRes', 33028: 'ModuleDiffBlock', 33029: 'PrmServerBlockRes', 33030: 'ARServerBlockRes', 33031: 'ARRPCBlockRes', 33032: 'ARVendorBlockRes', 33040: 'IODBlockRes_connect_end', 33041: 'IODBlockRes_plug', 33042: 'IOXBlockRes_connect', 33043: 'IOXBlockRes_plug', 33044: 'ReleaseBlockRes', 33046: 'IOXBlockRes_companion', 33047: 'IOXBlockRes_rt_class_3', 33048: 'IODBlockRes_connect_begin'}
IOD_WRITE_REQ_INDEX = {32768: 'ExpectedIdentificationData_subslot', 32769: 'RealIdentificationData_subslot', 32778: 'Diagnosis_channel_subslot', 32779: 'Diagnosis_all_subslot', 32780: 'Diagnosis_Maintenance_subslot', 32784: 'Maintenance_required_in_channel_subslot', 32785: 'Maintenance_demanded_in_channel_subslot', 32786: 'Maintenance_required_in_all_channels_subslot', 32787: 'Maintenance_demanded_in_all_channels_subslot', 32798: 'SubstitueValue_subslot', 32800: 'PDIRSubframeData_subslot', 32808: 'RecordInputDataObjectElement_subslot', 32809: 'RecordOutputDataObjectElement_subslot', 32810: 'PDPortDataReal_subslot', 32811: 'PDPortDataCheck_subslot', 32812: 'PDIRData_subslot', 32813: 'Expected_PDSyncData_subslot', 32815: 'PDPortDataAdjust_subslot', 32816: 'IsochronousModeData_subslot', 32817: 'Expected_PDTimeData_subslot', 32848: 'PDInterfaceMrpDataReal_subslot', 32849: 'PDInterfaceMrpDataCheck_subslot', 32850: 'PDInterfaceMrpDataAdjust_subslot', 32851: 'PDPortMrpDataAdjust_subslot', 32852: 'PDPortMrpDataReal_subslot', 32864: 'PDPortFODataReal_subslot', 32865: 'PDPortFODataCheck_subslot', 32866: 'PDPortFODataAdjust_subslot', 32880: 'PdNCDataCheck_subslot', 32881: 'PDInterfaceAdjust_subslot', 32882: 'PDPortStatistic_subslot', 32896: 'PDInterfaceDataReal_subslot', 32912: 'Expected_PDInterfaceFSUDataAdjust', 32928: 'Energy_saving_profile_record_0', 32944: 'CombinedObjectContainer', 32960: 'Sequence_events_profile_record_0', 45040: 'I&M0', 45041: 'I&M1', 45042: 'I&M2', 45043: 'I&M3', 45044: 'I&M4', 49152: 'Expect edIdentificationData_slot', 49153: 'RealId entificationData_slot', 49162: 'Diagno sis_channel_slot', 49163: 'Diagnosis_all_slot', 49164: 'Diagnosis_Maintenance_slot', 49168: 'Maintenance_required_in_channel_slot', 49169: 'Maintenance_demanded_in_channel_slot', 49170: 'Maintenance_required_in_all_channels_slot', 49171: 'Maintenance_demanded_in_all_channels_slot', 57344: 'ExpectedIdentificationData_AR', 57345: 'RealIdentificationData_AR', 57346: 'ModuleDiffBlock_AR', 57354: 'Diagnosis_channel_AR', 57355: 'Diagnosis_all_AR', 57356: 'Diagnosis_Maintenance_AR', 57360: 'Maintenance_required_in_channel_AR', 57361: 'Maintenance_demanded_in_channel_AR', 57362: 'Maintenance_required_in_all_channels_AR', 57363: 'Maintenance_demanded_in_all_channels_AR', 57408: 'WriteMultiple', 57424: 'ARFSUDataAdjust_AR', 61440: 'RealIdentificationData_API', 61450: 'Diagnosis_channel_API', 61451: 'Diagnosis_all_API', 61452: 'Diagnosis_Maintenance_API', 61456: 'Maintenance_required_in_channel_API', 61457: 'Maintenance_demanded_in_channel_API', 61458: 'Maintenance_required_in_all_channels_API', 61459: 'Maintenance_demanded_in_all_channels_API', 61472: 'ARData_API', 63500: 'Diagnosis_Maintenance_device', 63520: 'ARData', 63521: 'APIData', 63536: 'LogBookData', 63537: 'PdevData', 63552: 'I&M0FilterData', 63553: 'PDRealData', 63554: 'PDExpectedData', 63568: 'AutoConfiguration', 63584: 'GSD_upload', 63585: 'Nested_Diagnosis_info', 64511: 'Trigger_index_CMSM'}
AR_TYPE = {1: 'IOCARSingle', 6: 'IOSAR', 16: 'IOCARSingle_RT_CLASS_3', 32: 'IOCARSR'}
IOCR_TYPE = {1: 'InputCR', 2: 'OutputCR', 3: 'MulticastProviderCR', 4: 'MulticastConsumerCR'}
IOCR_BLOCK_REQ_IOCR_PROPERTIES = {1: 'RT_CLASS_1', 2: 'RT_CLASS_2', 3: 'RT_CLASS_3', 4: 'RT_CLASS_UDP'}
MAU_TYPE = {0: 'Radio', 30: '1000-BaseT-FD'}
MAU_EXTENSION = {0: 'None', 256: 'Polymeric-Optical-Fiber'}
LINKSTATE_LINK = {0: 'Reserved', 1: 'Up', 2: 'Down', 3: 'Testing', 4: 'Unknown', 5: 'Dormant', 6: 'NotPresent', 7: 'LowerLayerDown'}
LINKSTATE_PORT = {0: 'Unknown', 1: 'Disabled/Discarding', 2: 'Blocking', 3: 'Listening', 4: 'Learning', 5: 'Forwarding', 6: 'Broken', 7: 'Reserved'}
MEDIA_TYPE = {0: 'Unknown', 1: 'Copper cable', 2: 'Fiber optic cable', 3: 'Radio communication'}
RPC_INTERFACE_UUID = {'UUID_IO_DeviceInterface': UUID('dea00001-6c97-11d1-8271-00a02442df7d'), 'UUID_IO_ControllerInterface': UUID('dea00002-6c97-11d1-8271-00a02442df7d'), 'UUID_IO_SupervisorInterface': UUID('dea00003-6c97-11d1-8271-00a02442df7d'), 'UUID_IO_ParameterServerInterface': UUID('dea00004-6c97-11d1-8271-00a02442df7d')}

class BlockHeader(Packet):
    """Abstract packet to centralize block headers fields"""
    fields_desc = [ShortEnumField('block_type', None, BLOCK_TYPES_ENUM), ShortField('block_length', None), ByteField('block_version_high', 1), ByteField('block_version_low', 0)]

    def __new__(cls, name, bases, dct):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

class Block(Packet):
    """A generic block packet for PNIO RPC"""
    fields_desc = [BlockHeader, StrLenField('load', '', length_from=lambda pkt: pkt.block_length - 2)]
    block_type = 0

    def post_build(self, p, pay):
        if False:
            return 10
        if self.block_length is None:
            length = len(p) - 4
            p = p[:2] + struct.pack('!H', length) + p[4:]
        return Packet.post_build(self, p, pay)

    def extract_padding(self, s):
        if False:
            while True:
                i = 10
        length = self.payload_length()
        return (s[:length], s[length:])

    def payload_length(self):
        if False:
            for i in range(10):
                print('nop')
        ' A function for each block, to determine the length of\n        the payload '
        return 0

class IODControlReq(Block):
    """IODControl request block"""
    fields_desc = [BlockHeader, StrFixedLenField('padding', '', length=2), UUIDField('ARUUID', None), ShortField('SessionKey', 0), XShortField('AlarmSequenceNumber', 0), BitField('ControlCommand_reserved', 0, 9), BitField('ControlCommand_PrmBegin', 0, 1), BitField('ControlCommand_ReadyForRT_CLASS_3', 0, 1), BitField('ControlCommand_ReadyForCompanion', 0, 1), BitField('ControlCommand_Done', 0, 1), BitField('ControlCommand_Release', 0, 1), BitField('ControlCommand_ApplicationReady', 0, 1), BitField('ControlCommand_PrmEnd', 0, 1), XShortField('ControlBlockProperties', 0)]

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        if self.block_type is None:
            if self.ControlCommand_PrmBegin:
                p = struct.pack('!H', 280) + p[2:]
            elif self.ControlCommand_ReadyForRT_CLASS_3:
                p = struct.pack('!H', 279) + p[2:]
            elif self.ControlCommand_ReadyForCompanion:
                p = struct.pack('!H', 278) + p[2:]
            elif self.ControlCommand_Release:
                p = struct.pack('!H', 276) + p[2:]
            elif self.ControlCommand_ApplicationReady:
                if self.AlarmSequenceNumber > 0:
                    p = struct.pack('!H', 275) + p[2:]
                else:
                    p = struct.pack('!H', 274) + p[2:]
            elif self.ControlCommand_PrmEnd:
                if self.AlarmSequenceNumber > 0:
                    p = struct.pack('!H', 273) + p[2:]
                else:
                    p = struct.pack('!H', 272) + p[2:]
        return Block.post_build(self, p, pay)

    def get_response(self):
        if False:
            i = 10
            return i + 15
        'Generate the response block of this request.\n        Careful: it only sets the fields which can be set from the request\n        '
        res = IODControlRes()
        for field in ['ARUUID', 'SessionKey', 'AlarmSequenceNumber']:
            res.setfieldval(field, self.getfieldval(field))
        res.block_type = self.block_type + 32768
        return res

class IODControlRes(Block):
    """IODControl response block"""
    fields_desc = [BlockHeader, StrFixedLenField('padding', '', length=2), UUIDField('ARUUID', None), ShortField('SessionKey', 0), XShortField('AlarmSequenceNumber', 0), BitField('ControlCommand_reserved', 0, 9), BitField('ControlCommand_PrmBegin', 0, 1), BitField('ControlCommand_ReadyForRT_CLASS_3', 0, 1), BitField('ControlCommand_ReadyForCompanion', 0, 1), BitField('ControlCommand_Done', 1, 1), BitField('ControlCommand_Release', 0, 1), BitField('ControlCommand_ApplicationReady', 0, 1), BitField('ControlCommand_PrmEnd', 0, 1), XShortField('ControlBlockProperties', 0)]
    block_type = 33040

class IODWriteReq(Block):
    """IODWrite request block"""
    fields_desc = [BlockHeader, ShortField('seqNum', 0), UUIDField('ARUUID', None), XIntField('API', 0), XShortField('slotNumber', 0), XShortField('subslotNumber', 0), StrFixedLenField('padding', '', length=2), XShortEnumField('index', 0, IOD_WRITE_REQ_INDEX), LenField('recordDataLength', None, fmt='I'), StrFixedLenField('RWPadding', '', length=24)]
    block_type = 8

    def payload_length(self):
        if False:
            i = 10
            return i + 15
        return self.recordDataLength

    def get_response(self):
        if False:
            print('Hello World!')
        'Generate the response block of this request.\n        Careful: it only sets the fields which can be set from the request\n        '
        res = IODWriteRes()
        for field in ['seqNum', 'ARUUID', 'API', 'slotNumber', 'subslotNumber', 'index']:
            res.setfieldval(field, self.getfieldval(field))
        return res

class IODWriteRes(Block):
    """IODWrite response block"""
    fields_desc = [BlockHeader, ShortField('seqNum', 0), UUIDField('ARUUID', None), XIntField('API', 0), XShortField('slotNumber', 0), XShortField('subslotNumber', 0), StrFixedLenField('padding', '', length=2), XShortEnumField('index', 0, IOD_WRITE_REQ_INDEX), LenField('recordDataLength', None, fmt='I'), XShortField('additionalValue1', 0), XShortField('additionalValue2', 0), IntEnumField('status', 0, ['OK']), StrFixedLenField('RWPadding', '', length=16)]
    block_type = 32776

class IODReadReq(Block):
    """IODRead request block"""
    fields_desc = [BlockHeader, ShortField('seqNum', 0), UUIDField('ARUUID', None), XIntField('API', 0), XShortField('slotNumber', 0), XShortField('subslotNumber', 0), StrFixedLenField('padding', '', length=2), XShortEnumField('index', 0, IOD_WRITE_REQ_INDEX), LenField('recordDataLength', None, fmt='I'), StrFixedLenField('RWPadding', '', length=24)]
    block_type = 9

    def payload_length(self):
        if False:
            return 10
        return self.recordDataLength

    def get_response(self):
        if False:
            return 10
        res = IODReadRes()
        for field in ['seqNum', 'ARUUID', 'API', 'slotNumber', 'subslotNumber', 'index']:
            res.setfieldval(field, self.getfieldval(field))
        return res

class IODReadRes(Block):
    """IODRead response block"""
    fields_desc = [BlockHeader, ShortField('seqNum', 0), UUIDField('ARUUID', None), XIntField('API', 0), XShortField('slotNumber', 0), XShortField('subslotNumber', 0), StrFixedLenField('padding', '', length=2), XShortEnumField('index', 0, IOD_WRITE_REQ_INDEX), LenField('recordDataLength', None, fmt='I'), XShortField('additionalValue1', 0), XShortField('additionalValue2', 0), StrFixedLenField('RWPadding', '', length=20)]
    block_type = 32777
F_PARAMETERS_BLOCK_ID = ['No_F_WD_Time2_No_F_iPar_CRC', 'No_F_WD_Time2_F_iPar_CRC', 'F_WD_Time2_No_F_iPar_CRC', 'F_WD_Time2_F_iPar_CRC', 'reserved_4', 'reserved_5', 'reserved_6', 'reserved_7']

class FParametersBlock(Packet):
    """F-Parameters configuration block"""
    name = 'F-Parameters Block'
    fields_desc = [BitField('F_Prm_Flag1_Reserved_7', 0, 1), BitField('F_CRC_Seed', 0, 1), BitEnumField('F_CRC_Length', 0, 2, ['CRC-24', 'depreciated', 'CRC-32', 'reserved']), BitEnumField('F_SIL', 2, 2, ['SIL_1', 'SIL_2', 'SIL_3', 'No_SIL']), BitField('F_Check_iPar', 0, 1), BitField('F_Check_SeqNr', 0, 1), BitEnumField('F_Par_Version', 1, 2, ['V1', 'V2', 'reserved_2', 'reserved_3']), BitEnumField('F_Block_ID', 0, 3, F_PARAMETERS_BLOCK_ID), BitField('F_Prm_Flag2_Reserved', 0, 2), BitField('F_Passivation', 0, 1), XShortField('F_Source_Add', 0), XShortField('F_Dest_Add', 0), ShortField('F_WD_Time', 0), ConditionalField(cond=lambda p: p.getfieldval('F_Block_ID') & 6 == 2, fld=ShortField('F_WD_Time_2', 0)), ConditionalField(cond=lambda p: p.getfieldval('F_Block_ID') & 5 == 1, fld=XIntField('F_iPar_CRC', 0)), XShortField('F_Par_CRC', 0)]
    overload_fields = {IODWriteReq: {'index': 256}}
bind_layers(IODWriteReq, FParametersBlock, index=256)
bind_layers(FParametersBlock, conf.padding_layer)

class PadFieldWithLen(PadField):
    """PadField which handles the i2len function to include padding"""

    def i2len(self, pkt, val):
        if False:
            return 10
        'get the length of the field, including the padding length'
        fld_len = self.fld.i2len(pkt, val)
        return fld_len + self.padlen(fld_len, pkt)

class IODWriteMultipleReq(Block):
    """IODWriteMultiple request"""
    fields_desc = [BlockHeader, ShortField('seqNum', 0), UUIDField('ARUUID', None), XIntField('API', 4294967295), XShortField('slotNumber', 65535), XShortField('subslotNumber', 65535), StrFixedLenField('padding', '', length=2), XShortEnumField('index', 0, IOD_WRITE_REQ_INDEX), FieldLenField('recordDataLength', None, fmt='I', length_of='blocks'), StrFixedLenField('RWPadding', '', length=24), FieldListField('blocks', [], PadFieldWithLen(PacketField('', None, IODWriteReq), 4), length_from=lambda pkt: pkt.recordDataLength)]
    block_type = 8
    index = 57408
    API = 4294967295
    slotNumber = 65535
    subslotNumber = 65535

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        if self.block_length is None:
            p = p[:2] + struct.pack('!H', 60) + p[4:]
        (fld, val) = self.getfield_and_val('blocks')
        if fld.i2count(self, val) > 0:
            length = len(val[-1])
            pad = fld.field.padlen(length, self)
            if pad > 0:
                p = p[:-pad]
                if self.recordDataLength is None:
                    val = struct.unpack('!I', p[36:40])[0]
                    val -= pad
                    p = p[:36] + struct.pack('!I', val) + p[40:]
        return Packet.post_build(self, p, pay)

    def get_response(self):
        if False:
            print('Hello World!')
        'Generate the response block of this request.\n        Careful: it only sets the fields which can be set from the request\n        '
        res = IODWriteMultipleRes()
        for field in ['seqNum', 'ARUUID', 'API', 'slotNumber', 'subslotNumber', 'index']:
            res.setfieldval(field, self.getfieldval(field))
        res_blocks = []
        for block in self.getfieldval('blocks'):
            res_blocks.append(block.get_response())
        res.setfieldval('blocks', res_blocks)
        return res

class IODWriteMultipleRes(Block):
    """IODWriteMultiple response"""
    fields_desc = [BlockHeader, ShortField('seqNum', 0), UUIDField('ARUUID', None), XIntField('API', 4294967295), XShortField('slotNumber', 65535), XShortField('subslotNumber', 65535), StrFixedLenField('padding', '', length=2), XShortEnumField('index', 0, IOD_WRITE_REQ_INDEX), FieldLenField('recordDataLength', None, fmt='I', length_of='blocks'), XShortField('additionalValue1', 0), XShortField('additionalValue2', 0), IntEnumField('status', 0, ['OK']), StrFixedLenField('RWPadding', '', length=16), FieldListField('blocks', [], PacketField('', None, IODWriteRes), length_from=lambda pkt: pkt.recordDataLength)]
    block_type = 32776
    index = 57408

    def post_build(self, p, pay):
        if False:
            for i in range(10):
                print('nop')
        if self.block_length is None:
            p = p[:2] + struct.pack('!H', 60) + p[4:]
        return Packet.post_build(self, p, pay)

class IM0Block(Block):
    """Identification and Maintenance 0"""
    fields_desc = [BlockHeader, ByteField('VendorIDHigh', 0), ByteField('VendorIDLow', 0), StrFixedLenField('OrderID', '', length=20), StrFixedLenField('IMSerialNumber', '', length=16), ShortField('IMHardwareRevision', 0), StrFixedLenField('IMSWRevisionPrefix', 'V', length=1), ByteField('IMSWRevisionFunctionalEnhancement', 0), ByteField('IMSWRevisionBugFix', 0), ByteField('IMSWRevisionInternalChange', 0), ShortField('IMRevisionCounter', 0), ShortField('IMProfileID', 0), ShortField('IMProfileSpecificType', 0), ByteField('IMVersionMajor', 1), ByteField('IMVersionMinor', 1), ShortField('IMSupported', 0)]
    block_type = 32

class IM1Block(Block):
    """Identification and Maintenance 1"""
    fields_desc = [BlockHeader, StrFixedLenField('IMTagFunction', '', length=32), StrFixedLenField('IMTagLocation', '', length=22)]
    block_type = 33

class IM2Block(Block):
    """Identification and Maintenance 2"""
    fields_desc = [BlockHeader, StrFixedLenField('IMDate', '', length=16)]
    block_type = 34

class IM3Block(Block):
    """Identification and Maintenance 3"""
    fields_desc = [BlockHeader, StrFixedLenField('IMDescriptor', '', length=54)]
    block_type = 35

class IM4Block(Block):
    """Identification and Maintenance 4"""
    fields_desc = [BlockHeader, StrFixedLenField('IMSignature', '', 54)]
    block_type = 36

class ARBlockReq(Block):
    """Application relationship block request"""
    fields_desc = [BlockHeader, XShortEnumField('ARType', 1, AR_TYPE), UUIDField('ARUUID', None), ShortField('SessionKey', 0), MACField('CMInitiatorMacAdd', None), UUIDField('CMInitiatorObjectUUID', None), BitField('ARProperties_PullModuleAlarmAllowed', 0, 1), BitEnumField('ARProperties_StartupMode', 0, 1, ['Legacy', 'Advanced']), BitField('ARProperties_reserved_3', 0, 6), BitField('ARProperties_reserved_2', 0, 12), BitField('ARProperties_AcknowledgeCompanionAR', 0, 1), BitEnumField('ARProperties_CompanionAR', 0, 2, ['Single_AR', 'First_AR', 'Companion_AR', 'reserved']), BitEnumField('ARProperties_DeviceAccess', 0, 1, ['ExpectedSubmodule', 'Controlled_by_IO_device_app']), BitField('ARProperties_reserved_1', 0, 3), BitEnumField('ARProperties_ParametrizationServer', 0, 1, ['External_PrmServer', 'CM_Initator']), BitField('ARProperties_SupervisorTakeoverAllowed', 0, 1), BitEnumField('ARProperties_State', 1, 3, {1: 'Active'}), ShortField('CMInitiatorActivityTimeoutFactor', 1000), ShortField('CMInitiatorUDPRTPort', 34962), FieldLenField('StationNameLength', None, fmt='H', length_of='CMInitiatorStationName'), StrLenField('CMInitiatorStationName', '', length_from=lambda pkt: pkt.StationNameLength)]
    block_type = 257

    def get_response(self):
        if False:
            i = 10
            return i + 15
        'Generate the response block of this request.\n        Careful: it only sets the fields which can be set from the request\n        '
        res = ARBlockRes()
        for field in ['ARType', 'ARUUID', 'SessionKey']:
            res.setfieldval(field, self.getfieldval(field))
        return res

class ARBlockRes(Block):
    """Application relationship block response"""
    fields_desc = [BlockHeader, XShortEnumField('ARType', 1, AR_TYPE), UUIDField('ARUUID', None), ShortField('SessionKey', 0), MACField('CMResponderMacAdd', None), ShortField('CMResponderUDPRTPort', 34962)]
    block_type = 33025

class IOCRAPIObject(Packet):
    """API item descriptor used in API description of IOCR blocks"""
    name = 'API item'
    fields_desc = [XShortField('SlotNumber', 0), XShortField('SubslotNumber', 0), ShortField('FrameOffset', 0)]

    def extract_padding(self, s):
        if False:
            return 10
        return (None, s)

class IOCRAPI(Packet):
    """API description used in IOCR block"""
    name = 'API'
    fields_desc = [XIntField('API', 0), FieldLenField('NumberOfIODataObjects', None, count_of='IODataObjects'), PacketListField('IODataObjects', [], IOCRAPIObject, count_from=lambda p: p.NumberOfIODataObjects), FieldLenField('NumberOfIOCS', None, count_of='IOCSs'), PacketListField('IOCSs', [], IOCRAPIObject, count_from=lambda p: p.NumberOfIOCS)]

    def extract_padding(self, s):
        if False:
            print('Hello World!')
        return (None, s)

class IOCRBlockReq(Block):
    """IO Connection Relationship block request"""
    fields_desc = [BlockHeader, XShortEnumField('IOCRType', 1, IOCR_TYPE), XShortField('IOCRReference', 1), XShortField('LT', 34962), BitField('IOCRProperties_reserved3', 0, 8), BitField('IOCRProperties_reserved2', 0, 11), BitField('IOCRProperties_reserved1', 0, 9), BitEnumField('IOCRProperties_RTClass', 0, 4, IOCR_BLOCK_REQ_IOCR_PROPERTIES), ShortField('DataLength', 40), XShortField('FrameID', 32768), ShortField('SendClockFactor', 32), ShortField('ReductionRatio', 32), ShortField('Phase', 1), ShortField('Sequence', 0), XIntField('FrameSendOffset', 4294967295), ShortField('WatchdogFactor', 10), ShortField('DataHoldFactor', 10), BitEnumField('IOCRTagHeader_IOUserPriority', 6, 3, {6: 'IOCRPriority'}), BitField('IOCRTagHeader_reserved', 0, 1), BitField('IOCRTagHeader_IOCRVLANID', 0, 12), MACField('IOCRMulticastMACAdd', None), FieldLenField('NumberOfAPIs', None, fmt='H', count_of='APIs'), PacketListField('APIs', [], IOCRAPI, count_from=lambda p: p.NumberOfAPIs)]
    block_type = 258

    def get_response(self):
        if False:
            while True:
                i = 10
        'Generate the response block of this request.\n        Careful: it only sets the fields which can be set from the request\n        '
        res = IOCRBlockRes()
        for field in ['IOCRType', 'IOCRReference', 'FrameID']:
            res.setfieldval(field, self.getfieldval(field))
        return res

class IOCRBlockRes(Block):
    """IO Connection Relationship block response"""
    fields_desc = [BlockHeader, XShortEnumField('IOCRType', 1, IOCR_TYPE), XShortField('IOCRReference', 1), XShortField('FrameID', 32768)]
    block_type = 33026

class AdjustLinkState(Block):
    fields_desc = [BlockHeader, StrFixedLenField('padding', '', length=2), XShortEnumField('LinkState', 0, LINKSTATE_LINK), ShortField('AdjustProperties', 0)]
    block_type = 539

class AdjustPeerToPeerBoundary(Block):
    fields_desc = [BlockHeader, StrFixedLenField('padding1', '', length=2), IntField('peerToPeerBoundary', 0), ShortField('adjustProperties', 0), PadField(ShortField('padding2', 0), 2)]
    block_type = 548

class AdjustDomainBoundary(Block):
    fields_desc = [BlockHeader, StrFixedLenField('padding1', '', length=2), IntEnumField('DomainBoundaryIngress', 0, {0: 'No Block', 1: 'Block'}), IntEnumField('DomainBoundaryEgress', 0, {0: 'No Block', 1: 'Block'}), ShortField('adjustProperties', 0), PadField(ShortField('padding2', 0), 2)]
    block_type = 521

class AdjustMulticastBoundary(Block):
    fields_desc = [BlockHeader, StrFixedLenField('padding1', '', length=2), IntField('MulticastAddress', 0), ShortField('adjustProperties', 0), PadField(ShortField('padding2', 0), 2)]
    block_type = 528

class AdjustMauType(Block):
    fields_desc = [BlockHeader, PadField(ShortField('padding', 0), 2), XShortEnumField('MAUType', 1, MAU_TYPE), ShortField('adjustProperties', 0)]
    block_type = 526

class AdjustMauTypeExtension(Block):
    fields_desc = [BlockHeader, PadField(ShortField('padding', 0), 2), XShortEnumField('MAUTypeExtension', 0, MAU_EXTENSION), ShortField('adjustProperties', 0)]
    block_type = 553

class AdjustDCPBoundary(Block):
    fields_desc = [BlockHeader, StrFixedLenField('padding1', '', length=2), IntField('dcpBoundary', 0), ShortField('adjustProperties', 0), PadField(ShortField('padding2', 0), 2)]
    block_type = 549
PDPORT_ADJUST_BLOCK_ASSOCIATION = {521: AdjustDomainBoundary, 526: AdjustMauType, 528: AdjustMulticastBoundary, 539: AdjustLinkState, 548: AdjustPeerToPeerBoundary, 549: AdjustDCPBoundary, 553: AdjustMauTypeExtension}

def _guess_pdportadjust_block(_pkt, *args, **kargs):
    if False:
        while True:
            i = 10
    cls = Block
    btype = struct.unpack('!H', _pkt[:2])[0]
    if btype in PDPORT_ADJUST_BLOCK_ASSOCIATION:
        cls = PDPORT_ADJUST_BLOCK_ASSOCIATION[btype]
    return cls(_pkt, *args, **kargs)

class PDPortDataAdjust(Block):
    fields_desc = [BlockHeader, StrFixedLenField('padding', '', length=2), XShortField('slotNumber', 0), XShortField('subslotNumber', 0), PacketListField('blocks', [], _guess_pdportadjust_block, length_from=lambda p: p.block_length)]
    block_type = 514

class ExpectedSubmoduleDataDescription(Packet):
    """Description of the data of a submodule"""
    name = 'Data Description'
    fields_desc = [XShortEnumField('DataDescription', 0, {1: 'Input', 2: 'Output'}), ShortField('SubmoduleDataLength', 0), ByteField('LengthIOCS', 0), ByteField('LengthIOPS', 0)]

    def extract_padding(self, s):
        if False:
            for i in range(10):
                print('nop')
        return (None, s)

class ExpectedSubmodule(Packet):
    """Description of a submodule in an API of an expected submodule"""
    name = 'Submodule'
    fields_desc = [XShortField('SubslotNumber', 0), XIntField('SubmoduleIdentNumber', 0), XByteField('SubmoduleProperties_reserved_2', 0), BitField('SubmoduleProperties_reserved_1', 0, 2), BitField('SubmoduleProperties_DiscardIOXS', 0, 1), BitField('SubmoduleProperties_ReduceOutputSubmoduleDataLength', 0, 1), BitField('SubmoduleProperties_ReduceInputSubmoduleDataLength', 0, 1), BitField('SubmoduleProperties_SharedInput', 0, 1), BitEnumField('SubmoduleProperties_Type', 0, 2, ['NO_IO', 'INPUT', 'OUTPUT', 'INPUT_OUTPUT']), PacketListField('DataDescription', [], ExpectedSubmoduleDataDescription, count_from=lambda p: 2 if p.SubmoduleProperties_Type == 3 else 1)]

    def extract_padding(self, s):
        if False:
            while True:
                i = 10
        return (None, s)

class ExpectedSubmoduleAPI(Packet):
    """Description of an API in the expected submodules blocks"""
    name = 'API'
    fields_desc = [XIntField('API', 0), XShortField('SlotNumber', 0), XIntField('ModuleIdentNumber', 0), XShortField('ModuleProperties', 0), FieldLenField('NumberOfSubmodules', None, fmt='H', count_of='Submodules'), PacketListField('Submodules', [], ExpectedSubmodule, count_from=lambda p: p.NumberOfSubmodules)]

    def extract_padding(self, s):
        if False:
            print('Hello World!')
        return (None, s)

class ExpectedSubmoduleBlockReq(Block):
    """Expected submodule block request"""
    fields_desc = [BlockHeader, FieldLenField('NumberOfAPIs', None, fmt='H', count_of='APIs'), PacketListField('APIs', [], ExpectedSubmoduleAPI, count_from=lambda p: p.NumberOfAPIs)]
    block_type = 260

    def get_response(self):
        if False:
            i = 10
            return i + 15
        'Generate the response block of this request.\n        Careful: it only sets the fields which can be set from the request\n        '
        return None
ALARM_CR_TYPE = {1: 'AlarmCR'}
ALARM_CR_TRANSPORT = {0: 'RTA_CLASS_1', 1: 'RTA_CLASS_UDP'}

class AlarmCRBlockReq(Block):
    """Alarm CR block request"""
    fields_desc = [BlockHeader, XShortEnumField('AlarmCRType', 1, ALARM_CR_TYPE), ShortField('LT', 34962), BitField('AlarmCRProperties_Priority', 0, 1), BitEnumField('AlarmCRProperties_Transport', 0, 1, ALARM_CR_TRANSPORT), BitField('AlarmCRProperties_Reserved1', 0, 22), BitField('AlarmCRProperties_Reserved2', 0, 8), ShortField('RTATimeoutFactor', 1), ShortField('RTARetries', 3), ShortField('LocalAlarmReference', 3), ShortField('MaxAlarmDataLength', 200), ShortField('AlarmCRTagHeaderHigh', 49152), ShortField('AlarmCRTagHeaderLow', 40960)]
    block_type = 259

    def post_build(self, p, pay):
        if False:
            for i in range(10):
                print('nop')
        if self.AlarmCRProperties_Transport == 1:
            p = p[:8] + struct.pack('!H', 2048) + p[10:]
        return Block.post_build(self, p, pay)

    def get_response(self):
        if False:
            return 10
        'Generate the response block of this request.\n        Careful: it only sets the fields which can be set from the request\n        '
        res = AlarmCRBlockRes()
        for field in ['AlarmCRType', 'LocalAlarmReference']:
            res.setfieldval(field, self.getfieldval(field))
        res.block_type = self.block_type + 32768
        return res

class AlarmCRBlockRes(Block):
    fields_desc = [BlockHeader, XShortEnumField('AlarmCRType', 1, ALARM_CR_TYPE), ShortField('LocalAlarmReference', 0), ShortField('MaxAlarmDataLength', 0)]
    block_type = 33027

class AlarmItem(Packet):
    fields_desc = [XShortField('UserStructureIdentifier', 0), PacketField('load', '', Raw)]

    def extract_padding(self, s):
        if False:
            print('Hello World!')
        return (None, s)

class MaintenanceItem(AlarmItem):
    fields_desc = [XShortField('UserStructureIdentifier', 0), BlockHeader, StrFixedLenField('padding', '', length=2), XIntField('MaintenanceStatus', 0)]

class DiagnosisItem(AlarmItem):
    fields_desc = [XShortField('UserStructureIdentifier', 0), XShortField('ChannelNumber', 0), XShortField('ChannelProperties', 0), XShortField('ChannelErrorType', 0), ConditionalField(cond=lambda p: p.getfieldval('UserStructureIdentifier') in [32770, 32771], fld=XShortField('ExtChannelErrorType', 0)), ConditionalField(cond=lambda p: p.getfieldval('UserStructureIdentifier') in [32770, 32771], fld=XIntField('ExtChannelAddValue', 0)), ConditionalField(cond=lambda p: p.getfieldval('UserStructureIdentifier') == 32771, fld=XIntField('QualifiedChannelQualifier', 0))]

class UploadRetrievalItem(AlarmItem):
    fields_desc = [XShortField('UserStructureIdentifier', 0), BlockHeader, StrFixedLenField('padding', '', length=2), XIntField('URRecordIndex', 0), XIntField('URRecordLength', 0)]

class iParameterItem(AlarmItem):
    fields_desc = [XShortField('UserStructureIdentifier', 0), BlockHeader, StrFixedLenField('padding', '', length=2), XIntField('iPar_Req_Header', 0), XIntField('Max_Segm_Size', 0), XIntField('Transfer_Index', 0), XIntField('Total_iPar_Size', 0)]
PE_OPERATIONAL_MODE = {0: 'PE_PowerOff', 240: 'PE_Operate', 254: 'PE_SleepModeWOL', 255: 'PE_ReadyToOperate'}
PE_OPERATIONAL_MODE.update({i: 'PE_EnergySavingMode_{}'.format(i) for i in range(1, 32)})
PE_OPERATIONAL_MODE.update({i: 'Reserved' for i in range(32, 240)})
PE_OPERATIONAL_MODE.update({i: 'Reserved' for i in range(241, 254)})

class PE_AlarmItem(AlarmItem):
    fields_desc = [XShortField('UserStructureIdentifier', 0), BlockHeader, ByteEnumField('PE_OperationalMode', 0, PE_OPERATIONAL_MODE)]

class RS_AlarmItem(AlarmItem):
    fields_desc = [XShortField('UserStructureIdentifier', 0), XShortField('RS_AlarmInfo', 0)]

class PRAL_AlarmItem(AlarmItem):
    fields_desc = [XShortField('UserStructureIdentifier', 0), XShortField('ChannelNumber', 0), XShortField('PRAL_ChannelProperties', 0), XShortField('PRAL_Reason', 0), XShortField('PRAL_ExtReason', 0), StrLenField('PRAL_ReasonAddValue', '', length_from=lambda x: x.len - 10)]
PNIO_RPC_ALARM_ASSOCIATION = {'8000': DiagnosisItem, '8002': DiagnosisItem, '8003': DiagnosisItem, '8100': MaintenanceItem, '8200': UploadRetrievalItem, '8201': iParameterItem, '8300': RS_AlarmItem, '8301': RS_AlarmItem, '8302': RS_AlarmItem, '8310': PE_AlarmItem, '8320': PRAL_AlarmItem}

def _guess_alarm_payload(_pkt, *args, **kargs):
    if False:
        i = 10
        return i + 15
    cls = AlarmItem
    btype = bytes_hex(_pkt[:2]).decode('utf8')
    if btype in PNIO_RPC_ALARM_ASSOCIATION:
        cls = PNIO_RPC_ALARM_ASSOCIATION[btype]
    return cls(_pkt, *args, **kargs)

class AlarmNotificationPDU(Block):
    fields_desc = [BlockHeader, ShortField('AlarmType', 0), XIntField('API', 0), ShortField('SlotNumber', 0), ShortField('SubslotNumber', 0), XIntField('ModuleIdentNumber', 0), XIntField('SubmoduleIdentNUmber', 0), XShortField('AlarmSpecifier', 0), PacketListField('AlarmPayload', [], _guess_alarm_payload)]

class AlarmNotification_High(AlarmNotificationPDU):
    block_type = 1

class AlarmNotification_Low(AlarmNotificationPDU):
    block_type = 2
PDU_TYPE_TYPE = {1: 'RTA_TYPE_DATA', 2: 'RTA_TYPE_NACK', 3: 'RTA_TYPE_ACK', 4: 'RTA_TYPE_ERR', 5: 'RTA_TYPE_FREQ', 6: 'RTA_TYPE_FRSP'}
PDU_TYPE_TYPE.update({i: 'Reserved' for i in range(7, 16)})
PDU_TYPE_VERSION = {0: 'Reserved', 1: 'Version 1', 2: 'Version 2'}
PDU_TYPE_VERSION.update({i: 'Reserved' for i in range(3, 16)})

class PNIORealTimeAcyclicPDUHeader(Packet):
    fields_desc = [ShortField('AlarmDstEndpoint', 0), ShortField('AlarmSrcEndpoint', 0), BitEnumField('PDUTypeType', 0, 4, PDU_TYPE_TYPE), BitEnumField('PDUTypeVersion', 0, 4, PDU_TYPE_VERSION), BitField('AddFlags', 0, 8), XShortField('SendSeqNum', 0), XShortField('AckSeqNum', 0), XShortField('VarPartLen', 0)]

    def __new__(cls, name, bases, dct):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

class Alarm_Low(Packet):
    fields_desc = [PNIORealTimeAcyclicPDUHeader, PacketField('RTA_SDU', None, AlarmNotification_Low)]

class Alarm_High(Packet):
    fields_desc = [PNIORealTimeAcyclicPDUHeader, PacketField('RTA_SDU', None, AlarmNotification_High)]
PNIO_RPC_BLOCK_ASSOCIATION = {'0020': IM0Block, '0021': IM1Block, '0022': IM2Block, '0023': IM3Block, '0024': IM4Block, '0101': ARBlockReq, '0102': IOCRBlockReq, '0103': AlarmCRBlockReq, '0104': ExpectedSubmoduleBlockReq, '0110': IODControlReq, '0111': IODControlReq, '0112': IODControlReq, '0113': IODControlReq, '0114': IODControlReq, '0116': IODControlReq, '0117': IODControlReq, '0118': IODControlReq, '0202': PDPortDataAdjust, '8101': ARBlockRes, '8102': IOCRBlockRes, '8103': AlarmCRBlockRes, '8110': IODControlRes, '8111': IODControlRes, '8112': IODControlRes, '8113': IODControlRes, '8114': IODControlRes, '8116': IODControlRes, '8117': IODControlRes, '8118': IODControlRes}

def _guess_block_class(_pkt, *args, **kargs):
    if False:
        i = 10
        return i + 15
    cls = Block
    if _pkt[:2] == b'\x00\x08':
        if _pkt[34:36] == b'\xe0@':
            cls = IODWriteMultipleReq
        else:
            cls = IODWriteReq
    elif _pkt[:2] == b'\x00\t':
        cls = IODReadReq
    elif _pkt[:2] == b'\x80\x08':
        if _pkt[34:36] == b'\xe0@':
            cls = IODWriteMultipleRes
        else:
            cls = IODWriteRes
    elif _pkt[:2] == b'\x80\t':
        cls = IODReadRes
    else:
        btype = bytes_hex(_pkt[:2]).decode('utf8')
        if btype in PNIO_RPC_BLOCK_ASSOCIATION:
            cls = PNIO_RPC_BLOCK_ASSOCIATION[btype]
    return cls(_pkt, *args, **kargs)

def dce_rpc_endianess(pkt):
    if False:
        while True:
            i = 10
    'determine the symbol for the endianness of a the DCE/RPC'
    try:
        endianness = pkt.underlayer.endian
    except AttributeError:
        return '!'
    if endianness == 0:
        return '>'
    elif endianness == 1:
        return '<'
    else:
        return '!'

class NDRData(Packet):
    """Base NDRData to centralize some fields. It can't be instantiated"""
    fields_desc = [EField(FieldLenField('args_length', None, fmt='I', length_of='blocks'), endianness_from=dce_rpc_endianess), EField(FieldLenField('max_count', None, fmt='I', length_of='blocks'), endianness_from=dce_rpc_endianess), EField(IntField('offset', 0), endianness_from=dce_rpc_endianess), EField(FieldLenField('actual_count', None, fmt='I', length_of='blocks'), endianness_from=dce_rpc_endianess), PacketListField('blocks', [], _guess_block_class, length_from=lambda p: p.args_length)]

    def __new__(cls, name, bases, dct):
        if False:
            print('Hello World!')
        raise NotImplementedError()

class PNIOServiceReqPDU(Packet):
    """PNIO PDU for RPC Request"""
    fields_desc = [EField(FieldLenField('args_max', None, fmt='I', length_of='blocks'), endianness_from=dce_rpc_endianess), NDRData]
    overload_fields = {DceRpc4: {'object': RandUUID('dea00000-6c97-11d1-8271-******'), 'if_id': RPC_INTERFACE_UUID['UUID_IO_DeviceInterface'], 'ptype': 0}}

    @classmethod
    def can_handle(cls, pkt, rpc):
        if False:
            i = 10
            return i + 15
        'heuristic guess_payload_class'
        if rpc.ptype == 0 and str(rpc.object).startswith('dea00000-6c97-11d1-8271-'):
            return True
        return False
DceRpc4Payload.register_possible_payload(PNIOServiceReqPDU)

class PNIOServiceResPDU(Packet):
    """PNIO PDU for RPC Response"""
    fields_desc = [EField(IntEnumField('status', 0, ['OK']), endianness_from=dce_rpc_endianess), NDRData]
    overload_fields = {DceRpc4: {'object': RandUUID('dea00000-6c97-11d1-8271-******'), 'if_id': RPC_INTERFACE_UUID['UUID_IO_ControllerInterface'], 'ptype': 2}}

    @classmethod
    def can_handle(cls, pkt, rpc):
        if False:
            while True:
                i = 10
        'heuristic guess_payload_class'
        if rpc.ptype == 2 and str(rpc.object).startswith('dea00000-6c97-11d1-8271-'):
            return True
        return False
DceRpc4Payload.register_possible_payload(PNIOServiceResPDU)