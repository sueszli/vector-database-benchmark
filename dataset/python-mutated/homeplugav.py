"""
HomePlugAV Layer for Scapy

Copyright (C) FlUxIuS (Sebastien Dudek)

HomePlugAV Management Message Type
Key (type value) : Description
"""
import struct
from scapy.packet import Packet, bind_layers
from scapy.fields import BitField, ByteEnumField, ByteField, ConditionalField, EnumField, FieldLenField, IntField, LEIntField, LELongField, LEShortEnumField, LEShortField, MACField, OUIField, PacketListField, ShortField, StrFixedLenField, StrLenField, X3BytesField, XByteField, XIntField, XLongField, XShortField
from scapy.layers.l2 import Ether
HPAVTypeList = {40960: "'Get Device/sw version Request'", 40961: "'Get Device/sw version Confirmation'", 40968: "'Read MAC Memory Request'", 40969: "'Read MAC Memory Confirmation'", 40972: "'Start MAC Request'", 40973: "'Start MAC Confirmation'", 40976: "'Get NVM Parameters Request'", 40977: "'Get NVM Parameters Confirmation'", 40988: "'Reset Device Request'", 40989: "'Reset Device Confirmation'", 40992: "'Write Module Data Request'", 40996: "'Read Module Data Request'", 40997: "'Read Module Data Confirmation'", 41000: "'Write Module Data to NVM Request'", 41001: "'Write Module Data to NVM Confirmation'", 41012: "'Sniffer Request'", 41013: "'Sniffer Confirmation'", 41014: "'Sniffer Indicates'", 41016: "'Network Information Request'", 41017: "'Network Information Confirmation'", 41032: "'Loopback Request'", 41033: "'Loopback Request Confirmation'", 41040: "'Set Encryption Key Request'", 41041: "'Set Encryption Key Request Confirmation'", 41048: "'Read Configuration Block Request'", 41049: "'Read Configuration Block Confirmation'", 41058: "'Embedded Host Action Required Indication'"}
HPAVversionList = {0: '1.0', 1: '1.1'}
HPAVDeviceIDList = {0: 'Unknown', 1: "'INT6000'", 2: "'INT6300'", 3: "'INT6400'", 4: "'AR7400'", 5: "'AR6405'", 32: "'QCA7450/QCA7420'", 33: "'QCA6410/QCA6411'", 34: "'QCA7000'"}
StationRole = {0: "'Station'", 1: "'Proxy coordinator'", 2: "'Central coordinator'"}
StatusCodes = {0: "'Success'", 16: "'Invalid Address'", 20: "'Invalid Length'"}
DefaultVendor = 'Qualcomm'
QualcommTypeList = {40964: 'VS_WR_MEM', 40980: 'VS_RSVD_1', 40984: 'VS_RSVD_2', 41004: 'VS_WD_RPT', 41008: 'VS_LNK_STATS', 41020: 'VS_RSVD_3', 41024: 'VS_CP_RPT', 41028: 'VS_ARPC', 41044: 'VS_MFG_STRING', 41052: 'VS_SET_SDRAM', 41056: 'VS_HOST_ACTION', 41064: 'VS_OP_ATTRIBUTES', 41068: 'VS_ENET_SETTINGS', 41072: 'VS_TONE_MAP_CHAR', 41076: 'VS_NW_INFO_STATS', 41080: 'VS_SLAVE_MEM', 41084: 'VS_FAC_DEFAULTS', 41085: 'VS_FAC_DEFAULTS_CONFIRM', 41092: 'VS_MULTICAST_INFO', 41096: 'VS_CLASSIFICATION', 41104: 'VS_RX_TONE_MAP_CHAR', 41108: 'VS_SET_LED_BEHAVIOR', 41112: 'VS_WRITE_AND_EXECUTE_APPLET', 41116: 'VS_MDIO_COMMAND', 41120: 'VS_SLAVE_REG', 41124: 'VS_BANDWIDTH_LIMITING', 41128: 'VS_SNID_OPERATION', 41132: 'VS_NN_MITIGATE', 41136: 'VS_MODULE_OPERATION', 41140: 'VS_DIAG_NETWORK_PROBE', 41144: 'VS_PL_LINK_STATUS', 41148: 'VS_GPIO_STATE_CHANGE', 41152: 'VS_CONN_ADD', 41156: 'VS_CONN_MOD', 41160: 'VS_CONN_REL', 41164: 'VS_CONN_INFO', 41168: 'VS_MULTIPORT_LNK_STA', 41180: 'VS_EM_ID_TABLE', 41184: 'VS_STANDBY', 41188: 'VS_SLEEPSCHEDULE', 41192: 'VS_SLEEPSCHEDULE_NOTIFICATION', 41200: 'VS_MICROCONTROLLER_DIAG', 41208: 'VS_GET_PROPERTY', 41216: 'VS_SET_PROPERTY', 41220: 'VS_PHYSWITCH_MDIO', 41228: 'VS_SELFTEST_ONETIME_CONFIG', 41232: 'VS_SELFTEST_RESULTS', 41236: 'VS_MDU_TRAFFIC_STATS', 41240: 'VS_FORWARD_CONFIG', 41472: 'VS_HYBRID_INFO'}
EofPadList = [40960, 41016]

def FragmentCond(pkt):
    if False:
        while True:
            i = 10
    '\n        A fragmentation field condition\n        TODO: To complete\n    '
    return pkt.version == 1

class MACManagementHeader(Packet):
    name = 'MACManagementHeader '
    if DefaultVendor == 'Qualcomm':
        HPAVTypeList.update(QualcommTypeList)
    fields_desc = [ByteEnumField('version', 0, HPAVversionList), EnumField('HPtype', 40960, HPAVTypeList, '<H')]

class VendorMME(Packet):
    name = 'VendorMME '
    fields_desc = [OUIField('OUI', 45138)]

class GetDeviceVersion(Packet):
    name = 'GetDeviceVersion'
    fields_desc = [ByteEnumField('Status', 0, StatusCodes), ByteEnumField('DeviceID', 32, HPAVDeviceIDList), FieldLenField('VersionLen', None, count_of='DeviceVersion', fmt='B'), StrLenField('DeviceVersion', b'NoVersion\x00', length_from=lambda pkt: pkt.VersionLen), StrLenField('DeviceVersion_pad', b'\xcc\xcc\xcc\xcc\xcc' + b'\x00' * 59, length_from=lambda pkt: 64 - pkt.VersionLen), ByteEnumField('Upgradable', 0, {0: 'False', 1: 'True'})]

class NetworkInformationRequest(Packet):
    name = 'NetworkInformationRequest'
    fields_desc = []

class NetworkInfoV10(Packet):
    """
        Network Information Element
    """
    name = 'NetworkInfo'
    fields_desc = [StrFixedLenField('NetworkID', b'\x00\x00\x00\x00\x00\x00\x00', 7), XByteField('ShortNetworkID', 0), XByteField('TerminalEID', 1), ByteEnumField('StationRole', 0, StationRole), MACField('CCoMACAdress', '00:00:00:00:00:00'), XByteField('CCoTerminalEID', 1)]

    def extract_padding(self, p):
        if False:
            print('Hello World!')
        return (b'', p)

class StationInfoV10(Packet):
    """
        Station Information Element
    """
    name = 'StationInfo'
    fields_desc = [MACField('StationMAC', '00:00:00:00:00:00'), XByteField('StationTerminalEID', 1), MACField('firstnodeMAC', 'ff:ff:ff:ff:ff:ff'), XByteField('TXaverage', 0), XByteField('RXaverage', 0)]

    def extract_padding(self, p):
        if False:
            for i in range(10):
                print('nop')
        return (b'', p)

class NetworkInfoV11(Packet):
    """
        Network Information Element
    """
    name = 'NetworkInfo'
    fields_desc = [StrFixedLenField('NetworkID', b'\x00\x00\x00\x00\x00\x00\x00', 7), ShortField('reserved_1', 0), XByteField('ShortNetworkID', 0), XByteField('TerminalEID', 1), IntField('reserved_2', 0), ByteEnumField('StationRole', 0, StationRole), MACField('CCoMACAdress', '00:00:00:00:00:00'), XByteField('CCoTerminalEID', 1), X3BytesField('reserved_3', 0)]

    def extract_padding(self, p):
        if False:
            print('Hello World!')
        return (b'', p)

class StationInfoV11(Packet):
    """
        Station Information Element
    """
    name = 'StationInfo'
    fields_desc = [MACField('StationMAC', '00:00:00:00:00:00'), XByteField('StationTerminalEID', 1), X3BytesField('reserved_s2', 0), MACField('firstnodeMAC', 'ff:ff:ff:ff:ff:ff'), LEShortField('TXaverage', 0), BitField('RxCoupling', 0, 4), BitField('TxCoupling', 0, 4), XByteField('reserved_s3', 0), LEShortField('RXaverage', 0), XByteField('reserved_s4', 0)]

    def extract_padding(self, p):
        if False:
            print('Hello World!')
        return (b'', p)

class NetworkInfoConfirmationV10(Packet):
    """
        Network Information Confirmation following the MAC Management version 1.0  # noqa: E501
    """
    name = 'NetworkInfoConfirmation'
    fields_desc = [XByteField('LogicalNetworksNumber', 1), PacketListField('NetworksInfos', '', NetworkInfoV10, length_from=lambda pkt: pkt.LogicalNetworksNumber * 17), XByteField('StationsNumber', 1), PacketListField('StationsInfos', '', StationInfoV10, length_from=lambda pkt: pkt.StationsNumber * 21)]

class NetworkInfoConfirmationV11(Packet):
    """
        Network Information Confirmation following the MAC Management version 1.1  # noqa: E501
        This introduce few 'crazy' reserved bytes -> have fun!
    """
    name = 'NetworkInfoConfirmation'
    fields_desc = [StrFixedLenField('reserved_n1', b'\x00\x00:\x00\x00', 5), XByteField('LogicalNetworksNumber', 1), PacketListField('NetworksInfos', '', NetworkInfoV11, length_from=lambda pkt: pkt.LogicalNetworksNumber * 26), XByteField('StationsNumber', 1), StrFixedLenField('reserverd_s1', b'\x00\x00\x00\x00\x00', 5), PacketListField('StationsInfos', '', StationInfoV11, length_from=lambda pkt: pkt.StationsNumber * 23)]
ActionsList = {2: "'PIB Update Ready'", 4: "'Loader (Bootloader)'"}

class HostActionRequired(Packet):
    """
        Embedded Host Action Required Indice
    """
    name = 'HostActionRequired'
    fields_desc = [ByteEnumField('ActionRequired', 2, ActionsList)]

class LoopbackRequest(Packet):
    name = 'LoopbackRequest'
    fields_desc = [ByteField('Duration', 1), ByteField('reserved_l1', 1), ShortField('LRlength', 0)]

class LoopbackConfirmation(Packet):
    name = 'LoopbackConfirmation'
    fields_desc = [ByteEnumField('Status', 0, StatusCodes), ByteField('Duration', 1), ShortField('LRlength', 0)]

class SetEncryptionKeyRequest(Packet):
    name = 'SetEncryptionKeyRequest'
    fields_desc = [XByteField('EKS', 0), StrFixedLenField('NMK', b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', 16), XByteField('PayloadEncKeySelect', 0), MACField('DestinationMAC', 'ff:ff:ff:ff:ff:ff'), StrFixedLenField('DAK', b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', 16)]
SetEncKey_Status = {0: 'Success', 16: 'Invalid EKS', 17: 'Invalid PKS'}

class SetEncryptionKeyConfirmation(Packet):
    name = 'SetEncryptionKeyConfirmation'
    fields_desc = [ByteEnumField('Status', 0, SetEncKey_Status)]

class QUAResetFactoryConfirm(Packet):
    name = 'QUAResetFactoryConfirm'
    fields_desc = [ByteEnumField('Status', 0, StatusCodes)]

class GetNVMParametersRequest(Packet):
    name = 'Get NVM Parameters Request'
    fields_desc = []

class GetNVMParametersConfirmation(Packet):
    name = 'Get NVM Parameters Confirmation'
    fields_desc = [ByteEnumField('Status', 0, StatusCodes), LEIntField('NVMType', 19), LEIntField('NVMPageSize', 256), LEIntField('NVMBlockSize', 65536), LEIntField('NVMMemorySize', 1048576)]
SnifferControlList = {0: "'Disabled'", 1: "'Enabled'"}
SnifferTypeCodes = {0: "'Regular'"}

class SnifferRequest(Packet):
    name = 'SnifferRequest'
    fields_desc = [ByteEnumField('SnifferControl', 0, SnifferControlList)]
SnifferCodes = {0: "'Success'", 16: "'Invalid Control'"}

class SnifferConfirmation(Packet):
    name = 'SnifferConfirmation'
    fields_desc = [ByteEnumField('Status', 0, StatusCodes)]
DirectionCodes = {0: "'Tx'", 1: "'Rx'"}
ANCodes = {0: "'In-home'", 1: "'Access'"}

class SnifferIndicate(Packet):
    name = 'SnifferIndicate'
    fields_desc = [ByteEnumField('SnifferType', 0, SnifferTypeCodes), ByteEnumField('Direction', 0, DirectionCodes), LELongField('SystemTime', 0), LEIntField('BeaconTime', 0), XByteField('ShortNetworkID', 0), ByteField('SourceTermEqID', 0), ByteField('DestTermEqID', 0), ByteField('LinkID', 0), XByteField('PayloadEncrKeySelect', 15), ByteField('PendingPHYblock', 0), ByteField('BitLoadingEstim', 0), BitField('ToneMapIndex', 0, size=5), BitField('NumberofSymbols', 0, size=2), BitField('PHYblockSize', 0, size=1), XShortField('FrameLength', 0), XByteField('ReversegrandLength', 0), BitField('RequestSACKtrans', 0, size=1), BitField('DataMACstreamCMD', 0, size=3), BitField('ManNACFrameStreamCMD', 0, size=3), BitField('reserved_1', 0, size=6), BitField('MultinetBroadcast', 0, size=1), BitField('DifferentCPPHYclock', 0, size=1), BitField('Multicast', 0, size=1), X3BytesField('FrameControlCheckSeq', 0), XByteField('ShortNetworkID_', 0), IntField('BeaconTimestamp', 0), XShortField('BeaconTransOffset_0', 0), XShortField('BeaconTransOffset_1', 0), XShortField('BeaconTransOffset_2', 0), XShortField('BeaconTransOffset_3', 0), X3BytesField('FrameContrchkSeq', 0)]

class ReadMACMemoryRequest(Packet):
    name = 'ReadMACMemoryRequest'
    fields_desc = [LEIntField('Address', 0), LEIntField('Length', 1024)]
ReadMACStatus = {0: 'Success', 16: 'Invalid Address', 20: 'Invalid Length'}

class ReadMACMemoryConfirmation(Packet):
    name = 'ReadMACMemoryConfirmation'
    fields_desc = [ByteEnumField('Status', 0, ReadMACStatus), LEIntField('Address', 0), FieldLenField('MACLen', None, length_of='MACData', fmt='<H'), StrLenField('MACData', b'\x00', length_from=lambda pkt: pkt.MACLen)]
OperationList = {0: 'Read', 17: 'Write'}

class ModuleOperationRequest(Packet):
    name = 'ModuleOperationRequest'
    fields_desc = [XIntField('reserved', 0), XByteField('NumOpData', 1), LEShortEnumField('operation', 0, OperationList), LEShortField('OPDataLength', None), XIntField('reserved_1', 0), ConditionalField(LEIntField('SessionID', 0), lambda pkt: 17 == pkt.operation), ConditionalField(XByteField('ModuleIDX', 0), lambda pkt: 17 == pkt.operation), LEShortField('ModuleID', 28674), LEShortField('ModuleSubID', 0), ConditionalField(LEShortField('ReadDataLen', 1400), lambda pkt: 0 == pkt.operation), ConditionalField(LEIntField('ReadOffset', 0), lambda pkt: 0 == pkt.operation), ConditionalField(FieldLenField('WriteDataLen', None, count_of='ModuleData', fmt='<H'), lambda pkt: 17 == pkt.operation), ConditionalField(LEIntField('WriteOffset', 0), lambda pkt: 17 == pkt.operation), ConditionalField(StrLenField('ModuleData', b'\x00', length_from=lambda pkt: pkt.WriteDataLen), lambda pkt: 17 == pkt.operation)]

    def post_build(self, p, pay):
        if False:
            for i in range(10):
                print('nop')
        if self.operation == 0:
            if self.OPDataLength is None:
                _len = 18
                p = p[:7] + struct.pack('!H', _len) + p[9:]
        if self.operation == 17:
            if self.OPDataLength is None:
                _len = 23 + len(self.ModuleData)
                p = p[:7] + struct.pack('!H', _len) + p[9:]
            if self.WriteDataLen is None:
                _len = len(self.ModuleData)
                p = p[:22] + struct.pack('!H', _len) + p[24:]
        return p + pay

class ModuleOperationConfirmation(Packet):
    name = 'ModuleOperationConfirmation'
    fields_desc = [LEShortField('Status', 0), LEShortField('ErrorCode', 0), XIntField('reserved', 0), XByteField('NumOpData', 1), LEShortEnumField('operation', 0, OperationList), LEShortField('OPDataLength', 18), XIntField('reserved_1', 0), ConditionalField(LEIntField('SessionID', 0), lambda pkt: 17 == pkt.operation), ConditionalField(XByteField('ModuleIDX', 0), lambda pkt: 17 == pkt.operation), LEShortField('ModuleID', 28674), LEShortField('ModuleSubID', 0), ConditionalField(FieldLenField('ReadDataLen', None, count_of='ModuleData', fmt='<H'), lambda pkt: 0 == pkt.operation), ConditionalField(LEIntField('ReadOffset', 0), lambda pkt: 0 == pkt.operation), ConditionalField(StrLenField('ModuleData', b'\x00', length_from=lambda pkt: pkt.ReadDataLen), lambda pkt: 0 == pkt.operation), ConditionalField(LEShortField('WriteDataLen', 0), lambda pkt: 17 == pkt.operation), ConditionalField(LEIntField('WriteOffset', 0), lambda pkt: 17 == pkt.operation)]

    def post_build(self, p, pay):
        if False:
            i = 10
            return i + 15
        if self.operation == 17:
            if self.OPDataLength is None:
                _len = 18 + len(self.ModuleData)
                p = p[:7] + struct.pack('h', _len) + p[9:]
        if self.operation == 0:
            if self.OPDataLength is None:
                _len = 23 + len(self.ModuleData)
                p = p[:7] + struct.pack('h', _len) + p[9:]
            if self.WriteDataLen is None:
                _len = len(self.ModuleData)
                p = p[:17] + struct.pack('h', _len) + p[19:]
        return p + pay
ModuleIDList = {0: 'MAC Soft-Loader Image', 1: 'MAC Software Image', 2: 'PIB', 16: 'Write Alternate Flash Location'}

def chksum32(data):
    if False:
        print('Hello World!')
    cksum = 0
    for i in range(0, len(data), 4):
        cksum = (cksum ^ struct.unpack('<I', data[i:i + 4])[0]) & 4294967295
    return ~cksum & 4294967295

class ReadModuleDataRequest(Packet):
    name = 'ReadModuleDataRequest'
    fields_desc = [ByteEnumField('ModuleID', 2, ModuleIDList), XByteField('reserved', 0), LEShortField('Length', 1024), LEIntField('Offset', 0)]

class ReadModuleDataConfirmation(Packet):
    name = 'ReadModuleDataConfirmation'
    fields_desc = [ByteEnumField('Status', 0, StatusCodes), X3BytesField('reserved_1', 0), ByteEnumField('ModuleID', 2, ModuleIDList), XByteField('reserved_2', 0), FieldLenField('DataLen', None, count_of='ModuleData', fmt='<H'), LEIntField('Offset', 0), LEIntField('checksum', None), StrLenField('ModuleData', b'\x00', length_from=lambda pkt: pkt.DataLen)]

    def post_build(self, p, pay):
        if False:
            while True:
                i = 10
        if self.DataLen is None:
            _len = len(self.ModuleData)
            p = p[:6] + struct.pack('h', _len) + p[8:]
        if self.checksum is None and p:
            ck = chksum32(self.ModuleData)
            p = p[:12] + struct.pack('I', ck) + p[16:]
        return p + pay

class WriteModuleDataRequest(Packet):
    name = 'WriteModuleDataRequest'
    fields_desc = [ByteEnumField('ModuleID', 2, ModuleIDList), XByteField('reserved_1', 0), FieldLenField('DataLen', None, count_of='ModuleData', fmt='<H'), LEIntField('Offset', 0), LEIntField('checksum', None), StrLenField('ModuleData', b'\x00', length_from=lambda pkt: pkt.DataLen)]

    def post_build(self, p, pay):
        if False:
            for i in range(10):
                print('nop')
        if self.DataLen is None:
            _len = len(self.ModuleData)
            p = p[:2] + struct.pack('<H', _len) + p[4:]
        if self.checksum is None and p:
            ck = chksum32(self.ModuleData)
            p = p[:8] + struct.pack('<I', ck) + p[12:]
        return p + pay

class ClassifierPriorityMap(Packet):
    name = 'ClassifierPriorityMap'
    fields_desc = [LEIntField('Priority', 0), LEIntField('PID', 0), LEIntField('IndividualOperand', 0), StrFixedLenField('ClassifierValue', b'\x00' * 16, 16)]

    def extract_padding(self, p):
        if False:
            return 10
        return (b'', p)

class ClassifierObj(Packet):
    name = 'ClassifierObj'
    fields_desc = [LEIntField('ClassifierPID', 0), LEIntField('IndividualOperand', 0), StrFixedLenField('ClassifierValue', b'\x00' * 16, 16)]

    def extract_padding(self, p):
        if False:
            return 10
        return (b'', p)

class AutoConnection(Packet):
    name = 'AutoConnection'
    fields_desc = [XByteField('Action', 0), XByteField('ClassificationOperand', 0), XShortField('NumClassifiers', 0), PacketListField('ClassifierObjs', '', ClassifierObj, length_from=lambda x: 24), XShortField('CSPECversion', 0), XByteField('ConnCAP', 0), XByteField('ConnCoQoSPrio', 0), ShortField('ConnRate', 0), LEIntField('ConnTTL', 0), ShortField('version', 0), StrFixedLenField('VlanTag', b'\x00' * 4, 4), XIntField('reserved_1', 0), StrFixedLenField('reserved_2', b'\x00' * 14, 14)]

    def extract_padding(self, p):
        if False:
            while True:
                i = 10
        return (b'', p)

class PeerNode(Packet):
    name = 'PeerNodes'
    fields_desc = [XByteField('PeerTEI', 0), MACField('PIBMACAddr', '00:00:00:00:00:00')]

    def extract_padding(self, p):
        if False:
            print('Hello World!')
        return (b'', p)

class AggregateConfigEntrie(Packet):
    name = 'AggregateConfigEntrie'
    fields_desc = [XByteField('TrafficTypeID', 0), XByteField('AggregationConfigID', 0)]

    def extract_padding(self, p):
        if False:
            for i in range(10):
                print('nop')
        return (b'', p)

class RSVD_CustomAggregationParameter(Packet):
    name = 'RSVD_CustomAggregationParameter'
    fields_desc = [XIntField('CustomAggregationParameter', 0)]

    def extract_padding(self, p):
        if False:
            return 10
        return (b'', p)

class PrescalerValue(Packet):
    name = 'PrescalerValue'
    fields_desc = [XIntField('prescaler', 0)]

    def extract_padding(self, p):
        if False:
            for i in range(10):
                print('nop')
        return (b'', p)

class GPIOMap(Packet):
    name = 'GPIOMap'
    fields_desc = [XByteField('GPIOvalue', 0)]

    def extract_padding(self, p):
        if False:
            while True:
                i = 10
        return (b'', p)

class ReservedPercentageForCap(Packet):
    name = 'ReservedPercentageForCap'
    fields_desc = [XByteField('CAPpercent', 0)]

    def extract_padding(self, p):
        if False:
            return 10
        return (b'', p)

class ConfigBit(Packet):
    name = 'ConfigBit'
    fields_desc = [BitField('OverrideSoundCap', 0, 1), BitField('OverrideFailHoldDefaults', 0, 1), BitField('OverrideResourceDefaults', 0, 1), BitField('OverrideContentionWindowDefaults', 0, 1), BitField('OverrideUnplugDetectionDefaults', 0, 1), BitField('OverrideResoundDefaults', 0, 1), BitField('OverrideExpiryDefaults', 0, 1), BitField('DisableWorseChannelTrigger', 0, 1), BitField('DisableBetterChannelTrigger', 0, 1), BitField('DisableNetworkEventTrigger', 0, 1), BitField('rsv1', 0, 6)]

class ContentionWindowTable(Packet):
    name = 'ContentionWindowTable'
    fields_desc = [XShortField('element', 0)]

    def extract_padding(self, p):
        if False:
            while True:
                i = 10
        return (b'', p)

class BackoffDeferalCountTable(Packet):
    name = 'BackoffDeferalCountTable'
    fields_desc = [XByteField('element', 0)]

    def extract_padding(self, p):
        if False:
            print('Hello World!')
        return (b'', p)

class BehaviorBlockArray(Packet):
    name = 'BehaviorBlockArray'
    fields_desc = [XByteField('BehId', 0), XByteField('NoOfSteps', 0), XByteField('DurationInMs', 0), XShortField('GPIOMaskBits_1', 0), XShortField('GPIOMaskBits_2', 0), XShortField('GPIOMaskBits_3', 0), XShortField('GPIOMaskBits_4', 0), XShortField('GPIOMaskBits_5', 0), XShortField('GPIOMaskBits_6', 0), XIntField('reserved_beh', 0)]

    def extract_padding(self, p):
        if False:
            for i in range(10):
                print('nop')
        return (b'', p)

class EventBlockArray(Packet):
    name = 'EventBlockArray'
    fields_desc = [XByteField('EventPriorityID', 0), XByteField('EventID', 0), XByteField('BehID_1', 0), XByteField('BehID_2', 0), XByteField('BehID_3', 0), XShortField('ParticipatingGPIOs', 0), XByteField('EventAttributes', 0), XShortField('reserved_evb', 0)]

    def extract_padding(self, p):
        if False:
            i = 10
            return i + 15
        return (b'', p)

class ModulePIB(Packet):
    """
        Simple Module PIB Decoder.
            /!/ A wrong slice would produce 'bad' results
    """
    name = 'ModulePIB'
    __slots__ = ['_ModulePIB__offset', '_ModulePIB__length']
    fields_desc = [ConditionalField(XByteField('FirmwareMajorVersion', 0), lambda pkt: 0 == pkt.__offset and 1 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('PIBMinorVersion', 0), lambda pkt: 1 >= pkt.__offset and 2 <= pkt.__offset + pkt.__length), ConditionalField(XShortField('reserved_1', 0), lambda pkt: 2 >= pkt.__offset and 4 <= pkt.__offset + pkt.__length), ConditionalField(XShortField('PIBLength', 0), lambda pkt: 4 >= pkt.__offset and 6 <= pkt.__offset + pkt.__length), ConditionalField(XShortField('reserved_2', 0), lambda pkt: 6 >= pkt.__offset and 8 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('checksumPIB', None), lambda pkt: 8 >= pkt.__offset and 12 <= pkt.__offset + pkt.__length), ConditionalField(MACField('PIBMACAddr', '00:00:00:00:00:00'), lambda pkt: 12 >= pkt.__offset and 18 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('DAK', b'\x00' * 16, 16), lambda pkt: 18 >= pkt.__offset and 34 <= pkt.__offset + pkt.__length), ConditionalField(XShortField('reserved_3', 0), lambda pkt: 34 >= pkt.__offset and 36 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('ManufactorID', b'\x00' * 64, 64), lambda pkt: 36 >= pkt.__offset and 100 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('NMK', b'\x00' * 16, 16), lambda pkt: 100 >= pkt.__offset and 116 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('UserID', b'\x00' * 64, 64), lambda pkt: 116 >= pkt.__offset and 180 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('AVLN_ID', b'\x00' * 64, 64), lambda pkt: 180 >= pkt.__offset and 244 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('CCoSelection', 0), lambda pkt: 244 >= pkt.__offset and 245 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('CoExistSelection', 0), lambda pkt: 245 >= pkt.__offset and 246 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('PLFreqSelection', 0), lambda pkt: 246 >= pkt.__offset and 247 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('H3CDowngradeShld', 0), lambda pkt: 247 >= pkt.__offset and 248 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('PreferredNID', b'\x00' * 7, 7), lambda pkt: 248 >= pkt.__offset and 255 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('AutoFWUpgradeable', 0), lambda pkt: 255 >= pkt.__offset and 256 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('MDUConfiguration', 0), lambda pkt: 256 >= pkt.__offset and 257 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('MDURole', 0), lambda pkt: 257 >= pkt.__offset and 258 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('SnifferEnabled', 0), lambda pkt: 258 >= pkt.__offset and 259 <= pkt.__offset + pkt.__length), ConditionalField(MACField('SnifferMACAddrRetrn', '00:00:00:00:00:00'), lambda pkt: 259 >= pkt.__offset and 265 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('WireTapEnable', 0), lambda pkt: 265 >= pkt.__offset and 266 <= pkt.__offset + pkt.__length), ConditionalField(XShortField('reserved_4', 0), lambda pkt: 266 >= pkt.__offset and 268 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('StaticNetworkEnabled', 0), lambda pkt: 268 >= pkt.__offset and 269 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('LD_TEI', 0), lambda pkt: 269 >= pkt.__offset and 270 <= pkt.__offset + pkt.__length), ConditionalField(MACField('CCo_MACAdd', '00:00:00:00:00:00'), lambda pkt: 270 >= pkt.__offset and 276 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('SNID', 0), lambda pkt: 276 >= pkt.__offset and 277 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('NumOfPeerNodes', 0), lambda pkt: 277 >= pkt.__offset and 278 <= pkt.__offset + pkt.__length), ConditionalField(PacketListField('PeerNodes', '', PeerNode, length_from=lambda x: 56), lambda pkt: 278 >= pkt.__offset and 284 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('reserved_5', b'\x00' * 62, 62), lambda pkt: 326 >= pkt.__offset and 334 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('OverideModeDefaults', 0), lambda pkt: 396 >= pkt.__offset and 397 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('DisableFlowControl', 0), lambda pkt: 397 >= pkt.__offset and 398 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('AdvertisementCapabilities', 0), lambda pkt: 398 >= pkt.__offset and 399 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('OverrideMeteringDefaults', 0), lambda pkt: 399 >= pkt.__offset and 400 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('MaxFramesPerSec', 0), lambda pkt: 400 >= pkt.__offset and 404 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('DisableAutoNegotiation', 0), lambda pkt: 404 >= pkt.__offset and 405 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('EnetSpeedSetting', 0), lambda pkt: 405 >= pkt.__offset and 406 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('EnetDuplexSetting', 0), lambda pkt: 406 >= pkt.__offset and 407 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('DisableTxFlowControl', 0), lambda pkt: 407 >= pkt.__offset and 408 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('DisableRxFlowControl', 0), lambda pkt: 408 >= pkt.__offset and 409 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('PhyAddressSelection', 0), lambda pkt: 409 >= pkt.__offset and 410 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('PhyAddressSelection_Data', 0), lambda pkt: 410 >= pkt.__offset and 411 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('reserved_6', 0), lambda pkt: 411 >= pkt.__offset and 412 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('Force33MHz', 0), lambda pkt: 412 >= pkt.__offset and 413 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('LinkStatusOnPowerline', 0), lambda pkt: 413 >= pkt.__offset and 414 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('OverrideIdDefaults', 0), lambda pkt: 414 >= pkt.__offset and 415 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('OverrideSubIdDefaults', 0), lambda pkt: 415 >= pkt.__offset and 416 <= pkt.__offset + pkt.__length), ConditionalField(XShortField('PCIDeviceID', 0), lambda pkt: 416 >= pkt.__offset and 418 <= pkt.__offset + pkt.__length), ConditionalField(XShortField('PCIVendorID', 0), lambda pkt: 418 >= pkt.__offset and 420 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('reserved_7', 0), lambda pkt: 420 >= pkt.__offset and 421 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('PCIClassCode', 0), lambda pkt: 421 >= pkt.__offset and 422 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('PCIClassCodeSubClass', 0), lambda pkt: 422 >= pkt.__offset and 423 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('PCIRevisionID', 0), lambda pkt: 423 >= pkt.__offset and 424 <= pkt.__offset + pkt.__length), ConditionalField(XShortField('PCISubsystemID', 0), lambda pkt: 424 >= pkt.__offset and 426 <= pkt.__offset + pkt.__length), ConditionalField(XShortField('PCISybsystemVendorID', 0), lambda pkt: 426 >= pkt.__offset and 428 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('reserved_8', b'\x00' * 64, 64), lambda pkt: 428 >= pkt.__offset and 492 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('OverrideIGMPDefaults', 0), lambda pkt: 492 >= pkt.__offset and 493 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('ConfigFlags', 0), lambda pkt: 493 >= pkt.__offset and 494 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('NumCpToSend_PLFrames', 0), lambda pkt: 494 >= pkt.__offset and 495 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('reserved_9', b'\x00' * 29, 29), lambda pkt: 495 >= pkt.__offset and 524 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('UniCastPriority', 0), lambda pkt: 524 >= pkt.__offset and 525 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('McastPriority', 0), lambda pkt: 525 >= pkt.__offset and 526 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('IGMPPriority', 0), lambda pkt: 526 >= pkt.__offset and 527 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('AVStreamPriority', 0), lambda pkt: 527 >= pkt.__offset and 528 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('PriorityTTL_0', 0), lambda pkt: 528 >= pkt.__offset and 532 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('PriorityTTL_1', 0), lambda pkt: 532 >= pkt.__offset and 536 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('PriorityTTL_2', 0), lambda pkt: 536 >= pkt.__offset and 540 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('PriorityTTL_3', 0), lambda pkt: 540 >= pkt.__offset and 544 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('EnableVLANOver', 0), lambda pkt: 544 >= pkt.__offset and 545 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('EnableTOSOver', 0), lambda pkt: 545 >= pkt.__offset and 546 <= pkt.__offset + pkt.__length), ConditionalField(XShortField('reserved_10', 0), lambda pkt: 546 >= pkt.__offset and 548 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('VLANPrioTOSPrecMatrix', 0), lambda pkt: 548 >= pkt.__offset and 552 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('NumClassifierPriorityMaps', 0), lambda pkt: 552 >= pkt.__offset and 556 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('NumAutoConnections', 0), lambda pkt: 556 >= pkt.__offset and 560 <= pkt.__offset + pkt.__length), ConditionalField(PacketListField('ClassifierPriorityMaps', '', ClassifierPriorityMap, length_from=lambda x: 224), lambda pkt: 560 >= pkt.__offset and 580 <= pkt.__offset + pkt.__length), ConditionalField(PacketListField('AutoConnections', '', AutoConnection, length_from=lambda x: 1600), lambda pkt: 784 >= pkt.__offset and 878 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('NumberOfConfigEntries', 0), lambda pkt: 2384 >= pkt.__offset and 2385 <= pkt.__offset + pkt.__length), ConditionalField(PacketListField('AggregateConfigEntries', '', AggregateConfigEntrie, length_from=lambda x: 16), lambda pkt: 2385 >= pkt.__offset and 2401 <= pkt.__offset + pkt.__length), ConditionalField(PacketListField('RSVD_CustomAggregationParameters', '', RSVD_CustomAggregationParameter, length_from=lambda x: 48), lambda pkt: 2401 >= pkt.__offset and 2449 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('reserved_11', b'\x00' * 123, 123), lambda pkt: 2449 >= pkt.__offset and 2572 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('ToneMaskType', 0), lambda pkt: 2572 >= pkt.__offset and 2576 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('ToneMaskEnabled', 0), lambda pkt: 2576 >= pkt.__offset and 2580 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('StartTone', 0), lambda pkt: 2580 >= pkt.__offset and 2584 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('EndTone', 0), lambda pkt: 2584 >= pkt.__offset and 2588 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('reserved_12', b'\x00' * 12, 12), lambda pkt: 2588 >= pkt.__offset and 2600 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('PsdIndex', 0), lambda pkt: 2600 >= pkt.__offset and 2604 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('TxPrescalerType', 0), lambda pkt: 2604 >= pkt.__offset and 2608 <= pkt.__offset + pkt.__length), ConditionalField(PacketListField('PrescalerValues', '', PrescalerValue, length_from=lambda x: 3600), lambda pkt: 2608 >= pkt.__offset and 2612 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('reserved_13', b'\x00' * 1484, 1484), lambda pkt: 6208 >= pkt.__offset and 7692 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('AllowNEKRotation', 0), lambda pkt: 7692 >= pkt.__offset and 7696 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('OverrideLocalNEK', 0), lambda pkt: 7696 >= pkt.__offset and 7700 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('LocalNEKToUse', b'\x00' * 16, 16), lambda pkt: 7700 >= pkt.__offset and 7716 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('OverrideNEKRotationTimer', 0), lambda pkt: 7716 >= pkt.__offset and 7720 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('NEKRotationTime_Min', 0), lambda pkt: 7720 >= pkt.__offset and 7724 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('reserved_14', b'\x00' * 96, 96), lambda pkt: 7724 >= pkt.__offset and 7820 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('AVLNMembership', 0), lambda pkt: 7820 >= pkt.__offset and 7824 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('SimpleConnectTimeout', 0), lambda pkt: 7824 >= pkt.__offset and 7828 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('EnableLEDThroughputIndicate', 0), lambda pkt: 7828 >= pkt.__offset and 7829 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('MidLEDThroughputThreshold_Mbps', 0), lambda pkt: 7829 >= pkt.__offset and 7830 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('HighLEDThroughputThreshold_Mbps', 0), lambda pkt: 7830 >= pkt.__offset and 7831 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('reserved_15', 0), lambda pkt: 7831 >= pkt.__offset and 7832 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('EnableUnicastQuieriesToMember', 0), lambda pkt: 7832 >= pkt.__offset and 7833 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('DisableMLDGroupIDCheckInMAC', 0), lambda pkt: 7833 >= pkt.__offset and 7834 <= pkt.__offset + pkt.__length), ConditionalField(XShortField('EnableReportsToNonQuerierHosts', 0), lambda pkt: 7834 >= pkt.__offset and 7836 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('DisableExpireGroupMembershipInterval', 0), lambda pkt: 7836 >= pkt.__offset and 7840 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('DisableLEDTestLights', 0), lambda pkt: 7840 >= pkt.__offset and 7844 <= pkt.__offset + pkt.__length), ConditionalField(PacketListField('GPIOMaps', '', GPIOMap, length_from=lambda x: 12), lambda pkt: 7844 >= pkt.__offset and 7856 <= pkt.__offset + pkt.__length), ConditionalField(XLongField('reserved_16', 0), lambda pkt: 7856 >= pkt.__offset and 7864 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('EnableTrafficClass_DSCPOver', 0), lambda pkt: 7864 >= pkt.__offset and 7865 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('TrafficClass_DSCPMatrices', b'\x00' * 64, 64), lambda pkt: 7865 >= pkt.__offset and 7929 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('GPIOControl', 0), lambda pkt: 7929 >= pkt.__offset and 7930 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('LEDControl', b'\x00' * 32, 32), lambda pkt: 7930 >= pkt.__offset and 7962 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('OverrideMinButtonPressHoldTime', 0), lambda pkt: 7962 >= pkt.__offset and 7966 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('MinButtonPressHoldTime', 0), lambda pkt: 7966 >= pkt.__offset and 7970 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('reserved_17', b'\x00' * 22, 22), lambda pkt: 7970 >= pkt.__offset and 7992 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('MemoryProfile', 0), lambda pkt: 7992 >= pkt.__offset and 7996 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('DisableAllLEDFlashOnWarmReboot', 0), lambda pkt: 7996 >= pkt.__offset and 8000 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('UplinkLimit_bps', 0), lambda pkt: 8000 >= pkt.__offset and 8004 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('DownlinkLimit_bps', 0), lambda pkt: 8004 >= pkt.__offset and 8008 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('MDUStaticSNID', 0), lambda pkt: 8008 >= pkt.__offset and 8012 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('MitigateEnabled', 0), lambda pkt: 8012 >= pkt.__offset and 8013 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('CorrelThreshold', 0), lambda pkt: 8013 >= pkt.__offset and 8017 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('ScaledTxGain', 0), lambda pkt: 8017 >= pkt.__offset and 8021 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('ResourceThresholdEnabled', 0), lambda pkt: 8021 >= pkt.__offset and 8022 <= pkt.__offset + pkt.__length), ConditionalField(PacketListField('ReservedPercentageForCaps', '', ReservedPercentageForCap, length_from=lambda x: 4), lambda pkt: 8022 >= pkt.__offset and 8026 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('PowerSavingMode', 0), lambda pkt: 8026 >= pkt.__offset and 8027 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('PowerLEDDutyCycle', 0), lambda pkt: 8027 >= pkt.__offset and 8028 <= pkt.__offset + pkt.__length), ConditionalField(XShortField('reserved_18', 0), lambda pkt: 8028 >= pkt.__offset and 8030 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('LinkUpDurationBeforeReset_ms', 0), lambda pkt: 8030 >= pkt.__offset and 8034 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('PowerLEDPeriod_ms', 0), lambda pkt: 8034 >= pkt.__offset and 8038 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('LinkDownDurationBeforeLowPowerMode_ms', 0), lambda pkt: 8038 >= pkt.__offset and 8042 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('reserved_19', 0), lambda pkt: 8042 >= pkt.__offset and 8046 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('AfeGainBusMode', 0), lambda pkt: 8046 >= pkt.__offset and 8047 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('EnableDynamicPsd', 0), lambda pkt: 8047 >= pkt.__offset and 8048 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('ReservedPercentageForTxStreams', 0), lambda pkt: 8048 >= pkt.__offset and 8049 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('ReservedPercentageForRxStreams', 0), lambda pkt: 8049 >= pkt.__offset and 8050 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('reserved_20', b'\x00' * 22, 22), lambda pkt: 8050 >= pkt.__offset and 8072 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('LegacyNetworkUpgradeEnable', 0), lambda pkt: 8072 >= pkt.__offset and 8076 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('unknown', 0), lambda pkt: 8076 >= pkt.__offset and 8080 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('MMETTL_us', 0), lambda pkt: 8080 >= pkt.__offset and 8084 <= pkt.__offset + pkt.__length), ConditionalField(PacketListField('ConfigBits', '', ConfigBit, length_from=lambda x: 2), lambda pkt: 8084 >= pkt.__offset and 8086 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('TxToneMapExpiry_ms', 0), lambda pkt: 8086 >= pkt.__offset and 8090 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('RxToneMapExpiry_ms', 0), lambda pkt: 8090 >= pkt.__offset and 8094 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('TimeoutToResound_ms', 0), lambda pkt: 8094 >= pkt.__offset and 8098 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('MissingSackThresholdForUnplugDetection', 0), lambda pkt: 8098 >= pkt.__offset and 8102 <= pkt.__offset + pkt.__length), ConditionalField(LEIntField('UnplugTimeout_ms', 0), lambda pkt: 8102 >= pkt.__offset and 8106 <= pkt.__offset + pkt.__length), ConditionalField(PacketListField('ContentionWindowTableES', '', ContentionWindowTable, length_from=lambda x: 8), lambda pkt: 8106 >= pkt.__offset and 8114 <= pkt.__offset + pkt.__length), ConditionalField(PacketListField('BackoffDeferalCountTableES', '', BackoffDeferalCountTable, length_from=lambda x: 4), lambda pkt: 8114 >= pkt.__offset and 8118 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('GoodSoundCountThreshold', 0), lambda pkt: 8118 >= pkt.__offset and 8119 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('SoundCountThreshold_GoodSoundCountPass', 0), lambda pkt: 8119 >= pkt.__offset and 8120 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('SoundCountThreshold_GoodSoundCountFail', 0), lambda pkt: 8120 >= pkt.__offset and 8121 <= pkt.__offset + pkt.__length), ConditionalField(XShortField('reserved_21', 0), lambda pkt: 8121 >= pkt.__offset and 8123 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('ExclusiveTxPbs_percentage', 0), lambda pkt: 8123 >= pkt.__offset and 8124 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('ExclusiveRxPbs_percentage', 0), lambda pkt: 8124 >= pkt.__offset and 8125 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('OptimizationBackwardCompatible', 0), lambda pkt: 8125 >= pkt.__offset and 8126 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('reserved_21b', 0), lambda pkt: 8126 >= pkt.__offset and 8127 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('MaxPbsPerSymbol', 0), lambda pkt: 8127 >= pkt.__offset and 8128 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('MaxModulation', 0), lambda pkt: 8128 >= pkt.__offset and 8129 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('ContinuousRx', 0), lambda pkt: 8129 >= pkt.__offset and 8130 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('reserved_22', b'\x00' * 6, 6), lambda pkt: 8130 >= pkt.__offset and 8136 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('PBControlStatus', 0), lambda pkt: 8136 >= pkt.__offset and 8137 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('STAMembershipMaskEnabled', 0), lambda pkt: 8137 >= pkt.__offset and 8138 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('ExitDefaultEnabled', 0), lambda pkt: 8138 >= pkt.__offset and 8139 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('RejectDefaultEnabled', 0), lambda pkt: 8139 >= pkt.__offset and 8140 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('ChainingEnabled', 0), lambda pkt: 8140 >= pkt.__offset and 8141 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('VendorSpecificNMK', b'\x00' * 16, 16), lambda pkt: 8141 >= pkt.__offset and 8157 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('LocalMACAddressLimit', 0), lambda pkt: 8157 >= pkt.__offset and 8158 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('OverrideBridgeTableAgingTime', 0), lambda pkt: 8158 >= pkt.__offset and 8159 <= pkt.__offset + pkt.__length), ConditionalField(XShortField('LocalBridgeTableAgingTime_min', 0), lambda pkt: 8159 >= pkt.__offset and 8161 <= pkt.__offset + pkt.__length), ConditionalField(XShortField('RemoteBridgeTableAgingTime_min', 0), lambda pkt: 8161 >= pkt.__offset and 8163 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('PhySyncReference', 0), lambda pkt: 8163 >= pkt.__offset and 8167 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('reserved_23', 0), lambda pkt: 8167 >= pkt.__offset and 8168 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('reserved_24', 0), lambda pkt: 8168 >= pkt.__offset and 8172 <= pkt.__offset + pkt.__length), ConditionalField(XIntField('reserved_25', 0), lambda pkt: 8172 >= pkt.__offset and 8176 <= pkt.__offset + pkt.__length), ConditionalField(StrFixedLenField('reserved_26', b'\x00' * 24, 24), lambda pkt: 8176 >= pkt.__offset and 8200 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('OverrideDefaultLedEventBehavior', 128), lambda pkt: 8200 >= pkt.__offset and 8201 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('ReportToHostInfo', 0), lambda pkt: 8201 >= pkt.__offset and 8202 <= pkt.__offset + pkt.__length), ConditionalField(X3BytesField('reserved_27', 0), lambda pkt: 8202 >= pkt.__offset and 8205 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('NumBehaviors', 0), lambda pkt: 8205 >= pkt.__offset and 8206 <= pkt.__offset + pkt.__length), ConditionalField(PacketListField('BehaviorBlockArrayES', '', BehaviorBlockArray, length_from=lambda x: 1200), lambda pkt: 8206 >= pkt.__offset and 9406 <= pkt.__offset + pkt.__length), ConditionalField(XByteField('NumEvents', 0), lambda pkt: 9406 >= pkt.__offset and 9407 <= pkt.__offset + pkt.__length), ConditionalField(PacketListField('EventBlockArrayES', '', EventBlockArray, length_from=lambda x: 550), lambda pkt: 9407 >= pkt.__offset and 9957 <= pkt.__offset + pkt.__length)]

    def __init__(self, packet='', offset=0, length=1024):
        if False:
            while True:
                i = 10
        self.__offset = offset
        self.__length = length
        return super(ModulePIB, self).__init__(packet)
StartMACCodes = {0: 'Success'}

class StartMACRequest(Packet):
    name = 'StartMACRequest'
    fields_desc = [ByteEnumField('ModuleID', 0, StartMACCodes), X3BytesField('reserver_1', 0), LEIntField('ImgLoadStartAddr', 0), LEIntField('ImgLength', 0), LEIntField('ImgCheckSum', 0), LEIntField('ImgStartAddr', 0)]

class StartMACConfirmation(Packet):
    name = 'StartMACConfirmation'
    fields_desc = [ByteEnumField('Status', 0, StartMACCodes), XByteField('ModuleID', 0)]
ResetDeviceCodes = {0: 'Success'}

class ResetDeviceRequest(Packet):
    name = 'ResetDeviceRequest'
    fields_desc = []

class ResetDeviceConfirmation(Packet):
    name = 'ResetDeviceConfirmation'
    fields_desc = [ByteEnumField('Status', 0, ResetDeviceCodes)]
ReadConfBlockCodes = {0: 'Success'}

class ReadConfBlockRequest(Packet):
    name = 'ReadConfBlockRequest'
    fields_desc = []
CBImgTCodes = {0: 'Generic Image', 1: 'Synopsis configuration', 2: 'Denali configuration', 3: 'Denali applet', 4: 'Runtime firmware', 5: 'OAS client', 6: 'Custom image', 7: 'Memory control applet', 8: 'Power management applet', 9: 'OAS client IP stack', 10: 'OAS client TR069', 11: 'SoftLoader', 12: 'Flash layout', 13: 'Unknown', 14: 'Chain manifest', 15: 'Runtime parameters', 16: 'Custom module in scratch', 17: 'Custom module update applet'}

class ConfBlock(Packet):
    name = 'ConfBlock'
    fields_desc = [LEIntField('HeaderVersionNum', 0), LEIntField('ImgAddrNVM', 0), LEIntField('ImgAddrSDRAM', 0), LEIntField('ImgLength', 0), LEIntField('ImgCheckSum', 0), LEIntField('EntryPoint', 0), XByteField('HeaderMinVersion', 0), ByteEnumField('HeaderImgType', 0, CBImgTCodes), XShortField('HeaderIgnoreMask', 0), LEIntField('HeaderModuleID', 0), LEIntField('HeaderModuleSubID', 0), LEIntField('AddrNextHeaderNVM', 0), LEIntField('HeaderChecksum', 0), LEIntField('SDRAMsize', 0), LEIntField('SDRAMConfRegister', 0), LEIntField('SDRAMTimingRegister_0', 0), LEIntField('SDRAMTimingRegister_1', 0), LEIntField('SDRAMControlRegister', 0), LEIntField('SDRAMRefreshRegister', 0), LEIntField('MACClockRegister', 0), LEIntField('reserved_1', 0)]

class ReadConfBlockConfirmation(Packet):
    name = 'ReadConfBlockConfirmation'
    fields_desc = [ByteEnumField('Status', 0, ReadConfBlockCodes), FieldLenField('BlockLen', None, count_of='ConfigurationBlock', fmt='B'), PacketListField('ConfigurationBlock', None, ConfBlock, length_from=lambda pkt: pkt.BlockLen)]

class WriteModuleData2NVMRequest(Packet):
    name = 'WriteModuleData2NVMRequest'
    fields_desc = [ByteEnumField('ModuleID', 2, ModuleIDList)]

class WriteModuleData2NVMConfirmation(Packet):
    name = 'WriteModuleData2NVMConfirmation'
    fields_desc = [ByteEnumField('Status', 0, StatusCodes), ByteEnumField('ModuleID', 2, ModuleIDList)]

class HomePlugAV(Packet):
    """
        HomePlugAV Packet - by default => gets devices information
    """
    name = 'HomePlugAV '
    fields_desc = [MACManagementHeader, ConditionalField(XShortField('FragmentInfo', 0), FragmentCond), ConditionalField(PacketListField('VendorField', VendorMME(), VendorMME, length_from=lambda x: 3), lambda pkt: pkt.version == 0)]

    def answers(self, other):
        if False:
            while True:
                i = 10
        return isinstance(self, HomePlugAV)
bind_layers(Ether, HomePlugAV, {'type': 35041})
bind_layers(HomePlugAV, GetDeviceVersion, HPtype=40961)
bind_layers(HomePlugAV, StartMACRequest, HPtype=40972)
bind_layers(HomePlugAV, StartMACConfirmation, HPtype=40973)
bind_layers(HomePlugAV, ResetDeviceRequest, HPtype=40988)
bind_layers(HomePlugAV, ResetDeviceConfirmation, HPtype=40989)
bind_layers(HomePlugAV, NetworkInformationRequest, HPtype=41016)
bind_layers(HomePlugAV, ReadMACMemoryRequest, HPtype=40968)
bind_layers(HomePlugAV, ReadMACMemoryConfirmation, HPtype=40969)
bind_layers(HomePlugAV, ReadModuleDataRequest, HPtype=40996)
bind_layers(HomePlugAV, ReadModuleDataConfirmation, HPtype=40997)
bind_layers(HomePlugAV, ModuleOperationRequest, HPtype=41136)
bind_layers(HomePlugAV, ModuleOperationConfirmation, HPtype=41137)
bind_layers(HomePlugAV, WriteModuleDataRequest, HPtype=40992)
bind_layers(HomePlugAV, WriteModuleData2NVMRequest, HPtype=41000)
bind_layers(HomePlugAV, WriteModuleData2NVMConfirmation, HPtype=41001)
bind_layers(HomePlugAV, NetworkInfoConfirmationV10, HPtype=41017, version=0)
bind_layers(HomePlugAV, NetworkInfoConfirmationV11, HPtype=41017, version=1)
bind_layers(NetworkInfoConfirmationV10, NetworkInfoV10, HPtype=41017, version=0)
bind_layers(NetworkInfoConfirmationV11, NetworkInfoV11, HPtype=41017, version=1)
bind_layers(HomePlugAV, HostActionRequired, HPtype=41058)
bind_layers(HomePlugAV, LoopbackRequest, HPtype=41032)
bind_layers(HomePlugAV, LoopbackConfirmation, HPtype=41033)
bind_layers(HomePlugAV, SetEncryptionKeyRequest, HPtype=41040)
bind_layers(HomePlugAV, SetEncryptionKeyConfirmation, HPtype=41041)
bind_layers(HomePlugAV, ReadConfBlockRequest, HPtype=41048)
bind_layers(HomePlugAV, ReadConfBlockConfirmation, HPtype=41049)
bind_layers(HomePlugAV, QUAResetFactoryConfirm, HPtype=41085)
bind_layers(HomePlugAV, GetNVMParametersRequest, HPtype=40976)
bind_layers(HomePlugAV, GetNVMParametersConfirmation, HPtype=40977)
bind_layers(HomePlugAV, SnifferRequest, HPtype=41012)
bind_layers(HomePlugAV, SnifferConfirmation, HPtype=41013)
bind_layers(HomePlugAV, SnifferIndicate, HPtype=41014)
'\n    Credit song : "Western Spaguetti - We are terrorists"\n'