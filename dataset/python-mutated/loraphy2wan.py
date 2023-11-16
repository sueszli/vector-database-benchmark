"""
LoRa PHY to WAN Layer

Initially developed @PentHertz
and improved at @Trend Micro

Spec: lorawantm_specification v1.1
"""
from scapy.packet import Packet
from scapy.fields import BitEnumField, BitField, BitFieldLenField, ByteEnumField, ByteField, ConditionalField, IntField, LEShortField, MayEnd, MultipleTypeField, PacketField, PacketListField, StrField, StrFixedLenField, X3BytesField, XBitField, XByteField, XIntField, XLE3BytesField, XLEIntField, XShortField

class FCtrl_DownLink(Packet):
    name = 'FCtrl_DownLink'
    fields_desc = [BitField('ADR', 0, 1), BitField('ADRACKReq', 0, 1), BitField('ACK', 0, 1), BitField('FPending', 0, 1), BitFieldLenField('FOptsLen', 0, 4)]

    def extract_padding(self, p):
        if False:
            i = 10
            return i + 15
        return ('', p)

class FCtrl_Link(Packet):
    name = 'FCtrl_UpLink'
    fields_desc = [BitField('ADR', 0, 1), BitField('ADRACKReq', 0, 1), BitField('ACK', 0, 1), BitField('UpClassB_DownFPending', 0, 1), BitFieldLenField('FOptsLen', 0, 4)]

    def extract_padding(self, p):
        if False:
            i = 10
            return i + 15
        return ('', p)

class FCtrl_UpLink(Packet):
    name = 'FCtrl_UpLink'
    fields_desc = [BitField('ADR', 0, 1), BitField('ADRACKReq', 0, 1), BitField('ACK', 0, 1), BitField('ClassB', 0, 1), BitFieldLenField('FOptsLen', 0, 4)]

    def extract_padding(self, p):
        if False:
            i = 10
            return i + 15
        return ('', p)

class DevAddrElem(Packet):
    name = 'DevAddrElem'
    fields_desc = [XByteField('NwkID', 0), XLE3BytesField('NwkAddr', b'\x00' * 3)]
CIDs_up = {1: 'ResetInd', 2: 'LinkCheckReq', 3: 'LinkADRReq', 4: 'DutyCycleReq', 5: 'RXParamSetupReq', 6: 'DevStatusReq', 7: 'NewChannelReq', 8: 'RXTimingSetupReq', 9: 'TxParamSetupReq', 10: 'DlChannelReq', 11: 'RekeyInd', 12: 'ADRParamSetupReq', 13: 'DeviceTimeReq', 14: 'ForceRejoinReq', 15: 'RejoinParamSetupReq'}
CIDs_down = {1: 'ResetConf', 2: 'LinkCheckAns', 3: 'LinkADRAns', 4: 'DutyCycleAns', 5: 'RXParamSetupAns', 6: 'DevStatusAns', 7: 'NewChannelAns', 8: 'RXTimingSetupAns', 9: 'TxParamSetupAns', 10: 'DlChannelAns', 11: 'RekeyConf', 12: 'ADRParamSetupAns', 13: 'DeviceTimeAns', 15: 'RejoinParamSetupAns'}

class ResetInd(Packet):
    name = 'ResetInd'
    fields_desc = [ByteField('Dev_version', 0)]

class ResetConf(Packet):
    name = 'ResetConf'
    fields_desc = [ByteField('Serv_version', 0)]

class LinkCheckReq(Packet):
    name = 'LinkCheckReq'

class LinkCheckAns(Packet):
    name = 'LinkCheckAns'
    fields_desc = [ByteField('Margin', 0), ByteField('GwCnt', 0)]

class DataRate_TXPower(Packet):
    name = 'DataRate_TXPower'
    fields_desc = [XBitField('DataRate', 0, 4), XBitField('TXPower', 0, 4)]

class Redundancy(Packet):
    name = 'Redundancy'
    fields_desc = [XBitField('RFU', 0, 1), XBitField('ChMaskCntl', 0, 3), XBitField('NbTrans', 0, 4)]

class LinkADRReq(Packet):
    name = 'LinkADRReq'
    fields_desc = [DataRate_TXPower, XShortField('ChMask', 0), Redundancy]

class LinkADRAns_Status(Packet):
    name = 'LinkADRAns_Status'
    fields_desc = [BitField('RFU', 0, 5), BitField('PowerACK', 0, 1), BitField('DataRate', 0, 1), BitField('ChannelMaskACK', 0, 1)]

class LinkADRAns(Packet):
    name = 'LinkADRAns'
    fields_desc = [PacketField('status', LinkADRAns_Status(), LinkADRAns_Status)]

class DutyCyclePL(Packet):
    name = 'DutyCyclePL'
    fields_desc = [BitField('MaxDCycle', 0, 4)]

class DutyCycleReq(Packet):
    name = 'DutyCycleReq'
    fields_desc = [DutyCyclePL]

class DutyCycleAns(Packet):
    name = 'DutyCycleAns'
    fields_desc = []

class DLsettings(Packet):
    name = 'DLsettings'
    fields_desc = [BitField('OptNeg', 0, 1), XBitField('RX1DRoffset', 0, 3), XBitField('RX2_Data_rate', 0, 4)]

class RXParamSetupReq(Packet):
    name = 'RXParamSetupReq'
    fields_desc = [DLsettings, X3BytesField('Frequency', 0)]

class RXParamSetupAns_Status(Packet):
    name = 'RXParamSetupAns_Status'
    fields_desc = [XBitField('RFU', 0, 5), BitField('RX1DRoffsetACK', 0, 1), BitField('RX2DatarateACK', 0, 1), BitField('ChannelACK', 0, 1)]

class RXParamSetupAns(Packet):
    name = 'RXParamSetupAns'
    fields_desc = [RXParamSetupAns_Status]
Battery_state = {0: 'End-device connected to external source', 255: 'Battery level unknown'}

class DevStatusReq(Packet):
    name = 'DevStatusReq'
    fields_desc = [ByteEnumField('Battery', 0, Battery_state), ByteField('Margin', 0)]

class DevStatusAns_Status(Packet):
    name = 'DevStatusAns_Status'
    fields_desc = [XBitField('RFU', 0, 2), XBitField('Margin', 0, 6)]

class DevStatusAns(Packet):
    name = 'DevStatusAns'
    fields_desc = [DevStatusAns_Status]

class DrRange(Packet):
    name = 'DrRange'
    fields_desc = [XBitField('MaxDR', 0, 4), XBitField('MinDR', 0, 4)]

class NewChannelReq(Packet):
    name = 'NewChannelReq'
    fields_desc = [ByteField('ChIndex', 0), X3BytesField('Freq', 0), DrRange]

class NewChannelAns_Status(Packet):
    name = 'NewChannelAns_Status'
    fields_desc = [XBitField('RFU', 0, 6), BitField('Dataraterangeok', 0, 1), BitField('Channelfrequencyok', 0, 1)]

class NewChannelAns(Packet):
    name = 'NewChannelAns'
    fields_desc = [NewChannelAns_Status]

class RXTimingSetupReq_Settings(Packet):
    name = 'RXTimingSetupReq_Settings'
    fields_desc = [XBitField('RFU', 0, 4), XBitField('Del', 0, 4)]

class RXTimingSetupReq(Packet):
    name = 'RXTimingSetupReq'
    fields_desc = [RXTimingSetupReq_Settings]

class RXTimingSetupAns(Packet):
    name = 'RXTimingSetupAns'
    fields_desc = []
MaxEIRPs = {0: '8 dbm', 1: '10 dbm', 2: '12 dbm', 3: '13 dbm', 4: '14 dbm', 5: '16 dbm', 6: '18 dbm', 7: '20 dbm', 8: '21 dbm', 9: '24 dbm', 10: '26 dbm', 11: '27 dbm', 12: '29 dbm', 13: '30 dbm', 14: '33 dbm', 15: '36 dbm'}
DwellTimes = {0: 'No limit', 1: '400 ms'}

class EIRP_DwellTime(Packet):
    name = 'EIRP_DwellTime'
    fields_desc = [BitField('RFU', 0, 2), BitEnumField('DownlinkDwellTime', 0, 1, DwellTimes), BitEnumField('UplinkDwellTime', 0, 1, DwellTimes), BitEnumField('MaxEIRP', 0, 4, MaxEIRPs)]

class TxParamSetupReq(Packet):
    name = 'TxParamSetupReq'
    fields_desc = [EIRP_DwellTime]

class TxParamSetupAns(Packet):
    name = 'TxParamSetupAns'
    fields_desc = []

class DlChannelReq(Packet):
    name = 'DlChannelReq'
    fields_desc = [ByteField('ChIndex', 0), X3BytesField('Freq', 0)]

class DlChannelAns(Packet):
    name = 'DlChannelAns'
    fields_desc = [ByteField('Status', 0)]

class DevLoraWANversion(Packet):
    name = 'DevLoraWANversion'
    fields_desc = [BitField('RFU', 0, 4), BitField('Minor', 1, 4)]

class RekeyInd(Packet):
    name = 'RekeyInd'
    fields_desc = [PacketListField('LoRaWANversion', b'', DevLoraWANversion, length_from=lambda pkt: 1)]

class RekeyConf(Packet):
    name = 'RekeyConf'
    fields_desc = [ByteField('ServerVersion', 0)]

class ADRparam(Packet):
    name = 'ADRparam'
    fields_desc = [BitField('Limit_exp', 0, 4), BitField('Delay_exp', 0, 4)]

class ADRParamSetupReq(Packet):
    name = 'ADRParamSetupReq'
    fields_desc = [ADRparam]

class ADRParamSetupAns(Packet):
    name = 'ADRParamSetupReq'
    fields_desc = []

class DeviceTimeReq(Packet):
    name = 'DeviceTimeReq'
    fields_desc = []

class DeviceTimeAns(Packet):
    name = 'DeviceTimeAns'
    fields_desc = [IntField('SecondsSinceEpoch', 0), ByteField('FracSecond', 0)]

class ForceRejoinReq(Packet):
    name = 'ForceRejoinReq'
    fields_desc = [BitField('RFU', 0, 2), BitField('Period', 0, 3), BitField('Max_Retries', 0, 3), BitField('RFU2', 0, 1), BitField('RejoinType', 0, 3), BitField('DR', 0, 4)]

class RejoinParamSetupReq(Packet):
    name = 'RejoinParamSetupReq'
    fields_desc = [BitField('MaxTimeN', 0, 4), BitField('MaxCountN', 0, 4)]

class RejoinParamSetupAns(Packet):
    name = 'RejoinParamSetupAns'
    fields_desc = [BitField('RFU', 0, 7), BitField('TimeOK', 0, 1)]

class MACCommand_up(Packet):
    name = 'MACCommand_up'
    fields_desc = [ByteEnumField('CID', 0, CIDs_up), ConditionalField(PacketListField('Reset', b'', ResetInd, length_from=lambda pkt: 1), lambda pkt: pkt.CID == 1), ConditionalField(PacketListField('LinkCheck', b'', LinkCheckReq, length_from=lambda pkt: 0), lambda pkt: pkt.CID == 2), ConditionalField(PacketListField('LinkADR', b'', LinkADRReq, length_from=lambda pkt: 4), lambda pkt: pkt.CID == 3), ConditionalField(PacketListField('DutyCycle', b'', DutyCycleReq, length_from=lambda pkt: 4), lambda pkt: pkt.CID == 4), ConditionalField(PacketListField('RXParamSetup', b'', RXParamSetupReq, length_from=lambda pkt: 4), lambda pkt: pkt.CID == 5), ConditionalField(PacketListField('DevStatus', b'', DevStatusReq, length_from=lambda pkt: 2), lambda pkt: pkt.CID == 6), ConditionalField(PacketListField('NewChannel', b'', NewChannelReq, length_from=lambda pkt: 5), lambda pkt: pkt.CID == 7), ConditionalField(PacketListField('RXTimingSetup', b'', RXTimingSetupReq, length_from=lambda pkt: 1), lambda pkt: pkt.CID == 8), ConditionalField(PacketListField('TxParamSetup', b'', TxParamSetupReq, length_from=lambda pkt: 1), lambda pkt: pkt.CID == 9), ConditionalField(PacketListField('DlChannel', b'', DlChannelReq, length_from=lambda pkt: 4), lambda pkt: pkt.CID == 10), ConditionalField(PacketListField('Rekey', b'', RekeyInd, length_from=lambda pkt: 1), lambda pkt: pkt.CID == 11), ConditionalField(PacketListField('ADRParamSetup', b'', ADRParamSetupReq, length_from=lambda pkt: 1), lambda pkt: pkt.CID == 12), ConditionalField(PacketListField('DeviceTime', b'', DeviceTimeReq, length_from=lambda pkt: 0), lambda pkt: pkt.CID == 13), ConditionalField(PacketListField('ForceRejoin', b'', ForceRejoinReq, length_from=lambda pkt: 2), lambda pkt: pkt.CID == 14), ConditionalField(PacketListField('RejoinParamSetup', b'', RejoinParamSetupReq, length_from=lambda pkt: 1), lambda pkt: pkt.CID == 15)]

    def extract_padding(self, p):
        if False:
            i = 10
            return i + 15
        return ('', p)

class MACCommand_down(Packet):
    name = 'MACCommand_down'
    fields_desc = [ByteEnumField('CID', 0, CIDs_up), ConditionalField(PacketListField('Reset', b'', ResetConf, length_from=lambda pkt: 1), lambda pkt: pkt.CID == 1), ConditionalField(PacketListField('LinkCheck', b'', LinkCheckAns, length_from=lambda pkt: 2), lambda pkt: pkt.CID == 2), ConditionalField(PacketListField('LinkADR', b'', LinkADRAns, length_from=lambda pkt: 0), lambda pkt: pkt.CID == 3), ConditionalField(PacketListField('DutyCycle', b'', DutyCycleAns, length_from=lambda pkt: 4), lambda pkt: pkt.CID == 4), ConditionalField(PacketListField('RXParamSetup', b'', RXParamSetupAns, length_from=lambda pkt: 1), lambda pkt: pkt.CID == 5), ConditionalField(PacketListField('DevStatusAns', b'', RXParamSetupAns, length_from=lambda pkt: 1), lambda pkt: pkt.CID == 6), ConditionalField(PacketListField('NewChannel', b'', NewChannelAns, length_from=lambda pkt: 1), lambda pkt: pkt.CID == 7), ConditionalField(PacketListField('RXTimingSetup', b'', RXTimingSetupAns, length_from=lambda pkt: 0), lambda pkt: pkt.CID == 8), ConditionalField(PacketListField('TxParamSetup', b'', TxParamSetupAns, length_from=lambda pkt: 0), lambda pkt: pkt.CID == 9), ConditionalField(PacketListField('DlChannel', b'', DlChannelAns, length_from=lambda pkt: 1), lambda pkt: pkt.CID == 10), ConditionalField(PacketListField('Rekey', b'', RekeyConf, length_from=lambda pkt: 1), lambda pkt: pkt.CID == 11), ConditionalField(PacketListField('ADRParamSetup', b'', ADRParamSetupAns, length_from=lambda pkt: 0), lambda pkt: pkt.CID == 12), ConditionalField(PacketListField('DeviceTime', b'', DeviceTimeAns, length_from=lambda pkt: 5), lambda pkt: pkt.CID == 13), ConditionalField(PacketListField('RejoinParamSetup', b'', RejoinParamSetupAns, length_from=lambda pkt: 1), lambda pkt: pkt.CID == 15)]

class FOpts(Packet):
    name = 'FOpts'
    fields_desc = [ConditionalField(PacketListField('FOpts_up', b'', MACCommand_up, length_from=lambda pkt: pkt.FCtrl[0].FOptsLen), lambda pkt: pkt.FCtrl[0].FOptsLen > 0 and pkt.MType & 1 == 0 and (pkt.MType >= 2)), ConditionalField(PacketListField('FOpts_down', b'', MACCommand_down, length_from=lambda pkt: pkt.FCtrl[0].FOptsLen), lambda pkt: pkt.FCtrl[0].FOptsLen > 0 and pkt.MType & 1 == 1 and (pkt.MType <= 5))]

def FOptsDownShow(pkt):
    if False:
        return 10
    try:
        if pkt.FCtrl[0].FOptsLen > 0 and pkt.MType & 1 == 1 and (pkt.MType <= 5) and (pkt.MType & 5 > 0):
            return True
        return False
    except Exception:
        return False

def FOptsUpShow(pkt):
    if False:
        for i in range(10):
            print('nop')
    try:
        if pkt.FCtrl[0].FOptsLen > 0 and pkt.MType & 1 == 0 and (pkt.MType >= 2) and (pkt.MType & 6 > 0):
            return True
        return False
    except Exception:
        return False

class FHDR(Packet):
    name = 'FHDR'
    fields_desc = [ConditionalField(PacketListField('DevAddr', b'', DevAddrElem, length_from=lambda pkt: 4), lambda pkt: pkt.MType >= 2 and pkt.MType <= 5), ConditionalField(PacketListField('FCtrl', b'', FCtrl_Link, length_from=lambda pkt: 1), lambda pkt: pkt.MType & 1 == 1 and pkt.MType <= 5 and (pkt.MType & 2 > 0) or (pkt.MType & 1 == 0 and pkt.MType >= 2)), ConditionalField(LEShortField('FCnt', 0), lambda pkt: pkt.MType >= 2 and pkt.MType <= 5), ConditionalField(PacketListField('FOpts_up', b'', MACCommand_up, length_from=lambda pkt: pkt.FCtrl[0].FOptsLen), FOptsUpShow), ConditionalField(PacketListField('FOpts_down', b'', MACCommand_down, length_from=lambda pkt: pkt.FCtrl[0].FOptsLen), FOptsDownShow)]
FPorts = {0: 'NwkSKey'}
JoinReqTypes = {255: 'Join-request', 0: 'Rejoin-request type 0', 1: 'Rejoin-request type 1', 2: 'Rejoin-request type 2'}

class Join_Request(Packet):
    name = 'Join_Request'
    fields_desc = [StrFixedLenField('AppEUI', b'\x00' * 8, 8), StrFixedLenField('DevEUI', b'\x00' * 8, 8), LEShortField('DevNonce', 0)]

class Join_Accept(Packet):
    name = 'Join_Accept'
    dcflist = False
    fields_desc = [XLE3BytesField('JoinAppNonce', 0), XLE3BytesField('NetID', 0), XLEIntField('DevAddr', 0), DLsettings, XByteField('RxDelay', 0), ConditionalField(StrFixedLenField('CFList', b'\x00' * 16, 16), lambda pkt: Join_Accept.dcflist is True)]

    def extract_padding(self, p):
        if False:
            i = 10
            return i + 15
        return ('', p)

    def __init__(self, packet=''):
        if False:
            i = 10
            return i + 15
        if len(packet) > 18:
            Join_Accept.dcflist = True
        super(Join_Accept, self).__init__(packet)
RejoinType = {0: 'NetID+DevEUI', 1: 'JoinEUI+DevEUI', 2: 'NetID+DevEUI'}

class RejoinReq(Packet):
    name = 'RejoinReq'
    fields_desc = [ByteField('Type', 0), X3BytesField('NetID', 0), StrFixedLenField('DevEUI', b'\x00' * 8), XShortField('RJcount0', 0)]

def dpload_type(pkt):
    if False:
        while True:
            i = 10
    if pkt.MType == 5 or pkt.MType == 3:
        return 0
    elif pkt.MType == 4 or pkt.MType == 2:
        return 1
    return None
datapayload_list = [(StrField('DataPayload', '', remain=4), lambda pkt: dpload_type(pkt) == 0), (StrField('DataPayload', '', remain=6), lambda pkt: dpload_type(pkt) == 1)]

class FRMPayload(Packet):
    name = 'FRMPayload'
    fields_desc = [ConditionalField(MultipleTypeField(datapayload_list, StrField('DataPayload', '', remain=4)), lambda pkt: dpload_type(pkt) is not None), ConditionalField(PacketListField('Join_Request_Field', b'', Join_Request, length_from=lambda pkt: 18), lambda pkt: pkt.MType == 0), ConditionalField(PacketListField('Join_Accept_Field', b'', Join_Accept, count_from=lambda pkt: 1), lambda pkt: pkt.MType == 1 and LoRa.encrypted is False), ConditionalField(StrField('Join_Accept_Encrypted', 0), lambda pkt: pkt.MType == 1 and LoRa.encrypted is True), ConditionalField(PacketListField('ReJoin_Request_Field', b'', RejoinReq, length_from=lambda pkt: 14), lambda pkt: pkt.MType == 7)]

class MACPayload(Packet):
    name = 'MACPayload'
    eFPort = False
    fields_desc = [FHDR, ConditionalField(ByteEnumField('FPort', 0, FPorts), lambda pkt: pkt.MType >= 2 and pkt.MType <= 5 and (pkt.FCtrl[0].FOptsLen == 0)), FRMPayload]
MTypes = {0: 'Join-request', 1: 'Join-accept', 2: 'Unconfirmed Data Up', 3: 'Unconfirmed Data Down', 4: 'Confirmed Data Up', 5: 'Confirmed Data Down', 6: 'Rejoin-request', 7: 'Proprietary'}

class MHDR(Packet):
    name = 'MHDR'
    fields_desc = [BitEnumField('MType', 0, 3, MTypes), BitField('RFU', 0, 3), BitField('Major', 0, 2)]

class PHYPayload(Packet):
    name = 'PHYPayload'
    fields_desc = [MHDR, MACPayload, MayEnd(ConditionalField(XIntField('MIC', 0), lambda pkt: pkt.MType != 1 or LoRa.encrypted is False))]

class LoRa(Packet):
    name = 'LoRa'
    version = '1.1'
    encrypted = True
    fields_desc = [XBitField('Preamble', 0, 4), XBitField('PHDR', 0, 16), XBitField('PHDR_CRC', 0, 4), PHYPayload, ConditionalField(XShortField('CRC', 0), lambda pkt: pkt.MType & 1 == 0)]