import struct
from scapy.fields import BitField, ByteEnumField, ByteField, ConditionalField, MayEnd, ObservableDict, StrField, X3BytesField, XByteEnumField, XByteField, XShortEnumField
from scapy.packet import Packet, bind_layers, NoPayload
from scapy.config import conf
from scapy.error import log_loading
from scapy.utils import PeriodicSenderThread
from scapy.plist import _PacketIterable
from scapy.contrib.isotp import ISOTP
from typing import Dict, Any
try:
    if conf.contribs['KWP']['treat-response-pending-as-answer']:
        pass
except KeyError:
    log_loading.info('Specify "conf.contribs[\'KWP\'] = {\'treat-response-pending-as-answer\': True}" to treat a negative response \'requestCorrectlyReceived-ResponsePending\' as answer of a request. \nThe default value is False.')
    conf.contribs['KWP'] = {'treat-response-pending-as-answer': False}

class KWP(ISOTP):
    services = ObservableDict({16: 'StartDiagnosticSession', 17: 'ECUReset', 20: 'ClearDiagnosticInformation', 23: 'ReadStatusOfDiagnosticTroubleCodes', 24: 'ReadDiagnosticTroubleCodesByStatus', 26: 'ReadECUIdentification', 33: 'ReadDataByLocalIdentifier', 34: 'ReadDataByIdentifier', 35: 'ReadMemoryByAddress', 39: 'SecurityAccess', 40: 'DisableNormalMessageTransmission', 41: 'EnableNormalMessageTransmission', 44: 'DynamicallyDefineLocalIdentifier', 46: 'WriteDataByIdentifier', 48: 'InputOutputControlByLocalIdentifier', 49: 'StartRoutineByLocalIdentifier', 50: 'StopRoutineByLocalIdentifier', 51: 'RequestRoutineResultsByLocalIdentifier', 52: 'RequestDownload', 53: 'RequestUpload', 54: 'TransferData', 55: 'RequestTransferExit', 59: 'WriteDataByLocalIdentifier', 61: 'WriteMemoryByAddress', 62: 'TesterPresent', 133: 'ControlDTCSetting', 134: 'ResponseOnEvent', 80: 'StartDiagnosticSessionPositiveResponse', 81: 'ECUResetPositiveResponse', 84: 'ClearDiagnosticInformationPositiveResponse', 87: 'ReadStatusOfDiagnosticTroubleCodesPositiveResponse', 88: 'ReadDiagnosticTroubleCodesByStatusPositiveResponse', 90: 'ReadECUIdentificationPositiveResponse', 97: 'ReadDataByLocalIdentifierPositiveResponse', 98: 'ReadDataByIdentifierPositiveResponse', 99: 'ReadMemoryByAddressPositiveResponse', 103: 'SecurityAccessPositiveResponse', 104: 'DisableNormalMessageTransmissionPositiveResponse', 105: 'EnableNormalMessageTransmissionPositiveResponse', 108: 'DynamicallyDefineLocalIdentifierPositiveResponse', 110: 'WriteDataByIdentifierPositiveResponse', 112: 'InputOutputControlByLocalIdentifierPositiveResponse', 113: 'StartRoutineByLocalIdentifierPositiveResponse', 114: 'StopRoutineByLocalIdentifierPositiveResponse', 115: 'RequestRoutineResultsByLocalIdentifierPositiveResponse', 116: 'RequestDownloadPositiveResponse', 117: 'RequestUploadPositiveResponse', 118: 'TransferDataPositiveResponse', 119: 'RequestTransferExitPositiveResponse', 123: 'WriteDataByLocalIdentifierPositiveResponse', 125: 'WriteMemoryByAddressPositiveResponse', 126: 'TesterPresentPositiveResponse', 197: 'ControlDTCSettingPositiveResponse', 198: 'ResponseOnEventPositiveResponse', 127: 'NegativeResponse'})
    name = 'KWP'
    fields_desc = [XByteEnumField('service', 0, services)]

    def answers(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, type(self)):
            return False
        if self.service == 127:
            return self.payload.answers(other)
        if self.service == other.service + 64:
            if isinstance(self.payload, NoPayload) or isinstance(other.payload, NoPayload):
                return len(self) <= len(other)
            else:
                return self.payload.answers(other.payload)
        return False

    def hashret(self):
        if False:
            for i in range(10):
                print('nop')
        if self.service == 127:
            return struct.pack('B', self.requestServiceId & ~64)
        else:
            return struct.pack('B', self.service & ~64)

class KWP_SDS(Packet):
    diagnosticSessionTypes = ObservableDict({129: 'defaultSession', 133: 'programmingSession', 137: 'standBySession', 144: 'EcuPassiveSession', 146: 'extendedDiagnosticSession'})
    name = 'StartDiagnosticSession'
    fields_desc = [ByteEnumField('diagnosticSession', 0, diagnosticSessionTypes)]
bind_layers(KWP, KWP_SDS, service=16)

class KWP_SDSPR(Packet):
    name = 'StartDiagnosticSessionPositiveResponse'
    fields_desc = [ByteEnumField('diagnosticSession', 0, KWP_SDS.diagnosticSessionTypes)]

    def answers(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, KWP_SDS) and other.diagnosticSession == self.diagnosticSession
bind_layers(KWP, KWP_SDSPR, service=80)

class KWP_ER(Packet):
    resetModes = {0: 'reserved', 1: 'powerOnReset', 130: 'nonvolatileMemoryReset'}
    name = 'ECUReset'
    fields_desc = [ByteEnumField('resetMode', 0, resetModes)]
bind_layers(KWP, KWP_ER, service=17)

class KWP_ERPR(Packet):
    name = 'ECUResetPositiveResponse'

    def answers(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, KWP_ER)
bind_layers(KWP, KWP_ERPR, service=81)

class KWP_SA(Packet):
    name = 'SecurityAccess'
    fields_desc = [ByteField('accessMode', 0), ConditionalField(StrField('key', b''), lambda pkt: pkt.accessMode % 2 == 0)]
bind_layers(KWP, KWP_SA, service=39)

class KWP_SAPR(Packet):
    name = 'SecurityAccessPositiveResponse'
    fields_desc = [ByteField('accessMode', 0), ConditionalField(StrField('seed', b''), lambda pkt: pkt.accessMode % 2 == 1)]

    def answers(self, other):
        if False:
            return 10
        return isinstance(other, KWP_SA) and other.accessMode == self.accessMode
bind_layers(KWP, KWP_SAPR, service=103)

class KWP_IOCBLI(Packet):
    name = 'InputOutputControlByLocalIdentifier'
    inputOutputControlParameters = {0: 'Return Control to ECU', 1: 'Report Current State', 4: 'Reset to Default', 5: 'Freeze Current State', 7: 'Short Term Adjustment', 8: 'Long Term Adjustment'}
    fields_desc = [XByteField('localIdentifier', 0), XByteEnumField('inputOutputControlParameter', 0, inputOutputControlParameters), StrField('controlState', b'', fmt='B')]
bind_layers(KWP, KWP_IOCBLI, service=48)

class KWP_IOCBLIPR(Packet):
    name = 'InputOutputControlByLocalIdentifierPositiveResponse'
    fields_desc = [XByteField('localIdentifier', 0), XByteEnumField('inputOutputControlParameter', 0, KWP_IOCBLI.inputOutputControlParameters), StrField('controlState', b'', fmt='B')]

    def answers(self, other):
        if False:
            return 10
        return isinstance(other, KWP_IOCBLI) and other.localIdentifier == self.localIdentifier
bind_layers(KWP, KWP_IOCBLIPR, service=112)

class KWP_DNMT(Packet):
    responseTypes = {1: 'responseRequired', 2: 'noResponse'}
    name = 'DisableNormalMessageTransmission'
    fields_desc = [ByteEnumField('responseRequired', 0, responseTypes)]
bind_layers(KWP, KWP_DNMT, service=40)

class KWP_DNMTPR(Packet):
    name = 'DisableNormalMessageTransmissionPositiveResponse'

    def answers(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, KWP_DNMT)
bind_layers(KWP, KWP_DNMTPR, service=104)

class KWP_ENMT(Packet):
    responseTypes = {1: 'responseRequired', 2: 'noResponse'}
    name = 'EnableNormalMessageTransmission'
    fields_desc = [ByteEnumField('responseRequired', 1, responseTypes)]
bind_layers(KWP, KWP_ENMT, service=41)

class KWP_ENMTPR(Packet):
    name = 'EnableNormalMessageTransmissionPositiveResponse'

    def answers(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, KWP_DNMT)
bind_layers(KWP, KWP_ENMTPR, service=105)

class KWP_TP(Packet):
    responseTypes = {1: 'responseRequired', 2: 'noResponse'}
    name = 'TesterPresent'
    fields_desc = [ByteEnumField('responseRequired', 1, responseTypes)]
bind_layers(KWP, KWP_TP, service=62)

class KWP_TPPR(Packet):
    name = 'TesterPresentPositiveResponse'

    def answers(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, KWP_TP)
bind_layers(KWP, KWP_TPPR, service=126)

class KWP_CDTCS(Packet):
    responseTypes = {1: 'responseRequired', 2: 'noResponse'}
    DTCGroups = {0: 'allPowertrainDTCs', 16384: 'allChassisDTCs', 32768: 'allBodyDTCs', 49152: 'allNetworkDTCs', 65280: 'allDTCs'}
    DTCSettingModes = {0: 'Reserved', 1: 'on', 2: 'off'}
    name = 'ControlDTCSetting'
    fields_desc = [ByteEnumField('responseRequired', 1, responseTypes), XShortEnumField('groupOfDTC', 0, DTCGroups), ByteEnumField('DTCSettingMode', 0, DTCSettingModes)]
bind_layers(KWP, KWP_CDTCS, service=133)

class KWP_CDTCSPR(Packet):
    name = 'ControlDTCSettingPositiveResponse'

    def answers(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, KWP_CDTCS)
bind_layers(KWP, KWP_CDTCSPR, service=197)

class KWP_ROE(Packet):
    responseTypes = {1: 'responseRequired', 2: 'noResponse'}
    eventWindowTimes = {0: 'reserved', 1: 'testerPresentRequired', 2: 'infiniteTimeToResponse', 128: 'noEventWindow'}
    eventTypes = {128: 'reportActivatedEvents', 129: 'stopResponseOnEvent', 130: 'onNewDTC', 131: 'onTimerInterrupt', 132: 'onChangeOfRecordValue', 160: 'onComparisonOfValues'}
    name = 'ResponseOnEvent'
    fields_desc = [ByteEnumField('responseRequired', 1, responseTypes), ByteEnumField('eventWindowTime', 0, eventWindowTimes), MayEnd(ByteEnumField('eventType', 0, eventTypes)), ByteField('eventParameter', 0), ByteEnumField('serviceToRespond', 0, KWP.services), ByteField('serviceParameter', 0)]
bind_layers(KWP, KWP_ROE, service=134)

class KWP_ROEPR(Packet):
    name = 'ResponseOnEventPositiveResponse'
    fields_desc = [ByteField('numberOfActivatedEvents', 0), MayEnd(ByteEnumField('eventWindowTime', 0, KWP_ROE.eventWindowTimes)), ByteEnumField('eventType', 0, KWP_ROE.eventTypes)]

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, KWP_ROE) and other.eventType == self.eventType
bind_layers(KWP, KWP_ROEPR, service=198)

class KWP_RDBLI(Packet):
    localIdentifiers = ObservableDict({224: 'Development Data', 225: 'ECU Serial Number', 226: 'DBCom Data', 227: 'Operating System Version', 228: 'Ecu Reprogramming Identification', 229: 'Vehicle Information', 230: 'Flash Info 1', 231: 'Flash Info 2', 232: 'System Diagnostic general parameter data', 233: 'System Diagnostic global parameter data', 234: 'Ecu Configuration', 235: 'Diagnostic Protocol Information'})
    name = 'ReadDataByLocalIdentifier'
    fields_desc = [XByteEnumField('recordLocalIdentifier', 0, localIdentifiers)]
bind_layers(KWP, KWP_RDBLI, service=33)

class KWP_RDBLIPR(Packet):
    name = 'ReadDataByLocalIdentifierPositiveResponse'
    fields_desc = [XByteEnumField('recordLocalIdentifier', 0, KWP_RDBLI.localIdentifiers)]

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, KWP_RDBLI) and self.recordLocalIdentifier == other.recordLocalIdentifier
bind_layers(KWP, KWP_RDBLIPR, service=97)

class KWP_WDBLI(Packet):
    name = 'WriteDataByLocalIdentifier'
    fields_desc = [XByteEnumField('recordLocalIdentifier', 0, KWP_RDBLI.localIdentifiers)]
bind_layers(KWP, KWP_WDBLI, service=59)

class KWP_WDBLIPR(Packet):
    name = 'WriteDataByLocalIdentifierPositiveResponse'
    fields_desc = [XByteEnumField('recordLocalIdentifier', 0, KWP_RDBLI.localIdentifiers)]

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, KWP_WDBLI) and self.recordLocalIdentifier == other.recordLocalIdentifier
bind_layers(KWP, KWP_WDBLIPR, service=123)

class KWP_RDBI(Packet):
    dataIdentifiers = ObservableDict()
    name = 'ReadDataByIdentifier'
    fields_desc = [XShortEnumField('identifier', 0, dataIdentifiers)]
bind_layers(KWP, KWP_RDBI, service=34)

class KWP_RDBIPR(Packet):
    name = 'ReadDataByIdentifierPositiveResponse'
    fields_desc = [XShortEnumField('identifier', 0, KWP_RDBI.dataIdentifiers)]

    def answers(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, KWP_RDBI) and self.identifier == other.identifier
bind_layers(KWP, KWP_RDBIPR, service=98)

class KWP_RMBA(Packet):
    name = 'ReadMemoryByAddress'
    fields_desc = [X3BytesField('memoryAddress', 0), ByteField('memorySize', 0)]
bind_layers(KWP, KWP_RMBA, service=35)

class KWP_RMBAPR(Packet):
    name = 'ReadMemoryByAddressPositiveResponse'
    fields_desc = [StrField('dataRecord', b'', fmt='B')]

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, KWP_RMBA)
bind_layers(KWP, KWP_RMBAPR, service=99)

class KWP_DDLI(Packet):
    name = 'DynamicallyDefineLocalIdentifier'
    definitionModes = {1: 'defineByLocalIdentifier', 2: 'defineByMemoryAddress', 3: 'defineByIdentifier', 4: 'clearDynamicallyDefinedLocalIdentifier'}
    fields_desc = [XByteField('dynamicallyDefineLocalIdentifier', 0), ByteEnumField('definitionMode', 0, definitionModes), StrField('dataRecord', b'', fmt='B')]
bind_layers(KWP, KWP_DDLI, service=44)

class KWP_DDLIPR(Packet):
    name = 'DynamicallyDefineLocalIdentifierPositiveResponse'
    fields_desc = [XByteField('dynamicallyDefineLocalIdentifier', 0)]

    def answers(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, KWP_DDLI) and other.dynamicallyDefineLocalIdentifier == self.dynamicallyDefineLocalIdentifier
bind_layers(KWP, KWP_DDLIPR, service=108)

class KWP_WDBI(Packet):
    name = 'WriteDataByIdentifier'
    fields_desc = [XShortEnumField('identifier', 0, KWP_RDBI.dataIdentifiers)]
bind_layers(KWP, KWP_WDBI, service=46)

class KWP_WDBIPR(Packet):
    name = 'WriteDataByIdentifierPositiveResponse'
    fields_desc = [XShortEnumField('identifier', 0, KWP_RDBI.dataIdentifiers)]

    def answers(self, other):
        if False:
            return 10
        return isinstance(other, KWP_WDBI) and other.identifier == self.identifier
bind_layers(KWP, KWP_WDBIPR, service=110)

class KWP_WMBA(Packet):
    name = 'WriteMemoryByAddress'
    fields_desc = [X3BytesField('memoryAddress', 0), ByteField('memorySize', 0), StrField('dataRecord', b'', fmt='B')]
bind_layers(KWP, KWP_WMBA, service=61)

class KWP_WMBAPR(Packet):
    name = 'WriteMemoryByAddressPositiveResponse'
    fields_desc = [X3BytesField('memoryAddress', 0)]

    def answers(self, other):
        if False:
            return 10
        return isinstance(other, KWP_WMBA) and other.memoryAddress == self.memoryAddress
bind_layers(KWP, KWP_WMBAPR, service=125)

class KWP_CDI(Packet):
    DTCGroups = {0: 'allPowertrainDTCs', 16384: 'allChassisDTCs', 32768: 'allBodyDTCs', 49152: 'allNetworkDTCs', 65280: 'allDTCs'}
    name = 'ClearDiagnosticInformation'
    fields_desc = [XShortEnumField('groupOfDTC', 0, DTCGroups)]
bind_layers(KWP, KWP_CDI, service=20)

class KWP_CDIPR(Packet):
    name = 'ClearDiagnosticInformationPositiveResponse'
    fields_desc = [XShortEnumField('groupOfDTC', 0, KWP_CDI.DTCGroups)]

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, KWP_CDI) and self.groupOfDTC == other.groupOfDTC
bind_layers(KWP, KWP_CDIPR, service=84)

class KWP_RSODTC(Packet):
    name = 'ReadStatusOfDiagnosticTroubleCodes'
    fields_desc = [XShortEnumField('groupOfDTC', 0, KWP_CDI.DTCGroups)]
bind_layers(KWP, KWP_RSODTC, service=23)

class KWP_RSODTCPR(Packet):
    name = 'ReadStatusOfDiagnosticTroubleCodesPositiveResponse'
    fields_desc = [ByteField('numberOfDTC', 0)]

    def answers(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, KWP_RSODTC)
bind_layers(KWP, KWP_RSODTCPR, service=87)

class KWP_RECUI(Packet):
    name = 'ReadECUIdentification'
    localIdentifiers = ObservableDict({134: 'DCS ECU Identification', 135: 'DCX / MMC ECU Identification', 136: 'VIN (Original)', 137: 'Diagnostic Variant Code', 144: 'VIN (Current)', 150: 'Calibration Identification', 151: 'Calibration Verification Number', 154: 'ECU Code Fingerprint', 152: 'ECU Data Fingerprint', 156: 'ECU Code Software Identification', 157: 'ECU Data Software Identification', 158: 'ECU Boot Software Identification', 159: 'ECU Boot Fingerprint'})
    fields_desc = [XByteEnumField('localIdentifier', 0, localIdentifiers)]
bind_layers(KWP, KWP_RECUI, service=26)

class KWP_RECUIPR(Packet):
    name = 'ReadECUIdentificationPositiveResponse'
    fields_desc = [XByteEnumField('localIdentifier', 0, KWP_RECUI.localIdentifiers)]

    def answers(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, KWP_RECUI) and self.localIdentifier == other.localIdentifier
bind_layers(KWP, KWP_RECUIPR, service=90)

class KWP_SRBLI(Packet):
    routineLocalIdentifiers = ObservableDict({224: 'FlashEraseRoutine', 225: 'FlashCheckRoutine', 226: 'Tell-TaleRetentionStack', 227: 'RequestDTCsFromShadowErrorMemory', 228: 'RequestEnvironmentDataFromShadowErrorMemory', 229: 'RequestEventInformation', 230: 'RequestEventEnvironmentData', 231: 'RequestSoftwareModuleInformation', 232: 'ClearTell-TaleRetentionStack', 233: 'ClearEventInformation'})
    name = 'StartRoutineByLocalIdentifier'
    fields_desc = [XByteEnumField('routineLocalIdentifier', 0, routineLocalIdentifiers)]
bind_layers(KWP, KWP_SRBLI, service=49)

class KWP_SRBLIPR(Packet):
    name = 'StartRoutineByLocalIdentifierPositiveResponse'
    fields_desc = [XByteEnumField('routineLocalIdentifier', 0, KWP_SRBLI.routineLocalIdentifiers)]

    def answers(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, KWP_SRBLI) and other.routineLocalIdentifier == self.routineLocalIdentifier
bind_layers(KWP, KWP_SRBLIPR, service=113)

class KWP_STRBLI(Packet):
    name = 'StopRoutineByLocalIdentifier'
    fields_desc = [XByteEnumField('routineLocalIdentifier', 0, KWP_SRBLI.routineLocalIdentifiers)]
bind_layers(KWP, KWP_STRBLI, service=50)

class KWP_STRBLIPR(Packet):
    name = 'StopRoutineByLocalIdentifierPositiveResponse'
    fields_desc = [XByteEnumField('routineLocalIdentifier', 0, KWP_SRBLI.routineLocalIdentifiers)]

    def answers(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, KWP_STRBLI) and other.routineLocalIdentifier == self.routineLocalIdentifier
bind_layers(KWP, KWP_STRBLIPR, service=114)

class KWP_RRRBLI(Packet):
    name = 'RequestRoutineResultsByLocalIdentifier'
    fields_desc = [XByteEnumField('routineLocalIdentifier', 0, KWP_SRBLI.routineLocalIdentifiers)]
bind_layers(KWP, KWP_RRRBLI, service=51)

class KWP_RRRBLIPR(Packet):
    name = 'RequestRoutineResultsByLocalIdentifierPositiveResponse'
    fields_desc = [XByteEnumField('routineLocalIdentifier', 0, KWP_SRBLI.routineLocalIdentifiers)]

    def answers(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, KWP_RRRBLI) and other.routineLocalIdentifier == self.routineLocalIdentifier
bind_layers(KWP, KWP_RRRBLIPR, service=115)

class KWP_RD(Packet):
    name = 'RequestDownload'
    fields_desc = [X3BytesField('memoryAddress', 0), BitField('compression', 0, 4), BitField('encryption', 0, 4), X3BytesField('uncompressedMemorySize', 0)]
bind_layers(KWP, KWP_RD, service=52)

class KWP_RDPR(Packet):
    name = 'RequestDownloadPositiveResponse'
    fields_desc = [StrField('maxNumberOfBlockLength', b'', fmt='B')]

    def answers(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, KWP_RD)
bind_layers(KWP, KWP_RDPR, service=116)

class KWP_RU(Packet):
    name = 'RequestUpload'
    fields_desc = [X3BytesField('memoryAddress', 0), BitField('compression', 0, 4), BitField('encryption', 0, 4), X3BytesField('uncompressedMemorySize', 0)]
bind_layers(KWP, KWP_RU, service=53)

class KWP_RUPR(Packet):
    name = 'RequestUploadPositiveResponse'
    fields_desc = [StrField('maxNumberOfBlockLength', b'', fmt='B')]

    def answers(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, KWP_RU)
bind_layers(KWP, KWP_RUPR, service=117)

class KWP_TD(Packet):
    name = 'TransferData'
    fields_desc = [ByteField('blockSequenceCounter', 0), StrField('transferDataRequestParameter', b'', fmt='B')]
bind_layers(KWP, KWP_TD, service=54)

class KWP_TDPR(Packet):
    name = 'TransferDataPositiveResponse'
    fields_desc = [ByteField('blockSequenceCounter', 0), StrField('transferDataRequestParameter', b'', fmt='B')]

    def answers(self, other):
        if False:
            return 10
        return isinstance(other, KWP_TD) and other.blockSequenceCounter == self.blockSequenceCounter
bind_layers(KWP, KWP_TDPR, service=118)

class KWP_RTE(Packet):
    name = 'RequestTransferExit'
    fields_desc = [StrField('transferDataRequestParameter', b'', fmt='B')]
bind_layers(KWP, KWP_RTE, service=55)

class KWP_RTEPR(Packet):
    name = 'RequestTransferExitPositiveResponse'
    fields_desc = [StrField('transferDataRequestParameter', b'', fmt='B')]

    def answers(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, KWP_RTE)
bind_layers(KWP, KWP_RTEPR, service=119)

class KWP_NR(Packet):
    negativeResponseCodes = {0: 'positiveResponse', 16: 'generalReject', 17: 'serviceNotSupported', 18: 'subFunctionNotSupported-InvalidFormat', 33: 'busyRepeatRequest', 34: 'conditionsNotCorrect-RequestSequenceError', 35: 'routineNotComplete', 49: 'requestOutOfRange', 51: 'securityAccessDenied-SecurityAccessRequested', 53: 'invalidKey', 54: 'exceedNumberOfAttempts', 55: 'requiredTimeDelayNotExpired', 64: 'downloadNotAccepted', 80: 'uploadNotAccepted', 113: 'transferSuspended', 120: 'requestCorrectlyReceived-ResponsePending', 128: 'subFunctionNotSupportedInActiveDiagnosticSession', 154: 'dataDecompressionFailed', 155: 'dataDecryptionFailed', 160: 'EcuNotResponding', 161: 'EcuAddressUnknown'}
    name = 'NegativeResponse'
    fields_desc = [MayEnd(XByteEnumField('requestServiceId', 0, KWP.services)), ByteEnumField('negativeResponseCode', 0, negativeResponseCodes)]

    def answers(self, other):
        if False:
            return 10
        return self.requestServiceId == other.service and (self.negativeResponseCode != 120 or conf.contribs['KWP']['treat-response-pending-as-answer'])
bind_layers(KWP, KWP_NR, service=127)

class KWP_TesterPresentSender(PeriodicSenderThread):

    def __init__(self, sock, pkt=KWP() / KWP_TP(responseRequired=2), interval=2):
        if False:
            return 10
        ' Thread that sends TesterPresent packets periodically\n\n        :param sock: socket where packet is sent periodically\n        :param pkt: packet to send\n        :param interval: interval between two packets\n        '
        PeriodicSenderThread.__init__(self, sock, pkt, interval)

    def run(self):
        if False:
            return 10
        while not self._stopped.is_set():
            for p in self._pkts:
                self._socket.sr1(p, timeout=0.3, verbose=False)
                self._stopped.wait(timeout=self._interval)
                if self._stopped.is_set() or self._socket.closed:
                    break