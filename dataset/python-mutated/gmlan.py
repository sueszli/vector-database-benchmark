import struct
from scapy.contrib.automotive import log_automotive
from scapy.fields import ByteEnumField, ConditionalField, FieldListField, MayEnd, MultipleTypeField, ObservableDict, PacketField, PacketListField, ShortField, StrField, StrFixedLenField, X3BytesField, XByteEnumField, XByteField, XIntField, XShortEnumField, XShortField
from scapy.packet import Packet, bind_layers, NoPayload
from scapy.config import conf
from scapy.contrib.isotp import ISOTP
'\nGMLAN\n'
try:
    if conf.contribs['GMLAN']['treat-response-pending-as-answer']:
        pass
except KeyError:
    log_automotive.info('Specify "conf.contribs[\'GMLAN\'] = {\'treat-response-pending-as-answer\': True}" to treat a negative response \'RequestCorrectlyReceived-ResponsePending\' as answer of a request. \nThe default value is False.')
    conf.contribs['GMLAN'] = {'treat-response-pending-as-answer': False}
conf.contribs['GMLAN']['GMLAN_ECU_AddressingScheme'] = None

class GMLAN(ISOTP):

    @staticmethod
    def determine_len(x):
        if False:
            return 10
        if conf.contribs['GMLAN']['GMLAN_ECU_AddressingScheme'] is None:
            log_automotive.warning("Define conf.contribs['GMLAN']['GMLAN_ECU_AddressingScheme']! Assign either 2,3 or 4")
        if conf.contribs['GMLAN']['GMLAN_ECU_AddressingScheme'] not in [2, 3, 4]:
            log_automotive.warning("Define conf.contribs['GMLAN']['GMLAN_ECU_AddressingScheme']! Assign either 2,3 or 4")
        return conf.contribs['GMLAN']['GMLAN_ECU_AddressingScheme'] == x
    services = ObservableDict({4: 'ClearDiagnosticInformation', 16: 'InitiateDiagnosticOperation', 18: 'ReadFailureRecordData', 26: 'ReadDataByIdentifier', 32: 'ReturnToNormalOperation', 34: 'ReadDataByParameterIdentifier', 35: 'ReadMemoryByAddress', 39: 'SecurityAccess', 40: 'DisableNormalCommunication', 44: 'DynamicallyDefineMessage', 45: 'DefinePIDByAddress', 52: 'RequestDownload', 54: 'TransferData', 59: 'WriteDataByIdentifier', 62: 'TesterPresent', 68: 'ClearDiagnosticInformationPositiveResponse', 80: 'InitiateDiagnosticOperationPositiveResponse', 82: 'ReadFailureRecordDataPositiveResponse', 90: 'ReadDataByIdentifierPositiveResponse', 96: 'ReturnToNormalOperationPositiveResponse', 98: 'ReadDataByParameterIdentifierPositiveResponse', 99: 'ReadMemoryByAddressPositiveResponse', 103: 'SecurityAccessPositiveResponse', 104: 'DisableNormalCommunicationPositiveResponse', 108: 'DynamicallyDefineMessagePositiveResponse', 109: 'DefinePIDByAddressPositiveResponse', 116: 'RequestDownloadPositiveResponse', 118: 'TransferDataPositiveResponse', 123: 'WriteDataByIdentifierPositiveResponse', 126: 'TesterPresentPositiveResponse', 127: 'NegativeResponse', 162: 'ReportProgrammingState', 165: 'ProgrammingMode', 169: 'ReadDiagnosticInformation', 170: 'ReadDataByPacketIdentifier', 174: 'DeviceControl', 226: 'ReportProgrammingStatePositiveResponse', 229: 'ProgrammingModePositiveResponse', 233: 'ReadDiagnosticInformationPositiveResponse', 234: 'ReadDataByPacketIdentifierPositiveResponse', 238: 'DeviceControlPositiveResponse'})
    name = 'General Motors Local Area Network'
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
                return True
            else:
                return self.payload.answers(other.payload)
        return False

    def hashret(self):
        if False:
            print('Hello World!')
        if self.service == 127:
            return struct.pack('B', self.requestServiceId & ~64)
        return struct.pack('B', self.service & ~64)

class GMLAN_IDO(Packet):
    subfunctions = {2: 'disableAllDTCs', 3: 'enableDTCsDuringDevCntrl', 4: 'wakeUpLinks'}
    name = 'InitiateDiagnosticOperation'
    fields_desc = [ByteEnumField('subfunction', 0, subfunctions)]
bind_layers(GMLAN, GMLAN_IDO, service=16)

class GMLAN_DTC(Packet):
    name = 'GMLAN DTC information'
    fields_desc = [XByteField('failureRecordNumber', 0), XByteField('DTCHighByte', 0), XByteField('DTCLowByte', 0), XByteField('DTCFailureType', 0)]

    def extract_padding(self, p):
        if False:
            for i in range(10):
                print('nop')
        return ('', p)

class GMLAN_RFRD(Packet):
    subfunctions = {1: 'readFailureRecordIdentifiers', 2: 'readFailureRecordParameters'}
    name = 'ReadFailureRecordData'
    fields_desc = [ByteEnumField('subfunction', 0, subfunctions), ConditionalField(PacketField('dtc', b'', GMLAN_DTC), lambda pkt: pkt.subfunction == 2)]
bind_layers(GMLAN, GMLAN_RFRD, service=18)

class GMLAN_RFRDPR(Packet):
    name = 'ReadFailureRecordDataPositiveResponse'
    fields_desc = [ByteEnumField('subfunction', 0, GMLAN_RFRD.subfunctions)]

    def answers(self, other):
        if False:
            return 10
        return isinstance(other, GMLAN_RFRD) and other.subfunction == self.subfunction
bind_layers(GMLAN, GMLAN_RFRDPR, service=82)

class GMLAN_RFRDPR_RFRI(Packet):
    failureRecordDataStructureIdentifiers = {0: 'PID', 1: 'DPID'}
    name = 'ReadFailureRecordDataPositiveResponse_readFailureRecordIdentifiers'
    fields_desc = [ByteEnumField('failureRecordDataStructureIdentifier', 0, failureRecordDataStructureIdentifiers), PacketListField('dtcs', [], GMLAN_DTC)]
bind_layers(GMLAN_RFRDPR, GMLAN_RFRDPR_RFRI, subfunction=1)

class GMLAN_RFRDPR_RFRP(Packet):
    name = 'ReadFailureRecordDataPositiveResponse_readFailureRecordParameters'
    fields_desc = [PacketField('dtc', b'', GMLAN_DTC)]
bind_layers(GMLAN_RFRDPR, GMLAN_RFRDPR_RFRP, subfunction=2)

class GMLAN_RDBI(Packet):
    dataIdentifiers = ObservableDict({144: '$90: VehicleIdentificationNumber (VIN)', 146: '$92: SystemSupplierId (SYSSUPPID)', 151: '$97: SystemNameOrEngineType (SNOET)', 152: '$98: RepairShopCodeOrTesterSerialNumber (RSCOTSN)', 153: '$99: ProgrammingDate (PD)', 154: '$9a: DiagnosticDataIdentifier (DDI)', 155: '$9b: XmlConfigurationCompatibilityIdentifier (XMLCCID)', 156: '$9C: XmlDataFilePartNumber (XMLDFPN)', 157: '$9D: XmlDataFileAlphaCode (XMLDFAC)', 159: '$9F: PreviousStoredRepairShopCodeOrTesterSerialNumbers (PSRSCOTSN)', 160: '$A0: manufacturers_enable_counter (MEC)', 161: '$A1: ECUConfigurationOrCustomizationData (ECUCOCGD) 1', 162: '$A2: ECUConfigurationOrCustomizationData (ECUCOCGD) 2', 163: '$A3: ECUConfigurationOrCustomizationData (ECUCOCGD) 3', 164: '$A4: ECUConfigurationOrCustomizationData (ECUCOCGD) 4', 165: '$A5: ECUConfigurationOrCustomizationData (ECUCOCGD) 5', 166: '$A6: ECUConfigurationOrCustomizationData (ECUCOCGD) 6', 167: '$A7: ECUConfigurationOrCustomizationData (ECUCOCGD) 7', 168: '$A8: ECUConfigurationOrCustomizationData (ECUCOCGD) 8', 176: '$B0: ECUDiagnosticAddress (ECUADDR)', 177: '$B1: ECUFunctionalSystemsAndVirtualDevices (ECUFSAVD)', 178: '$B2: GM ManufacturingData (GMMD)', 179: '$B3: Data Universal Numbering System Identification (DUNS)', 180: '$B4: Manufacturing Traceability Characters (MTC)', 181: '$B5: GM BroadcastCode (GMBC)', 182: '$B6: GM Target Vehicle (GMTV)', 183: '$B7: GM Software Usage Description (GMSUD)', 184: '$B8: GM Bench Verification Information (GMBVI)', 185: '$B9: Subnet_Config_List_HighSpeed (SCLHS)', 186: '$BA: Subnet_Config_List_LowSpeed (SCLLS)', 187: '$BB: Subnet_Config_List_MidSpeed (SCLMS)', 188: '$BC: Subnet_Config_List_NonCan 1 (SCLNC 1)', 189: '$BD: Subnet_Config_List_NonCan 2 (SCLNC 2)', 190: '$BE: Subnet_Config_List_LIN (SCLLIN)', 191: '$BF: Subnet_Config_List_GMLANChassisExpansionBus (SCLGCEB)', 192: '$C0: BootSoftwarePartNumber (BSPN)', 193: '$C1: SoftwareModuleIdentifier (SWMI) 01', 194: '$C2: SoftwareModuleIdentifier (SWMI) 02', 195: '$C3: SoftwareModuleIdentifier (SWMI) 03', 196: '$C4: SoftwareModuleIdentifier (SWMI) 04', 197: '$C5: SoftwareModuleIdentifier (SWMI) 05', 198: '$C6: SoftwareModuleIdentifier (SWMI) 06', 199: '$C7: SoftwareModuleIdentifier (SWMI) 07', 200: '$C8: SoftwareModuleIdentifier (SWMI) 08', 201: '$C9: SoftwareModuleIdentifier (SWMI) 09', 202: '$CA: SoftwareModuleIdentifier (SWMI) 10', 203: '$CB: EndModelPartNumber', 204: '$CC: BaseModelPartNumber (BMPN)', 208: '$D0: BootSoftwarePartNumberAlphaCode', 209: '$D1: SoftwareModuleIdentifierAlphaCode (SWMIAC) 01', 210: '$D2: SoftwareModuleIdentifierAlphaCode (SWMIAC) 02', 211: '$D3: SoftwareModuleIdentifierAlphaCode (SWMIAC) 03', 212: '$D4: SoftwareModuleIdentifierAlphaCode (SWMIAC) 04', 213: '$D5: SoftwareModuleIdentifierAlphaCode (SWMIAC) 05', 214: '$D6: SoftwareModuleIdentifierAlphaCode (SWMIAC) 06', 215: '$D7: SoftwareModuleIdentifierAlphaCode (SWMIAC) 07', 216: '$D8: SoftwareModuleIdentifierAlphaCode (SWMIAC) 08', 217: '$D9: SoftwareModuleIdentifierAlphaCode (SWMIAC) 09', 218: '$DA: SoftwareModuleIdentifierAlphaCode (SWMIAC) 10', 219: '$DB: EndModelPartNumberAlphaCode', 220: '$DC: BaseModelPartNumberAlphaCode', 221: '$DD: SoftwareModuleIdentifierDataIdentifiers (SWMIDID)', 222: '$DE: GMLANIdentificationData (GMLANID)', 223: '$DF: ECUOdometerValue (ECUODO)', 224: '$E0: VehicleLevelDataRecord (VLDR) 0', 225: '$E1: VehicleLevelDataRecord (VLDR) 1', 226: '$E2: VehicleLevelDataRecord (VLDR) 2', 227: '$E3: VehicleLevelDataRecord (VLDR) 3', 228: '$E4: VehicleLevelDataRecord (VLDR) 4', 229: '$E5: VehicleLevelDataRecord (VLDR) 5', 230: '$E6: VehicleLevelDataRecord (VLDR) 6', 231: '$E7: VehicleLevelDataRecord (VLDR) 7', 232: '$E8: Subnet_Config_List_GMLANPowertrainExpansionBus (SCLGPEB)', 233: '$E9: Subnet_Config_List_GMLANFrontObjectExpansionBus (SCLGFOEB)', 234: '$EA: Subnet_Config_List_GMLANRearObjectExpansionBus (SCLGROEB)', 235: '$EB: Subnet_Config_List_GMLANExpansionBus1 (SCLGEB1)', 236: '$EC: Subnet_Config_List_GMLANExpansionBus2 (SCLGEB2)', 237: '$ED: Subnet_Config_List_GMLANExpansionBus3 (SCLGEB3)', 238: '$EE: Subnet_Config_List_GMLANExpansionBus4 (SCLGEB4)', 239: '$EF: Subnet_Config_List_GMLANExpansionBus5 (SCLGEB5)'})
    name = 'ReadDataByIdentifier'
    fields_desc = [XByteEnumField('dataIdentifier', 0, dataIdentifiers)]
bind_layers(GMLAN, GMLAN_RDBI, service=26)

class GMLAN_RDBIPR(Packet):
    name = 'ReadDataByIdentifierPositiveResponse'
    fields_desc = [XByteEnumField('dataIdentifier', 0, GMLAN_RDBI.dataIdentifiers)]

    def answers(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, GMLAN_RDBI) and other.dataIdentifier == self.dataIdentifier
bind_layers(GMLAN, GMLAN_RDBIPR, service=90)

class GMLAN_RDBPI(Packet):
    dataIdentifiers = ObservableDict({5: 'OBD_EngineCoolantTemperature', 12: 'OBD_EngineRPM', 31: 'OBD_TimeSinceEngineStart'})
    name = 'ReadDataByParameterIdentifier'
    fields_desc = [FieldListField('identifiers', [], XShortEnumField('parameterIdentifier', 0, dataIdentifiers))]
bind_layers(GMLAN, GMLAN_RDBPI, service=34)

class GMLAN_RDBPIPR(Packet):
    name = 'ReadDataByParameterIdentifierPositiveResponse'
    fields_desc = [XShortEnumField('parameterIdentifier', 0, GMLAN_RDBPI.dataIdentifiers)]

    def answers(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, GMLAN_RDBPI) and self.parameterIdentifier in other.identifiers
bind_layers(GMLAN, GMLAN_RDBPIPR, service=98)

class GMLAN_RDBPKTI(Packet):
    name = 'ReadDataByPacketIdentifier'
    subfunctions = {0: 'stopSending', 1: 'sendOneResponse', 2: 'scheduleAtSlowRate', 3: 'scheduleAtMediumRate', 4: 'scheduleAtFastRate'}
    fields_desc = [XByteEnumField('subfunction', 0, subfunctions), ConditionalField(FieldListField('request_DPIDs', [], XByteField('', 0)), lambda pkt: pkt.subfunction > 0)]
bind_layers(GMLAN, GMLAN_RDBPKTI, service=170)

class GMLAN_RMBA(Packet):
    name = 'ReadMemoryByAddress'
    fields_desc = [MultipleTypeField([(XShortField('memoryAddress', 0), lambda pkt: GMLAN.determine_len(2)), (X3BytesField('memoryAddress', 0), lambda pkt: GMLAN.determine_len(3)), (XIntField('memoryAddress', 0), lambda pkt: GMLAN.determine_len(4))], XIntField('memoryAddress', 0)), XShortField('memorySize', 0)]
bind_layers(GMLAN, GMLAN_RMBA, service=35)

class GMLAN_RMBAPR(Packet):
    name = 'ReadMemoryByAddressPositiveResponse'
    fields_desc = [MultipleTypeField([(XShortField('memoryAddress', 0), lambda pkt: GMLAN.determine_len(2)), (X3BytesField('memoryAddress', 0), lambda pkt: GMLAN.determine_len(3)), (XIntField('memoryAddress', 0), lambda pkt: GMLAN.determine_len(4))], XIntField('memoryAddress', 0)), StrField('dataRecord', b'', fmt='B')]

    def answers(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, GMLAN_RMBA) and other.memoryAddress == self.memoryAddress
bind_layers(GMLAN, GMLAN_RMBAPR, service=99)

class GMLAN_SA(Packet):
    subfunctions = {0: 'ReservedByDocument', 1: 'SPSrequestSeed', 2: 'SPSsendKey', 3: 'DevCtrlrequestSeed', 4: 'DevCtrlsendKey', 255: 'ReservedByDocument'}
    for i in range(5, 10 + 1):
        subfunctions[i] = 'ReservedByDocument'
    for i in range(11, 250 + 1):
        subfunctions[i] = 'Reserved for vehicle manufacturer specific needs'
    for i in range(251, 254 + 1):
        subfunctions[i] = 'Reserved for ECU or system supplier manufacturing needs'
    name = 'SecurityAccess'
    fields_desc = [ByteEnumField('subfunction', 0, subfunctions), ConditionalField(XShortField('securityKey', 0), lambda pkt: pkt.subfunction % 2 == 0)]
bind_layers(GMLAN, GMLAN_SA, service=39)

class GMLAN_SAPR(Packet):
    name = 'SecurityAccessPositiveResponse'
    fields_desc = [ByteEnumField('subfunction', 0, GMLAN_SA.subfunctions), ConditionalField(XShortField('securitySeed', 0), lambda pkt: pkt.subfunction % 2 == 1)]

    def answers(self, other):
        if False:
            return 10
        return isinstance(other, GMLAN_SA) and other.subfunction == self.subfunction
bind_layers(GMLAN, GMLAN_SAPR, service=103)

class GMLAN_DDM(Packet):
    name = 'DynamicallyDefineMessage'
    fields_desc = [XByteField('DPIDIdentifier', 0), StrField('PIDData', b'\x00\x00')]
bind_layers(GMLAN, GMLAN_DDM, service=44)

class GMLAN_DDMPR(Packet):
    name = 'DynamicallyDefineMessagePositiveResponse'
    fields_desc = [XByteField('DPIDIdentifier', 0)]

    def answers(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, GMLAN_DDM) and other.DPIDIdentifier == self.DPIDIdentifier
bind_layers(GMLAN, GMLAN_DDMPR, service=108)

class GMLAN_DPBA(Packet):
    name = 'DefinePIDByAddress'
    fields_desc = [XShortField('parameterIdentifier', 0), MultipleTypeField([(XShortField('memoryAddress', 0), lambda pkt: GMLAN.determine_len(2)), (X3BytesField('memoryAddress', 0), lambda pkt: GMLAN.determine_len(3)), (XIntField('memoryAddress', 0), lambda pkt: GMLAN.determine_len(4))], XIntField('memoryAddress', 0)), XByteField('memorySize', 0)]
bind_layers(GMLAN, GMLAN_DPBA, service=45)

class GMLAN_DPBAPR(Packet):
    name = 'DefinePIDByAddressPositiveResponse'
    fields_desc = [XShortField('parameterIdentifier', 0)]

    def answers(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, GMLAN_DPBA) and other.parameterIdentifier == self.parameterIdentifier
bind_layers(GMLAN, GMLAN_DPBAPR, service=109)

class GMLAN_RD(Packet):
    name = 'RequestDownload'
    fields_desc = [XByteField('dataFormatIdentifier', 0), MultipleTypeField([(XShortField('memorySize', 0), lambda pkt: GMLAN.determine_len(2)), (X3BytesField('memorySize', 0), lambda pkt: GMLAN.determine_len(3)), (XIntField('memorySize', 0), lambda pkt: GMLAN.determine_len(4))], XIntField('memorySize', 0))]
bind_layers(GMLAN, GMLAN_RD, service=52)

class GMLAN_TD(Packet):
    subfunctions = {0: 'download', 128: 'downloadAndExecuteOrExecute'}
    name = 'TransferData'
    fields_desc = [ByteEnumField('subfunction', 0, subfunctions), MultipleTypeField([(XShortField('startingAddress', 0), lambda pkt: GMLAN.determine_len(2)), (X3BytesField('startingAddress', 0), lambda pkt: GMLAN.determine_len(3)), (XIntField('startingAddress', 0), lambda pkt: GMLAN.determine_len(4))], XIntField('startingAddress', 0)), StrField('dataRecord', b'')]
bind_layers(GMLAN, GMLAN_TD, service=54)

class GMLAN_WDBI(Packet):
    name = 'WriteDataByIdentifier'
    fields_desc = [XByteEnumField('dataIdentifier', 0, GMLAN_RDBI.dataIdentifiers), StrField('dataRecord', b'')]
bind_layers(GMLAN, GMLAN_WDBI, service=59)

class GMLAN_WDBIPR(Packet):
    name = 'WriteDataByIdentifierPositiveResponse'
    fields_desc = [XByteEnumField('dataIdentifier', 0, GMLAN_RDBI.dataIdentifiers)]

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, GMLAN_WDBI) and other.dataIdentifier == self.dataIdentifier
bind_layers(GMLAN, GMLAN_WDBIPR, service=123)

class GMLAN_RPSPR(Packet):
    programmedStates = {0: 'fully programmed', 1: 'no op s/w or cal data', 2: 'op s/w present, cal data missing', 3: 's/w present, default or no start cal present', 80: 'General Memory Fault', 81: 'RAM Memory Fault', 82: 'NVRAM Memory Fault', 83: 'Boot Memory Failure', 84: 'Flash Memory Failure', 85: 'EEPROM Memory Failure'}
    name = 'ReportProgrammedStatePositiveResponse'
    fields_desc = [ByteEnumField('programmedState', 0, programmedStates)]
bind_layers(GMLAN, GMLAN_RPSPR, service=226)

class GMLAN_PM(Packet):
    subfunctions = {1: 'requestProgrammingMode', 2: 'requestProgrammingMode_HighSpeed', 3: 'enableProgrammingMode'}
    name = 'ProgrammingMode'
    fields_desc = [ByteEnumField('subfunction', 0, subfunctions)]
bind_layers(GMLAN, GMLAN_PM, service=165)

class GMLAN_RDI(Packet):
    subfunctions = {128: 'readStatusOfDTCByDTCNumber', 129: 'readStatusOfDTCByStatusMask', 130: 'sendOnChangeDTCCount'}
    name = 'ReadDiagnosticInformation'
    fields_desc = [ByteEnumField('subfunction', 0, subfunctions)]
bind_layers(GMLAN, GMLAN_RDI, service=169)

class GMLAN_RDI_BN(Packet):
    name = 'ReadStatusOfDTCByDTCNumber'
    fields_desc = [XByteField('DTCHighByte', 0), XByteField('DTCLowByte', 0), XByteField('DTCFailureType', 0)]
bind_layers(GMLAN_RDI, GMLAN_RDI_BN, subfunction=128)

class GMLAN_RDI_BM(Packet):
    name = 'ReadStatusOfDTCByStatusMask'
    fields_desc = [XByteField('DTCStatusMask', 0)]
bind_layers(GMLAN_RDI, GMLAN_RDI_BM, subfunction=129)

class GMLAN_RDI_BC(Packet):
    name = 'SendOnChangeDTCCount'
    fields_desc = [XByteField('DTCStatusMask', 0)]
bind_layers(GMLAN_RDI, GMLAN_RDI_BC, subfunction=130)

class GMLAN_DC(Packet):
    name = 'DeviceControl'
    fields_desc = [XByteField('CPIDNumber', 0), StrFixedLenField('CPIDControlBytes', b'', 5)]
bind_layers(GMLAN, GMLAN_DC, service=174)

class GMLAN_DCPR(Packet):
    name = 'DeviceControlPositiveResponse'
    fields_desc = [XByteField('CPIDNumber', 0)]

    def answers(self, other):
        if False:
            return 10
        return isinstance(other, GMLAN_DC) and other.CPIDNumber == self.CPIDNumber
bind_layers(GMLAN, GMLAN_DCPR, service=238)

class GMLAN_NR(Packet):
    negativeResponseCodes = {17: 'ServiceNotSupported', 18: 'SubFunctionNotSupported', 34: 'ConditionsNotCorrectOrRequestSequenceError', 49: 'RequestOutOfRange', 53: 'InvalidKey', 54: 'ExceedNumberOfAttempts', 55: 'RequiredTimeDelayNotExpired', 120: 'RequestCorrectlyReceived-ResponsePending', 129: 'SchedulerFull', 131: 'VoltageOutOfRange', 133: 'GeneralProgrammingFailure', 137: 'DeviceTypeError', 153: 'ReadyForDownload-DTCStored', 227: 'DeviceControlLimitsExceeded'}
    name = 'NegativeResponse'
    fields_desc = [XByteEnumField('requestServiceId', 0, GMLAN.services), MayEnd(ByteEnumField('returnCode', 0, negativeResponseCodes)), ShortField('deviceControlLimitExceeded', 0)]

    def answers(self, other):
        if False:
            while True:
                i = 10
        return self.requestServiceId == other.service and (self.returnCode != 120 or conf.contribs['GMLAN']['treat-response-pending-as-answer'])
bind_layers(GMLAN, GMLAN_NR, service=127)