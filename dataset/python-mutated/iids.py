from scapy.fields import FieldLenField, FieldListField, StrFixedLenField, ByteField, ShortField, FlagsField, XByteField, PacketListField
from scapy.packet import Packet, bind_layers
from scapy.contrib.automotive.obd.packet import OBD_Packet
from scapy.contrib.automotive.obd.services import OBD_S09

class OBD_S09_PR_Record(Packet):
    fields_desc = [XByteField('iid', 0)]

class OBD_S09_PR(Packet):
    name = 'Infotype IDs'
    fields_desc = [PacketListField('data_records', [], OBD_S09_PR_Record)]

    def answers(self, other):
        if False:
            return 10
        return isinstance(other, OBD_S09) and all((r.iid in other.iid for r in self.data_records))

class OBD_IID00(OBD_Packet):
    name = 'IID_00_Service9SupportedInformationTypes'
    fields_desc = [FlagsField('supported_iids', 0, 32, ['IID20', 'IID1F', 'IID1E', 'IID1D', 'IID1C', 'IID1B', 'IID1A', 'IID19', 'IID18', 'IID17', 'IID16', 'IID15', 'IID14', 'IID13', 'IID12', 'IID11', 'IID10', 'IID0F', 'IID0E', 'IID0D', 'IID0C', 'IID0B', 'IID0A', 'IID09', 'IID08', 'IID07', 'IID06', 'IID05', 'IID04', 'IID03', 'IID02', 'IID01'])]

class _OBD_IID_MessageCount(OBD_Packet):
    fields_desc = [ByteField('message_count', 0)]

class OBD_IID01(_OBD_IID_MessageCount):
    name = 'IID_01_VinMessageCount'

class OBD_IID03(_OBD_IID_MessageCount):
    name = 'IID_03_CalibrationIdMessageCount'

class OBD_IID05(_OBD_IID_MessageCount):
    name = 'IID_05_CalibrationVerificationNumbersMessageCount'

class OBD_IID07(_OBD_IID_MessageCount):
    name = 'IID_07_InUsePerformanceTrackingMessageCount'

class OBD_IID09(_OBD_IID_MessageCount):
    name = 'IID_09_EcuNameMessageCount'

class OBD_IID02(OBD_Packet):
    name = 'IID_02_VehicleIdentificationNumber'
    fields_desc = [FieldLenField('count', None, count_of='vehicle_identification_numbers', fmt='B'), FieldListField('vehicle_identification_numbers', [], StrFixedLenField('', b'', 17), count_from=lambda pkt: pkt.count)]

class OBD_IID04(OBD_Packet):
    name = 'IID_04_CalibrationId'
    fields_desc = [FieldLenField('count', None, count_of='calibration_identifications', fmt='B'), FieldListField('calibration_identifications', [], StrFixedLenField('', b'', 16), count_from=lambda pkt: pkt.count)]

class OBD_IID06(OBD_Packet):
    name = 'IID_06_CalibrationVerificationNumbers'
    fields_desc = [FieldLenField('count', None, count_of='calibration_verification_numbers', fmt='B'), FieldListField('calibration_verification_numbers', [], StrFixedLenField('', b'', 4), count_from=lambda pkt: pkt.count)]

class OBD_IID08(OBD_Packet):
    name = 'IID_08_InUsePerformanceTracking'
    fields_desc = [FieldLenField('count', None, count_of='data', fmt='B'), FieldListField('data', [], ShortField('', 0), count_from=lambda pkt: pkt.count)]

class OBD_IID0A(OBD_Packet):
    name = 'IID_0A_EcuName'
    fields_desc = [FieldLenField('count', None, count_of='ecu_names', fmt='B'), FieldListField('ecu_names', [], StrFixedLenField('', b'', 20), count_from=lambda pkt: pkt.count)]

class OBD_IID0B(OBD_Packet):
    name = 'IID_0B_InUsePerformanceTrackingForCompressionIgnitionVehicles'
    fields_desc = [FieldLenField('count', None, count_of='data', fmt='B'), FieldListField('data', [], ShortField('', 0), count_from=lambda pkt: pkt.count)]
bind_layers(OBD_S09_PR_Record, OBD_IID00, iid=0)
bind_layers(OBD_S09_PR_Record, OBD_IID01, iid=1)
bind_layers(OBD_S09_PR_Record, OBD_IID02, iid=2)
bind_layers(OBD_S09_PR_Record, OBD_IID03, iid=3)
bind_layers(OBD_S09_PR_Record, OBD_IID04, iid=4)
bind_layers(OBD_S09_PR_Record, OBD_IID05, iid=5)
bind_layers(OBD_S09_PR_Record, OBD_IID06, iid=6)
bind_layers(OBD_S09_PR_Record, OBD_IID07, iid=7)
bind_layers(OBD_S09_PR_Record, OBD_IID08, iid=8)
bind_layers(OBD_S09_PR_Record, OBD_IID09, iid=9)
bind_layers(OBD_S09_PR_Record, OBD_IID0A, iid=10)
bind_layers(OBD_S09_PR_Record, OBD_IID0B, iid=11)