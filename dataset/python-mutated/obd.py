import struct
from scapy.contrib.automotive import log_automotive
from scapy.contrib.automotive.obd.iid.iids import *
from scapy.contrib.automotive.obd.mid.mids import *
from scapy.contrib.automotive.obd.pid.pids import *
from scapy.contrib.automotive.obd.tid.tids import *
from scapy.contrib.automotive.obd.services import *
from scapy.packet import bind_layers, NoPayload
from scapy.config import conf
from scapy.fields import XByteEnumField
from scapy.contrib.isotp import ISOTP
try:
    if conf.contribs['OBD']['treat-response-pending-as-answer']:
        pass
except KeyError:
    log_automotive.info('Specify "conf.contribs[\'OBD\'] = {\'treat-response-pending-as-answer\': True}" to treat a negative response \'requestCorrectlyReceived-ResponsePending\' as answer of a request. \nThe default value is False.')
    conf.contribs['OBD'] = {'treat-response-pending-as-answer': False}

class OBD(ISOTP):
    services = {1: 'CurrentPowertrainDiagnosticDataRequest', 2: 'PowertrainFreezeFrameDataRequest', 3: 'EmissionRelatedDiagnosticTroubleCodesRequest', 4: 'ClearResetDiagnosticTroubleCodesRequest', 5: 'OxygenSensorMonitoringTestResultsRequest', 6: 'OnBoardMonitoringTestResultsRequest', 7: 'PendingEmissionRelatedDiagnosticTroubleCodesRequest', 8: 'ControlOperationRequest', 9: 'VehicleInformationRequest', 10: 'PermanentDiagnosticTroubleCodesRequest', 65: 'CurrentPowertrainDiagnosticDataResponse', 66: 'PowertrainFreezeFrameDataResponse', 67: 'EmissionRelatedDiagnosticTroubleCodesResponse', 68: 'ClearResetDiagnosticTroubleCodesResponse', 69: 'OxygenSensorMonitoringTestResultsResponse', 70: 'OnBoardMonitoringTestResultsResponse', 71: 'PendingEmissionRelatedDiagnosticTroubleCodesResponse', 72: 'ControlOperationResponse', 73: 'VehicleInformationResponse', 74: 'PermanentDiagnosticTroubleCodesResponse', 127: 'NegativeResponse'}
    name = 'On-board diagnostics'
    fields_desc = [XByteEnumField('service', 0, services)]

    def hashret(self):
        if False:
            i = 10
            return i + 15
        if self.service == 127:
            return struct.pack('B', self.request_service_id & ~64)
        return struct.pack('B', self.service & ~64)

    def answers(self, other):
        if False:
            for i in range(10):
                print('nop')
        if other.__class__ != self.__class__:
            return False
        if self.service == 127:
            return self.payload.answers(other)
        if self.service == other.service + 64:
            if isinstance(self.payload, NoPayload) or isinstance(other.payload, NoPayload):
                return True
            else:
                return self.payload.answers(other.payload)
        return False
bind_layers(OBD, OBD_S01, service=1)
bind_layers(OBD, OBD_S02, service=2)
bind_layers(OBD, OBD_S03, service=3)
bind_layers(OBD, OBD_S04, service=4)
bind_layers(OBD, OBD_S06, service=6)
bind_layers(OBD, OBD_S07, service=7)
bind_layers(OBD, OBD_S08, service=8)
bind_layers(OBD, OBD_S09, service=9)
bind_layers(OBD, OBD_S0A, service=10)
bind_layers(OBD, OBD_S01_PR, service=65)
bind_layers(OBD, OBD_S02_PR, service=66)
bind_layers(OBD, OBD_S03_PR, service=67)
bind_layers(OBD, OBD_S04_PR, service=68)
bind_layers(OBD, OBD_S06_PR, service=70)
bind_layers(OBD, OBD_S07_PR, service=71)
bind_layers(OBD, OBD_S08_PR, service=72)
bind_layers(OBD, OBD_S09_PR, service=73)
bind_layers(OBD, OBD_S0A_PR, service=74)
bind_layers(OBD, OBD_NR, service=127)