from scapy.contrib.automotive.uds import UDS_DSCPR, UDS_ERPR, UDS_SAPR, UDS_RDBPIPR, UDS_CCPR, UDS_TPPR, UDS_RDPR, UDS
from scapy.packet import Packet
from scapy.contrib.automotive.ecu import EcuState
__all__ = ['UDS_DSCPR_modify_ecu_state', 'UDS_CCPR_modify_ecu_state', 'UDS_ERPR_modify_ecu_state', 'UDS_RDBPIPR_modify_ecu_state', 'UDS_TPPR_modify_ecu_state', 'UDS_SAPR_modify_ecu_state', 'UDS_RDPR_modify_ecu_state']

@EcuState.extend_pkt_with_modifier(UDS_DSCPR)
def UDS_DSCPR_modify_ecu_state(self, req, state):
    if False:
        for i in range(10):
            print('nop')
    state.session = self.diagnosticSessionType
    try:
        del state.security_level
    except AttributeError:
        pass

@EcuState.extend_pkt_with_modifier(UDS_ERPR)
def UDS_ERPR_modify_ecu_state(self, req, state):
    if False:
        return 10
    state.reset()
    state.session = 1

@EcuState.extend_pkt_with_modifier(UDS_SAPR)
def UDS_SAPR_modify_ecu_state(self, req, state):
    if False:
        print('Hello World!')
    if self.securityAccessType % 2 == 0 and self.securityAccessType > 0 and (len(req) >= 3):
        state.security_level = self.securityAccessType
    elif self.securityAccessType % 2 == 1 and self.securityAccessType > 0 and (len(req) >= 3) and (not any(self.securitySeed)):
        state.security_level = self.securityAccessType + 1

@EcuState.extend_pkt_with_modifier(UDS_CCPR)
def UDS_CCPR_modify_ecu_state(self, req, state):
    if False:
        print('Hello World!')
    state.communication_control = self.controlType

@EcuState.extend_pkt_with_modifier(UDS_TPPR)
def UDS_TPPR_modify_ecu_state(self, req, state):
    if False:
        i = 10
        return i + 15
    state.tp = 1

@EcuState.extend_pkt_with_modifier(UDS_RDBPIPR)
def UDS_RDBPIPR_modify_ecu_state(self, req, state):
    if False:
        print('Hello World!')
    state.pdid = self.periodicDataIdentifier

@EcuState.extend_pkt_with_modifier(UDS_RDPR)
def UDS_RDPR_modify_ecu_state(self, req, state):
    if False:
        while True:
            i = 10
    oldstr = getattr(state, 'req_download', '')
    newstr = str(req.fields)
    state.req_download = oldstr if newstr in oldstr else oldstr + newstr

@EcuState.extend_pkt_with_modifier(UDS)
def UDS_modify_ecu_state(self, req, state):
    if False:
        for i in range(10):
            print('nop')
    if self.service == 119:
        try:
            state.download_complete = state.req_download
        except (KeyError, AttributeError):
            pass
        state.req_download = ''