from scapy.packet import Packet
from scapy.contrib.automotive.ecu import EcuState
from scapy.contrib.automotive.gm.gmlan import GMLAN, GMLAN_SAPR
__all__ = ['GMLAN_modify_ecu_state', 'GMLAN_SAPR_modify_ecu_state']

@EcuState.extend_pkt_with_modifier(GMLAN)
def GMLAN_modify_ecu_state(self, req, state):
    if False:
        print('Hello World!')
    if self.service == 80:
        state.session = 3
    elif self.service == 96:
        state.reset()
        state.session = 1
    elif self.service == 104:
        state.communication_control = 1
    elif self.service == 229:
        state.session = 2
    elif self.service == 116 and len(req) > 3:
        state.request_download = 1
    elif self.service == 126:
        state.tp = 1

@EcuState.extend_pkt_with_modifier(GMLAN_SAPR)
def GMLAN_SAPR_modify_ecu_state(self, req, state):
    if False:
        for i in range(10):
            print('nop')
    if self.subfunction % 2 == 0 and self.subfunction > 0 and (len(req) >= 3):
        state.security_level = self.subfunction
    elif self.subfunction % 2 == 1 and self.subfunction > 0 and (len(req) >= 3) and (not any(self.securitySeed)):
        state.security_level = self.securityAccessType + 1