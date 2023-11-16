"""
    MACControl
    ~~~~~~~~~~

    :author:    Thomas Tannhaeuser, hecke@naberius.de

    :description:

        This module provides Scapy layers for the MACControl protocol messages:
            - Pause
            - Gate
            - Report
            - Register/REQ/ACK
            - Class Based Flow Control

        normative references:
            - IEEE 802.3x


    :NOTES:
        - this is based on the MACControl dissector used by Wireshark
          (https://github.com/wireshark/wireshark/blob/master/epan/dissectors/packet-maccontrol.c)

"""
from scapy.compat import orb
from scapy.data import ETHER_TYPES
from scapy.error import Scapy_Exception
from scapy.fields import IntField, ByteField, ByteEnumField, ShortField, BitField
from scapy.layers.dot11 import Packet
from scapy.layers.l2 import Ether, Dot1Q, bind_layers
MAC_CONTROL_ETHER_TYPE = 34824
ETHER_TYPES[MAC_CONTROL_ETHER_TYPE] = 'MAC_CONTROL'
ETHER_SPEED_MBIT_10 = 1
ETHER_SPEED_MBIT_100 = 2
ETHER_SPEED_MBIT_1000 = 4

class MACControl(Packet):
    DEFAULT_DST_MAC = '01:80:c2:00:00:01'
    OP_CODE_PAUSE = 1
    OP_CODE_GATE = 2
    OP_CODE_REPORT = 3
    OP_CODE_REGISTER_REQ = 4
    OP_CODE_REGISTER = 5
    OP_CODE_REGISTER_ACK = 6
    OP_CODE_CLASS_BASED_FLOW_CONTROL = 257
    OP_CODES = {OP_CODE_PAUSE: 'pause', OP_CODE_GATE: 'gate', OP_CODE_REPORT: 'report', OP_CODE_REGISTER_REQ: 'register req', OP_CODE_REGISTER: 'register', OP_CODE_REGISTER_ACK: 'register_ack', OP_CODE_CLASS_BASED_FLOW_CONTROL: 'class based flow control'}
    '\n    flags used by Register* messages\n    '
    FLAG_REGISTER = 1
    FLAG_DEREGISTER = 2
    FLAG_ACK = 3
    FLAG_NACK = 4
    REGISTER_FLAGS = {FLAG_REGISTER: 'register', FLAG_DEREGISTER: 'deregister', FLAG_ACK: 'ack', FLAG_NACK: 'nack'}

    def guess_payload_class(self, payload):
        if False:
            while True:
                i = 10
        try:
            op_code = (orb(payload[0]) << 8) + orb(payload[1])
            return MAC_CTRL_CLASSES[op_code]
        except KeyError:
            pass
        return Packet.guess_payload_class(self, payload)

    def _get_underlayers_size(self):
        if False:
            while True:
                i = 10
        '\n        get the total size of all under layers\n        :return: number of bytes\n        '
        under_layer = self.underlayer
        under_layers_size = 0
        while under_layer and isinstance(under_layer, Dot1Q):
            under_layers_size += 4
            under_layer = under_layer.underlayer
        if under_layer and isinstance(under_layer, Ether):
            under_layers_size += 14 + 4
        return under_layers_size

    def post_build(self, pkt, pay):
        if False:
            while True:
                i = 10
        '\n        add padding to the frame if required.\n\n        note that padding is only added if pay is None/empty. this allows us to add  # noqa: E501\n        any payload after the MACControl* PDU if needed (piggybacking).\n        '
        if not pay:
            under_layers_size = self._get_underlayers_size()
            frame_size = len(pkt) + under_layers_size
            if frame_size < 64:
                return pkt + b'\x00' * (64 - frame_size)
        return pkt + pay

class MACControlInvalidSpeedException(Scapy_Exception):
    pass

class MACControlPause(MACControl):
    fields_desc = [ShortField('_op_code', MACControl.OP_CODE_PAUSE), ShortField('pause_time', 0)]

    def get_pause_time(self, speed=ETHER_SPEED_MBIT_1000):
        if False:
            print('Hello World!')
        '\n        get pause time for given link speed in seconds\n\n        :param speed: select link speed to get the pause time for, must be ETHER_SPEED_MBIT_[10,100,1000]  # noqa: E501\n        :return: pause time in seconds\n        :raises MACControlInvalidSpeedException: on invalid speed selector\n        '
        try:
            return self.pause_time * {ETHER_SPEED_MBIT_10: 1e-07 * 512, ETHER_SPEED_MBIT_100: 1e-08 * 512, ETHER_SPEED_MBIT_1000: 1e-09 * 512 * 2}[speed]
        except KeyError:
            raise MACControlInvalidSpeedException('Invalid speed selector given. Must be one of ETHER_SPEED_MBIT_[10,100,1000]')

class MACControlGate(MACControl):
    fields_desc = [ShortField('_op_code', MACControl.OP_CODE_GATE), IntField('timestamp', 0)]

class MACControlReport(MACControl):
    fields_desc = [ShortField('_op_code', MACControl.OP_CODE_REPORT), IntField('timestamp', 0), ByteEnumField('flags', 0, MACControl.REGISTER_FLAGS), ByteField('pending_grants', 0)]

class MACControlRegisterReq(MACControl):
    fields_desc = [ShortField('_op_code', MACControl.OP_CODE_REGISTER_REQ), IntField('timestamp', 0), ShortField('assigned_port', 0), ByteEnumField('flags', 0, MACControl.REGISTER_FLAGS), ShortField('sync_time', 0), ByteField('echoed_pending_grants', 0)]

class MACControlRegister(MACControl):
    fields_desc = [ShortField('_op_code', MACControl.OP_CODE_REGISTER), IntField('timestamp', 0), ByteEnumField('flags', 0, MACControl.REGISTER_FLAGS), ShortField('echoed_assigned_port', 0), ShortField('echoed_sync_time', 0)]

class MACControlRegisterAck(MACControl):
    fields_desc = [ShortField('_op_code', MACControl.OP_CODE_REGISTER_ACK), IntField('timestamp', 0), ByteEnumField('flags', 0, MACControl.REGISTER_FLAGS), ShortField('echoed_assigned_port', 0), ShortField('echoed_sync_time', 0)]

class MACControlClassBasedFlowControl(MACControl):
    fields_desc = [ShortField('_op_code', MACControl.OP_CODE_CLASS_BASED_FLOW_CONTROL), ByteField('_reserved', 0), BitField('c7_enabled', 0, 1), BitField('c6_enabled', 0, 1), BitField('c5_enabled', 0, 1), BitField('c4_enabled', 0, 1), BitField('c3_enabled', 0, 1), BitField('c2_enabled', 0, 1), BitField('c1_enabled', 0, 1), BitField('c0_enabled', 0, 1), ShortField('c0_pause_time', 0), ShortField('c1_pause_time', 0), ShortField('c2_pause_time', 0), ShortField('c3_pause_time', 0), ShortField('c4_pause_time', 0), ShortField('c5_pause_time', 0), ShortField('c6_pause_time', 0), ShortField('c7_pause_time', 0)]
MAC_CTRL_CLASSES = {MACControl.OP_CODE_PAUSE: MACControlPause, MACControl.OP_CODE_GATE: MACControlGate, MACControl.OP_CODE_REPORT: MACControlReport, MACControl.OP_CODE_REGISTER_REQ: MACControlRegisterReq, MACControl.OP_CODE_REGISTER: MACControlRegister, MACControl.OP_CODE_REGISTER_ACK: MACControlRegisterAck, MACControl.OP_CODE_CLASS_BASED_FLOW_CONTROL: MACControlClassBasedFlowControl}
bind_layers(Ether, MACControl, type=MAC_CONTROL_ETHER_TYPE)
bind_layers(Dot1Q, MACControl, type=MAC_CONTROL_ETHER_TYPE)