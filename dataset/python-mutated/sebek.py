"""
Sebek: kernel module for data collection on honeypots.
"""
from scapy.fields import FieldLenField, IPField, IntField, ShortEnumField, ShortField, StrFixedLenField, StrLenField, XIntField, ByteEnumField
from scapy.packet import Packet, bind_layers
from scapy.layers.inet import UDP
from scapy.data import IP_PROTOS

class SebekHead(Packet):
    name = 'Sebek header'
    fields_desc = [XIntField('magic', 13684944), ShortField('version', 1), ShortEnumField('type', 0, {'read': 0, 'write': 1, 'socket': 2, 'open': 3}), IntField('counter', 0), IntField('time_sec', 0), IntField('time_usec', 0)]

    def mysummary(self):
        if False:
            print('Hello World!')
        return self.sprintf('Sebek Header v%SebekHead.version% %SebekHead.type%')

class SebekV1(Packet):
    name = 'Sebek v1'
    fields_desc = [IntField('pid', 0), IntField('uid', 0), IntField('fd', 0), StrFixedLenField('cmd', '', 12), FieldLenField('data_length', None, 'data', fmt='I'), StrLenField('data', '', length_from=lambda x: x.data_length)]

    def mysummary(self):
        if False:
            return 10
        if isinstance(self.underlayer, SebekHead):
            return self.underlayer.sprintf('Sebek v1 %SebekHead.type% (%SebekV1.cmd%)')
        else:
            return self.sprintf('Sebek v1 (%SebekV1.cmd%)')

class SebekV3(Packet):
    name = 'Sebek v3'
    fields_desc = [IntField('parent_pid', 0), IntField('pid', 0), IntField('uid', 0), IntField('fd', 0), IntField('inode', 0), StrFixedLenField('cmd', '', 12), FieldLenField('data_length', None, 'data', fmt='I'), StrLenField('data', '', length_from=lambda x: x.data_length)]

    def mysummary(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self.underlayer, SebekHead):
            return self.underlayer.sprintf('Sebek v%SebekHead.version% %SebekHead.type% (%SebekV3.cmd%)')
        else:
            return self.sprintf('Sebek v3 (%SebekV3.cmd%)')

class SebekV2(SebekV3):

    def mysummary(self):
        if False:
            return 10
        if isinstance(self.underlayer, SebekHead):
            return self.underlayer.sprintf('Sebek v%SebekHead.version% %SebekHead.type% (%SebekV2.cmd%)')
        else:
            return self.sprintf('Sebek v2 (%SebekV2.cmd%)')

class SebekV3Sock(Packet):
    name = 'Sebek v2 socket'
    fields_desc = [IntField('parent_pid', 0), IntField('pid', 0), IntField('uid', 0), IntField('fd', 0), IntField('inode', 0), StrFixedLenField('cmd', '', 12), IntField('data_length', 15), IPField('dip', '127.0.0.1'), ShortField('dport', 0), IPField('sip', '127.0.0.1'), ShortField('sport', 0), ShortEnumField('call', 0, {'bind': 2, 'connect': 3, 'listen': 4, 'accept': 5, 'sendmsg': 16, 'recvmsg': 17, 'sendto': 11, 'recvfrom': 12}), ByteEnumField('proto', 0, IP_PROTOS)]

    def mysummary(self):
        if False:
            print('Hello World!')
        if isinstance(self.underlayer, SebekHead):
            return self.underlayer.sprintf('Sebek v%SebekHead.version% %SebekHead.type% (%SebekV3Sock.cmd%)')
        else:
            return self.sprintf('Sebek v3 socket (%SebekV3Sock.cmd%)')

class SebekV2Sock(SebekV3Sock):

    def mysummary(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self.underlayer, SebekHead):
            return self.underlayer.sprintf('Sebek v%SebekHead.version% %SebekHead.type% (%SebekV2Sock.cmd%)')
        else:
            return self.sprintf('Sebek v2 socket (%SebekV2Sock.cmd%)')
bind_layers(UDP, SebekHead, sport=1101)
bind_layers(UDP, SebekHead, dport=1101)
bind_layers(UDP, SebekHead, dport=1101, sport=1101)
bind_layers(SebekHead, SebekV1, version=1)
bind_layers(SebekHead, SebekV2Sock, version=2, type=2)
bind_layers(SebekHead, SebekV2, version=2)
bind_layers(SebekHead, SebekV3Sock, version=3, type=2)
bind_layers(SebekHead, SebekV3, version=3)