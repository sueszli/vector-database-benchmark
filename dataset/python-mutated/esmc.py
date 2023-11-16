from scapy.packet import Packet, bind_layers
from scapy.fields import BitField, ByteField, XByteField, ShortField, XStrFixedLenField
from scapy.contrib.slowprot import SlowProtocol
from scapy.compat import orb

class ESMC(Packet):
    name = 'ESMC'
    fields_desc = [XStrFixedLenField('ituOui', b'\x00\x19\xa7', 3), ShortField('ituSubtype', 1), BitField('version', 1, 4), BitField('event', 0, 1), BitField('reserved1', 0, 3), XStrFixedLenField('reserved2', b'\x00' * 3, 3)]

    def guess_payload_class(self, payload):
        if False:
            for i in range(10):
                print('nop')
        if orb(payload[0]) == 1:
            return QLTLV
        if orb(payload[0]) == 2:
            return EQLTLV
        return Packet.guess_payload_class(self, payload)

class QLTLV(ESMC):
    name = 'QLTLV'
    fields_desc = [ByteField('type', 1), ShortField('length', 4), XByteField('ssmCode', 15)]

class EQLTLV(ESMC):
    name = 'EQLTLV'
    fields_desc = [ByteField('type', 2), ShortField('length', 20), XByteField('enhancedSsmCode', 255), XStrFixedLenField('clockIdentity', b'\x00' * 8, 8), ByteField('flag', 0), ByteField('cascaded_eEEcs', 1), ByteField('cascaded_EEcs', 0), XStrFixedLenField('reserved', b'\x00' * 5, 5)]
bind_layers(SlowProtocol, ESMC, subtype=10)