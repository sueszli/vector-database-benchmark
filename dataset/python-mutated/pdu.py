from typing import Tuple, Optional
from scapy.layers.inet import UDP
from scapy.fields import IntField, XIntField, PacketListField
from scapy.packet import Packet, bind_bottom_up

class PDU(Packet):
    """
    Single PDU Packet inside PDUTransport list.
    Contains ID and payload length, and later - raw load.
    It's free to interpret using bind_layers/bind_bottom_up method

    Based off this document:

    https://www.autosar.org/fileadmin/standards/classic/22-11/AUTOSAR_SWS_IPDUMultiplexer.pdf # noqa: E501
    """
    name = 'PDU'
    fields_desc = [XIntField('pdu_id', 0), IntField('pdu_payload_len', 0)]

    def extract_padding(self, s):
        if False:
            for i in range(10):
                print('nop')
        return (s[:self.pdu_payload_len], s[self.pdu_payload_len:])

class PDUTransport(Packet):
    """
    Packet representing PDUTransport containing multiple PDUs
    FIXME: Support CAN messages as well.
    """
    name = 'PDUTransport'
    fields_desc = [PacketListField('pdus', [PDU()], PDU)]
bind_bottom_up(UDP, PDUTransport, dport=60000)