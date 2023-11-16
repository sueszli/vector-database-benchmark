"""
    CLNS Extension
    ~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2014, 2015 BENOCS GmbH, Berlin (Germany)
    :author:    Marcel Patzlaff, mpatzlaff@benocs.com

    :description:

        This module provides a registration function and a generic PDU
        for OSI Connectionless-mode Network Services (such as IS-IS).
"""
from scapy.config import conf
from scapy.fields import ByteEnumField, PacketField
from scapy.layers.l2 import LLC
from scapy.packet import Packet, bind_top_down, bind_bottom_up
from scapy.compat import orb
network_layer_protocol_ids = {0: 'Null', 8: 'Q.933', 128: 'IEEE SNAP', 129: 'ISO 8438 CLNP', 130: 'ISO 9542 ES-IS', 131: 'ISO 10589 IS-IS', 142: 'IPv6', 176: 'FRF.9', 177: 'FRF.12', 192: 'TRILL', 193: 'IEEE 802.aq', 204: 'IPv4', 207: 'PPP'}
_cln_protocols = {}

class _GenericClnsPdu(Packet):
    name = 'Generic CLNS PDU'
    fields_desc = [ByteEnumField('nlpid', 0, network_layer_protocol_ids), PacketField('rawdata', None, conf.raw_layer)]

def _create_cln_pdu(s, **kwargs):
    if False:
        return 10
    pdu_cls = conf.raw_layer
    if len(s) >= 1:
        nlpid = orb(s[0])
        pdu_cls = _cln_protocols.get(nlpid, _GenericClnsPdu)
    return pdu_cls(s, **kwargs)

@conf.commands.register
def register_cln_protocol(nlpid, cln_protocol_class):
    if False:
        for i in range(10):
            print('nop')
    if nlpid is None or cln_protocol_class is None:
        return
    chk = _cln_protocols.get(nlpid, None)
    if chk is not None and chk != cln_protocol_class:
        raise ValueError('different protocol already registered!')
    _cln_protocols[nlpid] = cln_protocol_class
    bind_top_down(LLC, cln_protocol_class, dsap=254, ssap=254, ctrl=3)
bind_top_down(LLC, _GenericClnsPdu, dsap=254, ssap=254, ctrl=3)
bind_bottom_up(LLC, _create_cln_pdu, dsap=254, ssap=254, ctrl=3)