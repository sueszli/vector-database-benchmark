"""
ERSPAN - Encapsulated Remote SPAN

https://datatracker.ietf.org/doc/html/draft-foschiano-erspan-03
"""
from scapy.packet import Packet, bind_layers
from scapy.fields import BitField, BitEnumField, XIntField, XShortField
from scapy.layers.l2 import Ether, GRE

class ERSPAN(Packet):
    """
    A generic ERSPAN packet
    """
    name = 'ERSPAN'
    fields_desc = []

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            while True:
                i = 10
        if _pkt:
            ver = _pkt[0] >> 4
            if ver == 1:
                return ERSPAN_II
            elif ver == 2:
                return ERSPAN_III
            else:
                return ERSPAN_I
        if cls == ERSPAN:
            return ERSPAN_II
        return cls

class ERSPAN_I(ERSPAN):
    name = 'ERSPAN I'
    match_subclass = True
    fields_desc = []

class ERSPAN_II(ERSPAN):
    name = 'ERSPAN II'
    match_subclass = True
    fields_desc = [BitField('ver', 1, 4), BitField('vlan', 0, 12), BitField('cos', 0, 3), BitField('en', 0, 2), BitField('t', 0, 1), BitField('session_id', 0, 10), BitField('reserved', 0, 12), BitField('index', 0, 20)]

class ERSPAN_III(ERSPAN):
    name = 'ERSPAN III'
    match_subclass = True
    fields_desc = [BitField('ver', 2, 4), BitField('vlan', 0, 12), BitField('cos', 0, 3), BitField('bso', 0, 2), BitField('t', 0, 1), BitField('session_id', 0, 10), XIntField('timestamp', 0), XShortField('sgt_other', 0), BitField('p', 0, 1), BitEnumField('ft', 0, 5, {0: 'Ethernet', 2: 'IP'}), BitField('hw', 0, 6), BitField('d', 0, 1), BitEnumField('gra', 0, 2, {0: '100us', 1: '100ns', 2: 'IEEE 1588'}), BitField('o', 0, 1)]

class ERSPAN_PlatformSpecific(Packet):
    name = 'PlatformSpecific'
    fields_desc = [BitField('platf_id', 0, 6), BitField('info1', 0, 26), XIntField('info2', 0)]
bind_layers(ERSPAN_I, Ether)
bind_layers(ERSPAN_II, Ether)
bind_layers(ERSPAN_III, Ether, o=0)
bind_layers(ERSPAN_III, ERSPAN_PlatformSpecific, o=1)
bind_layers(ERSPAN_PlatformSpecific, Ether)
bind_layers(GRE, ERSPAN, proto=35006, seqnum_present=0)
bind_layers(GRE, ERSPAN_II, proto=35006, seqnum_present=1)
bind_layers(GRE, ERSPAN_III, proto=8939)