"""
    DTP Scapy Extension
    ~~~~~~~~~~~~~~~~~~~

    :version: 2008-12-22
    :author: Jochen Bartl <lobo@c3a.de>

    :Thanks:

    - TLV code derived from the CDP implementation of scapy. (Thanks to Nicolas Bareil and Arnaud Ebalard)  # noqa: E501
"""
import struct
from scapy.packet import Packet, bind_layers
from scapy.fields import ByteField, FieldLenField, MACField, PacketListField, ShortField, StrLenField, XShortField
from scapy.layers.l2 import SNAP, Dot3, LLC
from scapy.sendrecv import sendp
from scapy.config import conf
from scapy.volatile import RandMAC

class DtpGenericTlv(Packet):
    name = 'DTP Generic TLV'
    fields_desc = [XShortField('type', 1), FieldLenField('length', None, length_of=lambda pkt: pkt.value + 4), StrLenField('value', '', length_from=lambda pkt: pkt.length - 4)]

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            i = 10
            return i + 15
        if _pkt and len(_pkt) >= 2:
            t = struct.unpack('!H', _pkt[:2])[0]
            cls = _DTP_TLV_CLS.get(t, 'DtpGenericTlv')
        return cls

    def guess_payload_class(self, p):
        if False:
            return 10
        return conf.padding_layer

class DTPDomain(DtpGenericTlv):
    name = 'DTP Domain'
    fields_desc = [ShortField('type', 1), FieldLenField('length', None, 'domain', adjust=lambda pkt, x: x + 4), StrLenField('domain', b'\x00', length_from=lambda pkt: pkt.length - 4)]

class DTPStatus(DtpGenericTlv):
    name = 'DTP Status'
    fields_desc = [ShortField('type', 2), FieldLenField('length', None, 'status', adjust=lambda pkt, x: x + 4), StrLenField('status', b'\x03', length_from=lambda pkt: pkt.length - 4)]

class DTPType(DtpGenericTlv):
    name = 'DTP Type'
    fields_desc = [ShortField('type', 3), FieldLenField('length', None, 'dtptype', adjust=lambda pkt, x: x + 4), StrLenField('dtptype', b'\xa5', length_from=lambda pkt: pkt.length - 4)]

class DTPNeighbor(DtpGenericTlv):
    name = 'DTP Neighbor'
    fields_desc = [ShortField('type', 4), ShortField('len', 10), MACField('neighbor', None)]
_DTP_TLV_CLS = {1: DTPDomain, 2: DTPStatus, 3: DTPType, 4: DTPNeighbor}

class DTP(Packet):
    name = 'DTP'
    fields_desc = [ByteField('ver', 1), PacketListField('tlvlist', [], DtpGenericTlv)]
bind_layers(SNAP, DTP, code=8196, OUI=12)

def negotiate_trunk(iface=conf.iface, mymac=str(RandMAC())):
    if False:
        i = 10
        return i + 15
    print('Trying to negotiate a trunk on interface %s' % iface)
    p = Dot3(src=mymac, dst='01:00:0c:cc:cc:cc') / LLC()
    p /= SNAP()
    p /= DTP(tlvlist=[DTPDomain(), DTPStatus(), DTPType(), DTPNeighbor(neighbor=mymac)])
    sendp(p)