from scapy.packet import Packet, bind_layers
from scapy.fields import BitField, ByteEnumField, ByteField, FieldLenField, FieldListField, IPField, PacketListField, ShortField, XShortField
from scapy.compat import orb
from scapy.layers.inet import IP
from scapy.contrib.igmp import IGMP
from scapy.config import conf
' Based on the following references\n http://www.iana.org/assignments/igmp-type-numbers\n http://www.rfc-editor.org/rfc/pdfrfc/rfc3376.txt.pdf\n\n'

class IGMPv3(IGMP):
    """IGMP Message Class for v3.

    This class is derived from class Packet.
    The fields defined below are a
    direct interpretation of the v3 Membership Query Message.
    Fields 'type'  through 'qqic' are directly assignable.
    For 'numsrc', do not assign a value.
    Instead add to the 'srcaddrs' list to auto-set 'numsrc'. To
    assign values to 'srcaddrs', use the following methods::

      c = IGMPv3()
      c.srcaddrs = ['1.2.3.4', '5.6.7.8']
      c.srcaddrs += ['192.168.10.24']

    At this point, 'c.numsrc' is three (3)

    'chksum' is automagically calculated before the packet is sent.

    'mrcode' is also the Advertisement Interval field

    """
    name = 'IGMPv3'
    igmpv3types = {17: 'Membership Query', 34: 'Version 3 Membership Report', 48: 'Multicast Router Advertisement', 49: 'Multicast Router Solicitation', 50: 'Multicast Router Termination'}
    fields_desc = [ByteEnumField('type', 17, igmpv3types), ByteField('mrcode', 20), XShortField('chksum', None)]

    def encode_maxrespcode(self):
        if False:
            print('Hello World!')
        'Encode and replace the mrcode value to its IGMPv3 encoded time value if needed,  # noqa: E501\n        as specified in rfc3376#section-4.1.1.\n\n        If value < 128, return the value specified. If >= 128, encode as a floating  # noqa: E501\n        point value. Value can be 0 - 31744.\n        '
        value = self.mrcode
        if value < 128:
            code = value
        elif value > 31743:
            code = 255
        else:
            exp = 0
            value >>= 3
            while value > 31:
                exp += 1
                value >>= 1
            exp <<= 4
            code = 128 | exp | value & 15
        self.mrcode = code

    def mysummary(self):
        if False:
            print('Hello World!')
        'Display a summary of the IGMPv3 object.'
        if isinstance(self.underlayer, IP):
            return self.underlayer.sprintf('IGMPv3: %IP.src% > %IP.dst% %IGMPv3.type%')
        else:
            return self.sprintf('IGMPv3 %IGMPv3.type%')

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            print('Hello World!')
        if _pkt and len(_pkt) >= 4:
            if orb(_pkt[0]) in [18, 22, 23]:
                return IGMP
            elif orb(_pkt[0]) == 17 and len(_pkt) < 12:
                return IGMP
        return IGMPv3

class IGMPv3mq(Packet):
    """IGMPv3 Membership Query.
    Payload of IGMPv3 when type=0x11"""
    name = 'IGMPv3mq'
    fields_desc = [IPField('gaddr', '0.0.0.0'), BitField('resv', 0, 4), BitField('s', 0, 1), BitField('qrv', 0, 3), ByteField('qqic', 0), FieldLenField('numsrc', None, count_of='srcaddrs'), FieldListField('srcaddrs', None, IPField('sa', '0.0.0.0'), count_from=lambda x: x.numsrc)]

class IGMPv3gr(Packet):
    """IGMP Group Record for IGMPv3 Membership Report

    This class is derived from class Packet and should be added in the records
    of an instantiation of class IGMPv3mr.
    """
    name = 'IGMPv3gr'
    igmpv3grtypes = {1: 'Mode Is Include', 2: 'Mode Is Exclude', 3: 'Change To Include Mode', 4: 'Change To Exclude Mode', 5: 'Allow New Sources', 6: 'Block Old Sources'}
    fields_desc = [ByteEnumField('rtype', 1, igmpv3grtypes), ByteField('auxdlen', 0), FieldLenField('numsrc', None, count_of='srcaddrs'), IPField('maddr', '0.0.0.0'), FieldListField('srcaddrs', [], IPField('sa', '0.0.0.0'), count_from=lambda x: x.numsrc)]

    def mysummary(self):
        if False:
            while True:
                i = 10
        'Display a summary of the IGMPv3 group record.'
        return self.sprintf('IGMPv3 Group Record %IGMPv3gr.type% %IGMPv3gr.maddr%')

    def default_payload_class(self, payload):
        if False:
            print('Hello World!')
        return conf.padding_layer

class IGMPv3mr(Packet):
    """IGMP Membership Report extension for IGMPv3.
    Payload of IGMPv3 when type=0x22"""
    name = 'IGMPv3mr'
    fields_desc = [XShortField('res2', 0), FieldLenField('numgrp', None, count_of='records'), PacketListField('records', [], IGMPv3gr, count_from=lambda x: x.numgrp)]

class IGMPv3mra(Packet):
    """IGMP Multicast Router Advertisement extension for IGMPv3.
    Payload of IGMPv3 when type=0x30"""
    name = 'IGMPv3mra'
    fields_desc = [ShortField('qryIntvl', 0), ShortField('robust', 0)]
bind_layers(IP, IGMPv3, frag=0, proto=2, ttl=1, tos=192, dst='224.0.0.22')
bind_layers(IGMPv3, IGMPv3mq, type=17)
bind_layers(IGMPv3, IGMPv3mr, type=34, mrcode=0)
bind_layers(IGMPv3, IGMPv3mra, type=48)