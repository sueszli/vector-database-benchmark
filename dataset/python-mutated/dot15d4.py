"""
Wireless MAC according to IEEE 802.15.4.
"""
import struct
from scapy.compat import orb, chb
from scapy.error import warning
from scapy.config import conf
from scapy.data import DLT_IEEE802_15_4_WITHFCS, DLT_IEEE802_15_4_NOFCS
from scapy.packet import Packet, bind_layers
from scapy.fields import BitEnumField, BitField, ByteEnumField, ByteField, ConditionalField, Emph, FCSField, Field, FieldListField, LELongField, MultipleTypeField, PacketField, StrFixedLenField, XByteField, XLEIntField, XLEShortField

class dot15d4AddressField(Field):
    __slots__ = ['adjust', 'length_of']

    def __init__(self, name, default, length_of=None, fmt='<H', adjust=None):
        if False:
            return 10
        Field.__init__(self, name, default, fmt)
        self.length_of = length_of
        if adjust is not None:
            self.adjust = adjust
        else:
            self.adjust = lambda pkt, x: self.lengthFromAddrMode(pkt, x)

    def i2repr(self, pkt, x):
        if False:
            print('Hello World!')
        'Convert internal value to a nice representation'
        if len(hex(self.i2m(pkt, x))) < 7:
            return hex(self.i2m(pkt, x))
        else:
            x = '%016x' % self.i2m(pkt, x)
            return ':'.join(['%s%s' % (x[i], x[i + 1]) for i in range(0, len(x), 2)])

    def addfield(self, pkt, s, val):
        if False:
            print('Hello World!')
        'Add an internal value to a string'
        if self.adjust(pkt, self.length_of) == 2:
            return s + struct.pack(self.fmt[0] + 'H', val)
        elif self.adjust(pkt, self.length_of) == 8:
            return s + struct.pack(self.fmt[0] + 'Q', val)
        else:
            return s

    def getfield(self, pkt, s):
        if False:
            while True:
                i = 10
        if self.adjust(pkt, self.length_of) == 2:
            return (s[2:], self.m2i(pkt, struct.unpack(self.fmt[0] + 'H', s[:2])[0]))
        elif self.adjust(pkt, self.length_of) == 8:
            return (s[8:], self.m2i(pkt, struct.unpack(self.fmt[0] + 'Q', s[:8])[0]))
        else:
            raise Exception('impossible case')

    def lengthFromAddrMode(self, pkt, x):
        if False:
            i = 10
            return i + 15
        addrmode = 0
        pkttop = pkt.underlayer
        if pkttop is None:
            warning('No underlayer to guess address mode')
            return 0
        while True:
            try:
                addrmode = pkttop.getfieldval(x)
                break
            except Exception:
                if pkttop.underlayer is None:
                    break
                pkttop = pkttop.underlayer
        if addrmode == 2:
            return 2
        elif addrmode == 3:
            return 8
        return 0

class Dot15d4(Packet):
    name = '802.15.4'
    fields_desc = [BitField('fcf_reserved_1', 0, 1), BitEnumField('fcf_panidcompress', 0, 1, [False, True]), BitEnumField('fcf_ackreq', 0, 1, [False, True]), BitEnumField('fcf_pending', 0, 1, [False, True]), BitEnumField('fcf_security', 0, 1, [False, True]), Emph(BitEnumField('fcf_frametype', 0, 3, {0: 'Beacon', 1: 'Data', 2: 'Ack', 3: 'Command'})), BitEnumField('fcf_srcaddrmode', 0, 2, {0: 'None', 1: 'Reserved', 2: 'Short', 3: 'Long'}), BitField('fcf_framever', 0, 2), BitEnumField('fcf_destaddrmode', 2, 2, {0: 'None', 1: 'Reserved', 2: 'Short', 3: 'Long'}), BitField('fcf_reserved_2', 0, 2), Emph(ByteField('seqnum', 1))]

    def mysummary(self):
        if False:
            for i in range(10):
                print('nop')
        return self.sprintf('802.15.4 %Dot15d4.fcf_frametype% ackreq(%Dot15d4.fcf_ackreq%) ( %Dot15d4.fcf_destaddrmode% -> %Dot15d4.fcf_srcaddrmode% ) Seq#%Dot15d4.seqnum%')

    def guess_payload_class(self, payload):
        if False:
            i = 10
            return i + 15
        if self.fcf_frametype == 0:
            return Dot15d4Beacon
        elif self.fcf_frametype == 1:
            return Dot15d4Data
        elif self.fcf_frametype == 2:
            return Dot15d4Ack
        elif self.fcf_frametype == 3:
            return Dot15d4Cmd
        else:
            return Packet.guess_payload_class(self, payload)

    def answers(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, Dot15d4):
            if self.fcf_frametype == 2:
                if self.seqnum != other.seqnum:
                    return 0
                elif other.fcf_ackreq == 1:
                    return 1
        return 0

    def post_build(self, p, pay):
        if False:
            return 10
        if self.fcf_frametype == 2 and self.fcf_destaddrmode != 0:
            self.fcf_destaddrmode = 0
            return p[:1] + chb((self.fcf_srcaddrmode << 6) + (self.fcf_framever << 4)) + p[2:] + pay
        else:
            return p + pay

class Dot15d4FCS(Dot15d4):
    """
    This class is a drop-in replacement for the Dot15d4 class above, except
    it expects a FCS/checksum in the input, and produces one in the output.
    This provides the user flexibility, as many 802.15.4 interfaces will have an AUTO_CRC setting  # noqa: E501
    that will validate the FCS/CRC in firmware, and add it automatically when transmitting.  # noqa: E501
    """
    name = '802.15.4 - FCS'
    match_subclass = True
    fields_desc = Dot15d4.fields_desc + [FCSField('fcs', None, fmt='<H')]

    def compute_fcs(self, data):
        if False:
            while True:
                i = 10
        crc = 0
        for i in range(0, len(data)):
            c = orb(data[i])
            q = (crc ^ c) & 15
            crc = crc // 16 ^ q * 4225
            q = (crc ^ c // 16) & 15
            crc = crc // 16 ^ q * 4225
        return struct.pack('<H', crc)

    def post_build(self, p, pay):
        if False:
            for i in range(10):
                print('nop')
        p = Dot15d4.post_build(self, p, pay)
        if self.fcs is None:
            p = p[:-2]
            p = p + self.compute_fcs(p)
        return p

class Dot15d4Ack(Packet):
    name = '802.15.4 Ack'
    fields_desc = []

class Dot15d4AuxSecurityHeader(Packet):
    name = '802.15.4 Auxiliary Security Header'
    fields_desc = [BitField('sec_sc_reserved', 0, 3), BitEnumField('sec_sc_keyidmode', 0, 2, {0: 'Implicit', 1: '1oKeyIndex', 2: '4o-KeySource-1oKeyIndex', 3: '8o-KeySource-1oKeyIndex'}), BitEnumField('sec_sc_seclevel', 0, 3, {0: 'None', 1: 'MIC-32', 2: 'MIC-64', 3: 'MIC-128', 4: 'ENC', 5: 'ENC-MIC-32', 6: 'ENC-MIC-64', 7: 'ENC-MIC-128'}), XLEIntField('sec_framecounter', 0), MultipleTypeField([(XLEIntField('sec_keyid_keysource', 0), lambda pkt: pkt.getfieldval('sec_sc_keyidmode') == 2), (LELongField('sec_keyid_keysource', 0), lambda pkt: pkt.getfieldval('sec_sc_keyidmode') == 3)], StrFixedLenField('sec_keyid_keysource', '', length=0)), ConditionalField(XByteField('sec_keyid_keyindex', 255), lambda pkt: pkt.getfieldval('sec_sc_keyidmode') != 0)]

class Dot15d4Data(Packet):
    name = '802.15.4 Data'
    fields_desc = [XLEShortField('dest_panid', 65535), dot15d4AddressField('dest_addr', 65535, length_of='fcf_destaddrmode'), ConditionalField(XLEShortField('src_panid', 0), lambda pkt: util_srcpanid_present(pkt)), ConditionalField(dot15d4AddressField('src_addr', None, length_of='fcf_srcaddrmode'), lambda pkt: pkt.underlayer.getfieldval('fcf_srcaddrmode') != 0), ConditionalField(PacketField('aux_sec_header', Dot15d4AuxSecurityHeader(), Dot15d4AuxSecurityHeader), lambda pkt: pkt.underlayer.getfieldval('fcf_security') is True)]

    def guess_payload_class(self, payload):
        if False:
            return 10
        from scapy.layers.sixlowpan import SixLoWPAN
        from scapy.layers.zigbee import ZigbeeNWK
        if conf.dot15d4_protocol == 'sixlowpan':
            return SixLoWPAN
        elif conf.dot15d4_protocol == 'zigbee':
            return ZigbeeNWK
        else:
            if conf.dot15d4_protocol is None:
                _msg = 'Please set conf.dot15d4_protocol to select a ' + '802.15.4 protocol. Values must be in the list: '
            else:
                _msg = 'Unknown conf.dot15d4_protocol value: must be in '
            warning(_msg + "['sixlowpan', 'zigbee']" + ' Defaulting to SixLoWPAN')
            return SixLoWPAN

    def mysummary(self):
        if False:
            return 10
        return self.sprintf('802.15.4 Data ( %Dot15d4Data.src_panid%:%Dot15d4Data.src_addr% -> %Dot15d4Data.dest_panid%:%Dot15d4Data.dest_addr% )')

class Dot15d4Beacon(Packet):
    name = '802.15.4 Beacon'
    fields_desc = [XLEShortField('src_panid', 0), dot15d4AddressField('src_addr', None, length_of='fcf_srcaddrmode'), ConditionalField(PacketField('aux_sec_header', Dot15d4AuxSecurityHeader(), Dot15d4AuxSecurityHeader), lambda pkt: pkt.underlayer.getfieldval('fcf_security') is True), BitField('sf_sforder', 15, 4), BitField('sf_beaconorder', 15, 4), BitEnumField('sf_assocpermit', 0, 1, [False, True]), BitEnumField('sf_pancoord', 0, 1, [False, True]), BitField('sf_reserved', 0, 1), BitEnumField('sf_battlifeextend', 0, 1, [False, True]), BitField('sf_finalcapslot', 15, 4), BitEnumField('gts_spec_permit', 1, 1, [False, True]), BitField('gts_spec_reserved', 0, 4), BitField('gts_spec_desccount', 0, 3), ConditionalField(BitField('gts_dir_reserved', 0, 1), lambda pkt: pkt.getfieldval('gts_spec_desccount') != 0), ConditionalField(BitField('gts_dir_mask', 0, 7), lambda pkt: pkt.getfieldval('gts_spec_desccount') != 0), BitField('pa_reserved_1', 0, 1), BitField('pa_num_long', 0, 3), BitField('pa_reserved_2', 0, 1), BitField('pa_num_short', 0, 3), FieldListField('pa_short_addresses', [], XLEShortField('', 0), count_from=lambda pkt: pkt.pa_num_short), FieldListField('pa_long_addresses', [], dot15d4AddressField('', 0, adjust=lambda pkt, x: 8), count_from=lambda pkt: pkt.pa_num_long)]

    def mysummary(self):
        if False:
            print('Hello World!')
        return self.sprintf('802.15.4 Beacon ( %Dot15d4Beacon.src_panid%:%Dot15d4Beacon.src_addr% ) assocPermit(%Dot15d4Beacon.sf_assocpermit%) panCoord(%Dot15d4Beacon.sf_pancoord%)')

class Dot15d4Cmd(Packet):
    name = '802.15.4 Command'
    fields_desc = [XLEShortField('dest_panid', 65535), dot15d4AddressField('dest_addr', 0, length_of='fcf_destaddrmode'), ConditionalField(XLEShortField('src_panid', 0), lambda pkt: util_srcpanid_present(pkt)), ConditionalField(dot15d4AddressField('src_addr', None, length_of='fcf_srcaddrmode'), lambda pkt: pkt.underlayer.getfieldval('fcf_srcaddrmode') != 0), ConditionalField(PacketField('aux_sec_header', Dot15d4AuxSecurityHeader(), Dot15d4AuxSecurityHeader), lambda pkt: pkt.underlayer.getfieldval('fcf_security') is True), ByteEnumField('cmd_id', 0, {1: 'AssocReq', 2: 'AssocResp', 3: 'DisassocNotify', 4: 'DataReq', 5: 'PANIDConflictNotify', 6: 'OrphanNotify', 7: 'BeaconReq', 8: 'CoordRealign', 9: 'GTSReq'})]

    def mysummary(self):
        if False:
            print('Hello World!')
        return self.sprintf('802.15.4 Command %Dot15d4Cmd.cmd_id% ( %Dot15dCmd.src_panid%:%Dot15d4Cmd.src_addr% -> %Dot15d4Cmd.dest_panid%:%Dot15d4Cmd.dest_addr% )')

    def guess_payload_class(self, payload):
        if False:
            while True:
                i = 10
        if self.cmd_id == 1:
            return Dot15d4CmdAssocReq
        elif self.cmd_id == 2:
            return Dot15d4CmdAssocResp
        elif self.cmd_id == 3:
            return Dot15d4CmdDisassociation
        elif self.cmd_id == 8:
            return Dot15d4CmdCoordRealign
        elif self.cmd_id == 9:
            return Dot15d4CmdGTSReq
        else:
            return Packet.guess_payload_class(self, payload)

class Dot15d4CmdCoordRealign(Packet):
    name = '802.15.4 Coordinator Realign Command'
    fields_desc = [XLEShortField('panid', 65535), XLEShortField('coord_address', 0), ByteField('channel', 0), XLEShortField('dev_address', 65535)]

    def mysummary(self):
        if False:
            i = 10
            return i + 15
        return self.sprintf('802.15.4 Coordinator Realign Payload ( PAN ID: %Dot15dCmdCoordRealign.pan_id% : channel %Dot15d4CmdCoordRealign.channel% )')

    def guess_payload_class(self, payload):
        if False:
            for i in range(10):
                print('nop')
        if len(payload) == 1:
            return Dot15d4CmdCoordRealignPage
        else:
            return Packet.guess_payload_class(self, payload)

class Dot15d4CmdCoordRealignPage(Packet):
    name = '802.15.4 Coordinator Realign Page'
    fields_desc = [ByteField('channel_page', 0)]

def util_srcpanid_present(pkt):
    if False:
        while True:
            i = 10
    'A source PAN ID is included if and only if both src addr mode != 0 and PAN ID Compression in FCF == 0'
    if pkt.underlayer.getfieldval('fcf_srcaddrmode') != 0 and pkt.underlayer.getfieldval('fcf_panidcompress') == 0:
        return True
    else:
        return False

class Dot15d4CmdAssocReq(Packet):
    name = '802.15.4 Association Request Payload'
    fields_desc = [BitField('allocate_address', 0, 1), BitField('security_capability', 0, 1), BitField('reserved2', 0, 1), BitField('reserved1', 0, 1), BitField('receiver_on_when_idle', 0, 1), BitField('power_source', 0, 1), BitField('device_type', 0, 1), BitField('alternate_pan_coordinator', 0, 1)]

    def mysummary(self):
        if False:
            return 10
        return self.sprintf('802.15.4 Association Request Payload ( Alt PAN Coord: %Dot15d4CmdAssocReq.alternate_pan_coordinator% Device Type: %Dot15d4CmdAssocReq.device_type% )')

class Dot15d4CmdAssocResp(Packet):
    name = '802.15.4 Association Response Payload'
    fields_desc = [XLEShortField('short_address', 65535), ByteEnumField('association_status', 0, {0: 'successful', 1: 'PAN_at_capacity', 2: 'PAN_access_denied'})]

    def mysummary(self):
        if False:
            print('Hello World!')
        return self.sprintf('802.15.4 Association Response Payload ( Association Status: %Dot15d4CmdAssocResp.association_status% Assigned Address: %Dot15d4CmdAssocResp.short_address% )')

class Dot15d4CmdDisassociation(Packet):
    name = '802.15.4 Disassociation Notification Payload'
    fields_desc = [ByteEnumField('disassociation_reason', 2, {1: 'coord_wishes_device_to_leave', 2: 'device_wishes_to_leave'})]

    def mysummary(self):
        if False:
            print('Hello World!')
        return self.sprintf('802.15.4 Disassociation Notification Payload ( Disassociation Reason %Dot15d4CmdDisassociation.disassociation_reason% )')

class Dot15d4CmdGTSReq(Packet):
    name = '802.15.4 GTS request command'
    fields_desc = [BitField('reserved', 0, 2), BitField('charact_type', 0, 1), BitField('gts_dir', 0, 1), BitField('gts_len', 0, 4)]

    def mysummary(self):
        if False:
            while True:
                i = 10
        return self.sprintf('802.15.4 GTS Request Command ( %Dot15d4CmdGTSReq.gts_len% : %Dot15d4CmdGTSReq.gts_dir% )')
bind_layers(Dot15d4, Dot15d4Beacon, fcf_frametype=0)
bind_layers(Dot15d4, Dot15d4Data, fcf_frametype=1)
bind_layers(Dot15d4, Dot15d4Ack, fcf_frametype=2)
bind_layers(Dot15d4, Dot15d4Cmd, fcf_frametype=3)
conf.l2types.register(DLT_IEEE802_15_4_WITHFCS, Dot15d4FCS)
conf.l2types.register(DLT_IEEE802_15_4_NOFCS, Dot15d4)