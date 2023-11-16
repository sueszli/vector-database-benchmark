import copy
from scapy.compat import raw
from scapy.error import Scapy_Exception
from scapy.config import conf
from scapy.packet import Packet, bind_layers
from scapy.layers.l2 import Ether
from scapy.layers.inet import UDP
from scapy.fields import XShortEnumField, BitEnumField, XBitField, BitField, StrField, PacketListField, StrFixedLenField, ShortField, FlagsField, ByteField, XIntField, X3BytesField
PNIO_FRAME_IDS = {32: 'PTCP-RTSyncPDU-followup', 128: 'PTCP-RTSyncPDU', 64513: 'Alarm High', 65025: 'Alarm Low', 65276: 'DCP-Hello-Req', 65277: 'DCP-Get-Set', 65278: 'DCP-Identify-ReqPDU', 65279: 'DCP-Identify-ResPDU', 65280: 'PTCP-AnnouncePDU', 65312: 'PTCP-FollowUpPDU', 65344: 'PTCP-DelayReqPDU', 65345: 'PTCP-DelayResPDU-followup', 65346: 'PTCP-DelayFuResPDU', 65347: 'PTCP-DelayResPDU'}

def i2s_frameid(x):
    if False:
        while True:
            i = 10
    ' Get representation name of a pnio frame ID\n\n    :param x: a key of the PNIO_FRAME_IDS dictionary\n    :returns: str\n    '
    try:
        return PNIO_FRAME_IDS[x]
    except KeyError:
        pass
    if 256 <= x < 4096:
        return 'RT_CLASS_3 (%4x)' % x
    if 32768 <= x < 49152:
        return 'RT_CLASS_1 (%4x)' % x
    if 49152 <= x < 64512:
        return 'RT_CLASS_UDP (%4x)' % x
    if 65408 <= x < 65424:
        return 'FragmentationFrameID (%4x)' % x
    return x

def s2i_frameid(x):
    if False:
        return 10
    ' Get pnio frame ID from a representation name\n\n    Performs a reverse look-up in PNIO_FRAME_IDS dictionary\n\n    :param x: a value of PNIO_FRAME_IDS dict\n    :returns: integer\n    '
    try:
        return {'RT_CLASS_3': 256, 'RT_CLASS_1': 32768, 'RT_CLASS_UDP': 49152, 'FragmentationFrameID': 65408}[x]
    except KeyError:
        pass
    try:
        return next((key for (key, value) in PNIO_FRAME_IDS.items() if value == x))
    except StopIteration:
        pass
    return x

class ProfinetIO(Packet):
    """ Basic PROFINET IO dispatcher """
    fields_desc = [XShortEnumField('frameID', 0, (i2s_frameid, s2i_frameid))]

    def guess_payload_class(self, payload):
        if False:
            i = 10
            return i + 15
        if self.frameID in [65278, 65279, 65277]:
            from scapy.contrib.pnio_dcp import ProfinetDCP
            return ProfinetDCP
        elif self.frameID == 65025:
            from scapy.contrib.pnio_rpc import Alarm_Low
            return Alarm_Low
        elif self.frameID == 64513:
            from scapy.contrib.pnio_rpc import Alarm_High
            return Alarm_High
        elif 256 <= self.frameID < 4096 or 32768 <= self.frameID < 64512:
            return PNIORealTimeCyclicPDU
        return super(ProfinetIO, self).guess_payload_class(payload)
bind_layers(Ether, ProfinetIO, type=34962)
bind_layers(UDP, ProfinetIO, dport=34962)
conf.contribs['PNIO_RTC'] = {}

class PNIORealTime_IOxS(Packet):
    """ IOCS and IOPS packets for PROFINET Real-Time payload """
    name = 'PNIO RTC IOxS'
    fields_desc = [BitEnumField('dataState', 1, 1, ['bad', 'good']), BitEnumField('instance', 0, 2, ['subslot', 'slot', 'device', 'controller']), XBitField('reserved', 0, 4), BitField('extension', 0, 1)]

    @classmethod
    def is_extension_set(cls, _pkt, _lst, p, _remain):
        if False:
            return 10
        ret = cls if isinstance(p, type(None)) or p.extension != 0 else None
        return ret

    @classmethod
    def get_len(cls):
        if False:
            print('Hello World!')
        return sum((type(fld).i2len(None, 0) for fld in cls.fields_desc))

    def guess_payload_class(self, p):
        if False:
            for i in range(10):
                print('nop')
        return conf.padding_layer

class PNIORealTimeCyclicDefaultRawData(Packet):
    name = 'PROFINET IO Real Time Cyclic Default Raw Data'
    fields_desc = [StrField('data', '', remain=4)]

    def guess_payload_class(self, payload):
        if False:
            while True:
                i = 10
        return conf.padding_layer

class PNIORealTimeCyclicPDU(Packet):
    """ PROFINET cyclic real-time """
    __slots__ = ['_len', '_layout']
    name = 'PROFINET Real-Time'
    fields_desc = [PacketListField('data', [], next_cls_cb=lambda pkt, lst, p, remain: pkt.next_cls_cb(lst, p, remain)), StrFixedLenField('padding', '', length_from=lambda p: p.get_padding_length()), ShortField('cycleCounter', 0), FlagsField('dataStatus', 53, 8, ['primary', 'redundancy', 'validData', 'reserved_1', 'run', 'no_problem', 'reserved_2', 'ignore']), ByteField('transferStatus', 0)]

    def pre_dissect(self, s):
        if False:
            while True:
                i = 10
        self._len = min(1440, len(s))
        return s

    def get_padding_length(self):
        if False:
            return 10
        if hasattr(self, '_len'):
            pad_len = self._len - sum((len(raw(pkt)) for pkt in self.getfieldval('data'))) - 2 - 1 - 1
        else:
            pad_len = len(self.getfieldval('padding'))
        assert 0 <= pad_len <= 40
        q = self
        while not isinstance(q, UDP) and hasattr(q, 'underlayer'):
            q = q.underlayer
        if isinstance(q, UDP):
            assert 0 <= pad_len <= 12
        return pad_len

    def next_cls_cb(self, _lst, _p, _remain):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, '_layout') and isinstance(self._layout, list):
            try:
                return self._layout.pop(0)
            except IndexError:
                self._layout = None
                return None
        ether_layer = None
        q = self
        while not isinstance(q, Ether) and hasattr(q, 'underlayer'):
            q = q.underlayer
        if isinstance(q, Ether):
            ether_layer = q
        pnio_layer = None
        q = self
        while not isinstance(q, ProfinetIO) and hasattr(q, 'underlayer'):
            q = q.underlayer
        if isinstance(q, ProfinetIO):
            pnio_layer = q
        self._layout = [PNIORealTimeCyclicDefaultRawData]
        if not (ether_layer is None and pnio_layer is None):
            layout = type(self).get_layout_from_config(ether_layer.src, ether_layer.dst, pnio_layer.frameID)
            if not isinstance(layout, type(None)):
                self._layout = layout
        return self._layout.pop(0)

    @staticmethod
    def get_layout_from_config(ether_src, ether_dst, frame_id):
        if False:
            i = 10
            return i + 15
        try:
            return copy.deepcopy(conf.contribs['PNIO_RTC'][ether_src, ether_dst, frame_id])
        except KeyError:
            return None

    @staticmethod
    def build_fixed_len_raw_type(length):
        if False:
            while True:
                i = 10
        return type('FixedLenRawPacketLen{}'.format(length), (conf.raw_layer,), {'name': 'FixedLenRawPacketLen{}'.format(length), 'fields_desc': [StrFixedLenField('data', '', length=length)], 'get_data_length': lambda _: length, 'guess_payload_class': lambda self, p: conf.padding_layer})
profisafe_control_flags = ['iPar_EN', 'OA_Req', 'R_cons_nr', 'Use_TO2', 'activate_FV', 'Toggle_h', 'ChF_Ack', 'Loopcheck']
profisafe_status_flags = ['iPar_OK', 'Device_Fault/ChF_Ack_Req', 'CE_CRC', 'WD_timeout', 'FV_activated', 'Toggle_d', 'cons_nr_R', 'reserved']

class PROFIsafeCRCSeed(Packet):
    __slots__ = ['_len'] + Packet.__slots__

    def guess_payload_class(self, p):
        if False:
            for i in range(10):
                print('nop')
        return conf.padding_layer

    def get_data_length(self):
        if False:
            print('Hello World!')
        ' Must be overridden in a subclass to return the correct value '
        raise Scapy_Exception('This method must be overridden in a specific subclass')

    def get_mandatory_fields_len(self):
        if False:
            return 10
        return 5

    @staticmethod
    def get_max_data_length():
        if False:
            return 10
        return 13

class PROFIsafeControlCRCSeed(PROFIsafeCRCSeed):
    name = 'PROFISafe Control Message with F_CRC_Seed=1'
    fields_desc = [StrFixedLenField('data', '', length_from=lambda p: p.get_data_length()), FlagsField('control', 0, 8, profisafe_control_flags), XIntField('crc', 0)]

class PROFIsafeStatusCRCSeed(PROFIsafeCRCSeed):
    name = 'PROFISafe Status Message with F_CRC_Seed=1'
    fields_desc = [StrFixedLenField('data', '', length_from=lambda p: p.get_data_length()), FlagsField('status', 0, 8, profisafe_status_flags), XIntField('crc', 0)]

class PROFIsafe(Packet):
    __slots__ = ['_len'] + Packet.__slots__

    def guess_payload_class(self, p):
        if False:
            while True:
                i = 10
        return conf.padding_layer

    def get_data_length(self):
        if False:
            for i in range(10):
                print('nop')
        ' Must be overridden in a subclass to return the correct value '
        raise Scapy_Exception('This method must be overridden in a specific subclass')

    def get_mandatory_fields_len(self):
        if False:
            return 10
        return 4

    @staticmethod
    def get_max_data_length():
        if False:
            print('Hello World!')
        return 12

    @staticmethod
    def build_PROFIsafe_class(cls, data_length):
        if False:
            print('Hello World!')
        assert cls.get_max_data_length() >= data_length
        return type('{}Len{}'.format(cls.__name__, data_length), (cls,), {'get_data_length': lambda _: data_length})

class PROFIsafeControl(PROFIsafe):
    name = 'PROFISafe Control Message with F_CRC_Seed=0'
    fields_desc = [StrFixedLenField('data', '', length_from=lambda p: p.get_data_length()), FlagsField('control', 0, 8, profisafe_control_flags), X3BytesField('crc', 0)]

class PROFIsafeStatus(PROFIsafe):
    name = 'PROFISafe Status Message with F_CRC_Seed=0'
    fields_desc = [StrFixedLenField('data', '', length_from=lambda p: p.get_data_length()), FlagsField('status', 0, 8, profisafe_status_flags), X3BytesField('crc', 0)]