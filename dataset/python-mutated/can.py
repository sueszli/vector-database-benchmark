"""A minimal implementation of the CANopen protocol, based on
Wireshark dissectors. See https://wiki.wireshark.org/CANopen

"""
import os
import gzip
import struct
from scapy.config import conf
from scapy.compat import chb, hex_bytes
from scapy.data import DLT_CAN_SOCKETCAN
from scapy.fields import FieldLenField, FlagsField, StrLenField, ThreeBytesField, XBitField, ScalingField, ConditionalField, LenField, ShortField
from scapy.volatile import RandFloat, RandBinFloat
from scapy.packet import Packet, bind_layers
from scapy.layers.l2 import CookedLinux
from scapy.error import Scapy_Exception
from scapy.plist import PacketList
from scapy.supersocket import SuperSocket
from scapy.utils import _ByteStream
from typing import Tuple, Optional, Type, List, Union, Callable, IO, Any, cast
__all__ = ['CAN', 'SignalPacket', 'SignalField', 'LESignedSignalField', 'LEUnsignedSignalField', 'LEFloatSignalField', 'BEFloatSignalField', 'BESignedSignalField', 'BEUnsignedSignalField', 'rdcandump', 'CandumpReader', 'SignalHeader', 'CAN_MTU', 'CAN_MAX_IDENTIFIER', 'CAN_MAX_DLEN', 'CAN_INV_FILTER', 'CANFD', 'CAN_FD_MTU', 'CAN_FD_MAX_DLEN']
CAN_MAX_IDENTIFIER = (1 << 29) - 1
CAN_MTU = 16
CAN_MAX_DLEN = 8
CAN_INV_FILTER = 536870912
CAN_FD_MTU = 72
CAN_FD_MAX_DLEN = 64
conf.contribs['CAN'] = {'swap-bytes': False, 'remove-padding': True}

class CAN(Packet):
    """A implementation of CAN messages.

    Dissection of CAN messages from Wireshark captures and Linux PF_CAN sockets
    are supported from protocol specification.
    See https://wiki.wireshark.org/CANopen for further information on
    the Wireshark dissector. Linux PF_CAN and Wireshark use different
    endianness for the first 32 bit of a CAN message. This dissector can be
    configured for both use cases.

    Configuration ``swap-bytes``:
        Wireshark dissection:
            >>> conf.contribs['CAN']['swap-bytes'] = False

        PF_CAN Socket dissection:
            >>> conf.contribs['CAN']['swap-bytes'] = True

    Configuration ``remove-padding``:
    Linux PF_CAN Sockets always return 16 bytes per CAN frame receive.
    This implicates that CAN frames get padded from the Linux PF_CAN socket
    with zeros up to 8 bytes of data. The real length from the CAN frame on
    the wire is given by the length field. To obtain only the CAN frame from
    the wire, this additional padding has to be removed. Nevertheless, for
    corner cases, it might be useful to also get the padding. This can be
    configured through the **remove-padding** configuration.

    Truncate CAN frame based on length field:
        >>> conf.contribs['CAN']['remove-padding'] = True

    Show entire CAN frame received from socket:
        >>> conf.contribs['CAN']['remove-padding'] = False

    """
    fields_desc = [FlagsField('flags', 0, 3, ['error', 'remote_transmission_request', 'extended']), XBitField('identifier', 0, 29), FieldLenField('length', None, length_of='data', fmt='B'), ThreeBytesField('reserved', 0), StrLenField('data', b'', length_from=lambda pkt: int(pkt.length))]

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            return 10
        if _pkt:
            fdf_set = len(_pkt) > 5 and _pkt[5] & 4 and (not _pkt[5] & 248)
            if fdf_set:
                return CANFD
            elif len(_pkt) > 4 and _pkt[4] > 8:
                return CANFD
        return CAN

    @staticmethod
    def inv_endianness(pkt):
        if False:
            while True:
                i = 10
        'Invert the order of the first four bytes of a CAN packet\n\n        This method is meant to be used specifically to convert a CAN packet\n        between the pcap format and the SocketCAN format\n\n        :param pkt: bytes str of the CAN packet\n        :return: bytes str with the first four bytes swapped\n        '
        len_partial = len(pkt) - 4
        return struct.pack('<I{}s'.format(len_partial), *struct.unpack('>I{}s'.format(len_partial), pkt))

    def pre_dissect(self, s):
        if False:
            for i in range(10):
                print('nop')
        'Implements the swap-bytes functionality when dissecting '
        if conf.contribs['CAN']['swap-bytes']:
            data = CAN.inv_endianness(s)
            return data
        return s

    def post_dissect(self, s):
        if False:
            return 10
        self.raw_packet_cache = None
        return s

    def post_build(self, pkt, pay):
        if False:
            return 10
        'Implements the swap-bytes functionality for Packet build.\n\n        This is based on a copy of the Packet.self_build default method.\n        The goal is to affect only the CAN layer data and keep\n        under layers (e.g CookedLinux) unchanged\n        '
        if conf.contribs['CAN']['swap-bytes']:
            data = CAN.inv_endianness(pkt)
            return data + pay
        return pkt + pay

    def extract_padding(self, p):
        if False:
            i = 10
            return i + 15
        if conf.contribs['CAN']['remove-padding']:
            return (b'', None)
        else:
            return (b'', p)
conf.l2types.register(DLT_CAN_SOCKETCAN, CAN)
bind_layers(CookedLinux, CAN, proto=12)

class CANFD(CAN):
    """
    This class is used for distinction of CAN FD packets.
    """
    fields_desc = [FlagsField('flags', 0, 3, ['error', 'remote_transmission_request', 'extended']), XBitField('identifier', 0, 29), FieldLenField('length', None, length_of='data', fmt='B'), FlagsField('fd_flags', 4, 8, ['bit_rate_switch', 'error_state_indicator', 'fd_frame']), ShortField('reserved', 0), StrLenField('data', b'', length_from=lambda pkt: int(pkt.length))]

    def post_build(self, pkt, pay):
        if False:
            print('Hello World!')
        data = super(CANFD, self).post_build(pkt, pay)
        length = data[4]
        if 8 < length <= 24:
            wire_length = length + -length % 4
        elif 24 < length <= 64:
            wire_length = length + -length % 8
        elif length > 64:
            raise NotImplementedError
        else:
            wire_length = length
        pad = b'\x00' * (wire_length - length)
        return data[0:4] + chb(wire_length) + data[5:] + pad
bind_layers(CookedLinux, CANFD, proto=13)

class SignalField(ScalingField):
    """SignalField is a base class for signal data, usually transmitted from
    CAN messages in automotive applications. Most vehicle manufacturers
    describe their vehicle internal signals by so called data base CAN (DBC)
    files. All necessary functions to easily create Scapy dissectors similar
    to signal descriptions from DBC files are provided by this base class.

    SignalField instances should only be used together with SignalPacket
    classes since SignalPackets enforce length checks for CAN messages.

    """
    __slots__ = ['start', 'size']

    def __init__(self, name, default, start, size, scaling=1, unit='', offset=0, ndigits=3, fmt='B'):
        if False:
            while True:
                i = 10
        ScalingField.__init__(self, name, default, scaling, unit, offset, ndigits, fmt)
        self.start = start
        self.size = abs(size)
        if fmt[-1] == 'f' and self.size != 32:
            raise Scapy_Exception('SignalField size has to be 32 for floats')
    _lookup_table = [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25, 24, 39, 38, 37, 36, 35, 34, 33, 32, 47, 46, 45, 44, 43, 42, 41, 40, 55, 54, 53, 52, 51, 50, 49, 48, 63, 62, 61, 60, 59, 58, 57, 56]

    @staticmethod
    def _msb_lookup(start):
        if False:
            i = 10
            return i + 15
        try:
            return SignalField._lookup_table.index(start)
        except ValueError:
            raise Scapy_Exception('Only 64 bits for all SignalFields are supported')

    @staticmethod
    def _lsb_lookup(start, size):
        if False:
            return 10
        try:
            return SignalField._lookup_table[SignalField._msb_lookup(start) + size - 1]
        except IndexError:
            raise Scapy_Exception('Only 64 bits for all SignalFields are supported')

    @staticmethod
    def _convert_to_unsigned(number, bit_length):
        if False:
            while True:
                i = 10
        if number & 1 << bit_length - 1:
            mask = 2 ** bit_length
            return mask + number
        return number

    @staticmethod
    def _convert_to_signed(number, bit_length):
        if False:
            print('Hello World!')
        mask = 2 ** bit_length - 1
        if number & 1 << bit_length - 1:
            return number | ~mask
        return number & mask

    def _is_little_endian(self):
        if False:
            print('Hello World!')
        return self.fmt[0] == '<'

    def _is_signed_number(self):
        if False:
            while True:
                i = 10
        return self.fmt[-1].islower()

    def _is_float_number(self):
        if False:
            print('Hello World!')
        return self.fmt[-1] == 'f'

    def addfield(self, pkt, s, val):
        if False:
            print('Hello World!')
        if not isinstance(pkt, SignalPacket):
            raise Scapy_Exception('Only use SignalFields in a SignalPacket')
        val = self.i2m(pkt, val)
        if self._is_little_endian():
            msb_pos = self.start + self.size - 1
            lsb_pos = self.start
            shift = lsb_pos
            fmt = '<Q'
        else:
            msb_pos = self.start
            lsb_pos = self._lsb_lookup(self.start, self.size)
            shift = 64 - self._msb_lookup(msb_pos) - self.size
            fmt = '>Q'
        field_len = max(msb_pos, lsb_pos) // 8 + 1
        if len(s) < field_len:
            s += b'\x00' * (field_len - len(s))
        if self._is_float_number():
            int_val = struct.unpack(self.fmt[0] + 'I', struct.pack(self.fmt, val))[0]
        elif self._is_signed_number():
            int_val = self._convert_to_unsigned(int(val), self.size)
        else:
            int_val = cast(int, val)
        pkt_val = struct.unpack(fmt, (s + b'\x00' * 8)[:8])[0]
        pkt_val |= int_val << shift
        tmp_s = struct.pack(fmt, pkt_val)
        return tmp_s[:len(s)]

    def getfield(self, pkt, s):
        if False:
            while True:
                i = 10
        if not isinstance(pkt, SignalPacket):
            raise Scapy_Exception('Only use SignalFields in a SignalPacket')
        if isinstance(s, tuple):
            (s, _) = s
        if self._is_little_endian():
            msb_pos = self.start + self.size - 1
            lsb_pos = self.start
            shift = self.start
            fmt = '<Q'
        else:
            msb_pos = self.start
            lsb_pos = self._lsb_lookup(self.start, self.size)
            shift = 64 - self._msb_lookup(self.start) - self.size
            fmt = '>Q'
        field_len = max(msb_pos, lsb_pos) // 8 + 1
        if pkt.wirelen is None:
            pkt.wirelen = field_len
        pkt.wirelen = max(pkt.wirelen, field_len)
        fld_val = struct.unpack(fmt, (s + b'\x00' * 8)[:8])[0] >> shift
        fld_val &= (1 << self.size) - 1
        if self._is_float_number():
            fld_val = struct.unpack(self.fmt, struct.pack(self.fmt[0] + 'I', fld_val))[0]
        elif self._is_signed_number():
            fld_val = self._convert_to_signed(fld_val, self.size)
        return (s, self.m2i(pkt, fld_val))

    def randval(self):
        if False:
            print('Hello World!')
        if self._is_float_number():
            return RandBinFloat(0, 0)
        if self._is_signed_number():
            min_val = -2 ** (self.size - 1)
            max_val = 2 ** (self.size - 1) - 1
        else:
            min_val = 0
            max_val = 2 ** self.size - 1
        min_val = round(min_val * self.scaling + self.offset, self.ndigits)
        max_val = round(max_val * self.scaling + self.offset, self.ndigits)
        return RandFloat(min(min_val, max_val), max(min_val, max_val))

    def i2len(self, pkt, x):
        if False:
            return 10
        return int(float(self.size) / 8)

class LEUnsignedSignalField(SignalField):

    def __init__(self, name, default, start, size, scaling=1, unit='', offset=0, ndigits=3):
        if False:
            while True:
                i = 10
        SignalField.__init__(self, name, default, start, size, scaling, unit, offset, ndigits, '<B')

class LESignedSignalField(SignalField):

    def __init__(self, name, default, start, size, scaling=1, unit='', offset=0, ndigits=3):
        if False:
            while True:
                i = 10
        SignalField.__init__(self, name, default, start, size, scaling, unit, offset, ndigits, '<b')

class BEUnsignedSignalField(SignalField):

    def __init__(self, name, default, start, size, scaling=1, unit='', offset=0, ndigits=3):
        if False:
            i = 10
            return i + 15
        SignalField.__init__(self, name, default, start, size, scaling, unit, offset, ndigits, '>B')

class BESignedSignalField(SignalField):

    def __init__(self, name, default, start, size, scaling=1, unit='', offset=0, ndigits=3):
        if False:
            print('Hello World!')
        SignalField.__init__(self, name, default, start, size, scaling, unit, offset, ndigits, '>b')

class LEFloatSignalField(SignalField):

    def __init__(self, name, default, start, scaling=1, unit='', offset=0, ndigits=3):
        if False:
            for i in range(10):
                print('nop')
        SignalField.__init__(self, name, default, start, 32, scaling, unit, offset, ndigits, '<f')

class BEFloatSignalField(SignalField):

    def __init__(self, name, default, start, scaling=1, unit='', offset=0, ndigits=3):
        if False:
            for i in range(10):
                print('nop')
        SignalField.__init__(self, name, default, start, 32, scaling, unit, offset, ndigits, '>f')

class SignalPacket(Packet):
    """Special implementation of Packet.

    This class enforces the correct wirelen of a CAN message for
    signal transmitting in automotive applications.
    Furthermore, the dissection order of SignalFields in fields_desc is
    deduced by the start index of a field.
    """

    def pre_dissect(self, s):
        if False:
            print('Hello World!')
        if not all((isinstance(f, SignalField) or (isinstance(f, ConditionalField) and isinstance(f.fld, SignalField)) for f in self.fields_desc)):
            raise Scapy_Exception('Use only SignalFields in a SignalPacket')
        return s

    def post_dissect(self, s):
        if False:
            return 10
        'SignalFields can be dissected on packets with unordered fields.\n\n        The order of SignalFields is defined from the start parameter.\n        After a build, the consumed bytes of the length of all SignalFields\n        have to be removed from the SignalPacket.\n        '
        if self.wirelen is not None and self.wirelen > 8:
            raise Scapy_Exception('Only 64 bits for all SignalFields are supported')
        self.raw_packet_cache = None
        return s[self.wirelen:]

class SignalHeader(CAN):
    """Special implementation of a CAN Packet to allow dynamic binding.

    This class can be provided to CANSockets as basecls.

    Example:
        >>> class floatSignals(SignalPacket):
        >>>     fields_desc = [
        >>>         LEFloatSignalField("floatSignal2", default=0, start=32),
        >>>         BEFloatSignalField("floatSignal1", default=0, start=7)]
        >>>
        >>> bind_layers(SignalHeader, floatSignals, identifier=0x321)
        >>>
        >>> dbc_sock = CANSocket("can0", basecls=SignalHeader)

    All CAN messages received from this dbc_sock CANSocket will be interpreted
    as SignalHeader. Through Scapys ``bind_layers`` mechanism, all CAN messages
    with CAN identifier 0x321 will interpret the payload bytes of these
    CAN messages as floatSignals packet.
    """
    fields_desc = [FlagsField('flags', 0, 3, ['error', 'remote_transmission_request', 'extended']), XBitField('identifier', 0, 29), LenField('length', None, fmt='B'), FlagsField('fd_flags', 0, 8, ['bit_rate_switch', 'error_state_indicator', 'fd_frame']), ShortField('reserved', 0)]

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            print('Hello World!')
        return SignalHeader

    def extract_padding(self, s):
        if False:
            for i in range(10):
                print('nop')
        return (s, None)

def rdcandump(filename, count=-1, interface=None):
    if False:
        for i in range(10):
            print('nop')
    ' Read a candump log file and return a packet list.\n\n    :param filename: Filename of the file to read from.\n                     Also gzip files are accepted.\n    :param count: Read only <count> packets. Specify -1 to read all packets.\n    :param interface: Return only packets from a specified interface\n    :return: A PacketList object containing the read files\n    '
    with CandumpReader(filename, interface) as fdesc:
        return fdesc.read_all(count=count)

class CandumpReader:
    """A stateful candump reader. Each packet is returned as a CAN packet.

    Creates a CandumpReader object

    :param filename: filename of a candump logfile, compressed or
                     uncompressed, or a already opened file object.
    :param interface: Name of a interface, if candump contains messages
                      of multiple interfaces and only one messages from a
                      specific interface are wanted.
    """
    nonblocking_socket = True

    def __init__(self, filename, interface=None):
        if False:
            i = 10
            return i + 15
        (self.filename, self.f) = self.open(filename)
        self.ifilter = None
        if interface is not None:
            if isinstance(interface, str):
                self.ifilter = [interface]
            else:
                self.ifilter = interface

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    @staticmethod
    def open(filename):
        if False:
            print('Hello World!')
        'Open function to handle three types of input data.\n\n        If filename of a regular candump log file is provided, this function\n        opens the file and returns the file object.\n        If filename of a gzip compressed candump log file is provided, the\n        required gzip open function is used to obtain the necessary file\n        object, which gets returned.\n        If a fileobject or ByteIO is provided, the filename is gathered for\n        internal use. No further steps are performed on this object.\n\n        :param filename: Can be a string, specifying a candump log file or a\n                         gzip compressed candump log file. Also already opened\n                         file objects are allowed.\n        :return: A opened file object for further use.\n        '
        'Open (if necessary) filename.'
        if isinstance(filename, str):
            try:
                fdesc = gzip.open(filename, 'rb')
                fdesc.read(1)
                fdesc.seek(0)
            except IOError:
                fdesc = open(filename, 'rb')
            return (filename, fdesc)
        else:
            name = getattr(filename, 'name', 'No name')
            return (name, filename)

    def next(self):
        if False:
            return 10
        'Implements the iterator protocol on a set of packets\n\n        :return: Next readable CAN Packet from the specified file\n        '
        try:
            pkt = None
            while pkt is None:
                pkt = self.read_packet()
        except EOFError:
            raise StopIteration
        return pkt
    __next__ = next

    def read_packet(self, size=CAN_MTU):
        if False:
            i = 10
            return i + 15
        'Read a packet from the specified file.\n\n        This function will raise EOFError when no more packets are available.\n\n        :param size: Not used. Just here to follow the function signature for\n                     SuperSocket emulation.\n        :return: A single packet read from the file or None if filters apply\n        '
        line = self.f.readline()
        line = line.lstrip()
        if len(line) < 16:
            raise EOFError
        is_log_file_format = line[0] == ord(b'(')
        fd_flags = None
        if is_log_file_format:
            (t_b, intf, f) = line.split()
            if b'##' in f:
                (idn, data) = f.split(b'##')
                fd_flags = data[0]
                data = data[1:]
            else:
                (idn, data) = f.split(b'#')
            le = None
            t = float(t_b[1:-1])
        else:
            (h, data) = line.split(b']')
            (intf, idn, le) = h.split()
            t = None
        if self.ifilter is not None and intf.decode('ASCII') not in self.ifilter:
            return None
        data = data.replace(b' ', b'')
        data = data.strip()
        if len(data) <= 8 and fd_flags is None:
            pkt = CAN(identifier=int(idn, 16), data=hex_bytes(data))
        else:
            pkt = CANFD(identifier=int(idn, 16), fd_flags=fd_flags, data=hex_bytes(data))
        if le is not None:
            pkt.length = int(le[1:])
        else:
            pkt.length = len(pkt.data)
        if len(idn) > 3:
            pkt.flags = 4
        if t is not None:
            pkt.time = t
        return pkt

    def dispatch(self, callback):
        if False:
            print('Hello World!')
        'Call the specified callback routine for each packet read\n\n        This is just a convenience function for the main loop\n        that allows for easy launching of packet processing in a\n        thread.\n        '
        for p in self:
            callback(p)

    def read_all(self, count=-1):
        if False:
            return 10
        'Read a specific number or all packets from a candump file.\n\n        :param count: Specify a specific number of packets to be read.\n                      All packets can be read by count=-1.\n        :return: A PacketList object containing read CAN messages\n        '
        res = []
        while count != 0:
            try:
                p = self.read_packet()
                if p is None:
                    continue
            except EOFError:
                break
            count -= 1
            res.append(p)
        return PacketList(res, name=os.path.basename(self.filename))

    def recv(self, size=CAN_MTU):
        if False:
            for i in range(10):
                print('nop')
        'Emulation of SuperSocket'
        try:
            return self.read_packet(size=size)
        except EOFError:
            return None

    def fileno(self):
        if False:
            return 10
        'Emulation of SuperSocket'
        return self.f.fileno()

    @property
    def closed(self):
        if False:
            for i in range(10):
                print('nop')
        return self.f.closed

    def close(self):
        if False:
            while True:
                i = 10
        'Emulation of SuperSocket'
        return self.f.close()

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, exc_type, exc_value, tracback):
        if False:
            while True:
                i = 10
        self.close()

    @staticmethod
    def select(sockets, remain=None):
        if False:
            while True:
                i = 10
        'Emulation of SuperSocket'
        return [s for s in sockets if isinstance(s, CandumpReader) and (not s.closed)]