"""
IProxyParser implementation for version two of the PROXY protocol.
"""
import binascii
import struct
from typing import Callable, Tuple, Type, Union
from zope.interface import implementer
from constantly import ValueConstant, Values
from typing_extensions import Literal
from twisted.internet import address
from twisted.python import compat
from . import _info, _interfaces
from ._exceptions import InvalidNetworkProtocol, InvalidProxyHeader, MissingAddressData, convertError

class NetFamily(Values):
    """
    Values for the 'family' field.
    """
    UNSPEC = ValueConstant(0)
    INET = ValueConstant(16)
    INET6 = ValueConstant(32)
    UNIX = ValueConstant(48)

class NetProtocol(Values):
    """
    Values for 'protocol' field.
    """
    UNSPEC = ValueConstant(0)
    STREAM = ValueConstant(1)
    DGRAM = ValueConstant(2)
_HIGH = 240
_LOW = 15
_LOCALCOMMAND = 'LOCAL'
_PROXYCOMMAND = 'PROXY'

@implementer(_interfaces.IProxyParser)
class V2Parser:
    """
    PROXY protocol version two header parser.

    Version two of the PROXY protocol is a binary format.
    """
    PREFIX = b'\r\n\r\n\x00\r\nQUIT\n'
    VERSIONS = [32]
    COMMANDS = {0: _LOCALCOMMAND, 1: _PROXYCOMMAND}
    ADDRESSFORMATS = {17: '!4s4s2H', 18: '!4s4s2H', 33: '!16s16s2H', 34: '!16s16s2H', 49: '!108s108s', 50: '!108s108s'}

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.buffer = b''

    def feed(self, data: bytes) -> Union[Tuple[_info.ProxyInfo, bytes], Tuple[None, None]]:
        if False:
            while True:
                i = 10
        '\n        Consume a chunk of data and attempt to parse it.\n\n        @param data: A bytestring.\n        @type data: bytes\n\n        @return: A two-tuple containing, in order, a L{_interfaces.IProxyInfo}\n            and any bytes fed to the parser that followed the end of the\n            header.  Both of these values are None until a complete header is\n            parsed.\n\n        @raises InvalidProxyHeader: If the bytes fed to the parser create an\n            invalid PROXY header.\n        '
        self.buffer += data
        if len(self.buffer) < 16:
            raise InvalidProxyHeader()
        size = struct.unpack('!H', self.buffer[14:16])[0] + 16
        if len(self.buffer) < size:
            return (None, None)
        (header, remaining) = (self.buffer[:size], self.buffer[size:])
        self.buffer = b''
        info = self.parse(header)
        return (info, remaining)

    @staticmethod
    def _bytesToIPv4(bytestring: bytes) -> bytes:
        if False:
            return 10
        '\n        Convert packed 32-bit IPv4 address bytes into a dotted-quad ASCII bytes\n        representation of that address.\n\n        @param bytestring: 4 octets representing an IPv4 address.\n        @type bytestring: L{bytes}\n\n        @return: a dotted-quad notation IPv4 address.\n        @rtype: L{bytes}\n        '
        return b'.'.join((('%i' % (ord(b),)).encode('ascii') for b in compat.iterbytes(bytestring)))

    @staticmethod
    def _bytesToIPv6(bytestring: bytes) -> bytes:
        if False:
            print('Hello World!')
        '\n        Convert packed 128-bit IPv6 address bytes into a colon-separated ASCII\n        bytes representation of that address.\n\n        @param bytestring: 16 octets representing an IPv6 address.\n        @type bytestring: L{bytes}\n\n        @return: a dotted-quad notation IPv6 address.\n        @rtype: L{bytes}\n        '
        hexString = binascii.b2a_hex(bytestring)
        return b':'.join((f'{int(hexString[b:b + 4], 16):x}'.encode('ascii') for b in range(0, 32, 4)))

    @classmethod
    def parse(cls, line: bytes) -> _info.ProxyInfo:
        if False:
            return 10
        '\n        Parse a bytestring as a full PROXY protocol header.\n\n        @param line: A bytestring that represents a valid HAProxy PROXY\n            protocol version 2 header.\n        @type line: bytes\n\n        @return: A L{_interfaces.IProxyInfo} containing the\n            parsed data.\n\n        @raises InvalidProxyHeader: If the bytestring does not represent a\n            valid PROXY header.\n        '
        prefix = line[:12]
        addrInfo = None
        with convertError(IndexError, InvalidProxyHeader):
            versionCommand = ord(line[12:13])
            familyProto = ord(line[13:14])
        if prefix != cls.PREFIX:
            raise InvalidProxyHeader()
        (version, command) = (versionCommand & _HIGH, versionCommand & _LOW)
        if version not in cls.VERSIONS or command not in cls.COMMANDS:
            raise InvalidProxyHeader()
        if cls.COMMANDS[command] == _LOCALCOMMAND:
            return _info.ProxyInfo(line, None, None)
        (family, netproto) = (familyProto & _HIGH, familyProto & _LOW)
        with convertError(ValueError, InvalidNetworkProtocol):
            family = NetFamily.lookupByValue(family)
            netproto = NetProtocol.lookupByValue(netproto)
        if family is NetFamily.UNSPEC or netproto is NetProtocol.UNSPEC:
            return _info.ProxyInfo(line, None, None)
        addressFormat = cls.ADDRESSFORMATS[familyProto]
        addrInfo = line[16:16 + struct.calcsize(addressFormat)]
        if family is NetFamily.UNIX:
            with convertError(struct.error, MissingAddressData):
                (source, dest) = struct.unpack(addressFormat, addrInfo)
            return _info.ProxyInfo(line, address.UNIXAddress(source.rstrip(b'\x00')), address.UNIXAddress(dest.rstrip(b'\x00')))
        addrType: Union[Literal['TCP'], Literal['UDP']] = 'TCP'
        if netproto is NetProtocol.DGRAM:
            addrType = 'UDP'
        addrCls: Union[Type[address.IPv4Address], Type[address.IPv6Address]] = address.IPv4Address
        addrParser: Callable[[bytes], bytes] = cls._bytesToIPv4
        if family is NetFamily.INET6:
            addrCls = address.IPv6Address
            addrParser = cls._bytesToIPv6
        with convertError(struct.error, MissingAddressData):
            info = struct.unpack(addressFormat, addrInfo)
            (source, dest, sPort, dPort) = info
        return _info.ProxyInfo(line, addrCls(addrType, addrParser(source).decode(), sPort), addrCls(addrType, addrParser(dest).decode(), dPort))