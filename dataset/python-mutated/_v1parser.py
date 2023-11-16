"""
IProxyParser implementation for version one of the PROXY protocol.
"""
from typing import Tuple, Union
from zope.interface import implementer
from twisted.internet import address
from . import _info, _interfaces
from ._exceptions import InvalidNetworkProtocol, InvalidProxyHeader, MissingAddressData, convertError

@implementer(_interfaces.IProxyParser)
class V1Parser:
    """
    PROXY protocol version one header parser.

    Version one of the PROXY protocol is a human readable format represented
    by a single, newline delimited binary string that contains all of the
    relevant source and destination data.
    """
    PROXYSTR = b'PROXY'
    UNKNOWN_PROTO = b'UNKNOWN'
    TCP4_PROTO = b'TCP4'
    TCP6_PROTO = b'TCP6'
    ALLOWED_NET_PROTOS = (TCP4_PROTO, TCP6_PROTO, UNKNOWN_PROTO)
    NEWLINE = b'\r\n'

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.buffer = b''

    def feed(self, data: bytes) -> Union[Tuple[_info.ProxyInfo, bytes], Tuple[None, None]]:
        if False:
            print('Hello World!')
        '\n        Consume a chunk of data and attempt to parse it.\n\n        @param data: A bytestring.\n        @type data: L{bytes}\n\n        @return: A two-tuple containing, in order, a\n            L{_interfaces.IProxyInfo} and any bytes fed to the\n            parser that followed the end of the header.  Both of these values\n            are None until a complete header is parsed.\n\n        @raises InvalidProxyHeader: If the bytes fed to the parser create an\n            invalid PROXY header.\n        '
        self.buffer += data
        if len(self.buffer) > 107 and self.NEWLINE not in self.buffer:
            raise InvalidProxyHeader()
        lines = self.buffer.split(self.NEWLINE, 1)
        if not len(lines) > 1:
            return (None, None)
        self.buffer = b''
        remaining = lines.pop()
        header = lines.pop()
        info = self.parse(header)
        return (info, remaining)

    @classmethod
    def parse(cls, line: bytes) -> _info.ProxyInfo:
        if False:
            return 10
        '\n        Parse a bytestring as a full PROXY protocol header line.\n\n        @param line: A bytestring that represents a valid HAProxy PROXY\n            protocol header line.\n        @type line: bytes\n\n        @return: A L{_interfaces.IProxyInfo} containing the parsed data.\n\n        @raises InvalidProxyHeader: If the bytestring does not represent a\n            valid PROXY header.\n\n        @raises InvalidNetworkProtocol: When no protocol can be parsed or is\n            not one of the allowed values.\n\n        @raises MissingAddressData: When the protocol is TCP* but the header\n            does not contain a complete set of addresses and ports.\n        '
        originalLine = line
        proxyStr = None
        networkProtocol = None
        sourceAddr = None
        sourcePort = None
        destAddr = None
        destPort = None
        with convertError(ValueError, InvalidProxyHeader):
            (proxyStr, line) = line.split(b' ', 1)
        if proxyStr != cls.PROXYSTR:
            raise InvalidProxyHeader()
        with convertError(ValueError, InvalidNetworkProtocol):
            (networkProtocol, line) = line.split(b' ', 1)
        if networkProtocol not in cls.ALLOWED_NET_PROTOS:
            raise InvalidNetworkProtocol()
        if networkProtocol == cls.UNKNOWN_PROTO:
            return _info.ProxyInfo(originalLine, None, None)
        with convertError(ValueError, MissingAddressData):
            (sourceAddr, line) = line.split(b' ', 1)
        with convertError(ValueError, MissingAddressData):
            (destAddr, line) = line.split(b' ', 1)
        with convertError(ValueError, MissingAddressData):
            (sourcePort, line) = line.split(b' ', 1)
        with convertError(ValueError, MissingAddressData):
            destPort = line.split(b' ')[0]
        if networkProtocol == cls.TCP4_PROTO:
            return _info.ProxyInfo(originalLine, address.IPv4Address('TCP', sourceAddr.decode(), int(sourcePort)), address.IPv4Address('TCP', destAddr.decode(), int(destPort)))
        return _info.ProxyInfo(originalLine, address.IPv6Address('TCP', sourceAddr.decode(), int(sourcePort)), address.IPv6Address('TCP', destAddr.decode(), int(destPort)))