"""
Test cases for L{twisted.protocols.haproxy.V2Parser}.
"""
from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
V2_SIGNATURE = b'\r\n\r\n\x00\r\nQUIT\n'

def _makeHeaderIPv6(sig: bytes=V2_SIGNATURE, verCom: bytes=b'!', famProto: bytes=b'!', addrLength: bytes=b'\x00$', addrs: bytes=(b'\x00' * 15 + b'\x01') * 2, ports: bytes=b'\x1f\x90"\xb8') -> bytes:
    if False:
        print('Hello World!')
    '\n    Construct a version 2 IPv6 header with custom bytes.\n\n    @param sig: The protocol signature; defaults to valid L{V2_SIGNATURE}.\n    @type sig: L{bytes}\n\n    @param verCom: Protocol version and command.  Defaults to V2 PROXY.\n    @type verCom: L{bytes}\n\n    @param famProto: Address family and protocol.  Defaults to AF_INET6/STREAM.\n    @type famProto: L{bytes}\n\n    @param addrLength: Network-endian byte length of payload.  Defaults to\n        description of default addrs/ports.\n    @type addrLength: L{bytes}\n\n    @param addrs: Address payload.  Defaults to C{::1} for source and\n        destination.\n    @type addrs: L{bytes}\n\n    @param ports: Source and destination ports.  Defaults to 8080 for source\n        8888 for destination.\n    @type ports: L{bytes}\n\n    @return: A packet with header, addresses, and ports.\n    @rtype: L{bytes}\n    '
    return sig + verCom + famProto + addrLength + addrs + ports

def _makeHeaderIPv4(sig: bytes=V2_SIGNATURE, verCom: bytes=b'!', famProto: bytes=b'\x11', addrLength: bytes=b'\x00\x0c', addrs: bytes=b'\x7f\x00\x00\x01\x7f\x00\x00\x01', ports: bytes=b'\x1f\x90"\xb8') -> bytes:
    if False:
        for i in range(10):
            print('nop')
    '\n    Construct a version 2 IPv4 header with custom bytes.\n\n    @param sig: The protocol signature; defaults to valid L{V2_SIGNATURE}.\n    @type sig: L{bytes}\n\n    @param verCom: Protocol version and command.  Defaults to V2 PROXY.\n    @type verCom: L{bytes}\n\n    @param famProto: Address family and protocol.  Defaults to AF_INET/STREAM.\n    @type famProto: L{bytes}\n\n    @param addrLength: Network-endian byte length of payload.  Defaults to\n        description of default addrs/ports.\n    @type addrLength: L{bytes}\n\n    @param addrs: Address payload.  Defaults to 127.0.0.1 for source and\n        destination.\n    @type addrs: L{bytes}\n\n    @param ports: Source and destination ports.  Defaults to 8080 for source\n        8888 for destination.\n    @type ports: L{bytes}\n\n    @return: A packet with header, addresses, and ports.\n    @rtype: L{bytes}\n    '
    return sig + verCom + famProto + addrLength + addrs + ports

def _makeHeaderUnix(sig: bytes=V2_SIGNATURE, verCom: bytes=b'!', famProto: bytes=b'1', addrLength: bytes=b'\x00\xd8', addrs: bytes=(b'/home/tests/mysockets/sock' + b'\x00' * 82) * 2) -> bytes:
    if False:
        print('Hello World!')
    '\n    Construct a version 2 IPv4 header with custom bytes.\n\n    @param sig: The protocol signature; defaults to valid L{V2_SIGNATURE}.\n    @type sig: L{bytes}\n\n    @param verCom: Protocol version and command.  Defaults to V2 PROXY.\n    @type verCom: L{bytes}\n\n    @param famProto: Address family and protocol.  Defaults to AF_UNIX/STREAM.\n    @type famProto: L{bytes}\n\n    @param addrLength: Network-endian byte length of payload.  Defaults to 108\n        bytes for 2 null terminated paths.\n    @type addrLength: L{bytes}\n\n    @param addrs: Address payload.  Defaults to C{/home/tests/mysockets/sock}\n        for source and destination paths.\n    @type addrs: L{bytes}\n\n    @return: A packet with header, addresses, and8 ports.\n    @rtype: L{bytes}\n    '
    return sig + verCom + famProto + addrLength + addrs

class V2ParserTests(unittest.TestCase):
    """
    Test L{twisted.protocols.haproxy.V2Parser} behaviour.
    """

    def test_happyPathIPv4(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test if a well formed IPv4 header is parsed without error.\n        '
        header = _makeHeaderIPv4()
        self.assertTrue(_v2parser.V2Parser.parse(header))

    def test_happyPathIPv6(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test if a well formed IPv6 header is parsed without error.\n        '
        header = _makeHeaderIPv6()
        self.assertTrue(_v2parser.V2Parser.parse(header))

    def test_happyPathUnix(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test if a well formed UNIX header is parsed without error.\n        '
        header = _makeHeaderUnix()
        self.assertTrue(_v2parser.V2Parser.parse(header))

    def test_invalidSignature(self) -> None:
        if False:
            return 10
        '\n        Test if an invalid signature block raises InvalidProxyError.\n        '
        header = _makeHeaderIPv4(sig=b'\x00' * 12)
        self.assertRaises(InvalidProxyHeader, _v2parser.V2Parser.parse, header)

    def test_invalidVersion(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test if an invalid version raises InvalidProxyError.\n        '
        header = _makeHeaderIPv4(verCom=b'\x11')
        self.assertRaises(InvalidProxyHeader, _v2parser.V2Parser.parse, header)

    def test_invalidCommand(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test if an invalid command raises InvalidProxyError.\n        '
        header = _makeHeaderIPv4(verCom=b'#')
        self.assertRaises(InvalidProxyHeader, _v2parser.V2Parser.parse, header)

    def test_invalidFamily(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if an invalid family raises InvalidProxyError.\n        '
        header = _makeHeaderIPv4(famProto=b'@')
        self.assertRaises(InvalidProxyHeader, _v2parser.V2Parser.parse, header)

    def test_invalidProto(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test if an invalid protocol raises InvalidProxyError.\n        '
        header = _makeHeaderIPv4(famProto=b'$')
        self.assertRaises(InvalidProxyHeader, _v2parser.V2Parser.parse, header)

    def test_localCommandIpv4(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that local does not return endpoint data for IPv4 connections.\n        '
        header = _makeHeaderIPv4(verCom=b' ')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_localCommandIpv6(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test that local does not return endpoint data for IPv6 connections.\n        '
        header = _makeHeaderIPv6(verCom=b' ')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_localCommandUnix(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test that local does not return endpoint data for UNIX connections.\n        '
        header = _makeHeaderUnix(verCom=b' ')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_proxyCommandIpv4(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test that proxy returns endpoint data for IPv4 connections.\n        '
        header = _makeHeaderIPv4(verCom=b'!')
        info = _v2parser.V2Parser.parse(header)
        self.assertTrue(info.source)
        self.assertIsInstance(info.source, address.IPv4Address)
        self.assertTrue(info.destination)
        self.assertIsInstance(info.destination, address.IPv4Address)

    def test_proxyCommandIpv6(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test that proxy returns endpoint data for IPv6 connections.\n        '
        header = _makeHeaderIPv6(verCom=b'!')
        info = _v2parser.V2Parser.parse(header)
        self.assertTrue(info.source)
        self.assertIsInstance(info.source, address.IPv6Address)
        self.assertTrue(info.destination)
        self.assertIsInstance(info.destination, address.IPv6Address)

    def test_proxyCommandUnix(self) -> None:
        if False:
            return 10
        '\n        Test that proxy returns endpoint data for UNIX connections.\n        '
        header = _makeHeaderUnix(verCom=b'!')
        info = _v2parser.V2Parser.parse(header)
        self.assertTrue(info.source)
        self.assertIsInstance(info.source, address.UNIXAddress)
        self.assertTrue(info.destination)
        self.assertIsInstance(info.destination, address.UNIXAddress)

    def test_unspecFamilyIpv4(self) -> None:
        if False:
            return 10
        '\n        Test that UNSPEC does not return endpoint data for IPv4 connections.\n        '
        header = _makeHeaderIPv4(famProto=b'\x01')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_unspecFamilyIpv6(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test that UNSPEC does not return endpoint data for IPv6 connections.\n        '
        header = _makeHeaderIPv6(famProto=b'\x01')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_unspecFamilyUnix(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test that UNSPEC does not return endpoint data for UNIX connections.\n        '
        header = _makeHeaderUnix(famProto=b'\x01')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_unspecProtoIpv4(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test that UNSPEC does not return endpoint data for IPv4 connections.\n        '
        header = _makeHeaderIPv4(famProto=b'\x10')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_unspecProtoIpv6(self) -> None:
        if False:
            return 10
        '\n        Test that UNSPEC does not return endpoint data for IPv6 connections.\n        '
        header = _makeHeaderIPv6(famProto=b' ')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_unspecProtoUnix(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test that UNSPEC does not return endpoint data for UNIX connections.\n        '
        header = _makeHeaderUnix(famProto=b'0')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_overflowIpv4(self) -> None:
        if False:
            return 10
        '\n        Test that overflow bits are preserved during feed parsing for IPv4.\n        '
        testValue = b'TEST DATA\r\n\r\nTEST DATA'
        header = _makeHeaderIPv4() + testValue
        parser = _v2parser.V2Parser()
        (info, overflow) = parser.feed(header)
        self.assertTrue(info)
        self.assertEqual(overflow, testValue)

    def test_overflowIpv6(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that overflow bits are preserved during feed parsing for IPv6.\n        '
        testValue = b'TEST DATA\r\n\r\nTEST DATA'
        header = _makeHeaderIPv6() + testValue
        parser = _v2parser.V2Parser()
        (info, overflow) = parser.feed(header)
        self.assertTrue(info)
        self.assertEqual(overflow, testValue)

    def test_overflowUnix(self) -> None:
        if False:
            return 10
        '\n        Test that overflow bits are preserved during feed parsing for Unix.\n        '
        testValue = b'TEST DATA\r\n\r\nTEST DATA'
        header = _makeHeaderUnix() + testValue
        parser = _v2parser.V2Parser()
        (info, overflow) = parser.feed(header)
        self.assertTrue(info)
        self.assertEqual(overflow, testValue)

    def test_segmentTooSmall(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that an initial payload of less than 16 bytes fails.\n        '
        testValue = b'NEEDMOREDATA'
        parser = _v2parser.V2Parser()
        self.assertRaises(InvalidProxyHeader, parser.feed, testValue)