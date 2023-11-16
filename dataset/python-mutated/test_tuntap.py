"""
Tests for L{twisted.pair.tuntap}.
"""
import os
import socket
import struct
from collections import deque
from errno import EAGAIN, EBADF, EINVAL, ENODEV, ENOENT, EPERM, EWOULDBLOCK
from itertools import cycle
from random import randrange
from signal import SIGINT
from typing import Optional
from twisted.python.reflect import ObjectNotFound, namedAny
platformSkip: Optional[str]
try:
    namedAny('fcntl.ioctl')
except (ObjectNotFound, AttributeError):
    platformSkip = 'Platform is missing fcntl/ioctl support'
else:
    platformSkip = None
from zope.interface import Interface, implementer
from zope.interface.verify import verifyObject
from twisted.internet.error import CannotListenError
from twisted.internet.interfaces import IAddress, IListeningPort, IReactorFDSet
from twisted.internet.protocol import AbstractDatagramProtocol, DatagramProtocol, Factory
from twisted.internet.task import Clock
from twisted.pair.ethernet import EthernetProtocol
from twisted.pair.ip import IPProtocol
from twisted.pair.raw import IRawPacketProtocol
from twisted.pair.rawudp import RawUDPProtocol
from twisted.python.compat import iterbytes
from twisted.python.log import addObserver, removeObserver, textFromEventDict
from twisted.python.reflect import fullyQualifiedName
from twisted.trial.unittest import SkipTest, SynchronousTestCase
_RealSystem = object
_IInputOutputSystem = Interface
if not platformSkip:
    from twisted.pair.testing import _H, _PI_SIZE, MemoryIOSystem, Tunnel, _ethernet, _ip, _IPv4, _udp
    from twisted.pair.tuntap import _IFNAMSIZ, _TUNSETIFF, TunnelAddress, TunnelFlags, TuntapPort, _IInputOutputSystem, _RealSystem
else:
    skip = platformSkip

@implementer(IReactorFDSet)
class ReactorFDSet:
    """
    An implementation of L{IReactorFDSet} which only keeps track of which
    descriptors have been registered for reading and writing.

    This implementation isn't actually capable of determining readability or
    writeability and generates no events for the descriptors registered with
    it.

    @ivar _readers: A L{set} of L{IReadDescriptor} providers which the reactor
        is supposedly monitoring for read events.

    @ivar _writers: A L{set} of L{IWriteDescriptor} providers which the reactor
        is supposedly monitoring for write events.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._readers = set()
        self._writers = set()
        self.addReader = self._readers.add
        self.addWriter = self._writers.add

    def removeReader(self, reader):
        if False:
            return 10
        self._readers.discard(reader)

    def removeWriter(self, writer):
        if False:
            return 10
        self._writers.discard(writer)

    def getReaders(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self._readers)

    def getWriters(self):
        if False:
            print('Hello World!')
        return iter(self._writers)

    def removeAll(self):
        if False:
            print('Hello World!')
        try:
            return list(self._readers | self._writers)
        finally:
            self._readers = set()
            self._writers = set()
verifyObject(IReactorFDSet, ReactorFDSet())

class FSSetClock(Clock, ReactorFDSet):
    """
    An L{FSSetClock} is a L{IReactorFDSet} and an L{IReactorClock}.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        Clock.__init__(self)
        ReactorFDSet.__init__(self)

class TunHelper:
    """
    A helper for tests of tun-related functionality (ip-level tunnels).
    """

    @property
    def TUNNEL_TYPE(self):
        if False:
            for i in range(10):
                print('nop')
        return TunnelFlags.IFF_TUN | TunnelFlags.IFF_NO_PI

    def __init__(self, tunnelRemote, tunnelLocal):
        if False:
            while True:
                i = 10
        '\n        @param tunnelRemote: The source address for UDP datagrams originated\n            from this helper.  This is an IPv4 dotted-quad string.\n        @type tunnelRemote: L{bytes}\n\n        @param tunnelLocal: The destination address for UDP datagrams\n            originated from this helper.  This is an IPv4 dotted-quad string.\n        @type tunnelLocal: L{bytes}\n        '
        self.tunnelRemote = tunnelRemote
        self.tunnelLocal = tunnelLocal

    def encapsulate(self, source, destination, payload):
        if False:
            i = 10
            return i + 15
        '\n        Construct an ip datagram containing a udp datagram containing the given\n        application-level payload.\n\n        @param source: The source port for the UDP datagram being encapsulated.\n        @type source: L{int}\n\n        @param destination: The destination port for the UDP datagram being\n            encapsulated.\n        @type destination: L{int}\n\n        @param payload: The application data to include in the udp datagram.\n        @type payload: L{bytes}\n\n        @return: An ethernet frame.\n        @rtype: L{bytes}\n        '
        return _ip(src=self.tunnelRemote, dst=self.tunnelLocal, payload=_udp(src=source, dst=destination, payload=payload))

    def parser(self):
        if False:
            while True:
                i = 10
        '\n        Get a function for parsing a datagram read from a I{tun} device.\n\n        @return: A function which accepts a datagram exactly as might be read\n            from a I{tun} device.  The datagram is expected to ultimately carry\n            a UDP datagram.  When called, it returns a L{list} of L{tuple}s.\n            Each tuple has the UDP application data as the first element and\n            the sender address as the second element.\n        '
        datagrams = []
        receiver = DatagramProtocol()

        def capture(*args):
            if False:
                return 10
            datagrams.append(args)
        receiver.datagramReceived = capture
        udp = RawUDPProtocol()
        udp.addProto(12345, receiver)
        ip = IPProtocol()
        ip.addProto(17, udp)

        def parse(data):
            if False:
                return 10
            ip.datagramReceived(data, False, None, None, None)
            return datagrams
        return parse

class TapHelper:
    """
    A helper for tests of tap-related functionality (ethernet-level tunnels).
    """

    @property
    def TUNNEL_TYPE(self):
        if False:
            print('Hello World!')
        flag = TunnelFlags.IFF_TAP
        if not self.pi:
            flag |= TunnelFlags.IFF_NO_PI
        return flag

    def __init__(self, tunnelRemote, tunnelLocal, pi):
        if False:
            return 10
        '\n        @param tunnelRemote: The source address for UDP datagrams originated\n            from this helper.  This is an IPv4 dotted-quad string.\n        @type tunnelRemote: L{bytes}\n\n        @param tunnelLocal: The destination address for UDP datagrams\n            originated from this helper.  This is an IPv4 dotted-quad string.\n        @type tunnelLocal: L{bytes}\n\n        @param pi: A flag indicating whether this helper will generate and\n            consume a protocol information (PI) header.\n        @type pi: L{bool}\n        '
        self.tunnelRemote = tunnelRemote
        self.tunnelLocal = tunnelLocal
        self.pi = pi

    def encapsulate(self, source, destination, payload):
        if False:
            i = 10
            return i + 15
        '\n        Construct an ethernet frame containing an ip datagram containing a udp\n        datagram containing the given application-level payload.\n\n        @param source: The source port for the UDP datagram being encapsulated.\n        @type source: L{int}\n\n        @param destination: The destination port for the UDP datagram being\n            encapsulated.\n        @type destination: L{int}\n\n        @param payload: The application data to include in the udp datagram.\n        @type payload: L{bytes}\n\n        @return: An ethernet frame.\n        @rtype: L{bytes}\n        '
        tun = TunHelper(self.tunnelRemote, self.tunnelLocal)
        ip = tun.encapsulate(source, destination, payload)
        frame = _ethernet(src=b'\x00\x00\x00\x00\x00\x00', dst=b'\xff\xff\xff\xff\xff\xff', protocol=_IPv4, payload=ip)
        if self.pi:
            protocol = _IPv4
            flags = 0
            frame = _H(flags) + _H(protocol) + frame
        return frame

    def parser(self):
        if False:
            print('Hello World!')
        '\n        Get a function for parsing a datagram read from a I{tap} device.\n\n        @return: A function which accepts a datagram exactly as might be read\n            from a I{tap} device.  The datagram is expected to ultimately carry\n            a UDP datagram.  When called, it returns a L{list} of L{tuple}s.\n            Each tuple has the UDP application data as the first element and\n            the sender address as the second element.\n        '
        datagrams = []
        receiver = DatagramProtocol()

        def capture(*args):
            if False:
                for i in range(10):
                    print('nop')
            datagrams.append(args)
        receiver.datagramReceived = capture
        udp = RawUDPProtocol()
        udp.addProto(12345, receiver)
        ip = IPProtocol()
        ip.addProto(17, udp)
        ether = EthernetProtocol()
        ether.addProto(2048, ip)

        def parser(datagram):
            if False:
                for i in range(10):
                    print('nop')
            if self.pi:
                datagram = datagram[_PI_SIZE:]
            ether.datagramReceived(datagram)
            return datagrams
        return parser

class TunnelTests(SynchronousTestCase):
    """
    L{Tunnel} is mostly tested by other test cases but some tests don't fit
    there.  Those tests are here.
    """

    def test_blockingRead(self):
        if False:
            while True:
                i = 10
        '\n        Blocking reads are not implemented by L{Tunnel.read}.  Attempting one\n        results in L{NotImplementedError} being raised.\n        '
        tunnel = Tunnel(MemoryIOSystem(), os.O_RDONLY, None)
        self.assertRaises(NotImplementedError, tunnel.read, 1024)

class TunnelDeviceTestsMixin:
    """
    A mixin defining tests that apply to L{_IInputOutputSystem}
    implementations.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        '\n        Create the L{_IInputOutputSystem} provider under test and open a tunnel\n        using it.\n        '
        self.system = self.createSystem()
        self.fileno = self.system.open(b'/dev/net/tun', os.O_RDWR | os.O_NONBLOCK)
        self.addCleanup(self.system.close, self.fileno)
        mode = self.helper.TUNNEL_TYPE
        config = struct.pack('%dsH' % (_IFNAMSIZ,), self._TUNNEL_DEVICE, mode.value)
        self.system.ioctl(self.fileno, _TUNSETIFF, config)

    def test_interface(self):
        if False:
            while True:
                i = 10
        '\n        The object under test provides L{_IInputOutputSystem}.\n        '
        self.assertTrue(verifyObject(_IInputOutputSystem, self.system))

    def _invalidFileDescriptor(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get an invalid file descriptor.\n\n        @return: An integer which is not a valid file descriptor at the time of\n            this call.  After any future system call which allocates a new file\n            descriptor, there is no guarantee the returned file descriptor will\n            still be invalid.\n        '
        fd = self.system.open(b'/dev/net/tun', os.O_RDWR)
        self.system.close(fd)
        return fd

    def test_readEBADF(self):
        if False:
            while True:
                i = 10
        "\n        The device's C{read} implementation raises L{OSError} with an errno of\n        C{EBADF} when called on a file descriptor which is not valid (ie, which\n        has no associated file description).\n        "
        fd = self._invalidFileDescriptor()
        exc = self.assertRaises(OSError, self.system.read, fd, 1024)
        self.assertEqual(EBADF, exc.errno)

    def test_writeEBADF(self):
        if False:
            return 10
        "\n        The device's C{write} implementation raises L{OSError} with an errno of\n        C{EBADF} when called on a file descriptor which is not valid (ie, which\n        has no associated file description).\n        "
        fd = self._invalidFileDescriptor()
        exc = self.assertRaises(OSError, self.system.write, fd, b'bytes')
        self.assertEqual(EBADF, exc.errno)

    def test_closeEBADF(self):
        if False:
            while True:
                i = 10
        "\n        The device's C{close} implementation raises L{OSError} with an errno of\n        C{EBADF} when called on a file descriptor which is not valid (ie, which\n        has no associated file description).\n        "
        fd = self._invalidFileDescriptor()
        exc = self.assertRaises(OSError, self.system.close, fd)
        self.assertEqual(EBADF, exc.errno)

    def test_ioctlEBADF(self):
        if False:
            return 10
        "\n        The device's C{ioctl} implementation raises L{OSError} with an errno of\n        C{EBADF} when called on a file descriptor which is not valid (ie, which\n        has no associated file description).\n        "
        fd = self._invalidFileDescriptor()
        exc = self.assertRaises(IOError, self.system.ioctl, fd, _TUNSETIFF, b'tap0')
        self.assertEqual(EBADF, exc.errno)

    def test_ioctlEINVAL(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The device's C{ioctl} implementation raises L{IOError} with an errno of\n        C{EINVAL} when called with a request (second argument) which is not a\n        supported operation.\n        "
        request = 3735928559
        exc = self.assertRaises(IOError, self.system.ioctl, self.fileno, request, b'garbage')
        self.assertEqual(EINVAL, exc.errno)

    def test_receive(self):
        if False:
            i = 10
            return i + 15
        '\n        If a UDP datagram is sent to an address reachable by the tunnel device\n        then it can be read out of the tunnel device.\n        '
        parse = self.helper.parser()
        found = False
        for i in range(100):
            key = randrange(2 ** 64)
            message = b'hello world:%d' % (key,)
            source = self.system.sendUDP(message, (self._TUNNEL_REMOTE, 12345))
            for j in range(100):
                try:
                    packet = self.system.read(self.fileno, 1024)
                except OSError as e:
                    if e.errno in (EAGAIN, EWOULDBLOCK):
                        break
                    raise
                else:
                    datagrams = parse(packet)
                    if (message, source) in datagrams:
                        found = True
                        break
                    del datagrams[:]
            if found:
                break
        if not found:
            self.fail('Never saw probe UDP packet on tunnel')

    def test_send(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If a UDP datagram is written the tunnel device then it is received by\n        the network to which it is addressed.\n        '
        key = randrange(2 ** 64)
        message = b'hello world:%d' % (key,)
        self.addCleanup(socket.setdefaulttimeout, socket.getdefaulttimeout())
        socket.setdefaulttimeout(120)
        port = self.system.receiveUDP(self.fileno, self._TUNNEL_LOCAL, 12345)
        packet = self.helper.encapsulate(50000, 12345, message)
        self.system.write(self.fileno, packet)
        packet = port.recv(1024)
        self.assertEqual(message, packet)

class FakeDeviceTestsMixin:
    """
    Define a mixin for use with test cases that require an
    L{_IInputOutputSystem} provider.  This mixin hands out L{MemoryIOSystem}
    instances as the provider of that interface.
    """
    _TUNNEL_DEVICE = b'tap-twistedtest'
    _TUNNEL_LOCAL = b'172.16.2.1'
    _TUNNEL_REMOTE = b'172.16.2.2'

    def createSystem(self):
        if False:
            print('Hello World!')
        '\n        Create and return a brand new L{MemoryIOSystem}.\n\n        The L{MemoryIOSystem} knows how to open new tunnel devices.\n\n        @return: The newly created I/O system object.\n        @rtype: L{MemoryIOSystem}\n        '
        system = MemoryIOSystem()
        system.registerSpecialDevice(Tunnel._DEVICE_NAME, Tunnel)
        return system

class FakeTapDeviceTests(FakeDeviceTestsMixin, TunnelDeviceTestsMixin, SynchronousTestCase):
    """
    Run various tap-type tunnel unit tests against an in-memory I/O system.
    """
setattr(FakeTapDeviceTests, 'helper', TapHelper(FakeTapDeviceTests._TUNNEL_REMOTE, FakeTapDeviceTests._TUNNEL_LOCAL, pi=False))

class FakeTapDeviceWithPITests(FakeDeviceTestsMixin, TunnelDeviceTestsMixin, SynchronousTestCase):
    """
    Run various tap-type tunnel unit tests against an in-memory I/O system with
    the PI header enabled.
    """
setattr(FakeTapDeviceWithPITests, 'helper', TapHelper(FakeTapDeviceTests._TUNNEL_REMOTE, FakeTapDeviceTests._TUNNEL_LOCAL, pi=True))

class FakeTunDeviceTests(FakeDeviceTestsMixin, TunnelDeviceTestsMixin, SynchronousTestCase):
    """
    Run various tun-type tunnel unit tests against an in-memory I/O system.
    """
setattr(FakeTunDeviceTests, 'helper', TunHelper(FakeTunDeviceTests._TUNNEL_REMOTE, FakeTunDeviceTests._TUNNEL_LOCAL))

@implementer(_IInputOutputSystem)
class TestRealSystem(_RealSystem):
    """
    Add extra skipping logic so tests that try to create real tunnel devices on
    platforms where those are not supported automatically get skipped.
    """

    def open(self, filename, *args, **kwargs):
        if False:
            return 10
        '\n        Attempt an open, but if the file is /dev/net/tun and it does not exist,\n        translate the error into L{SkipTest} so that tests that require\n        platform support for tuntap devices are skipped instead of failed.\n        '
        try:
            return super().open(filename, *args, **kwargs)
        except OSError as e:
            if e.errno in (ENOENT, ENODEV) and filename == b'/dev/net/tun':
                raise SkipTest('Platform lacks /dev/net/tun')
            raise

    def ioctl(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Attempt an ioctl, but translate permission denied errors into\n        L{SkipTest} so that tests that require elevated system privileges and\n        do not have them are skipped instead of failed.\n        '
        try:
            return super().ioctl(*args, **kwargs)
        except OSError as e:
            if EPERM == e.errno:
                raise SkipTest('Permission to configure device denied')
            raise

    def sendUDP(self, datagram, address):
        if False:
            print('Hello World!')
        '\n        Use the platform network stack to send a datagram to the given address.\n\n        @param datagram: A UDP datagram payload to send.\n        @type datagram: L{bytes}\n\n        @param address: The destination to which to send the datagram.\n        @type address: L{tuple} of (L{bytes}, L{int})\n\n        @return: The address from which the UDP datagram was sent.\n        @rtype: L{tuple} of (L{bytes}, L{int})\n        '
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('172.16.0.1', 0))
        s.sendto(datagram, address)
        return s.getsockname()

    def receiveUDP(self, fileno, host, port):
        if False:
            print('Hello World!')
        '\n        Use the platform network stack to receive a datagram sent to the given\n        address.\n\n        @param fileno: The file descriptor of the tunnel used to send the\n            datagram.  This is ignored because a real socket is used to receive\n            the datagram.\n        @type fileno: L{int}\n\n        @param host: The IPv4 address at which the datagram will be received.\n        @type host: L{bytes}\n\n        @param port: The UDP port number at which the datagram will be\n            received.\n        @type port: L{int}\n\n        @return: A L{socket.socket} which can be used to receive the specified\n            datagram.\n        '
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host, port))
        return s

class RealDeviceTestsMixin:
    """
    Define a mixin for use with test cases that require an
    L{_IInputOutputSystem} provider.  This mixin hands out L{TestRealSystem}
    instances as the provider of that interface.
    """
    skip = platformSkip

    def createSystem(self):
        if False:
            print('Hello World!')
        '\n        Create a real I/O system that can be used to open real tunnel device\n        provided by the underlying system and previously configured.\n\n        @return: The newly created I/O system object.\n        @rtype: L{TestRealSystem}\n        '
        return TestRealSystem()

class RealDeviceWithProtocolInformationTests(RealDeviceTestsMixin, TunnelDeviceTestsMixin, SynchronousTestCase):
    """
    Run various tap-type tunnel unit tests, with "protocol information" (PI)
    turned on, against a real I/O system.
    """
    _TUNNEL_DEVICE = b'tap-twtest-pi'
    _TUNNEL_LOCAL = b'172.16.1.1'
    _TUNNEL_REMOTE = b'172.16.1.2'
    helper = TapHelper(_TUNNEL_REMOTE, _TUNNEL_LOCAL, pi=True)

class RealDeviceWithoutProtocolInformationTests(RealDeviceTestsMixin, TunnelDeviceTestsMixin, SynchronousTestCase):
    """
    Run various tap-type tunnel unit tests, with "protocol information" (PI)
    turned off, against a real I/O system.
    """
    _TUNNEL_DEVICE = b'tap-twtest'
    _TUNNEL_LOCAL = b'172.16.0.1'
    _TUNNEL_REMOTE = b'172.16.0.2'
    helper = TapHelper(_TUNNEL_REMOTE, _TUNNEL_LOCAL, pi=False)

class TuntapPortTests(SynchronousTestCase):
    """
    Tests for L{TuntapPort} behavior that is independent of the tunnel type.
    """

    def test_interface(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A L{TuntapPort} instance provides L{IListeningPort}.\n        '
        port = TuntapPort(b'device', EthernetProtocol())
        self.assertTrue(verifyObject(IListeningPort, port))

    def test_realSystem(self):
        if False:
            i = 10
            return i + 15
        '\n        When not initialized with an I/O system, L{TuntapPort} uses a\n        L{_RealSystem}.\n        '
        port = TuntapPort(b'device', EthernetProtocol())
        self.assertIsInstance(port._system, _RealSystem)

class TunnelTestsMixin:
    """
    A mixin defining tests for L{TuntapPort}.

    These tests run against L{MemoryIOSystem} (proven equivalent to the real
    thing by the tests above) to avoid performing any real I/O.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        '\n        Create an in-memory I/O system and set up a L{TuntapPort} against it.\n        '
        self.name = b'tun0'
        self.system = MemoryIOSystem()
        self.system.registerSpecialDevice(Tunnel._DEVICE_NAME, Tunnel)
        self.protocol = self.factory.buildProtocol(TunnelAddress(self.helper.TUNNEL_TYPE, self.name))
        self.reactor = FSSetClock()
        self.port = TuntapPort(self.name, self.protocol, reactor=self.reactor, system=self.system)

    def _tunnelTypeOnly(self, flags):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mask off any flags except for L{TunnelType.IFF_TUN} and\n        L{TunnelType.IFF_TAP}.\n\n        @param flags: Flags from L{TunnelType} to mask.\n        @type flags: L{FlagConstant}\n\n        @return: The flags given by C{flags} except the two type flags.\n        @rtype: L{FlagConstant}\n        '
        return flags & (TunnelFlags.IFF_TUN | TunnelFlags.IFF_TAP)

    def test_startListeningOpensDevice(self):
        if False:
            i = 10
            return i + 15
        '\n        L{TuntapPort.startListening} opens the tunnel factory character special\n        device C{"/dev/net/tun"} and configures it as a I{tun} tunnel.\n        '
        system = self.system
        self.port.startListening()
        tunnel = self.system.getTunnel(self.port)
        expected = (system.O_RDWR | system.O_CLOEXEC | system.O_NONBLOCK, b'tun0' + b'\x00' * (_IFNAMSIZ - len(b'tun0')), self.port.interface, False, True)
        actual = (tunnel.openFlags, tunnel.requestedName, tunnel.name, tunnel.blocking, tunnel.closeOnExec)
        self.assertEqual(expected, actual)

    def test_startListeningSetsConnected(self):
        if False:
            return 10
        '\n        L{TuntapPort.startListening} sets C{connected} on the port object to\n        C{True}.\n        '
        self.port.startListening()
        self.assertTrue(self.port.connected)

    def test_startListeningConnectsProtocol(self):
        if False:
            print('Hello World!')
        '\n        L{TuntapPort.startListening} calls C{makeConnection} on the protocol\n        the port was initialized with, passing the port as an argument.\n        '
        self.port.startListening()
        self.assertIs(self.port, self.protocol.transport)

    def test_startListeningStartsReading(self):
        if False:
            return 10
        "\n        L{TuntapPort.startListening} passes the port instance to the reactor's\n        C{addReader} method to begin watching the port's file descriptor for\n        data to read.\n        "
        self.port.startListening()
        self.assertIn(self.port, self.reactor.getReaders())

    def test_startListeningHandlesOpenFailure(self):
        if False:
            while True:
                i = 10
        '\n        L{TuntapPort.startListening} raises L{CannotListenError} if opening the\n        tunnel factory character special device fails.\n        '
        self.system.permissions.remove('open')
        self.assertRaises(CannotListenError, self.port.startListening)

    def test_startListeningHandlesConfigureFailure(self):
        if False:
            return 10
        '\n        L{TuntapPort.startListening} raises L{CannotListenError} if the\n        C{ioctl} call to configure the tunnel device fails.\n        '
        self.system.permissions.remove('ioctl')
        self.assertRaises(CannotListenError, self.port.startListening)

    def _stopPort(self, port):
        if False:
            print('Hello World!')
        '\n        Verify that the C{stopListening} method of an L{IListeningPort} removes\n        that port from the reactor\'s "readers" set and also that the\n        L{Deferred} returned by that method fires with L{None}.\n\n        @param port: The port object to stop.\n        @type port: L{IListeningPort} provider\n        '
        stopped = port.stopListening()
        self.assertNotIn(port, self.reactor.getReaders())
        self.reactor.advance(0)
        self.assertIsNone(self.successResultOf(stopped))

    def test_stopListeningStopsReading(self):
        if False:
            i = 10
            return i + 15
        "\n        L{TuntapPort.stopListening} returns a L{Deferred} which fires after the\n        port has been removed from the reactor's reader list by passing it to\n        the reactor's C{removeReader} method.\n        "
        self.port.startListening()
        fileno = self.port.fileno()
        self._stopPort(self.port)
        self.assertNotIn(fileno, self.system._openFiles)

    def test_stopListeningUnsetsConnected(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        After the L{Deferred} returned by L{TuntapPort.stopListening} fires,\n        the C{connected} attribute of the port object is set to C{False}.\n        '
        self.port.startListening()
        self._stopPort(self.port)
        self.assertFalse(self.port.connected)

    def test_stopListeningStopsProtocol(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{TuntapPort.stopListening} calls C{doStop} on the protocol the port\n        was initialized with.\n        '
        self.port.startListening()
        self._stopPort(self.port)
        self.assertIsNone(self.protocol.transport)

    def test_stopListeningWhenStopped(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{TuntapPort.stopListening} returns a L{Deferred} which succeeds\n        immediately if it is called when the port is not listening.\n        '
        stopped = self.port.stopListening()
        self.assertIsNone(self.successResultOf(stopped))

    def test_multipleStopListening(self):
        if False:
            while True:
                i = 10
        '\n        It is safe and a no-op to call L{TuntapPort.stopListening} more than\n        once with no intervening L{TuntapPort.startListening} call.\n        '
        self.port.startListening()
        self.port.stopListening()
        second = self.port.stopListening()
        self.reactor.advance(0)
        self.assertIsNone(self.successResultOf(second))

    def test_loseConnection(self):
        if False:
            i = 10
            return i + 15
        '\n        L{TuntapPort.loseConnection} stops the port and is deprecated.\n        '
        self.port.startListening()
        self.port.loseConnection()
        self.reactor.advance(0)
        self.assertFalse(self.port.connected)
        warnings = self.flushWarnings([self.test_loseConnection])
        self.assertEqual(DeprecationWarning, warnings[0]['category'])
        self.assertEqual('twisted.pair.tuntap.TuntapPort.loseConnection was deprecated in Twisted 14.0.0; please use twisted.pair.tuntap.TuntapPort.stopListening instead', warnings[0]['message'])
        self.assertEqual(1, len(warnings))

    def _stopsReadingTest(self, style):
        if False:
            return 10
        '\n        Test that L{TuntapPort.doRead} has no side-effects under a certain\n        exception condition.\n\n        @param style: An exception instance to arrange for the (python wrapper\n            around the) underlying platform I{read} call to fail with.\n\n        @raise C{self.failureException}: If there are any observable\n            side-effects.\n        '
        self.port.startListening()
        tunnel = self.system.getTunnel(self.port)
        tunnel.nonBlockingExceptionStyle = style
        self.port.doRead()
        self.assertEqual([], self.protocol.received)

    def test_eagainStopsReading(self):
        if False:
            while True:
                i = 10
        '\n        Once L{TuntapPort.doRead} encounters an I{EAGAIN} errno from a C{read}\n        call, it returns.\n        '
        self._stopsReadingTest(Tunnel.EAGAIN_STYLE)

    def test_ewouldblockStopsReading(self):
        if False:
            i = 10
            return i + 15
        '\n        Once L{TuntapPort.doRead} encounters an I{EWOULDBLOCK} errno from a\n        C{read} call, it returns.\n        '
        self._stopsReadingTest(Tunnel.EWOULDBLOCK_STYLE)

    def test_eintrblockStopsReading(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Once L{TuntapPort.doRead} encounters an I{EINTR} errno from a C{read}\n        call, it returns.\n        '
        self._stopsReadingTest(Tunnel.EINTR_STYLE)

    def test_unhandledReadError(self):
        if False:
            return 10
        '\n        If L{Tuntap.doRead} encounters any exception other than one explicitly\n        handled by the code, the exception propagates to the caller.\n        '

        class UnexpectedException(Exception):
            pass
        self.assertRaises(UnexpectedException, self._stopsReadingTest, UnexpectedException())

    def test_unhandledEnvironmentReadError(self):
        if False:
            return 10
        '\n        Just like C{test_unhandledReadError}, but for the case where the\n        exception that is not explicitly handled happens to be of type\n        C{EnvironmentError} (C{OSError} or C{IOError}).\n        '
        self.assertRaises(IOError, self._stopsReadingTest, IOError(EPERM, 'Operation not permitted'))

    def test_doReadSmallDatagram(self):
        if False:
            print('Hello World!')
        "\n        L{TuntapPort.doRead} reads a datagram of fewer than\n        C{TuntapPort.maxPacketSize} from the port's file descriptor and passes\n        it to its protocol's C{datagramReceived} method.\n        "
        datagram = b'x' * (self.port.maxPacketSize - 1)
        self.port.startListening()
        tunnel = self.system.getTunnel(self.port)
        tunnel.readBuffer.append(datagram)
        self.port.doRead()
        self.assertEqual([datagram], self.protocol.received)

    def test_doReadLargeDatagram(self):
        if False:
            while True:
                i = 10
        "\n        L{TuntapPort.doRead} reads the first part of a datagram of more than\n        C{TuntapPort.maxPacketSize} from the port's file descriptor and passes\n        the truncated data to its protocol's C{datagramReceived} method.\n        "
        datagram = b'x' * self.port.maxPacketSize
        self.port.startListening()
        tunnel = self.system.getTunnel(self.port)
        tunnel.readBuffer.append(datagram + b'y')
        self.port.doRead()
        self.assertEqual([datagram], self.protocol.received)

    def test_doReadSeveralDatagrams(self):
        if False:
            return 10
        '\n        L{TuntapPort.doRead} reads several datagrams, of up to\n        C{TuntapPort.maxThroughput} bytes total, before returning.\n        '
        values = cycle(iterbytes(b'abcdefghijklmnopqrstuvwxyz'))
        total = 0
        datagrams = []
        while total < self.port.maxThroughput:
            datagrams.append(next(values) * self.port.maxPacketSize)
            total += self.port.maxPacketSize
        self.port.startListening()
        tunnel = self.system.getTunnel(self.port)
        tunnel.readBuffer.extend(datagrams)
        tunnel.readBuffer.append(b'excessive datagram, not to be read')
        self.port.doRead()
        self.assertEqual(datagrams, self.protocol.received)

    def _datagramReceivedException(self):
        if False:
            return 10
        '\n        Deliver some data to a L{TuntapPort} hooked up to an application\n        protocol that raises an exception from its C{datagramReceived} method.\n\n        @return: Whatever L{AttributeError} exceptions are logged.\n        '
        self.port.startListening()
        self.system.getTunnel(self.port).readBuffer.append(b'ping')
        self.protocol.received = None
        self.port.doRead()
        return self.flushLoggedErrors(AttributeError)

    def test_datagramReceivedException(self):
        if False:
            i = 10
            return i + 15
        "\n        If the protocol's C{datagramReceived} method raises an exception, the\n        exception is logged.\n        "
        errors = self._datagramReceivedException()
        self.assertEqual(1, len(errors))

    def test_datagramReceivedExceptionIdentifiesProtocol(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The exception raised by C{datagramReceived} is logged with a message\n        identifying the offending protocol.\n        '
        messages = []
        addObserver(messages.append)
        self.addCleanup(removeObserver, messages.append)
        self._datagramReceivedException()
        error = next((m for m in messages if m['isError']))
        message = textFromEventDict(error)
        self.assertEqual('Unhandled exception from %s.datagramReceived' % (fullyQualifiedName(self.protocol.__class__),), message.splitlines()[0])

    def test_write(self):
        if False:
            while True:
                i = 10
        '\n        L{TuntapPort.write} sends a datagram into the tunnel.\n        '
        datagram = b'a b c d e f g'
        self.port.startListening()
        self.port.write(datagram)
        self.assertEqual(self.system.getTunnel(self.port).writeBuffer, deque([datagram]))

    def test_interruptedWrite(self):
        if False:
            while True:
                i = 10
        '\n        If the platform write call is interrupted (causing the Python wrapper\n        to raise C{IOError} with errno set to C{EINTR}), the write is re-tried.\n        '
        self.port.startListening()
        tunnel = self.system.getTunnel(self.port)
        tunnel.pendingSignals.append(SIGINT)
        self.port.write(b'hello, world')
        self.assertEqual(deque([b'hello, world']), tunnel.writeBuffer)

    def test_unhandledWriteError(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Any exception raised by the underlying write call, except for EINTR, is\n        propagated to the caller.\n        '
        self.port.startListening()
        tunnel = self.system.getTunnel(self.port)
        self.assertRaises(IOError, self.port.write, b'x' * tunnel.SEND_BUFFER_SIZE + b'y')

    def test_writeSequence(self):
        if False:
            print('Hello World!')
        '\n        L{TuntapPort.writeSequence} sends a datagram into the tunnel by\n        concatenating the byte strings in the list passed to it.\n        '
        datagram = [b'a', b'b', b'c', b'd']
        self.port.startListening()
        self.port.writeSequence(datagram)
        self.assertEqual(self.system.getTunnel(self.port).writeBuffer, deque([b''.join(datagram)]))

    def test_getHost(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        L{TuntapPort.getHost} returns a L{TunnelAddress} including the tunnel's\n        type and name.\n        "
        self.port.startListening()
        address = self.port.getHost()
        self.assertEqual(TunnelAddress(self._tunnelTypeOnly(self.helper.TUNNEL_TYPE), self.system.getTunnel(self.port).name), address)

    def test_listeningString(self):
        if False:
            print('Hello World!')
        '\n        The string representation of a L{TuntapPort} instance includes the\n        tunnel type and interface and the protocol associated with the port.\n        '
        self.port.startListening()
        self.assertRegex(str(self.port), fullyQualifiedName(self.protocol.__class__))
        expected = ' listening on {}/{}>'.format(self._tunnelTypeOnly(self.helper.TUNNEL_TYPE).name, self.system.getTunnel(self.port).name)
        self.assertTrue(str(self.port).find(expected) != -1)

    def test_unlisteningString(self):
        if False:
            i = 10
            return i + 15
        '\n        The string representation of a L{TuntapPort} instance includes the\n        tunnel type and interface and the protocol associated with the port.\n        '
        self.assertRegex(str(self.port), fullyQualifiedName(self.protocol.__class__))
        expected = ' not listening on {}/{}>'.format(self._tunnelTypeOnly(self.helper.TUNNEL_TYPE).name, self.name)
        self.assertTrue(str(self.port).find(expected) != -1)

    def test_logPrefix(self):
        if False:
            print('Hello World!')
        '\n        L{TuntapPort.logPrefix} returns a string identifying the application\n        protocol and the type of tunnel.\n        '
        self.assertEqual('%s (%s)' % (self.protocol.__class__.__name__, self._tunnelTypeOnly(self.helper.TUNNEL_TYPE).name), self.port.logPrefix())

class TunnelAddressTests(SynchronousTestCase):
    """
    Tests for L{TunnelAddress}.
    """

    def test_interfaces(self):
        if False:
            while True:
                i = 10
        '\n        A L{TunnelAddress} instances provides L{IAddress}.\n        '
        self.assertTrue(verifyObject(IAddress, TunnelAddress(TunnelFlags.IFF_TAP, 'tap0')))

    def test_indexing(self):
        if False:
            i = 10
            return i + 15
        '\n        A L{TunnelAddress} instance can be indexed to retrieve either the byte\n        string C{"TUNTAP"} or the name of the tunnel interface, while\n        triggering a deprecation warning.\n        '
        address = TunnelAddress(TunnelFlags.IFF_TAP, 'tap0')
        self.assertEqual('TUNTAP', address[0])
        self.assertEqual('tap0', address[1])
        warnings = self.flushWarnings([self.test_indexing])
        message = 'TunnelAddress.__getitem__ is deprecated since Twisted 14.0.0  Use attributes instead.'
        self.assertEqual(DeprecationWarning, warnings[0]['category'])
        self.assertEqual(message, warnings[0]['message'])
        self.assertEqual(DeprecationWarning, warnings[1]['category'])
        self.assertEqual(message, warnings[1]['message'])
        self.assertEqual(2, len(warnings))

    def test_repr(self):
        if False:
            i = 10
            return i + 15
        '\n        The string representation of a L{TunnelAddress} instance includes the\n        class name and the values of the C{type} and C{name} attributes.\n        '
        self.assertRegex(repr(TunnelAddress(TunnelFlags.IFF_TUN, name=b'device')), "TunnelAddress type=IFF_TUN name=b'device'>")

class TunnelAddressEqualityTests(SynchronousTestCase):
    """
    Tests for the implementation of equality (C{==} and C{!=}) for
    L{TunnelAddress}.
    """

    def setUp(self):
        if False:
            return 10
        self.first = TunnelAddress(TunnelFlags.IFF_TUN, b'device')
        self.second = TunnelAddress(TunnelFlags.IFF_TUN | TunnelFlags.IFF_TUN, b'device')
        self.variedType = TunnelAddress(TunnelFlags.IFF_TAP, b'tap1')
        self.variedName = TunnelAddress(TunnelFlags.IFF_TUN, b'tun1')

    def test_selfComparesEqual(self):
        if False:
            while True:
                i = 10
        '\n        A L{TunnelAddress} compares equal to itself.\n        '
        self.assertTrue(self.first == self.first)

    def test_selfNotComparesNotEqual(self):
        if False:
            while True:
                i = 10
        "\n        A L{TunnelAddress} doesn't compare not equal to itself.\n        "
        self.assertFalse(self.first != self.first)

    def test_sameAttributesComparesEqual(self):
        if False:
            print('Hello World!')
        '\n        Two L{TunnelAddress} instances with the same value for the C{type} and\n        C{name} attributes compare equal to each other.\n        '
        self.assertTrue(self.first == self.second)

    def test_sameAttributesNotComparesNotEqual(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Two L{TunnelAddress} instances with the same value for the C{type} and\n        C{name} attributes don't compare not equal to each other.\n        "
        self.assertFalse(self.first != self.second)

    def test_differentTypeComparesNotEqual(self):
        if False:
            while True:
                i = 10
        "\n        Two L{TunnelAddress} instances that differ only by the value of their\n        type don't compare equal to each other.\n        "
        self.assertFalse(self.first == self.variedType)

    def test_differentTypeNotComparesEqual(self):
        if False:
            return 10
        '\n        Two L{TunnelAddress} instances that differ only by the value of their\n        type compare not equal to each other.\n        '
        self.assertTrue(self.first != self.variedType)

    def test_differentNameComparesNotEqual(self):
        if False:
            return 10
        "\n        Two L{TunnelAddress} instances that differ only by the value of their\n        name don't compare equal to each other.\n        "
        self.assertFalse(self.first == self.variedName)

    def test_differentNameNotComparesEqual(self):
        if False:
            print('Hello World!')
        '\n        Two L{TunnelAddress} instances that differ only by the value of their\n        name compare not equal to each other.\n        '
        self.assertTrue(self.first != self.variedName)

    def test_differentClassNotComparesEqual(self):
        if False:
            return 10
        "\n        A L{TunnelAddress} doesn't compare equal to an instance of another\n        class.\n        "
        self.assertFalse(self.first == self)

    def test_differentClassComparesNotEqual(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A L{TunnelAddress} compares not equal to an instance of another class.\n        '
        self.assertTrue(self.first != self)

@implementer(IRawPacketProtocol)
class IPRecordingProtocol(AbstractDatagramProtocol):
    """
    A protocol which merely records the datagrams delivered to it.
    """

    def startProtocol(self):
        if False:
            while True:
                i = 10
        self.received = []

    def datagramReceived(self, datagram, partial=False, dest=None, source=None, protocol=None):
        if False:
            for i in range(10):
                print('nop')
        self.received.append(datagram)

    def addProto(self, num, proto):
        if False:
            while True:
                i = 10
        pass

class TunTests(TunnelTestsMixin, SynchronousTestCase):
    """
    Tests for L{TuntapPort} when used to open a Linux I{tun} tunnel.
    """
    factory = Factory()
    factory.protocol = IPRecordingProtocol
    helper = TunHelper(None, None)

class EthernetRecordingProtocol(EthernetProtocol):
    """
    A protocol which merely records the datagrams delivered to it.
    """

    def startProtocol(self):
        if False:
            for i in range(10):
                print('nop')
        self.received = []

    def datagramReceived(self, datagram, partial=False):
        if False:
            print('Hello World!')
        self.received.append(datagram)

class TapTests(TunnelTestsMixin, SynchronousTestCase):
    """
    Tests for L{TuntapPort} when used to open a Linux I{tap} tunnel.
    """
    factory = Factory()
    factory.protocol = EthernetRecordingProtocol
    helper = TapHelper(None, None, pi=False)

class IOSystemTestsMixin:
    """
    Tests that apply to any L{_IInputOutputSystem} implementation.
    """

    def test_noSuchDevice(self):
        if False:
            return 10
        '\n        L{_IInputOutputSystem.open} raises L{OSError} when called with a\n        non-existent device path.\n        '
        system = self.createSystem()
        self.assertRaises(OSError, system.open, b'/dev/there-is-no-such-device-ever', os.O_RDWR)

class MemoryIOSystemTests(IOSystemTestsMixin, SynchronousTestCase, FakeDeviceTestsMixin):
    """
    General L{_IInputOutputSystem} tests applied to L{MemoryIOSystem}.
    """

class RealIOSystemTests(IOSystemTestsMixin, SynchronousTestCase, RealDeviceTestsMixin):
    """
    General L{_IInputOutputSystem} tests applied to L{_RealSystem}.
    """