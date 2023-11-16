"""
Tests for L{twisted.conch.telnet}.
"""
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest

@implementer(telnet.ITelnetProtocol)
class TestProtocol:
    localEnableable = ()
    remoteEnableable = ()

    def __init__(self):
        if False:
            print('Hello World!')
        self.data = b''
        self.subcmd = []
        self.calls = []
        self.enabledLocal = []
        self.enabledRemote = []
        self.disabledLocal = []
        self.disabledRemote = []

    def makeConnection(self, transport):
        if False:
            while True:
                i = 10
        d = transport.negotiationMap = {}
        d[b'\x12'] = self.neg_TEST_COMMAND
        d = transport.commandMap = transport.commandMap.copy()
        for cmd in ('EOR', 'NOP', 'DM', 'BRK', 'IP', 'AO', 'AYT', 'EC', 'EL', 'GA'):
            d[getattr(telnet, cmd)] = lambda arg, cmd=cmd: self.calls.append(cmd)

    def dataReceived(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.data += data

    def connectionLost(self, reason):
        if False:
            i = 10
            return i + 15
        pass

    def neg_TEST_COMMAND(self, payload):
        if False:
            print('Hello World!')
        self.subcmd = payload

    def enableLocal(self, option):
        if False:
            for i in range(10):
                print('nop')
        if option in self.localEnableable:
            self.enabledLocal.append(option)
            return True
        return False

    def disableLocal(self, option):
        if False:
            for i in range(10):
                print('nop')
        self.disabledLocal.append(option)

    def enableRemote(self, option):
        if False:
            return 10
        if option in self.remoteEnableable:
            self.enabledRemote.append(option)
            return True
        return False

    def disableRemote(self, option):
        if False:
            return 10
        self.disabledRemote.append(option)

    def connectionMade(self):
        if False:
            while True:
                i = 10
        pass

    def unhandledCommand(self, command, argument):
        if False:
            print('Hello World!')
        pass

    def unhandledSubnegotiation(self, command, data):
        if False:
            return 10
        pass

class InterfacesTests(unittest.TestCase):

    def test_interface(self):
        if False:
            print('Hello World!')
        '\n        L{telnet.TelnetProtocol} implements L{telnet.ITelnetProtocol}\n        '
        p = telnet.TelnetProtocol()
        verifyObject(telnet.ITelnetProtocol, p)

class TelnetTransportTests(unittest.TestCase):
    """
    Tests for L{telnet.TelnetTransport}.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.p = telnet.TelnetTransport(TestProtocol)
        self.t = proto_helpers.StringTransport()
        self.p.makeConnection(self.t)

    def testRegularBytes(self):
        if False:
            print('Hello World!')
        h = self.p.protocol
        L = [b'here are some bytes la la la', b'some more arrive here', b'lots of bytes to play with', b'la la la', b'ta de da', b'dum']
        for b in L:
            self.p.dataReceived(b)
        self.assertEqual(h.data, b''.join(L))

    def testNewlineHandling(self):
        if False:
            print('Hello World!')
        h = self.p.protocol
        L = [b'here is the first line\r\n', b'here is the second line\r\x00', b'here is the third line\r\n', b'here is the last line\r\x00']
        for b in L:
            self.p.dataReceived(b)
        self.assertEqual(h.data, L[0][:-2] + b'\n' + L[1][:-2] + b'\r' + L[2][:-2] + b'\n' + L[3][:-2] + b'\r')

    def testIACEscape(self):
        if False:
            print('Hello World!')
        h = self.p.protocol
        L = [b'here are some bytes\xff\xff with an embedded IAC', b'and here is a test of a border escape\xff', b'\xff did you get that IAC?']
        for b in L:
            self.p.dataReceived(b)
        self.assertEqual(h.data, b''.join(L).replace(b'\xff\xff', b'\xff'))

    def _simpleCommandTest(self, cmdName):
        if False:
            return 10
        h = self.p.protocol
        cmd = telnet.IAC + getattr(telnet, cmdName)
        L = [b"Here's some bytes, tra la la", b'But ono!' + cmd + b' an interrupt']
        for b in L:
            self.p.dataReceived(b)
        self.assertEqual(h.calls, [cmdName])
        self.assertEqual(h.data, b''.join(L).replace(cmd, b''))

    def testInterrupt(self):
        if False:
            for i in range(10):
                print('nop')
        self._simpleCommandTest('IP')

    def testEndOfRecord(self):
        if False:
            return 10
        self._simpleCommandTest('EOR')

    def testNoOperation(self):
        if False:
            for i in range(10):
                print('nop')
        self._simpleCommandTest('NOP')

    def testDataMark(self):
        if False:
            print('Hello World!')
        self._simpleCommandTest('DM')

    def testBreak(self):
        if False:
            while True:
                i = 10
        self._simpleCommandTest('BRK')

    def testAbortOutput(self):
        if False:
            i = 10
            return i + 15
        self._simpleCommandTest('AO')

    def testAreYouThere(self):
        if False:
            print('Hello World!')
        self._simpleCommandTest('AYT')

    def testEraseCharacter(self):
        if False:
            for i in range(10):
                print('nop')
        self._simpleCommandTest('EC')

    def testEraseLine(self):
        if False:
            for i in range(10):
                print('nop')
        self._simpleCommandTest('EL')

    def testGoAhead(self):
        if False:
            while True:
                i = 10
        self._simpleCommandTest('GA')

    def testSubnegotiation(self):
        if False:
            while True:
                i = 10
        h = self.p.protocol
        cmd = telnet.IAC + telnet.SB + b'\x12hello world' + telnet.IAC + telnet.SE
        L = [b'These are some bytes but soon' + cmd, b'there will be some more']
        for b in L:
            self.p.dataReceived(b)
        self.assertEqual(h.data, b''.join(L).replace(cmd, b''))
        self.assertEqual(h.subcmd, list(iterbytes(b'hello world')))

    def testSubnegotiationWithEmbeddedSE(self):
        if False:
            i = 10
            return i + 15
        h = self.p.protocol
        cmd = telnet.IAC + telnet.SB + b'\x12' + telnet.SE + telnet.IAC + telnet.SE
        L = [b'Some bytes are here' + cmd + b'and here', b'and here']
        for b in L:
            self.p.dataReceived(b)
        self.assertEqual(h.data, b''.join(L).replace(cmd, b''))
        self.assertEqual(h.subcmd, [telnet.SE])

    def testBoundarySubnegotiation(self):
        if False:
            return 10
        cmd = telnet.IAC + telnet.SB + b'\x12' + telnet.SE + b'hello' + telnet.IAC + telnet.SE
        for i in range(len(cmd)):
            h = self.p.protocol = TestProtocol()
            h.makeConnection(self.p)
            (a, b) = (cmd[:i], cmd[i:])
            L = [b'first part' + a, b + b'last part']
            for data in L:
                self.p.dataReceived(data)
            self.assertEqual(h.data, b''.join(L).replace(cmd, b''))
            self.assertEqual(h.subcmd, [telnet.SE] + list(iterbytes(b'hello')))

    def _enabledHelper(self, o, eL=[], eR=[], dL=[], dR=[]):
        if False:
            i = 10
            return i + 15
        self.assertEqual(o.enabledLocal, eL)
        self.assertEqual(o.enabledRemote, eR)
        self.assertEqual(o.disabledLocal, dL)
        self.assertEqual(o.disabledRemote, dR)

    def testRefuseWill(self):
        if False:
            for i in range(10):
                print('nop')
        cmd = telnet.IAC + telnet.WILL + b'\x12'
        data = b'surrounding bytes' + cmd + b'to spice things up'
        self.p.dataReceived(data)
        self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
        self.assertEqual(self.t.value(), telnet.IAC + telnet.DONT + b'\x12')
        self._enabledHelper(self.p.protocol)

    def testRefuseDo(self):
        if False:
            for i in range(10):
                print('nop')
        cmd = telnet.IAC + telnet.DO + b'\x12'
        data = b'surrounding bytes' + cmd + b'to spice things up'
        self.p.dataReceived(data)
        self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
        self.assertEqual(self.t.value(), telnet.IAC + telnet.WONT + b'\x12')
        self._enabledHelper(self.p.protocol)

    def testAcceptDo(self):
        if False:
            for i in range(10):
                print('nop')
        cmd = telnet.IAC + telnet.DO + b'\x19'
        data = b'padding' + cmd + b'trailer'
        h = self.p.protocol
        h.localEnableable = (b'\x19',)
        self.p.dataReceived(data)
        self.assertEqual(self.t.value(), telnet.IAC + telnet.WILL + b'\x19')
        self._enabledHelper(h, eL=[b'\x19'])

    def testAcceptWill(self):
        if False:
            i = 10
            return i + 15
        cmd = telnet.IAC + telnet.WILL + b'\x91'
        data = b'header' + cmd + b'padding'
        h = self.p.protocol
        h.remoteEnableable = (b'\x91',)
        self.p.dataReceived(data)
        self.assertEqual(self.t.value(), telnet.IAC + telnet.DO + b'\x91')
        self._enabledHelper(h, eR=[b'\x91'])

    def testAcceptWont(self):
        if False:
            while True:
                i = 10
        cmd = telnet.IAC + telnet.WONT + b')'
        s = self.p.getOptionState(b')')
        s.him.state = 'yes'
        data = b'fiddle dee' + cmd
        self.p.dataReceived(data)
        self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
        self.assertEqual(self.t.value(), telnet.IAC + telnet.DONT + b')')
        self.assertEqual(s.him.state, 'no')
        self._enabledHelper(self.p.protocol, dR=[b')'])

    def testAcceptDont(self):
        if False:
            i = 10
            return i + 15
        cmd = telnet.IAC + telnet.DONT + b')'
        s = self.p.getOptionState(b')')
        s.us.state = 'yes'
        data = b'fiddle dum ' + cmd
        self.p.dataReceived(data)
        self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
        self.assertEqual(self.t.value(), telnet.IAC + telnet.WONT + b')')
        self.assertEqual(s.us.state, 'no')
        self._enabledHelper(self.p.protocol, dL=[b')'])

    def testIgnoreWont(self):
        if False:
            while True:
                i = 10
        cmd = telnet.IAC + telnet.WONT + b'G'
        data = b'dum de dum' + cmd + b'tra la la'
        self.p.dataReceived(data)
        self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
        self.assertEqual(self.t.value(), b'')
        self._enabledHelper(self.p.protocol)

    def testIgnoreDont(self):
        if False:
            i = 10
            return i + 15
        cmd = telnet.IAC + telnet.DONT + b'G'
        data = b'dum de dum' + cmd + b'tra la la'
        self.p.dataReceived(data)
        self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
        self.assertEqual(self.t.value(), b'')
        self._enabledHelper(self.p.protocol)

    def testIgnoreWill(self):
        if False:
            while True:
                i = 10
        cmd = telnet.IAC + telnet.WILL + b'V'
        s = self.p.getOptionState(b'V')
        s.him.state = 'yes'
        data = b'tra la la' + cmd + b'dum de dum'
        self.p.dataReceived(data)
        self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
        self.assertEqual(self.t.value(), b'')
        self._enabledHelper(self.p.protocol)

    def testIgnoreDo(self):
        if False:
            return 10
        cmd = telnet.IAC + telnet.DO + b'V'
        s = self.p.getOptionState(b'V')
        s.us.state = 'yes'
        data = b'tra la la' + cmd + b'dum de dum'
        self.p.dataReceived(data)
        self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
        self.assertEqual(self.t.value(), b'')
        self._enabledHelper(self.p.protocol)

    def testAcceptedEnableRequest(self):
        if False:
            return 10
        d = self.p.do(b'B')
        h = self.p.protocol
        h.remoteEnableable = (b'B',)
        self.assertEqual(self.t.value(), telnet.IAC + telnet.DO + b'B')
        self.p.dataReceived(telnet.IAC + telnet.WILL + b'B')
        d.addCallback(self.assertEqual, True)
        d.addCallback(lambda _: self._enabledHelper(h, eR=[b'B']))
        return d

    def test_refusedEnableRequest(self):
        if False:
            return 10
        '\n        If the peer refuses to enable an option we request it to enable, the\n        L{Deferred} returned by L{TelnetProtocol.do} fires with an\n        L{OptionRefused} L{Failure}.\n        '
        self.p.protocol.remoteEnableable = (b'B',)
        d = self.p.do(b'B')
        self.assertEqual(self.t.value(), telnet.IAC + telnet.DO + b'B')
        s = self.p.getOptionState(b'B')
        self.assertEqual(s.him.state, 'no')
        self.assertEqual(s.us.state, 'no')
        self.assertTrue(s.him.negotiating)
        self.assertFalse(s.us.negotiating)
        self.p.dataReceived(telnet.IAC + telnet.WONT + b'B')
        d = self.assertFailure(d, telnet.OptionRefused)
        d.addCallback(lambda ignored: self._enabledHelper(self.p.protocol))
        d.addCallback(lambda ignored: self.assertFalse(s.him.negotiating))
        return d

    def test_refusedEnableOffer(self):
        if False:
            while True:
                i = 10
        '\n        If the peer refuses to allow us to enable an option, the L{Deferred}\n        returned by L{TelnetProtocol.will} fires with an L{OptionRefused}\n        L{Failure}.\n        '
        self.p.protocol.localEnableable = (b'B',)
        d = self.p.will(b'B')
        self.assertEqual(self.t.value(), telnet.IAC + telnet.WILL + b'B')
        s = self.p.getOptionState(b'B')
        self.assertEqual(s.him.state, 'no')
        self.assertEqual(s.us.state, 'no')
        self.assertFalse(s.him.negotiating)
        self.assertTrue(s.us.negotiating)
        self.p.dataReceived(telnet.IAC + telnet.DONT + b'B')
        d = self.assertFailure(d, telnet.OptionRefused)
        d.addCallback(lambda ignored: self._enabledHelper(self.p.protocol))
        d.addCallback(lambda ignored: self.assertFalse(s.us.negotiating))
        return d

    def testAcceptedDisableRequest(self):
        if False:
            print('Hello World!')
        s = self.p.getOptionState(b'B')
        s.him.state = 'yes'
        d = self.p.dont(b'B')
        self.assertEqual(self.t.value(), telnet.IAC + telnet.DONT + b'B')
        self.p.dataReceived(telnet.IAC + telnet.WONT + b'B')
        d.addCallback(self.assertEqual, True)
        d.addCallback(lambda _: self._enabledHelper(self.p.protocol, dR=[b'B']))
        return d

    def testNegotiationBlocksFurtherNegotiation(self):
        if False:
            for i in range(10):
                print('nop')
        s = self.p.getOptionState(b'$')
        s.him.state = 'yes'
        self.p.dont(b'$')

        def _do(x):
            if False:
                return 10
            d = self.p.do(b'$')
            return self.assertFailure(d, telnet.AlreadyNegotiating)

        def _dont(x):
            if False:
                return 10
            d = self.p.dont(b'$')
            return self.assertFailure(d, telnet.AlreadyNegotiating)

        def _final(x):
            if False:
                while True:
                    i = 10
            self.p.dataReceived(telnet.IAC + telnet.WONT + b'$')
            self._enabledHelper(self.p.protocol, dR=[b'$'])
            self.p.protocol.remoteEnableable = (b'$',)
            d = self.p.do(b'$')
            self.p.dataReceived(telnet.IAC + telnet.WILL + b'$')
            d.addCallback(self.assertEqual, True)
            d.addCallback(lambda _: self._enabledHelper(self.p.protocol, eR=[b'$'], dR=[b'$']))
            return d
        d = _do(None)
        d.addCallback(_dont)
        d.addCallback(_final)
        return d

    def testSuperfluousDisableRequestRaises(self):
        if False:
            for i in range(10):
                print('nop')
        d = self.p.dont(b'\xab')
        return self.assertFailure(d, telnet.AlreadyDisabled)

    def testSuperfluousEnableRequestRaises(self):
        if False:
            return 10
        s = self.p.getOptionState(b'\xab')
        s.him.state = 'yes'
        d = self.p.do(b'\xab')
        return self.assertFailure(d, telnet.AlreadyEnabled)

    def testLostConnectionFailsDeferreds(self):
        if False:
            while True:
                i = 10
        d1 = self.p.do(b'\x12')
        d2 = self.p.do(b'#')
        d3 = self.p.do(b'4')

        class TestException(Exception):
            pass
        self.p.connectionLost(TestException('Total failure!'))
        d1 = self.assertFailure(d1, TestException)
        d2 = self.assertFailure(d2, TestException)
        d3 = self.assertFailure(d3, TestException)
        return defer.gatherResults([d1, d2, d3])

class TestTelnet(telnet.Telnet):
    """
    A trivial extension of the telnet protocol class useful to unit tests.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        telnet.Telnet.__init__(self)
        self.events = []

    def applicationDataReceived(self, data):
        if False:
            print('Hello World!')
        '\n        Record the given data in C{self.events}.\n        '
        self.events.append(('bytes', data))

    def unhandledCommand(self, command, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Record the given command in C{self.events}.\n        '
        self.events.append(('command', command, data))

    def unhandledSubnegotiation(self, command, data):
        if False:
            print('Hello World!')
        '\n        Record the given subnegotiation command in C{self.events}.\n        '
        self.events.append(('negotiate', command, data))

class TelnetTests(unittest.TestCase):
    """
    Tests for L{telnet.Telnet}.

    L{telnet.Telnet} implements the TELNET protocol (RFC 854), including option
    and suboption negotiation, and option state tracking.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create an unconnected L{telnet.Telnet} to be used by tests.\n        '
        self.protocol = TestTelnet()

    def test_enableLocal(self):
        if False:
            print('Hello World!')
        '\n        L{telnet.Telnet.enableLocal} should reject all options, since\n        L{telnet.Telnet} does not know how to implement any options.\n        '
        self.assertFalse(self.protocol.enableLocal(b'\x00'))

    def test_enableRemote(self):
        if False:
            return 10
        '\n        L{telnet.Telnet.enableRemote} should reject all options, since\n        L{telnet.Telnet} does not know how to implement any options.\n        '
        self.assertFalse(self.protocol.enableRemote(b'\x00'))

    def test_disableLocal(self):
        if False:
            while True:
                i = 10
        '\n        It is an error for L{telnet.Telnet.disableLocal} to be called, since\n        L{telnet.Telnet.enableLocal} will never allow any options to be enabled\n        locally.  If a subclass overrides enableLocal, it must also override\n        disableLocal.\n        '
        self.assertRaises(NotImplementedError, self.protocol.disableLocal, b'\x00')

    def test_disableRemote(self):
        if False:
            while True:
                i = 10
        '\n        It is an error for L{telnet.Telnet.disableRemote} to be called, since\n        L{telnet.Telnet.enableRemote} will never allow any options to be\n        enabled remotely.  If a subclass overrides enableRemote, it must also\n        override disableRemote.\n        '
        self.assertRaises(NotImplementedError, self.protocol.disableRemote, b'\x00')

    def test_requestNegotiation(self):
        if False:
            while True:
                i = 10
        '\n        L{telnet.Telnet.requestNegotiation} formats the feature byte and the\n        payload bytes into the subnegotiation format and sends them.\n\n        See RFC 855.\n        '
        transport = proto_helpers.StringTransport()
        self.protocol.makeConnection(transport)
        self.protocol.requestNegotiation(b'\x01', b'\x02\x03')
        self.assertEqual(transport.value(), b'\xff\xfa\x01\x02\x03\xff\xf0')

    def test_requestNegotiationEscapesIAC(self):
        if False:
            return 10
        '\n        If the payload for a subnegotiation includes I{IAC}, it is escaped by\n        L{telnet.Telnet.requestNegotiation} with another I{IAC}.\n\n        See RFC 855.\n        '
        transport = proto_helpers.StringTransport()
        self.protocol.makeConnection(transport)
        self.protocol.requestNegotiation(b'\x01', b'\xff')
        self.assertEqual(transport.value(), b'\xff\xfa\x01\xff\xff\xff\xf0')

    def _deliver(self, data, *expected):
        if False:
            while True:
                i = 10
        "\n        Pass the given bytes to the protocol's C{dataReceived} method and\n        assert that the given events occur.\n        "
        received = self.protocol.events = []
        self.protocol.dataReceived(data)
        self.assertEqual(received, list(expected))

    def test_oneApplicationDataByte(self):
        if False:
            while True:
                i = 10
        '\n        One application-data byte in the default state gets delivered right\n        away.\n        '
        self._deliver(b'a', ('bytes', b'a'))

    def test_twoApplicationDataBytes(self):
        if False:
            print('Hello World!')
        '\n        Two application-data bytes in the default state get delivered\n        together.\n        '
        self._deliver(b'bc', ('bytes', b'bc'))

    def test_threeApplicationDataBytes(self):
        if False:
            while True:
                i = 10
        "\n        Three application-data bytes followed by a control byte get\n        delivered, but the control byte doesn't.\n        "
        self._deliver(b'def' + telnet.IAC, ('bytes', b'def'))

    def test_escapedControl(self):
        if False:
            print('Hello World!')
        '\n        IAC in the escaped state gets delivered and so does another\n        application-data byte following it.\n        '
        self._deliver(telnet.IAC)
        self._deliver(telnet.IAC + b'g', ('bytes', telnet.IAC + b'g'))

    def test_carriageReturn(self):
        if False:
            return 10
        '\n        A carriage return only puts the protocol into the newline state.  A\n        linefeed in the newline state causes just the newline to be\n        delivered.  A nul in the newline state causes a carriage return to\n        be delivered.  An IAC in the newline state causes a carriage return\n        to be delivered and puts the protocol into the escaped state.\n        Anything else causes a carriage return and that thing to be\n        delivered.\n        '
        self._deliver(b'\r')
        self._deliver(b'\n', ('bytes', b'\n'))
        self._deliver(b'\r\n', ('bytes', b'\n'))
        self._deliver(b'\r')
        self._deliver(b'\x00', ('bytes', b'\r'))
        self._deliver(b'\r\x00', ('bytes', b'\r'))
        self._deliver(b'\r')
        self._deliver(b'a', ('bytes', b'\ra'))
        self._deliver(b'\ra', ('bytes', b'\ra'))
        self._deliver(b'\r')
        self._deliver(telnet.IAC + telnet.IAC + b'x', ('bytes', b'\r' + telnet.IAC + b'x'))

    def test_applicationDataBeforeSimpleCommand(self):
        if False:
            return 10
        '\n        Application bytes received before a command are delivered before the\n        command is processed.\n        '
        self._deliver(b'x' + telnet.IAC + telnet.NOP, ('bytes', b'x'), ('command', telnet.NOP, None))

    def test_applicationDataBeforeCommand(self):
        if False:
            return 10
        '\n        Application bytes received before a WILL/WONT/DO/DONT are delivered\n        before the command is processed.\n        '
        self.protocol.commandMap = {}
        self._deliver(b'y' + telnet.IAC + telnet.WILL + b'\x00', ('bytes', b'y'), ('command', telnet.WILL, b'\x00'))

    def test_applicationDataBeforeSubnegotiation(self):
        if False:
            i = 10
            return i + 15
        '\n        Application bytes received before a subnegotiation command are\n        delivered before the negotiation is processed.\n        '
        self._deliver(b'z' + telnet.IAC + telnet.SB + b'Qx' + telnet.IAC + telnet.SE, ('bytes', b'z'), ('negotiate', b'Q', [b'x']))