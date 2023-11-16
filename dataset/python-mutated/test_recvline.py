"""
Tests for L{twisted.conch.recvline} and fixtures for testing related
functionality.
"""
import os
import sys
from unittest import skipIf
from twisted.conch import recvline
from twisted.conch.insults import insults
from twisted.cred import portal
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.python import components, filepath, reflect
from twisted.python.compat import iterbytes
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
stdio = requireModule('twisted.conch.stdio')
properEnv = dict(os.environ)
properEnv['PYTHONPATH'] = os.pathsep.join(sys.path)

class ArrowsTests(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.underlyingTransport = StringTransport()
        self.pt = insults.ServerProtocol()
        self.p = recvline.HistoricRecvLine()
        self.pt.protocolFactory = lambda : self.p
        self.pt.factory = self
        self.pt.makeConnection(self.underlyingTransport)

    def test_printableCharacters(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When L{HistoricRecvLine} receives a printable character,\n        it adds it to the current line buffer.\n        '
        self.p.keystrokeReceived(b'x', None)
        self.p.keystrokeReceived(b'y', None)
        self.p.keystrokeReceived(b'z', None)
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))

    def test_horizontalArrows(self):
        if False:
            return 10
        '\n        When L{HistoricRecvLine} receives a LEFT_ARROW or\n        RIGHT_ARROW keystroke it moves the cursor left or right\n        in the current line buffer, respectively.\n        '
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'xyz'):
            kR(ch)
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
        kR(self.pt.RIGHT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
        kR(self.pt.LEFT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'xy', b'z'))
        kR(self.pt.LEFT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'x', b'yz'))
        kR(self.pt.LEFT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'', b'xyz'))
        kR(self.pt.LEFT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'', b'xyz'))
        kR(self.pt.RIGHT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'x', b'yz'))
        kR(self.pt.RIGHT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'xy', b'z'))
        kR(self.pt.RIGHT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
        kR(self.pt.RIGHT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))

    def test_newline(self):
        if False:
            return 10
        '\n        When {HistoricRecvLine} receives a newline, it adds the current\n        line buffer to the end of its history buffer.\n        '
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'xyz\nabc\n123\n'):
            kR(ch)
        self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz', b'abc', b'123'), ()))
        kR(b'c')
        kR(b'b')
        kR(b'a')
        self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz', b'abc', b'123'), ()))
        kR(b'\n')
        self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz', b'abc', b'123', b'cba'), ()))

    def test_verticalArrows(self):
        if False:
            while True:
                i = 10
        '\n        When L{HistoricRecvLine} receives UP_ARROW or DOWN_ARROW\n        keystrokes it move the current index in the current history\n        buffer up or down, and resets the current line buffer to the\n        previous or next line in history, respectively for each.\n        '
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'xyz\nabc\n123\n'):
            kR(ch)
        self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz', b'abc', b'123'), ()))
        self.assertEqual(self.p.currentLineBuffer(), (b'', b''))
        kR(self.pt.UP_ARROW)
        self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz', b'abc'), (b'123',)))
        self.assertEqual(self.p.currentLineBuffer(), (b'123', b''))
        kR(self.pt.UP_ARROW)
        self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz',), (b'abc', b'123')))
        self.assertEqual(self.p.currentLineBuffer(), (b'abc', b''))
        kR(self.pt.UP_ARROW)
        self.assertEqual(self.p.currentHistoryBuffer(), ((), (b'xyz', b'abc', b'123')))
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
        kR(self.pt.UP_ARROW)
        self.assertEqual(self.p.currentHistoryBuffer(), ((), (b'xyz', b'abc', b'123')))
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
        for i in range(4):
            kR(self.pt.DOWN_ARROW)
        self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz', b'abc', b'123'), ()))

    def test_home(self):
        if False:
            return 10
        '\n        When L{HistoricRecvLine} receives a HOME keystroke it moves the\n        cursor to the beginning of the current line buffer.\n        '
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'hello, world'):
            kR(ch)
        self.assertEqual(self.p.currentLineBuffer(), (b'hello, world', b''))
        kR(self.pt.HOME)
        self.assertEqual(self.p.currentLineBuffer(), (b'', b'hello, world'))

    def test_end(self):
        if False:
            i = 10
            return i + 15
        '\n        When L{HistoricRecvLine} receives an END keystroke it moves the cursor\n        to the end of the current line buffer.\n        '
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'hello, world'):
            kR(ch)
        self.assertEqual(self.p.currentLineBuffer(), (b'hello, world', b''))
        kR(self.pt.HOME)
        kR(self.pt.END)
        self.assertEqual(self.p.currentLineBuffer(), (b'hello, world', b''))

    def test_backspace(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When L{HistoricRecvLine} receives a BACKSPACE keystroke it deletes\n        the character immediately before the cursor.\n        '
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'xyz'):
            kR(ch)
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
        kR(self.pt.BACKSPACE)
        self.assertEqual(self.p.currentLineBuffer(), (b'xy', b''))
        kR(self.pt.LEFT_ARROW)
        kR(self.pt.BACKSPACE)
        self.assertEqual(self.p.currentLineBuffer(), (b'', b'y'))
        kR(self.pt.BACKSPACE)
        self.assertEqual(self.p.currentLineBuffer(), (b'', b'y'))

    def test_delete(self):
        if False:
            print('Hello World!')
        '\n        When L{HistoricRecvLine} receives a DELETE keystroke, it\n        delets the character immediately after the cursor.\n        '
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'xyz'):
            kR(ch)
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
        kR(self.pt.DELETE)
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
        kR(self.pt.LEFT_ARROW)
        kR(self.pt.DELETE)
        self.assertEqual(self.p.currentLineBuffer(), (b'xy', b''))
        kR(self.pt.LEFT_ARROW)
        kR(self.pt.DELETE)
        self.assertEqual(self.p.currentLineBuffer(), (b'x', b''))
        kR(self.pt.LEFT_ARROW)
        kR(self.pt.DELETE)
        self.assertEqual(self.p.currentLineBuffer(), (b'', b''))
        kR(self.pt.DELETE)
        self.assertEqual(self.p.currentLineBuffer(), (b'', b''))

    def test_insert(self):
        if False:
            return 10
        '\n        When not in INSERT mode, L{HistoricRecvLine} inserts the typed\n        character at the cursor before the next character.\n        '
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'xyz'):
            kR(ch)
        kR(self.pt.LEFT_ARROW)
        kR(b'A')
        self.assertEqual(self.p.currentLineBuffer(), (b'xyA', b'z'))
        kR(self.pt.LEFT_ARROW)
        kR(b'B')
        self.assertEqual(self.p.currentLineBuffer(), (b'xyB', b'Az'))

    def test_typeover(self):
        if False:
            return 10
        '\n        When in INSERT mode and upon receiving a keystroke with a printable\n        character, L{HistoricRecvLine} replaces the character at\n        the cursor with the typed character rather than inserting before.\n        Ah, the ironies of INSERT mode.\n        '
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'xyz'):
            kR(ch)
        kR(self.pt.INSERT)
        kR(self.pt.LEFT_ARROW)
        kR(b'A')
        self.assertEqual(self.p.currentLineBuffer(), (b'xyA', b''))
        kR(self.pt.LEFT_ARROW)
        kR(b'B')
        self.assertEqual(self.p.currentLineBuffer(), (b'xyB', b''))

    def test_unprintableCharacters(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When L{HistoricRecvLine} receives a keystroke for an unprintable\n        function key with no assigned behavior, the line buffer is unmodified.\n        '
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        pt = self.pt
        for ch in (pt.F1, pt.F2, pt.F3, pt.F4, pt.F5, pt.F6, pt.F7, pt.F8, pt.F9, pt.F10, pt.F11, pt.F12, pt.PGUP, pt.PGDN):
            kR(ch)
            self.assertEqual(self.p.currentLineBuffer(), (b'', b''))
from twisted.conch import telnet
from twisted.conch.insults import helper
from twisted.conch.test.loopback import LoopbackRelay

class EchoServer(recvline.HistoricRecvLine):

    def lineReceived(self, line):
        if False:
            for i in range(10):
                print('nop')
        self.terminal.write(line + b'\n' + self.ps[self.pn])
left = b'\x1b[D'
right = b'\x1b[C'
up = b'\x1b[A'
down = b'\x1b[B'
insert = b'\x1b[2~'
home = b'\x1b[1~'
delete = b'\x1b[3~'
end = b'\x1b[4~'
backspace = b'\x7f'
from twisted.cred import checkers
try:
    from twisted.conch.manhole_ssh import ConchFactory, TerminalRealm, TerminalSession, TerminalSessionTransport, TerminalUser
    from twisted.conch.ssh import channel, connection, keys, session, transport, userauth
except ImportError:
    ssh = False
else:
    ssh = True

    class SessionChannel(channel.SSHChannel):
        name = b'session'

        def __init__(self, protocolFactory, protocolArgs, protocolKwArgs, width, height, *a, **kw):
            if False:
                i = 10
                return i + 15
            channel.SSHChannel.__init__(self, *a, **kw)
            self.protocolFactory = protocolFactory
            self.protocolArgs = protocolArgs
            self.protocolKwArgs = protocolKwArgs
            self.width = width
            self.height = height

        def channelOpen(self, data):
            if False:
                i = 10
                return i + 15
            term = session.packRequest_pty_req(b'vt102', (self.height, self.width, 0, 0), b'')
            self.conn.sendRequest(self, b'pty-req', term)
            self.conn.sendRequest(self, b'shell', b'')
            self._protocolInstance = self.protocolFactory(*self.protocolArgs, **self.protocolKwArgs)
            self._protocolInstance.factory = self
            self._protocolInstance.makeConnection(self)

        def closed(self):
            if False:
                for i in range(10):
                    print('nop')
            self._protocolInstance.connectionLost(error.ConnectionDone())

        def dataReceived(self, data):
            if False:
                return 10
            self._protocolInstance.dataReceived(data)

    class TestConnection(connection.SSHConnection):

        def __init__(self, protocolFactory, protocolArgs, protocolKwArgs, width, height, *a, **kw):
            if False:
                while True:
                    i = 10
            connection.SSHConnection.__init__(self, *a, **kw)
            self.protocolFactory = protocolFactory
            self.protocolArgs = protocolArgs
            self.protocolKwArgs = protocolKwArgs
            self.width = width
            self.height = height

        def serviceStarted(self):
            if False:
                print('Hello World!')
            self.__channel = SessionChannel(self.protocolFactory, self.protocolArgs, self.protocolKwArgs, self.width, self.height)
            self.openChannel(self.__channel)

        def write(self, data):
            if False:
                print('Hello World!')
            return self.__channel.write(data)

    class TestAuth(userauth.SSHUserAuthClient):

        def __init__(self, username, password, *a, **kw):
            if False:
                return 10
            userauth.SSHUserAuthClient.__init__(self, username, *a, **kw)
            self.password = password

        def getPassword(self):
            if False:
                while True:
                    i = 10
            return defer.succeed(self.password)

    class TestTransport(transport.SSHClientTransport):

        def __init__(self, protocolFactory, protocolArgs, protocolKwArgs, username, password, width, height, *a, **kw):
            if False:
                while True:
                    i = 10
            self.protocolFactory = protocolFactory
            self.protocolArgs = protocolArgs
            self.protocolKwArgs = protocolKwArgs
            self.username = username
            self.password = password
            self.width = width
            self.height = height

        def verifyHostKey(self, hostKey, fingerprint):
            if False:
                for i in range(10):
                    print('nop')
            return defer.succeed(True)

        def connectionSecure(self):
            if False:
                while True:
                    i = 10
            self.__connection = TestConnection(self.protocolFactory, self.protocolArgs, self.protocolKwArgs, self.width, self.height)
            self.requestService(TestAuth(self.username, self.password, self.__connection))

        def write(self, data):
            if False:
                i = 10
                return i + 15
            return self.__connection.write(data)

    class TestSessionTransport(TerminalSessionTransport):

        def protocolFactory(self):
            if False:
                print('Hello World!')
            return self.avatar.conn.transport.factory.serverProtocol()

    class TestSession(TerminalSession):
        transportFactory = TestSessionTransport

    class TestUser(TerminalUser):
        pass
    components.registerAdapter(TestSession, TestUser, session.ISession)

class NotifyingExpectableBuffer(helper.ExpectableBuffer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.onConnection = defer.Deferred()
        self.onDisconnection = defer.Deferred()

    def connectionMade(self):
        if False:
            while True:
                i = 10
        helper.ExpectableBuffer.connectionMade(self)
        self.onConnection.callback(self)

    def connectionLost(self, reason):
        if False:
            print('Hello World!')
        self.onDisconnection.errback(reason)

class _BaseMixin:
    WIDTH = 80
    HEIGHT = 24

    def _assertBuffer(self, lines):
        if False:
            return 10
        receivedLines = self.recvlineClient.__bytes__().splitlines()
        expectedLines = lines + [b''] * (self.HEIGHT - len(lines) - 1)
        self.assertEqual(receivedLines, expectedLines)

    def _trivialTest(self, inputLine, output):
        if False:
            i = 10
            return i + 15
        done = self.recvlineClient.expect(b'done')
        self._testwrite(inputLine)

        def finished(ign):
            if False:
                print('Hello World!')
            self._assertBuffer(output)
        return done.addCallback(finished)

class _SSHMixin(_BaseMixin):

    def setUp(self):
        if False:
            while True:
                i = 10
        if not ssh:
            raise SkipTest("cryptography requirements missing, can't run historic recvline tests over ssh")
        (u, p) = (b'testuser', b'testpass')
        rlm = TerminalRealm()
        rlm.userFactory = TestUser
        rlm.chainedProtocolFactory = lambda : insultsServer
        checker = checkers.InMemoryUsernamePasswordDatabaseDontUse()
        checker.addUser(u, p)
        ptl = portal.Portal(rlm)
        ptl.registerChecker(checker)
        sshFactory = ConchFactory(ptl)
        sshKey = keys._getPersistentRSAKey(filepath.FilePath(self.mktemp()), keySize=1024)
        sshFactory.publicKeys[b'ssh-rsa'] = sshKey
        sshFactory.privateKeys[b'ssh-rsa'] = sshKey
        sshFactory.serverProtocol = self.serverProtocol
        sshFactory.startFactory()
        recvlineServer = self.serverProtocol()
        insultsServer = insults.ServerProtocol(lambda : recvlineServer)
        sshServer = sshFactory.buildProtocol(None)
        clientTransport = LoopbackRelay(sshServer)
        recvlineClient = NotifyingExpectableBuffer()
        insultsClient = insults.ClientProtocol(lambda : recvlineClient)
        sshClient = TestTransport(lambda : insultsClient, (), {}, u, p, self.WIDTH, self.HEIGHT)
        serverTransport = LoopbackRelay(sshClient)
        sshClient.makeConnection(clientTransport)
        sshServer.makeConnection(serverTransport)
        self.recvlineClient = recvlineClient
        self.sshClient = sshClient
        self.sshServer = sshServer
        self.clientTransport = clientTransport
        self.serverTransport = serverTransport
        return recvlineClient.onConnection

    def _testwrite(self, data):
        if False:
            print('Hello World!')
        self.sshClient.write(data)
from twisted.conch.test import test_telnet

class TestInsultsClientProtocol(insults.ClientProtocol, test_telnet.TestProtocol):
    pass

class TestInsultsServerProtocol(insults.ServerProtocol, test_telnet.TestProtocol):
    pass

class _TelnetMixin(_BaseMixin):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        recvlineServer = self.serverProtocol()
        insultsServer = TestInsultsServerProtocol(lambda : recvlineServer)
        telnetServer = telnet.TelnetTransport(lambda : insultsServer)
        clientTransport = LoopbackRelay(telnetServer)
        recvlineClient = NotifyingExpectableBuffer()
        insultsClient = TestInsultsClientProtocol(lambda : recvlineClient)
        telnetClient = telnet.TelnetTransport(lambda : insultsClient)
        serverTransport = LoopbackRelay(telnetClient)
        telnetClient.makeConnection(clientTransport)
        telnetServer.makeConnection(serverTransport)
        serverTransport.clearBuffer()
        clientTransport.clearBuffer()
        self.recvlineClient = recvlineClient
        self.telnetClient = telnetClient
        self.clientTransport = clientTransport
        self.serverTransport = serverTransport
        return recvlineClient.onConnection

    def _testwrite(self, data):
        if False:
            while True:
                i = 10
        self.telnetClient.write(data)

class _StdioMixin(_BaseMixin):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        testTerminal = NotifyingExpectableBuffer()
        insultsClient = insults.ClientProtocol(lambda : testTerminal)
        processClient = stdio.TerminalProcessProtocol(insultsClient)
        exe = sys.executable
        module = stdio.__file__
        if module.endswith('.pyc') or module.endswith('.pyo'):
            module = module[:-1]
        args = [exe, module, reflect.qual(self.serverProtocol)]
        from twisted.internet import reactor
        clientTransport = reactor.spawnProcess(processClient, exe, args, env=properEnv, usePTY=True)
        self.recvlineClient = self.testTerminal = testTerminal
        self.processClient = processClient
        self.clientTransport = clientTransport
        return defer.gatherResults(filter(None, [processClient.onConnection, testTerminal.expect(b'>>> ')]))

    def tearDown(self):
        if False:
            while True:
                i = 10
        try:
            self.clientTransport.signalProcess('KILL')
        except (error.ProcessExitedAlready, OSError):
            pass

        def trap(failure):
            if False:
                return 10
            failure.trap(error.ProcessTerminated)
            self.assertIsNone(failure.value.exitCode)
            self.assertEqual(failure.value.status, 9)
        return self.testTerminal.onDisconnection.addErrback(trap)

    def _testwrite(self, data):
        if False:
            return 10
        self.clientTransport.write(data)

class RecvlineLoopbackMixin:
    serverProtocol = EchoServer

    def testSimple(self):
        if False:
            print('Hello World!')
        return self._trivialTest(b'first line\ndone', [b'>>> first line', b'first line', b'>>> done'])

    def testLeftArrow(self):
        if False:
            for i in range(10):
                print('nop')
        return self._trivialTest(insert + b'first line' + left * 4 + b'xxxx\ndone', [b'>>> first xxxx', b'first xxxx', b'>>> done'])

    def testRightArrow(self):
        if False:
            for i in range(10):
                print('nop')
        return self._trivialTest(insert + b'right line' + left * 4 + right * 2 + b'xx\ndone', [b'>>> right lixx', b'right lixx', b'>>> done'])

    def testBackspace(self):
        if False:
            i = 10
            return i + 15
        return self._trivialTest(b'second line' + backspace * 4 + b'xxxx\ndone', [b'>>> second xxxx', b'second xxxx', b'>>> done'])

    def testDelete(self):
        if False:
            i = 10
            return i + 15
        return self._trivialTest(b'delete xxxx' + left * 4 + delete * 4 + b'line\ndone', [b'>>> delete line', b'delete line', b'>>> done'])

    def testInsert(self):
        if False:
            while True:
                i = 10
        return self._trivialTest(b'third ine' + left * 3 + b'l\ndone', [b'>>> third line', b'third line', b'>>> done'])

    def testTypeover(self):
        if False:
            while True:
                i = 10
        return self._trivialTest(b'fourth xine' + left * 4 + insert + b'l\ndone', [b'>>> fourth line', b'fourth line', b'>>> done'])

    def testHome(self):
        if False:
            return 10
        return self._trivialTest(insert + b'blah line' + home + b'home\ndone', [b'>>> home line', b'home line', b'>>> done'])

    def testEnd(self):
        if False:
            for i in range(10):
                print('nop')
        return self._trivialTest(b'end ' + left * 4 + end + b'line\ndone', [b'>>> end line', b'end line', b'>>> done'])

class RecvlineLoopbackTelnetTests(_TelnetMixin, TestCase, RecvlineLoopbackMixin):
    pass

class RecvlineLoopbackSSHTests(_SSHMixin, TestCase, RecvlineLoopbackMixin):
    pass

@skipIf(not stdio, "Terminal requirements missing, can't run recvline tests over stdio")
class RecvlineLoopbackStdioTests(_StdioMixin, TestCase, RecvlineLoopbackMixin):
    pass

class HistoricRecvlineLoopbackMixin:
    serverProtocol = EchoServer

    def testUpArrow(self):
        if False:
            while True:
                i = 10
        return self._trivialTest(b'first line\n' + up + b'\ndone', [b'>>> first line', b'first line', b'>>> first line', b'first line', b'>>> done'])

    def test_DownArrowToPartialLineInHistory(self):
        if False:
            while True:
                i = 10
        '\n        Pressing down arrow to visit an entry that was added to the\n        history by pressing the up arrow instead of return does not\n        raise a L{TypeError}.\n\n        @see: U{http://twistedmatrix.com/trac/ticket/9031}\n\n        @return: A L{defer.Deferred} that fires when C{b"done"} is\n            echoed back.\n        '
        return self._trivialTest(b'first line\n' + b'partial line' + up + down + b'\ndone', [b'>>> first line', b'first line', b'>>> partial line', b'partial line', b'>>> done'])

    def testDownArrow(self):
        if False:
            while True:
                i = 10
        return self._trivialTest(b'first line\nsecond line\n' + up * 2 + down + b'\ndone', [b'>>> first line', b'first line', b'>>> second line', b'second line', b'>>> second line', b'second line', b'>>> done'])

class HistoricRecvlineLoopbackTelnetTests(_TelnetMixin, TestCase, HistoricRecvlineLoopbackMixin):
    pass

class HistoricRecvlineLoopbackSSHTests(_SSHMixin, TestCase, HistoricRecvlineLoopbackMixin):
    pass

@skipIf(not stdio, "Terminal requirements missing, can't run historic recvline tests over stdio")
class HistoricRecvlineLoopbackStdioTests(_StdioMixin, TestCase, HistoricRecvlineLoopbackMixin):
    pass

class TransportSequenceTests(TestCase):
    """
    L{twisted.conch.recvline.TransportSequence}
    """

    def test_invalidSequence(self):
        if False:
            return 10
        '\n        Initializing a L{recvline.TransportSequence} with no args\n        raises an assertion.\n        '
        self.assertRaises(AssertionError, recvline.TransportSequence)