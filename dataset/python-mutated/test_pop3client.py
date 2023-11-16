import inspect
import sys
from typing import List
from unittest import skipIf
from zope.interface import directlyProvides
import twisted.mail._pop3client
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.mail.pop3 import AdvancedPOP3Client as POP3Client, InsecureAuthenticationDisallowed, ServerErrorResponse
from twisted.mail.test import pop3testserver
from twisted.protocols import basic, loopback
from twisted.python import log
from twisted.trial.unittest import TestCase
try:
    from twisted.test.ssl_helpers import ClientTLSContext, ServerTLSContext
except ImportError:
    ClientTLSContext = None
    ServerTLSContext = None

class StringTransportWithConnectionLosing(StringTransport):

    def loseConnection(self):
        if False:
            i = 10
            return i + 15
        self.protocol.connectionLost(error.ConnectionDone())
capCache = {b'TOP': None, b'LOGIN-DELAY': b'180', b'UIDL': None, b'STLS': None, b'USER': None, b'SASL': b'LOGIN'}

def setUp(greet=True):
    if False:
        for i in range(10):
            print('nop')
    p = POP3Client()
    p._capCache = capCache
    t = StringTransportWithConnectionLosing()
    t.protocol = p
    p.makeConnection(t)
    if greet:
        p.dataReceived(b'+OK Hello!\r\n')
    return (p, t)

def strip(f):
    if False:
        while True:
            i = 10
    return lambda result, f=f: f()

class POP3ClientLoginTests(TestCase):

    def testNegativeGreeting(self):
        if False:
            return 10
        (p, t) = setUp(greet=False)
        p.allowInsecureLogin = True
        d = p.login(b'username', b'password')
        p.dataReceived(b'-ERR Offline for maintenance\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'Offline for maintenance'))

    def testOkUser(self):
        if False:
            for i in range(10):
                print('nop')
        (p, t) = setUp()
        d = p.user(b'username')
        self.assertEqual(t.value(), b'USER username\r\n')
        p.dataReceived(b'+OK send password\r\n')
        return d.addCallback(self.assertEqual, b'send password')

    def testBadUser(self):
        if False:
            print('Hello World!')
        (p, t) = setUp()
        d = p.user(b'username')
        self.assertEqual(t.value(), b'USER username\r\n')
        p.dataReceived(b'-ERR account suspended\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'account suspended'))

    def testOkPass(self):
        if False:
            while True:
                i = 10
        (p, t) = setUp()
        d = p.password(b'password')
        self.assertEqual(t.value(), b'PASS password\r\n')
        p.dataReceived(b"+OK you're in!\r\n")
        return d.addCallback(self.assertEqual, b"you're in!")

    def testBadPass(self):
        if False:
            while True:
                i = 10
        (p, t) = setUp()
        d = p.password(b'password')
        self.assertEqual(t.value(), b'PASS password\r\n')
        p.dataReceived(b'-ERR go away\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'go away'))

    def testOkLogin(self):
        if False:
            return 10
        (p, t) = setUp()
        p.allowInsecureLogin = True
        d = p.login(b'username', b'password')
        self.assertEqual(t.value(), b'USER username\r\n')
        p.dataReceived(b'+OK go ahead\r\n')
        self.assertEqual(t.value(), b'USER username\r\nPASS password\r\n')
        p.dataReceived(b'+OK password accepted\r\n')
        return d.addCallback(self.assertEqual, b'password accepted')

    def testBadPasswordLogin(self):
        if False:
            i = 10
            return i + 15
        (p, t) = setUp()
        p.allowInsecureLogin = True
        d = p.login(b'username', b'password')
        self.assertEqual(t.value(), b'USER username\r\n')
        p.dataReceived(b'+OK waiting on you\r\n')
        self.assertEqual(t.value(), b'USER username\r\nPASS password\r\n')
        p.dataReceived(b'-ERR bogus login\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'bogus login'))

    def testBadUsernameLogin(self):
        if False:
            print('Hello World!')
        (p, t) = setUp()
        p.allowInsecureLogin = True
        d = p.login(b'username', b'password')
        self.assertEqual(t.value(), b'USER username\r\n')
        p.dataReceived(b'-ERR bogus login\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'bogus login'))

    def testServerGreeting(self):
        if False:
            for i in range(10):
                print('nop')
        (p, t) = setUp(greet=False)
        p.dataReceived(b'+OK lalala this has no challenge\r\n')
        self.assertEqual(p.serverChallenge, None)

    def testServerGreetingWithChallenge(self):
        if False:
            print('Hello World!')
        (p, t) = setUp(greet=False)
        p.dataReceived(b'+OK <here is the challenge>\r\n')
        self.assertEqual(p.serverChallenge, b'<here is the challenge>')

    def testAPOP(self):
        if False:
            while True:
                i = 10
        (p, t) = setUp(greet=False)
        p.dataReceived(b'+OK <challenge string goes here>\r\n')
        d = p.login(b'username', b'password')
        self.assertEqual(t.value(), b'APOP username f34f1e464d0d7927607753129cabe39a\r\n')
        p.dataReceived(b'+OK Welcome!\r\n')
        return d.addCallback(self.assertEqual, b'Welcome!')

    def testInsecureLoginRaisesException(self):
        if False:
            for i in range(10):
                print('nop')
        (p, t) = setUp(greet=False)
        p.dataReceived(b'+OK Howdy\r\n')
        d = p.login(b'username', b'password')
        self.assertFalse(t.value())
        return self.assertFailure(d, InsecureAuthenticationDisallowed)

    def testSSLTransportConsideredSecure(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        If a server doesn't offer APOP but the transport is secured using\n        SSL or TLS, a plaintext login should be allowed, not rejected with\n        an InsecureAuthenticationDisallowed exception.\n        "
        (p, t) = setUp(greet=False)
        directlyProvides(t, interfaces.ISSLTransport)
        p.dataReceived(b'+OK Howdy\r\n')
        d = p.login(b'username', b'password')
        self.assertEqual(t.value(), b'USER username\r\n')
        t.clear()
        p.dataReceived(b'+OK\r\n')
        self.assertEqual(t.value(), b'PASS password\r\n')
        p.dataReceived(b'+OK\r\n')
        return d

class ListConsumer:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.data = {}

    def consume(self, result):
        if False:
            for i in range(10):
                print('nop')
        (item, value) = result
        self.data.setdefault(item, []).append(value)

class MessageConsumer:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.data = []

    def consume(self, line):
        if False:
            return 10
        self.data.append(line)

class POP3ClientListTests(TestCase):

    def testListSize(self):
        if False:
            return 10
        (p, t) = setUp()
        d = p.listSize()
        self.assertEqual(t.value(), b'LIST\r\n')
        p.dataReceived(b'+OK Here it comes\r\n')
        p.dataReceived(b'1 3\r\n2 2\r\n3 1\r\n.\r\n')
        return d.addCallback(self.assertEqual, [3, 2, 1])

    def testListSizeWithConsumer(self):
        if False:
            print('Hello World!')
        (p, t) = setUp()
        c = ListConsumer()
        f = c.consume
        d = p.listSize(f)
        self.assertEqual(t.value(), b'LIST\r\n')
        p.dataReceived(b'+OK Here it comes\r\n')
        p.dataReceived(b'1 3\r\n2 2\r\n3 1\r\n')
        self.assertEqual(c.data, {0: [3], 1: [2], 2: [1]})
        p.dataReceived(b'5 3\r\n6 2\r\n7 1\r\n')
        self.assertEqual(c.data, {0: [3], 1: [2], 2: [1], 4: [3], 5: [2], 6: [1]})
        p.dataReceived(b'.\r\n')
        return d.addCallback(self.assertIdentical, f)

    def testFailedListSize(self):
        if False:
            while True:
                i = 10
        (p, t) = setUp()
        d = p.listSize()
        self.assertEqual(t.value(), b'LIST\r\n')
        p.dataReceived(b'-ERR Fatal doom server exploded\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'Fatal doom server exploded'))

    def testListUID(self):
        if False:
            while True:
                i = 10
        (p, t) = setUp()
        d = p.listUID()
        self.assertEqual(t.value(), b'UIDL\r\n')
        p.dataReceived(b'+OK Here it comes\r\n')
        p.dataReceived(b'1 abc\r\n2 def\r\n3 ghi\r\n.\r\n')
        return d.addCallback(self.assertEqual, [b'abc', b'def', b'ghi'])

    def testListUIDWithConsumer(self):
        if False:
            for i in range(10):
                print('nop')
        (p, t) = setUp()
        c = ListConsumer()
        f = c.consume
        d = p.listUID(f)
        self.assertEqual(t.value(), b'UIDL\r\n')
        p.dataReceived(b'+OK Here it comes\r\n')
        p.dataReceived(b'1 xyz\r\n2 abc\r\n5 mno\r\n')
        self.assertEqual(c.data, {0: [b'xyz'], 1: [b'abc'], 4: [b'mno']})
        p.dataReceived(b'.\r\n')
        return d.addCallback(self.assertIdentical, f)

    def testFailedListUID(self):
        if False:
            print('Hello World!')
        (p, t) = setUp()
        d = p.listUID()
        self.assertEqual(t.value(), b'UIDL\r\n')
        p.dataReceived(b'-ERR Fatal doom server exploded\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'Fatal doom server exploded'))

class POP3ClientMessageTests(TestCase):

    def testRetrieve(self):
        if False:
            return 10
        (p, t) = setUp()
        d = p.retrieve(7)
        self.assertEqual(t.value(), b'RETR 8\r\n')
        p.dataReceived(b'+OK Message incoming\r\n')
        p.dataReceived(b'La la la here is message text\r\n')
        p.dataReceived(b'..Further message text tra la la\r\n')
        p.dataReceived(b'.\r\n')
        return d.addCallback(self.assertEqual, [b'La la la here is message text', b'.Further message text tra la la'])

    def testRetrieveWithConsumer(self):
        if False:
            print('Hello World!')
        (p, t) = setUp()
        c = MessageConsumer()
        f = c.consume
        d = p.retrieve(7, f)
        self.assertEqual(t.value(), b'RETR 8\r\n')
        p.dataReceived(b'+OK Message incoming\r\n')
        p.dataReceived(b'La la la here is message text\r\n')
        p.dataReceived(b'..Further message text\r\n.\r\n')
        return d.addCallback(self._cbTestRetrieveWithConsumer, f, c)

    def _cbTestRetrieveWithConsumer(self, result, f, c):
        if False:
            i = 10
            return i + 15
        self.assertIdentical(result, f)
        self.assertEqual(c.data, [b'La la la here is message text', b'.Further message text'])

    def testPartialRetrieve(self):
        if False:
            for i in range(10):
                print('nop')
        (p, t) = setUp()
        d = p.retrieve(7, lines=2)
        self.assertEqual(t.value(), b'TOP 8 2\r\n')
        p.dataReceived(b'+OK 2 lines on the way\r\n')
        p.dataReceived(b'Line the first!  Woop\r\n')
        p.dataReceived(b'Line the last!  Bye\r\n')
        p.dataReceived(b'.\r\n')
        return d.addCallback(self.assertEqual, [b'Line the first!  Woop', b'Line the last!  Bye'])

    def testPartialRetrieveWithConsumer(self):
        if False:
            return 10
        (p, t) = setUp()
        c = MessageConsumer()
        f = c.consume
        d = p.retrieve(7, f, lines=2)
        self.assertEqual(t.value(), b'TOP 8 2\r\n')
        p.dataReceived(b'+OK 2 lines on the way\r\n')
        p.dataReceived(b'Line the first!  Woop\r\n')
        p.dataReceived(b'Line the last!  Bye\r\n')
        p.dataReceived(b'.\r\n')
        return d.addCallback(self._cbTestPartialRetrieveWithConsumer, f, c)

    def _cbTestPartialRetrieveWithConsumer(self, result, f, c):
        if False:
            for i in range(10):
                print('nop')
        self.assertIdentical(result, f)
        self.assertEqual(c.data, [b'Line the first!  Woop', b'Line the last!  Bye'])

    def testFailedRetrieve(self):
        if False:
            for i in range(10):
                print('nop')
        (p, t) = setUp()
        d = p.retrieve(0)
        self.assertEqual(t.value(), b'RETR 1\r\n')
        p.dataReceived(b'-ERR Fatal doom server exploded\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'Fatal doom server exploded'))

    def test_concurrentRetrieves(self):
        if False:
            i = 10
            return i + 15
        '\n        Issue three retrieve calls immediately without waiting for any to\n        succeed and make sure they all do succeed eventually.\n        '
        (p, t) = setUp()
        messages = [p.retrieve(i).addCallback(self.assertEqual, [b'First line of %d.' % (i + 1,), b'Second line of %d.' % (i + 1,)]) for i in range(3)]
        for i in range(1, 4):
            self.assertEqual(t.value(), b'RETR %d\r\n' % (i,))
            t.clear()
            p.dataReceived(b'+OK 2 lines on the way\r\n')
            p.dataReceived(b'First line of %d.\r\n' % (i,))
            p.dataReceived(b'Second line of %d.\r\n' % (i,))
            self.assertEqual(t.value(), b'')
            p.dataReceived(b'.\r\n')
        return defer.DeferredList(messages, fireOnOneErrback=True)

class POP3ClientMiscTests(TestCase):

    def testCapability(self):
        if False:
            for i in range(10):
                print('nop')
        (p, t) = setUp()
        d = p.capabilities(useCache=0)
        self.assertEqual(t.value(), b'CAPA\r\n')
        p.dataReceived(b'+OK Capabilities on the way\r\n')
        p.dataReceived(b'X\r\nY\r\nZ\r\nA 1 2 3\r\nB 1 2\r\nC 1\r\n.\r\n')
        return d.addCallback(self.assertEqual, {b'X': None, b'Y': None, b'Z': None, b'A': [b'1', b'2', b'3'], b'B': [b'1', b'2'], b'C': [b'1']})

    def testCapabilityError(self):
        if False:
            print('Hello World!')
        (p, t) = setUp()
        d = p.capabilities(useCache=0)
        self.assertEqual(t.value(), b'CAPA\r\n')
        p.dataReceived(b'-ERR This server is lame!\r\n')
        return d.addCallback(self.assertEqual, {})

    def testStat(self):
        if False:
            for i in range(10):
                print('nop')
        (p, t) = setUp()
        d = p.stat()
        self.assertEqual(t.value(), b'STAT\r\n')
        p.dataReceived(b'+OK 1 1212\r\n')
        return d.addCallback(self.assertEqual, (1, 1212))

    def testStatError(self):
        if False:
            while True:
                i = 10
        (p, t) = setUp()
        d = p.stat()
        self.assertEqual(t.value(), b'STAT\r\n')
        p.dataReceived(b'-ERR This server is lame!\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'This server is lame!'))

    def testNoop(self):
        if False:
            for i in range(10):
                print('nop')
        (p, t) = setUp()
        d = p.noop()
        self.assertEqual(t.value(), b'NOOP\r\n')
        p.dataReceived(b'+OK No-op to you too!\r\n')
        return d.addCallback(self.assertEqual, b'No-op to you too!')

    def testNoopError(self):
        if False:
            while True:
                i = 10
        (p, t) = setUp()
        d = p.noop()
        self.assertEqual(t.value(), b'NOOP\r\n')
        p.dataReceived(b'-ERR This server is lame!\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'This server is lame!'))

    def testRset(self):
        if False:
            while True:
                i = 10
        (p, t) = setUp()
        d = p.reset()
        self.assertEqual(t.value(), b'RSET\r\n')
        p.dataReceived(b'+OK Reset state\r\n')
        return d.addCallback(self.assertEqual, b'Reset state')

    def testRsetError(self):
        if False:
            i = 10
            return i + 15
        (p, t) = setUp()
        d = p.reset()
        self.assertEqual(t.value(), b'RSET\r\n')
        p.dataReceived(b'-ERR This server is lame!\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'This server is lame!'))

    def testDelete(self):
        if False:
            print('Hello World!')
        (p, t) = setUp()
        d = p.delete(3)
        self.assertEqual(t.value(), b'DELE 4\r\n')
        p.dataReceived(b'+OK Hasta la vista\r\n')
        return d.addCallback(self.assertEqual, b'Hasta la vista')

    def testDeleteError(self):
        if False:
            while True:
                i = 10
        (p, t) = setUp()
        d = p.delete(3)
        self.assertEqual(t.value(), b'DELE 4\r\n')
        p.dataReceived(b'-ERR Winner is not you.\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'Winner is not you.'))

class SimpleClient(POP3Client):

    def __init__(self, deferred, contextFactory=None):
        if False:
            return 10
        self.deferred = deferred
        self.allowInsecureLogin = True

    def serverGreeting(self, challenge):
        if False:
            for i in range(10):
                print('nop')
        self.deferred.callback(None)

class POP3HelperMixin:
    serverCTX = None
    clientCTX = None

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        d = defer.Deferred()
        self.server = pop3testserver.POP3TestServer(contextFactory=self.serverCTX)
        self.client = SimpleClient(d, contextFactory=self.clientCTX)
        self.client.timeout = 30
        self.connected = d

    def tearDown(self):
        if False:
            return 10
        del self.server
        del self.client
        del self.connected

    def _cbStopClient(self, ignore):
        if False:
            while True:
                i = 10
        self.client.transport.loseConnection()

    def _ebGeneral(self, failure):
        if False:
            return 10
        self.client.transport.loseConnection()
        self.server.transport.loseConnection()
        return failure

    def loopback(self):
        if False:
            while True:
                i = 10
        return loopback.loopbackTCP(self.server, self.client, noisy=False)

class TLSServerFactory(protocol.ServerFactory):

    class protocol(basic.LineReceiver):
        context = None
        output: List[bytes] = []

        def connectionMade(self):
            if False:
                for i in range(10):
                    print('nop')
            self.factory.input = []
            self.output = self.output[:]
            for line in self.output.pop(0):
                self.sendLine(line)

        def lineReceived(self, line):
            if False:
                return 10
            self.factory.input.append(line)
            [self.sendLine(l) for l in self.output.pop(0)]
            if line == b'STLS':
                self.transport.startTLS(self.context)

@skipIf(not ClientTLSContext, 'OpenSSL not present')
@skipIf(not interfaces.IReactorSSL(reactor, None), 'OpenSSL not present')
class POP3TLSTests(TestCase):
    """
    Tests for POP3Client's support for TLS connections.
    """

    def test_startTLS(self):
        if False:
            print('Hello World!')
        '\n        POP3Client.startTLS starts a TLS session over its existing TCP\n        connection.\n        '
        sf = TLSServerFactory()
        sf.protocol.output = [[b'+OK'], [b'+OK', b'STLS', b'.'], [b'+OK'], [b'+OK', b'.'], [b'+OK']]
        sf.protocol.context = ServerTLSContext()
        port = reactor.listenTCP(0, sf, interface='127.0.0.1')
        self.addCleanup(port.stopListening)
        H = port.getHost().host
        P = port.getHost().port
        connLostDeferred = defer.Deferred()
        cp = SimpleClient(defer.Deferred(), ClientTLSContext())

        def connectionLost(reason):
            if False:
                i = 10
                return i + 15
            SimpleClient.connectionLost(cp, reason)
            connLostDeferred.callback(None)
        cp.connectionLost = connectionLost
        cf = protocol.ClientFactory()
        cf.protocol = lambda : cp
        conn = reactor.connectTCP(H, P, cf)

        def cbConnected(ignored):
            if False:
                print('Hello World!')
            log.msg('Connected to server; starting TLS')
            return cp.startTLS()

        def cbStartedTLS(ignored):
            if False:
                while True:
                    i = 10
            log.msg('Started TLS; disconnecting')
            return cp.quit()

        def cbDisconnected(ign):
            if False:
                i = 10
                return i + 15
            log.msg('Disconnected; asserting correct input received')
            self.assertEqual(sf.input, [b'CAPA', b'STLS', b'CAPA', b'QUIT'])

        def cleanup(result):
            if False:
                for i in range(10):
                    print('nop')
            log.msg('Asserted correct input; disconnecting client and shutting down server')
            conn.disconnect()
            return connLostDeferred
        cp.deferred.addCallback(cbConnected)
        cp.deferred.addCallback(cbStartedTLS)
        cp.deferred.addCallback(cbDisconnected)
        cp.deferred.addBoth(cleanup)
        return cp.deferred

class POP3TimeoutTests(POP3HelperMixin, TestCase):

    def testTimeout(self):
        if False:
            return 10

        def login():
            if False:
                print('Hello World!')
            d = self.client.login('test', 'twisted')
            d.addCallback(loggedIn)
            d.addErrback(timedOut)
            return d

        def loggedIn(result):
            if False:
                for i in range(10):
                    print('nop')
            self.fail('Successfully logged in!?  Impossible!')

        def timedOut(failure):
            if False:
                while True:
                    i = 10
            failure.trap(error.TimeoutError)
            self._cbStopClient(None)

        def quit():
            if False:
                return 10
            return self.client.quit()
        self.client.timeout = 0.01
        pop3testserver.TIMEOUT_RESPONSE = True
        methods = [login, quit]
        map(self.connected.addCallback, map(strip, methods))
        self.connected.addCallback(self._cbStopClient)
        self.connected.addErrback(self._ebGeneral)
        return self.loopback()

class POP3ClientModuleStructureTests(TestCase):
    """
    Miscellaneous tests more to do with module/package structure than
    anything to do with the POP3 client.
    """

    def test_all(self):
        if False:
            print('Hello World!')
        '\n        twisted.mail._pop3client.__all__ should be empty because all classes\n        should be imported through twisted.mail.pop3.\n        '
        self.assertEqual(twisted.mail._pop3client.__all__, [])

    def test_import(self):
        if False:
            while True:
                i = 10
        '\n        Every public class in twisted.mail._pop3client should be available as\n        a member of twisted.mail.pop3 with the exception of\n        twisted.mail._pop3client.POP3Client which should be available as\n        twisted.mail.pop3.AdvancedClient.\n        '
        publicClasses = [c[0] for c in inspect.getmembers(sys.modules['twisted.mail._pop3client'], inspect.isclass) if not c[0][0] == '_']
        for pc in publicClasses:
            if not pc == 'POP3Client':
                self.assertTrue(hasattr(twisted.mail.pop3, pc), f'{pc} not in {twisted.mail.pop3}')
            else:
                self.assertTrue(hasattr(twisted.mail.pop3, 'AdvancedPOP3Client'))