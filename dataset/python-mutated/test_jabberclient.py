"""
Tests for L{twisted.words.protocols.jabber.client}
"""
from hashlib import sha1
from unittest import skipIf
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import client, error, jid, xmlstream
from twisted.words.protocols.jabber.sasl import SASLInitiatingInitializer
from twisted.words.xish import utility
try:
    from twisted.internet import ssl
except ImportError:
    ssl = None
    skipWhenNoSSL = (True, 'SSL not available')
else:
    skipWhenNoSSL = (False, '')
IQ_AUTH_GET = '/iq[@type="get"]/query[@xmlns="jabber:iq:auth"]'
IQ_AUTH_SET = '/iq[@type="set"]/query[@xmlns="jabber:iq:auth"]'
NS_BIND = 'urn:ietf:params:xml:ns:xmpp-bind'
IQ_BIND_SET = '/iq[@type="set"]/bind[@xmlns="%s"]' % NS_BIND
NS_SESSION = 'urn:ietf:params:xml:ns:xmpp-session'
IQ_SESSION_SET = '/iq[@type="set"]/session[@xmlns="%s"]' % NS_SESSION

class CheckVersionInitializerTests(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        a = xmlstream.Authenticator()
        xs = xmlstream.XmlStream(a)
        self.init = client.CheckVersionInitializer(xs)

    def testSupported(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test supported version number 1.0\n        '
        self.init.xmlstream.version = (1, 0)
        self.init.initialize()

    def testNotSupported(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test unsupported version number 0.0, and check exception.\n        '
        self.init.xmlstream.version = (0, 0)
        exc = self.assertRaises(error.StreamError, self.init.initialize)
        self.assertEqual('unsupported-version', exc.condition)

class InitiatingInitializerHarness:
    """
    Testing harness for interacting with XML stream initializers.

    This sets up an L{utility.XmlPipe} to create a communication channel between
    the initializer and the stubbed receiving entity. It features a sink and
    source side that both act similarly to a real L{xmlstream.XmlStream}. The
    sink is augmented with an authenticator to which initializers can be added.

    The harness also provides some utility methods to work with event observers
    and deferreds.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.output = []
        self.pipe = utility.XmlPipe()
        self.xmlstream = self.pipe.sink
        self.authenticator = xmlstream.ConnectAuthenticator('example.org')
        self.xmlstream.authenticator = self.authenticator

    def waitFor(self, event, handler):
        if False:
            return 10
        '\n        Observe an output event, returning a deferred.\n\n        The returned deferred will be fired when the given event has been\n        observed on the source end of the L{XmlPipe} tied to the protocol\n        under test. The handler is added as the first callback.\n\n        @param event: The event to be observed. See\n            L{utility.EventDispatcher.addOnetimeObserver}.\n        @param handler: The handler to be called with the observed event object.\n        @rtype: L{defer.Deferred}.\n        '
        d = defer.Deferred()
        d.addCallback(handler)
        self.pipe.source.addOnetimeObserver(event, d.callback)
        return d

class IQAuthInitializerTests(InitiatingInitializerHarness, unittest.TestCase):
    """
    Tests for L{client.IQAuthInitializer}.
    """

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.init = client.IQAuthInitializer(self.xmlstream)
        self.authenticator.jid = jid.JID('user@example.com/resource')
        self.authenticator.password = 'secret'

    def testPlainText(self):
        if False:
            print('Hello World!')
        '\n        Test plain-text authentication.\n\n        Act as a server supporting plain-text authentication and expect the\n        C{password} field to be filled with the password. Then act as if\n        authentication succeeds.\n        '

        def onAuthGet(iq):
            if False:
                print('Hello World!')
            '\n            Called when the initializer sent a query for authentication methods.\n\n            The response informs the client that plain-text authentication\n            is supported.\n            '
            response = xmlstream.toResponse(iq, 'result')
            response.addElement(('jabber:iq:auth', 'query'))
            response.query.addElement('username')
            response.query.addElement('password')
            response.query.addElement('resource')
            d = self.waitFor(IQ_AUTH_SET, onAuthSet)
            self.pipe.source.send(response)
            return d

        def onAuthSet(iq):
            if False:
                return 10
            '\n            Called when the initializer sent the authentication request.\n\n            The server checks the credentials and responds with an empty result\n            signalling success.\n            '
            self.assertEqual('user', str(iq.query.username))
            self.assertEqual('secret', str(iq.query.password))
            self.assertEqual('resource', str(iq.query.resource))
            response = xmlstream.toResponse(iq, 'result')
            self.pipe.source.send(response)
        d1 = self.waitFor(IQ_AUTH_GET, onAuthGet)
        d2 = self.init.initialize()
        return defer.gatherResults([d1, d2])

    def testDigest(self):
        if False:
            i = 10
            return i + 15
        '\n        Test digest authentication.\n\n        Act as a server supporting digest authentication and expect the\n        C{digest} field to be filled with a sha1 digest of the concatenated\n        stream session identifier and password. Then act as if authentication\n        succeeds.\n        '

        def onAuthGet(iq):
            if False:
                print('Hello World!')
            '\n            Called when the initializer sent a query for authentication methods.\n\n            The response informs the client that digest authentication is\n            supported.\n            '
            response = xmlstream.toResponse(iq, 'result')
            response.addElement(('jabber:iq:auth', 'query'))
            response.query.addElement('username')
            response.query.addElement('digest')
            response.query.addElement('resource')
            d = self.waitFor(IQ_AUTH_SET, onAuthSet)
            self.pipe.source.send(response)
            return d

        def onAuthSet(iq):
            if False:
                while True:
                    i = 10
            '\n            Called when the initializer sent the authentication request.\n\n            The server checks the credentials and responds with an empty result\n            signalling success.\n            '
            self.assertEqual('user', str(iq.query.username))
            self.assertEqual(sha1(b'12345secret').hexdigest(), str(iq.query.digest))
            self.assertEqual('resource', str(iq.query.resource))
            response = xmlstream.toResponse(iq, 'result')
            self.pipe.source.send(response)
        self.xmlstream.sid = '12345'
        d1 = self.waitFor(IQ_AUTH_GET, onAuthGet)
        d2 = self.init.initialize()
        return defer.gatherResults([d1, d2])

    def testFailRequestFields(self):
        if False:
            i = 10
            return i + 15
        '\n        Test initializer failure of request for fields for authentication.\n        '

        def onAuthGet(iq):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Called when the initializer sent a query for authentication methods.\n\n            The server responds that the client is not authorized to authenticate.\n            '
            response = error.StanzaError('not-authorized').toResponse(iq)
            self.pipe.source.send(response)
        d1 = self.waitFor(IQ_AUTH_GET, onAuthGet)
        d2 = self.init.initialize()
        self.assertFailure(d2, error.StanzaError)
        return defer.gatherResults([d1, d2])

    def testFailAuth(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test initializer failure to authenticate.\n        '

        def onAuthGet(iq):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Called when the initializer sent a query for authentication methods.\n\n            The response informs the client that plain-text authentication\n            is supported.\n            '
            response = xmlstream.toResponse(iq, 'result')
            response.addElement(('jabber:iq:auth', 'query'))
            response.query.addElement('username')
            response.query.addElement('password')
            response.query.addElement('resource')
            d = self.waitFor(IQ_AUTH_SET, onAuthSet)
            self.pipe.source.send(response)
            return d

        def onAuthSet(iq):
            if False:
                i = 10
                return i + 15
            '\n            Called when the initializer sent the authentication request.\n\n            The server checks the credentials and responds with a not-authorized\n            stanza error.\n            '
            response = error.StanzaError('not-authorized').toResponse(iq)
            self.pipe.source.send(response)
        d1 = self.waitFor(IQ_AUTH_GET, onAuthGet)
        d2 = self.init.initialize()
        self.assertFailure(d2, error.StanzaError)
        return defer.gatherResults([d1, d2])

class BindInitializerTests(InitiatingInitializerHarness, unittest.TestCase):
    """
    Tests for L{client.BindInitializer}.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.init = client.BindInitializer(self.xmlstream)
        self.authenticator.jid = jid.JID('user@example.com/resource')

    def testBasic(self):
        if False:
            print('Hello World!')
        '\n        Set up a stream, and act as if resource binding succeeds.\n        '

        def onBind(iq):
            if False:
                i = 10
                return i + 15
            response = xmlstream.toResponse(iq, 'result')
            response.addElement((NS_BIND, 'bind'))
            response.bind.addElement('jid', content='user@example.com/other resource')
            self.pipe.source.send(response)

        def cb(result):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(jid.JID('user@example.com/other resource'), self.authenticator.jid)
        d1 = self.waitFor(IQ_BIND_SET, onBind)
        d2 = self.init.start()
        d2.addCallback(cb)
        return defer.gatherResults([d1, d2])

    def testFailure(self):
        if False:
            print('Hello World!')
        '\n        Set up a stream, and act as if resource binding fails.\n        '

        def onBind(iq):
            if False:
                return 10
            response = error.StanzaError('conflict').toResponse(iq)
            self.pipe.source.send(response)
        d1 = self.waitFor(IQ_BIND_SET, onBind)
        d2 = self.init.start()
        self.assertFailure(d2, error.StanzaError)
        return defer.gatherResults([d1, d2])

class SessionInitializerTests(InitiatingInitializerHarness, unittest.TestCase):
    """
    Tests for L{client.SessionInitializer}.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.init = client.SessionInitializer(self.xmlstream)

    def testSuccess(self):
        if False:
            print('Hello World!')
        '\n        Set up a stream, and act as if session establishment succeeds.\n        '

        def onSession(iq):
            if False:
                while True:
                    i = 10
            response = xmlstream.toResponse(iq, 'result')
            self.pipe.source.send(response)
        d1 = self.waitFor(IQ_SESSION_SET, onSession)
        d2 = self.init.start()
        return defer.gatherResults([d1, d2])

    def testFailure(self):
        if False:
            print('Hello World!')
        '\n        Set up a stream, and act as if session establishment fails.\n        '

        def onSession(iq):
            if False:
                i = 10
                return i + 15
            response = error.StanzaError('forbidden').toResponse(iq)
            self.pipe.source.send(response)
        d1 = self.waitFor(IQ_SESSION_SET, onSession)
        d2 = self.init.start()
        self.assertFailure(d2, error.StanzaError)
        return defer.gatherResults([d1, d2])

class BasicAuthenticatorTests(unittest.TestCase):
    """
    Test for both BasicAuthenticator and basicClientFactory.
    """

    def test_basic(self):
        if False:
            print('Hello World!')
        '\n        Authenticator and stream are properly constructed by the factory.\n\n        The L{xmlstream.XmlStream} protocol created by the factory has the new\n        L{client.BasicAuthenticator} instance in its C{authenticator}\n        attribute.  It is set up with C{jid} and C{password} as passed to the\n        factory, C{otherHost} taken from the client JID. The stream futher has\n        two initializers, for TLS and authentication, of which the first has\n        its C{required} attribute set to C{True}.\n        '
        self.client_jid = jid.JID('user@example.com/resource')
        xs = client.basicClientFactory(self.client_jid, 'secret').buildProtocol(None)
        self.assertEqual('example.com', xs.authenticator.otherHost)
        self.assertEqual(self.client_jid, xs.authenticator.jid)
        self.assertEqual('secret', xs.authenticator.password)
        (tls, auth) = xs.initializers
        self.assertIsInstance(tls, xmlstream.TLSInitiatingInitializer)
        self.assertIsInstance(auth, client.IQAuthInitializer)
        self.assertFalse(tls.required)

class XMPPAuthenticatorTests(unittest.TestCase):
    """
    Test for both XMPPAuthenticator and XMPPClientFactory.
    """

    def test_basic(self):
        if False:
            print('Hello World!')
        '\n        Test basic operations.\n\n        Setup an XMPPClientFactory, which sets up an XMPPAuthenticator, and let\n        it produce a protocol instance. Then inspect the instance variables of\n        the authenticator and XML stream objects.\n        '
        self.client_jid = jid.JID('user@example.com/resource')
        xs = client.XMPPClientFactory(self.client_jid, 'secret').buildProtocol(None)
        self.assertEqual('example.com', xs.authenticator.otherHost)
        self.assertEqual(self.client_jid, xs.authenticator.jid)
        self.assertEqual('secret', xs.authenticator.password)
        (version, tls, sasl, bind, session) = xs.initializers
        self.assertIsInstance(tls, xmlstream.TLSInitiatingInitializer)
        self.assertIsInstance(sasl, SASLInitiatingInitializer)
        self.assertIsInstance(bind, client.BindInitializer)
        self.assertIsInstance(session, client.SessionInitializer)
        self.assertTrue(tls.required)
        self.assertTrue(sasl.required)
        self.assertTrue(bind.required)
        self.assertFalse(session.required)

    @skipIf(*skipWhenNoSSL)
    def test_tlsConfiguration(self):
        if False:
            i = 10
            return i + 15
        '\n        A TLS configuration is passed to the TLS initializer.\n        '
        configs = []

        def init(self, xs, required=True, configurationForTLS=None):
            if False:
                while True:
                    i = 10
            configs.append(configurationForTLS)
        self.client_jid = jid.JID('user@example.com/resource')
        configurationForTLS = ssl.CertificateOptions()
        factory = client.XMPPClientFactory(self.client_jid, 'secret', configurationForTLS=configurationForTLS)
        self.patch(xmlstream.TLSInitiatingInitializer, '__init__', init)
        xs = factory.buildProtocol(None)
        (version, tls, sasl, bind, session) = xs.initializers
        self.assertIsInstance(tls, xmlstream.TLSInitiatingInitializer)
        self.assertIs(configurationForTLS, configs[0])