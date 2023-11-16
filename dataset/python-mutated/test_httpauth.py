"""
Tests for L{twisted.web._auth}.
"""
import base64
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.cred import error, portal
from twisted.cred.checkers import ANONYMOUS, AllowAnonymousAccess, InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import IUsernamePassword
from twisted.internet.address import IPv4Address
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import EventLoggingObserver
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial import unittest
from twisted.web._auth import basic, digest
from twisted.web._auth.basic import BasicCredentialFactory
from twisted.web._auth.wrapper import HTTPAuthSessionWrapper, UnauthorizedResource
from twisted.web.iweb import ICredentialFactory
from twisted.web.resource import IResource, Resource, getChildForRequest
from twisted.web.server import NOT_DONE_YET
from twisted.web.static import Data
from twisted.web.test.test_web import DummyRequest

def b64encode(s):
    if False:
        print('Hello World!')
    return base64.b64encode(s).strip()

class BasicAuthTestsMixin:
    """
    L{TestCase} mixin class which defines a number of tests for
    L{basic.BasicCredentialFactory}.  Because this mixin defines C{setUp}, it
    must be inherited before L{TestCase}.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.request = self.makeRequest()
        self.realm = b'foo'
        self.username = b'dreid'
        self.password = b'S3CuR1Ty'
        self.credentialFactory = basic.BasicCredentialFactory(self.realm)

    def makeRequest(self, method=b'GET', clientAddress=None):
        if False:
            i = 10
            return i + 15
        '\n        Create a request object to be passed to\n        L{basic.BasicCredentialFactory.decode} along with a response value.\n        Override this in a subclass.\n        '
        raise NotImplementedError(f'{self.__class__!r} did not implement makeRequest')

    def test_interface(self):
        if False:
            return 10
        '\n        L{BasicCredentialFactory} implements L{ICredentialFactory}.\n        '
        self.assertTrue(verifyObject(ICredentialFactory, self.credentialFactory))

    def test_usernamePassword(self):
        if False:
            return 10
        '\n        L{basic.BasicCredentialFactory.decode} turns a base64-encoded response\n        into a L{UsernamePassword} object with a password which reflects the\n        one which was encoded in the response.\n        '
        response = b64encode(b''.join([self.username, b':', self.password]))
        creds = self.credentialFactory.decode(response, self.request)
        self.assertTrue(IUsernamePassword.providedBy(creds))
        self.assertTrue(creds.checkPassword(self.password))
        self.assertFalse(creds.checkPassword(self.password + b'wrong'))

    def test_incorrectPadding(self):
        if False:
            i = 10
            return i + 15
        '\n        L{basic.BasicCredentialFactory.decode} decodes a base64-encoded\n        response with incorrect padding.\n        '
        response = b64encode(b''.join([self.username, b':', self.password]))
        response = response.strip(b'=')
        creds = self.credentialFactory.decode(response, self.request)
        self.assertTrue(verifyObject(IUsernamePassword, creds))
        self.assertTrue(creds.checkPassword(self.password))

    def test_invalidEncoding(self):
        if False:
            i = 10
            return i + 15
        '\n        L{basic.BasicCredentialFactory.decode} raises L{LoginFailed} if passed\n        a response which is not base64-encoded.\n        '
        response = b'x'
        self.assertRaises(error.LoginFailed, self.credentialFactory.decode, response, self.makeRequest())

    def test_invalidCredentials(self):
        if False:
            return 10
        '\n        L{basic.BasicCredentialFactory.decode} raises L{LoginFailed} when\n        passed a response which is not valid base64-encoded text.\n        '
        response = b64encode(b'123abc+/')
        self.assertRaises(error.LoginFailed, self.credentialFactory.decode, response, self.makeRequest())

class RequestMixin:

    def makeRequest(self, method=b'GET', clientAddress=None):
        if False:
            return 10
        '\n        Create a L{DummyRequest} (change me to create a\n        L{twisted.web.http.Request} instead).\n        '
        if clientAddress is None:
            clientAddress = IPv4Address('TCP', 'localhost', 1234)
        request = DummyRequest(b'/')
        request.method = method
        request.client = clientAddress
        return request

class BasicAuthTests(RequestMixin, BasicAuthTestsMixin, unittest.TestCase):
    """
    Basic authentication tests which use L{twisted.web.http.Request}.
    """

class DigestAuthTests(RequestMixin, unittest.TestCase):
    """
    Digest authentication tests which use L{twisted.web.http.Request}.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        '\n        Create a DigestCredentialFactory for testing\n        '
        self.realm = b'test realm'
        self.algorithm = b'md5'
        self.credentialFactory = digest.DigestCredentialFactory(self.algorithm, self.realm)
        self.request = self.makeRequest()

    def test_decode(self):
        if False:
            while True:
                i = 10
        '\n        L{digest.DigestCredentialFactory.decode} calls the C{decode} method on\n        L{twisted.cred.digest.DigestCredentialFactory} with the HTTP method and\n        host of the request.\n        '
        host = b'169.254.0.1'
        method = b'GET'
        done = [False]
        response = object()

        def check(_response, _method, _host):
            if False:
                while True:
                    i = 10
            self.assertEqual(response, _response)
            self.assertEqual(method, _method)
            self.assertEqual(host, _host)
            done[0] = True
        self.patch(self.credentialFactory.digest, 'decode', check)
        req = self.makeRequest(method, IPv4Address('TCP', host, 81))
        self.credentialFactory.decode(response, req)
        self.assertTrue(done[0])

    def test_interface(self):
        if False:
            return 10
        '\n        L{DigestCredentialFactory} implements L{ICredentialFactory}.\n        '
        self.assertTrue(verifyObject(ICredentialFactory, self.credentialFactory))

    def test_getChallenge(self):
        if False:
            print('Hello World!')
        "\n        The challenge issued by L{DigestCredentialFactory.getChallenge} must\n        include C{'qop'}, C{'realm'}, C{'algorithm'}, C{'nonce'}, and\n        C{'opaque'} keys.  The values for the C{'realm'} and C{'algorithm'}\n        keys must match the values supplied to the factory's initializer.\n        None of the values may have newlines in them.\n        "
        challenge = self.credentialFactory.getChallenge(self.request)
        self.assertEqual(challenge['qop'], b'auth')
        self.assertEqual(challenge['realm'], b'test realm')
        self.assertEqual(challenge['algorithm'], b'md5')
        self.assertIn('nonce', challenge)
        self.assertIn('opaque', challenge)
        for v in challenge.values():
            self.assertNotIn(b'\n', v)

    def test_getChallengeWithoutClientIP(self):
        if False:
            while True:
                i = 10
        '\n        L{DigestCredentialFactory.getChallenge} can issue a challenge even if\n        the L{Request} it is passed returns L{None} from C{getClientIP}.\n        '
        request = self.makeRequest(b'GET', None)
        challenge = self.credentialFactory.getChallenge(request)
        self.assertEqual(challenge['qop'], b'auth')
        self.assertEqual(challenge['realm'], b'test realm')
        self.assertEqual(challenge['algorithm'], b'md5')
        self.assertIn('nonce', challenge)
        self.assertIn('opaque', challenge)

class UnauthorizedResourceTests(RequestMixin, unittest.TestCase):
    """
    Tests for L{UnauthorizedResource}.
    """

    def test_getChildWithDefault(self):
        if False:
            print('Hello World!')
        '\n        An L{UnauthorizedResource} is every child of itself.\n        '
        resource = UnauthorizedResource([])
        self.assertIdentical(resource.getChildWithDefault('foo', None), resource)
        self.assertIdentical(resource.getChildWithDefault('bar', None), resource)

    def _unauthorizedRenderTest(self, request):
        if False:
            while True:
                i = 10
        '\n        Render L{UnauthorizedResource} for the given request object and verify\n        that the response code is I{Unauthorized} and that a I{WWW-Authenticate}\n        header is set in the response containing a challenge.\n        '
        resource = UnauthorizedResource([BasicCredentialFactory('example.com')])
        request.render(resource)
        self.assertEqual(request.responseCode, 401)
        self.assertEqual(request.responseHeaders.getRawHeaders(b'www-authenticate'), [b'basic realm="example.com"'])

    def test_render(self):
        if False:
            i = 10
            return i + 15
        '\n        L{UnauthorizedResource} renders with a 401 response code and a\n        I{WWW-Authenticate} header and puts a simple unauthorized message\n        into the response body.\n        '
        request = self.makeRequest()
        self._unauthorizedRenderTest(request)
        self.assertEqual(b'Unauthorized', b''.join(request.written))

    def test_renderHEAD(self):
        if False:
            print('Hello World!')
        '\n        The rendering behavior of L{UnauthorizedResource} for a I{HEAD} request\n        is like its handling of a I{GET} request, but no response body is\n        written.\n        '
        request = self.makeRequest(method=b'HEAD')
        self._unauthorizedRenderTest(request)
        self.assertEqual(b'', b''.join(request.written))

    def test_renderQuotesRealm(self):
        if False:
            i = 10
            return i + 15
        '\n        The realm value included in the I{WWW-Authenticate} header set in\n        the response when L{UnauthorizedResounrce} is rendered has quotes\n        and backslashes escaped.\n        '
        resource = UnauthorizedResource([BasicCredentialFactory('example\\"foo')])
        request = self.makeRequest()
        request.render(resource)
        self.assertEqual(request.responseHeaders.getRawHeaders(b'www-authenticate'), [b'basic realm="example\\\\\\"foo"'])

    def test_renderQuotesDigest(self):
        if False:
            while True:
                i = 10
        '\n        The digest value included in the I{WWW-Authenticate} header\n        set in the response when L{UnauthorizedResource} is rendered\n        has quotes and backslashes escaped.\n        '
        resource = UnauthorizedResource([digest.DigestCredentialFactory(b'md5', b'example\\"foo')])
        request = self.makeRequest()
        request.render(resource)
        authHeader = request.responseHeaders.getRawHeaders(b'www-authenticate')[0]
        self.assertIn(b'realm="example\\\\\\"foo"', authHeader)
        self.assertIn(b'hm="md5', authHeader)
implementer(portal.IRealm)

class Realm:
    """
    A simple L{IRealm} implementation which gives out L{WebAvatar} for any
    avatarId.

    @type loggedIn: C{int}
    @ivar loggedIn: The number of times C{requestAvatar} has been invoked for
        L{IResource}.

    @type loggedOut: C{int}
    @ivar loggedOut: The number of times the logout callback has been invoked.
    """

    def __init__(self, avatarFactory):
        if False:
            for i in range(10):
                print('nop')
        self.loggedOut = 0
        self.loggedIn = 0
        self.avatarFactory = avatarFactory

    def requestAvatar(self, avatarId, mind, *interfaces):
        if False:
            i = 10
            return i + 15
        if IResource in interfaces:
            self.loggedIn += 1
            return (IResource, self.avatarFactory(avatarId), self.logout)
        raise NotImplementedError()

    def logout(self):
        if False:
            print('Hello World!')
        self.loggedOut += 1

class HTTPAuthHeaderTests(unittest.TestCase):
    """
    Tests for L{HTTPAuthSessionWrapper}.
    """
    makeRequest = DummyRequest

    def setUp(self):
        if False:
            i = 10
            return i + 15
        '\n        Create a realm, portal, and L{HTTPAuthSessionWrapper} to use in the tests.\n        '
        self.username = b'foo bar'
        self.password = b'bar baz'
        self.avatarContent = b'contents of the avatar resource itself'
        self.childName = b'foo-child'
        self.childContent = b'contents of the foo child of the avatar'
        self.checker = InMemoryUsernamePasswordDatabaseDontUse()
        self.checker.addUser(self.username, self.password)
        self.avatar = Data(self.avatarContent, 'text/plain')
        self.avatar.putChild(self.childName, Data(self.childContent, 'text/plain'))
        self.avatars = {self.username: self.avatar}
        self.realm = Realm(self.avatars.get)
        self.portal = portal.Portal(self.realm, [self.checker])
        self.credentialFactories = []
        self.wrapper = HTTPAuthSessionWrapper(self.portal, self.credentialFactories)

    def _authorizedBasicLogin(self, request):
        if False:
            return 10
        '\n        Add an I{basic authorization} header to the given request and then\n        dispatch it, starting from C{self.wrapper} and returning the resulting\n        L{IResource}.\n        '
        authorization = b64encode(self.username + b':' + self.password)
        request.requestHeaders.addRawHeader(b'authorization', b'Basic ' + authorization)
        return getChildForRequest(self.wrapper, request)

    def test_getChildWithDefault(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Resource traversal which encounters an L{HTTPAuthSessionWrapper}\n        results in an L{UnauthorizedResource} instance when the request does\n        not have the required I{Authorization} headers.\n        '
        request = self.makeRequest([self.childName])
        child = getChildForRequest(self.wrapper, request)
        d = request.notifyFinish()

        def cbFinished(result):
            if False:
                return 10
            self.assertEqual(request.responseCode, 401)
        d.addCallback(cbFinished)
        request.render(child)
        return d

    def _invalidAuthorizationTest(self, response):
        if False:
            print('Hello World!')
        '\n        Create a request with the given value as the value of an\n        I{Authorization} header and perform resource traversal with it,\n        starting at C{self.wrapper}.  Assert that the result is a 401 response\n        code.  Return a L{Deferred} which fires when this is all done.\n        '
        self.credentialFactories.append(BasicCredentialFactory('example.com'))
        request = self.makeRequest([self.childName])
        request.requestHeaders.addRawHeader(b'authorization', response)
        child = getChildForRequest(self.wrapper, request)
        d = request.notifyFinish()

        def cbFinished(result):
            if False:
                print('Hello World!')
            self.assertEqual(request.responseCode, 401)
        d.addCallback(cbFinished)
        request.render(child)
        return d

    def test_getChildWithDefaultUnauthorizedUser(self):
        if False:
            print('Hello World!')
        '\n        Resource traversal which enouncters an L{HTTPAuthSessionWrapper}\n        results in an L{UnauthorizedResource} when the request has an\n        I{Authorization} header with a user which does not exist.\n        '
        return self._invalidAuthorizationTest(b'Basic ' + b64encode(b'foo:bar'))

    def test_getChildWithDefaultUnauthorizedPassword(self):
        if False:
            i = 10
            return i + 15
        '\n        Resource traversal which enouncters an L{HTTPAuthSessionWrapper}\n        results in an L{UnauthorizedResource} when the request has an\n        I{Authorization} header with a user which exists and the wrong\n        password.\n        '
        return self._invalidAuthorizationTest(b'Basic ' + b64encode(self.username + b':bar'))

    def test_getChildWithDefaultUnrecognizedScheme(self):
        if False:
            i = 10
            return i + 15
        '\n        Resource traversal which enouncters an L{HTTPAuthSessionWrapper}\n        results in an L{UnauthorizedResource} when the request has an\n        I{Authorization} header with an unrecognized scheme.\n        '
        return self._invalidAuthorizationTest(b'Quux foo bar baz')

    def test_getChildWithDefaultAuthorized(self):
        if False:
            return 10
        '\n        Resource traversal which encounters an L{HTTPAuthSessionWrapper}\n        results in an L{IResource} which renders the L{IResource} avatar\n        retrieved from the portal when the request has a valid I{Authorization}\n        header.\n        '
        self.credentialFactories.append(BasicCredentialFactory('example.com'))
        request = self.makeRequest([self.childName])
        child = self._authorizedBasicLogin(request)
        d = request.notifyFinish()

        def cbFinished(ignored):
            if False:
                while True:
                    i = 10
            self.assertEqual(request.written, [self.childContent])
        d.addCallback(cbFinished)
        request.render(child)
        return d

    def test_renderAuthorized(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Resource traversal which terminates at an L{HTTPAuthSessionWrapper}\n        and includes correct authentication headers results in the\n        L{IResource} avatar (not one of its children) retrieved from the\n        portal being rendered.\n        '
        self.credentialFactories.append(BasicCredentialFactory('example.com'))
        request = self.makeRequest([])
        child = self._authorizedBasicLogin(request)
        d = request.notifyFinish()

        def cbFinished(ignored):
            if False:
                return 10
            self.assertEqual(request.written, [self.avatarContent])
        d.addCallback(cbFinished)
        request.render(child)
        return d

    def test_getChallengeCalledWithRequest(self):
        if False:
            return 10
        '\n        When L{HTTPAuthSessionWrapper} finds an L{ICredentialFactory} to issue\n        a challenge, it calls the C{getChallenge} method with the request as an\n        argument.\n        '

        @implementer(ICredentialFactory)
        class DumbCredentialFactory:
            scheme = b'dumb'

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.requests = []

            def getChallenge(self, request):
                if False:
                    while True:
                        i = 10
                self.requests.append(request)
                return {}
        factory = DumbCredentialFactory()
        self.credentialFactories.append(factory)
        request = self.makeRequest([self.childName])
        child = getChildForRequest(self.wrapper, request)
        d = request.notifyFinish()

        def cbFinished(ignored):
            if False:
                i = 10
                return i + 15
            self.assertEqual(factory.requests, [request])
        d.addCallback(cbFinished)
        request.render(child)
        return d

    def _logoutTest(self):
        if False:
            i = 10
            return i + 15
        '\n        Issue a request for an authentication-protected resource using valid\n        credentials and then return the C{DummyRequest} instance which was\n        used.\n\n        This is a helper for tests about the behavior of the logout\n        callback.\n        '
        self.credentialFactories.append(BasicCredentialFactory('example.com'))

        class SlowerResource(Resource):

            def render(self, request):
                if False:
                    print('Hello World!')
                return NOT_DONE_YET
        self.avatar.putChild(self.childName, SlowerResource())
        request = self.makeRequest([self.childName])
        child = self._authorizedBasicLogin(request)
        request.render(child)
        self.assertEqual(self.realm.loggedOut, 0)
        return request

    def test_logout(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The realm's logout callback is invoked after the resource is rendered.\n        "
        request = self._logoutTest()
        request.finish()
        self.assertEqual(self.realm.loggedOut, 1)

    def test_logoutOnError(self):
        if False:
            print('Hello World!')
        "\n        The realm's logout callback is also invoked if there is an error\n        generating the response (for example, if the client disconnects\n        early).\n        "
        request = self._logoutTest()
        request.processingFailed(Failure(ConnectionDone('Simulated disconnect')))
        self.assertEqual(self.realm.loggedOut, 1)

    def test_decodeRaises(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Resource traversal which enouncters an L{HTTPAuthSessionWrapper}\n        results in an L{UnauthorizedResource} when the request has a I{Basic\n        Authorization} header which cannot be decoded using base64.\n        '
        self.credentialFactories.append(BasicCredentialFactory('example.com'))
        request = self.makeRequest([self.childName])
        request.requestHeaders.addRawHeader(b'authorization', b'Basic decode should fail')
        child = getChildForRequest(self.wrapper, request)
        self.assertIsInstance(child, UnauthorizedResource)

    def test_selectParseResponse(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{HTTPAuthSessionWrapper._selectParseHeader} returns a two-tuple giving\n        the L{ICredentialFactory} to use to parse the header and a string\n        containing the portion of the header which remains to be parsed.\n        '
        basicAuthorization = b'Basic abcdef123456'
        self.assertEqual(self.wrapper._selectParseHeader(basicAuthorization), (None, None))
        factory = BasicCredentialFactory('example.com')
        self.credentialFactories.append(factory)
        self.assertEqual(self.wrapper._selectParseHeader(basicAuthorization), (factory, b'abcdef123456'))

    def test_unexpectedDecodeError(self):
        if False:
            i = 10
            return i + 15
        "\n        Any unexpected exception raised by the credential factory's C{decode}\n        method results in a 500 response code and causes the exception to be\n        logged.\n        "
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)

        class UnexpectedException(Exception):
            pass

        class BadFactory:
            scheme = b'bad'

            def getChallenge(self, client):
                if False:
                    for i in range(10):
                        print('nop')
                return {}

            def decode(self, response, request):
                if False:
                    return 10
                raise UnexpectedException()
        self.credentialFactories.append(BadFactory())
        request = self.makeRequest([self.childName])
        request.requestHeaders.addRawHeader(b'authorization', b'Bad abc')
        child = getChildForRequest(self.wrapper, request)
        request.render(child)
        self.assertEqual(request.responseCode, 500)
        self.assertEquals(1, len(logObserver))
        self.assertIsInstance(logObserver[0]['log_failure'].value, UnexpectedException)
        self.assertEqual(len(self.flushLoggedErrors(UnexpectedException)), 1)

    def test_unexpectedLoginError(self):
        if False:
            while True:
                i = 10
        '\n        Any unexpected failure from L{Portal.login} results in a 500 response\n        code and causes the failure to be logged.\n        '
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)

        class UnexpectedException(Exception):
            pass

        class BrokenChecker:
            credentialInterfaces = (IUsernamePassword,)

            def requestAvatarId(self, credentials):
                if False:
                    i = 10
                    return i + 15
                raise UnexpectedException()
        self.portal.registerChecker(BrokenChecker())
        self.credentialFactories.append(BasicCredentialFactory('example.com'))
        request = self.makeRequest([self.childName])
        child = self._authorizedBasicLogin(request)
        request.render(child)
        self.assertEqual(request.responseCode, 500)
        self.assertEquals(1, len(logObserver))
        self.assertIsInstance(logObserver[0]['log_failure'].value, UnexpectedException)
        self.assertEqual(len(self.flushLoggedErrors(UnexpectedException)), 1)

    def test_anonymousAccess(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Anonymous requests are allowed if a L{Portal} has an anonymous checker\n        registered.\n        '
        unprotectedContents = b'contents of the unprotected child resource'
        self.avatars[ANONYMOUS] = Resource()
        self.avatars[ANONYMOUS].putChild(self.childName, Data(unprotectedContents, 'text/plain'))
        self.portal.registerChecker(AllowAnonymousAccess())
        self.credentialFactories.append(BasicCredentialFactory('example.com'))
        request = self.makeRequest([self.childName])
        child = getChildForRequest(self.wrapper, request)
        d = request.notifyFinish()

        def cbFinished(ignored):
            if False:
                while True:
                    i = 10
            self.assertEqual(request.written, [unprotectedContents])
        d.addCallback(cbFinished)
        request.render(child)
        return d