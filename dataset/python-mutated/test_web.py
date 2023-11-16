"""
Tests for various parts of L{twisted.web}.
"""
import os
import zlib
from io import BytesIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet import interfaces
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.task import Clock
from twisted.internet.testing import EventLoggingObserver, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python import failure, reflect
from twisted.python.compat import iterbytes
from twisted.python.filepath import FilePath
from twisted.trial import unittest
from twisted.web import error, http, iweb, resource, server
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET, Request, Site
from twisted.web.static import Data
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from ._util import assertIsFilesystemTemporary

class ResourceTests(unittest.TestCase):

    def testListEntities(self):
        if False:
            while True:
                i = 10
        r = resource.Resource()
        self.assertEqual([], r.listEntities())

class SimpleResource(resource.Resource):
    """
    @ivar _contentType: L{None} or a C{str} giving the value of the
        I{Content-Type} header in the response this resource will render.  If it
        is L{None}, no I{Content-Type} header will be set in the response.
    """

    def __init__(self, contentType=None):
        if False:
            i = 10
            return i + 15
        resource.Resource.__init__(self)
        self._contentType = contentType

    def render(self, request):
        if False:
            while True:
                i = 10
        if self._contentType is not None:
            request.responseHeaders.setRawHeaders(b'content-type', [self._contentType])
        if http.CACHED in (request.setLastModified(10), request.setETag(b'MatchingTag')):
            return b''
        else:
            return b'correct'

class ZeroLengthResource(resource.Resource):
    """
    A resource that always returns a zero-length response.
    """

    def render(self, request):
        if False:
            while True:
                i = 10
        return b''

class NoContentResource(resource.Resource):
    """
    A resource that always returns a 204 No Content response without setting
    Content-Length.
    """

    def render(self, request):
        if False:
            for i in range(10):
                print('nop')
        request.setResponseCode(http.NO_CONTENT)
        return b''

class SiteTest(unittest.TestCase):
    """
    Unit tests for L{server.Site}.
    """

    def getAutoExpiringSession(self, site):
        if False:
            return 10
        '\n        Create a new session which auto expires at cleanup.\n\n        @param site: The site on which the session is created.\n        @type site: L{server.Site}\n\n        @return: A newly created session.\n        @rtype: L{server.Session}\n        '
        session = site.makeSession()
        self.addCleanup(session.expire)
        return session

    def test_simplestSite(self):
        if False:
            while True:
                i = 10
        '\n        L{Site.getResourceFor} returns the C{b""} child of the root resource it\n        is constructed with when processing a request for I{/}.\n        '
        sres1 = SimpleResource()
        sres2 = SimpleResource()
        sres1.putChild(b'', sres2)
        site = server.Site(sres1)
        self.assertIdentical(site.getResourceFor(DummyRequest([b''])), sres2, 'Got the wrong resource.')

    def test_defaultRequestFactory(self):
        if False:
            while True:
                i = 10
        '\n        L{server.Request} is the default request factory.\n        '
        site = server.Site(resource=SimpleResource())
        self.assertIs(server.Request, site.requestFactory)

    def test_constructorRequestFactory(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Can be initialized with a custom requestFactory.\n        '
        customFactory = object()
        site = server.Site(resource=SimpleResource(), requestFactory=customFactory)
        self.assertIs(customFactory, site.requestFactory)

    def test_buildProtocol(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a C{Channel} whose C{site} and C{requestFactory} attributes are\n        assigned from the C{site} instance.\n        '
        site = server.Site(SimpleResource())
        channel = site.buildProtocol(None)
        self.assertIs(site, channel.site)
        self.assertIs(site.requestFactory, channel.requestFactory)

    def test_makeSession(self):
        if False:
            i = 10
            return i + 15
        '\n        L{site.getSession} generates a new C{Session} instance with an uid of\n        type L{bytes}.\n        '
        site = server.Site(resource.Resource())
        session = self.getAutoExpiringSession(site)
        self.assertIsInstance(session, server.Session)
        self.assertIsInstance(session.uid, bytes)

    def test_sessionUIDGeneration(self):
        if False:
            print('Hello World!')
        '\n        L{site.getSession} generates L{Session} objects with distinct UIDs from\n        a secure source of entropy.\n        '
        site = server.Site(resource.Resource())
        self.assertIdentical(site._entropy, os.urandom)

        def predictableEntropy(n):
            if False:
                print('Hello World!')
            predictableEntropy.x += 1
            return (chr(predictableEntropy.x) * n).encode('charmap')
        predictableEntropy.x = 0
        self.patch(site, '_entropy', predictableEntropy)
        a = self.getAutoExpiringSession(site)
        b = self.getAutoExpiringSession(site)
        self.assertEqual(a.uid, b'01' * 32)
        self.assertEqual(b.uid, b'02' * 32)
        self.assertEqual(site.counter, 2)

    def test_getSessionExistent(self):
        if False:
            return 10
        '\n        L{site.getSession} gets a previously generated session, by its unique\n        ID.\n        '
        site = server.Site(resource.Resource())
        createdSession = self.getAutoExpiringSession(site)
        retrievedSession = site.getSession(createdSession.uid)
        self.assertIs(createdSession, retrievedSession)

    def test_getSessionNonExistent(self):
        if False:
            i = 10
            return i + 15
        '\n        L{site.getSession} raises a L{KeyError} if the session is not found.\n        '
        site = server.Site(resource.Resource())
        self.assertRaises(KeyError, site.getSession, b'no-such-uid')

class SessionTests(unittest.TestCase):
    """
    Tests for L{server.Session}.
    """

    def setUp(self):
        if False:
            return 10
        '\n        Create a site with one active session using a deterministic, easily\n        controlled clock.\n        '
        self.clock = Clock()
        self.uid = b'unique'
        self.site = server.Site(resource.Resource(), reactor=self.clock)
        self.session = server.Session(self.site, self.uid)
        self.site.sessions[self.uid] = self.session

    def test_defaultReactor(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If no value is passed to L{server.Session.__init__}, the reactor\n        associated with the site is used.\n        '
        site = server.Site(resource.Resource(), reactor=Clock())
        session = server.Session(site, b'123')
        self.assertIdentical(session._reactor, site.reactor)

    def test_explicitReactor(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{Session} accepts the reactor to use as a parameter.\n        '
        site = server.Site(resource.Resource())
        otherReactor = Clock()
        session = server.Session(site, b'123', reactor=otherReactor)
        self.assertIdentical(session._reactor, otherReactor)

    def test_startCheckingExpiration(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{server.Session.startCheckingExpiration} causes the session to expire\n        after L{server.Session.sessionTimeout} seconds without activity.\n        '
        self.session.startCheckingExpiration()
        self.clock.advance(self.session.sessionTimeout - 1)
        self.assertIn(self.uid, self.site.sessions)
        self.clock.advance(1)
        self.assertNotIn(self.uid, self.site.sessions)
        self.assertFalse(self.clock.calls)

    def test_expire(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{server.Session.expire} expires the session.\n        '
        self.session.expire()
        self.assertNotIn(self.uid, self.site.sessions)
        self.assertFalse(self.clock.calls)

    def test_expireWhileChecking(self):
        if False:
            print('Hello World!')
        "\n        L{server.Session.expire} expires the session even if the timeout call\n        isn't due yet.\n        "
        self.session.startCheckingExpiration()
        self.test_expire()

    def test_notifyOnExpire(self):
        if False:
            return 10
        '\n        A function registered with L{server.Session.notifyOnExpire} is called\n        when the session expires.\n        '
        callbackRan = [False]

        def expired():
            if False:
                return 10
            callbackRan[0] = True
        self.session.notifyOnExpire(expired)
        self.session.expire()
        self.assertTrue(callbackRan[0])

    def test_touch(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{server.Session.touch} updates L{server.Session.lastModified} and\n        delays session timeout.\n        '
        self.clock.advance(3)
        self.session.touch()
        self.assertEqual(self.session.lastModified, 3)
        self.session.startCheckingExpiration()
        self.clock.advance(self.session.sessionTimeout - 1)
        self.session.touch()
        self.clock.advance(self.session.sessionTimeout - 1)
        self.assertIn(self.uid, self.site.sessions)
        self.clock.advance(1)
        self.assertNotIn(self.uid, self.site.sessions)

def httpBody(whole):
    if False:
        return 10
    return whole.split(b'\r\n\r\n', 1)[1]

def httpHeader(whole, key):
    if False:
        for i in range(10):
            print('nop')
    key = key.lower()
    headers = whole.split(b'\r\n\r\n', 1)[0]
    for header in headers.split(b'\r\n'):
        if header.lower().startswith(key):
            return header.split(b':', 1)[1].strip()
    return None

def httpCode(whole):
    if False:
        print('Hello World!')
    l1 = whole.split(b'\r\n', 1)[0]
    return int(l1.split()[1])

class ConditionalTests(unittest.TestCase):
    """
    web.server's handling of conditional requests for cache validation.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.resrc = SimpleResource()
        self.resrc.putChild(b'', self.resrc)
        self.resrc.putChild(b'with-content-type', SimpleResource(b'image/jpeg'))
        self.site = server.Site(self.resrc)
        self.site.startFactory()
        self.addCleanup(self.site.stopFactory)
        self.channel = self.site.buildProtocol(None)
        self.transport = http.StringTransport()
        self.transport.close = lambda *a, **kw: None
        self.transport.disconnecting = lambda *a, **kw: 0
        self.transport.getPeer = lambda *a, **kw: 'peer'
        self.transport.getHost = lambda *a, **kw: 'host'
        self.channel.makeConnection(self.transport)

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.channel.connectionLost(None)

    def _modifiedTest(self, modifiedSince=None, etag=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given the value C{modifiedSince} for the I{If-Modified-Since} header or\n        the value C{etag} for the I{If-Not-Match} header, verify that a response\n        with a 200 code, a default Content-Type, and the resource as the body is\n        returned.\n        '
        if modifiedSince is not None:
            validator = b'If-Modified-Since: ' + modifiedSince
        else:
            validator = b'If-Not-Match: ' + etag
        for line in [b'GET / HTTP/1.1', validator, b'']:
            self.channel.dataReceived(line + b'\r\n')
        result = self.transport.getvalue()
        self.assertEqual(httpCode(result), http.OK)
        self.assertEqual(httpBody(result), b'correct')
        self.assertEqual(httpHeader(result, b'Content-Type'), b'text/html')

    def test_modified(self):
        if False:
            i = 10
            return i + 15
        '\n        If a request is made with an I{If-Modified-Since} header value with\n        a timestamp indicating a time before the last modification of the\n        requested resource, a 200 response is returned along with a response\n        body containing the resource.\n        '
        self._modifiedTest(modifiedSince=http.datetimeToString(1))

    def test_unmodified(self):
        if False:
            while True:
                i = 10
        '\n        If a request is made with an I{If-Modified-Since} header value with a\n        timestamp indicating a time after the last modification of the request\n        resource, a 304 response is returned along with an empty response body\n        and no Content-Type header if the application does not set one.\n        '
        for line in [b'GET / HTTP/1.1', b'If-Modified-Since: ' + http.datetimeToString(100), b'']:
            self.channel.dataReceived(line + b'\r\n')
        result = self.transport.getvalue()
        self.assertEqual(httpCode(result), http.NOT_MODIFIED)
        self.assertEqual(httpBody(result), b'')
        self.assertEqual(httpHeader(result, b'Content-Type'), None)

    def test_invalidTimestamp(self):
        if False:
            return 10
        '\n        If a request is made with an I{If-Modified-Since} header value which\n        cannot be parsed, the header is treated as not having been present\n        and a normal 200 response is returned with a response body\n        containing the resource.\n        '
        self._modifiedTest(modifiedSince=b'like, maybe a week ago, I guess?')

    def test_invalidTimestampYear(self):
        if False:
            i = 10
            return i + 15
        '\n        If a request is made with an I{If-Modified-Since} header value which\n        contains a string in the year position which is not an integer, the\n        header is treated as not having been present and a normal 200\n        response is returned with a response body containing the resource.\n        '
        self._modifiedTest(modifiedSince=b'Thu, 01 Jan blah 00:00:10 GMT')

    def test_invalidTimestampTooLongAgo(self):
        if False:
            while True:
                i = 10
        '\n        If a request is made with an I{If-Modified-Since} header value which\n        contains a year before the epoch, the header is treated as not\n        having been present and a normal 200 response is returned with a\n        response body containing the resource.\n        '
        self._modifiedTest(modifiedSince=b'Thu, 01 Jan 1899 00:00:10 GMT')

    def test_invalidTimestampMonth(self):
        if False:
            return 10
        '\n        If a request is made with an I{If-Modified-Since} header value which\n        contains a string in the month position which is not a recognized\n        month abbreviation, the header is treated as not having been present\n        and a normal 200 response is returned with a response body\n        containing the resource.\n        '
        self._modifiedTest(modifiedSince=b'Thu, 01 Blah 1970 00:00:10 GMT')

    def test_etagMatchedNot(self):
        if False:
            return 10
        '\n        If a request is made with an I{If-None-Match} ETag which does not match\n        the current ETag of the requested resource, the header is treated as not\n        having been present and a normal 200 response is returned with a\n        response body containing the resource.\n        '
        self._modifiedTest(etag=b'unmatchedTag')

    def test_etagMatched(self):
        if False:
            i = 10
            return i + 15
        '\n        If a request is made with an I{If-None-Match} ETag which does match the\n        current ETag of the requested resource, a 304 response is returned along\n        with an empty response body.\n        '
        for line in [b'GET / HTTP/1.1', b'If-None-Match: MatchingTag', b'']:
            self.channel.dataReceived(line + b'\r\n')
        result = self.transport.getvalue()
        self.assertEqual(httpHeader(result, b'ETag'), b'MatchingTag')
        self.assertEqual(httpCode(result), http.NOT_MODIFIED)
        self.assertEqual(httpBody(result), b'')

    def test_unmodifiedWithContentType(self):
        if False:
            while True:
                i = 10
        '\n        Similar to L{test_etagMatched}, but the response should include a\n        I{Content-Type} header if the application explicitly sets one.\n\n        This I{Content-Type} header SHOULD NOT be present according to RFC 2616,\n        section 10.3.5.  It will only be present if the application explicitly\n        sets it.\n        '
        for line in [b'GET /with-content-type HTTP/1.1', b'If-None-Match: MatchingTag', b'']:
            self.channel.dataReceived(line + b'\r\n')
        result = self.transport.getvalue()
        self.assertEqual(httpCode(result), http.NOT_MODIFIED)
        self.assertEqual(httpBody(result), b'')
        self.assertEqual(httpHeader(result, b'Content-Type'), b'image/jpeg')

class RequestTests(unittest.TestCase):
    """
    Tests for the HTTP request class, L{server.Request}.
    """

    def test_interface(self):
        if False:
            print('Hello World!')
        '\n        L{server.Request} instances provide L{iweb.IRequest}.\n        '
        self.assertTrue(verifyObject(iweb.IRequest, server.Request(DummyChannel(), True)))

    def test_hashable(self):
        if False:
            print('Hello World!')
        '\n        L{server.Request} instances are hashable, thus can be put in a mapping.\n        '
        request = server.Request(DummyChannel(), True)
        hash(request)

    def testChildLink(self):
        if False:
            print('Hello World!')
        request = server.Request(DummyChannel(), 1)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        self.assertEqual(request.childLink(b'baz'), b'bar/baz')
        request = server.Request(DummyChannel(), 1)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar/', b'HTTP/1.0')
        self.assertEqual(request.childLink(b'baz'), b'baz')

    def testPrePathURLSimple(self):
        if False:
            while True:
                i = 10
        request = server.Request(DummyChannel(), 1)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        request.setHost(b'example.com', 80)
        self.assertEqual(request.prePathURL(), b'http://example.com/foo/bar')

    def testPrePathURLNonDefault(self):
        if False:
            print('Hello World!')
        d = DummyChannel()
        d.transport.port = 81
        request = server.Request(d, 1)
        request.setHost(b'example.com', 81)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        self.assertEqual(request.prePathURL(), b'http://example.com:81/foo/bar')

    def testPrePathURLSSLPort(self):
        if False:
            print('Hello World!')
        d = DummyChannel()
        d.transport.port = 443
        request = server.Request(d, 1)
        request.setHost(b'example.com', 443)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        self.assertEqual(request.prePathURL(), b'http://example.com:443/foo/bar')

    def testPrePathURLSSLPortAndSSL(self):
        if False:
            return 10
        d = DummyChannel()
        d.transport = DummyChannel.SSL()
        d.transport.port = 443
        request = server.Request(d, 1)
        request.setHost(b'example.com', 443)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        self.assertEqual(request.prePathURL(), b'https://example.com/foo/bar')

    def testPrePathURLHTTPPortAndSSL(self):
        if False:
            for i in range(10):
                print('nop')
        d = DummyChannel()
        d.transport = DummyChannel.SSL()
        d.transport.port = 80
        request = server.Request(d, 1)
        request.setHost(b'example.com', 80)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        self.assertEqual(request.prePathURL(), b'https://example.com:80/foo/bar')

    def testPrePathURLSSLNonDefault(self):
        if False:
            print('Hello World!')
        d = DummyChannel()
        d.transport = DummyChannel.SSL()
        d.transport.port = 81
        request = server.Request(d, 1)
        request.setHost(b'example.com', 81)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        self.assertEqual(request.prePathURL(), b'https://example.com:81/foo/bar')

    def testPrePathURLSetSSLHost(self):
        if False:
            return 10
        d = DummyChannel()
        d.transport.port = 81
        request = server.Request(d, 1)
        request.setHost(b'foo.com', 81, 1)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        self.assertEqual(request.prePathURL(), b'https://foo.com:81/foo/bar')

    def test_prePathURLQuoting(self):
        if False:
            return 10
        '\n        L{Request.prePathURL} quotes special characters in the URL segments to\n        preserve the original meaning.\n        '
        d = DummyChannel()
        request = server.Request(d, 1)
        request.setHost(b'example.com', 80)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo%2Fbar', b'HTTP/1.0')
        self.assertEqual(request.prePathURL(), b'http://example.com/foo%2Fbar')

    def test_processingFailedNoTracebackByDefault(self):
        if False:
            i = 10
            return i + 15
        '\n        By default, L{Request.processingFailed} does not write out the failure,\n        but give a generic error message, as L{Site.displayTracebacks} is\n        disabled by default.\n        '
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        d = DummyChannel()
        request = server.Request(d, 1)
        request.site = server.Site(resource.Resource())
        fail = failure.Failure(Exception('Oh no!'))
        request.processingFailed(fail)
        self.assertNotIn(b'Oh no!', request.transport.written.getvalue())
        self.assertIn(b'Processing Failed', request.transport.written.getvalue())
        self.assertEquals(1, len(logObserver))
        event = logObserver[0]
        f = event['log_failure']
        self.assertIsInstance(f.value, Exception)
        self.assertEquals(f.getErrorMessage(), 'Oh no!')
        self.assertEqual(1, len(self.flushLoggedErrors()))

    def test_processingFailedNoTraceback(self):
        if False:
            return 10
        '\n        L{Request.processingFailed} when the site has C{displayTracebacks} set\n        to C{False} does not write out the failure, but give a generic error\n        message.\n        '
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        d = DummyChannel()
        request = server.Request(d, 1)
        request.site = server.Site(resource.Resource())
        request.site.displayTracebacks = False
        fail = failure.Failure(Exception('Oh no!'))
        request.processingFailed(fail)
        self.assertNotIn(b'Oh no!', request.transport.written.getvalue())
        self.assertIn(b'Processing Failed', request.transport.written.getvalue())
        self.assertEquals(1, len(logObserver))
        event = logObserver[0]
        f = event['log_failure']
        self.assertIsInstance(f.value, Exception)
        self.assertEquals(f.getErrorMessage(), 'Oh no!')
        self.assertEqual(1, len(self.flushLoggedErrors()))

    def test_processingFailedDisplayTraceback(self):
        if False:
            i = 10
            return i + 15
        '\n        L{Request.processingFailed} when the site has C{displayTracebacks} set\n        to C{True} writes out the failure.\n        '
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        d = DummyChannel()
        request = server.Request(d, 1)
        request.site = server.Site(resource.Resource())
        request.site.displayTracebacks = True
        fail = failure.Failure(Exception('Oh no!'))
        request.processingFailed(fail)
        self.assertIn(b'Oh no!', request.transport.written.getvalue())
        event = logObserver[0]
        f = event['log_failure']
        self.assertIsInstance(f.value, Exception)
        self.assertEquals(f.getErrorMessage(), 'Oh no!')
        self.assertEqual(1, len(self.flushLoggedErrors()))

    def test_processingFailedDisplayTracebackHandlesUnicode(self):
        if False:
            return 10
        '\n        L{Request.processingFailed} when the site has C{displayTracebacks} set\n        to C{True} writes out the failure, making UTF-8 items into HTML\n        entities.\n        '
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        d = DummyChannel()
        request = server.Request(d, 1)
        request.site = server.Site(resource.Resource())
        request.site.displayTracebacks = True
        fail = failure.Failure(Exception('☃'))
        request.processingFailed(fail)
        self.assertIn(b'&#9731;', request.transport.written.getvalue())
        self.flushLoggedErrors(UnicodeError)
        event = logObserver[0]
        f = event['log_failure']
        self.assertIsInstance(f.value, Exception)
        self.assertEqual(1, len(self.flushLoggedErrors()))

    def test_sessionDifferentFromSecureSession(self):
        if False:
            i = 10
            return i + 15
        '\n        L{Request.session} and L{Request.secure_session} should be two separate\n        sessions with unique ids and different cookies.\n        '
        d = DummyChannel()
        d.transport = DummyChannel.SSL()
        request = server.Request(d, 1)
        request.site = server.Site(resource.Resource())
        request.sitepath = []
        secureSession = request.getSession()
        self.assertIsNotNone(secureSession)
        self.addCleanup(secureSession.expire)
        self.assertEqual(request.cookies[0].split(b'=')[0], b'TWISTED_SECURE_SESSION')
        session = request.getSession(forceNotSecure=True)
        self.assertIsNotNone(session)
        self.assertEqual(request.cookies[1].split(b'=')[0], b'TWISTED_SESSION')
        self.addCleanup(session.expire)
        self.assertNotEqual(session.uid, secureSession.uid)

    def test_sessionAttribute(self):
        if False:
            return 10
        '\n        On a L{Request}, the C{session} attribute retrieves the associated\n        L{Session} only if it has been initialized.  If the request is secure,\n        it retrieves the secure session.\n        '
        site = server.Site(resource.Resource())
        d = DummyChannel()
        d.transport = DummyChannel.SSL()
        request = server.Request(d, 1)
        request.site = site
        request.sitepath = []
        self.assertIs(request.session, None)
        insecureSession = request.getSession(forceNotSecure=True)
        self.addCleanup(insecureSession.expire)
        self.assertIs(request.session, None)
        secureSession = request.getSession()
        self.addCleanup(secureSession.expire)
        self.assertIsNot(secureSession, None)
        self.assertIsNot(secureSession, insecureSession)
        self.assertIs(request.session, secureSession)

    def test_sessionCaching(self):
        if False:
            while True:
                i = 10
        '\n        L{Request.getSession} creates the session object only once per request;\n        if it is called twice it returns the identical result.\n        '
        site = server.Site(resource.Resource())
        d = DummyChannel()
        request = server.Request(d, 1)
        request.site = site
        request.sitepath = []
        session1 = request.getSession()
        self.addCleanup(session1.expire)
        session2 = request.getSession()
        self.assertIs(session1, session2)

    def test_retrieveExistingSession(self):
        if False:
            return 10
        '\n        L{Request.getSession} retrieves an existing session if the relevant\n        cookie is set in the incoming request.\n        '
        site = server.Site(resource.Resource())
        d = DummyChannel()
        request = server.Request(d, 1)
        request.site = site
        request.sitepath = []
        mySession = server.Session(site, b'special-id')
        site.sessions[mySession.uid] = mySession
        request.received_cookies[b'TWISTED_SESSION'] = mySession.uid
        self.assertIs(request.getSession(), mySession)

    def test_retrieveNonExistentSession(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{Request.getSession} generates a new session if the session ID\n        advertised in the cookie from the incoming request is not found.\n        '
        site = server.Site(resource.Resource())
        d = DummyChannel()
        request = server.Request(d, 1)
        request.site = site
        request.sitepath = []
        request.received_cookies[b'TWISTED_SESSION'] = b'does-not-exist'
        session = request.getSession()
        self.assertIsNotNone(session)
        self.addCleanup(session.expire)
        self.assertTrue(request.cookies[0].startswith(b'TWISTED_SESSION='))
        self.assertNotIn(b'does-not-exist', request.cookies[0])

    def test_getSessionExpired(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{Request.getSession} generates a new session when the previous\n        session has expired.\n        '
        clock = Clock()
        site = server.Site(resource.Resource())
        d = DummyChannel()
        request = server.Request(d, 1)
        request.site = site
        request.sitepath = []

        def sessionFactoryWithClock(site, uid):
            if False:
                return 10
            '\n            Forward to normal session factory, but inject the clock.\n\n            @param site: The site on which the session is created.\n            @type site: L{server.Site}\n\n            @param uid: A unique identifier for the session.\n            @type uid: C{bytes}\n\n            @return: A newly created session.\n            @rtype: L{server.Session}\n            '
            session = sessionFactory(site, uid)
            session._reactor = clock
            return session
        sessionFactory = site.sessionFactory
        site.sessionFactory = sessionFactoryWithClock
        initialSession = request.getSession()
        clock.advance(sessionFactory.sessionTimeout)
        newSession = request.getSession()
        self.addCleanup(newSession.expire)
        self.assertIsNot(initialSession, newSession)
        self.assertNotEqual(initialSession.uid, newSession.uid)

    def test_OPTIONSStar(self):
        if False:
            return 10
        '\n        L{Request} handles OPTIONS * requests by doing a fast-path return of\n        200 OK.\n        '
        d = DummyChannel()
        request = server.Request(d, 1)
        request.setHost(b'example.com', 80)
        request.gotLength(0)
        request.requestReceived(b'OPTIONS', b'*', b'HTTP/1.1')
        response = d.transport.written.getvalue()
        self.assertTrue(response.startswith(b'HTTP/1.1 200 OK'))
        self.assertIn(b'Content-Length: 0\r\n', response)

    def test_rejectNonOPTIONSStar(self):
        if False:
            print('Hello World!')
        '\n        L{Request} handles any non-OPTIONS verb requesting the * path by doing\n        a fast-return 405 Method Not Allowed, indicating only the support for\n        OPTIONS.\n        '
        d = DummyChannel()
        request = server.Request(d, 1)
        request.setHost(b'example.com', 80)
        request.gotLength(0)
        request.requestReceived(b'GET', b'*', b'HTTP/1.1')
        response = d.transport.written.getvalue()
        self.assertTrue(response.startswith(b'HTTP/1.1 405 Method Not Allowed'))
        self.assertIn(b'Content-Length: 0\r\n', response)
        self.assertIn(b'Allow: OPTIONS\r\n', response)

    def test_noDefaultContentTypeOnZeroLengthResponse(self):
        if False:
            print('Hello World!')
        '\n        Responses with no length do not have a default content-type applied.\n        '
        resrc = ZeroLengthResource()
        resrc.putChild(b'', resrc)
        site = server.Site(resrc)
        d = DummyChannel()
        d.site = site
        request = server.Request(d, 1)
        request.site = site
        request.setHost(b'example.com', 80)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/', b'HTTP/1.1')
        self.assertNotIn(b'content-type', request.transport.written.getvalue().lower())

    def test_noDefaultContentTypeOn204Response(self):
        if False:
            print('Hello World!')
        '\n        Responses with a 204 status code have no default content-type applied.\n        '
        resrc = NoContentResource()
        resrc.putChild(b'', resrc)
        site = server.Site(resrc)
        d = DummyChannel()
        d.site = site
        request = server.Request(d, 1)
        request.site = site
        request.setHost(b'example.com', 80)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/', b'HTTP/1.1')
        response = request.transport.written.getvalue()
        self.assertTrue(response.startswith(b'HTTP/1.1 204 No Content\r\n'))
        self.assertNotIn(b'content-type', response.lower())

    def test_defaultSmallContentFile(self):
        if False:
            i = 10
            return i + 15
        "\n        L{http.Request} creates a L{BytesIO} if the content length is small and\n        the site doesn't offer to create one.\n        "
        request = server.Request(DummyChannel())
        request.gotLength(100000 - 1)
        self.assertIsInstance(request.content, BytesIO)

    def test_defaultLargerContentFile(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        L{http.Request} creates a temporary file on the filesystem if the\n        content length is larger and the site doesn't offer to create one.\n        "
        request = server.Request(DummyChannel())
        request.gotLength(100000)
        assertIsFilesystemTemporary(self, request.content)

    def test_defaultUnknownSizeContentFile(self):
        if False:
            while True:
                i = 10
        "\n        L{http.Request} creates a temporary file on the filesystem if the\n        content length is not known and the site doesn't offer to create one.\n        "
        request = server.Request(DummyChannel())
        request.gotLength(None)
        assertIsFilesystemTemporary(self, request.content)

    def test_siteSuppliedContentFile(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{http.Request} uses L{Site.getContentFile}, if it exists, to get a\n        file-like object for the request content.\n        '
        lengths = []
        contentFile = BytesIO()
        site = server.Site(resource.Resource())

        def getContentFile(length):
            if False:
                print('Hello World!')
            lengths.append(length)
            return contentFile
        site.getContentFile = getContentFile
        channel = DummyChannel()
        channel.site = site
        request = server.Request(channel)
        request.gotLength(12345)
        self.assertEqual([12345], lengths)
        self.assertIs(contentFile, request.content)

class GzipEncoderTests(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.channel = DummyChannel()
        staticResource = Data(b'Some data', 'text/plain')
        wrapped = resource.EncodingResourceWrapper(staticResource, [server.GzipEncoderFactory()])
        self.channel.site.resource.putChild(b'foo', wrapped)

    def test_interfaces(self):
        if False:
            while True:
                i = 10
        '\n        L{server.GzipEncoderFactory} implements the\n        L{iweb._IRequestEncoderFactory} and its C{encoderForRequest} returns an\n        instance of L{server._GzipEncoder} which implements\n        L{iweb._IRequestEncoder}.\n        '
        request = server.Request(self.channel, False)
        request.gotLength(0)
        request.requestHeaders.setRawHeaders(b'Accept-Encoding', [b'gzip,deflate'])
        factory = server.GzipEncoderFactory()
        self.assertTrue(verifyObject(iweb._IRequestEncoderFactory, factory))
        encoder = factory.encoderForRequest(request)
        self.assertTrue(verifyObject(iweb._IRequestEncoder, encoder))

    def test_encoding(self):
        if False:
            return 10
        '\n        If the client request passes a I{Accept-Encoding} header which mentions\n        gzip, L{server._GzipEncoder} automatically compresses the data.\n        '
        request = server.Request(self.channel, False)
        request.gotLength(0)
        request.requestHeaders.setRawHeaders(b'Accept-Encoding', [b'gzip,deflate'])
        request.requestReceived(b'GET', b'/foo', b'HTTP/1.0')
        data = self.channel.transport.written.getvalue()
        self.assertNotIn(b'Content-Length', data)
        self.assertIn(b'Content-Encoding: gzip\r\n', data)
        body = data[data.find(b'\r\n\r\n') + 4:]
        self.assertEqual(b'Some data', zlib.decompress(body, 16 + zlib.MAX_WBITS))

    def test_whitespaceInAcceptEncoding(self):
        if False:
            print('Hello World!')
        '\n        If the client request passes a I{Accept-Encoding} header which mentions\n        gzip, with whitespace inbetween the encoding name and the commas,\n        L{server._GzipEncoder} automatically compresses the data.\n        '
        request = server.Request(self.channel, False)
        request.gotLength(0)
        request.requestHeaders.setRawHeaders(b'Accept-Encoding', [b'deflate, gzip'])
        request.requestReceived(b'GET', b'/foo', b'HTTP/1.0')
        data = self.channel.transport.written.getvalue()
        self.assertNotIn(b'Content-Length', data)
        self.assertIn(b'Content-Encoding: gzip\r\n', data)
        body = data[data.find(b'\r\n\r\n') + 4:]
        self.assertEqual(b'Some data', zlib.decompress(body, 16 + zlib.MAX_WBITS))

    def test_nonEncoding(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        L{server.GzipEncoderFactory} doesn't return a L{server._GzipEncoder} if\n        the I{Accept-Encoding} header doesn't mention gzip support.\n        "
        request = server.Request(self.channel, False)
        request.gotLength(0)
        request.requestHeaders.setRawHeaders(b'Accept-Encoding', [b'foo,bar'])
        request.requestReceived(b'GET', b'/foo', b'HTTP/1.0')
        data = self.channel.transport.written.getvalue()
        self.assertIn(b'Content-Length', data)
        self.assertNotIn(b'Content-Encoding: gzip\r\n', data)
        body = data[data.find(b'\r\n\r\n') + 4:]
        self.assertEqual(b'Some data', body)

    def test_multipleAccept(self):
        if False:
            while True:
                i = 10
        '\n        If there are multiple I{Accept-Encoding} header,\n        L{server.GzipEncoderFactory} reads them properly to detect if gzip is\n        supported.\n        '
        request = server.Request(self.channel, False)
        request.gotLength(0)
        request.requestHeaders.setRawHeaders(b'Accept-Encoding', [b'deflate', b'gzip'])
        request.requestReceived(b'GET', b'/foo', b'HTTP/1.0')
        data = self.channel.transport.written.getvalue()
        self.assertNotIn(b'Content-Length', data)
        self.assertIn(b'Content-Encoding: gzip\r\n', data)
        body = data[data.find(b'\r\n\r\n') + 4:]
        self.assertEqual(b'Some data', zlib.decompress(body, 16 + zlib.MAX_WBITS))

    def test_alreadyEncoded(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the content is already encoded and the I{Content-Encoding} header is\n        set, L{server.GzipEncoderFactory} properly appends gzip to it.\n        '
        request = server.Request(self.channel, False)
        request.gotLength(0)
        request.requestHeaders.setRawHeaders(b'Accept-Encoding', [b'deflate', b'gzip'])
        request.responseHeaders.setRawHeaders(b'Content-Encoding', [b'deflate'])
        request.requestReceived(b'GET', b'/foo', b'HTTP/1.0')
        data = self.channel.transport.written.getvalue()
        self.assertNotIn(b'Content-Length', data)
        self.assertIn(b'Content-Encoding: deflate,gzip\r\n', data)
        body = data[data.find(b'\r\n\r\n') + 4:]
        self.assertEqual(b'Some data', zlib.decompress(body, 16 + zlib.MAX_WBITS))

    def test_multipleEncodingLines(self):
        if False:
            while True:
                i = 10
        '\n        If there are several I{Content-Encoding} headers,\n        L{server.GzipEncoderFactory} normalizes it and appends gzip to the\n        field value.\n        '
        request = server.Request(self.channel, False)
        request.gotLength(0)
        request.requestHeaders.setRawHeaders(b'Accept-Encoding', [b'deflate', b'gzip'])
        request.responseHeaders.setRawHeaders(b'Content-Encoding', [b'foo', b'bar'])
        request.requestReceived(b'GET', b'/foo', b'HTTP/1.0')
        data = self.channel.transport.written.getvalue()
        self.assertNotIn(b'Content-Length', data)
        self.assertIn(b'Content-Encoding: foo,bar,gzip\r\n', data)
        body = data[data.find(b'\r\n\r\n') + 4:]
        self.assertEqual(b'Some data', zlib.decompress(body, 16 + zlib.MAX_WBITS))

class RootResource(resource.Resource):
    isLeaf = 0

    def getChildWithDefault(self, name, request):
        if False:
            while True:
                i = 10
        request.rememberRootURL()
        return resource.Resource.getChildWithDefault(self, name, request)

    def render(self, request):
        if False:
            print('Hello World!')
        return ''

class RememberURLTests(unittest.TestCase):
    """
    Tests for L{server.Site}'s root request URL calculation.
    """

    def createServer(self, r):
        if False:
            while True:
                i = 10
        '\n        Create a L{server.Site} bound to a L{DummyChannel} and the\n        given resource as its root.\n\n        @param r: The root resource.\n        @type r: L{resource.Resource}\n\n        @return: The channel to which the site is bound.\n        @rtype: L{DummyChannel}\n        '
        chan = DummyChannel()
        chan.site = server.Site(r)
        return chan

    def testSimple(self):
        if False:
            print('Hello World!')
        "\n        The path component of the root URL of a L{server.Site} whose\n        root resource is below C{/} is that resource's path, and the\n        netloc component is the L{site.Server}'s own host and port.\n        "
        r = resource.Resource()
        r.isLeaf = 0
        rr = RootResource()
        r.putChild(b'foo', rr)
        rr.putChild(b'', rr)
        rr.putChild(b'bar', resource.Resource())
        chan = self.createServer(r)
        for url in [b'/foo/', b'/foo/bar', b'/foo/bar/baz', b'/foo/bar/']:
            request = server.Request(chan, 1)
            request.setHost(b'example.com', 81)
            request.gotLength(0)
            request.requestReceived(b'GET', url, b'HTTP/1.0')
            self.assertEqual(request.getRootURL(), b'http://example.com:81/foo')

    def testRoot(self):
        if False:
            i = 10
            return i + 15
        "\n        The path component of the root URL of a L{server.Site} whose\n        root resource is at C{/} is C{/}, and the netloc component is\n        the L{site.Server}'s own host and port.\n        "
        rr = RootResource()
        rr.putChild(b'', rr)
        rr.putChild(b'bar', resource.Resource())
        chan = self.createServer(rr)
        for url in [b'/', b'/bar', b'/bar/baz', b'/bar/']:
            request = server.Request(chan, 1)
            request.setHost(b'example.com', 81)
            request.gotLength(0)
            request.requestReceived(b'GET', url, b'HTTP/1.0')
            self.assertEqual(request.getRootURL(), b'http://example.com:81/')

class NewRenderResource(resource.Resource):

    def render_GET(self, request):
        if False:
            return 10
        return b'hi hi'

    def render_HEH(self, request):
        if False:
            i = 10
            return i + 15
        return b'ho ho'

@implementer(resource.IResource)
class HeadlessResource:
    """
    A resource that implements GET but not HEAD.
    """
    allowedMethods = [b'GET']

    def render(self, request):
        if False:
            print('Hello World!')
        '\n        Leave the request open for future writes.\n        '
        self.request = request
        if request.method not in self.allowedMethods:
            raise error.UnsupportedMethod(self.allowedMethods)
        self.request.write(b'some data')
        return server.NOT_DONE_YET

    def isLeaf(self):
        if False:
            while True:
                i = 10
        '\n        # IResource.isLeaf\n        '
        raise NotImplementedError()

    def getChildWithDefault(self, name, request):
        if False:
            return 10
        '\n        # IResource.getChildWithDefault\n        '
        raise NotImplementedError()

    def putChild(self, path, child):
        if False:
            print('Hello World!')
        '\n        # IResource.putChild\n        '
        raise NotImplementedError()

class NewRenderTests(unittest.TestCase):
    """
    Tests for L{server.Request.render}.
    """

    def _getReq(self, resource=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a request object with a stub channel and install the\n        passed resource at /newrender. If no resource is passed,\n        create one.\n        '
        d = DummyChannel()
        if resource is None:
            resource = NewRenderResource()
        d.site.resource.putChild(b'newrender', resource)
        d.transport.port = 81
        request = server.Request(d, 1)
        request.setHost(b'example.com', 81)
        request.gotLength(0)
        return request

    def testGoodMethods(self):
        if False:
            for i in range(10):
                print('nop')
        req = self._getReq()
        req.requestReceived(b'GET', b'/newrender', b'HTTP/1.0')
        self.assertEqual(req.transport.written.getvalue().splitlines()[-1], b'hi hi')
        req = self._getReq()
        req.requestReceived(b'HEH', b'/newrender', b'HTTP/1.0')
        self.assertEqual(req.transport.written.getvalue().splitlines()[-1], b'ho ho')

    def testBadMethods(self):
        if False:
            return 10
        req = self._getReq()
        req.requestReceived(b'CONNECT', b'/newrender', b'HTTP/1.0')
        self.assertEqual(req.code, 501)
        req = self._getReq()
        req.requestReceived(b'hlalauguG', b'/newrender', b'HTTP/1.0')
        self.assertEqual(req.code, 501)

    def test_notAllowedMethod(self):
        if False:
            return 10
        '\n        When trying to invoke a method not in the allowed method list, we get\n        a response saying it is not allowed.\n        '
        req = self._getReq()
        req.requestReceived(b'POST', b'/newrender', b'HTTP/1.0')
        self.assertEqual(req.code, 405)
        self.assertTrue(req.responseHeaders.hasHeader(b'allow'))
        raw_header = req.responseHeaders.getRawHeaders(b'allow')[0]
        allowed = sorted((h.strip() for h in raw_header.split(b',')))
        self.assertEqual([b'GET', b'HEAD', b'HEH'], allowed)

    def testImplicitHead(self):
        if False:
            for i in range(10):
                print('nop')
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        req = self._getReq()
        req.requestReceived(b'HEAD', b'/newrender', b'HTTP/1.0')
        self.assertEqual(req.code, 200)
        self.assertEqual(-1, req.transport.written.getvalue().find(b'hi hi'))
        self.assertEquals(1, len(logObserver))
        event = logObserver[0]
        self.assertEquals(event['log_level'], LogLevel.info)

    def test_unsupportedHead(self):
        if False:
            while True:
                i = 10
        '\n        HEAD requests against resource that only claim support for GET\n        should not include a body in the response.\n        '
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        resource = HeadlessResource()
        req = self._getReq(resource)
        req.requestReceived(b'HEAD', b'/newrender', b'HTTP/1.0')
        (headers, body) = req.transport.written.getvalue().split(b'\r\n\r\n')
        self.assertEqual(req.code, 200)
        self.assertEqual(body, b'')
        self.assertEquals(2, len(logObserver))

    def test_noBytesResult(self):
        if False:
            return 10
        '\n        When implemented C{render} method does not return bytes an internal\n        server error is returned.\n        '

        class RiggedRepr:

            def __repr__(self) -> str:
                if False:
                    while True:
                        i = 10
                return 'my>repr'
        result = RiggedRepr()
        no_bytes_resource = resource.Resource()
        no_bytes_resource.render = lambda request: result
        request = self._getReq(no_bytes_resource)
        request.requestReceived(b'GET', b'/newrender', b'HTTP/1.0')
        (headers, body) = request.transport.written.getvalue().split(b'\r\n\r\n')
        self.assertEqual(request.code, 500)
        expected = ['', '<html>', '  <head><title>500 - Request did not return bytes</title></head>', '  <body>', '    <h1>Request did not return bytes</h1>', '    <p>Request: <pre>&lt;%s&gt;</pre><br />Resource: <pre>&lt;%s&gt;</pre><br />Value: <pre>my&gt;repr</pre></p>' % (reflect.safe_repr(request)[1:-1], reflect.safe_repr(no_bytes_resource)[1:-1]), '  </body>', '</html>', '']
        self.assertEqual('\n'.join(expected).encode('ascii'), body)

class GettableResource(resource.Resource):
    """
    Used by AllowedMethodsTests to simulate an allowed method.
    """

    def render_GET(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def render_fred_render_ethel(self):
        if False:
            print('Hello World!')
        '\n        The unusual method name is designed to test the culling method\n        in C{twisted.web.resource._computeAllowedMethods}.\n        '
        pass

class AllowedMethodsTests(unittest.TestCase):
    """
    'C{twisted.web.resource._computeAllowedMethods} is provided by a
    default should the subclass not provide the method.
    """

    def _getReq(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate a dummy request for use by C{_computeAllowedMethod} tests.\n        '
        d = DummyChannel()
        d.site.resource.putChild(b'gettableresource', GettableResource())
        d.transport.port = 81
        request = server.Request(d, 1)
        request.setHost(b'example.com', 81)
        request.gotLength(0)
        return request

    def test_computeAllowedMethods(self):
        if False:
            return 10
        "\n        C{_computeAllowedMethods} will search through the\n        'gettableresource' for all attributes/methods of the form\n        'render_{method}' ('render_GET', for example) and return a list of\n        the methods. 'HEAD' will always be included from the\n        resource.Resource superclass.\n        "
        res = GettableResource()
        allowedMethods = resource._computeAllowedMethods(res)
        self.assertEqual(set(allowedMethods), {b'GET', b'HEAD', b'fred_render_ethel'})

    def test_notAllowed(self):
        if False:
            return 10
        "\n        When an unsupported method is requested, the default\n        L{_computeAllowedMethods} method will be called to determine the\n        allowed methods, and the HTTP 405 'Method Not Allowed' status will\n        be returned with the allowed methods will be returned in the\n        'Allow' header.\n        "
        req = self._getReq()
        req.requestReceived(b'POST', b'/gettableresource', b'HTTP/1.0')
        self.assertEqual(req.code, 405)
        self.assertEqual(set(req.responseHeaders.getRawHeaders(b'allow')[0].split(b', ')), {b'GET', b'HEAD', b'fred_render_ethel'})

    def test_notAllowedQuoting(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        When an unsupported method response is generated, an HTML message will\n        be displayed.  That message should include a quoted form of the URI and,\n        since that value come from a browser and shouldn't necessarily be\n        trusted.\n        "
        req = self._getReq()
        req.requestReceived(b'POST', b'/gettableresource?value=<script>bad', b'HTTP/1.0')
        self.assertEqual(req.code, 405)
        renderedPage = req.transport.written.getvalue()
        self.assertNotIn(b'<script>bad', renderedPage)
        self.assertIn(b'&lt;script&gt;bad', renderedPage)

    def test_notImplementedQuoting(self):
        if False:
            i = 10
            return i + 15
        "\n        When an not-implemented method response is generated, an HTML message\n        will be displayed.  That message should include a quoted form of the\n        requested method, since that value come from a browser and shouldn't\n        necessarily be trusted.\n        "
        req = self._getReq()
        req.requestReceived(b'<style>bad', b'/gettableresource', b'HTTP/1.0')
        self.assertEqual(req.code, 501)
        renderedPage = req.transport.written.getvalue()
        self.assertNotIn(b'<style>bad', renderedPage)
        self.assertIn(b'&lt;style&gt;bad', renderedPage)

class DummyRequestForLogTest(DummyRequest):
    uri = b'/dummy'
    code = 123
    clientproto = b'HTTP/1.0'
    sentLength = None
    client = IPv4Address('TCP', '1.2.3.4', 12345)

    def getClientIP(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        As L{getClientIP} is deprecated, no log formatter should call it.\n        '
        raise NotImplementedError('Call to deprecated getClientIP method (use getClientAddress instead)')

class AccessLogTestsMixin:
    """
    A mixin for L{TestCase} subclasses defining tests that apply to
    L{HTTPFactory} and its subclasses.
    """

    def factory(self, *args, **kwargs):
        if False:
            return 10
        '\n        Get the factory class to apply logging tests to.\n\n        Subclasses must override this method.\n        '
        raise NotImplementedError('Subclass failed to override factory')

    def test_combinedLogFormat(self):
        if False:
            print('Hello World!')
        "\n        The factory's C{log} method writes a I{combined log format} line to the\n        factory's log file.\n        "
        reactor = Clock()
        reactor.advance(1234567890)
        logPath = self.mktemp()
        factory = self.factory(logPath=logPath, reactor=reactor)
        factory.startFactory()
        try:
            factory.log(DummyRequestForLogTest(factory))
        finally:
            factory.stopFactory()
        self.assertEqual(b'"1.2.3.4" - - [13/Feb/2009:23:31:30 +0000] "GET /dummy HTTP/1.0" 123 - "-" "-"\n', FilePath(logPath).getContent())

    def test_logFormatOverride(self):
        if False:
            print('Hello World!')
        '\n        If the factory is initialized with a custom log formatter then that\n        formatter is used to generate lines for the log file.\n        '

        def notVeryGoodFormatter(timestamp, request):
            if False:
                for i in range(10):
                    print('nop')
            return 'this is a bad log format'
        reactor = Clock()
        reactor.advance(1234567890)
        logPath = self.mktemp()
        factory = self.factory(logPath=logPath, logFormatter=notVeryGoodFormatter)
        factory._reactor = reactor
        factory.startFactory()
        try:
            factory.log(DummyRequestForLogTest(factory))
        finally:
            factory.stopFactory()
        self.assertEqual(b'this is a bad log format\n', FilePath(logPath).getContent())

class HTTPFactoryAccessLogTests(AccessLogTestsMixin, unittest.TestCase):
    """
    Tests for L{http.HTTPFactory.log}.
    """
    factory = http.HTTPFactory

class SiteAccessLogTests(AccessLogTestsMixin, unittest.TestCase):
    """
    Tests for L{server.Site.log}.
    """

    def factory(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return server.Site(resource.Resource(), *args, **kwargs)

class CombinedLogFormatterTests(unittest.TestCase):
    """
    Tests for L{twisted.web.http.combinedLogFormatter}.
    """

    def test_interface(self):
        if False:
            while True:
                i = 10
        '\n        L{combinedLogFormatter} provides L{IAccessLogFormatter}.\n        '
        self.assertTrue(verifyObject(iweb.IAccessLogFormatter, http.combinedLogFormatter))

    def test_nonASCII(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Bytes in fields of the request which are not part of ASCII are escaped\n        in the result.\n        '
        reactor = Clock()
        reactor.advance(1234567890)
        timestamp = http.datetimeToLogString(reactor.seconds())
        request = DummyRequestForLogTest(http.HTTPFactory(reactor=reactor))
        request.client = IPv4Address('TCP', b'evil x-forwarded-for \x80', 12345)
        request.method = b'POS\x81'
        request.protocol = b'HTTP/1.\x82'
        request.requestHeaders.addRawHeader(b'referer', b'evil \x83')
        request.requestHeaders.addRawHeader(b'user-agent', b'evil \x84')
        line = http.combinedLogFormatter(timestamp, request)
        self.assertEqual('"evil x-forwarded-for \\x80" - - [13/Feb/2009:23:31:30 +0000] "POS\\x81 /dummy HTTP/1.0" 123 - "evil \\x83" "evil \\x84"', line)

    def test_clientAddrIPv6(self):
        if False:
            i = 10
            return i + 15
        '\n        A request from an IPv6 client is logged with that IP address.\n        '
        reactor = Clock()
        reactor.advance(1234567890)
        timestamp = http.datetimeToLogString(reactor.seconds())
        request = DummyRequestForLogTest(http.HTTPFactory(reactor=reactor))
        request.client = IPv6Address('TCP', b'::1', 12345)
        line = http.combinedLogFormatter(timestamp, request)
        self.assertEqual('"::1" - - [13/Feb/2009:23:31:30 +0000] "GET /dummy HTTP/1.0" 123 - "-" "-"', line)

    def test_clientAddrUnknown(self):
        if False:
            return 10
        '\n        A request made from an unknown address type is logged as C{"-"}.\n        '

        @implementer(interfaces.IAddress)
        class UnknowableAddress:
            """
            An L{IAddress} which L{combinedLogFormatter} cannot have
            foreknowledge of.
            """
        reactor = Clock()
        reactor.advance(1234567890)
        timestamp = http.datetimeToLogString(reactor.seconds())
        request = DummyRequestForLogTest(http.HTTPFactory(reactor=reactor))
        request.client = UnknowableAddress()
        line = http.combinedLogFormatter(timestamp, request)
        self.assertTrue(line.startswith('"-" '))

class ProxiedLogFormatterTests(unittest.TestCase):
    """
    Tests for L{twisted.web.http.proxiedLogFormatter}.
    """

    def test_interface(self):
        if False:
            return 10
        '\n        L{proxiedLogFormatter} provides L{IAccessLogFormatter}.\n        '
        self.assertTrue(verifyObject(iweb.IAccessLogFormatter, http.proxiedLogFormatter))

    def _xforwardedforTest(self, header):
        if False:
            print('Hello World!')
        '\n        Assert that a request with the given value in its I{X-Forwarded-For}\n        header is logged by L{proxiedLogFormatter} the same way it would have\n        been logged by L{combinedLogFormatter} but with 172.16.1.2 as the\n        client address instead of the normal value.\n\n        @param header: An I{X-Forwarded-For} header with left-most address of\n            172.16.1.2.\n        '
        reactor = Clock()
        reactor.advance(1234567890)
        timestamp = http.datetimeToLogString(reactor.seconds())
        request = DummyRequestForLogTest(http.HTTPFactory(reactor=reactor))
        expected = http.combinedLogFormatter(timestamp, request).replace('1.2.3.4', '172.16.1.2')
        request.requestHeaders.setRawHeaders(b'x-forwarded-for', [header])
        line = http.proxiedLogFormatter(timestamp, request)
        self.assertEqual(expected, line)

    def test_xforwardedfor(self):
        if False:
            return 10
        '\n        L{proxiedLogFormatter} logs the value of the I{X-Forwarded-For} header\n        in place of the client address field.\n        '
        self._xforwardedforTest(b'172.16.1.2, 10.0.0.3, 192.168.1.4')

    def test_extraForwardedSpaces(self):
        if False:
            print('Hello World!')
        '\n        Any extra spaces around the address in the I{X-Forwarded-For} header\n        are stripped and not included in the log string.\n        '
        self._xforwardedforTest(b' 172.16.1.2 , 10.0.0.3, 192.168.1.4')

class LogEscapingTests(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.logPath = self.mktemp()
        self.site = http.HTTPFactory(self.logPath)
        self.site.startFactory()
        self.request = DummyRequestForLogTest(self.site, False)

    def assertLogs(self, line):
        if False:
            print('Hello World!')
        "\n        Assert that if C{self.request} is logged using C{self.site} then\n        C{line} is written to the site's access log file.\n\n        @param line: The expected line.\n        @type line: L{bytes}\n\n        @raise self.failureException: If the log file contains something other\n            than the expected line.\n        "
        try:
            self.site.log(self.request)
        finally:
            self.site.stopFactory()
        logged = FilePath(self.logPath).getContent()
        self.assertEqual(line, logged)

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        '\n        A I{GET} request is logged with no extra escapes.\n        '
        self.site._logDateTime = '[%02d/%3s/%4d:%02d:%02d:%02d +0000]' % (25, 'Oct', 2004, 12, 31, 59)
        self.assertLogs(b'"1.2.3.4" - - [25/Oct/2004:12:31:59 +0000] "GET /dummy HTTP/1.0" 123 - "-" "-"\n')

    def test_methodQuote(self):
        if False:
            while True:
                i = 10
        '\n        If the HTTP request method includes a quote, the quote is escaped.\n        '
        self.site._logDateTime = '[%02d/%3s/%4d:%02d:%02d:%02d +0000]' % (25, 'Oct', 2004, 12, 31, 59)
        self.request.method = b'G"T'
        self.assertLogs(b'"1.2.3.4" - - [25/Oct/2004:12:31:59 +0000] "G\\"T /dummy HTTP/1.0" 123 - "-" "-"\n')

    def test_requestQuote(self):
        if False:
            while True:
                i = 10
        '\n        If the HTTP request path includes a quote, the quote is escaped.\n        '
        self.site._logDateTime = '[%02d/%3s/%4d:%02d:%02d:%02d +0000]' % (25, 'Oct', 2004, 12, 31, 59)
        self.request.uri = b'/dummy"withquote'
        self.assertLogs(b'"1.2.3.4" - - [25/Oct/2004:12:31:59 +0000] "GET /dummy\\"withquote HTTP/1.0" 123 - "-" "-"\n')

    def test_protoQuote(self):
        if False:
            i = 10
            return i + 15
        '\n        If the HTTP request version includes a quote, the quote is escaped.\n        '
        self.site._logDateTime = '[%02d/%3s/%4d:%02d:%02d:%02d +0000]' % (25, 'Oct', 2004, 12, 31, 59)
        self.request.clientproto = b'HT"P/1.0'
        self.assertLogs(b'"1.2.3.4" - - [25/Oct/2004:12:31:59 +0000] "GET /dummy HT\\"P/1.0" 123 - "-" "-"\n')

    def test_refererQuote(self):
        if False:
            i = 10
            return i + 15
        '\n        If the value of the I{Referer} header contains a quote, the quote is\n        escaped.\n        '
        self.site._logDateTime = '[%02d/%3s/%4d:%02d:%02d:%02d +0000]' % (25, 'Oct', 2004, 12, 31, 59)
        self.request.requestHeaders.addRawHeader(b'referer', b'http://malicious" ".website.invalid')
        self.assertLogs(b'"1.2.3.4" - - [25/Oct/2004:12:31:59 +0000] "GET /dummy HTTP/1.0" 123 - "http://malicious\\" \\".website.invalid" "-"\n')

    def test_userAgentQuote(self):
        if False:
            i = 10
            return i + 15
        '\n        If the value of the I{User-Agent} header contains a quote, the quote is\n        escaped.\n        '
        self.site._logDateTime = '[%02d/%3s/%4d:%02d:%02d:%02d +0000]' % (25, 'Oct', 2004, 12, 31, 59)
        self.request.requestHeaders.addRawHeader(b'user-agent', b'Malicious Web" Evil')
        self.assertLogs(b'"1.2.3.4" - - [25/Oct/2004:12:31:59 +0000] "GET /dummy HTTP/1.0" 123 - "-" "Malicious Web\\" Evil"\n')

class ServerAttributesTests(unittest.TestCase):
    """
    Tests that deprecated twisted.web.server attributes raise the appropriate
    deprecation warnings when used.
    """

    def test_deprecatedAttributeDateTimeString(self):
        if False:
            print('Hello World!')
        '\n        twisted.web.server.date_time_string should not be used; instead use\n        twisted.web.http.datetimeToString directly\n        '
        server.date_time_string
        warnings = self.flushWarnings(offendingFunctions=[self.test_deprecatedAttributeDateTimeString])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]['category'], DeprecationWarning)
        self.assertEqual(warnings[0]['message'], 'twisted.web.server.date_time_string was deprecated in Twisted 12.1.0: Please use twisted.web.http.datetimeToString instead')

    def test_deprecatedAttributeStringDateTime(self):
        if False:
            print('Hello World!')
        '\n        twisted.web.server.string_date_time should not be used; instead use\n        twisted.web.http.stringToDatetime directly\n        '
        server.string_date_time
        warnings = self.flushWarnings(offendingFunctions=[self.test_deprecatedAttributeStringDateTime])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]['category'], DeprecationWarning)
        self.assertEqual(warnings[0]['message'], 'twisted.web.server.string_date_time was deprecated in Twisted 12.1.0: Please use twisted.web.http.stringToDatetime instead')

class ExplicitHTTPFactoryReactor(unittest.TestCase):
    """
    L{http.HTTPFactory} accepts explicit reactor selection.
    """

    def test_explicitReactor(self):
        if False:
            print('Hello World!')
        '\n        L{http.HTTPFactory.__init__} accepts a reactor argument which is set on\n        L{http.HTTPFactory.reactor}.\n        '
        reactor = 'I am a reactor!'
        factory = http.HTTPFactory(reactor=reactor)
        self.assertIs(factory.reactor, reactor)

    def test_defaultReactor(self):
        if False:
            return 10
        '\n        Giving no reactor argument to L{http.HTTPFactory.__init__} means it\n        will select the global reactor.\n        '
        from twisted.internet import reactor
        factory = http.HTTPFactory()
        self.assertIs(factory.reactor, reactor)

class QueueResource(Resource):
    """
    Add all requests to an internal queue,
    without responding to the requests.
    You can access the requests from the queue and handle their response.
    """
    isLeaf = True

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()
        self.dispatchedRequests: List[Request] = []

    def render_GET(self, request: Request) -> int:
        if False:
            return 10
        self.dispatchedRequests.append(request)
        return NOT_DONE_YET

class TestRFC9112Section932(unittest.TestCase):
    """
    Verify that HTTP/1.1 request ordering is preserved.
    """

    def test_multipleRequestsInOneSegment(self) -> None:
        if False:
            print('Hello World!')
        '\n        Twisted MUST NOT respond to a second HTTP/1.1 request while the first\n        is still pending.\n        '
        qr = QueueResource()
        site = Site(qr)
        proto = site.buildProtocol(None)
        serverTransport = StringTransport()
        proto.makeConnection(serverTransport)
        proto.dataReceived(b'GET /first HTTP/1.1\r\nHost: a\r\n\r\nGET /second HTTP/1.1\r\nHost: a\r\n\r\n')
        self.assertEqual(len(qr.dispatchedRequests), 1)
        qr.dispatchedRequests[0].finish()
        self.assertEqual(len(qr.dispatchedRequests), 2)

    def test_multipleRequestsInDifferentSegments(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Twisted MUST NOT respond to a second HTTP/1.1 request while the first\n        is still pending, even if the second request is received in a separate\n        TCP package.\n        '
        qr = QueueResource()
        site = Site(qr)
        proto = site.buildProtocol(None)
        serverTransport = StringTransport()
        proto.makeConnection(serverTransport)
        raw_data = b'GET /first HTTP/1.1\r\nHost: a\r\n\r\nGET /second HTTP/1.1\r\nHost: a\r\n\r\n'
        for chunk in iterbytes(raw_data):
            proto.dataReceived(chunk)
        self.assertEqual(len(qr.dispatchedRequests), 1)
        qr.dispatchedRequests[0].finish()
        self.assertEqual(len(qr.dispatchedRequests), 2)