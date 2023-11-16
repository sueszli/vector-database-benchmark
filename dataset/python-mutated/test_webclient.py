"""
Tests L{twisted.web.client} helper APIs
"""
from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client

class URLJoinTests(unittest.TestCase):
    """
    Tests for L{client._urljoin}.
    """

    def test_noFragments(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{client._urljoin} does not include a fragment identifier in the\n        resulting URL if neither the base nor the new path include a fragment\n        identifier.\n        '
        self.assertEqual(client._urljoin(b'http://foo.com/bar', b'/quux'), b'http://foo.com/quux')
        self.assertEqual(client._urljoin(b'http://foo.com/bar#', b'/quux'), b'http://foo.com/quux')
        self.assertEqual(client._urljoin(b'http://foo.com/bar', b'/quux#'), b'http://foo.com/quux')

    def test_preserveFragments(self):
        if False:
            print('Hello World!')
        '\n        L{client._urljoin} preserves the fragment identifier from either the\n        new path or the base URL respectively, as specified in the HTTP 1.1 bis\n        draft.\n\n        @see: U{https://tools.ietf.org/html/draft-ietf-httpbis-p2-semantics-22#section-7.1.2}\n        '
        self.assertEqual(client._urljoin(b'http://foo.com/bar#frag', b'/quux'), b'http://foo.com/quux#frag')
        self.assertEqual(client._urljoin(b'http://foo.com/bar', b'/quux#frag2'), b'http://foo.com/quux#frag2')
        self.assertEqual(client._urljoin(b'http://foo.com/bar#frag', b'/quux#frag2'), b'http://foo.com/quux#frag2')

class URITests:
    """
    Abstract tests for L{twisted.web.client.URI}.

    Subclass this and L{unittest.TestCase}. Then provide a value for
    C{host} and C{uriHost}.

    @ivar host: A host specification for use in tests, must be L{bytes}.

    @ivar uriHost: The host specification in URI form, must be a L{bytes}. In
        most cases this is identical with C{host}. IPv6 address literals are an
        exception, according to RFC 3986 section 3.2.2, as they need to be
        enclosed in brackets. In this case this variable is different.
    """

    def makeURIString(self, template):
        if False:
            while True:
                i = 10
        '\n        Replace the string "HOST" in C{template} with this test\'s host.\n\n        Byte strings Python between (and including) versions 3.0 and 3.4\n        cannot be formatted using C{%} or C{format} so this does a simple\n        replace.\n\n        @type template: L{bytes}\n        @param template: A string containing "HOST".\n\n        @rtype: L{bytes}\n        @return: A string where "HOST" has been replaced by C{self.host}.\n        '
        self.assertIsInstance(self.host, bytes)
        self.assertIsInstance(self.uriHost, bytes)
        self.assertIsInstance(template, bytes)
        self.assertIn(b'HOST', template)
        return template.replace(b'HOST', self.uriHost)

    def assertURIEquals(self, uri, scheme, netloc, host, port, path, params=b'', query=b'', fragment=b''):
        if False:
            for i in range(10):
                print('nop')
        "\n        Assert that all of a L{client.URI}'s components match the expected\n        values.\n\n        @param uri: U{client.URI} instance whose attributes will be checked\n            for equality.\n\n        @type scheme: L{bytes}\n        @param scheme: URI scheme specifier.\n\n        @type netloc: L{bytes}\n        @param netloc: Network location component.\n\n        @type host: L{bytes}\n        @param host: Host name.\n\n        @type port: L{int}\n        @param port: Port number.\n\n        @type path: L{bytes}\n        @param path: Hierarchical path.\n\n        @type params: L{bytes}\n        @param params: Parameters for last path segment, defaults to C{b''}.\n\n        @type query: L{bytes}\n        @param query: Query string, defaults to C{b''}.\n\n        @type fragment: L{bytes}\n        @param fragment: Fragment identifier, defaults to C{b''}.\n        "
        self.assertEqual((scheme, netloc, host, port, path, params, query, fragment), (uri.scheme, uri.netloc, uri.host, uri.port, uri.path, uri.params, uri.query, uri.fragment))

    def test_parseDefaultPort(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{client.URI.fromBytes} by default assumes port 80 for the I{http}\n        scheme and 443 for the I{https} scheme.\n        '
        uri = client.URI.fromBytes(self.makeURIString(b'http://HOST'))
        self.assertEqual(80, uri.port)
        uri = client.URI.fromBytes(self.makeURIString(b'http://HOST:'))
        self.assertEqual(80, uri.port)
        uri = client.URI.fromBytes(self.makeURIString(b'https://HOST'))
        self.assertEqual(443, uri.port)

    def test_parseCustomDefaultPort(self):
        if False:
            while True:
                i = 10
        '\n        L{client.URI.fromBytes} accepts a C{defaultPort} parameter that\n        overrides the normal default port logic.\n        '
        uri = client.URI.fromBytes(self.makeURIString(b'http://HOST'), defaultPort=5144)
        self.assertEqual(5144, uri.port)
        uri = client.URI.fromBytes(self.makeURIString(b'https://HOST'), defaultPort=5144)
        self.assertEqual(5144, uri.port)

    def test_netlocHostPort(self):
        if False:
            print('Hello World!')
        '\n        Parsing a I{URI} splits the network location component into I{host} and\n        I{port}.\n        '
        uri = client.URI.fromBytes(self.makeURIString(b'http://HOST:5144'))
        self.assertEqual(5144, uri.port)
        self.assertEqual(self.host, uri.host)
        self.assertEqual(self.uriHost + b':5144', uri.netloc)
        uri = client.URI.fromBytes(self.makeURIString(b'http://HOST '))
        self.assertEqual(self.uriHost, uri.netloc)

    def test_path(self):
        if False:
            return 10
        '\n        Parse the path from a I{URI}.\n        '
        uri = self.makeURIString(b'http://HOST/foo/bar')
        parsed = client.URI.fromBytes(uri)
        self.assertURIEquals(parsed, scheme=b'http', netloc=self.uriHost, host=self.host, port=80, path=b'/foo/bar')
        self.assertEqual(uri, parsed.toBytes())

    def test_noPath(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The path of a I{URI} that has no path is the empty string.\n        '
        uri = self.makeURIString(b'http://HOST')
        parsed = client.URI.fromBytes(uri)
        self.assertURIEquals(parsed, scheme=b'http', netloc=self.uriHost, host=self.host, port=80, path=b'')
        self.assertEqual(uri, parsed.toBytes())

    def test_emptyPath(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The path of a I{URI} with an empty path is C{b'/'}.\n        "
        uri = self.makeURIString(b'http://HOST/')
        self.assertURIEquals(client.URI.fromBytes(uri), scheme=b'http', netloc=self.uriHost, host=self.host, port=80, path=b'/')

    def test_param(self):
        if False:
            while True:
                i = 10
        '\n        Parse I{URI} parameters from a I{URI}.\n        '
        uri = self.makeURIString(b'http://HOST/foo/bar;param')
        parsed = client.URI.fromBytes(uri)
        self.assertURIEquals(parsed, scheme=b'http', netloc=self.uriHost, host=self.host, port=80, path=b'/foo/bar', params=b'param')
        self.assertEqual(uri, parsed.toBytes())

    def test_query(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parse the query string from a I{URI}.\n        '
        uri = self.makeURIString(b'http://HOST/foo/bar;param?a=1&b=2')
        parsed = client.URI.fromBytes(uri)
        self.assertURIEquals(parsed, scheme=b'http', netloc=self.uriHost, host=self.host, port=80, path=b'/foo/bar', params=b'param', query=b'a=1&b=2')
        self.assertEqual(uri, parsed.toBytes())

    def test_fragment(self):
        if False:
            i = 10
            return i + 15
        '\n        Parse the fragment identifier from a I{URI}.\n        '
        uri = self.makeURIString(b'http://HOST/foo/bar;param?a=1&b=2#frag')
        parsed = client.URI.fromBytes(uri)
        self.assertURIEquals(parsed, scheme=b'http', netloc=self.uriHost, host=self.host, port=80, path=b'/foo/bar', params=b'param', query=b'a=1&b=2', fragment=b'frag')
        self.assertEqual(uri, parsed.toBytes())

    def test_originForm(self):
        if False:
            while True:
                i = 10
        '\n        L{client.URI.originForm} produces an absolute I{URI} path including\n        the I{URI} path.\n        '
        uri = client.URI.fromBytes(self.makeURIString(b'http://HOST/foo'))
        self.assertEqual(b'/foo', uri.originForm)

    def test_originFormComplex(self):
        if False:
            return 10
        '\n        L{client.URI.originForm} produces an absolute I{URI} path including\n        the I{URI} path, parameters and query string but excludes the fragment\n        identifier.\n        '
        uri = client.URI.fromBytes(self.makeURIString(b'http://HOST/foo;param?a=1#frag'))
        self.assertEqual(b'/foo;param?a=1', uri.originForm)

    def test_originFormNoPath(self):
        if False:
            i = 10
            return i + 15
        "\n        L{client.URI.originForm} produces a path of C{b'/'} when the I{URI}\n        specifies no path.\n        "
        uri = client.URI.fromBytes(self.makeURIString(b'http://HOST'))
        self.assertEqual(b'/', uri.originForm)

    def test_originFormEmptyPath(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        L{client.URI.originForm} produces a path of C{b'/'} when the I{URI}\n        specifies an empty path.\n        "
        uri = client.URI.fromBytes(self.makeURIString(b'http://HOST/'))
        self.assertEqual(b'/', uri.originForm)

    def test_externalUnicodeInterference(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{client.URI.fromBytes} parses the scheme, host, and path elements\n        into L{bytes}, even when passed an URL which has previously been passed\n        to L{urlparse} as a L{unicode} string.\n        '
        goodInput = self.makeURIString(b'http://HOST/path')
        badInput = goodInput.decode('ascii')
        urlparse(badInput)
        uri = client.URI.fromBytes(goodInput)
        self.assertIsInstance(uri.scheme, bytes)
        self.assertIsInstance(uri.host, bytes)
        self.assertIsInstance(uri.path, bytes)

class URITestsForHostname(URITests, unittest.TestCase):
    """
    Tests for L{twisted.web.client.URI} with host names.
    """
    uriHost = host = b'example.com'

class URITestsForIPv4(URITests, unittest.TestCase):
    """
    Tests for L{twisted.web.client.URI} with IPv4 host addresses.
    """
    uriHost = host = b'192.168.1.67'

class URITestsForIPv6(URITests, unittest.TestCase):
    """
    Tests for L{twisted.web.client.URI} with IPv6 host addresses.

    IPv6 addresses must always be surrounded by square braces in URIs. No
    attempt is made to test without.
    """
    host = b'fe80::20c:29ff:fea4:c60'
    uriHost = b'[fe80::20c:29ff:fea4:c60]'

    def test_hostBracketIPv6AddressLiteral(self):
        if False:
            print('Hello World!')
        '\n        Brackets around IPv6 addresses are stripped in the host field. The host\n        field is then exported with brackets in the output of\n        L{client.URI.toBytes}.\n        '
        uri = client.URI.fromBytes(b'http://[::1]:80/index.html')
        self.assertEqual(uri.host, b'::1')
        self.assertEqual(uri.netloc, b'[::1]:80')
        self.assertEqual(uri.toBytes(), b'http://[::1]:80/index.html')