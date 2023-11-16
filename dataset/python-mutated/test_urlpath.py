"""
Tests for L{twisted.python.urlpath}.
"""
from twisted.python import urlpath
from twisted.trial import unittest

class _BaseURLPathTests:
    """
    Tests for instantiated L{urlpath.URLPath}s.
    """

    def test_partsAreBytes(self):
        if False:
            while True:
                i = 10
        '\n        All of the attributes of L{urlpath.URLPath} should be L{bytes}.\n        '
        self.assertIsInstance(self.path.scheme, bytes)
        self.assertIsInstance(self.path.netloc, bytes)
        self.assertIsInstance(self.path.path, bytes)
        self.assertIsInstance(self.path.query, bytes)
        self.assertIsInstance(self.path.fragment, bytes)

    def test_strReturnsStr(self):
        if False:
            print('Hello World!')
        '\n        Calling C{str()} with a L{URLPath} will always return a L{str}.\n        '
        self.assertEqual(type(self.path.__str__()), str)

    def test_mutabilityWithText(self, stringType=str):
        if False:
            return 10
        '\n        Setting attributes on L{urlpath.URLPath} should change the value\n        returned by L{str}.\n\n        @param stringType: a callable to parameterize this test for different\n            text types.\n        @type stringType: 1-argument callable taking L{str} and returning\n            L{str} or L{bytes}.\n        '
        self.path.scheme = stringType('https')
        self.assertEqual(str(self.path), 'https://example.com/foo/bar?yes=no&no=yes#footer')
        self.path.netloc = stringType('another.example.invalid')
        self.assertEqual(str(self.path), 'https://another.example.invalid/foo/bar?yes=no&no=yes#footer')
        self.path.path = stringType('/hello')
        self.assertEqual(str(self.path), 'https://another.example.invalid/hello?yes=no&no=yes#footer')
        self.path.query = stringType('alpha=omega&opposites=same')
        self.assertEqual(str(self.path), 'https://another.example.invalid/hello?alpha=omega&opposites=same#footer')
        self.path.fragment = stringType('header')
        self.assertEqual(str(self.path), 'https://another.example.invalid/hello?alpha=omega&opposites=same#header')

    def test_mutabilityWithBytes(self):
        if False:
            while True:
                i = 10
        '\n        Same as L{test_mutabilityWithText} but for bytes.\n        '
        self.test_mutabilityWithText(lambda x: x.encode('ascii'))

    def test_allAttributesAreBytes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A created L{URLPath} has bytes attributes.\n        '
        self.assertIsInstance(self.path.scheme, bytes)
        self.assertIsInstance(self.path.netloc, bytes)
        self.assertIsInstance(self.path.path, bytes)
        self.assertIsInstance(self.path.query, bytes)
        self.assertIsInstance(self.path.fragment, bytes)

    def test_stringConversion(self):
        if False:
            return 10
        '\n        Calling C{str()} with a L{URLPath} will return the same URL that it was\n        constructed with.\n        '
        self.assertEqual(str(self.path), 'http://example.com/foo/bar?yes=no&no=yes#footer')

    def test_childString(self):
        if False:
            return 10
        '\n        Calling C{str()} with a C{URLPath.child()} will return a URL which is\n        the child of the URL it was instantiated with.\n        '
        self.assertEqual(str(self.path.child(b'hello')), 'http://example.com/foo/bar/hello')
        self.assertEqual(str(self.path.child(b'hello').child(b'')), 'http://example.com/foo/bar/hello/')
        self.assertEqual(str(self.path.child(b'hello', keepQuery=True)), 'http://example.com/foo/bar/hello?yes=no&no=yes')

    def test_siblingString(self):
        if False:
            print('Hello World!')
        '\n        Calling C{str()} with a C{URLPath.sibling()} will return a URL which is\n        the sibling of the URL it was instantiated with.\n        '
        self.assertEqual(str(self.path.sibling(b'baz')), 'http://example.com/foo/baz')
        self.assertEqual(str(self.path.sibling(b'baz', keepQuery=True)), 'http://example.com/foo/baz?yes=no&no=yes')
        self.assertEqual(str(self.path.child(b'').sibling(b'baz')), 'http://example.com/foo/bar/baz')

    def test_parentString(self):
        if False:
            print('Hello World!')
        '\n        Calling C{str()} with a C{URLPath.parent()} will return a URL which is\n        the parent of the URL it was instantiated with.\n        '
        self.assertEqual(str(self.path.parent()), 'http://example.com/')
        self.assertEqual(str(self.path.parent(keepQuery=True)), 'http://example.com/?yes=no&no=yes')
        self.assertEqual(str(self.path.child(b'').parent()), 'http://example.com/foo/')
        self.assertEqual(str(self.path.child(b'baz').parent()), 'http://example.com/foo/')
        self.assertEqual(str(self.path.parent().parent().parent().parent().parent()), 'http://example.com/')

    def test_hereString(self):
        if False:
            while True:
                i = 10
        '\n        Calling C{str()} with a C{URLPath.here()} will return a URL which is\n        the URL that it was instantiated with, without any file, query, or\n        fragment.\n        '
        self.assertEqual(str(self.path.here()), 'http://example.com/foo/')
        self.assertEqual(str(self.path.here(keepQuery=True)), 'http://example.com/foo/?yes=no&no=yes')
        self.assertEqual(str(self.path.child(b'').here()), 'http://example.com/foo/bar/')

    def test_doubleSlash(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calling L{urlpath.URLPath.click} on a L{urlpath.URLPath} with a\n        trailing slash with a relative URL containing a leading slash will\n        result in a URL with a single slash at the start of the path portion.\n        '
        self.assertEqual(str(self.path.click(b'/hello/world')).encode('ascii'), b'http://example.com/hello/world')

    def test_pathList(self):
        if False:
            i = 10
            return i + 15
        '\n        L{urlpath.URLPath.pathList} returns a L{list} of L{bytes}.\n        '
        self.assertEqual(self.path.child(b'%00%01%02').pathList(), [b'', b'foo', b'bar', b'%00%01%02'])
        self.assertEqual(self.path.child(b'%00%01%02').pathList(copy=False), [b'', b'foo', b'bar', b'%00%01%02'])
        self.assertEqual(self.path.child(b'%00%01%02').pathList(unquote=True), [b'', b'foo', b'bar', b'\x00\x01\x02'])

class BytesURLPathTests(_BaseURLPathTests, unittest.TestCase):
    """
    Tests for interacting with a L{URLPath} created with C{fromBytes}.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.path = urlpath.URLPath.fromBytes(b'http://example.com/foo/bar?yes=no&no=yes#footer')

    def test_mustBeBytes(self):
        if False:
            return 10
        '\n        L{URLPath.fromBytes} must take a L{bytes} argument.\n        '
        with self.assertRaises(ValueError):
            urlpath.URLPath.fromBytes(None)
        with self.assertRaises(ValueError):
            urlpath.URLPath.fromBytes('someurl')

    def test_withoutArguments(self):
        if False:
            i = 10
            return i + 15
        '\n        An instantiation with no arguments creates a usable L{URLPath} with\n        default arguments.\n        '
        url = urlpath.URLPath()
        self.assertEqual(str(url), 'http://localhost/')

    def test_partialArguments(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Leaving some optional arguments unfilled makes a L{URLPath} with those\n        optional arguments filled with defaults.\n        '
        url = urlpath.URLPath.fromBytes(b'http://google.com')
        self.assertEqual(url.scheme, b'http')
        self.assertEqual(url.netloc, b'google.com')
        self.assertEqual(url.path, b'/')
        self.assertEqual(url.fragment, b'')
        self.assertEqual(url.query, b'')
        self.assertEqual(str(url), 'http://google.com/')

    def test_nonASCIIBytes(self):
        if False:
            print('Hello World!')
        '\n        L{URLPath.fromBytes} can interpret non-ASCII bytes as percent-encoded\n        '
        url = urlpath.URLPath.fromBytes(b'http://example.com/\xff\x00')
        self.assertEqual(str(url), 'http://example.com/%FF%00')

class StringURLPathTests(_BaseURLPathTests, unittest.TestCase):
    """
    Tests for interacting with a L{URLPath} created with C{fromString} and a
    L{str} argument.
    """

    def setUp(self):
        if False:
            return 10
        self.path = urlpath.URLPath.fromString('http://example.com/foo/bar?yes=no&no=yes#footer')

    def test_mustBeStr(self):
        if False:
            print('Hello World!')
        '\n        C{URLPath.fromString} must take a L{str} or L{str} argument.\n        '
        with self.assertRaises(ValueError):
            urlpath.URLPath.fromString(None)
        with self.assertRaises(ValueError):
            urlpath.URLPath.fromString(b'someurl')

class UnicodeURLPathTests(_BaseURLPathTests, unittest.TestCase):
    """
    Tests for interacting with a L{URLPath} created with C{fromString} and a
    L{str} argument.
    """

    def setUp(self):
        if False:
            return 10
        self.path = urlpath.URLPath.fromString('http://example.com/foo/bar?yes=no&no=yes#footer')

    def test_nonASCIICharacters(self):
        if False:
            i = 10
            return i + 15
        '\n        L{URLPath.fromString} can load non-ASCII characters.\n        '
        url = urlpath.URLPath.fromString('http://example.com/ÿ\x00')
        self.assertEqual(str(url), 'http://example.com/%C3%BF%00')