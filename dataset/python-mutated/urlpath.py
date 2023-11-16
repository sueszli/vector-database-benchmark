"""
L{URLPath}, a representation of a URL.
"""
from typing import cast
from urllib.parse import quote as urlquote, unquote as urlunquote, urlunsplit
from hyperlink import URL as _URL
_allascii = b''.join([chr(x).encode('ascii') for x in range(1, 128)])

def _rereconstituter(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Attriute declaration to preserve mutability on L{URLPath}.\n\n    @param name: a public attribute name\n    @type name: native L{str}\n\n    @return: a descriptor which retrieves the private version of the attribute\n        on get and calls rerealize on set.\n    '
    privateName = '_' + name
    return property(lambda self: getattr(self, privateName), lambda self, value: setattr(self, privateName, value if isinstance(value, bytes) else value.encode('charmap')) or self._reconstitute())

class URLPath:
    """
    A representation of a URL.

    @ivar scheme: The scheme of the URL (e.g. 'http').
    @type scheme: L{bytes}

    @ivar netloc: The network location ("host").
    @type netloc: L{bytes}

    @ivar path: The path on the network location.
    @type path: L{bytes}

    @ivar query: The query argument (the portion after ?  in the URL).
    @type query: L{bytes}

    @ivar fragment: The page fragment (the portion after # in the URL).
    @type fragment: L{bytes}
    """

    def __init__(self, scheme=b'', netloc=b'localhost', path=b'', query=b'', fragment=b''):
        if False:
            return 10
        self._scheme = scheme or b'http'
        self._netloc = netloc
        self._path = path or b'/'
        self._query = query
        self._fragment = fragment
        self._reconstitute()

    def _reconstitute(self):
        if False:
            i = 10
            return i + 15
        '\n        Reconstitute this L{URLPath} from all its given attributes.\n        '
        urltext = urlquote(urlunsplit((self._scheme, self._netloc, self._path, self._query, self._fragment)), safe=_allascii)
        self._url = _URL.fromText(urltext.encode('ascii').decode('ascii'))
    scheme = _rereconstituter('scheme')
    netloc = _rereconstituter('netloc')
    path = _rereconstituter('path')
    query = _rereconstituter('query')
    fragment = _rereconstituter('fragment')

    @classmethod
    def _fromURL(cls, urlInstance):
        if False:
            i = 10
            return i + 15
        '\n        Reconstruct all the public instance variables of this L{URLPath} from\n        its underlying L{_URL}.\n\n        @param urlInstance: the object to base this L{URLPath} on.\n        @type urlInstance: L{_URL}\n\n        @return: a new L{URLPath}\n        '
        self = cls.__new__(cls)
        self._url = urlInstance.replace(path=urlInstance.path or [''])
        self._scheme = self._url.scheme.encode('ascii')
        self._netloc = self._url.authority().encode('ascii')
        self._path = _URL(path=self._url.path, rooted=True).asURI().asText().encode('ascii')
        self._query = _URL(query=self._url.query).asURI().asText().encode('ascii')[1:]
        self._fragment = self._url.fragment.encode('ascii')
        return self

    def pathList(self, unquote=False, copy=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        Split this URL's path into its components.\n\n        @param unquote: whether to remove %-encoding from the returned strings.\n\n        @param copy: (ignored, do not use)\n\n        @return: The components of C{self.path}\n        @rtype: L{list} of L{bytes}\n        "
        segments = self._url.path
        mapper = lambda x: x.encode('ascii')
        if unquote:
            mapper = lambda x, m=mapper: m(urlunquote(x))
        return [b''] + [mapper(segment) for segment in segments]

    @classmethod
    def fromString(klass, url):
        if False:
            i = 10
            return i + 15
        '\n        Make a L{URLPath} from a L{str} or L{unicode}.\n\n        @param url: A L{str} representation of a URL.\n        @type url: L{str} or L{unicode}.\n\n        @return: a new L{URLPath} derived from the given string.\n        @rtype: L{URLPath}\n        '
        if not isinstance(url, str):
            raise ValueError("'url' must be a str")
        return klass._fromURL(_URL.fromText(url))

    @classmethod
    def fromBytes(klass, url):
        if False:
            i = 10
            return i + 15
        '\n        Make a L{URLPath} from a L{bytes}.\n\n        @param url: A L{bytes} representation of a URL.\n        @type url: L{bytes}\n\n        @return: a new L{URLPath} derived from the given L{bytes}.\n        @rtype: L{URLPath}\n\n        @since: 15.4\n        '
        if not isinstance(url, bytes):
            raise ValueError("'url' must be bytes")
        quoted = urlquote(url, safe=_allascii)
        return klass.fromString(quoted)

    @classmethod
    def fromRequest(klass, request):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make a L{URLPath} from a L{twisted.web.http.Request}.\n\n        @param request: A L{twisted.web.http.Request} to make the L{URLPath}\n            from.\n\n        @return: a new L{URLPath} derived from the given request.\n        @rtype: L{URLPath}\n        '
        return klass.fromBytes(request.prePathURL())

    def _mod(self, newURL, keepQuery):
        if False:
            while True:
                i = 10
        '\n        Return a modified copy of C{self} using C{newURL}, keeping the query\n        string if C{keepQuery} is C{True}.\n\n        @param newURL: a L{URL} to derive a new L{URLPath} from\n        @type newURL: L{URL}\n\n        @param keepQuery: if C{True}, preserve the query parameters from\n            C{self} on the new L{URLPath}; if C{False}, give the new L{URLPath}\n            no query parameters.\n        @type keepQuery: L{bool}\n\n        @return: a new L{URLPath}\n        '
        return self._fromURL(newURL.replace(fragment='', query=self._url.query if keepQuery else ()))

    def sibling(self, path, keepQuery=False):
        if False:
            print('Hello World!')
        '\n        Get the sibling of the current L{URLPath}.  A sibling is a file which\n        is in the same directory as the current file.\n\n        @param path: The path of the sibling.\n        @type path: L{bytes}\n\n        @param keepQuery: Whether to keep the query parameters on the returned\n            L{URLPath}.\n        @type keepQuery: L{bool}\n\n        @return: a new L{URLPath}\n        '
        return self._mod(self._url.sibling(path.decode('ascii')), keepQuery)

    def child(self, path, keepQuery=False):
        if False:
            print('Hello World!')
        '\n        Get the child of this L{URLPath}.\n\n        @param path: The path of the child.\n        @type path: L{bytes}\n\n        @param keepQuery: Whether to keep the query parameters on the returned\n            L{URLPath}.\n        @type keepQuery: L{bool}\n\n        @return: a new L{URLPath}\n        '
        return self._mod(self._url.child(path.decode('ascii')), keepQuery)

    def parent(self, keepQuery=False):
        if False:
            return 10
        '\n        Get the parent directory of this L{URLPath}.\n\n        @param keepQuery: Whether to keep the query parameters on the returned\n            L{URLPath}.\n        @type keepQuery: L{bool}\n\n        @return: a new L{URLPath}\n        '
        return self._mod(self._url.click('..'), keepQuery)

    def here(self, keepQuery=False):
        if False:
            i = 10
            return i + 15
        '\n        Get the current directory of this L{URLPath}.\n\n        @param keepQuery: Whether to keep the query parameters on the returned\n            L{URLPath}.\n        @type keepQuery: L{bool}\n\n        @return: a new L{URLPath}\n        '
        return self._mod(self._url.click('.'), keepQuery)

    def click(self, st):
        if False:
            i = 10
            return i + 15
        '\n        Return a path which is the URL where a browser would presumably take\n        you if you clicked on a link with an HREF as given.\n\n        @param st: A relative URL, to be interpreted relative to C{self} as the\n            base URL.\n        @type st: L{bytes}\n\n        @return: a new L{URLPath}\n        '
        return self._fromURL(self._url.click(st.decode('ascii')))

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        '\n        The L{str} of a L{URLPath} is its URL text.\n        '
        return cast(str, self._url.asURI().asText())

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        The L{repr} of a L{URLPath} is an eval-able expression which will\n        construct a similar L{URLPath}.\n        '
        return 'URLPath(scheme={!r}, netloc={!r}, path={!r}, query={!r}, fragment={!r})'.format(self.scheme, self.netloc, self.path, self.query, self.fragment)