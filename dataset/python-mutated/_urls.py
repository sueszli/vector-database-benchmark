import typing
from urllib.parse import parse_qs, unquote
import idna
from ._types import QueryParamTypes, RawURL, URLTypes
from ._urlparse import urlencode, urlparse
from ._utils import primitive_value_to_str

class URL:
    """
    url = httpx.URL("HTTPS://jo%40email.com:a%20secret@müller.de:1234/pa%20th?search=ab#anchorlink")

    assert url.scheme == "https"
    assert url.username == "jo@email.com"
    assert url.password == "a secret"
    assert url.userinfo == b"jo%40email.com:a%20secret"
    assert url.host == "müller.de"
    assert url.raw_host == b"xn--mller-kva.de"
    assert url.port == 1234
    assert url.netloc == b"xn--mller-kva.de:1234"
    assert url.path == "/pa th"
    assert url.query == b"?search=ab"
    assert url.raw_path == b"/pa%20th?search=ab"
    assert url.fragment == "anchorlink"

    The components of a URL are broken down like this:

       https://jo%40email.com:a%20secret@müller.de:1234/pa%20th?search=ab#anchorlink
    [scheme]   [  username  ] [password] [ host ][port][ path ] [ query ] [fragment]
               [       userinfo        ] [   netloc   ][    raw_path    ]

    Note that:

    * `url.scheme` is normalized to always be lowercased.

    * `url.host` is normalized to always be lowercased. Internationalized domain
      names are represented in unicode, without IDNA encoding applied. For instance:

      url = httpx.URL("http://中国.icom.museum")
      assert url.host == "中国.icom.museum"
      url = httpx.URL("http://xn--fiqs8s.icom.museum")
      assert url.host == "中国.icom.museum"

    * `url.raw_host` is normalized to always be lowercased, and is IDNA encoded.

      url = httpx.URL("http://中国.icom.museum")
      assert url.raw_host == b"xn--fiqs8s.icom.museum"
      url = httpx.URL("http://xn--fiqs8s.icom.museum")
      assert url.raw_host == b"xn--fiqs8s.icom.museum"

    * `url.port` is either None or an integer. URLs that include the default port for
      "http", "https", "ws", "wss", and "ftp" schemes have their port normalized to `None`.

      assert httpx.URL("http://example.com") == httpx.URL("http://example.com:80")
      assert httpx.URL("http://example.com").port is None
      assert httpx.URL("http://example.com:80").port is None

    * `url.userinfo` is raw bytes, without URL escaping. Usually you'll want to work with
      `url.username` and `url.password` instead, which handle the URL escaping.

    * `url.raw_path` is raw bytes of both the path and query, without URL escaping.
      This portion is used as the target when constructing HTTP requests. Usually you'll
      want to work with `url.path` instead.

    * `url.query` is raw bytes, without URL escaping. A URL query string portion can only
      be properly URL escaped when decoding the parameter names and values themselves.
    """

    def __init__(self, url: typing.Union['URL', str]='', **kwargs: typing.Any) -> None:
        if False:
            print('Hello World!')
        if kwargs:
            allowed = {'scheme': str, 'username': str, 'password': str, 'userinfo': bytes, 'host': str, 'port': int, 'netloc': bytes, 'path': str, 'query': bytes, 'raw_path': bytes, 'fragment': str, 'params': object}
            for (key, value) in kwargs.items():
                if key not in allowed:
                    message = f'{key!r} is an invalid keyword argument for URL()'
                    raise TypeError(message)
                if value is not None and (not isinstance(value, allowed[key])):
                    expected = allowed[key].__name__
                    seen = type(value).__name__
                    message = f'Argument {key!r} must be {expected} but got {seen}'
                    raise TypeError(message)
                if isinstance(value, bytes):
                    kwargs[key] = value.decode('ascii')
            if 'params' in kwargs:
                params = kwargs.pop('params')
                kwargs['query'] = None if not params else str(QueryParams(params))
        if isinstance(url, str):
            self._uri_reference = urlparse(url, **kwargs)
        elif isinstance(url, URL):
            self._uri_reference = url._uri_reference.copy_with(**kwargs)
        else:
            raise TypeError(f'Invalid type for url.  Expected str or httpx.URL, got {type(url)}: {url!r}')

    @property
    def scheme(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        The URL scheme, such as "http", "https".\n        Always normalised to lowercase.\n        '
        return self._uri_reference.scheme

    @property
    def raw_scheme(self) -> bytes:
        if False:
            return 10
        '\n        The raw bytes representation of the URL scheme, such as b"http", b"https".\n        Always normalised to lowercase.\n        '
        return self._uri_reference.scheme.encode('ascii')

    @property
    def userinfo(self) -> bytes:
        if False:
            i = 10
            return i + 15
        '\n        The URL userinfo as a raw bytestring.\n        For example: b"jo%40email.com:a%20secret".\n        '
        return self._uri_reference.userinfo.encode('ascii')

    @property
    def username(self) -> str:
        if False:
            print('Hello World!')
        '\n        The URL username as a string, with URL decoding applied.\n        For example: "jo@email.com"\n        '
        userinfo = self._uri_reference.userinfo
        return unquote(userinfo.partition(':')[0])

    @property
    def password(self) -> str:
        if False:
            print('Hello World!')
        '\n        The URL password as a string, with URL decoding applied.\n        For example: "a secret"\n        '
        userinfo = self._uri_reference.userinfo
        return unquote(userinfo.partition(':')[2])

    @property
    def host(self) -> str:
        if False:
            return 10
        '\n        The URL host as a string.\n        Always normalized to lowercase, with IDNA hosts decoded into unicode.\n\n        Examples:\n\n        url = httpx.URL("http://www.EXAMPLE.org")\n        assert url.host == "www.example.org"\n\n        url = httpx.URL("http://中国.icom.museum")\n        assert url.host == "中国.icom.museum"\n\n        url = httpx.URL("http://xn--fiqs8s.icom.museum")\n        assert url.host == "中国.icom.museum"\n\n        url = httpx.URL("https://[::ffff:192.168.0.1]")\n        assert url.host == "::ffff:192.168.0.1"\n        '
        host: str = self._uri_reference.host
        if host.startswith('xn--'):
            host = idna.decode(host)
        return host

    @property
    def raw_host(self) -> bytes:
        if False:
            print('Hello World!')
        '\n        The raw bytes representation of the URL host.\n        Always normalized to lowercase, and IDNA encoded.\n\n        Examples:\n\n        url = httpx.URL("http://www.EXAMPLE.org")\n        assert url.raw_host == b"www.example.org"\n\n        url = httpx.URL("http://中国.icom.museum")\n        assert url.raw_host == b"xn--fiqs8s.icom.museum"\n\n        url = httpx.URL("http://xn--fiqs8s.icom.museum")\n        assert url.raw_host == b"xn--fiqs8s.icom.museum"\n\n        url = httpx.URL("https://[::ffff:192.168.0.1]")\n        assert url.raw_host == b"::ffff:192.168.0.1"\n        '
        return self._uri_reference.host.encode('ascii')

    @property
    def port(self) -> typing.Optional[int]:
        if False:
            while True:
                i = 10
        '\n        The URL port as an integer.\n\n        Note that the URL class performs port normalization as per the WHATWG spec.\n        Default ports for "http", "https", "ws", "wss", and "ftp" schemes are always\n        treated as `None`.\n\n        For example:\n\n        assert httpx.URL("http://www.example.com") == httpx.URL("http://www.example.com:80")\n        assert httpx.URL("http://www.example.com:80").port is None\n        '
        return self._uri_reference.port

    @property
    def netloc(self) -> bytes:
        if False:
            return 10
        '\n        Either `<host>` or `<host>:<port>` as bytes.\n        Always normalized to lowercase, and IDNA encoded.\n\n        This property may be used for generating the value of a request\n        "Host" header.\n        '
        return self._uri_reference.netloc.encode('ascii')

    @property
    def path(self) -> str:
        if False:
            return 10
        '\n        The URL path as a string. Excluding the query string, and URL decoded.\n\n        For example:\n\n        url = httpx.URL("https://example.com/pa%20th")\n        assert url.path == "/pa th"\n        '
        path = self._uri_reference.path or '/'
        return unquote(path)

    @property
    def query(self) -> bytes:
        if False:
            return 10
        '\n        The URL query string, as raw bytes, excluding the leading b"?".\n\n        This is necessarily a bytewise interface, because we cannot\n        perform URL decoding of this representation until we\'ve parsed\n        the keys and values into a QueryParams instance.\n\n        For example:\n\n        url = httpx.URL("https://example.com/?filter=some%20search%20terms")\n        assert url.query == b"filter=some%20search%20terms"\n        '
        query = self._uri_reference.query or ''
        return query.encode('ascii')

    @property
    def params(self) -> 'QueryParams':
        if False:
            print('Hello World!')
        '\n        The URL query parameters, neatly parsed and packaged into an immutable\n        multidict representation.\n        '
        return QueryParams(self._uri_reference.query)

    @property
    def raw_path(self) -> bytes:
        if False:
            return 10
        '\n        The complete URL path and query string as raw bytes.\n        Used as the target when constructing HTTP requests.\n\n        For example:\n\n        GET /users?search=some%20text HTTP/1.1\n        Host: www.example.org\n        Connection: close\n        '
        path = self._uri_reference.path or '/'
        if self._uri_reference.query is not None:
            path += '?' + self._uri_reference.query
        return path.encode('ascii')

    @property
    def fragment(self) -> str:
        if False:
            while True:
                i = 10
        "\n        The URL fragments, as used in HTML anchors.\n        As a string, without the leading '#'.\n        "
        return unquote(self._uri_reference.fragment or '')

    @property
    def raw(self) -> RawURL:
        if False:
            print('Hello World!')
        '\n        Provides the (scheme, host, port, target) for the outgoing request.\n\n        In older versions of `httpx` this was used in the low-level transport API.\n        We no longer use `RawURL`, and this property will be deprecated in a future release.\n        '
        return RawURL(self.raw_scheme, self.raw_host, self.port, self.raw_path)

    @property
    def is_absolute_url(self) -> bool:
        if False:
            print('Hello World!')
        "\n        Return `True` for absolute URLs such as 'http://example.com/path',\n        and `False` for relative URLs such as '/path'.\n        "
        return bool(self._uri_reference.scheme and self._uri_reference.host)

    @property
    def is_relative_url(self) -> bool:
        if False:
            print('Hello World!')
        "\n        Return `False` for absolute URLs such as 'http://example.com/path',\n        and `True` for relative URLs such as '/path'.\n        "
        return not self.is_absolute_url

    def copy_with(self, **kwargs: typing.Any) -> 'URL':
        if False:
            print('Hello World!')
        '\n        Copy this URL, returning a new URL with some components altered.\n        Accepts the same set of parameters as the components that are made\n        available via properties on the `URL` class.\n\n        For example:\n\n        url = httpx.URL("https://www.example.com").copy_with(username="jo@gmail.com", password="a secret")\n        assert url == "https://jo%40email.com:a%20secret@www.example.com"\n        '
        return URL(self, **kwargs)

    def copy_set_param(self, key: str, value: typing.Any=None) -> 'URL':
        if False:
            i = 10
            return i + 15
        return self.copy_with(params=self.params.set(key, value))

    def copy_add_param(self, key: str, value: typing.Any=None) -> 'URL':
        if False:
            for i in range(10):
                print('nop')
        return self.copy_with(params=self.params.add(key, value))

    def copy_remove_param(self, key: str) -> 'URL':
        if False:
            print('Hello World!')
        return self.copy_with(params=self.params.remove(key))

    def copy_merge_params(self, params: QueryParamTypes) -> 'URL':
        if False:
            i = 10
            return i + 15
        return self.copy_with(params=self.params.merge(params))

    def join(self, url: URLTypes) -> 'URL':
        if False:
            for i in range(10):
                print('nop')
        '\n        Return an absolute URL, using this URL as the base.\n\n        Eg.\n\n        url = httpx.URL("https://www.example.com/test")\n        url = url.join("/new/path")\n        assert url == "https://www.example.com/new/path"\n        '
        from urllib.parse import urljoin
        return URL(urljoin(str(self), str(URL(url))))

    def __hash__(self) -> int:
        if False:
            print('Hello World!')
        return hash(str(self))

    def __eq__(self, other: typing.Any) -> bool:
        if False:
            return 10
        return isinstance(other, (URL, str)) and str(self) == str(URL(other))

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return str(self._uri_reference)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        (scheme, userinfo, host, port, path, query, fragment) = self._uri_reference
        if ':' in userinfo:
            userinfo = f"{userinfo.split(':')[0]}:[secure]"
        authority = ''.join([f'{userinfo}@' if userinfo else '', f'[{host}]' if ':' in host else host, f':{port}' if port is not None else ''])
        url = ''.join([f'{self.scheme}:' if scheme else '', f'//{authority}' if authority else '', path, f'?{query}' if query is not None else '', f'#{fragment}' if fragment is not None else ''])
        return f'{self.__class__.__name__}({url!r})'

class QueryParams(typing.Mapping[str, str]):
    """
    URL query parameters, as a multi-dict.
    """

    def __init__(self, *args: typing.Optional[QueryParamTypes], **kwargs: typing.Any) -> None:
        if False:
            i = 10
            return i + 15
        assert len(args) < 2, 'Too many arguments.'
        assert not (args and kwargs), 'Cannot mix named and unnamed arguments.'
        value = args[0] if args else kwargs
        if value is None or isinstance(value, (str, bytes)):
            value = value.decode('ascii') if isinstance(value, bytes) else value
            self._dict = parse_qs(value, keep_blank_values=True)
        elif isinstance(value, QueryParams):
            self._dict = {k: list(v) for (k, v) in value._dict.items()}
        else:
            dict_value: typing.Dict[typing.Any, typing.List[typing.Any]] = {}
            if isinstance(value, (list, tuple)):
                for item in value:
                    dict_value.setdefault(item[0], []).append(item[1])
            else:
                dict_value = {k: list(v) if isinstance(v, (list, tuple)) else [v] for (k, v) in value.items()}
            self._dict = {str(k): [primitive_value_to_str(item) for item in v] for (k, v) in dict_value.items()}

    def keys(self) -> typing.KeysView[str]:
        if False:
            print('Hello World!')
        '\n        Return all the keys in the query params.\n\n        Usage:\n\n        q = httpx.QueryParams("a=123&a=456&b=789")\n        assert list(q.keys()) == ["a", "b"]\n        '
        return self._dict.keys()

    def values(self) -> typing.ValuesView[str]:
        if False:
            while True:
                i = 10
        '\n        Return all the values in the query params. If a key occurs more than once\n        only the first item for that key is returned.\n\n        Usage:\n\n        q = httpx.QueryParams("a=123&a=456&b=789")\n        assert list(q.values()) == ["123", "789"]\n        '
        return {k: v[0] for (k, v) in self._dict.items()}.values()

    def items(self) -> typing.ItemsView[str, str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return all items in the query params. If a key occurs more than once\n        only the first item for that key is returned.\n\n        Usage:\n\n        q = httpx.QueryParams("a=123&a=456&b=789")\n        assert list(q.items()) == [("a", "123"), ("b", "789")]\n        '
        return {k: v[0] for (k, v) in self._dict.items()}.items()

    def multi_items(self) -> typing.List[typing.Tuple[str, str]]:
        if False:
            print('Hello World!')
        '\n        Return all items in the query params. Allow duplicate keys to occur.\n\n        Usage:\n\n        q = httpx.QueryParams("a=123&a=456&b=789")\n        assert list(q.multi_items()) == [("a", "123"), ("a", "456"), ("b", "789")]\n        '
        multi_items: typing.List[typing.Tuple[str, str]] = []
        for (k, v) in self._dict.items():
            multi_items.extend([(k, i) for i in v])
        return multi_items

    def get(self, key: typing.Any, default: typing.Any=None) -> typing.Any:
        if False:
            while True:
                i = 10
        '\n        Get a value from the query param for a given key. If the key occurs\n        more than once, then only the first value is returned.\n\n        Usage:\n\n        q = httpx.QueryParams("a=123&a=456&b=789")\n        assert q.get("a") == "123"\n        '
        if key in self._dict:
            return self._dict[str(key)][0]
        return default

    def get_list(self, key: str) -> typing.List[str]:
        if False:
            return 10
        '\n        Get all values from the query param for a given key.\n\n        Usage:\n\n        q = httpx.QueryParams("a=123&a=456&b=789")\n        assert q.get_list("a") == ["123", "456"]\n        '
        return list(self._dict.get(str(key), []))

    def set(self, key: str, value: typing.Any=None) -> 'QueryParams':
        if False:
            print('Hello World!')
        '\n        Return a new QueryParams instance, setting the value of a key.\n\n        Usage:\n\n        q = httpx.QueryParams("a=123")\n        q = q.set("a", "456")\n        assert q == httpx.QueryParams("a=456")\n        '
        q = QueryParams()
        q._dict = dict(self._dict)
        q._dict[str(key)] = [primitive_value_to_str(value)]
        return q

    def add(self, key: str, value: typing.Any=None) -> 'QueryParams':
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a new QueryParams instance, setting or appending the value of a key.\n\n        Usage:\n\n        q = httpx.QueryParams("a=123")\n        q = q.add("a", "456")\n        assert q == httpx.QueryParams("a=123&a=456")\n        '
        q = QueryParams()
        q._dict = dict(self._dict)
        q._dict[str(key)] = q.get_list(key) + [primitive_value_to_str(value)]
        return q

    def remove(self, key: str) -> 'QueryParams':
        if False:
            while True:
                i = 10
        '\n        Return a new QueryParams instance, removing the value of a key.\n\n        Usage:\n\n        q = httpx.QueryParams("a=123")\n        q = q.remove("a")\n        assert q == httpx.QueryParams("")\n        '
        q = QueryParams()
        q._dict = dict(self._dict)
        q._dict.pop(str(key), None)
        return q

    def merge(self, params: typing.Optional[QueryParamTypes]=None) -> 'QueryParams':
        if False:
            i = 10
            return i + 15
        '\n        Return a new QueryParams instance, updated with.\n\n        Usage:\n\n        q = httpx.QueryParams("a=123")\n        q = q.merge({"b": "456"})\n        assert q == httpx.QueryParams("a=123&b=456")\n\n        q = httpx.QueryParams("a=123")\n        q = q.merge({"a": "456", "b": "789"})\n        assert q == httpx.QueryParams("a=456&b=789")\n        '
        q = QueryParams(params)
        q._dict = {**self._dict, **q._dict}
        return q

    def __getitem__(self, key: typing.Any) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._dict[key][0]

    def __contains__(self, key: typing.Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return key in self._dict

    def __iter__(self) -> typing.Iterator[typing.Any]:
        if False:
            print('Hello World!')
        return iter(self.keys())

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len(self._dict)

    def __bool__(self) -> bool:
        if False:
            return 10
        return bool(self._dict)

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash(str(self))

    def __eq__(self, other: typing.Any) -> bool:
        if False:
            i = 10
            return i + 15
        if not isinstance(other, self.__class__):
            return False
        return sorted(self.multi_items()) == sorted(other.multi_items())

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        "\n        Note that we use '%20' encoding for spaces, and treat '/' as a safe\n        character.\n\n        See https://github.com/encode/httpx/issues/2536 and\n        https://docs.python.org/3/library/urllib.parse.html#urllib.parse.urlencode\n        "
        return urlencode(self.multi_items())

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        class_name = self.__class__.__name__
        query_string = str(self)
        return f'{class_name}({query_string!r})'

    def update(self, params: typing.Optional[QueryParamTypes]=None) -> None:
        if False:
            while True:
                i = 10
        raise RuntimeError('QueryParams are immutable since 0.18.0. Use `q = q.merge(...)` to create an updated copy.')

    def __setitem__(self, key: str, value: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError('QueryParams are immutable since 0.18.0. Use `q = q.set(key, value)` to create an updated copy.')