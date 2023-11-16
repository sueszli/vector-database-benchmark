from __future__ import annotations
import json as _json
import typing
from urllib.parse import urlencode
from ._base_connection import _TYPE_BODY
from ._collections import HTTPHeaderDict
from .filepost import _TYPE_FIELDS, encode_multipart_formdata
from .response import BaseHTTPResponse
__all__ = ['RequestMethods']
_TYPE_ENCODE_URL_FIELDS = typing.Union[typing.Sequence[typing.Tuple[str, typing.Union[str, bytes]]], typing.Mapping[str, typing.Union[str, bytes]]]

class RequestMethods:
    """
    Convenience mixin for classes who implement a :meth:`urlopen` method, such
    as :class:`urllib3.HTTPConnectionPool` and
    :class:`urllib3.PoolManager`.

    Provides behavior for making common types of HTTP request methods and
    decides which type of request field encoding to use.

    Specifically,

    :meth:`.request_encode_url` is for sending requests whose fields are
    encoded in the URL (such as GET, HEAD, DELETE).

    :meth:`.request_encode_body` is for sending requests whose fields are
    encoded in the *body* of the request using multipart or www-form-urlencoded
    (such as for POST, PUT, PATCH).

    :meth:`.request` is for making any kind of request, it will look up the
    appropriate encoding format and use one of the above two methods to make
    the request.

    Initializer parameters:

    :param headers:
        Headers to include with all requests, unless other headers are given
        explicitly.
    """
    _encode_url_methods = {'DELETE', 'GET', 'HEAD', 'OPTIONS'}

    def __init__(self, headers: typing.Mapping[str, str] | None=None) -> None:
        if False:
            print('Hello World!')
        self.headers = headers or {}

    def urlopen(self, method: str, url: str, body: _TYPE_BODY | None=None, headers: typing.Mapping[str, str] | None=None, encode_multipart: bool=True, multipart_boundary: str | None=None, **kw: typing.Any) -> BaseHTTPResponse:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('Classes extending RequestMethods must implement their own ``urlopen`` method.')

    def request(self, method: str, url: str, body: _TYPE_BODY | None=None, fields: _TYPE_FIELDS | None=None, headers: typing.Mapping[str, str] | None=None, json: typing.Any | None=None, **urlopen_kw: typing.Any) -> BaseHTTPResponse:
        if False:
            return 10
        '\n        Make a request using :meth:`urlopen` with the appropriate encoding of\n        ``fields`` based on the ``method`` used.\n\n        This is a convenience method that requires the least amount of manual\n        effort. It can be used in most situations, while still having the\n        option to drop down to more specific methods when necessary, such as\n        :meth:`request_encode_url`, :meth:`request_encode_body`,\n        or even the lowest level :meth:`urlopen`.\n\n        :param method:\n            HTTP request method (such as GET, POST, PUT, etc.)\n\n        :param url:\n            The URL to perform the request on.\n\n        :param body:\n            Data to send in the request body, either :class:`str`, :class:`bytes`,\n            an iterable of :class:`str`/:class:`bytes`, or a file-like object.\n\n        :param fields:\n            Data to encode and send in the request body.  Values are processed\n            by :func:`urllib.parse.urlencode`.\n\n        :param headers:\n            Dictionary of custom headers to send, such as User-Agent,\n            If-None-Match, etc. If None, pool headers are used. If provided,\n            these headers completely replace any pool-specific headers.\n\n        :param json:\n            Data to encode and send as JSON with UTF-encoded in the request body.\n            The ``"Content-Type"`` header will be set to ``"application/json"``\n            unless specified otherwise.\n        '
        method = method.upper()
        if json is not None and body is not None:
            raise TypeError("request got values for both 'body' and 'json' parameters which are mutually exclusive")
        if json is not None:
            if headers is None:
                headers = self.headers.copy()
            if not 'content-type' in map(str.lower, headers.keys()):
                headers['Content-Type'] = 'application/json'
            body = _json.dumps(json, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        if body is not None:
            urlopen_kw['body'] = body
        if method in self._encode_url_methods:
            return self.request_encode_url(method, url, fields=fields, headers=headers, **urlopen_kw)
        else:
            return self.request_encode_body(method, url, fields=fields, headers=headers, **urlopen_kw)

    def request_encode_url(self, method: str, url: str, fields: _TYPE_ENCODE_URL_FIELDS | None=None, headers: typing.Mapping[str, str] | None=None, **urlopen_kw: str) -> BaseHTTPResponse:
        if False:
            i = 10
            return i + 15
        '\n        Make a request using :meth:`urlopen` with the ``fields`` encoded in\n        the url. This is useful for request methods like GET, HEAD, DELETE, etc.\n\n        :param method:\n            HTTP request method (such as GET, POST, PUT, etc.)\n\n        :param url:\n            The URL to perform the request on.\n\n        :param fields:\n            Data to encode and send in the request body.\n\n        :param headers:\n            Dictionary of custom headers to send, such as User-Agent,\n            If-None-Match, etc. If None, pool headers are used. If provided,\n            these headers completely replace any pool-specific headers.\n        '
        if headers is None:
            headers = self.headers
        extra_kw: dict[str, typing.Any] = {'headers': headers}
        extra_kw.update(urlopen_kw)
        if fields:
            url += '?' + urlencode(fields)
        return self.urlopen(method, url, **extra_kw)

    def request_encode_body(self, method: str, url: str, fields: _TYPE_FIELDS | None=None, headers: typing.Mapping[str, str] | None=None, encode_multipart: bool=True, multipart_boundary: str | None=None, **urlopen_kw: str) -> BaseHTTPResponse:
        if False:
            for i in range(10):
                print('nop')
        "\n        Make a request using :meth:`urlopen` with the ``fields`` encoded in\n        the body. This is useful for request methods like POST, PUT, PATCH, etc.\n\n        When ``encode_multipart=True`` (default), then\n        :func:`urllib3.encode_multipart_formdata` is used to encode\n        the payload with the appropriate content type. Otherwise\n        :func:`urllib.parse.urlencode` is used with the\n        'application/x-www-form-urlencoded' content type.\n\n        Multipart encoding must be used when posting files, and it's reasonably\n        safe to use it in other times too. However, it may break request\n        signing, such as with OAuth.\n\n        Supports an optional ``fields`` parameter of key/value strings AND\n        key/filetuple. A filetuple is a (filename, data, MIME type) tuple where\n        the MIME type is optional. For example::\n\n            fields = {\n                'foo': 'bar',\n                'fakefile': ('foofile.txt', 'contents of foofile'),\n                'realfile': ('barfile.txt', open('realfile').read()),\n                'typedfile': ('bazfile.bin', open('bazfile').read(),\n                              'image/jpeg'),\n                'nonamefile': 'contents of nonamefile field',\n            }\n\n        When uploading a file, providing a filename (the first parameter of the\n        tuple) is optional but recommended to best mimic behavior of browsers.\n\n        Note that if ``headers`` are supplied, the 'Content-Type' header will\n        be overwritten because it depends on the dynamic random boundary string\n        which is used to compose the body of the request. The random boundary\n        string can be explicitly set with the ``multipart_boundary`` parameter.\n\n        :param method:\n            HTTP request method (such as GET, POST, PUT, etc.)\n\n        :param url:\n            The URL to perform the request on.\n\n        :param fields:\n            Data to encode and send in the request body.\n\n        :param headers:\n            Dictionary of custom headers to send, such as User-Agent,\n            If-None-Match, etc. If None, pool headers are used. If provided,\n            these headers completely replace any pool-specific headers.\n\n        :param encode_multipart:\n            If True, encode the ``fields`` using the multipart/form-data MIME\n            format.\n\n        :param multipart_boundary:\n            If not specified, then a random boundary will be generated using\n            :func:`urllib3.filepost.choose_boundary`.\n        "
        if headers is None:
            headers = self.headers
        extra_kw: dict[str, typing.Any] = {'headers': HTTPHeaderDict(headers)}
        body: bytes | str
        if fields:
            if 'body' in urlopen_kw:
                raise TypeError("request got values for both 'fields' and 'body', can only specify one.")
            if encode_multipart:
                (body, content_type) = encode_multipart_formdata(fields, boundary=multipart_boundary)
            else:
                (body, content_type) = (urlencode(fields), 'application/x-www-form-urlencoded')
            extra_kw['body'] = body
            extra_kw['headers'].setdefault('Content-Type', content_type)
        extra_kw.update(urlopen_kw)
        return self.urlopen(method, url, **extra_kw)