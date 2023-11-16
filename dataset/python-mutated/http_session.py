import re
import time
import warnings
from typing import Any, Dict, Pattern, Tuple
import requests.adapters
import urllib3
from requests import PreparedRequest, Request, Session
from requests.adapters import HTTPAdapter
from streamlink.exceptions import PluginError, StreamlinkDeprecationWarning
from streamlink.packages.requests_file import FileAdapter
from streamlink.plugin.api import useragents
from streamlink.utils.parse import parse_json, parse_xml
try:
    from urllib3.util import create_urllib3_context
except ImportError:
    from urllib3.util.ssl_ import create_urllib3_context

class _HTTPResponse(urllib3.response.HTTPResponse):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        kwargs['enforce_content_length'] = True
        super().__init__(*args, **kwargs)
urllib3.connectionpool.HTTPConnectionPool.ResponseCls = _HTTPResponse
requests.adapters.HTTPResponse = _HTTPResponse

class Urllib3UtilUrlPercentReOverride:
    _re_percent_encoding: Pattern = getattr(urllib3.util.url, '_PERCENT_RE', getattr(urllib3.util.url, 'PERCENT_RE', re.compile('%[a-fA-F0-9]{2}')))

    @classmethod
    def subn(cls, repl: Any, string: str, count: Any=None) -> Tuple[str, int]:
        if False:
            return 10
        return (string, len(cls._re_percent_encoding.findall(string)))
urllib3.util.url._PERCENT_RE = urllib3.util.url.PERCENT_RE = Urllib3UtilUrlPercentReOverride
_VALID_REQUEST_ARGS = ('method', 'url', 'headers', 'files', 'data', 'params', 'auth', 'cookies', 'json')

class HTTPSession(Session):
    params: Dict

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.headers['User-Agent'] = useragents.FIREFOX
        self.timeout = 20.0
        self.mount('file://', FileAdapter())

    @classmethod
    def determine_json_encoding(cls, sample: bytes):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine which Unicode encoding the JSON text sample is encoded with\n\n        RFC4627 suggests that the encoding of JSON text can be determined\n        by checking the pattern of NULL bytes in first 4 octets of the text.\n        https://datatracker.ietf.org/doc/html/rfc4627#section-3\n\n        :param sample: a sample of at least 4 bytes of the JSON text\n        :return: the most likely encoding of the JSON text\n        '
        warnings.warn('Deprecated HTTPSession.determine_json_encoding() call', StreamlinkDeprecationWarning, stacklevel=1)
        data = int.from_bytes(sample[:4], 'big')
        if data & 4294967040 == 0:
            return 'UTF-32BE'
        elif data & 4278255360 == 0:
            return 'UTF-16BE'
        elif data & 16777215 == 0:
            return 'UTF-32LE'
        elif data & 16711935 == 0:
            return 'UTF-16LE'
        else:
            return 'UTF-8'

    @classmethod
    def json(cls, res, *args, **kwargs):
        if False:
            print('Hello World!')
        'Parses JSON from a response.'
        if res.encoding is None:
            return parse_json(res.content, *args, **kwargs)
        else:
            return parse_json(res.text, *args, **kwargs)

    @classmethod
    def xml(cls, res, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Parses XML from a response.'
        return parse_xml(res.text, *args, **kwargs)

    def resolve_url(self, url):
        if False:
            print('Hello World!')
        'Resolves any redirects and returns the final URL.'
        return self.get(url, stream=True).url

    @staticmethod
    def valid_request_args(**req_keywords) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        return {k: v for (k, v) in req_keywords.items() if k in _VALID_REQUEST_ARGS}

    def prepare_new_request(self, **req_keywords) -> PreparedRequest:
        if False:
            for i in range(10):
                print('nop')
        valid_args = self.valid_request_args(**req_keywords)
        valid_args.setdefault('method', 'GET')
        request = Request(**valid_args)
        return self.prepare_request(request)

    def request(self, method, url, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        acceptable_status = kwargs.pop('acceptable_status', [])
        exception = kwargs.pop('exception', PluginError)
        headers = kwargs.pop('headers', {})
        params = kwargs.pop('params', {})
        proxies = kwargs.pop('proxies', self.proxies)
        raise_for_status = kwargs.pop('raise_for_status', True)
        schema = kwargs.pop('schema', None)
        session = kwargs.pop('session', None)
        timeout = kwargs.pop('timeout', self.timeout)
        total_retries = kwargs.pop('retries', 0)
        retry_backoff = kwargs.pop('retry_backoff', 0.3)
        retry_max_backoff = kwargs.pop('retry_max_backoff', 10.0)
        retries = 0
        if session:
            headers.update(session.headers)
            params.update(session.params)
        while True:
            try:
                res = super().request(method, url, *args, headers=headers, params=params, timeout=timeout, proxies=proxies, **kwargs)
                if raise_for_status and res.status_code not in acceptable_status:
                    res.raise_for_status()
                break
            except KeyboardInterrupt:
                raise
            except Exception as rerr:
                if retries >= total_retries:
                    err = exception(f'Unable to open URL: {url} ({rerr})')
                    err.err = rerr
                    raise err from None
                retries += 1
                delay = min(retry_max_backoff, retry_backoff * 2 ** (retries - 1))
                time.sleep(delay)
        if schema:
            res = schema.validate(res.text, name='response text', exception=PluginError)
        return res

class TLSNoDHAdapter(HTTPAdapter):

    def init_poolmanager(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ctx = create_urllib3_context()
        ctx.load_default_certs()
        ciphers = ':'.join((cipher.get('name') for cipher in ctx.get_ciphers()))
        ciphers += ':!DH'
        ctx.set_ciphers(ciphers)
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

class TLSSecLevel1Adapter(HTTPAdapter):

    def init_poolmanager(self, *args, **kwargs):
        if False:
            print('Hello World!')
        ctx = create_urllib3_context()
        ctx.load_default_certs()
        ctx.set_ciphers('DEFAULT:@SECLEVEL=1')
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)
__all__ = ['HTTPSession', 'TLSNoDHAdapter', 'TLSSecLevel1Adapter']