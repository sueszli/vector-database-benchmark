import hashlib
import urllib
import base64
import hmac
from urllib.parse import ParseResult, urlparse
from typing import Union
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.policies import SansIOHTTPPolicy
from .._shared.utils import get_current_utc_time

class CallAutomationHMACCredentialsPolicy(SansIOHTTPPolicy):
    """Implementation of HMAC authentication policy.

    :param str host: The host of the endpoint url for Azure Communication Service resource
    :param access_key: The access key we use to authenticate to the service
    :type access_key: str or AzureKeyCredential
    :param bool decode_url: `True` if there is a need to decode the url. Default value is `False`
    """

    def __init__(self, host, acs_url, access_key, decode_url=False):
        if False:
            while True:
                i = 10
        super(CallAutomationHMACCredentialsPolicy, self).__init__()
        if host.startswith('https://'):
            self._host = host.replace('https://', '')
        if host.startswith('http://'):
            self._host = host.replace('http://', '')
        self._access_key = access_key
        self._decode_url = decode_url
        self._acs_url = acs_url

    def _compute_hmac(self, value):
        if False:
            i = 10
            return i + 15
        if isinstance(self._access_key, AzureKeyCredential):
            decoded_secret = base64.b64decode(self._access_key.key)
        else:
            decoded_secret = base64.b64decode(self._access_key)
        digest = hmac.new(decoded_secret, value.encode('utf-8'), hashlib.sha256).digest()
        return base64.b64encode(digest).decode('utf-8')

    def _sign_request(self, request):
        if False:
            i = 10
            return i + 15
        verb = request.http_request.method.upper()
        parsed_url: ParseResult = urlparse(request.http_request.url)
        query_url = parsed_url.path
        parsed_acs_url: ParseResult = urlparse(self._acs_url)
        if parsed_url.query:
            query_url += '?' + parsed_url.query
        try:
            from yarl import URL
            from azure.core.pipeline.transport import AioHttpTransport
            if isinstance(request.context.transport, AioHttpTransport) or isinstance(getattr(request.context.transport, '_transport', None), AioHttpTransport) or isinstance(getattr(getattr(request.context.transport, '_transport', None), '_transport', None), AioHttpTransport):
                query_url = str(URL(query_url))
        except (ImportError, TypeError):
            pass
        if self._decode_url:
            query_url = urllib.parse.unquote(query_url)
        signed_headers = 'x-ms-date;host;x-ms-content-sha256'
        utc_now = get_current_utc_time()
        if request.http_request.body is None:
            request.http_request.body = ''
        content_digest = hashlib.sha256(request.http_request.body.encode('utf-8')).digest()
        content_hash = base64.b64encode(content_digest).decode('utf-8')
        string_to_sign = verb + '\n' + query_url + '\n' + utc_now + ';' + parsed_acs_url.hostname + ';' + content_hash
        signature = self._compute_hmac(string_to_sign)
        signature_header = {'x-ms-host': parsed_acs_url.hostname, 'x-ms-date': utc_now, 'x-ms-content-sha256': content_hash, 'x-ms-return-client-request-id': 'true', 'Authorization': 'HMAC-SHA256 SignedHeaders=' + signed_headers + '&Signature=' + signature}
        request.http_request.headers.update(signature_header)
        return request

    def on_request(self, request):
        if False:
            print('Hello World!')
        self._sign_request(request)