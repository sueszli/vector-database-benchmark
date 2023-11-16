from __future__ import unicode_literals
from oauthlib.oauth2 import WebApplicationClient, InsecureTransportError
from oauthlib.oauth2 import is_secure_transport
from requests.auth import AuthBase

class OAuth2(AuthBase):
    """Adds proof of authorization (OAuth2 token) to the request."""

    def __init__(self, client_id=None, client=None, token=None):
        if False:
            for i in range(10):
                print('nop')
        'Construct a new OAuth 2 authorization object.\n\n        :param client_id: Client id obtained during registration\n        :param client: :class:`oauthlib.oauth2.Client` to be used. Default is\n                       WebApplicationClient which is useful for any\n                       hosted application but not mobile or desktop.\n        :param token: Token dictionary, must include access_token\n                      and token_type.\n        '
        self._client = client or WebApplicationClient(client_id, token=token)
        if token:
            for (k, v) in token.items():
                setattr(self._client, k, v)

    def __call__(self, r):
        if False:
            i = 10
            return i + 15
        'Append an OAuth 2 token to the request.\n\n        Note that currently HTTPS is required for all requests. There may be\n        a token type that allows for plain HTTP in the future and then this\n        should be updated to allow plain HTTP on a white list basis.\n        '
        if not is_secure_transport(r.url):
            raise InsecureTransportError()
        (r.url, r.headers, r.body) = self._client.add_token(r.url, http_method=r.method, body=r.body, headers=r.headers)
        return r