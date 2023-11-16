from __future__ import unicode_literals
import logging
from oauthlib.common import generate_token, urldecode
from oauthlib.oauth2 import WebApplicationClient, InsecureTransportError
from oauthlib.oauth2 import LegacyApplicationClient
from oauthlib.oauth2 import TokenExpiredError, is_secure_transport
import requests
log = logging.getLogger(__name__)

class TokenUpdated(Warning):

    def __init__(self, token):
        if False:
            print('Hello World!')
        super(TokenUpdated, self).__init__()
        self.token = token

class OAuth2Session(requests.Session):
    """Versatile OAuth 2 extension to :class:`requests.Session`.

    Supports any grant type adhering to :class:`oauthlib.oauth2.Client` spec
    including the four core OAuth 2 grants.

    Can be used to create authorization urls, fetch tokens and access protected
    resources using the :class:`requests.Session` interface you are used to.

    - :class:`oauthlib.oauth2.WebApplicationClient` (default): Authorization Code Grant
    - :class:`oauthlib.oauth2.MobileApplicationClient`: Implicit Grant
    - :class:`oauthlib.oauth2.LegacyApplicationClient`: Password Credentials Grant
    - :class:`oauthlib.oauth2.BackendApplicationClient`: Client Credentials Grant

    Note that the only time you will be using Implicit Grant from python is if
    you are driving a user agent able to obtain URL fragments.
    """

    def __init__(self, client_id=None, client=None, auto_refresh_url=None, auto_refresh_kwargs=None, scope=None, redirect_uri=None, token=None, state=None, token_updater=None, **kwargs):
        if False:
            while True:
                i = 10
        'Construct a new OAuth 2 client session.\n\n        :param client_id: Client id obtained during registration\n        :param client: :class:`oauthlib.oauth2.Client` to be used. Default is\n                       WebApplicationClient which is useful for any\n                       hosted application but not mobile or desktop.\n        :param scope: List of scopes you wish to request access to\n        :param redirect_uri: Redirect URI you registered as callback\n        :param token: Token dictionary, must include access_token\n                      and token_type.\n        :param state: State string used to prevent CSRF. This will be given\n                      when creating the authorization url and must be supplied\n                      when parsing the authorization response.\n                      Can be either a string or a no argument callable.\n        :auto_refresh_url: Refresh token endpoint URL, must be HTTPS. Supply\n                           this if you wish the client to automatically refresh\n                           your access tokens.\n        :auto_refresh_kwargs: Extra arguments to pass to the refresh token\n                              endpoint.\n        :token_updater: Method with one argument, token, to be used to update\n                        your token database on automatic token refresh. If not\n                        set a TokenUpdated warning will be raised when a token\n                        has been refreshed. This warning will carry the token\n                        in its token argument.\n        :param kwargs: Arguments to pass to the Session constructor.\n        '
        super(OAuth2Session, self).__init__(**kwargs)
        self._client = client or WebApplicationClient(client_id, token=token)
        self.token = token or {}
        self.scope = scope
        self.redirect_uri = redirect_uri
        self.state = state or generate_token
        self._state = state
        self.auto_refresh_url = auto_refresh_url
        self.auto_refresh_kwargs = auto_refresh_kwargs or {}
        self.token_updater = token_updater
        self.auth = lambda r: r
        self.compliance_hook = {'access_token_response': set(), 'refresh_token_response': set(), 'protected_request': set()}

    def new_state(self):
        if False:
            print('Hello World!')
        'Generates a state string to be used in authorizations.'
        try:
            self._state = self.state()
            log.debug('Generated new state %s.', self._state)
        except TypeError:
            self._state = self.state
            log.debug('Re-using previously supplied state %s.', self._state)
        return self._state

    @property
    def client_id(self):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._client, 'client_id', None)

    @client_id.setter
    def client_id(self, value):
        if False:
            i = 10
            return i + 15
        self._client.client_id = value

    @client_id.deleter
    def client_id(self):
        if False:
            while True:
                i = 10
        del self._client.client_id

    @property
    def token(self):
        if False:
            while True:
                i = 10
        return getattr(self._client, 'token', None)

    @token.setter
    def token(self, value):
        if False:
            i = 10
            return i + 15
        self._client.token = value
        self._client.populate_token_attributes(value)

    @property
    def access_token(self):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._client, 'access_token', None)

    @access_token.setter
    def access_token(self, value):
        if False:
            i = 10
            return i + 15
        self._client.access_token = value

    @access_token.deleter
    def access_token(self):
        if False:
            while True:
                i = 10
        del self._client.access_token

    @property
    def authorized(self):
        if False:
            print('Hello World!')
        'Boolean that indicates whether this session has an OAuth token\n        or not. If `self.authorized` is True, you can reasonably expect\n        OAuth-protected requests to the resource to succeed. If\n        `self.authorized` is False, you need the user to go through the OAuth\n        authentication dance before OAuth-protected requests to the resource\n        will succeed.\n        '
        return bool(self.access_token)

    def authorization_url(self, url, state=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Form an authorization URL.\n\n        :param url: Authorization endpoint url, must be HTTPS.\n        :param state: An optional state string for CSRF protection. If not\n                      given it will be generated for you.\n        :param kwargs: Extra parameters to include.\n        :return: authorization_url, state\n        '
        state = state or self.new_state()
        return (self._client.prepare_request_uri(url, redirect_uri=self.redirect_uri, scope=self.scope, state=state, **kwargs), state)

    def fetch_token(self, token_url, code=None, authorization_response=None, body='', auth=None, username=None, password=None, method='POST', force_querystring=False, timeout=None, headers=None, verify=True, proxies=None, include_client_id=None, client_secret=None, cert=None, **kwargs):
        if False:
            return 10
        'Generic method for fetching an access token from the token endpoint.\n\n        If you are using the MobileApplicationClient you will want to use\n        `token_from_fragment` instead of `fetch_token`.\n\n        The current implementation enforces the RFC guidelines.\n\n        :param token_url: Token endpoint URL, must use HTTPS.\n        :param code: Authorization code (used by WebApplicationClients).\n        :param authorization_response: Authorization response URL, the callback\n                                       URL of the request back to you. Used by\n                                       WebApplicationClients instead of code.\n        :param body: Optional application/x-www-form-urlencoded body to add the\n                     include in the token request. Prefer kwargs over body.\n        :param auth: An auth tuple or method as accepted by `requests`.\n        :param username: Username required by LegacyApplicationClients to appear\n                         in the request body.\n        :param password: Password required by LegacyApplicationClients to appear\n                         in the request body.\n        :param method: The HTTP method used to make the request. Defaults\n                       to POST, but may also be GET. Other methods should\n                       be added as needed.\n        :param force_querystring: If True, force the request body to be sent\n            in the querystring instead.\n        :param timeout: Timeout of the request in seconds.\n        :param headers: Dict to default request headers with.\n        :param verify: Verify SSL certificate.\n        :param proxies: The `proxies` argument is passed onto `requests`.\n        :param include_client_id: Should the request body include the\n                                  `client_id` parameter. Default is `None`,\n                                  which will attempt to autodetect. This can be\n                                  forced to always include (True) or never\n                                  include (False).\n        :param client_secret: The `client_secret` paired to the `client_id`.\n                              This is generally required unless provided in the\n                              `auth` tuple. If the value is `None`, it will be\n                              omitted from the request, however if the value is\n                              an empty string, an empty string will be sent.\n        :param cert: Client certificate to send for OAuth 2.0 Mutual-TLS Client\n                     Authentication (draft-ietf-oauth-mtls). Can either be the\n                     path of a file containing the private key and certificate or\n                     a tuple of two filenames for certificate and key.\n        :param kwargs: Extra parameters to include in the token request.\n        :return: A token dict\n        '
        if not is_secure_transport(token_url):
            raise InsecureTransportError()
        if not code and authorization_response:
            self._client.parse_request_uri_response(authorization_response, state=self._state)
            code = self._client.code
        elif not code and isinstance(self._client, WebApplicationClient):
            code = self._client.code
            if not code:
                raise ValueError('Please supply either code or authorization_response parameters.')
        if isinstance(self._client, LegacyApplicationClient):
            if username is None:
                raise ValueError('`LegacyApplicationClient` requires both the `username` and `password` parameters.')
            if password is None:
                raise ValueError('The required parameter `username` was supplied, but `password` was not.')
        if username is not None:
            kwargs['username'] = username
        if password is not None:
            kwargs['password'] = password
        if auth is not None:
            if include_client_id is None:
                include_client_id = False
        elif include_client_id is not True:
            client_id = self.client_id
            if client_id:
                log.debug('Encoding `client_id` "%s" with `client_secret` as Basic auth credentials.', client_id)
                client_secret = client_secret if client_secret is not None else ''
                auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
        if include_client_id:
            if client_secret is not None:
                kwargs['client_secret'] = client_secret
        body = self._client.prepare_request_body(code=code, body=body, redirect_uri=self.redirect_uri, include_client_id=include_client_id, **kwargs)
        headers = headers or {'Accept': 'application/json', 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'}
        self.token = {}
        request_kwargs = {}
        if method.upper() == 'POST':
            request_kwargs['params' if force_querystring else 'data'] = dict(urldecode(body))
        elif method.upper() == 'GET':
            request_kwargs['params'] = dict(urldecode(body))
        else:
            raise ValueError('The method kwarg must be POST or GET.')
        r = self.request(method=method, url=token_url, timeout=timeout, headers=headers, auth=auth, verify=verify, proxies=proxies, cert=cert, **request_kwargs)
        log.debug('Request to fetch token completed with status %s.', r.status_code)
        log.debug('Request url was %s', r.request.url)
        log.debug('Request headers were %s', r.request.headers)
        log.debug('Request body was %s', r.request.body)
        log.debug('Response headers were %s and content %s.', r.headers, r.text)
        log.debug('Invoking %d token response hooks.', len(self.compliance_hook['access_token_response']))
        for hook in self.compliance_hook['access_token_response']:
            log.debug('Invoking hook %s.', hook)
            r = hook(r)
        self._client.parse_request_body_response(r.text, scope=self.scope)
        self.token = self._client.token
        log.debug('Obtained token %s.', self.token)
        return self.token

    def token_from_fragment(self, authorization_response):
        if False:
            print('Hello World!')
        'Parse token from the URI fragment, used by MobileApplicationClients.\n\n        :param authorization_response: The full URL of the redirect back to you\n        :return: A token dict\n        '
        self._client.parse_request_uri_response(authorization_response, state=self._state)
        self.token = self._client.token
        return self.token

    def refresh_token(self, token_url, refresh_token=None, body='', auth=None, timeout=None, headers=None, verify=True, proxies=None, **kwargs):
        if False:
            while True:
                i = 10
        'Fetch a new access token using a refresh token.\n\n        :param token_url: The token endpoint, must be HTTPS.\n        :param refresh_token: The refresh_token to use.\n        :param body: Optional application/x-www-form-urlencoded body to add the\n                     include in the token request. Prefer kwargs over body.\n        :param auth: An auth tuple or method as accepted by `requests`.\n        :param timeout: Timeout of the request in seconds.\n        :param headers: A dict of headers to be used by `requests`.\n        :param verify: Verify SSL certificate.\n        :param proxies: The `proxies` argument will be passed to `requests`.\n        :param kwargs: Extra parameters to include in the token request.\n        :return: A token dict\n        '
        if not token_url:
            raise ValueError('No token endpoint set for auto_refresh.')
        if not is_secure_transport(token_url):
            raise InsecureTransportError()
        refresh_token = refresh_token or self.token.get('refresh_token')
        log.debug('Adding auto refresh key word arguments %s.', self.auto_refresh_kwargs)
        kwargs.update(self.auto_refresh_kwargs)
        body = self._client.prepare_refresh_body(body=body, refresh_token=refresh_token, scope=self.scope, **kwargs)
        log.debug('Prepared refresh token request body %s', body)
        if headers is None:
            headers = {'Accept': 'application/json', 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'}
        r = self.post(token_url, data=dict(urldecode(body)), auth=auth, timeout=timeout, headers=headers, verify=verify, withhold_token=True, proxies=proxies)
        log.debug('Request to refresh token completed with status %s.', r.status_code)
        log.debug('Response headers were %s and content %s.', r.headers, r.text)
        log.debug('Invoking %d token response hooks.', len(self.compliance_hook['refresh_token_response']))
        for hook in self.compliance_hook['refresh_token_response']:
            log.debug('Invoking hook %s.', hook)
            r = hook(r)
        self.token = self._client.parse_request_body_response(r.text, scope=self.scope)
        if not 'refresh_token' in self.token:
            log.debug('No new refresh token given. Re-using old.')
            self.token['refresh_token'] = refresh_token
        return self.token

    def request(self, method, url, data=None, headers=None, withhold_token=False, client_id=None, client_secret=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Intercept all requests and add the OAuth 2 token if present.'
        if not is_secure_transport(url):
            raise InsecureTransportError()
        if self.token and (not withhold_token):
            log.debug('Invoking %d protected resource request hooks.', len(self.compliance_hook['protected_request']))
            for hook in self.compliance_hook['protected_request']:
                log.debug('Invoking hook %s.', hook)
                (url, headers, data) = hook(url, headers, data)
            log.debug('Adding token %s to request.', self.token)
            try:
                (url, headers, data) = self._client.add_token(url, http_method=method, body=data, headers=headers)
            except TokenExpiredError:
                if self.auto_refresh_url:
                    log.debug('Auto refresh is set, attempting to refresh at %s.', self.auto_refresh_url)
                    auth = kwargs.pop('auth', None)
                    if client_id and client_secret and (auth is None):
                        log.debug('Encoding client_id "%s" with client_secret as Basic auth credentials.', client_id)
                        auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
                    token = self.refresh_token(self.auto_refresh_url, auth=auth, **kwargs)
                    if self.token_updater:
                        log.debug('Updating token to %s using %s.', token, self.token_updater)
                        self.token_updater(token)
                        (url, headers, data) = self._client.add_token(url, http_method=method, body=data, headers=headers)
                    else:
                        raise TokenUpdated(token)
                else:
                    raise
        log.debug('Requesting url %s using method %s.', url, method)
        log.debug('Supplying headers %s and data %s', headers, data)
        log.debug('Passing through key word arguments %s.', kwargs)
        return super(OAuth2Session, self).request(method, url, headers=headers, data=data, **kwargs)

    def register_compliance_hook(self, hook_type, hook):
        if False:
            print('Hello World!')
        'Register a hook for request/response tweaking.\n\n        Available hooks are:\n            access_token_response invoked before token parsing.\n            refresh_token_response invoked before refresh token parsing.\n            protected_request invoked before making a request.\n\n        If you find a new hook is needed please send a GitHub PR request\n        or open an issue.\n        '
        if hook_type not in self.compliance_hook:
            raise ValueError('Hook type %s is not in %s.', hook_type, self.compliance_hook)
        self.compliance_hook[hook_type].add(hook)