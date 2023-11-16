from __future__ import unicode_literals
try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse
import logging
from oauthlib.common import add_params_to_uri
from oauthlib.common import urldecode as _urldecode
from oauthlib.oauth1 import SIGNATURE_HMAC, SIGNATURE_RSA, SIGNATURE_TYPE_AUTH_HEADER
import requests
from . import OAuth1
log = logging.getLogger(__name__)

def urldecode(body):
    if False:
        while True:
            i = 10
    'Parse query or json to python dictionary'
    try:
        return _urldecode(body)
    except Exception:
        import json
        return json.loads(body)

class TokenRequestDenied(ValueError):

    def __init__(self, message, response):
        if False:
            while True:
                i = 10
        super(TokenRequestDenied, self).__init__(message)
        self.response = response

    @property
    def status_code(self):
        if False:
            return 10
        'For backwards-compatibility purposes'
        return self.response.status_code

class TokenMissing(ValueError):

    def __init__(self, message, response):
        if False:
            return 10
        super(TokenMissing, self).__init__(message)
        self.response = response

class VerifierMissing(ValueError):
    pass

class OAuth1Session(requests.Session):
    """Request signing and convenience methods for the oauth dance.

    What is the difference between OAuth1Session and OAuth1?

    OAuth1Session actually uses OAuth1 internally and its purpose is to assist
    in the OAuth workflow through convenience methods to prepare authorization
    URLs and parse the various token and redirection responses. It also provide
    rudimentary validation of responses.

    An example of the OAuth workflow using a basic CLI app and Twitter.

    >>> # Credentials obtained during the registration.
    >>> client_key = 'client key'
    >>> client_secret = 'secret'
    >>> callback_uri = 'https://127.0.0.1/callback'
    >>>
    >>> # Endpoints found in the OAuth provider API documentation
    >>> request_token_url = 'https://api.twitter.com/oauth/request_token'
    >>> authorization_url = 'https://api.twitter.com/oauth/authorize'
    >>> access_token_url = 'https://api.twitter.com/oauth/access_token'
    >>>
    >>> oauth_session = OAuth1Session(client_key,client_secret=client_secret, callback_uri=callback_uri)
    >>>
    >>> # First step, fetch the request token.
    >>> oauth_session.fetch_request_token(request_token_url)
    {
        'oauth_token': 'kjerht2309u',
        'oauth_token_secret': 'lsdajfh923874',
    }
    >>>
    >>> # Second step. Follow this link and authorize
    >>> oauth_session.authorization_url(authorization_url)
    'https://api.twitter.com/oauth/authorize?oauth_token=sdf0o9823sjdfsdf&oauth_callback=https%3A%2F%2F127.0.0.1%2Fcallback'
    >>>
    >>> # Third step. Fetch the access token
    >>> redirect_response = raw_input('Paste the full redirect URL here.')
    >>> oauth_session.parse_authorization_response(redirect_response)
    {
        'oauth_token: 'kjerht2309u',
        'oauth_token_secret: 'lsdajfh923874',
        'oauth_verifier: 'w34o8967345',
    }
    >>> oauth_session.fetch_access_token(access_token_url)
    {
        'oauth_token': 'sdf0o9823sjdfsdf',
        'oauth_token_secret': '2kjshdfp92i34asdasd',
    }
    >>> # Done. You can now make OAuth requests.
    >>> status_url = 'http://api.twitter.com/1/statuses/update.json'
    >>> new_status = {'status':  'hello world!'}
    >>> oauth_session.post(status_url, data=new_status)
    <Response [200]>
    """

    def __init__(self, client_key, client_secret=None, resource_owner_key=None, resource_owner_secret=None, callback_uri=None, signature_method=SIGNATURE_HMAC, signature_type=SIGNATURE_TYPE_AUTH_HEADER, rsa_key=None, verifier=None, client_class=None, force_include_body=False, **kwargs):
        if False:
            print('Hello World!')
        'Construct the OAuth 1 session.\n\n        :param client_key: A client specific identifier.\n        :param client_secret: A client specific secret used to create HMAC and\n                              plaintext signatures.\n        :param resource_owner_key: A resource owner key, also referred to as\n                                   request token or access token depending on\n                                   when in the workflow it is used.\n        :param resource_owner_secret: A resource owner secret obtained with\n                                      either a request or access token. Often\n                                      referred to as token secret.\n        :param callback_uri: The URL the user is redirect back to after\n                             authorization.\n        :param signature_method: Signature methods determine how the OAuth\n                                 signature is created. The three options are\n                                 oauthlib.oauth1.SIGNATURE_HMAC (default),\n                                 oauthlib.oauth1.SIGNATURE_RSA and\n                                 oauthlib.oauth1.SIGNATURE_PLAIN.\n        :param signature_type: Signature type decides where the OAuth\n                               parameters are added. Either in the\n                               Authorization header (default) or to the URL\n                               query parameters or the request body. Defined as\n                               oauthlib.oauth1.SIGNATURE_TYPE_AUTH_HEADER,\n                               oauthlib.oauth1.SIGNATURE_TYPE_QUERY and\n                               oauthlib.oauth1.SIGNATURE_TYPE_BODY\n                               respectively.\n        :param rsa_key: The private RSA key as a string. Can only be used with\n                        signature_method=oauthlib.oauth1.SIGNATURE_RSA.\n        :param verifier: A verifier string to prove authorization was granted.\n        :param client_class: A subclass of `oauthlib.oauth1.Client` to use with\n                             `requests_oauthlib.OAuth1` instead of the default\n        :param force_include_body: Always include the request body in the\n                                   signature creation.\n        :param **kwargs: Additional keyword arguments passed to `OAuth1`\n        '
        super(OAuth1Session, self).__init__()
        self._client = OAuth1(client_key, client_secret=client_secret, resource_owner_key=resource_owner_key, resource_owner_secret=resource_owner_secret, callback_uri=callback_uri, signature_method=signature_method, signature_type=signature_type, rsa_key=rsa_key, verifier=verifier, client_class=client_class, force_include_body=force_include_body, **kwargs)
        self.auth = self._client

    @property
    def token(self):
        if False:
            print('Hello World!')
        oauth_token = self._client.client.resource_owner_key
        oauth_token_secret = self._client.client.resource_owner_secret
        oauth_verifier = self._client.client.verifier
        token_dict = {}
        if oauth_token:
            token_dict['oauth_token'] = oauth_token
        if oauth_token_secret:
            token_dict['oauth_token_secret'] = oauth_token_secret
        if oauth_verifier:
            token_dict['oauth_verifier'] = oauth_verifier
        return token_dict

    @token.setter
    def token(self, value):
        if False:
            while True:
                i = 10
        self._populate_attributes(value)

    @property
    def authorized(self):
        if False:
            while True:
                i = 10
        'Boolean that indicates whether this session has an OAuth token\n        or not. If `self.authorized` is True, you can reasonably expect\n        OAuth-protected requests to the resource to succeed. If\n        `self.authorized` is False, you need the user to go through the OAuth\n        authentication dance before OAuth-protected requests to the resource\n        will succeed.\n        '
        if self._client.client.signature_method == SIGNATURE_RSA:
            return bool(self._client.client.resource_owner_key)
        else:
            return bool(self._client.client.client_secret) and bool(self._client.client.resource_owner_key) and bool(self._client.client.resource_owner_secret)

    def authorization_url(self, url, request_token=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "Create an authorization URL by appending request_token and optional\n        kwargs to url.\n\n        This is the second step in the OAuth 1 workflow. The user should be\n        redirected to this authorization URL, grant access to you, and then\n        be redirected back to you. The redirection back can either be specified\n        during client registration or by supplying a callback URI per request.\n\n        :param url: The authorization endpoint URL.\n        :param request_token: The previously obtained request token.\n        :param kwargs: Optional parameters to append to the URL.\n        :returns: The authorization URL with new parameters embedded.\n\n        An example using a registered default callback URI.\n\n        >>> request_token_url = 'https://api.twitter.com/oauth/request_token'\n        >>> authorization_url = 'https://api.twitter.com/oauth/authorize'\n        >>> oauth_session = OAuth1Session('client-key', client_secret='secret')\n        >>> oauth_session.fetch_request_token(request_token_url)\n        {\n            'oauth_token': 'sdf0o9823sjdfsdf',\n            'oauth_token_secret': '2kjshdfp92i34asdasd',\n        }\n        >>> oauth_session.authorization_url(authorization_url)\n        'https://api.twitter.com/oauth/authorize?oauth_token=sdf0o9823sjdfsdf'\n        >>> oauth_session.authorization_url(authorization_url, foo='bar')\n        'https://api.twitter.com/oauth/authorize?oauth_token=sdf0o9823sjdfsdf&foo=bar'\n\n        An example using an explicit callback URI.\n\n        >>> request_token_url = 'https://api.twitter.com/oauth/request_token'\n        >>> authorization_url = 'https://api.twitter.com/oauth/authorize'\n        >>> oauth_session = OAuth1Session('client-key', client_secret='secret', callback_uri='https://127.0.0.1/callback')\n        >>> oauth_session.fetch_request_token(request_token_url)\n        {\n            'oauth_token': 'sdf0o9823sjdfsdf',\n            'oauth_token_secret': '2kjshdfp92i34asdasd',\n        }\n        >>> oauth_session.authorization_url(authorization_url)\n        'https://api.twitter.com/oauth/authorize?oauth_token=sdf0o9823sjdfsdf&oauth_callback=https%3A%2F%2F127.0.0.1%2Fcallback'\n        "
        kwargs['oauth_token'] = request_token or self._client.client.resource_owner_key
        log.debug('Adding parameters %s to url %s', kwargs, url)
        return add_params_to_uri(url, kwargs.items())

    def fetch_request_token(self, url, realm=None, **request_kwargs):
        if False:
            while True:
                i = 10
        "Fetch a request token.\n\n        This is the first step in the OAuth 1 workflow. A request token is\n        obtained by making a signed post request to url. The token is then\n        parsed from the application/x-www-form-urlencoded response and ready\n        to be used to construct an authorization url.\n\n        :param url: The request token endpoint URL.\n        :param realm: A list of realms to request access to.\n        :param \\*\\*request_kwargs: Optional arguments passed to ''post''\n            function in ''requests.Session''\n        :returns: The response in dict format.\n\n        Note that a previously set callback_uri will be reset for your\n        convenience, or else signature creation will be incorrect on\n        consecutive requests.\n\n        >>> request_token_url = 'https://api.twitter.com/oauth/request_token'\n        >>> oauth_session = OAuth1Session('client-key', client_secret='secret')\n        >>> oauth_session.fetch_request_token(request_token_url)\n        {\n            'oauth_token': 'sdf0o9823sjdfsdf',\n            'oauth_token_secret': '2kjshdfp92i34asdasd',\n        }\n        "
        self._client.client.realm = ' '.join(realm) if realm else None
        token = self._fetch_token(url, **request_kwargs)
        log.debug('Resetting callback_uri and realm (not needed in next phase).')
        self._client.client.callback_uri = None
        self._client.client.realm = None
        return token

    def fetch_access_token(self, url, verifier=None, **request_kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Fetch an access token.\n\n        This is the final step in the OAuth 1 workflow. An access token is\n        obtained using all previously obtained credentials, including the\n        verifier from the authorization step.\n\n        Note that a previously set verifier will be reset for your\n        convenience, or else signature creation will be incorrect on\n        consecutive requests.\n\n        >>> access_token_url = 'https://api.twitter.com/oauth/access_token'\n        >>> redirect_response = 'https://127.0.0.1/callback?oauth_token=kjerht2309uf&oauth_token_secret=lsdajfh923874&oauth_verifier=w34o8967345'\n        >>> oauth_session = OAuth1Session('client-key', client_secret='secret')\n        >>> oauth_session.parse_authorization_response(redirect_response)\n        {\n            'oauth_token: 'kjerht2309u',\n            'oauth_token_secret: 'lsdajfh923874',\n            'oauth_verifier: 'w34o8967345',\n        }\n        >>> oauth_session.fetch_access_token(access_token_url)\n        {\n            'oauth_token': 'sdf0o9823sjdfsdf',\n            'oauth_token_secret': '2kjshdfp92i34asdasd',\n        }\n        "
        if verifier:
            self._client.client.verifier = verifier
        if not getattr(self._client.client, 'verifier', None):
            raise VerifierMissing('No client verifier has been set.')
        token = self._fetch_token(url, **request_kwargs)
        log.debug('Resetting verifier attribute, should not be used anymore.')
        self._client.client.verifier = None
        return token

    def parse_authorization_response(self, url):
        if False:
            return 10
        "Extract parameters from the post authorization redirect response URL.\n\n        :param url: The full URL that resulted from the user being redirected\n                    back from the OAuth provider to you, the client.\n        :returns: A dict of parameters extracted from the URL.\n\n        >>> redirect_response = 'https://127.0.0.1/callback?oauth_token=kjerht2309uf&oauth_token_secret=lsdajfh923874&oauth_verifier=w34o8967345'\n        >>> oauth_session = OAuth1Session('client-key', client_secret='secret')\n        >>> oauth_session.parse_authorization_response(redirect_response)\n        {\n            'oauth_token: 'kjerht2309u',\n            'oauth_token_secret: 'lsdajfh923874',\n            'oauth_verifier: 'w34o8967345',\n        }\n        "
        log.debug('Parsing token from query part of url %s', url)
        token = dict(urldecode(urlparse(url).query))
        log.debug('Updating internal client token attribute.')
        self._populate_attributes(token)
        self.token = token
        return token

    def _populate_attributes(self, token):
        if False:
            i = 10
            return i + 15
        if 'oauth_token' in token:
            self._client.client.resource_owner_key = token['oauth_token']
        else:
            raise TokenMissing('Response does not contain a token: {resp}'.format(resp=token), token)
        if 'oauth_token_secret' in token:
            self._client.client.resource_owner_secret = token['oauth_token_secret']
        if 'oauth_verifier' in token:
            self._client.client.verifier = token['oauth_verifier']

    def _fetch_token(self, url, **request_kwargs):
        if False:
            while True:
                i = 10
        log.debug('Fetching token from %s using client %s', url, self._client.client)
        r = self.post(url, **request_kwargs)
        if r.status_code >= 400:
            error = "Token request failed with code %s, response was '%s'."
            raise TokenRequestDenied(error % (r.status_code, r.text), r)
        log.debug('Decoding token from response "%s"', r.text)
        try:
            token = dict(urldecode(r.text.strip()))
        except ValueError as e:
            error = 'Unable to decode token from token response. This is commonly caused by an unsuccessful request where a non urlencoded error message is returned. The decoding error was %s' % e
            raise ValueError(error)
        log.debug('Obtained token %s', token)
        log.debug('Updating internal client attributes from token data.')
        self._populate_attributes(token)
        self.token = token
        return token

    def rebuild_auth(self, prepared_request, response):
        if False:
            return 10
        '\n        When being redirected we should always strip Authorization\n        header, since nonce may not be reused as per OAuth spec.\n        '
        if 'Authorization' in prepared_request.headers:
            prepared_request.headers.pop('Authorization', True)
            prepared_request.prepare_auth(self.auth)
        return