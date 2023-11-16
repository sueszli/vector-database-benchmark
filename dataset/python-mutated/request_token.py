"""
oauthlib.oauth1.rfc5849.endpoints.request_token
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module is an implementation of the request token provider logic of
OAuth 1.0 RFC 5849. It validates the correctness of request token requests,
creates and persists tokens as well as create the proper response to be
returned to the client.
"""
import logging
from oauthlib.common import urlencode
from .. import errors
from .base import BaseEndpoint
log = logging.getLogger(__name__)

class RequestTokenEndpoint(BaseEndpoint):
    """An endpoint responsible for providing OAuth 1 request tokens.

    Typical use is to instantiate with a request validator and invoke the
    ``create_request_token_response`` from a view function. The tuple returned
    has all information necessary (body, status, headers) to quickly form
    and return a proper response. See :doc:`/oauth1/validator` for details on which
    validator methods to implement for this endpoint.
    """

    def create_request_token(self, request, credentials):
        if False:
            return 10
        'Create and save a new request token.\n\n        :param request: OAuthlib request.\n        :type request: oauthlib.common.Request\n        :param credentials: A dict of extra token credentials.\n        :returns: The token as an urlencoded string.\n        '
        token = {'oauth_token': self.token_generator(), 'oauth_token_secret': self.token_generator(), 'oauth_callback_confirmed': 'true'}
        token.update(credentials)
        self.request_validator.save_request_token(token, request)
        return urlencode(token.items())

    def create_request_token_response(self, uri, http_method='GET', body=None, headers=None, credentials=None):
        if False:
            while True:
                i = 10
        "Create a request token response, with a new request token if valid.\n\n        :param uri: The full URI of the token request.\n        :param http_method: A valid HTTP verb, i.e. GET, POST, PUT, HEAD, etc.\n        :param body: The request body as a string.\n        :param headers: The request headers as a dict.\n        :param credentials: A list of extra credentials to include in the token.\n        :returns: A tuple of 3 elements.\n                  1. A dict of headers to set on the response.\n                  2. The response body as a string.\n                  3. The response status code as an integer.\n\n        An example of a valid request::\n\n            >>> from your_validator import your_validator\n            >>> from oauthlib.oauth1 import RequestTokenEndpoint\n            >>> endpoint = RequestTokenEndpoint(your_validator)\n            >>> h, b, s = endpoint.create_request_token_response(\n            ...     'https://your.provider/request_token?foo=bar',\n            ...     headers={\n            ...         'Authorization': 'OAuth realm=movies user, oauth_....'\n            ...     },\n            ...     credentials={\n            ...         'my_specific': 'argument',\n            ...     })\n            >>> h\n            {'Content-Type': 'application/x-www-form-urlencoded'}\n            >>> b\n            'oauth_token=lsdkfol23w54jlksdef&oauth_token_secret=qwe089234lkjsdf&oauth_callback_confirmed=true&my_specific=argument'\n            >>> s\n            200\n\n        An response to invalid request would have a different body and status::\n\n            >>> b\n            'error=invalid_request&description=missing+callback+uri'\n            >>> s\n            400\n\n        The same goes for an an unauthorized request:\n\n            >>> b\n            ''\n            >>> s\n            401\n        "
        resp_headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        try:
            request = self._create_request(uri, http_method, body, headers)
            (valid, processed_request) = self.validate_request_token_request(request)
            if valid:
                token = self.create_request_token(request, credentials or {})
                return (resp_headers, token, 200)
            else:
                return ({}, None, 401)
        except errors.OAuth1Error as e:
            return (resp_headers, e.urlencoded, e.status_code)

    def validate_request_token_request(self, request):
        if False:
            i = 10
            return i + 15
        'Validate a request token request.\n\n        :param request: OAuthlib request.\n        :type request: oauthlib.common.Request\n        :raises: OAuth1Error if the request is invalid.\n        :returns: A tuple of 2 elements.\n                  1. The validation result (True or False).\n                  2. The request object.\n        '
        self._check_transport_security(request)
        self._check_mandatory_parameters(request)
        if request.realm:
            request.realms = request.realm.split(' ')
        else:
            request.realms = self.request_validator.get_default_realms(request.client_key, request)
        if not self.request_validator.check_realms(request.realms):
            raise errors.InvalidRequestError(description='Invalid realm {}. Allowed are {!r}.'.format(request.realms, self.request_validator.realms))
        if not request.redirect_uri:
            raise errors.InvalidRequestError(description='Missing callback URI.')
        if not self.request_validator.validate_timestamp_and_nonce(request.client_key, request.timestamp, request.nonce, request, request_token=request.resource_owner_key):
            return (False, request)
        valid_client = self.request_validator.validate_client_key(request.client_key, request)
        if not valid_client:
            request.client_key = self.request_validator.dummy_client
        valid_realm = self.request_validator.validate_requested_realms(request.client_key, request.realms, request)
        valid_redirect = self.request_validator.validate_redirect_uri(request.client_key, request.redirect_uri, request)
        if not request.redirect_uri:
            raise NotImplementedError('Redirect URI must either be provided or set to a default during validation.')
        valid_signature = self._check_signature(request)
        request.validator_log['client'] = valid_client
        request.validator_log['realm'] = valid_realm
        request.validator_log['callback'] = valid_redirect
        request.validator_log['signature'] = valid_signature
        v = all((valid_client, valid_realm, valid_redirect, valid_signature))
        if not v:
            log.info('[Failure] request verification failed.')
            log.info('Valid client: %s.', valid_client)
            log.info('Valid realm: %s.', valid_realm)
            log.info('Valid callback: %s.', valid_redirect)
            log.info('Valid signature: %s.', valid_signature)
        return (v, request)