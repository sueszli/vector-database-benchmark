"""
oauthlib.oauth2.rfc6749.endpoint.revocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An implementation of the OAuth 2 `Token Revocation`_ spec (draft 11).

.. _`Token Revocation`: https://tools.ietf.org/html/draft-ietf-oauth-revocation-11
"""
import logging
from oauthlib.common import Request
from ..errors import OAuth2Error
from .base import BaseEndpoint, catch_errors_and_unavailability
log = logging.getLogger(__name__)

class RevocationEndpoint(BaseEndpoint):
    """Token revocation endpoint.

    Endpoint used by authenticated clients to revoke access and refresh tokens.
    Commonly this will be part of the Authorization Endpoint.
    """
    valid_token_types = ('access_token', 'refresh_token')
    valid_request_methods = ('POST',)

    def __init__(self, request_validator, supported_token_types=None, enable_jsonp=False):
        if False:
            for i in range(10):
                print('nop')
        BaseEndpoint.__init__(self)
        self.request_validator = request_validator
        self.supported_token_types = supported_token_types or self.valid_token_types
        self.enable_jsonp = enable_jsonp

    @catch_errors_and_unavailability
    def create_revocation_response(self, uri, http_method='POST', body=None, headers=None):
        if False:
            for i in range(10):
                print('nop')
        'Revoke supplied access or refresh token.\n\n\n        The authorization server responds with HTTP status code 200 if the\n        token has been revoked sucessfully or if the client submitted an\n        invalid token.\n\n        Note: invalid tokens do not cause an error response since the client\n        cannot handle such an error in a reasonable way.  Moreover, the purpose\n        of the revocation request, invalidating the particular token, is\n        already achieved.\n\n        The content of the response body is ignored by the client as all\n        necessary information is conveyed in the response code.\n\n        An invalid token type hint value is ignored by the authorization server\n        and does not influence the revocation response.\n        '
        resp_headers = {'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'Pragma': 'no-cache'}
        request = Request(uri, http_method=http_method, body=body, headers=headers)
        try:
            self.validate_revocation_request(request)
            log.debug('Token revocation valid for %r.', request)
        except OAuth2Error as e:
            log.debug('Client error during validation of %r. %r.', request, e)
            response_body = e.json
            if self.enable_jsonp and request.callback:
                response_body = '{}({});'.format(request.callback, response_body)
            resp_headers.update(e.headers)
            return (resp_headers, response_body, e.status_code)
        self.request_validator.revoke_token(request.token, request.token_type_hint, request)
        response_body = ''
        if self.enable_jsonp and request.callback:
            response_body = request.callback + '();'
        return ({}, response_body, 200)

    def validate_revocation_request(self, request):
        if False:
            while True:
                i = 10
        'Ensure the request is valid.\n\n        The client constructs the request by including the following parameters\n        using the "application/x-www-form-urlencoded" format in the HTTP\n        request entity-body:\n\n        token (REQUIRED).  The token that the client wants to get revoked.\n\n        token_type_hint (OPTIONAL).  A hint about the type of the token\n        submitted for revocation.  Clients MAY pass this parameter in order to\n        help the authorization server to optimize the token lookup.  If the\n        server is unable to locate the token using the given hint, it MUST\n        extend its search accross all of its supported token types.  An\n        authorization server MAY ignore this parameter, particularly if it is\n        able to detect the token type automatically.  This specification\n        defines two such values:\n\n                *  access_token: An Access Token as defined in [RFC6749],\n                    `section 1.4`_\n\n                *  refresh_token: A Refresh Token as defined in [RFC6749],\n                    `section 1.5`_\n\n                Specific implementations, profiles, and extensions of this\n                specification MAY define other values for this parameter using\n                the registry defined in `Section 4.1.2`_.\n\n        The client also includes its authentication credentials as described in\n        `Section 2.3`_. of [`RFC6749`_].\n\n        .. _`section 1.4`: https://tools.ietf.org/html/rfc6749#section-1.4\n        .. _`section 1.5`: https://tools.ietf.org/html/rfc6749#section-1.5\n        .. _`section 2.3`: https://tools.ietf.org/html/rfc6749#section-2.3\n        .. _`Section 4.1.2`: https://tools.ietf.org/html/draft-ietf-oauth-revocation-11#section-4.1.2\n        .. _`RFC6749`: https://tools.ietf.org/html/rfc6749\n        '
        self._raise_on_bad_method(request)
        self._raise_on_bad_post_request(request)
        self._raise_on_missing_token(request)
        self._raise_on_invalid_client(request)
        self._raise_on_unsupported_token(request)