"""
oauthlib.openid.connect.core.endpoints.userinfo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module is an implementation of userinfo endpoint.
"""
import json
import logging
from oauthlib.common import Request
from oauthlib.oauth2.rfc6749 import errors
from oauthlib.oauth2.rfc6749.endpoints.base import BaseEndpoint, catch_errors_and_unavailability
from oauthlib.oauth2.rfc6749.tokens import BearerToken
log = logging.getLogger(__name__)

class UserInfoEndpoint(BaseEndpoint):
    """Authorizes access to userinfo resource.
    """

    def __init__(self, request_validator):
        if False:
            i = 10
            return i + 15
        self.bearer = BearerToken(request_validator, None, None, None)
        self.request_validator = request_validator
        BaseEndpoint.__init__(self)

    @catch_errors_and_unavailability
    def create_userinfo_response(self, uri, http_method='GET', body=None, headers=None):
        if False:
            print('Hello World!')
        'Validate BearerToken and return userinfo from RequestValidator\n\n        The UserInfo Endpoint MUST return a\n        content-type header to indicate which format is being returned. The\n        content-type of the HTTP response MUST be application/json if the\n        response body is a text JSON object; the response body SHOULD be encoded\n        using UTF-8.\n        '
        request = Request(uri, http_method, body, headers)
        request.scopes = ['openid']
        self.validate_userinfo_request(request)
        claims = self.request_validator.get_userinfo_claims(request)
        if claims is None:
            log.error('Userinfo MUST have claims for %r.', request)
            raise errors.ServerError(status_code=500)
        if isinstance(claims, dict):
            resp_headers = {'Content-Type': 'application/json'}
            if 'sub' not in claims:
                log.error('Userinfo MUST have "sub" for %r.', request)
                raise errors.ServerError(status_code=500)
            body = json.dumps(claims)
        elif isinstance(claims, str):
            resp_headers = {'Content-Type': 'application/jwt'}
            body = claims
        else:
            log.error('Userinfo return unknown response for %r.', request)
            raise errors.ServerError(status_code=500)
        log.debug('Userinfo access valid for %r.', request)
        return (resp_headers, body, 200)

    def validate_userinfo_request(self, request):
        if False:
            print('Hello World!')
        'Ensure the request is valid.\n\n        5.3.1.  UserInfo Request\n        The Client sends the UserInfo Request using either HTTP GET or HTTP\n        POST. The Access Token obtained from an OpenID Connect Authentication\n        Request MUST be sent as a Bearer Token, per Section 2 of OAuth 2.0\n        Bearer Token Usage [RFC6750].\n\n        It is RECOMMENDED that the request use the HTTP GET method and the\n        Access Token be sent using the Authorization header field.\n\n        The following is a non-normative example of a UserInfo Request:\n\n        GET /userinfo HTTP/1.1\n        Host: server.example.com\n        Authorization: Bearer SlAV32hkKG\n\n        5.3.3. UserInfo Error Response\n        When an error condition occurs, the UserInfo Endpoint returns an Error\n        Response as defined in Section 3 of OAuth 2.0 Bearer Token Usage\n        [RFC6750]. (HTTP errors unrelated to RFC 6750 are returned to the User\n        Agent using the appropriate HTTP status code.)\n\n        The following is a non-normative example of a UserInfo Error Response:\n\n        HTTP/1.1 401 Unauthorized\n        WWW-Authenticate: Bearer error="invalid_token",\n                error_description="The Access Token expired"\n        '
        if not self.bearer.validate_request(request):
            raise errors.InvalidTokenError()
        if 'openid' not in request.scopes:
            raise errors.InsufficientScopeError()