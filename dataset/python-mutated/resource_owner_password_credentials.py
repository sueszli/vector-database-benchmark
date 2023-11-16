"""
oauthlib.oauth2.rfc6749.grant_types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import json
import logging
from .. import errors
from .base import GrantTypeBase
log = logging.getLogger(__name__)

class ResourceOwnerPasswordCredentialsGrant(GrantTypeBase):
    """`Resource Owner Password Credentials Grant`_

    The resource owner password credentials grant type is suitable in
    cases where the resource owner has a trust relationship with the
    client, such as the device operating system or a highly privileged
    application.  The authorization server should take special care when
    enabling this grant type and only allow it when other flows are not
    viable.

    This grant type is suitable for clients capable of obtaining the
    resource owner's credentials (username and password, typically using
    an interactive form).  It is also used to migrate existing clients
    using direct authentication schemes such as HTTP Basic or Digest
    authentication to OAuth by converting the stored credentials to an
    access token::

            +----------+
            | Resource |
            |  Owner   |
            |          |
            +----------+
                 v
                 |    Resource Owner
                (A) Password Credentials
                 |
                 v
            +---------+                                  +---------------+
            |         |>--(B)---- Resource Owner ------->|               |
            |         |         Password Credentials     | Authorization |
            | Client  |                                  |     Server    |
            |         |<--(C)---- Access Token ---------<|               |
            |         |    (w/ Optional Refresh Token)   |               |
            +---------+                                  +---------------+

    Figure 5: Resource Owner Password Credentials Flow

    The flow illustrated in Figure 5 includes the following steps:

    (A)  The resource owner provides the client with its username and
            password.

    (B)  The client requests an access token from the authorization
            server's token endpoint by including the credentials received
            from the resource owner.  When making the request, the client
            authenticates with the authorization server.

    (C)  The authorization server authenticates the client and validates
            the resource owner credentials, and if valid, issues an access
            token.

    .. _`Resource Owner Password Credentials Grant`: https://tools.ietf.org/html/rfc6749#section-4.3
    """

    def create_token_response(self, request, token_handler):
        if False:
            while True:
                i = 10
        'Return token or error in json format.\n\n        :param request: OAuthlib request.\n        :type request: oauthlib.common.Request\n        :param token_handler: A token handler instance, for example of type\n                              oauthlib.oauth2.BearerToken.\n\n        If the access token request is valid and authorized, the\n        authorization server issues an access token and optional refresh\n        token as described in `Section 5.1`_.  If the request failed client\n        authentication or is invalid, the authorization server returns an\n        error response as described in `Section 5.2`_.\n\n        .. _`Section 5.1`: https://tools.ietf.org/html/rfc6749#section-5.1\n        .. _`Section 5.2`: https://tools.ietf.org/html/rfc6749#section-5.2\n        '
        headers = self._get_default_headers()
        try:
            if self.request_validator.client_authentication_required(request):
                log.debug('Authenticating client, %r.', request)
                if not self.request_validator.authenticate_client(request):
                    log.debug('Client authentication failed, %r.', request)
                    raise errors.InvalidClientError(request=request)
            elif not self.request_validator.authenticate_client_id(request.client_id, request):
                log.debug('Client authentication failed, %r.', request)
                raise errors.InvalidClientError(request=request)
            log.debug('Validating access token request, %r.', request)
            self.validate_token_request(request)
        except errors.OAuth2Error as e:
            log.debug('Client error in token request, %s.', e)
            headers.update(e.headers)
            return (headers, e.json, e.status_code)
        token = token_handler.create_token(request, self.refresh_token)
        for modifier in self._token_modifiers:
            token = modifier(token)
        self.request_validator.save_token(token, request)
        log.debug('Issuing token %r to client id %r (%r) and username %s.', token, request.client_id, request.client, request.username)
        return (headers, json.dumps(token), 200)

    def validate_token_request(self, request):
        if False:
            return 10
        '\n        :param request: OAuthlib request.\n        :type request: oauthlib.common.Request\n\n        The client makes a request to the token endpoint by adding the\n        following parameters using the "application/x-www-form-urlencoded"\n        format per Appendix B with a character encoding of UTF-8 in the HTTP\n        request entity-body:\n\n        grant_type\n                REQUIRED.  Value MUST be set to "password".\n\n        username\n                REQUIRED.  The resource owner username.\n\n        password\n                REQUIRED.  The resource owner password.\n\n        scope\n                OPTIONAL.  The scope of the access request as described by\n                `Section 3.3`_.\n\n        If the client type is confidential or the client was issued client\n        credentials (or assigned other authentication requirements), the\n        client MUST authenticate with the authorization server as described\n        in `Section 3.2.1`_.\n\n        The authorization server MUST:\n\n        o  require client authentication for confidential clients or for any\n            client that was issued client credentials (or with other\n            authentication requirements),\n\n        o  authenticate the client if client authentication is included, and\n\n        o  validate the resource owner password credentials using its\n            existing password validation algorithm.\n\n        Since this access token request utilizes the resource owner\'s\n        password, the authorization server MUST protect the endpoint against\n        brute force attacks (e.g., using rate-limitation or generating\n        alerts).\n\n        .. _`Section 3.3`: https://tools.ietf.org/html/rfc6749#section-3.3\n        .. _`Section 3.2.1`: https://tools.ietf.org/html/rfc6749#section-3.2.1\n        '
        for validator in self.custom_validators.pre_token:
            validator(request)
        for param in ('grant_type', 'username', 'password'):
            if not getattr(request, param, None):
                raise errors.InvalidRequestError('Request is missing %s parameter.' % param, request=request)
        for param in ('grant_type', 'username', 'password', 'scope'):
            if param in request.duplicate_params:
                raise errors.InvalidRequestError(description='Duplicate %s parameter.' % param, request=request)
        if not request.grant_type == 'password':
            raise errors.UnsupportedGrantTypeError(request=request)
        log.debug('Validating username %s.', request.username)
        if not self.request_validator.validate_user(request.username, request.password, request.client, request):
            raise errors.InvalidGrantError('Invalid credentials given.', request=request)
        elif not hasattr(request.client, 'client_id'):
            raise NotImplementedError('Validate user must set the request.client.client_id attribute in authenticate_client.')
        log.debug('Authorizing access to user %r.', request.user)
        self.validate_grant_type(request)
        if request.client:
            request.client_id = request.client_id or request.client.client_id
        self.validate_scopes(request)
        for validator in self.custom_validators.post_token:
            validator(request)