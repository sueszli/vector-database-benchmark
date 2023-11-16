"""
oauthlib.oauth2.rfc6749
~~~~~~~~~~~~~~~~~~~~~~~

This module is an implementation of various logic needed
for consuming and providing OAuth 2.0 RFC6749.
"""
from ..parameters import prepare_token_request
from .base import Client

class LegacyApplicationClient(Client):
    """A public client using the resource owner password and username directly.

    The resource owner password credentials grant type is suitable in
    cases where the resource owner has a trust relationship with the
    client, such as the device operating system or a highly privileged
    application.  The authorization server should take special care when
    enabling this grant type, and only allow it when other flows are not
    viable.

    The grant type is suitable for clients capable of obtaining the
    resource owner's credentials (username and password, typically using
    an interactive form).  It is also used to migrate existing clients
    using direct authentication schemes such as HTTP Basic or Digest
    authentication to OAuth by converting the stored credentials to an
    access token.

    The method through which the client obtains the resource owner
    credentials is beyond the scope of this specification.  The client
    MUST discard the credentials once an access token has been obtained.
    """
    grant_type = 'password'

    def __init__(self, client_id, **kwargs):
        if False:
            return 10
        super().__init__(client_id, **kwargs)

    def prepare_request_body(self, username, password, body='', scope=None, include_client_id=False, **kwargs):
        if False:
            i = 10
            return i + 15
        'Add the resource owner password and username to the request body.\n\n        The client makes a request to the token endpoint by adding the\n        following parameters using the "application/x-www-form-urlencoded"\n        format per `Appendix B`_ in the HTTP request entity-body:\n\n        :param username:    The resource owner username.\n        :param password:    The resource owner password.\n        :param body: Existing request body (URL encoded string) to embed parameters\n                     into. This may contain extra paramters. Default \'\'.\n        :param scope:   The scope of the access request as described by\n                        `Section 3.3`_.\n        :param include_client_id: `True` to send the `client_id` in the\n                                  body of the upstream request. This is required\n                                  if the client is not authenticating with the\n                                  authorization server as described in\n                                  `Section 3.2.1`_. False otherwise (default).\n        :type include_client_id: Boolean\n        :param kwargs:  Extra credentials to include in the token request.\n\n        If the client type is confidential or the client was issued client\n        credentials (or assigned other authentication requirements), the\n        client MUST authenticate with the authorization server as described\n        in `Section 3.2.1`_.\n\n        The prepared body will include all provided credentials as well as\n        the ``grant_type`` parameter set to ``password``::\n\n            >>> from oauthlib.oauth2 import LegacyApplicationClient\n            >>> client = LegacyApplicationClient(\'your_id\')\n            >>> client.prepare_request_body(username=\'foo\', password=\'bar\', scope=[\'hello\', \'world\'])\n            \'grant_type=password&username=foo&scope=hello+world&password=bar\'\n\n        .. _`Appendix B`: https://tools.ietf.org/html/rfc6749#appendix-B\n        .. _`Section 3.3`: https://tools.ietf.org/html/rfc6749#section-3.3\n        .. _`Section 3.2.1`: https://tools.ietf.org/html/rfc6749#section-3.2.1\n        '
        kwargs['client_id'] = self.client_id
        kwargs['include_client_id'] = include_client_id
        scope = self.scope if scope is None else scope
        return prepare_token_request(self.grant_type, body=body, username=username, password=password, scope=scope, **kwargs)