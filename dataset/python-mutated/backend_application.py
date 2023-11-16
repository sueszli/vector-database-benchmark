"""
oauthlib.oauth2.rfc6749
~~~~~~~~~~~~~~~~~~~~~~~

This module is an implementation of various logic needed
for consuming and providing OAuth 2.0 RFC6749.
"""
from ..parameters import prepare_token_request
from .base import Client

class BackendApplicationClient(Client):
    """A public client utilizing the client credentials grant workflow.

    The client can request an access token using only its client
    credentials (or other supported means of authentication) when the
    client is requesting access to the protected resources under its
    control, or those of another resource owner which has been previously
    arranged with the authorization server (the method of which is beyond
    the scope of this specification).

    The client credentials grant type MUST only be used by confidential
    clients.

    Since the client authentication is used as the authorization grant,
    no additional authorization request is needed.
    """
    grant_type = 'client_credentials'

    def prepare_request_body(self, body='', scope=None, include_client_id=False, **kwargs):
        if False:
            i = 10
            return i + 15
        'Add the client credentials to the request body.\n\n        The client makes a request to the token endpoint by adding the\n        following parameters using the "application/x-www-form-urlencoded"\n        format per `Appendix B`_ in the HTTP request entity-body:\n\n        :param body: Existing request body (URL encoded string) to embed parameters\n                     into. This may contain extra paramters. Default \'\'.\n        :param scope:   The scope of the access request as described by\n                        `Section 3.3`_.\n\n        :param include_client_id: `True` to send the `client_id` in the\n                                  body of the upstream request. This is required\n                                  if the client is not authenticating with the\n                                  authorization server as described in\n                                  `Section 3.2.1`_. False otherwise (default).\n        :type include_client_id: Boolean\n\n        :param kwargs:  Extra credentials to include in the token request.\n\n        The client MUST authenticate with the authorization server as\n        described in `Section 3.2.1`_.\n\n        The prepared body will include all provided credentials as well as\n        the ``grant_type`` parameter set to ``client_credentials``::\n\n            >>> from oauthlib.oauth2 import BackendApplicationClient\n            >>> client = BackendApplicationClient(\'your_id\')\n            >>> client.prepare_request_body(scope=[\'hello\', \'world\'])\n            \'grant_type=client_credentials&scope=hello+world\'\n\n        .. _`Appendix B`: https://tools.ietf.org/html/rfc6749#appendix-B\n        .. _`Section 3.3`: https://tools.ietf.org/html/rfc6749#section-3.3\n        .. _`Section 3.2.1`: https://tools.ietf.org/html/rfc6749#section-3.2.1\n        '
        kwargs['client_id'] = self.client_id
        kwargs['include_client_id'] = include_client_id
        scope = self.scope if scope is None else scope
        return prepare_token_request(self.grant_type, body=body, scope=scope, **kwargs)