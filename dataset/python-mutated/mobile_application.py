"""
oauthlib.oauth2.rfc6749
~~~~~~~~~~~~~~~~~~~~~~~

This module is an implementation of various logic needed
for consuming and providing OAuth 2.0 RFC6749.
"""
from ..parameters import parse_implicit_response, prepare_grant_uri
from .base import Client

class MobileApplicationClient(Client):
    """A public client utilizing the implicit code grant workflow.

    A user-agent-based application is a public client in which the
    client code is downloaded from a web server and executes within a
    user-agent (e.g. web browser) on the device used by the resource
    owner.  Protocol data and credentials are easily accessible (and
    often visible) to the resource owner.  Since such applications
    reside within the user-agent, they can make seamless use of the
    user-agent capabilities when requesting authorization.

    The implicit grant type is used to obtain access tokens (it does not
    support the issuance of refresh tokens) and is optimized for public
    clients known to operate a particular redirection URI.  These clients
    are typically implemented in a browser using a scripting language
    such as JavaScript.

    As a redirection-based flow, the client must be capable of
    interacting with the resource owner's user-agent (typically a web
    browser) and capable of receiving incoming requests (via redirection)
    from the authorization server.

    Unlike the authorization code grant type in which the client makes
    separate requests for authorization and access token, the client
    receives the access token as the result of the authorization request.

    The implicit grant type does not include client authentication, and
    relies on the presence of the resource owner and the registration of
    the redirection URI.  Because the access token is encoded into the
    redirection URI, it may be exposed to the resource owner and other
    applications residing on the same device.
    """
    response_type = 'token'

    def prepare_request_uri(self, uri, redirect_uri=None, scope=None, state=None, **kwargs):
        if False:
            return 10
        'Prepare the implicit grant request URI.\n\n        The client constructs the request URI by adding the following\n        parameters to the query component of the authorization endpoint URI\n        using the "application/x-www-form-urlencoded" format, per `Appendix B`_:\n\n        :param redirect_uri:  OPTIONAL. The redirect URI must be an absolute URI\n                              and it should have been registerd with the OAuth\n                              provider prior to use. As described in `Section 3.1.2`_.\n\n        :param scope:  OPTIONAL. The scope of the access request as described by\n                       Section 3.3`_. These may be any string but are commonly\n                       URIs or various categories such as ``videos`` or ``documents``.\n\n        :param state:   RECOMMENDED.  An opaque value used by the client to maintain\n                        state between the request and callback.  The authorization\n                        server includes this value when redirecting the user-agent back\n                        to the client.  The parameter SHOULD be used for preventing\n                        cross-site request forgery as described in `Section 10.12`_.\n\n        :param kwargs:  Extra arguments to include in the request URI.\n\n        In addition to supplied parameters, OAuthLib will append the ``client_id``\n        that was provided in the constructor as well as the mandatory ``response_type``\n        argument, set to ``token``::\n\n            >>> from oauthlib.oauth2 import MobileApplicationClient\n            >>> client = MobileApplicationClient(\'your_id\')\n            >>> client.prepare_request_uri(\'https://example.com\')\n            \'https://example.com?client_id=your_id&response_type=token\'\n            >>> client.prepare_request_uri(\'https://example.com\', redirect_uri=\'https://a.b/callback\')\n            \'https://example.com?client_id=your_id&response_type=token&redirect_uri=https%3A%2F%2Fa.b%2Fcallback\'\n            >>> client.prepare_request_uri(\'https://example.com\', scope=[\'profile\', \'pictures\'])\n            \'https://example.com?client_id=your_id&response_type=token&scope=profile+pictures\'\n            >>> client.prepare_request_uri(\'https://example.com\', foo=\'bar\')\n            \'https://example.com?client_id=your_id&response_type=token&foo=bar\'\n\n        .. _`Appendix B`: https://tools.ietf.org/html/rfc6749#appendix-B\n        .. _`Section 2.2`: https://tools.ietf.org/html/rfc6749#section-2.2\n        .. _`Section 3.1.2`: https://tools.ietf.org/html/rfc6749#section-3.1.2\n        .. _`Section 3.3`: https://tools.ietf.org/html/rfc6749#section-3.3\n        .. _`Section 10.12`: https://tools.ietf.org/html/rfc6749#section-10.12\n        '
        scope = self.scope if scope is None else scope
        return prepare_grant_uri(uri, self.client_id, self.response_type, redirect_uri=redirect_uri, state=state, scope=scope, **kwargs)

    def parse_request_uri_response(self, uri, state=None, scope=None):
        if False:
            i = 10
            return i + 15
        'Parse the response URI fragment.\n\n        If the resource owner grants the access request, the authorization\n        server issues an access token and delivers it to the client by adding\n        the following parameters to the fragment component of the redirection\n        URI using the "application/x-www-form-urlencoded" format:\n\n        :param uri: The callback URI that resulted from the user being redirected\n                    back from the provider to you, the client.\n        :param state: The state provided in the authorization request.\n        :param scope: The scopes provided in the authorization request.\n        :return: Dictionary of token parameters.\n        :raises: OAuth2Error if response is invalid.\n\n        A successful response should always contain\n\n        **access_token**\n                The access token issued by the authorization server. Often\n                a random string.\n\n        **token_type**\n            The type of the token issued as described in `Section 7.1`_.\n            Commonly ``Bearer``.\n\n        **state**\n            If you provided the state parameter in the authorization phase, then\n            the provider is required to include that exact state value in the\n            response.\n\n        While it is not mandated it is recommended that the provider include\n\n        **expires_in**\n            The lifetime in seconds of the access token.  For\n            example, the value "3600" denotes that the access token will\n            expire in one hour from the time the response was generated.\n            If omitted, the authorization server SHOULD provide the\n            expiration time via other means or document the default value.\n\n        **scope**\n            Providers may supply this in all responses but are required to only\n            if it has changed since the authorization request.\n\n        A few example responses can be seen below::\n\n            >>> response_uri = \'https://example.com/callback#access_token=sdlfkj452&state=ss345asyht&token_type=Bearer&scope=hello+world\'\n            >>> from oauthlib.oauth2 import MobileApplicationClient\n            >>> client = MobileApplicationClient(\'your_id\')\n            >>> client.parse_request_uri_response(response_uri)\n            {\n                \'access_token\': \'sdlfkj452\',\n                \'token_type\': \'Bearer\',\n                \'state\': \'ss345asyht\',\n                \'scope\': [u\'hello\', u\'world\']\n            }\n            >>> client.parse_request_uri_response(response_uri, state=\'other\')\n            Traceback (most recent call last):\n                File "<stdin>", line 1, in <module>\n                File "oauthlib/oauth2/rfc6749/__init__.py", line 598, in parse_request_uri_response\n                    **scope**\n                File "oauthlib/oauth2/rfc6749/parameters.py", line 197, in parse_implicit_response\n                    raise ValueError("Mismatching or missing state in params.")\n            ValueError: Mismatching or missing state in params.\n            >>> def alert_scope_changed(message, old, new):\n            ...     print(message, old, new)\n            ...\n            >>> oauthlib.signals.scope_changed.connect(alert_scope_changed)\n            >>> client.parse_request_body_response(response_body, scope=[\'other\'])\n            (\'Scope has changed from "other" to "hello world".\', [\'other\'], [\'hello\', \'world\'])\n\n        .. _`Section 7.1`: https://tools.ietf.org/html/rfc6749#section-7.1\n        .. _`Section 3.3`: https://tools.ietf.org/html/rfc6749#section-3.3\n        '
        scope = self.scope if scope is None else scope
        self.token = parse_implicit_response(uri, state=state, scope=scope)
        self.populate_token_attributes(self.token)
        return self.token