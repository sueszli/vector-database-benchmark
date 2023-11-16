"""
oauthlib.openid.connect.core.request_validator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import logging
from oauthlib.oauth2.rfc6749.request_validator import RequestValidator as OAuth2RequestValidator
log = logging.getLogger(__name__)

class RequestValidator(OAuth2RequestValidator):

    def get_authorization_code_scopes(self, client_id, code, redirect_uri, request):
        if False:
            while True:
                i = 10
        ' Extracts scopes from saved authorization code.\n\n        The scopes returned by this method is used to route token requests\n        based on scopes passed to Authorization Code requests.\n\n        With that the token endpoint knows when to include OpenIDConnect\n        id_token in token response only based on authorization code scopes.\n\n        Only code param should be sufficient to retrieve grant code from\n        any storage you are using, `client_id` and `redirect_uri` can have a\n        blank value `""` don\'t forget to check it before using those values\n        in a select query if a database is used.\n\n        :param client_id: Unicode client identifier\n        :param code: Unicode authorization code grant\n        :param redirect_uri: Unicode absolute URI\n        :return: A list of scope\n\n        Method is used by:\n            - Authorization Token Grant Dispatcher\n        '
        raise NotImplementedError('Subclasses must implement this method.')

    def get_authorization_code_nonce(self, client_id, code, redirect_uri, request):
        if False:
            for i in range(10):
                print('nop')
        ' Extracts nonce from saved authorization code.\n\n        If present in the Authentication Request, Authorization\n        Servers MUST include a nonce Claim in the ID Token with the\n        Claim Value being the nonce value sent in the Authentication\n        Request. Authorization Servers SHOULD perform no other\n        processing on nonce values used. The nonce value is a\n        case-sensitive string.\n\n        Only code param should be sufficient to retrieve grant code from\n        any storage you are using. However, `client_id` and `redirect_uri`\n        have been validated and can be used also.\n\n        :param client_id: Unicode client identifier\n        :param code: Unicode authorization code grant\n        :param redirect_uri: Unicode absolute URI\n        :return: Unicode nonce\n\n        Method is used by:\n            - Authorization Token Grant Dispatcher\n        '
        raise NotImplementedError('Subclasses must implement this method.')

    def get_jwt_bearer_token(self, token, token_handler, request):
        if False:
            for i in range(10):
                print('nop')
        'Get JWT Bearer token or OpenID Connect ID token\n\n        If using OpenID Connect this SHOULD call `oauthlib.oauth2.RequestValidator.get_id_token`\n\n        :param token: A Bearer token dict\n        :param token_handler: the token handler (BearerToken class)\n        :param request: OAuthlib request.\n        :type request: oauthlib.common.Request\n        :return: The JWT Bearer token or OpenID Connect ID token (a JWS signed JWT)\n\n        Method is used by JWT Bearer and OpenID Connect tokens:\n            - JWTToken.create_token\n        '
        raise NotImplementedError('Subclasses must implement this method.')

    def get_id_token(self, token, token_handler, request):
        if False:
            while True:
                i = 10
        'Get OpenID Connect ID token\n\n        This method is OPTIONAL and is NOT RECOMMENDED.\n        `finalize_id_token` SHOULD be implemented instead. However, if you\n        want a full control over the minting of the `id_token`, you\n        MAY want to override `get_id_token` instead of using\n        `finalize_id_token`.\n\n        In the OpenID Connect workflows when an ID Token is requested this method is called.\n        Subclasses should implement the construction, signing and optional encryption of the\n        ID Token as described in the OpenID Connect spec.\n\n        In addition to the standard OAuth2 request properties, the request may also contain\n        these OIDC specific properties which are useful to this method:\n\n            - nonce, if workflow is implicit or hybrid and it was provided\n            - claims, if provided to the original Authorization Code request\n\n        The token parameter is a dict which may contain an ``access_token`` entry, in which\n        case the resulting ID Token *should* include a calculated ``at_hash`` claim.\n\n        Similarly, when the request parameter has a ``code`` property defined, the ID Token\n        *should* include a calculated ``c_hash`` claim.\n\n        http://openid.net/specs/openid-connect-core-1_0.html (sections `3.1.3.6`_, `3.2.2.10`_, `3.3.2.11`_)\n\n        .. _`3.1.3.6`: http://openid.net/specs/openid-connect-core-1_0.html#CodeIDToken\n        .. _`3.2.2.10`: http://openid.net/specs/openid-connect-core-1_0.html#ImplicitIDToken\n        .. _`3.3.2.11`: http://openid.net/specs/openid-connect-core-1_0.html#HybridIDToken\n\n        :param token: A Bearer token dict\n        :param token_handler: the token handler (BearerToken class)\n        :param request: OAuthlib request.\n        :type request: oauthlib.common.Request\n        :return: The ID Token (a JWS signed JWT)\n        '
        return None

    def finalize_id_token(self, id_token, token, token_handler, request):
        if False:
            print('Hello World!')
        'Finalize OpenID Connect ID token & Sign or Encrypt.\n\n        In the OpenID Connect workflows when an ID Token is requested\n        this method is called.  Subclasses should implement the\n        construction, signing and optional encryption of the ID Token\n        as described in the OpenID Connect spec.\n\n        The `id_token` parameter is a dict containing a couple of OIDC\n        technical fields related to the specification. Prepopulated\n        attributes are:\n\n        - `aud`, equals to `request.client_id`.\n        - `iat`, equals to current time.\n        - `nonce`, if present, is equals to the `nonce` from the\n          authorization request.\n        - `at_hash`, hash of `access_token`, if relevant.\n        - `c_hash`, hash of `code`, if relevant.\n\n        This method MUST provide required fields as below:\n\n        - `iss`, REQUIRED. Issuer Identifier for the Issuer of the response.\n        - `sub`, REQUIRED. Subject Identifier\n        - `exp`, REQUIRED. Expiration time on or after which the ID\n          Token MUST NOT be accepted by the RP when performing\n          authentication with the OP.\n\n        Additionals claims must be added, note that `request.scope`\n        should be used to determine the list of claims.\n\n        More information can be found at `OpenID Connect Core#Claims`_\n\n        .. _`OpenID Connect Core#Claims`: https://openid.net/specs/openid-connect-core-1_0.html#Claims\n\n        :param id_token: A dict containing technical fields of id_token\n        :param token: A Bearer token dict\n        :param token_handler: the token handler (BearerToken class)\n        :param request: OAuthlib request.\n        :type request: oauthlib.common.Request\n        :return: The ID Token (a JWS signed JWT or JWE encrypted JWT)\n        '
        raise NotImplementedError('Subclasses must implement this method.')

    def validate_jwt_bearer_token(self, token, scopes, request):
        if False:
            i = 10
            return i + 15
        'Ensure the JWT Bearer token or OpenID Connect ID token are valids and authorized access to scopes.\n\n        If using OpenID Connect this SHOULD call `oauthlib.oauth2.RequestValidator.get_id_token`\n\n        If not using OpenID Connect this can `return None` to avoid 5xx rather 401/3 response.\n\n        OpenID connect core 1.0 describe how to validate an id_token:\n            - http://openid.net/specs/openid-connect-core-1_0.html#IDTokenValidation\n            - http://openid.net/specs/openid-connect-core-1_0.html#ImplicitIDTValidation\n            - http://openid.net/specs/openid-connect-core-1_0.html#HybridIDTValidation\n            - http://openid.net/specs/openid-connect-core-1_0.html#HybridIDTValidation2\n\n        :param token: Unicode Bearer token\n        :param scopes: List of scopes (defined by you)\n        :param request: OAuthlib request.\n        :type request: oauthlib.common.Request\n        :rtype: True or False\n\n        Method is indirectly used by all core OpenID connect JWT token issuing grant types:\n            - Authorization Code Grant\n            - Implicit Grant\n            - Hybrid Grant\n        '
        raise NotImplementedError('Subclasses must implement this method.')

    def validate_id_token(self, token, scopes, request):
        if False:
            while True:
                i = 10
        'Ensure the id token is valid and authorized access to scopes.\n\n        OpenID connect core 1.0 describe how to validate an id_token:\n            - http://openid.net/specs/openid-connect-core-1_0.html#IDTokenValidation\n            - http://openid.net/specs/openid-connect-core-1_0.html#ImplicitIDTValidation\n            - http://openid.net/specs/openid-connect-core-1_0.html#HybridIDTValidation\n            - http://openid.net/specs/openid-connect-core-1_0.html#HybridIDTValidation2\n\n        :param token: Unicode Bearer token\n        :param scopes: List of scopes (defined by you)\n        :param request: OAuthlib request.\n        :type request: oauthlib.common.Request\n        :rtype: True or False\n\n        Method is indirectly used by all core OpenID connect JWT token issuing grant types:\n            - Authorization Code Grant\n            - Implicit Grant\n            - Hybrid Grant\n        '
        raise NotImplementedError('Subclasses must implement this method.')

    def validate_silent_authorization(self, request):
        if False:
            while True:
                i = 10
        'Ensure the logged in user has authorized silent OpenID authorization.\n\n        Silent OpenID authorization allows access tokens and id tokens to be\n        granted to clients without any user prompt or interaction.\n\n        :param request: OAuthlib request.\n        :type request: oauthlib.common.Request\n        :rtype: True or False\n\n        Method is used by:\n            - OpenIDConnectAuthCode\n            - OpenIDConnectImplicit\n            - OpenIDConnectHybrid\n        '
        raise NotImplementedError('Subclasses must implement this method.')

    def validate_silent_login(self, request):
        if False:
            while True:
                i = 10
        'Ensure session user has authorized silent OpenID login.\n\n        If no user is logged in or has not authorized silent login, this\n        method should return False.\n\n        If the user is logged in but associated with multiple accounts and\n        not selected which one to link to the token then this method should\n        raise an oauthlib.oauth2.AccountSelectionRequired error.\n\n        :param request: OAuthlib request.\n        :type request: oauthlib.common.Request\n        :rtype: True or False\n\n        Method is used by:\n            - OpenIDConnectAuthCode\n            - OpenIDConnectImplicit\n            - OpenIDConnectHybrid\n        '
        raise NotImplementedError('Subclasses must implement this method.')

    def validate_user_match(self, id_token_hint, scopes, claims, request):
        if False:
            while True:
                i = 10
        'Ensure client supplied user id hint matches session user.\n\n        If the sub claim or id_token_hint is supplied then the session\n        user must match the given ID.\n\n        :param id_token_hint: User identifier string.\n        :param scopes: List of OAuth 2 scopes and OpenID claims (strings).\n        :param claims: OpenID Connect claims dict.\n        :param request: OAuthlib request.\n        :type request: oauthlib.common.Request\n        :rtype: True or False\n\n        Method is used by:\n            - OpenIDConnectAuthCode\n            - OpenIDConnectImplicit\n            - OpenIDConnectHybrid\n        '
        raise NotImplementedError('Subclasses must implement this method.')

    def get_userinfo_claims(self, request):
        if False:
            return 10
        "Return the UserInfo claims in JSON or Signed or Encrypted.\n\n        The UserInfo Claims MUST be returned as the members of a JSON object\n         unless a signed or encrypted response was requested during Client\n         Registration. The Claims defined in Section 5.1 can be returned, as can\n         additional Claims not specified there.\n\n        For privacy reasons, OpenID Providers MAY elect to not return values for\n        some requested Claims.\n\n        If a Claim is not returned, that Claim Name SHOULD be omitted from the\n        JSON object representing the Claims; it SHOULD NOT be present with a\n        null or empty string value.\n\n        The sub (subject) Claim MUST always be returned in the UserInfo\n        Response.\n\n        Upon receipt of the UserInfo Request, the UserInfo Endpoint MUST return\n        the JSON Serialization of the UserInfo Response as in Section 13.3 in\n        the HTTP response body unless a different format was specified during\n        Registration [OpenID.Registration].\n\n        If the UserInfo Response is signed and/or encrypted, then the Claims are\n        returned in a JWT and the content-type MUST be application/jwt. The\n        response MAY be encrypted without also being signed. If both signing and\n        encryption are requested, the response MUST be signed then encrypted,\n        with the result being a Nested JWT, as defined in [JWT].\n\n        If signed, the UserInfo Response SHOULD contain the Claims iss (issuer)\n        and aud (audience) as members. The iss value SHOULD be the OP's Issuer\n        Identifier URL. The aud value SHOULD be or include the RP's Client ID\n        value.\n\n        :param request: OAuthlib request.\n        :type request: oauthlib.common.Request\n        :rtype: Claims as a dict OR JWT/JWS/JWE as a string\n\n        Method is used by:\n            UserInfoEndpoint\n        "

    def refresh_id_token(self, request):
        if False:
            return 10
        'Whether the id token should be refreshed. Default, True\n\n        :param request: OAuthlib request.\n        :type request: oauthlib.common.Request\n        :rtype: True or False\n\n        Method is used by:\n            RefreshTokenGrant\n        '
        return True