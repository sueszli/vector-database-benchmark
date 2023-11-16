import pytest
from django.contrib.auth import get_user
from django.contrib.auth.models import AnonymousUser
from django.test import RequestFactory, TestCase
from django.urls import reverse
from django.utils import timezone
from pytest_django.asserts import assertRedirects
from oauth2_provider.exceptions import ClientIdMissmatch, InvalidIDTokenError, InvalidOIDCClientError, InvalidOIDCRedirectURIError
from oauth2_provider.models import get_access_token_model, get_id_token_model, get_refresh_token_model
from oauth2_provider.oauth2_validators import OAuth2Validator
from oauth2_provider.settings import oauth2_settings
from oauth2_provider.views.oidc import RPInitiatedLogoutView, _load_id_token, _validate_claims, validate_logout_request
from . import presets

@pytest.mark.usefixtures('oauth2_settings')
@pytest.mark.oauth2_settings(presets.OIDC_SETTINGS_RW)
class TestConnectDiscoveryInfoView(TestCase):

    def test_get_connect_discovery_info(self):
        if False:
            i = 10
            return i + 15
        expected_response = {'issuer': 'http://localhost/o', 'authorization_endpoint': 'http://localhost/o/authorize/', 'token_endpoint': 'http://localhost/o/token/', 'userinfo_endpoint': 'http://localhost/o/userinfo/', 'jwks_uri': 'http://localhost/o/.well-known/jwks.json', 'scopes_supported': ['read', 'write', 'openid'], 'response_types_supported': ['code', 'token', 'id_token', 'id_token token', 'code token', 'code id_token', 'code id_token token'], 'subject_types_supported': ['public'], 'id_token_signing_alg_values_supported': ['RS256', 'HS256'], 'token_endpoint_auth_methods_supported': ['client_secret_post', 'client_secret_basic'], 'claims_supported': ['sub']}
        response = self.client.get('/o/.well-known/openid-configuration')
        self.assertEqual(response.status_code, 200)
        assert response.json() == expected_response

    def test_get_connect_discovery_info_deprecated(self):
        if False:
            return 10
        expected_response = {'issuer': 'http://localhost/o', 'authorization_endpoint': 'http://localhost/o/authorize/', 'token_endpoint': 'http://localhost/o/token/', 'userinfo_endpoint': 'http://localhost/o/userinfo/', 'jwks_uri': 'http://localhost/o/.well-known/jwks.json', 'scopes_supported': ['read', 'write', 'openid'], 'response_types_supported': ['code', 'token', 'id_token', 'id_token token', 'code token', 'code id_token', 'code id_token token'], 'subject_types_supported': ['public'], 'id_token_signing_alg_values_supported': ['RS256', 'HS256'], 'token_endpoint_auth_methods_supported': ['client_secret_post', 'client_secret_basic'], 'claims_supported': ['sub']}
        response = self.client.get('/o/.well-known/openid-configuration/')
        self.assertEqual(response.status_code, 200)
        assert response.json() == expected_response

    def expect_json_response_with_rp_logout(self, base):
        if False:
            while True:
                i = 10
        expected_response = {'issuer': f'{base}', 'authorization_endpoint': f'{base}/authorize/', 'token_endpoint': f'{base}/token/', 'userinfo_endpoint': f'{base}/userinfo/', 'jwks_uri': f'{base}/.well-known/jwks.json', 'scopes_supported': ['read', 'write', 'openid'], 'response_types_supported': ['code', 'token', 'id_token', 'id_token token', 'code token', 'code id_token', 'code id_token token'], 'subject_types_supported': ['public'], 'id_token_signing_alg_values_supported': ['RS256', 'HS256'], 'token_endpoint_auth_methods_supported': ['client_secret_post', 'client_secret_basic'], 'claims_supported': ['sub'], 'end_session_endpoint': f'{base}/logout/'}
        response = self.client.get(reverse('oauth2_provider:oidc-connect-discovery-info'))
        self.assertEqual(response.status_code, 200)
        assert response.json() == expected_response

    def test_get_connect_discovery_info_with_rp_logout(self):
        if False:
            print('Hello World!')
        self.oauth2_settings.OIDC_RP_INITIATED_LOGOUT_ENABLED = True
        self.expect_json_response_with_rp_logout(self.oauth2_settings.OIDC_ISS_ENDPOINT)

    def test_get_connect_discovery_info_without_issuer_url(self):
        if False:
            return 10
        self.oauth2_settings.OIDC_ISS_ENDPOINT = None
        self.oauth2_settings.OIDC_USERINFO_ENDPOINT = None
        expected_response = {'issuer': 'http://testserver/o', 'authorization_endpoint': 'http://testserver/o/authorize/', 'token_endpoint': 'http://testserver/o/token/', 'userinfo_endpoint': 'http://testserver/o/userinfo/', 'jwks_uri': 'http://testserver/o/.well-known/jwks.json', 'scopes_supported': ['read', 'write', 'openid'], 'response_types_supported': ['code', 'token', 'id_token', 'id_token token', 'code token', 'code id_token', 'code id_token token'], 'subject_types_supported': ['public'], 'id_token_signing_alg_values_supported': ['RS256', 'HS256'], 'token_endpoint_auth_methods_supported': ['client_secret_post', 'client_secret_basic'], 'claims_supported': ['sub']}
        response = self.client.get(reverse('oauth2_provider:oidc-connect-discovery-info'))
        self.assertEqual(response.status_code, 200)
        assert response.json() == expected_response

    def test_get_connect_discovery_info_without_issuer_url_with_rp_logout(self):
        if False:
            for i in range(10):
                print('nop')
        self.oauth2_settings.OIDC_RP_INITIATED_LOGOUT_ENABLED = True
        self.oauth2_settings.OIDC_ISS_ENDPOINT = None
        self.oauth2_settings.OIDC_USERINFO_ENDPOINT = None
        self.expect_json_response_with_rp_logout('http://testserver/o')

    def test_get_connect_discovery_info_without_rsa_key(self):
        if False:
            i = 10
            return i + 15
        self.oauth2_settings.OIDC_RSA_PRIVATE_KEY = None
        response = self.client.get(reverse('oauth2_provider:oidc-connect-discovery-info'))
        self.assertEqual(response.status_code, 200)
        assert response.json()['id_token_signing_alg_values_supported'] == ['HS256']

@pytest.mark.usefixtures('oauth2_settings')
@pytest.mark.oauth2_settings(presets.OIDC_SETTINGS_RW)
class TestJwksInfoView(TestCase):

    def test_get_jwks_info(self):
        if False:
            i = 10
            return i + 15
        self.oauth2_settings.OIDC_RSA_PRIVATE_KEYS_INACTIVE = []
        expected_response = {'keys': [{'alg': 'RS256', 'use': 'sig', 'kid': 's4a1o8mFEd1tATAIH96caMlu4hOxzBUaI2QTqbYNBHs', 'e': 'AQAB', 'kty': 'RSA', 'n': 'mwmIeYdjZkLgalTuhvvwjvnB5vVQc7G9DHgOm20Hw524bLVTk49IXJ2Scw42HOmowWWX-oMVT_ca3ZvVIeffVSN1-TxVy2zB65s0wDMwhiMoPv35z9IKHGMZgl9vlyso_2b7daVF_FQDdgIayUn8TQylBxEU1RFfW0QSYOBdAt8'}]}
        response = self.client.get(reverse('oauth2_provider:jwks-info'))
        self.assertEqual(response.status_code, 200)
        assert response.json() == expected_response

    def test_get_jwks_info_no_rsa_key(self):
        if False:
            return 10
        self.oauth2_settings.OIDC_RSA_PRIVATE_KEY = None
        response = self.client.get(reverse('oauth2_provider:jwks-info'))
        self.assertEqual(response.status_code, 200)
        assert response.json() == {'keys': []}

    def test_get_jwks_info_multiple_rsa_keys(self):
        if False:
            return 10
        expected_response = {'keys': [{'alg': 'RS256', 'e': 'AQAB', 'kid': 's4a1o8mFEd1tATAIH96caMlu4hOxzBUaI2QTqbYNBHs', 'kty': 'RSA', 'n': 'mwmIeYdjZkLgalTuhvvwjvnB5vVQc7G9DHgOm20Hw524bLVTk49IXJ2Scw42HOmowWWX-oMVT_ca3ZvVIeffVSN1-TxVy2zB65s0wDMwhiMoPv35z9IKHGMZgl9vlyso_2b7daVF_FQDdgIayUn8TQylBxEU1RFfW0QSYOBdAt8', 'use': 'sig'}, {'alg': 'RS256', 'e': 'AQAB', 'kid': 'AJ_IkYJUFWqiKKE2FvPIESroTvownbaj0OzL939oIIE', 'kty': 'RSA', 'n': '0qVzbcWg_fgygZ0liTaFeodD2bkinhj8gPJ9P2rPzvqG6ImI9YKkEk8Dxcc7eWcudnw5iEL8wx_tgooaRiHiYfUrFBBXfA15D_15PdX_5gG8rQbJ7XMxQrYoRUcVm2wQDB4fIuR7sTPqx9p8OR4f--BixOfM5Oa7SEUtQ8kvrlE', 'use': 'sig'}]}
        response = self.client.get(reverse('oauth2_provider:jwks-info'))
        self.assertEqual(response.status_code, 200)
        assert response.json() == expected_response

def mock_request():
    if False:
        print('Hello World!')
    '\n    Dummy request with an AnonymousUser attached.\n    '
    return mock_request_for(AnonymousUser())

def mock_request_for(user):
    if False:
        i = 10
        return i + 15
    '\n    Dummy request with the `user` attached.\n    '
    request = RequestFactory().get('')
    request.user = user
    return request

@pytest.mark.django_db
@pytest.mark.parametrize('ALWAYS_PROMPT', [True, False])
def test_deprecated_validate_logout_request(oidc_tokens, public_application, other_user, rp_settings, ALWAYS_PROMPT):
    if False:
        i = 10
        return i + 15
    rp_settings.OIDC_RP_INITIATED_LOGOUT_ALWAYS_PROMPT = ALWAYS_PROMPT
    oidc_tokens = oidc_tokens
    application = oidc_tokens.application
    client_id = application.client_id
    id_token = oidc_tokens.id_token
    assert validate_logout_request(request=mock_request_for(oidc_tokens.user), id_token_hint=None, client_id=None, post_logout_redirect_uri=None) == (True, (None, None), None)
    assert validate_logout_request(request=mock_request_for(oidc_tokens.user), id_token_hint=None, client_id=client_id, post_logout_redirect_uri=None) == (True, (None, application), None)
    assert validate_logout_request(request=mock_request_for(oidc_tokens.user), id_token_hint=None, client_id=client_id, post_logout_redirect_uri='http://example.org') == (True, ('http://example.org', application), None)
    assert validate_logout_request(request=mock_request_for(oidc_tokens.user), id_token_hint=id_token, client_id=None, post_logout_redirect_uri='http://example.org') == (ALWAYS_PROMPT, ('http://example.org', application), oidc_tokens.user)
    assert validate_logout_request(request=mock_request_for(other_user), id_token_hint=id_token, client_id=None, post_logout_redirect_uri='http://example.org') == (True, ('http://example.org', application), oidc_tokens.user)
    assert validate_logout_request(request=mock_request_for(oidc_tokens.user), id_token_hint=id_token, client_id=client_id, post_logout_redirect_uri='http://example.org') == (ALWAYS_PROMPT, ('http://example.org', application), oidc_tokens.user)
    with pytest.raises(InvalidIDTokenError):
        validate_logout_request(request=mock_request_for(oidc_tokens.user), id_token_hint='111', client_id=public_application.client_id, post_logout_redirect_uri='http://other.org')
    with pytest.raises(ClientIdMissmatch):
        validate_logout_request(request=mock_request_for(oidc_tokens.user), id_token_hint=id_token, client_id=public_application.client_id, post_logout_redirect_uri='http://other.org')
    with pytest.raises(InvalidOIDCClientError):
        validate_logout_request(request=mock_request_for(oidc_tokens.user), id_token_hint=None, client_id=None, post_logout_redirect_uri='http://example.org')
    with pytest.raises(InvalidOIDCRedirectURIError):
        validate_logout_request(request=mock_request_for(oidc_tokens.user), id_token_hint=None, client_id=client_id, post_logout_redirect_uri='example.org')
    with pytest.raises(InvalidOIDCRedirectURIError):
        validate_logout_request(request=mock_request_for(oidc_tokens.user), id_token_hint=None, client_id=client_id, post_logout_redirect_uri='imap://example.org')
    with pytest.raises(InvalidOIDCRedirectURIError):
        validate_logout_request(request=mock_request_for(oidc_tokens.user), id_token_hint=None, client_id=client_id, post_logout_redirect_uri='http://other.org')
    with pytest.raises(InvalidOIDCRedirectURIError):
        rp_settings.OIDC_RP_INITIATED_LOGOUT_STRICT_REDIRECT_URIS = True
        validate_logout_request(request=mock_request_for(oidc_tokens.user), id_token_hint=None, client_id=public_application.client_id, post_logout_redirect_uri='http://other.org')

@pytest.mark.django_db
def test_validate_logout_request(oidc_tokens, public_application, rp_settings):
    if False:
        print('Hello World!')
    oidc_tokens = oidc_tokens
    application = oidc_tokens.application
    client_id = application.client_id
    id_token = oidc_tokens.id_token
    view = RPInitiatedLogoutView()
    view.request = mock_request_for(oidc_tokens.user)
    assert view.validate_logout_request(id_token_hint=None, client_id=None, post_logout_redirect_uri=None) == (None, None)
    assert view.validate_logout_request(id_token_hint=None, client_id=client_id, post_logout_redirect_uri=None) == (application, None)
    assert view.validate_logout_request(id_token_hint=None, client_id=client_id, post_logout_redirect_uri='http://example.org') == (application, None)
    assert view.validate_logout_request(id_token_hint=id_token, client_id=None, post_logout_redirect_uri='http://example.org') == (application, oidc_tokens.user)
    assert view.validate_logout_request(id_token_hint=id_token, client_id=client_id, post_logout_redirect_uri='http://example.org') == (application, oidc_tokens.user)
    with pytest.raises(InvalidIDTokenError):
        view.validate_logout_request(id_token_hint='111', client_id=public_application.client_id, post_logout_redirect_uri='http://other.org')
    with pytest.raises(ClientIdMissmatch):
        view.validate_logout_request(id_token_hint=id_token, client_id=public_application.client_id, post_logout_redirect_uri='http://other.org')
    with pytest.raises(InvalidOIDCClientError):
        view.validate_logout_request(id_token_hint=None, client_id=None, post_logout_redirect_uri='http://example.org')
    with pytest.raises(InvalidOIDCRedirectURIError):
        view.validate_logout_request(id_token_hint=None, client_id=client_id, post_logout_redirect_uri='example.org')
    with pytest.raises(InvalidOIDCRedirectURIError):
        view.validate_logout_request(id_token_hint=None, client_id=client_id, post_logout_redirect_uri='imap://example.org')
    with pytest.raises(InvalidOIDCRedirectURIError):
        view.validate_logout_request(id_token_hint=None, client_id=client_id, post_logout_redirect_uri='http://other.org')
    with pytest.raises(InvalidOIDCRedirectURIError):
        rp_settings.OIDC_RP_INITIATED_LOGOUT_STRICT_REDIRECT_URIS = True
        view.validate_logout_request(id_token_hint=None, client_id=public_application.client_id, post_logout_redirect_uri='http://other.org')

@pytest.mark.django_db
@pytest.mark.parametrize('ALWAYS_PROMPT', [True, False])
def test_must_prompt(oidc_tokens, other_user, rp_settings, ALWAYS_PROMPT):
    if False:
        return 10
    rp_settings.OIDC_RP_INITIATED_LOGOUT_ALWAYS_PROMPT = ALWAYS_PROMPT
    oidc_tokens = oidc_tokens
    assert RPInitiatedLogoutView(request=mock_request_for(oidc_tokens.user)).must_prompt(None) is True
    assert RPInitiatedLogoutView(request=mock_request_for(oidc_tokens.user)).must_prompt(oidc_tokens.user) == ALWAYS_PROMPT
    assert RPInitiatedLogoutView(request=mock_request_for(other_user)).must_prompt(oidc_tokens.user) is True

def test__load_id_token():
    if False:
        return 10
    assert _load_id_token('Not a Valid ID Token.') == (None, None)

def is_logged_in(client):
    if False:
        for i in range(10):
            print('nop')
    return get_user(client).is_authenticated

@pytest.mark.django_db
def test_rp_initiated_logout_get(logged_in_client, rp_settings):
    if False:
        i = 10
        return i + 15
    rsp = logged_in_client.get(reverse('oauth2_provider:rp-initiated-logout'), data={})
    assert rsp.status_code == 200
    assert is_logged_in(logged_in_client)

@pytest.mark.django_db
def test_rp_initiated_logout_get_id_token(logged_in_client, oidc_tokens, rp_settings):
    if False:
        print('Hello World!')
    rsp = logged_in_client.get(reverse('oauth2_provider:rp-initiated-logout'), data={'id_token_hint': oidc_tokens.id_token})
    assert rsp.status_code == 302
    assert rsp['Location'] == 'http://testserver/'
    assert not is_logged_in(logged_in_client)

@pytest.mark.django_db
def test_rp_initiated_logout_get_revoked_id_token(logged_in_client, oidc_tokens, rp_settings):
    if False:
        print('Hello World!')
    validator = oauth2_settings.OAUTH2_VALIDATOR_CLASS()
    validator._load_id_token(oidc_tokens.id_token).revoke()
    rsp = logged_in_client.get(reverse('oauth2_provider:rp-initiated-logout'), data={'id_token_hint': oidc_tokens.id_token})
    assert rsp.status_code == 400
    assert is_logged_in(logged_in_client)

@pytest.mark.django_db
def test_rp_initiated_logout_get_id_token_redirect(logged_in_client, oidc_tokens, rp_settings):
    if False:
        print('Hello World!')
    rsp = logged_in_client.get(reverse('oauth2_provider:rp-initiated-logout'), data={'id_token_hint': oidc_tokens.id_token, 'post_logout_redirect_uri': 'http://example.org'})
    assert rsp.status_code == 302
    assert rsp['Location'] == 'http://example.org'
    assert not is_logged_in(logged_in_client)

@pytest.mark.django_db
def test_rp_initiated_logout_get_id_token_redirect_with_state(logged_in_client, oidc_tokens, rp_settings):
    if False:
        return 10
    rsp = logged_in_client.get(reverse('oauth2_provider:rp-initiated-logout'), data={'id_token_hint': oidc_tokens.id_token, 'post_logout_redirect_uri': 'http://example.org', 'state': '987654321'})
    assert rsp.status_code == 302
    assert rsp['Location'] == 'http://example.org?state=987654321'
    assert not is_logged_in(logged_in_client)

@pytest.mark.django_db
def test_rp_initiated_logout_get_id_token_missmatch_client_id(logged_in_client, oidc_tokens, public_application, rp_settings):
    if False:
        i = 10
        return i + 15
    rsp = logged_in_client.get(reverse('oauth2_provider:rp-initiated-logout'), data={'id_token_hint': oidc_tokens.id_token, 'client_id': public_application.client_id})
    assert rsp.status_code == 400
    assert is_logged_in(logged_in_client)

@pytest.mark.django_db
def test_rp_initiated_logout_public_client_redirect_client_id(logged_in_client, oidc_non_confidential_tokens, public_application, rp_settings):
    if False:
        print('Hello World!')
    rsp = logged_in_client.get(reverse('oauth2_provider:rp-initiated-logout'), data={'id_token_hint': oidc_non_confidential_tokens.id_token, 'client_id': public_application.client_id, 'post_logout_redirect_uri': 'http://other.org'})
    assert rsp.status_code == 302
    assert not is_logged_in(logged_in_client)

@pytest.mark.django_db
def test_rp_initiated_logout_public_client_strict_redirect_client_id(logged_in_client, oidc_non_confidential_tokens, public_application, oauth2_settings):
    if False:
        i = 10
        return i + 15
    oauth2_settings.update(presets.OIDC_SETTINGS_RP_LOGOUT_STRICT_REDIRECT_URI)
    rsp = logged_in_client.get(reverse('oauth2_provider:rp-initiated-logout'), data={'id_token_hint': oidc_non_confidential_tokens.id_token, 'client_id': public_application.client_id, 'post_logout_redirect_uri': 'http://other.org'})
    assert rsp.status_code == 400
    assert is_logged_in(logged_in_client)

@pytest.mark.django_db
def test_rp_initiated_logout_get_client_id(logged_in_client, oidc_tokens, rp_settings):
    if False:
        print('Hello World!')
    rsp = logged_in_client.get(reverse('oauth2_provider:rp-initiated-logout'), data={'client_id': oidc_tokens.application.client_id})
    assert rsp.status_code == 200
    assert is_logged_in(logged_in_client)

@pytest.mark.django_db
def test_rp_initiated_logout_post(logged_in_client, oidc_tokens, rp_settings):
    if False:
        print('Hello World!')
    form_data = {'client_id': oidc_tokens.application.client_id}
    rsp = logged_in_client.post(reverse('oauth2_provider:rp-initiated-logout'), form_data)
    assert rsp.status_code == 400
    assert is_logged_in(logged_in_client)

@pytest.mark.django_db
def test_rp_initiated_logout_post_allowed(logged_in_client, oidc_tokens, rp_settings):
    if False:
        i = 10
        return i + 15
    form_data = {'client_id': oidc_tokens.application.client_id, 'allow': True}
    rsp = logged_in_client.post(reverse('oauth2_provider:rp-initiated-logout'), form_data)
    assert rsp.status_code == 302
    assert rsp['Location'] == 'http://testserver/'
    assert not is_logged_in(logged_in_client)

@pytest.mark.django_db
def test_rp_initiated_logout_post_no_session(client, oidc_tokens, rp_settings):
    if False:
        return 10
    form_data = {'client_id': oidc_tokens.application.client_id, 'allow': True}
    rsp = client.post(reverse('oauth2_provider:rp-initiated-logout'), form_data)
    assert rsp.status_code == 302
    assert rsp['Location'] == 'http://testserver/'
    assert not is_logged_in(client)

@pytest.mark.django_db
@pytest.mark.oauth2_settings(presets.OIDC_SETTINGS_RP_LOGOUT)
def test_rp_initiated_logout_expired_tokens_accept(logged_in_client, application, expired_id_token):
    if False:
        print('Hello World!')
    rsp = logged_in_client.get(reverse('oauth2_provider:rp-initiated-logout'), data={'id_token_hint': expired_id_token, 'client_id': application.client_id})
    assert rsp.status_code == 302
    assert not is_logged_in(logged_in_client)

@pytest.mark.django_db
@pytest.mark.oauth2_settings(presets.OIDC_SETTINGS_RP_LOGOUT_DENY_EXPIRED)
def test_rp_initiated_logout_expired_tokens_deny(logged_in_client, application, expired_id_token):
    if False:
        i = 10
        return i + 15
    rsp = logged_in_client.get(reverse('oauth2_provider:rp-initiated-logout'), data={'id_token_hint': expired_id_token, 'client_id': application.client_id})
    assert rsp.status_code == 400
    assert is_logged_in(logged_in_client)

@pytest.mark.django_db
@pytest.mark.oauth2_settings(presets.OIDC_SETTINGS_RP_LOGOUT)
def test_load_id_token_accept_expired(expired_id_token):
    if False:
        print('Hello World!')
    (id_token, _) = _load_id_token(expired_id_token)
    assert isinstance(id_token, get_id_token_model())

@pytest.mark.django_db
@pytest.mark.oauth2_settings(presets.OIDC_SETTINGS_RP_LOGOUT)
def test_load_id_token_wrong_aud(id_token_wrong_aud):
    if False:
        return 10
    (id_token, claims) = _load_id_token(id_token_wrong_aud)
    assert id_token is None
    assert claims is None

@pytest.mark.django_db
@pytest.mark.oauth2_settings(presets.OIDC_SETTINGS_RP_LOGOUT_DENY_EXPIRED)
def test_load_id_token_deny_expired(expired_id_token):
    if False:
        return 10
    (id_token, claims) = _load_id_token(expired_id_token)
    assert id_token is None
    assert claims is None

@pytest.mark.django_db
@pytest.mark.oauth2_settings(presets.OIDC_SETTINGS_RP_LOGOUT)
def test_validate_claims_wrong_iss(id_token_wrong_iss):
    if False:
        print('Hello World!')
    (id_token, claims) = _load_id_token(id_token_wrong_iss)
    assert id_token is not None
    assert claims is not None
    assert not _validate_claims(mock_request(), claims)

@pytest.mark.django_db
@pytest.mark.oauth2_settings(presets.OIDC_SETTINGS_RP_LOGOUT)
def test_validate_claims(oidc_tokens):
    if False:
        for i in range(10):
            print('nop')
    (id_token, claims) = _load_id_token(oidc_tokens.id_token)
    assert claims is not None
    assert _validate_claims(mock_request_for(oidc_tokens.user), claims)

@pytest.mark.django_db
@pytest.mark.parametrize('method', ['get', 'post'])
def test_userinfo_endpoint(oidc_tokens, client, method):
    if False:
        i = 10
        return i + 15
    auth_header = 'Bearer %s' % oidc_tokens.access_token
    rsp = getattr(client, method)(reverse('oauth2_provider:user-info'), HTTP_AUTHORIZATION=auth_header)
    data = rsp.json()
    assert 'sub' in data
    assert data['sub'] == str(oidc_tokens.user.pk)

@pytest.mark.django_db
def test_userinfo_endpoint_bad_token(oidc_tokens, client):
    if False:
        for i in range(10):
            print('nop')
    rsp = client.get(reverse('oauth2_provider:user-info'))
    assert rsp.status_code == 401
    rsp = client.get(reverse('oauth2_provider:user-info'), HTTP_AUTHORIZATION='Bearer not-a-real-token')
    assert rsp.status_code == 401

@pytest.mark.django_db
def test_token_deletion_on_logout(oidc_tokens, logged_in_client, rp_settings):
    if False:
        print('Hello World!')
    AccessToken = get_access_token_model()
    IDToken = get_id_token_model()
    RefreshToken = get_refresh_token_model()
    assert AccessToken.objects.count() == 1
    assert IDToken.objects.count() == 1
    assert RefreshToken.objects.count() == 1
    rsp = logged_in_client.get(reverse('oauth2_provider:rp-initiated-logout'), data={'id_token_hint': oidc_tokens.id_token, 'client_id': oidc_tokens.application.client_id})
    assert rsp.status_code == 302
    assert not is_logged_in(logged_in_client)
    assert all([token.is_expired() for token in AccessToken.objects.all()])
    assert all([token.is_expired() for token in IDToken.objects.all()])
    assert all([token.revoked <= timezone.now() for token in RefreshToken.objects.all()])

@pytest.mark.django_db
def test_token_deletion_on_logout_expired_session(oidc_tokens, client, rp_settings):
    if False:
        return 10
    AccessToken = get_access_token_model()
    IDToken = get_id_token_model()
    RefreshToken = get_refresh_token_model()
    assert AccessToken.objects.count() == 1
    assert IDToken.objects.count() == 1
    assert RefreshToken.objects.count() == 1
    rsp = client.get(reverse('oauth2_provider:rp-initiated-logout'), data={'id_token_hint': oidc_tokens.id_token, 'client_id': oidc_tokens.application.client_id})
    assert rsp.status_code == 200
    assert not is_logged_in(client)
    access_token = AccessToken.objects.get()
    assert not access_token.is_expired()
    id_token = IDToken.objects.get()
    assert not id_token.is_expired()
    refresh_token = RefreshToken.objects.get()
    assert refresh_token.revoked is None
    rsp = client.post(reverse('oauth2_provider:rp-initiated-logout'), data={'id_token_hint': oidc_tokens.id_token, 'client_id': oidc_tokens.application.client_id, 'allow': True})
    assertRedirects(rsp, 'http://testserver/', fetch_redirect_response=False)
    assert not is_logged_in(client)
    assert all((token.is_expired() for token in AccessToken.objects.all()))
    assert all((token.is_expired() for token in IDToken.objects.all()))
    assert all((token.revoked <= timezone.now() for token in RefreshToken.objects.all()))

@pytest.mark.django_db
@pytest.mark.oauth2_settings(presets.OIDC_SETTINGS_RP_LOGOUT_KEEP_TOKENS)
def test_token_deletion_on_logout_disabled(oidc_tokens, logged_in_client, rp_settings):
    if False:
        return 10
    rp_settings.OIDC_RP_INITIATED_LOGOUT_DELETE_TOKENS = False
    AccessToken = get_access_token_model()
    IDToken = get_id_token_model()
    RefreshToken = get_refresh_token_model()
    assert AccessToken.objects.count() == 1
    assert IDToken.objects.count() == 1
    assert RefreshToken.objects.count() == 1
    rsp = logged_in_client.get(reverse('oauth2_provider:rp-initiated-logout'), data={'id_token_hint': oidc_tokens.id_token, 'client_id': oidc_tokens.application.client_id})
    assert rsp.status_code == 302
    assert not is_logged_in(logged_in_client)
    assert AccessToken.objects.count() == 1
    assert not any([token.is_expired() for token in AccessToken.objects.all()])
    assert IDToken.objects.count() == 1
    assert not any([token.is_expired() for token in IDToken.objects.all()])
    assert RefreshToken.objects.count() == 1
    assert not any([token.revoked is not None for token in RefreshToken.objects.all()])
EXAMPLE_EMAIL = 'example.email@example.com'

def claim_user_email(request):
    if False:
        print('Hello World!')
    return EXAMPLE_EMAIL

@pytest.mark.django_db
def test_userinfo_endpoint_custom_claims_callable(oidc_tokens, client, oauth2_settings):
    if False:
        print('Hello World!')

    class CustomValidator(OAuth2Validator):
        oidc_claim_scope = None

        def get_additional_claims(self):
            if False:
                while True:
                    i = 10
            return {'username': claim_user_email, 'email': claim_user_email}
    oidc_tokens.oauth2_settings.OAUTH2_VALIDATOR_CLASS = CustomValidator
    auth_header = 'Bearer %s' % oidc_tokens.access_token
    rsp = client.get(reverse('oauth2_provider:user-info'), HTTP_AUTHORIZATION=auth_header)
    data = rsp.json()
    assert 'sub' in data
    assert data['sub'] == str(oidc_tokens.user.pk)
    assert 'username' in data
    assert data['username'] == EXAMPLE_EMAIL
    assert 'email' in data
    assert data['email'] == EXAMPLE_EMAIL

@pytest.mark.django_db
def test_userinfo_endpoint_custom_claims_email_scope_callable(oidc_email_scope_tokens, client, oauth2_settings):
    if False:
        print('Hello World!')

    class CustomValidator(OAuth2Validator):

        def get_additional_claims(self):
            if False:
                while True:
                    i = 10
            return {'username': claim_user_email, 'email': claim_user_email}
    oidc_email_scope_tokens.oauth2_settings.OAUTH2_VALIDATOR_CLASS = CustomValidator
    auth_header = 'Bearer %s' % oidc_email_scope_tokens.access_token
    rsp = client.get(reverse('oauth2_provider:user-info'), HTTP_AUTHORIZATION=auth_header)
    data = rsp.json()
    assert 'sub' in data
    assert data['sub'] == str(oidc_email_scope_tokens.user.pk)
    assert 'username' not in data
    assert 'email' in data
    assert data['email'] == EXAMPLE_EMAIL

@pytest.mark.django_db
def test_userinfo_endpoint_custom_claims_plain(oidc_tokens, client, oauth2_settings):
    if False:
        i = 10
        return i + 15

    class CustomValidator(OAuth2Validator):
        oidc_claim_scope = None

        def get_additional_claims(self, request):
            if False:
                i = 10
                return i + 15
            return {'username': EXAMPLE_EMAIL, 'email': EXAMPLE_EMAIL}
    oidc_tokens.oauth2_settings.OAUTH2_VALIDATOR_CLASS = CustomValidator
    auth_header = 'Bearer %s' % oidc_tokens.access_token
    rsp = client.get(reverse('oauth2_provider:user-info'), HTTP_AUTHORIZATION=auth_header)
    data = rsp.json()
    assert 'sub' in data
    assert data['sub'] == str(oidc_tokens.user.pk)
    assert 'username' in data
    assert data['username'] == EXAMPLE_EMAIL
    assert 'email' in data
    assert data['email'] == EXAMPLE_EMAIL

@pytest.mark.django_db
def test_userinfo_endpoint_custom_claims_email_scopeplain(oidc_email_scope_tokens, client, oauth2_settings):
    if False:
        i = 10
        return i + 15

    class CustomValidator(OAuth2Validator):

        def get_additional_claims(self, request):
            if False:
                while True:
                    i = 10
            return {'username': EXAMPLE_EMAIL, 'email': EXAMPLE_EMAIL}
    oidc_email_scope_tokens.oauth2_settings.OAUTH2_VALIDATOR_CLASS = CustomValidator
    auth_header = 'Bearer %s' % oidc_email_scope_tokens.access_token
    rsp = client.get(reverse('oauth2_provider:user-info'), HTTP_AUTHORIZATION=auth_header)
    data = rsp.json()
    assert 'sub' in data
    assert data['sub'] == str(oidc_email_scope_tokens.user.pk)
    assert 'username' not in data
    assert 'email' in data
    assert data['email'] == EXAMPLE_EMAIL