import base64
import datetime
import hashlib
import json
from urllib.parse import parse_qs, urlparse
import pytest
from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import RequestFactory, TestCase
from django.urls import reverse
from django.utils import timezone
from django.utils.crypto import get_random_string
from jwcrypto import jwt
from oauthlib.oauth2.rfc6749 import errors as oauthlib_errors
from oauth2_provider.models import get_access_token_model, get_application_model, get_grant_model, get_refresh_token_model
from oauth2_provider.views import ProtectedResourceView
from . import presets
from .utils import get_basic_auth_header
Application = get_application_model()
AccessToken = get_access_token_model()
Grant = get_grant_model()
RefreshToken = get_refresh_token_model()
UserModel = get_user_model()
CLEARTEXT_SECRET = '1234567890abcdefghijklmnopqrstuvwxyz'

class ResourceView(ProtectedResourceView):

    def get(self, request, *args, **kwargs):
        if False:
            print('Hello World!')
        return 'This is a protected resource'

@pytest.mark.usefixtures('oauth2_settings')
class BaseTest(TestCase):
    factory = RequestFactory()

    @classmethod
    def setUpTestData(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.test_user = UserModel.objects.create_user('test_user', 'test@example.com', '123456')
        cls.dev_user = UserModel.objects.create_user('dev_user', 'dev@example.com', '123456')
        cls.application = Application.objects.create(name='Test Application', redirect_uris='http://localhost http://example.com http://example.org custom-scheme://example.com', user=cls.dev_user, client_type=Application.CLIENT_CONFIDENTIAL, authorization_grant_type=Application.GRANT_AUTHORIZATION_CODE, client_secret=CLEARTEXT_SECRET)

    def setUp(self):
        if False:
            while True:
                i = 10
        self.oauth2_settings.ALLOWED_REDIRECT_URI_SCHEMES = ['http', 'custom-scheme']
        self.oauth2_settings.PKCE_REQUIRED = False

class TestRegressionIssue315(BaseTest):
    """
    Test to avoid regression for the issue 315: request object
    was being reassigned when getting AuthorizationView
    """

    def test_request_is_not_overwritten(self):
        if False:
            print('Hello World!')
        self.oauth2_settings.PKCE_REQUIRED = False
        self.client.login(username='test_user', password='123456')
        response = self.client.get(reverse('oauth2_provider:authorize'), {'client_id': self.application.client_id, 'response_type': 'code', 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org'})
        self.assertEqual(response.status_code, 200)
        assert 'request' not in response.context_data

@pytest.mark.oauth2_settings(presets.DEFAULT_SCOPES_RW)
class TestAuthorizationCodeView(BaseTest):

    def test_skip_authorization_completely(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If application.skip_authorization = True, should skip the authorization page.\n        '
        self.oauth2_settings.PKCE_REQUIRED = False
        self.client.login(username='test_user', password='123456')
        self.application.skip_authorization = True
        self.application.save()
        response = self.client.get(reverse('oauth2_provider:authorize'), {'client_id': self.application.client_id, 'response_type': 'code', 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org'})
        self.assertEqual(response.status_code, 302)

    def test_pre_auth_invalid_client(self):
        if False:
            i = 10
            return i + 15
        '\n        Test error for an invalid client_id with response_type: code\n        '
        self.client.login(username='test_user', password='123456')
        query_data = {'client_id': 'fakeclientid', 'response_type': 'code'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.context_data['url'], '?error=invalid_request&error_description=Invalid+client_id+parameter+value.')

    def test_pre_auth_valid_client(self):
        if False:
            while True:
                i = 10
        '\n        Test response for a valid client_id with response_type: code\n        '
        self.oauth2_settings.PKCE_REQUIRED = False
        self.client.login(username='test_user', password='123456')
        query_data = {'client_id': self.application.client_id, 'response_type': 'code', 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('form', response.context)
        form = response.context['form']
        self.assertEqual(form['redirect_uri'].value(), 'http://example.org')
        self.assertEqual(form['state'].value(), 'random_state_string')
        self.assertEqual(form['scope'].value(), 'read write')
        self.assertEqual(form['client_id'].value(), self.application.client_id)

    def test_pre_auth_valid_client_custom_redirect_uri_scheme(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test response for a valid client_id with response_type: code\n        using a non-standard, but allowed, redirect_uri scheme.\n        '
        self.client.login(username='test_user', password='123456')
        query_data = {'client_id': self.application.client_id, 'response_type': 'code', 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'custom-scheme://example.com'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('form', response.context)
        form = response.context['form']
        self.assertEqual(form['redirect_uri'].value(), 'custom-scheme://example.com')
        self.assertEqual(form['state'].value(), 'random_state_string')
        self.assertEqual(form['scope'].value(), 'read write')
        self.assertEqual(form['client_id'].value(), self.application.client_id)

    def test_pre_auth_approval_prompt(self):
        if False:
            print('Hello World!')
        tok = AccessToken.objects.create(user=self.test_user, token='1234567890', application=self.application, expires=timezone.now() + datetime.timedelta(days=1), scope='read write')
        self.client.login(username='test_user', password='123456')
        query_data = {'client_id': self.application.client_id, 'response_type': 'code', 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org', 'approval_prompt': 'auto'}
        url = reverse('oauth2_provider:authorize')
        response = self.client.get(url, data=query_data)
        self.assertEqual(response.status_code, 302)
        tok.scope = 'read'
        tok.save()
        response = self.client.get(url, data=query_data)
        self.assertEqual(response.status_code, 200)

    def test_pre_auth_approval_prompt_default(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.oauth2_settings.REQUEST_APPROVAL_PROMPT, 'force')
        AccessToken.objects.create(user=self.test_user, token='1234567890', application=self.application, expires=timezone.now() + datetime.timedelta(days=1), scope='read write')
        self.client.login(username='test_user', password='123456')
        query_data = {'client_id': self.application.client_id, 'response_type': 'code', 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertEqual(response.status_code, 200)

    def test_pre_auth_approval_prompt_default_override(self):
        if False:
            for i in range(10):
                print('nop')
        self.oauth2_settings.REQUEST_APPROVAL_PROMPT = 'auto'
        AccessToken.objects.create(user=self.test_user, token='1234567890', application=self.application, expires=timezone.now() + datetime.timedelta(days=1), scope='read write')
        self.client.login(username='test_user', password='123456')
        query_data = {'client_id': self.application.client_id, 'response_type': 'code', 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertEqual(response.status_code, 302)

    def test_pre_auth_default_redirect(self):
        if False:
            print('Hello World!')
        '\n        Test for default redirect uri if omitted from query string with response_type: code\n        '
        self.client.login(username='test_user', password='123456')
        self.application.redirect_uris = 'http://localhost'
        self.application.save()
        query_data = {'client_id': self.application.client_id, 'response_type': 'code'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertEqual(response.status_code, 200)
        form = response.context['form']
        self.assertEqual(form['redirect_uri'].value(), 'http://localhost')

    def test_pre_auth_missing_redirect(self):
        if False:
            i = 10
            return i + 15
        '\n        Test response if redirect_uri is missing and multiple URIs are registered.\n        @see https://datatracker.ietf.org/doc/html/rfc6749#section-3.1.2.3\n        '
        self.client.login(username='test_user', password='123456')
        query_data = {'client_id': self.application.client_id, 'response_type': 'code'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertEqual(response.status_code, 400)

    def test_pre_auth_forbibben_redirect(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test error when passing a forbidden redirect_uri in query string with response_type: code\n        '
        self.client.login(username='test_user', password='123456')
        query_data = {'client_id': self.application.client_id, 'response_type': 'code', 'redirect_uri': 'http://forbidden.it'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertEqual(response.status_code, 400)

    def test_pre_auth_wrong_response_type(self):
        if False:
            while True:
                i = 10
        '\n        Test error when passing a wrong response_type in query string\n        '
        self.client.login(username='test_user', password='123456')
        query_data = {'client_id': self.application.client_id, 'response_type': 'WRONG', 'redirect_uri': 'http://example.org'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertEqual(response.status_code, 302)
        self.assertIn('error=unsupported_response_type', response['Location'])

    def test_code_post_auth_allow(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test authorization code is given for an allowed request with response_type: code\n        '
        self.client.login(username='test_user', password='123456')
        form_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=form_data)
        self.assertEqual(response.status_code, 302)
        self.assertIn('http://example.org?', response['Location'])
        self.assertIn('state=random_state_string', response['Location'])
        self.assertIn('code=', response['Location'])

    def test_code_post_auth_deny(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test error when resource owner deny access\n        '
        self.client.login(username='test_user', password='123456')
        form_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': False}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=form_data)
        self.assertEqual(response.status_code, 302)
        self.assertIn('error=access_denied', response['Location'])
        self.assertIn('state=random_state_string', response['Location'])

    def test_code_post_auth_deny_no_state(self):
        if False:
            while True:
                i = 10
        '\n        Test optional state when resource owner deny access\n        '
        self.client.login(username='test_user', password='123456')
        form_data = {'client_id': self.application.client_id, 'scope': 'read write', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': False}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=form_data)
        self.assertEqual(response.status_code, 302)
        self.assertIn('error=access_denied', response['Location'])
        self.assertNotIn('state', response['Location'])

    def test_code_post_auth_bad_responsetype(self):
        if False:
            i = 10
            return i + 15
        '\n        Test authorization code is given for an allowed request with a response_type not supported\n        '
        self.client.login(username='test_user', password='123456')
        form_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org', 'response_type': 'UNKNOWN', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=form_data)
        self.assertEqual(response.status_code, 302)
        self.assertIn('http://example.org?error', response['Location'])

    def test_code_post_auth_forbidden_redirect_uri(self):
        if False:
            while True:
                i = 10
        '\n        Test authorization code is given for an allowed request with a forbidden redirect_uri\n        '
        self.client.login(username='test_user', password='123456')
        form_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://forbidden.it', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=form_data)
        self.assertEqual(response.status_code, 400)

    def test_code_post_auth_malicious_redirect_uri(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test validation of a malicious redirect_uri\n        '
        self.client.login(username='test_user', password='123456')
        form_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': '/../', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=form_data)
        self.assertEqual(response.status_code, 400)

    def test_code_post_auth_allow_custom_redirect_uri_scheme(self):
        if False:
            print('Hello World!')
        '\n        Test authorization code is given for an allowed request with response_type: code\n        using a non-standard, but allowed, redirect_uri scheme.\n        '
        self.client.login(username='test_user', password='123456')
        form_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'custom-scheme://example.com', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=form_data)
        self.assertEqual(response.status_code, 302)
        self.assertIn('custom-scheme://example.com?', response['Location'])
        self.assertIn('state=random_state_string', response['Location'])
        self.assertIn('code=', response['Location'])

    def test_code_post_auth_deny_custom_redirect_uri_scheme(self):
        if False:
            while True:
                i = 10
        '\n        Test error when resource owner deny access\n        using a non-standard, but allowed, redirect_uri scheme.\n        '
        self.client.login(username='test_user', password='123456')
        form_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'custom-scheme://example.com', 'response_type': 'code', 'allow': False}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=form_data)
        self.assertEqual(response.status_code, 302)
        self.assertIn('custom-scheme://example.com?', response['Location'])
        self.assertIn('error=access_denied', response['Location'])
        self.assertIn('state=random_state_string', response['Location'])

    def test_code_post_auth_redirection_uri_with_querystring(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that a redirection uri with query string is allowed\n        and query string is retained on redirection.\n        See https://rfc-editor.org/rfc/rfc6749.html#section-3.1.2\n        '
        self.client.login(username='test_user', password='123456')
        form_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.com?foo=bar', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=form_data)
        self.assertEqual(response.status_code, 302)
        self.assertIn('http://example.com?foo=bar', response['Location'])
        self.assertIn('code=', response['Location'])
        self.assertIn('state=random_state_string', response['Location'])

    def test_code_post_auth_failing_redirection_uri_with_querystring(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that in case of error the querystring of the redirection uri is preserved\n\n        See https://github.com/jazzband/django-oauth-toolkit/issues/238\n        '
        self.client.login(username='test_user', password='123456')
        form_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.com?foo=bar', 'response_type': 'code', 'allow': False}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=form_data)
        self.assertEqual(response.status_code, 302)
        self.assertIn('http://example.com?', response['Location'])
        self.assertIn('error=access_denied', response['Location'])
        self.assertIn('state=random_state_string', response['Location'])
        self.assertIn('foo=bar', response['Location'])

    def test_code_post_auth_fails_when_redirect_uri_path_is_invalid(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that a redirection uri is matched using scheme + netloc + path\n        '
        self.client.login(username='test_user', password='123456')
        form_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.com/a?foo=bar', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=form_data)
        self.assertEqual(response.status_code, 400)

@pytest.mark.oauth2_settings(presets.OIDC_SETTINGS_RW)
class TestOIDCAuthorizationCodeView(BaseTest):

    def test_id_token_skip_authorization_completely(self):
        if False:
            return 10
        '\n        If application.skip_authorization = True, should skip the authorization page.\n        '
        self.client.login(username='test_user', password='123456')
        self.application.skip_authorization = True
        self.application.save()
        query_data = {'client_id': self.application.client_id, 'response_type': 'code', 'state': 'random_state_string', 'scope': 'openid', 'redirect_uri': 'http://example.org'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertEqual(response.status_code, 302)

    def test_id_token_pre_auth_valid_client(self):
        if False:
            return 10
        '\n        Test response for a valid client_id with response_type: code\n        '
        self.client.login(username='test_user', password='123456')
        query_data = {'client_id': self.application.client_id, 'response_type': 'code', 'state': 'random_state_string', 'scope': 'openid', 'redirect_uri': 'http://example.org'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('form', response.context)
        form = response.context['form']
        self.assertEqual(form['redirect_uri'].value(), 'http://example.org')
        self.assertEqual(form['state'].value(), 'random_state_string')
        self.assertEqual(form['scope'].value(), 'openid')
        self.assertEqual(form['client_id'].value(), self.application.client_id)

    def test_id_token_code_post_auth_allow(self):
        if False:
            print('Hello World!')
        '\n        Test authorization code is given for an allowed request with response_type: code\n        '
        self.client.login(username='test_user', password='123456')
        form_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'openid', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=form_data)
        self.assertEqual(response.status_code, 302)
        self.assertIn('http://example.org?', response['Location'])
        self.assertIn('state=random_state_string', response['Location'])
        self.assertIn('code=', response['Location'])

    def test_prompt_login(self):
        if False:
            print('Hello World!')
        '\n        Test response for redirect when supplied with prompt: login\n        '
        self.oauth2_settings.PKCE_REQUIRED = False
        self.client.login(username='test_user', password='123456')
        query_data = {'client_id': self.application.client_id, 'response_type': 'code', 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org', 'prompt': 'login'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertEqual(response.status_code, 302)
        (scheme, netloc, path, params, query, fragment) = urlparse(response['Location'])
        self.assertEqual(path, settings.LOGIN_URL)
        parsed_query = parse_qs(query)
        next = parsed_query['next'][0]
        self.assertIn('redirect_uri=http%3A%2F%2Fexample.org', next)
        self.assertIn('state=random_state_string', next)
        self.assertIn('scope=read+write', next)
        self.assertIn(f'client_id={self.application.client_id}', next)
        self.assertNotIn('prompt=login', next)

class BaseAuthorizationCodeTokenView(BaseTest):

    def get_auth(self, scope='read write'):
        if False:
            print('Hello World!')
        '\n        Helper method to retrieve a valid authorization code\n        '
        authcode_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': scope, 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=authcode_data)
        query_dict = parse_qs(urlparse(response['Location']).query)
        return query_dict['code'].pop()

    def generate_pkce_codes(self, algorithm, length=43):
        if False:
            while True:
                i = 10
        '\n        Helper method to generate pkce codes\n        '
        code_verifier = get_random_string(length)
        if algorithm == 'S256':
            code_challenge = base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest()).decode().rstrip('=')
        else:
            code_challenge = code_verifier
        return (code_verifier, code_challenge)

    def get_pkce_auth(self, code_challenge, code_challenge_method):
        if False:
            print('Hello World!')
        '\n        Helper method to retrieve a valid authorization code using pkce\n        '
        authcode_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True, 'code_challenge': code_challenge, 'code_challenge_method': code_challenge_method}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=authcode_data)
        query_dict = parse_qs(urlparse(response['Location']).query)
        return query_dict['code'].pop()

@pytest.mark.oauth2_settings(presets.DEFAULT_SCOPES_RW)
class TestAuthorizationCodeTokenView(BaseAuthorizationCodeTokenView):

    def test_basic_auth(self):
        if False:
            i = 10
            return i + 15
        '\n        Request an access token using basic authentication for client authentication\n        '
        self.client.login(username='test_user', password='123456')
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content.decode('utf-8'))
        self.assertEqual(content['token_type'], 'Bearer')
        self.assertEqual(content['scope'], 'read write')
        self.assertEqual(content['expires_in'], self.oauth2_settings.ACCESS_TOKEN_EXPIRE_SECONDS)

    def test_refresh(self):
        if False:
            return 10
        '\n        Request an access token using a refresh token\n        '
        self.client.login(username='test_user', password='123456')
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        content = json.loads(response.content.decode('utf-8'))
        self.assertTrue('refresh_token' in content)
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        token_request_data = {'grant_type': 'refresh_token', 'refresh_token': content['refresh_token'], 'scope': content['scope']}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content.decode('utf-8'))
        self.assertTrue('access_token' in content)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 400)
        content = json.loads(response.content.decode('utf-8'))
        self.assertTrue('invalid_grant' in content.values())

    def test_refresh_with_grace_period(self):
        if False:
            return 10
        '\n        Request an access token using a refresh token\n        '
        self.oauth2_settings.REFRESH_TOKEN_GRACE_PERIOD_SECONDS = 120
        self.client.login(username='test_user', password='123456')
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        content = json.loads(response.content.decode('utf-8'))
        self.assertTrue('refresh_token' in content)
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        token_request_data = {'grant_type': 'refresh_token', 'refresh_token': content['refresh_token'], 'scope': content['scope']}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content.decode('utf-8'))
        self.assertTrue('access_token' in content)
        first_access_token = content['access_token']
        first_refresh_token = content['refresh_token']
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content.decode('utf-8'))
        self.assertTrue('access_token' in content)
        self.assertEqual(content['access_token'], first_access_token)
        self.assertTrue('refresh_token' in content)
        self.assertEqual(content['refresh_token'], first_refresh_token)

    def test_refresh_invalidates_old_tokens(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure existing refresh tokens are cleaned up when issuing new ones\n        '
        self.client.login(username='test_user', password='123456')
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        content = json.loads(response.content.decode('utf-8'))
        rt = content['refresh_token']
        at = content['access_token']
        token_request_data = {'grant_type': 'refresh_token', 'refresh_token': rt, 'scope': content['scope']}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 200)
        refresh_token = RefreshToken.objects.filter(token=rt).first()
        self.assertIsNotNone(refresh_token.revoked)
        self.assertFalse(AccessToken.objects.filter(token=at).exists())

    def test_refresh_no_scopes(self):
        if False:
            return 10
        '\n        Request an access token using a refresh token without passing any scope\n        '
        self.client.login(username='test_user', password='123456')
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        content = json.loads(response.content.decode('utf-8'))
        self.assertTrue('refresh_token' in content)
        token_request_data = {'grant_type': 'refresh_token', 'refresh_token': content['refresh_token']}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content.decode('utf-8'))
        self.assertTrue('access_token' in content)

    def test_refresh_bad_scopes(self):
        if False:
            print('Hello World!')
        '\n        Request an access token using a refresh token and wrong scopes\n        '
        self.client.login(username='test_user', password='123456')
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        content = json.loads(response.content.decode('utf-8'))
        self.assertTrue('refresh_token' in content)
        token_request_data = {'grant_type': 'refresh_token', 'refresh_token': content['refresh_token'], 'scope': 'read write nuke'}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 400)

    def test_refresh_fail_repeating_requests(self):
        if False:
            print('Hello World!')
        '\n        Try refreshing an access token with the same refresh token more than once\n        '
        self.client.login(username='test_user', password='123456')
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        content = json.loads(response.content.decode('utf-8'))
        self.assertTrue('refresh_token' in content)
        token_request_data = {'grant_type': 'refresh_token', 'refresh_token': content['refresh_token'], 'scope': content['scope']}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 400)

    def test_refresh_repeating_requests(self):
        if False:
            print('Hello World!')
        '\n        Trying to refresh an access token with the same refresh token more than\n        once succeeds in the grace period and fails outside\n        '
        self.oauth2_settings.REFRESH_TOKEN_GRACE_PERIOD_SECONDS = 120
        self.client.login(username='test_user', password='123456')
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        content = json.loads(response.content.decode('utf-8'))
        self.assertTrue('refresh_token' in content)
        token_request_data = {'grant_type': 'refresh_token', 'refresh_token': content['refresh_token'], 'scope': content['scope']}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 200)
        rt = RefreshToken.objects.get(token=content['refresh_token'])
        self.assertIsNotNone(rt.revoked)
        rt.revoked = timezone.now() - datetime.timedelta(minutes=10)
        rt.save()
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 400)

    def test_refresh_repeating_requests_non_rotating_tokens(self):
        if False:
            while True:
                i = 10
        '\n        Try refreshing an access token with the same refresh token more than once when not rotating tokens.\n        '
        self.client.login(username='test_user', password='123456')
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        content = json.loads(response.content.decode('utf-8'))
        self.assertTrue('refresh_token' in content)
        token_request_data = {'grant_type': 'refresh_token', 'refresh_token': content['refresh_token'], 'scope': content['scope']}
        self.oauth2_settings.ROTATE_REFRESH_TOKEN = False
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 200)

    def test_refresh_with_deleted_token(self):
        if False:
            return 10
        '\n        Ensure that using a deleted refresh token returns 400\n        '
        self.client.login(username='test_user', password='123456')
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'scope': 'read write', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        content = json.loads(response.content.decode('utf-8'))
        rt = content['refresh_token']
        token_request_data = {'grant_type': 'refresh_token', 'refresh_token': rt, 'scope': 'read write'}
        AccessToken.objects.filter(token=content['access_token']).delete()
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 400)

    def test_basic_auth_bad_authcode(self):
        if False:
            print('Hello World!')
        '\n        Request an access token using a bad authorization code\n        '
        self.client.login(username='test_user', password='123456')
        token_request_data = {'grant_type': 'authorization_code', 'code': 'BLAH', 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 400)

    def test_basic_auth_bad_granttype(self):
        if False:
            i = 10
            return i + 15
        '\n        Request an access token using a bad grant_type string\n        '
        self.client.login(username='test_user', password='123456')
        token_request_data = {'grant_type': 'UNKNOWN', 'code': 'BLAH', 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 400)

    def test_basic_auth_grant_expired(self):
        if False:
            i = 10
            return i + 15
        '\n        Request an access token using an expired grant token\n        '
        self.client.login(username='test_user', password='123456')
        g = Grant(application=self.application, user=self.test_user, code='BLAH', expires=timezone.now(), redirect_uri='', scope='')
        g.save()
        token_request_data = {'grant_type': 'authorization_code', 'code': 'BLAH', 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 400)

    def test_basic_auth_bad_secret(self):
        if False:
            i = 10
            return i + 15
        '\n        Request an access token using basic authentication for client authentication\n        '
        self.client.login(username='test_user', password='123456')
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, 'BOOM!')
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 401)

    def test_basic_auth_wrong_auth_type(self):
        if False:
            i = 10
            return i + 15
        '\n        Request an access token using basic authentication for client authentication\n        '
        self.client.login(username='test_user', password='123456')
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        user_pass = '{0}:{1}'.format(self.application.client_id, CLEARTEXT_SECRET)
        auth_string = base64.b64encode(user_pass.encode('utf-8'))
        auth_headers = {'HTTP_AUTHORIZATION': 'Wrong ' + auth_string.decode('utf-8')}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 401)

    def test_request_body_params(self):
        if False:
            print('Hello World!')
        '\n        Request an access token using client_type: public\n        '
        self.client.login(username='test_user', password='123456')
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org', 'client_id': self.application.client_id, 'client_secret': CLEARTEXT_SECRET}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content.decode('utf-8'))
        self.assertEqual(content['token_type'], 'Bearer')
        self.assertEqual(content['scope'], 'read write')
        self.assertEqual(content['expires_in'], self.oauth2_settings.ACCESS_TOKEN_EXPIRE_SECONDS)

    def test_public(self):
        if False:
            print('Hello World!')
        '\n        Request an access token using client_type: public\n        '
        self.client.login(username='test_user', password='123456')
        self.application.client_type = Application.CLIENT_PUBLIC
        self.application.save()
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org', 'client_id': self.application.client_id}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content.decode('utf-8'))
        self.assertEqual(content['token_type'], 'Bearer')
        self.assertEqual(content['scope'], 'read write')
        self.assertEqual(content['expires_in'], self.oauth2_settings.ACCESS_TOKEN_EXPIRE_SECONDS)

    def test_public_pkce_S256_authorize_get(self):
        if False:
            print('Hello World!')
        '\n        Request an access token using client_type: public\n        and PKCE enabled. Tests if the authorize get is successful\n        for the S256 algorithm and form data are properly passed.\n        '
        self.client.login(username='test_user', password='123456')
        self.application.client_type = Application.CLIENT_PUBLIC
        self.application.save()
        (code_verifier, code_challenge) = self.generate_pkce_codes('S256')
        query_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True, 'code_challenge': code_challenge, 'code_challenge_method': 'S256'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertContains(response, 'value="S256"', count=1, status_code=200)
        self.assertContains(response, 'value="{0}"'.format(code_challenge), count=1, status_code=200)

    def test_public_pkce_plain_authorize_get(self):
        if False:
            print('Hello World!')
        '\n        Request an access token using client_type: public\n        and PKCE enabled. Tests if the authorize get is successful\n        for the plain algorithm and form data are properly passed.\n        '
        self.client.login(username='test_user', password='123456')
        self.application.client_type = Application.CLIENT_PUBLIC
        self.application.save()
        (code_verifier, code_challenge) = self.generate_pkce_codes('plain')
        query_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True, 'code_challenge': code_challenge, 'code_challenge_method': 'plain'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertContains(response, 'value="plain"', count=1, status_code=200)
        self.assertContains(response, 'value="{0}"'.format(code_challenge), count=1, status_code=200)

    def test_public_pkce_S256(self):
        if False:
            return 10
        '\n        Request an access token using client_type: public\n        and PKCE enabled with the S256 algorithm\n        '
        self.client.login(username='test_user', password='123456')
        self.application.client_type = Application.CLIENT_PUBLIC
        self.application.save()
        (code_verifier, code_challenge) = self.generate_pkce_codes('S256')
        authorization_code = self.get_pkce_auth(code_challenge, 'S256')
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org', 'client_id': self.application.client_id, 'code_verifier': code_verifier}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content.decode('utf-8'))
        self.assertEqual(content['token_type'], 'Bearer')
        self.assertEqual(content['scope'], 'read write')
        self.assertEqual(content['expires_in'], self.oauth2_settings.ACCESS_TOKEN_EXPIRE_SECONDS)

    def test_public_pkce_plain(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Request an access token using client_type: public\n        and PKCE enabled with the plain algorithm\n        '
        self.client.login(username='test_user', password='123456')
        self.application.client_type = Application.CLIENT_PUBLIC
        self.application.save()
        (code_verifier, code_challenge) = self.generate_pkce_codes('plain')
        authorization_code = self.get_pkce_auth(code_challenge, 'plain')
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org', 'client_id': self.application.client_id, 'code_verifier': code_verifier}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content.decode('utf-8'))
        self.assertEqual(content['token_type'], 'Bearer')
        self.assertEqual(content['scope'], 'read write')
        self.assertEqual(content['expires_in'], self.oauth2_settings.ACCESS_TOKEN_EXPIRE_SECONDS)

    def test_public_pkce_invalid_algorithm(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Request an access token using client_type: public\n        and PKCE enabled with an invalid algorithm\n        '
        self.client.login(username='test_user', password='123456')
        self.application.client_type = Application.CLIENT_PUBLIC
        self.application.save()
        (code_verifier, code_challenge) = self.generate_pkce_codes('invalid')
        query_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True, 'code_challenge': code_challenge, 'code_challenge_method': 'invalid'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertEqual(response.status_code, 302)
        self.assertIn('error=invalid_request', response['Location'])

    def test_public_pkce_missing_code_challenge(self):
        if False:
            while True:
                i = 10
        '\n        Request an access token using client_type: public\n        and PKCE enabled but with the code_challenge missing\n        '
        self.oauth2_settings.PKCE_REQUIRED = True
        self.client.login(username='test_user', password='123456')
        self.application.client_type = Application.CLIENT_PUBLIC
        self.application.skip_authorization = True
        self.application.save()
        (code_verifier, code_challenge) = self.generate_pkce_codes('S256')
        query_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True, 'code_challenge_method': 'S256'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertEqual(response.status_code, 302)
        self.assertIn('error=invalid_request', response['Location'])

    def test_public_pkce_missing_code_challenge_method(self):
        if False:
            return 10
        '\n        Request an access token using client_type: public\n        and PKCE enabled but with the code_challenge_method missing\n        '
        self.client.login(username='test_user', password='123456')
        self.application.client_type = Application.CLIENT_PUBLIC
        self.application.save()
        (code_verifier, code_challenge) = self.generate_pkce_codes('S256')
        query_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True, 'code_challenge': code_challenge}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertEqual(response.status_code, 200)

    def test_public_pkce_S256_invalid_code_verifier(self):
        if False:
            print('Hello World!')
        '\n        Request an access token using client_type: public\n        and PKCE enabled with the S256 algorithm and an invalid code_verifier\n        '
        self.client.login(username='test_user', password='123456')
        self.application.client_type = Application.CLIENT_PUBLIC
        self.application.save()
        (code_verifier, code_challenge) = self.generate_pkce_codes('S256')
        authorization_code = self.get_pkce_auth(code_challenge, 'S256')
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org', 'client_id': self.application.client_id, 'code_verifier': 'invalid'}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data)
        self.assertEqual(response.status_code, 400)

    def test_public_pkce_plain_invalid_code_verifier(self):
        if False:
            return 10
        '\n        Request an access token using client_type: public\n        and PKCE enabled with the plain algorithm and an invalid code_verifier\n        '
        self.client.login(username='test_user', password='123456')
        self.application.client_type = Application.CLIENT_PUBLIC
        self.application.save()
        (code_verifier, code_challenge) = self.generate_pkce_codes('plain')
        authorization_code = self.get_pkce_auth(code_challenge, 'plain')
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org', 'client_id': self.application.client_id, 'code_verifier': 'invalid'}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data)
        self.assertEqual(response.status_code, 400)

    def test_public_pkce_S256_missing_code_verifier(self):
        if False:
            while True:
                i = 10
        '\n        Request an access token using client_type: public\n        and PKCE enabled with the S256 algorithm and the code_verifier missing\n        '
        self.client.login(username='test_user', password='123456')
        self.application.client_type = Application.CLIENT_PUBLIC
        self.application.save()
        (code_verifier, code_challenge) = self.generate_pkce_codes('S256')
        authorization_code = self.get_pkce_auth(code_challenge, 'S256')
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org', 'client_id': self.application.client_id}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data)
        self.assertEqual(response.status_code, 400)

    def test_public_pkce_plain_missing_code_verifier(self):
        if False:
            i = 10
            return i + 15
        '\n        Request an access token using client_type: public\n        and PKCE enabled with the plain algorithm and the code_verifier missing\n        '
        self.client.login(username='test_user', password='123456')
        self.application.client_type = Application.CLIENT_PUBLIC
        self.application.save()
        (code_verifier, code_challenge) = self.generate_pkce_codes('plain')
        authorization_code = self.get_pkce_auth(code_challenge, 'plain')
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org', 'client_id': self.application.client_id}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data)
        self.assertEqual(response.status_code, 400)

    def test_malicious_redirect_uri(self):
        if False:
            print('Hello World!')
        '\n        Request an access token using client_type: public and ensure redirect_uri is\n        properly validated.\n        '
        self.client.login(username='test_user', password='123456')
        self.application.client_type = Application.CLIENT_PUBLIC
        self.application.save()
        authorization_code = self.get_auth()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': '/../', 'client_id': self.application.client_id}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data['error'], 'invalid_request')
        self.assertEqual(data['error_description'], oauthlib_errors.MismatchingRedirectURIError.description)

    def test_code_exchange_succeed_when_redirect_uri_match(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests code exchange succeed when redirect uri matches the one used for code request\n        '
        self.client.login(username='test_user', password='123456')
        authcode_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org?foo=bar', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=authcode_data)
        query_dict = parse_qs(urlparse(response['Location']).query)
        authorization_code = query_dict['code'].pop()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org?foo=bar'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content.decode('utf-8'))
        self.assertEqual(content['token_type'], 'Bearer')
        self.assertEqual(content['scope'], 'read write')
        self.assertEqual(content['expires_in'], self.oauth2_settings.ACCESS_TOKEN_EXPIRE_SECONDS)

    def test_code_exchange_fails_when_redirect_uri_does_not_match(self):
        if False:
            return 10
        '\n        Tests code exchange fails when redirect uri does not match the one used for code request\n        '
        self.client.login(username='test_user', password='123456')
        authcode_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org?foo=bar', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=authcode_data)
        query_dict = parse_qs(urlparse(response['Location']).query)
        authorization_code = query_dict['code'].pop()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org?foo=baraa'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data['error'], 'invalid_request')
        self.assertEqual(data['error_description'], oauthlib_errors.MismatchingRedirectURIError.description)

    def test_code_exchange_succeed_when_redirect_uri_match_with_multiple_query_params(self):
        if False:
            while True:
                i = 10
        '\n        Tests code exchange succeed when redirect uri matches the one used for code request\n        '
        self.client.login(username='test_user', password='123456')
        self.application.redirect_uris = 'http://localhost http://example.com?foo=bar'
        self.application.save()
        authcode_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.com?bar=baz&foo=bar', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=authcode_data)
        query_dict = parse_qs(urlparse(response['Location']).query)
        authorization_code = query_dict['code'].pop()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.com?bar=baz&foo=bar'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content.decode('utf-8'))
        self.assertEqual(content['token_type'], 'Bearer')
        self.assertEqual(content['scope'], 'read write')
        self.assertEqual(content['expires_in'], self.oauth2_settings.ACCESS_TOKEN_EXPIRE_SECONDS)

@pytest.mark.oauth2_settings(presets.OIDC_SETTINGS_RW)
class TestOIDCAuthorizationCodeTokenView(BaseAuthorizationCodeTokenView):

    @classmethod
    def setUpTestData(cls):
        if False:
            while True:
                i = 10
        super().setUpTestData()
        cls.application.algorithm = Application.RS256_ALGORITHM
        cls.application.save()

    def test_id_token_public(self):
        if False:
            return 10
        '\n        Request an access token using client_type: public\n        '
        self.client.login(username='test_user', password='123456')
        self.application.client_type = Application.CLIENT_PUBLIC
        self.application.save()
        authorization_code = self.get_auth(scope='openid')
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org', 'client_id': self.application.client_id, 'scope': 'openid'}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content.decode('utf-8'))
        self.assertEqual(content['token_type'], 'Bearer')
        self.assertEqual(content['scope'], 'openid')
        self.assertIn('access_token', content)
        self.assertIn('id_token', content)
        self.assertEqual(content['expires_in'], self.oauth2_settings.ACCESS_TOKEN_EXPIRE_SECONDS)

    def test_id_token_code_exchange_succeed_when_redirect_uri_match_with_multiple_query_params(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests code exchange succeed when redirect uri matches the one used for code request\n        '
        self.client.login(username='test_user', password='123456')
        self.application.redirect_uris = 'http://localhost http://example.com?foo=bar'
        self.application.save()
        authcode_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'openid', 'redirect_uri': 'http://example.com?bar=baz&foo=bar', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=authcode_data)
        query_dict = parse_qs(urlparse(response['Location']).query)
        authorization_code = query_dict['code'].pop()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.com?bar=baz&foo=bar'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content.decode('utf-8'))
        self.assertEqual(content['token_type'], 'Bearer')
        self.assertEqual(content['scope'], 'openid')
        self.assertIn('access_token', content)
        self.assertIn('id_token', content)
        self.assertEqual(content['expires_in'], self.oauth2_settings.ACCESS_TOKEN_EXPIRE_SECONDS)

@pytest.mark.oauth2_settings(presets.OIDC_SETTINGS_RW)
class TestOIDCAuthorizationCodeHSAlgorithm(BaseAuthorizationCodeTokenView):

    @classmethod
    def setUpTestData(cls):
        if False:
            while True:
                i = 10
        super().setUpTestData()
        cls.application.algorithm = Application.HS256_ALGORITHM
        cls.application.save()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.oauth2_settings.OIDC_RSA_PRIVATE_KEY = None

    def test_id_token(self):
        if False:
            print('Hello World!')
        '\n        Request an access token using an HS256 application\n        '
        self.client.login(username='test_user', password='123456')
        authorization_code = self.get_auth(scope='openid')
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org', 'client_id': self.application.client_id, 'client_secret': CLEARTEXT_SECRET, 'scope': 'openid'}
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data)
        self.assertEqual(response.status_code, 200)
        content = response.json()
        self.assertEqual(content['token_type'], 'Bearer')
        self.assertEqual(content['scope'], 'openid')
        self.assertIn('access_token', content)
        self.assertIn('id_token', content)
        self.assertEqual(content['expires_in'], self.oauth2_settings.ACCESS_TOKEN_EXPIRE_SECONDS)
        key = self.application.jwk_key
        assert key.key_type == 'oct'
        jwt_token = jwt.JWT(key=key, jwt=content['id_token'])
        claims = json.loads(jwt_token.claims)
        assert claims['sub'] == '1'

@pytest.mark.oauth2_settings(presets.DEFAULT_SCOPES_RW)
class TestAuthorizationCodeProtectedResource(BaseTest):

    def test_resource_access_allowed(self):
        if False:
            i = 10
            return i + 15
        self.client.login(username='test_user', password='123456')
        authcode_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'read write', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=authcode_data)
        query_dict = parse_qs(urlparse(response['Location']).query)
        authorization_code = query_dict['code'].pop()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        content = json.loads(response.content.decode('utf-8'))
        access_token = content['access_token']
        auth_headers = {'HTTP_AUTHORIZATION': 'Bearer ' + access_token}
        request = self.factory.get('/fake-resource', **auth_headers)
        request.user = self.test_user
        view = ResourceView.as_view()
        response = view(request)
        self.assertEqual(response, 'This is a protected resource')

    def test_resource_access_deny(self):
        if False:
            return 10
        auth_headers = {'HTTP_AUTHORIZATION': 'Bearer ' + 'faketoken'}
        request = self.factory.get('/fake-resource', **auth_headers)
        request.user = self.test_user
        view = ResourceView.as_view()
        response = view(request)
        self.assertEqual(response.status_code, 403)

@pytest.mark.oauth2_settings(presets.OIDC_SETTINGS_RW)
class TestOIDCAuthorizationCodeProtectedResource(BaseTest):

    @classmethod
    def setUpTestData(cls):
        if False:
            return 10
        super().setUpTestData()
        cls.application.algorithm = Application.RS256_ALGORITHM
        cls.application.save()

    def test_id_token_resource_access_allowed(self):
        if False:
            return 10
        self.client.login(username='test_user', password='123456')
        authcode_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'openid', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=authcode_data)
        query_dict = parse_qs(urlparse(response['Location']).query)
        authorization_code = query_dict['code'].pop()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        content = json.loads(response.content.decode('utf-8'))
        access_token = content['access_token']
        id_token = content['id_token']
        auth_headers = {'HTTP_AUTHORIZATION': 'Bearer ' + access_token}
        request = self.factory.get('/fake-resource', **auth_headers)
        request.user = self.test_user
        view = ResourceView.as_view()
        response = view(request)
        self.assertEqual(response, 'This is a protected resource')
        auth_headers = {'HTTP_AUTHORIZATION': 'Bearer ' + id_token}
        request = self.factory.get('/fake-resource', **auth_headers)
        request.user = self.test_user
        view = ResourceView.as_view()
        response = view(request)
        self.assertEqual(response, 'This is a protected resource')

@pytest.mark.oauth2_settings(presets.DEFAULT_SCOPES_RO)
class TestDefaultScopes(BaseTest):

    def test_pre_auth_default_scopes(self):
        if False:
            while True:
                i = 10
        '\n        Test response for a valid client_id with response_type: code using default scopes\n        '
        self.client.login(username='test_user', password='123456')
        query_data = {'client_id': self.application.client_id, 'response_type': 'code', 'state': 'random_state_string', 'redirect_uri': 'http://example.org'}
        response = self.client.get(reverse('oauth2_provider:authorize'), data=query_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('form', response.context)
        form = response.context['form']
        self.assertEqual(form['redirect_uri'].value(), 'http://example.org')
        self.assertEqual(form['state'].value(), 'random_state_string')
        self.assertEqual(form['scope'].value(), 'read')
        self.assertEqual(form['client_id'].value(), self.application.client_id)