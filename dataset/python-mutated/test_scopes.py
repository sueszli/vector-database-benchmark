import json
from urllib.parse import parse_qs, urlparse
import pytest
from django.contrib.auth import get_user_model
from django.core.exceptions import ImproperlyConfigured
from django.test import RequestFactory, TestCase
from django.urls import reverse
from oauth2_provider.models import get_access_token_model, get_application_model, get_grant_model
from oauth2_provider.views import ReadWriteScopedResourceView, ScopedProtectedResourceView
from .utils import get_basic_auth_header
Application = get_application_model()
AccessToken = get_access_token_model()
Grant = get_grant_model()
UserModel = get_user_model()
CLEARTEXT_SECRET = '1234567890abcdefghijklmnopqrstuvwxyz'

class ScopeResourceView(ScopedProtectedResourceView):
    required_scopes = ['scope1']

    def get(self, request, *args, **kwargs):
        if False:
            return 10
        return 'This is a protected resource'

class MultiScopeResourceView(ScopedProtectedResourceView):
    required_scopes = ['scope1', 'scope2']

    def get(self, request, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return 'This is a protected resource'

class ReadWriteResourceView(ReadWriteScopedResourceView):

    def get(self, request, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return 'This is a read protected resource'

    def post(self, request, *args, **kwargs):
        if False:
            while True:
                i = 10
        return 'This is a write protected resource'
SCOPE_SETTINGS = {'SCOPES': {'read': 'Read scope', 'write': 'Write scope', 'scope1': 'Custom scope 1', 'scope2': 'Custom scope 2', 'scope3': 'Custom scope 3'}}

@pytest.mark.usefixtures('oauth2_settings')
@pytest.mark.oauth2_settings(SCOPE_SETTINGS)
class BaseTest(TestCase):
    factory = RequestFactory()

    @classmethod
    def setUpTestData(cls):
        if False:
            return 10
        cls.test_user = UserModel.objects.create_user('test_user', 'test@example.com', '123456')
        cls.dev_user = UserModel.objects.create_user('dev_user', 'dev@example.com', '123456')
        cls.application = Application.objects.create(name='Test Application', redirect_uris='http://localhost http://example.com http://example.org', user=cls.dev_user, client_type=Application.CLIENT_CONFIDENTIAL, authorization_grant_type=Application.GRANT_AUTHORIZATION_CODE, client_secret=CLEARTEXT_SECRET)

class TestScopesSave(BaseTest):

    def test_scopes_saved_in_grant(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test scopes are properly saved in grant\n        '
        self.oauth2_settings.PKCE_REQUIRED = False
        self.client.login(username='test_user', password='123456')
        authcode_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'scope1 scope2', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=authcode_data)
        query_dict = parse_qs(urlparse(response['Location']).query)
        authorization_code = query_dict['code'].pop()
        grant = Grant.objects.get(code=authorization_code)
        self.assertEqual(grant.scope, 'scope1 scope2')

    def test_scopes_save_in_access_token(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test scopes are properly saved in access token\n        '
        self.oauth2_settings.PKCE_REQUIRED = False
        self.client.login(username='test_user', password='123456')
        authcode_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'scope1 scope2', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=authcode_data)
        query_dict = parse_qs(urlparse(response['Location']).query)
        authorization_code = query_dict['code'].pop()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        content = json.loads(response.content.decode('utf-8'))
        access_token = content['access_token']
        at = AccessToken.objects.get(token=access_token)
        self.assertEqual(at.scope, 'scope1 scope2')

class TestScopesProtection(BaseTest):

    def test_scopes_protection_valid(self):
        if False:
            print('Hello World!')
        '\n        Test access to a scope protected resource with correct scopes provided\n        '
        self.oauth2_settings.PKCE_REQUIRED = False
        self.client.login(username='test_user', password='123456')
        authcode_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'scope1 scope2', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True}
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
        view = ScopeResourceView.as_view()
        response = view(request)
        self.assertEqual(response, 'This is a protected resource')

    def test_scopes_protection_fail(self):
        if False:
            i = 10
            return i + 15
        '\n        Test access to a scope protected resource with wrong scopes provided\n        '
        self.oauth2_settings.PKCE_REQUIRED = False
        self.client.login(username='test_user', password='123456')
        authcode_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'scope2', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True}
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
        view = ScopeResourceView.as_view()
        response = view(request)
        self.assertEqual(response.status_code, 403)

    def test_multi_scope_fail(self):
        if False:
            i = 10
            return i + 15
        '\n        Test access to a multi-scope protected resource with wrong scopes provided\n        '
        self.oauth2_settings.PKCE_REQUIRED = False
        self.client.login(username='test_user', password='123456')
        authcode_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'scope1 scope3', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True}
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
        view = MultiScopeResourceView.as_view()
        response = view(request)
        self.assertEqual(response.status_code, 403)

    def test_multi_scope_valid(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test access to a multi-scope protected resource with correct scopes provided\n        '
        self.oauth2_settings.PKCE_REQUIRED = False
        self.client.login(username='test_user', password='123456')
        authcode_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': 'scope1 scope2', 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True}
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
        view = MultiScopeResourceView.as_view()
        response = view(request)
        self.assertEqual(response, 'This is a protected resource')

class TestReadWriteScope(BaseTest):

    def get_access_token(self, scopes):
        if False:
            for i in range(10):
                print('nop')
        self.oauth2_settings.PKCE_REQUIRED = False
        self.client.login(username='test_user', password='123456')
        authcode_data = {'client_id': self.application.client_id, 'state': 'random_state_string', 'scope': scopes, 'redirect_uri': 'http://example.org', 'response_type': 'code', 'allow': True}
        response = self.client.post(reverse('oauth2_provider:authorize'), data=authcode_data)
        query_dict = parse_qs(urlparse(response['Location']).query)
        authorization_code = query_dict['code'].pop()
        token_request_data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': 'http://example.org'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        content = json.loads(response.content.decode('utf-8'))
        return content['access_token']

    def test_improperly_configured(self):
        if False:
            print('Hello World!')
        self.oauth2_settings.SCOPES = {'scope1': 'Scope 1'}
        request = self.factory.get('/fake')
        view = ReadWriteResourceView.as_view()
        self.assertRaises(ImproperlyConfigured, view, request)
        self.oauth2_settings.SCOPES = {'read': 'Read Scope', 'write': 'Write Scope'}
        self.oauth2_settings.READ_SCOPE = 'ciccia'
        view = ReadWriteResourceView.as_view()
        self.assertRaises(ImproperlyConfigured, view, request)

    def test_properly_configured(self):
        if False:
            i = 10
            return i + 15
        self.oauth2_settings.SCOPES = {'scope1': 'Scope 1'}
        request = self.factory.get('/fake')
        view = ReadWriteResourceView.as_view()
        self.assertRaises(ImproperlyConfigured, view, request)
        self.oauth2_settings.SCOPES = {'read': 'Read Scope', 'write': 'Write Scope'}
        self.oauth2_settings.READ_SCOPE = 'ciccia'
        view = ReadWriteResourceView.as_view()
        self.assertRaises(ImproperlyConfigured, view, request)

    def test_has_read_scope(self):
        if False:
            i = 10
            return i + 15
        access_token = self.get_access_token('read')
        auth_headers = {'HTTP_AUTHORIZATION': 'Bearer ' + access_token}
        request = self.factory.get('/fake-resource', **auth_headers)
        request.user = self.test_user
        view = ReadWriteResourceView.as_view()
        response = view(request)
        self.assertEqual(response, 'This is a read protected resource')

    def test_no_read_scope(self):
        if False:
            i = 10
            return i + 15
        access_token = self.get_access_token('scope1')
        auth_headers = {'HTTP_AUTHORIZATION': 'Bearer ' + access_token}
        request = self.factory.get('/fake-resource', **auth_headers)
        request.user = self.test_user
        view = ReadWriteResourceView.as_view()
        response = view(request)
        self.assertEqual(response.status_code, 403)

    def test_has_write_scope(self):
        if False:
            while True:
                i = 10
        access_token = self.get_access_token('write')
        auth_headers = {'HTTP_AUTHORIZATION': 'Bearer ' + access_token}
        request = self.factory.post('/fake-resource', **auth_headers)
        request.user = self.test_user
        view = ReadWriteResourceView.as_view()
        response = view(request)
        self.assertEqual(response, 'This is a write protected resource')

    def test_no_write_scope(self):
        if False:
            for i in range(10):
                print('nop')
        access_token = self.get_access_token('scope1')
        auth_headers = {'HTTP_AUTHORIZATION': 'Bearer ' + access_token}
        request = self.factory.post('/fake-resource', **auth_headers)
        request.user = self.test_user
        view = ReadWriteResourceView.as_view()
        response = view(request)
        self.assertEqual(response.status_code, 403)