import calendar
import datetime
import pytest
from django.conf import settings
from django.conf.urls import include
from django.contrib.auth import get_user_model
from django.http import HttpResponse
from django.test import TestCase, override_settings
from django.urls import path
from django.utils import timezone
from oauthlib.common import Request
from oauth2_provider.models import get_access_token_model, get_application_model
from oauth2_provider.oauth2_validators import OAuth2Validator
from oauth2_provider.settings import oauth2_settings
from oauth2_provider.views import ScopedProtectedResourceView
from . import presets
try:
    from unittest import mock
except ImportError:
    import mock
Application = get_application_model()
AccessToken = get_access_token_model()
UserModel = get_user_model()
exp = datetime.datetime.now() + datetime.timedelta(days=1)

class ScopeResourceView(ScopedProtectedResourceView):
    required_scopes = ['dolphin']

    def get(self, request, *args, **kwargs):
        if False:
            while True:
                i = 10
        return HttpResponse('This is a protected resource', 200)

    def post(self, request, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return HttpResponse('This is a protected resource', 200)

def mocked_requests_post(url, data, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Mock the response from the authentication server\n    '

    class MockResponse:

        def __init__(self, json_data, status_code):
            if False:
                for i in range(10):
                    print('nop')
            self.json_data = json_data
            self.status_code = status_code

        def json(self):
            if False:
                i = 10
                return i + 15
            return self.json_data
    if 'token' in data and data['token'] and (data['token'] != '12345678900'):
        return MockResponse({'active': True, 'scope': 'read write dolphin', 'client_id': 'client_id_{}'.format(data['token']), 'username': '{}_user'.format(data['token']), 'exp': int(calendar.timegm(exp.timetuple()))}, 200)
    return MockResponse({'active': False}, 200)
urlpatterns = [path('oauth2/', include('oauth2_provider.urls')), path('oauth2-test-resource/', ScopeResourceView.as_view())]

@override_settings(ROOT_URLCONF=__name__)
@pytest.mark.usefixtures('oauth2_settings')
@pytest.mark.oauth2_settings(presets.INTROSPECTION_SETTINGS)
class TestTokenIntrospectionAuth(TestCase):
    """
    Tests for Authorization through token introspection
    """

    @classmethod
    def setUpTestData(cls):
        if False:
            i = 10
            return i + 15
        cls.validator = OAuth2Validator()
        cls.request = mock.MagicMock(wraps=Request)
        cls.resource_server_user = UserModel.objects.create_user('resource_server', 'test@example.com', '123456')
        cls.application = Application.objects.create(name='Test Application', redirect_uris='http://localhost http://example.com http://example.org', user=cls.resource_server_user, client_type=Application.CLIENT_CONFIDENTIAL, authorization_grant_type=Application.GRANT_AUTHORIZATION_CODE)
        cls.resource_server_token = AccessToken.objects.create(user=cls.resource_server_user, token='12345678900', application=cls.application, expires=timezone.now() + datetime.timedelta(days=1), scope='introspection')
        cls.invalid_token = AccessToken.objects.create(user=cls.resource_server_user, token='12345678901', application=cls.application, expires=timezone.now() + datetime.timedelta(days=-1), scope='read write dolphin')

    def setUp(self):
        if False:
            while True:
                i = 10
        self.oauth2_settings.RESOURCE_SERVER_AUTH_TOKEN = self.resource_server_token.token

    @mock.patch('requests.post', side_effect=mocked_requests_post)
    def test_get_token_from_authentication_server_not_existing_token(self, mock_get):
        if False:
            i = 10
            return i + 15
        '\n        Test method _get_token_from_authentication_server with non existing token\n        '
        token = self.validator._get_token_from_authentication_server(self.resource_server_token.token, self.oauth2_settings.RESOURCE_SERVER_INTROSPECTION_URL, self.oauth2_settings.RESOURCE_SERVER_AUTH_TOKEN, self.oauth2_settings.RESOURCE_SERVER_INTROSPECTION_CREDENTIALS)
        self.assertIsNone(token)

    @mock.patch('requests.post', side_effect=mocked_requests_post)
    def test_get_token_from_authentication_server_existing_token(self, mock_get):
        if False:
            print('Hello World!')
        '\n        Test method _get_token_from_authentication_server with existing token\n        '
        token = self.validator._get_token_from_authentication_server('foo', self.oauth2_settings.RESOURCE_SERVER_INTROSPECTION_URL, self.oauth2_settings.RESOURCE_SERVER_AUTH_TOKEN, self.oauth2_settings.RESOURCE_SERVER_INTROSPECTION_CREDENTIALS)
        self.assertIsInstance(token, AccessToken)
        self.assertEqual(token.user.username, 'foo_user')
        self.assertEqual(token.scope, 'read write dolphin')

    @mock.patch('requests.post', side_effect=mocked_requests_post)
    def test_get_token_from_authentication_server_expires_timezone(self, mock_get):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test method _get_token_from_authentication_server for projects with USE_TZ False\n        '
        settings_use_tz_backup = settings.USE_TZ
        settings.USE_TZ = False
        try:
            self.validator._get_token_from_authentication_server('foo', oauth2_settings.RESOURCE_SERVER_INTROSPECTION_URL, oauth2_settings.RESOURCE_SERVER_AUTH_TOKEN, oauth2_settings.RESOURCE_SERVER_INTROSPECTION_CREDENTIALS)
        except ValueError as exception:
            self.fail(str(exception))
        finally:
            settings.USE_TZ = settings_use_tz_backup

    @mock.patch('requests.post', side_effect=mocked_requests_post)
    def test_validate_bearer_token(self, mock_get):
        if False:
            i = 10
            return i + 15
        '\n        Test method validate_bearer_token\n        '
        self.assertFalse(self.validator.validate_bearer_token(None, ['dolphin'], self.request))
        self.assertTrue(self.validator.validate_bearer_token(self.resource_server_token.token, ['introspection'], self.request))
        self.assertTrue(self.validator.validate_bearer_token(self.invalid_token.token, ['dolphin'], self.request))
        self.assertTrue(self.validator.validate_bearer_token('butzi', ['dolphin'], self.request))
        self.assertFalse(self.validator.validate_bearer_token('foo', ['kaudawelsch'], self.request))
        self.assertFalse(self.validator.validate_bearer_token('butz', ['kaudawelsch'], self.request))
        self.assertTrue(self.validator.validate_bearer_token('butzi', ['dolphin'], self.request))

    @mock.patch('requests.post', side_effect=mocked_requests_post)
    def test_get_resource(self, mock_get):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that we can access the resource with a get request and a remotely validated token\n        '
        auth_headers = {'HTTP_AUTHORIZATION': 'Bearer bar'}
        response = self.client.get('/oauth2-test-resource/', **auth_headers)
        self.assertEqual(response.content.decode('utf-8'), 'This is a protected resource')

    @mock.patch('requests.post', side_effect=mocked_requests_post)
    def test_post_resource(self, mock_get):
        if False:
            return 10
        '\n        Test that we can access the resource with a post request and a remotely validated token\n        '
        auth_headers = {'HTTP_AUTHORIZATION': 'Bearer batz'}
        response = self.client.post('/oauth2-test-resource/', **auth_headers)
        self.assertEqual(response.content.decode('utf-8'), 'This is a protected resource')