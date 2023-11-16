import json
import pytest
from django.contrib.auth import get_user_model
from django.test import RequestFactory, TestCase
from django.urls import reverse
from oauth2_provider.models import get_application_model
from oauth2_provider.views import ProtectedResourceView
from .utils import get_basic_auth_header
Application = get_application_model()
UserModel = get_user_model()
CLEARTEXT_SECRET = '1234567890abcdefghijklmnopqrstuvwxyz'

class ResourceView(ProtectedResourceView):

    def get(self, request, *args, **kwargs):
        if False:
            return 10
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
        cls.application = Application.objects.create(name='Test Password Application', user=cls.dev_user, client_type=Application.CLIENT_PUBLIC, authorization_grant_type=Application.GRANT_PASSWORD, client_secret=CLEARTEXT_SECRET)

class TestPasswordTokenView(BaseTest):

    def test_get_token(self):
        if False:
            while True:
                i = 10
        '\n        Request an access token using Resource Owner Password Flow\n        '
        token_request_data = {'grant_type': 'password', 'username': 'test_user', 'password': '123456'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content.decode('utf-8'))
        self.assertEqual(content['token_type'], 'Bearer')
        self.assertEqual(set(content['scope'].split()), {'read', 'write'})
        self.assertEqual(content['expires_in'], self.oauth2_settings.ACCESS_TOKEN_EXPIRE_SECONDS)

    def test_bad_credentials(self):
        if False:
            print('Hello World!')
        '\n        Request an access token using Resource Owner Password Flow\n        '
        token_request_data = {'grant_type': 'password', 'username': 'test_user', 'password': 'NOT_MY_PASS'}
        auth_headers = get_basic_auth_header(self.application.client_id, CLEARTEXT_SECRET)
        response = self.client.post(reverse('oauth2_provider:token'), data=token_request_data, **auth_headers)
        self.assertEqual(response.status_code, 400)

class TestPasswordProtectedResource(BaseTest):

    def test_password_resource_access_allowed(self):
        if False:
            while True:
                i = 10
        token_request_data = {'grant_type': 'password', 'username': 'test_user', 'password': '123456'}
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