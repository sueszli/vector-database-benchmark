from unittest import TestCase, mock
import pytest
from django.contrib.auth import get_user_model
from django.test import RequestFactory
from django.utils import timezone
from pytest_django.fixtures import SettingsWrapper
from pytest_mock import MockerFixture
from rest_framework.authtoken.models import Token
from custom_auth.oauth.serializers import GithubLoginSerializer, GoogleLoginSerializer, OAuthLoginSerializer
from users.models import SignUpType
UserModel = get_user_model()

@pytest.mark.django_db
class OAuthLoginSerializerTestCase(TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.test_email = 'testytester@example.com'
        self.test_first_name = 'testy'
        self.test_last_name = 'tester'
        self.test_id = 'test-id'
        self.mock_user_data = {'email': self.test_email, 'first_name': self.test_first_name, 'last_name': self.test_last_name, 'google_user_id': self.test_id}
        rf = RequestFactory()
        self.request = rf.post('placeholer-login-url')

    @mock.patch('custom_auth.oauth.serializers.get_user_info')
    def test_create(self, mock_get_user_info):
        if False:
            for i in range(10):
                print('nop')
        access_token = 'access-token'
        sign_up_type = 'NO_INVITE'
        data = {'access_token': access_token, 'sign_up_type': sign_up_type}
        serializer = OAuthLoginSerializer(data=data, context={'request': self.request})
        serializer.get_user_info = lambda : self.mock_user_data
        serializer.is_valid()
        response = serializer.save()
        assert UserModel.objects.filter(email=self.test_email, sign_up_type=sign_up_type).exists()
        assert isinstance(response, Token)
        assert (timezone.now() - response.user.last_login).seconds < 5
        assert response.user.email == self.test_email

class GoogleLoginSerializerTestCase(TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        rf = RequestFactory()
        self.request = rf.post('placeholer-login-url')

    @mock.patch('custom_auth.oauth.serializers.get_user_info')
    def test_get_user_info(self, mock_get_user_info):
        if False:
            print('Hello World!')
        access_token = 'some-access-token'
        serializer = GoogleLoginSerializer(data={'access_token': access_token}, context={'request': self.request})
        serializer.is_valid()
        serializer.get_user_info()
        mock_get_user_info.assert_called_with(access_token)

class GithubLoginSerializerTestCase(TestCase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        rf = RequestFactory()
        self.request = rf.post('placeholer-login-url')

    @mock.patch('custom_auth.oauth.serializers.GithubUser')
    def test_get_user_info(self, MockGithubUser):
        if False:
            i = 10
            return i + 15
        access_token = 'some-access-token'
        serializer = GithubLoginSerializer(data={'access_token': access_token}, context={'request': self.request})
        mock_github_user = mock.MagicMock()
        MockGithubUser.return_value = mock_github_user
        serializer.is_valid()
        serializer.get_user_info()
        MockGithubUser.assert_called_with(code=access_token)
        mock_github_user.get_user_info.assert_called()

def test_OAuthLoginSerializer_calls_is_authentication_method_valid_correctly_if_auth_controller_is_installed(settings, rf, mocker, db):
    if False:
        print('Hello World!')
    settings.AUTH_CONTROLLER_INSTALLED = True
    request = rf.post('/some-login/url')
    user_email = 'test_user@test.com'
    mocked_auth_controller = mocker.MagicMock()
    mocker.patch.dict('sys.modules', {'auth_controller.controller': mocked_auth_controller})
    serializer = OAuthLoginSerializer(data={'access_token': 'some_token'}, context={'request': request})
    serializer.get_user_info = lambda : {'email': user_email}
    serializer.is_valid(raise_exception=True)
    serializer.save()
    mocked_auth_controller.is_authentication_method_valid.assert_called_with(request, email=user_email, raise_exception=True)

def test_OAuthLoginSerializer_allows_registration_if_sign_up_type_is_invite_link(settings: SettingsWrapper, rf: RequestFactory, mocker: MockerFixture, db: None):
    if False:
        for i in range(10):
            print('nop')
    settings.ALLOW_REGISTRATION_WITHOUT_INVITE = False
    request = rf.post('/api/v1/auth/users/')
    user_email = 'test_user@test.com'
    serializer = OAuthLoginSerializer(data={'access_token': 'some_token', 'sign_up_type': SignUpType.INVITE_LINK.value}, context={'request': request})
    serializer.get_user_info = lambda : {'email': user_email}
    serializer.is_valid(raise_exception=True)
    user = serializer.save()
    assert user