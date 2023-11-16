from datetime import datetime, timezone
from django.conf import settings
from django.contrib.auth import authenticate
from django.contrib.auth.backends import RemoteUserBackend
from django.contrib.auth.middleware import RemoteUserMiddleware
from django.contrib.auth.models import User
from django.middleware.csrf import _get_new_csrf_string, _mask_cipher_secret
from django.test import Client, TestCase, modify_settings, override_settings

@override_settings(ROOT_URLCONF='auth_tests.urls')
class RemoteUserTest(TestCase):
    middleware = 'django.contrib.auth.middleware.RemoteUserMiddleware'
    backend = 'django.contrib.auth.backends.RemoteUserBackend'
    header = 'REMOTE_USER'
    email_header = 'REMOTE_EMAIL'
    known_user = 'knownuser'
    known_user2 = 'knownuser2'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.patched_settings = modify_settings(AUTHENTICATION_BACKENDS={'append': self.backend}, MIDDLEWARE={'append': self.middleware})
        self.patched_settings.enable()

    def tearDown(self):
        if False:
            return 10
        self.patched_settings.disable()

    def test_no_remote_user(self):
        if False:
            i = 10
            return i + 15
        'Users are not created when remote user is not specified.'
        num_users = User.objects.count()
        response = self.client.get('/remote_user/')
        self.assertTrue(response.context['user'].is_anonymous)
        self.assertEqual(User.objects.count(), num_users)
        response = self.client.get('/remote_user/', **{self.header: None})
        self.assertTrue(response.context['user'].is_anonymous)
        self.assertEqual(User.objects.count(), num_users)
        response = self.client.get('/remote_user/', **{self.header: ''})
        self.assertTrue(response.context['user'].is_anonymous)
        self.assertEqual(User.objects.count(), num_users)

    def test_csrf_validation_passes_after_process_request_login(self):
        if False:
            i = 10
            return i + 15
        '\n        CSRF check must access the CSRF token from the session or cookie,\n        rather than the request, as rotate_token() may have been called by an\n        authentication middleware during the process_request() phase.\n        '
        csrf_client = Client(enforce_csrf_checks=True)
        csrf_secret = _get_new_csrf_string()
        csrf_token = _mask_cipher_secret(csrf_secret)
        csrf_token_form = _mask_cipher_secret(csrf_secret)
        headers = {self.header: 'fakeuser'}
        data = {'csrfmiddlewaretoken': csrf_token_form}
        csrf_client.cookies.load({settings.CSRF_COOKIE_NAME: csrf_token})
        response = csrf_client.post('/remote_user/', **headers)
        self.assertEqual(response.status_code, 403)
        self.assertIn(b'CSRF verification failed.', response.content)
        csrf_client.cookies.load({settings.CSRF_COOKIE_NAME: csrf_token})
        response = csrf_client.post('/remote_user/', data, **headers)
        self.assertEqual(response.status_code, 200)

    def test_unknown_user(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests the case where the username passed in the header does not exist\n        as a User.\n        '
        num_users = User.objects.count()
        response = self.client.get('/remote_user/', **{self.header: 'newuser'})
        self.assertEqual(response.context['user'].username, 'newuser')
        self.assertEqual(User.objects.count(), num_users + 1)
        User.objects.get(username='newuser')
        response = self.client.get('/remote_user/', **{self.header: 'newuser'})
        self.assertEqual(User.objects.count(), num_users + 1)

    def test_known_user(self):
        if False:
            while True:
                i = 10
        '\n        Tests the case where the username passed in the header is a valid User.\n        '
        User.objects.create(username='knownuser')
        User.objects.create(username='knownuser2')
        num_users = User.objects.count()
        response = self.client.get('/remote_user/', **{self.header: self.known_user})
        self.assertEqual(response.context['user'].username, 'knownuser')
        self.assertEqual(User.objects.count(), num_users)
        response = self.client.get('/remote_user/', **{self.header: self.known_user2})
        self.assertEqual(response.context['user'].username, 'knownuser2')
        self.assertEqual(User.objects.count(), num_users)

    def test_last_login(self):
        if False:
            while True:
                i = 10
        "\n        A user's last_login is set the first time they make a\n        request but not updated in subsequent requests with the same session.\n        "
        user = User.objects.create(username='knownuser')
        default_login = datetime(2000, 1, 1)
        if settings.USE_TZ:
            default_login = default_login.replace(tzinfo=timezone.utc)
        user.last_login = default_login
        user.save()
        response = self.client.get('/remote_user/', **{self.header: self.known_user})
        self.assertNotEqual(default_login, response.context['user'].last_login)
        user = User.objects.get(username='knownuser')
        user.last_login = default_login
        user.save()
        response = self.client.get('/remote_user/', **{self.header: self.known_user})
        self.assertEqual(default_login, response.context['user'].last_login)

    def test_header_disappears(self):
        if False:
            i = 10
            return i + 15
        '\n        A logged in user is logged out automatically when\n        the REMOTE_USER header disappears during the same browser session.\n        '
        User.objects.create(username='knownuser')
        response = self.client.get('/remote_user/', **{self.header: self.known_user})
        self.assertEqual(response.context['user'].username, 'knownuser')
        response = self.client.get('/remote_user/')
        self.assertTrue(response.context['user'].is_anonymous)
        User.objects.create_user(username='modeluser', password='foo')
        self.client.login(username='modeluser', password='foo')
        authenticate(username='modeluser', password='foo')
        response = self.client.get('/remote_user/')
        self.assertEqual(response.context['user'].username, 'modeluser')

    def test_user_switch_forces_new_login(self):
        if False:
            print('Hello World!')
        '\n        If the username in the header changes between requests\n        that the original user is logged out\n        '
        User.objects.create(username='knownuser')
        response = self.client.get('/remote_user/', **{self.header: self.known_user})
        self.assertEqual(response.context['user'].username, 'knownuser')
        response = self.client.get('/remote_user/', **{self.header: 'newnewuser'})
        self.assertNotEqual(response.context['user'].username, 'knownuser')

    def test_inactive_user(self):
        if False:
            while True:
                i = 10
        User.objects.create(username='knownuser', is_active=False)
        response = self.client.get('/remote_user/', **{self.header: 'knownuser'})
        self.assertTrue(response.context['user'].is_anonymous)

class RemoteUserNoCreateBackend(RemoteUserBackend):
    """Backend that doesn't create unknown users."""
    create_unknown_user = False

class RemoteUserNoCreateTest(RemoteUserTest):
    """
    Contains the same tests as RemoteUserTest, but using a custom auth backend
    class that doesn't create unknown users.
    """
    backend = 'auth_tests.test_remote_user.RemoteUserNoCreateBackend'

    def test_unknown_user(self):
        if False:
            return 10
        num_users = User.objects.count()
        response = self.client.get('/remote_user/', **{self.header: 'newuser'})
        self.assertTrue(response.context['user'].is_anonymous)
        self.assertEqual(User.objects.count(), num_users)

class AllowAllUsersRemoteUserBackendTest(RemoteUserTest):
    """Backend that allows inactive users."""
    backend = 'django.contrib.auth.backends.AllowAllUsersRemoteUserBackend'

    def test_inactive_user(self):
        if False:
            return 10
        user = User.objects.create(username='knownuser', is_active=False)
        response = self.client.get('/remote_user/', **{self.header: self.known_user})
        self.assertEqual(response.context['user'].username, user.username)

class CustomRemoteUserBackend(RemoteUserBackend):
    """
    Backend that overrides RemoteUserBackend methods.
    """

    def clean_username(self, username):
        if False:
            return 10
        '\n        Grabs username before the @ character.\n        '
        return username.split('@')[0]

    def configure_user(self, request, user, created=True):
        if False:
            print('Hello World!')
        "\n        Sets user's email address using the email specified in an HTTP header.\n        Sets user's last name for existing users.\n        "
        user.email = request.META.get(RemoteUserTest.email_header, '')
        if not created:
            user.last_name = user.username
        user.save()
        return user

class RemoteUserCustomTest(RemoteUserTest):
    """
    Tests a custom RemoteUserBackend subclass that overrides the clean_username
    and configure_user methods.
    """
    backend = 'auth_tests.test_remote_user.CustomRemoteUserBackend'
    known_user = 'knownuser@example.com'
    known_user2 = 'knownuser2@example.com'

    def test_known_user(self):
        if False:
            while True:
                i = 10
        '\n        The strings passed in REMOTE_USER should be cleaned and the known users\n        should not have been configured with an email address.\n        '
        super().test_known_user()
        knownuser = User.objects.get(username='knownuser')
        knownuser2 = User.objects.get(username='knownuser2')
        self.assertEqual(knownuser.email, '')
        self.assertEqual(knownuser2.email, '')
        self.assertEqual(knownuser.last_name, 'knownuser')
        self.assertEqual(knownuser2.last_name, 'knownuser2')

    def test_unknown_user(self):
        if False:
            i = 10
            return i + 15
        '\n        The unknown user created should be configured with an email address\n        provided in the request header.\n        '
        num_users = User.objects.count()
        response = self.client.get('/remote_user/', **{self.header: 'newuser', self.email_header: 'user@example.com'})
        self.assertEqual(response.context['user'].username, 'newuser')
        self.assertEqual(response.context['user'].email, 'user@example.com')
        self.assertEqual(response.context['user'].last_name, '')
        self.assertEqual(User.objects.count(), num_users + 1)
        newuser = User.objects.get(username='newuser')
        self.assertEqual(newuser.email, 'user@example.com')

class CustomHeaderMiddleware(RemoteUserMiddleware):
    """
    Middleware that overrides custom HTTP auth user header.
    """
    header = 'HTTP_AUTHUSER'

class CustomHeaderRemoteUserTest(RemoteUserTest):
    """
    Tests a custom RemoteUserMiddleware subclass with custom HTTP auth user
    header.
    """
    middleware = 'auth_tests.test_remote_user.CustomHeaderMiddleware'
    header = 'HTTP_AUTHUSER'

class PersistentRemoteUserTest(RemoteUserTest):
    """
    PersistentRemoteUserMiddleware keeps the user logged in even if the
    subsequent calls do not contain the header value.
    """
    middleware = 'django.contrib.auth.middleware.PersistentRemoteUserMiddleware'
    require_header = False

    def test_header_disappears(self):
        if False:
            return 10
        '\n        A logged in user is kept logged in even if the REMOTE_USER header\n        disappears during the same browser session.\n        '
        User.objects.create(username='knownuser')
        response = self.client.get('/remote_user/', **{self.header: self.known_user})
        self.assertEqual(response.context['user'].username, 'knownuser')
        response = self.client.get('/remote_user/')
        self.assertFalse(response.context['user'].is_anonymous)
        self.assertEqual(response.context['user'].username, 'knownuser')