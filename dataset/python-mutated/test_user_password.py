from django.test import override_settings
from sentry.models.user import User
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import control_silo_test

@control_silo_test(stable=True)
class UserPasswordTest(APITestCase):
    endpoint = 'sentry-api-0-user-password'
    method = 'put'

    def setUp(self):
        if False:
            return 10
        self.user = self.create_user(email='a@example.com', is_managed=False, name='example name')
        self.user.set_password('helloworld!')
        self.user.save()
        self.login_as(self.user)

    def test_change_password(self):
        if False:
            print('Hello World!')
        old_password = self.user.password
        self.get_success_response('me', status_code=204, **{'password': 'helloworld!', 'passwordNew': 'testpassword', 'passwordVerify': 'testpassword'})
        user = User.objects.get(id=self.user.id)
        assert old_password != user.password

    @override_settings(AUTH_PASSWORD_VALIDATORS=[{'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', 'OPTIONS': {'min_length': 8}}])
    def test_password_too_short(self):
        if False:
            for i in range(10):
                print('nop')
        self.get_error_response('me', status_code=400, **{'password': 'helloworld!', 'passwordNew': 'hi', 'passwordVerify': 'hi'})

    def test_no_password(self):
        if False:
            for i in range(10):
                print('nop')
        self.get_error_response('me', status_code=400, **{'password': 'helloworld!'})
        self.get_error_response('me', status_code=400)

    def test_require_current_password(self):
        if False:
            print('Hello World!')
        self.get_error_response('me', status_code=400, **{'password': 'wrongpassword', 'passwordNew': 'testpassword', 'passwordVerify': 'passworddoesntmatch'})

    def test_verifies_mismatch_password(self):
        if False:
            print('Hello World!')
        self.get_error_response('me', status_code=400, **{'password': 'helloworld!', 'passwordNew': 'testpassword', 'passwordVerify': 'passworddoesntmatch'})

    def test_managed_unable_change_password(self):
        if False:
            i = 10
            return i + 15
        user = self.create_user(email='new@example.com', is_managed=True)
        self.login_as(user)
        self.get_error_response(user.id, status_code=400, **{'passwordNew': 'newpassword', 'passwordVerify': 'newpassword'})

    def test_unusable_password_unable_change_password(self):
        if False:
            print('Hello World!')
        user = self.create_user(email='new@example.com')
        user.set_unusable_password()
        user.save()
        self.login_as(user)
        self.get_error_response(user.id, status_code=400, **{'passwordNew': 'newpassword', 'passwordVerify': 'newpassword'})

    def test_cannot_change_other_user_password(self):
        if False:
            while True:
                i = 10
        user = self.create_user(email='new@example.com', is_superuser=False)
        self.login_as(user)
        self.get_error_response(self.user.id, status_code=403, **{'password': 'helloworld!', 'passwordNew': 'newpassword', 'passwordVerify': 'newpassword'})

    def test_superuser_can_change_other_user_password(self):
        if False:
            for i in range(10):
                print('nop')
        user = self.create_user(email='new@example.com', is_superuser=True)
        self.login_as(user, superuser=True)
        self.get_success_response(self.user.id, status_code=204, **{'password': 'helloworld!', 'passwordNew': 'newpassword', 'passwordVerify': 'newpassword'})