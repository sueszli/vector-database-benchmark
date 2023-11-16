from threading import local
from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend
from . import app_settings
from .app_settings import AuthenticationMethod
from .utils import filter_users_by_email, filter_users_by_username
_stash = local()

class AuthenticationBackend(ModelBackend):

    def authenticate(self, request, **credentials):
        if False:
            for i in range(10):
                print('nop')
        ret = None
        if app_settings.AUTHENTICATION_METHOD == AuthenticationMethod.EMAIL:
            ret = self._authenticate_by_email(**credentials)
        elif app_settings.AUTHENTICATION_METHOD == AuthenticationMethod.USERNAME_EMAIL:
            ret = self._authenticate_by_email(**credentials)
            if not ret:
                ret = self._authenticate_by_username(**credentials)
        else:
            ret = self._authenticate_by_username(**credentials)
        return ret

    def _authenticate_by_username(self, **credentials):
        if False:
            while True:
                i = 10
        username_field = app_settings.USER_MODEL_USERNAME_FIELD
        username = credentials.get('username')
        password = credentials.get('password')
        User = get_user_model()
        if not username_field or username is None or password is None:
            return None
        try:
            user = filter_users_by_username(username).get()
        except User.DoesNotExist:
            get_user_model()().set_password(password)
            return None
        else:
            if self._check_password(user, password):
                return user

    def _authenticate_by_email(self, **credentials):
        if False:
            print('Hello World!')
        email = credentials.get('email', credentials.get('username'))
        if email:
            for user in filter_users_by_email(email, prefer_verified=True):
                if self._check_password(user, credentials['password']):
                    return user
        return None

    def _check_password(self, user, password):
        if False:
            for i in range(10):
                print('nop')
        ret = user.check_password(password)
        if ret:
            ret = self.user_can_authenticate(user)
            if not ret:
                self._stash_user(user)
        return ret

    @classmethod
    def _stash_user(cls, user):
        if False:
            i = 10
            return i + 15
        "Now, be aware, the following is quite ugly, let me explain:\n\n        Even if the user credentials match, the authentication can fail because\n        Django's default ModelBackend calls user_can_authenticate(), which\n        checks `is_active`. Now, earlier versions of allauth did not do this\n        and simply returned the user as authenticated, even in case of\n        `is_active=False`. For allauth scope, this does not pose a problem, as\n        these users are properly redirected to an account inactive page.\n\n        This does pose a problem when the allauth backend is used in a\n        different context where allauth is not responsible for the login. Then,\n        by not checking on `user_can_authenticate()` users will allow to become\n        authenticated whereas according to Django logic this should not be\n        allowed.\n\n        In order to preserve the allauth behavior while respecting Django's\n        logic, we stash a user for which the password check succeeded but\n        `user_can_authenticate()` failed. In the allauth authentication logic,\n        we can then unstash this user and proceed pointing the user to the\n        account inactive page.\n        "
        global _stash
        ret = getattr(_stash, 'user', None)
        _stash.user = user
        return ret

    @classmethod
    def unstash_authenticated_user(cls):
        if False:
            while True:
                i = 10
        return cls._stash_user(None)