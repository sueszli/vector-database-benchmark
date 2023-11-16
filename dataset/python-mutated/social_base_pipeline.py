from social_core.exceptions import AuthException
from django.utils.translation import gettext_lazy as _

class AuthNotFound(AuthException):

    def __init__(self, backend, email_or_uid, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.email_or_uid = email_or_uid
        super(AuthNotFound, self).__init__(backend, *args, **kwargs)

    def __str__(self):
        if False:
            while True:
                i = 10
        return _('An account cannot be found for {0}').format(self.email_or_uid)

class AuthInactive(AuthException):

    def __str__(self):
        if False:
            return 10
        return _('Your account is inactive')

def check_user_found_or_created(backend, details, user=None, *args, **kwargs):
    if False:
        while True:
            i = 10
    if not user:
        email_or_uid = details.get('email') or kwargs.get('email') or kwargs.get('uid') or '???'
        raise AuthNotFound(backend, email_or_uid)

def set_is_active_for_new_user(strategy, details, user=None, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    if kwargs.get('is_new', False):
        details['is_active'] = True
        return {'details': details}

def prevent_inactive_login(backend, details, user=None, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    if user and (not user.is_active):
        raise AuthInactive(backend)