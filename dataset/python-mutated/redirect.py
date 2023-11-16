from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from common.exceptions import JMSException
from . import const

class NeedMoreInfoError(Exception):
    error = ''
    msg = ''

    def __init__(self, error='', msg=''):
        if False:
            for i in range(10):
                print('nop')
        if error:
            self.error = error
        if msg:
            self.msg = msg

    def as_data(self):
        if False:
            return 10
        return {'error': self.error, 'msg': self.msg}

class NeedRedirectError(JMSException):

    def __init__(self, url, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.url = url

class MFARequiredError(NeedMoreInfoError):
    msg = const.mfa_required_msg
    error = 'mfa_required'

    def __init__(self, error='', msg='', mfa_types=()):
        if False:
            while True:
                i = 10
        super().__init__(error=error, msg=msg)
        self.choices = mfa_types

    def as_data(self):
        if False:
            return 10
        return {'error': self.error, 'msg': self.msg, 'data': {'choices': self.choices, 'url': reverse('api-auth:mfa-challenge')}}

class LoginConfirmBaseError(NeedMoreInfoError):

    def __init__(self, ticket_id, **kwargs):
        if False:
            while True:
                i = 10
        self.ticket_id = ticket_id
        super().__init__(**kwargs)

    def as_data(self):
        if False:
            print('Hello World!')
        return {'error': self.error, 'msg': self.msg, 'data': {'ticket_id': self.ticket_id}}

class LoginConfirmWaitError(LoginConfirmBaseError):
    msg = const.login_confirm_wait_msg
    error = 'login_confirm_wait'

class LoginConfirmOtherError(LoginConfirmBaseError):
    error = 'login_confirm_error'

    def __init__(self, ticket_id, status, username):
        if False:
            while True:
                i = 10
        self.username = username
        msg = const.login_confirm_error_msg.format(status)
        super().__init__(ticket_id=ticket_id, msg=msg)

    def as_data(self):
        if False:
            print('Hello World!')
        ret = super().as_data()
        ret['data']['username'] = self.username
        return ret

class PasswordTooSimple(NeedRedirectError):
    default_code = 'passwd_too_simple'
    default_detail = _('Your password is too simple, please change it for security')

    def __init__(self, url, *args, **kwargs):
        if False:
            return 10
        super().__init__(url, *args, **kwargs)

class PasswordNeedUpdate(NeedRedirectError):
    default_code = 'passwd_need_update'
    default_detail = _('You should to change your password before login')

    def __init__(self, url, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(url, *args, **kwargs)

class PasswordRequireResetError(NeedRedirectError):
    default_code = 'passwd_has_expired'
    default_detail = _('Your password has expired, please reset before logging in')

    def __init__(self, url, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(url, *args, **kwargs)

class MFAUnsetError(NeedRedirectError):
    error = const.reason_mfa_unset
    msg = const.mfa_unset_msg

    def __init__(self, url, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(url, *args, **kwargs)