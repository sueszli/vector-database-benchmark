from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.utils import timezone
from django.utils.translation import gettext as _
from rest_framework import authentication, exceptions
from common.auth import signature
from common.utils import get_object_or_none, get_request_ip_or_data, contains_ip
from ..models import AccessKey, PrivateToken

def date_more_than(d, seconds):
    if False:
        print('Hello World!')
    return d is None or (timezone.now() - d).seconds > seconds

def after_authenticate_update_date(user, token=None):
    if False:
        while True:
            i = 10
    if date_more_than(user.date_api_key_last_used, 60):
        user.date_api_key_last_used = timezone.now()
        user.save(update_fields=['date_api_key_last_used'])
    if token and hasattr(token, 'date_last_used') and date_more_than(token.date_last_used, 60):
        token.date_last_used = timezone.now()
        token.save(update_fields=['date_last_used'])

class AccessTokenAuthentication(authentication.BaseAuthentication):
    keyword = 'Bearer'
    model = get_user_model()

    def authenticate(self, request):
        if False:
            for i in range(10):
                print('nop')
        auth = authentication.get_authorization_header(request).split()
        if not auth or auth[0].lower() != self.keyword.lower().encode():
            return None
        if len(auth) == 1:
            msg = _('Invalid token header. No credentials provided.')
            raise exceptions.AuthenticationFailed(msg)
        elif len(auth) > 2:
            msg = _('Invalid token header. Sign string should not contain spaces.')
            raise exceptions.AuthenticationFailed(msg)
        try:
            token = auth[1].decode()
        except UnicodeError:
            msg = _('Invalid token header. Sign string should not contain invalid characters.')
            raise exceptions.AuthenticationFailed(msg)
        (user, header) = self.authenticate_credentials(token)
        after_authenticate_update_date(user)
        return (user, header)

    @staticmethod
    def authenticate_credentials(token):
        if False:
            while True:
                i = 10
        model = get_user_model()
        user_id = cache.get(token)
        user = get_object_or_none(model, id=user_id)
        if not user:
            msg = _('Invalid token or cache refreshed.')
            raise exceptions.AuthenticationFailed(msg)
        return (user, None)

    def authenticate_header(self, request):
        if False:
            return 10
        return self.keyword

class PrivateTokenAuthentication(authentication.TokenAuthentication):
    model = PrivateToken

    def authenticate(self, request):
        if False:
            print('Hello World!')
        user_token = super().authenticate(request)
        if not user_token:
            return
        (user, token) = user_token
        after_authenticate_update_date(user, token)
        return (user, token)

class SessionAuthentication(authentication.SessionAuthentication):

    def authenticate(self, request):
        if False:
            i = 10
            return i + 15
        '\n        Returns a `User` if the request session currently has a logged in user.\n        Otherwise, returns `None`.\n        '
        user = getattr(request._request, 'user', None)
        if not user or not user.is_active:
            return None
        try:
            self.enforce_csrf(request)
        except exceptions.AuthenticationFailed:
            return None
        return (user, None)

class SignatureAuthentication(signature.SignatureAuthentication):
    model = get_user_model()

    def fetch_user_data(self, key_id, algorithm='hmac-sha256'):
        if False:
            i = 10
            return i + 15
        try:
            key = AccessKey.objects.get(id=key_id)
            if not key.is_active:
                return (None, None)
            (user, secret) = (key.user, str(key.secret))
            after_authenticate_update_date(user, key)
            return (user, secret)
        except (AccessKey.DoesNotExist, exceptions.ValidationError):
            return (None, None)

    def is_ip_allow(self, key_id, request):
        if False:
            i = 10
            return i + 15
        try:
            ak = AccessKey.objects.get(id=key_id)
            ip_group = ak.ip_group
            ip = get_request_ip_or_data(request)
            if not contains_ip(ip, ip_group):
                return False
            return True
        except (AccessKey.DoesNotExist, exceptions.ValidationError):
            return False