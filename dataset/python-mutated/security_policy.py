import datetime
from pyramid.authentication import SessionAuthenticationHelper, extract_http_basic_credentials
from pyramid.authorization import ACLHelper
from pyramid.httpexceptions import HTTPUnauthorized
from pyramid.interfaces import ISecurityPolicy
from pyramid.security import Allowed
from zope.interface import implementer
from warehouse.accounts.interfaces import IPasswordBreachedService, IUserService
from warehouse.accounts.models import DisableReason, User
from warehouse.cache.http import add_vary_callback
from warehouse.email import send_password_compromised_email_hibp
from warehouse.errors import BasicAuthAccountFrozen, BasicAuthBreachedPassword, BasicAuthFailedPassword, WarehouseDenied
from warehouse.events.tags import EventTag
from warehouse.packaging.models import TwoFactorRequireable
from warehouse.utils.security_policy import AuthenticationMethod, principals_for

def _format_exc_status(exc, message):
    if False:
        i = 10
        return i + 15
    exc.status = f'{exc.status_code} {message}'
    return exc

def _basic_auth_check(username, password, request):
    if False:
        for i in range(10):
            print('nop')
    if not request.matched_route:
        return False
    if request.matched_route.name != 'forklift.legacy.file_upload':
        return False
    login_service = request.find_service(IUserService, context=None)
    breach_service = request.find_service(IPasswordBreachedService, context=None)
    userid = login_service.find_userid(username)
    request._unauthenticated_userid = userid
    if userid is not None:
        user = login_service.get_user(userid)
        if login_service.check_password(user.id, password, tags=['mechanism:basic_auth', 'method:auth', 'auth_method:basic']):
            (is_disabled, disabled_for) = login_service.is_disabled(user.id)
            if is_disabled:
                if disabled_for == DisableReason.CompromisedPassword:
                    raise _format_exc_status(BasicAuthBreachedPassword(), breach_service.failure_message_plain)
                elif disabled_for == DisableReason.AccountFrozen:
                    raise _format_exc_status(BasicAuthAccountFrozen(), 'Account is frozen.')
                else:
                    raise _format_exc_status(HTTPUnauthorized(), 'Account is disabled.')
            if breach_service.check_password(password, tags=['method:auth', 'auth_method:basic']):
                send_password_compromised_email_hibp(request, user)
                login_service.disable_password(user.id, request, reason=DisableReason.CompromisedPassword)
                raise _format_exc_status(BasicAuthBreachedPassword(), breach_service.failure_message_plain)
            login_service.update_user(user.id, last_login=datetime.datetime.utcnow())
            user.record_event(tag=EventTag.Account.LoginSuccess, request=request, additional={'auth_method': 'basic'})
            return True
        else:
            user.record_event(tag=EventTag.Account.LoginFailure, request=request, additional={'reason': 'invalid_password', 'auth_method': 'basic'})
            raise _format_exc_status(BasicAuthFailedPassword(), 'Invalid or non-existent authentication information. See {projecthelp} for more information.'.format(projecthelp=request.help_url(_anchor='invalid-auth')))
    return False

@implementer(ISecurityPolicy)
class SessionSecurityPolicy:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._session_helper = SessionAuthenticationHelper()
        self._acl = ACLHelper()

    def identity(self, request):
        if False:
            i = 10
            return i + 15
        request.add_response_callback(add_vary_callback('Cookie'))
        request.authentication_method = AuthenticationMethod.SESSION
        if request.banned.by_ip(request.remote_addr):
            return None
        if not request.matched_route:
            return None
        if request.matched_route.name == 'forklift.legacy.file_upload':
            return None
        userid = self._session_helper.authenticated_userid(request)
        request._unauthenticated_userid = userid
        if userid is None:
            return None
        login_service = request.find_service(IUserService, context=None)
        user = login_service.get_user(userid)
        if user is None:
            return None
        (is_disabled, _) = login_service.is_disabled(userid)
        if is_disabled:
            request.session.invalidate()
            request.session.flash('Session invalidated', queue='error')
            return None
        if request.session.password_outdated(login_service.get_password_timestamp(userid)):
            request.session.invalidate()
            request.session.flash('Session invalidated by password change', queue='error')
            return None
        return user

    def forget(self, request, **kw):
        if False:
            while True:
                i = 10
        return self._session_helper.forget(request, **kw)

    def remember(self, request, userid, **kw):
        if False:
            print('Hello World!')
        return self._session_helper.remember(request, userid, **kw)

    def authenticated_userid(self, request):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def permits(self, request, context, permission):
        if False:
            print('Hello World!')
        return _permits_for_user_policy(self._acl, request, context, permission)

@implementer(ISecurityPolicy)
class BasicAuthSecurityPolicy:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._acl = ACLHelper()

    def identity(self, request):
        if False:
            i = 10
            return i + 15
        request.add_response_callback(add_vary_callback('Authorization'))
        request.authentication_method = AuthenticationMethod.BASIC_AUTH
        if request.banned.by_ip(request.remote_addr):
            return None
        credentials = extract_http_basic_credentials(request)
        if credentials is None:
            return None
        (username, password) = credentials
        if not _basic_auth_check(username, password, request):
            return None
        login_service = request.find_service(IUserService, context=None)
        return login_service.get_user_by_username(username)

    def forget(self, request, **kw):
        if False:
            print('Hello World!')
        return []

    def remember(self, request, userid, **kw):
        if False:
            print('Hello World!')
        return [('WWW-Authenticate', 'Basic realm="Realm"')]

    def authenticated_userid(self, request):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def permits(self, request, context, permission):
        if False:
            i = 10
            return i + 15
        return _permits_for_user_policy(self._acl, request, context, permission)

def _permits_for_user_policy(acl, request, context, permission):
    if False:
        i = 10
        return i + 15
    assert isinstance(request.identity, User)
    res = acl.permits(context, principals_for(request.identity), permission)
    if isinstance(res, Allowed) and (not request.identity.has_primary_verified_email) and request.matched_route.name.startswith('manage') and (request.matched_route.name != 'manage.account'):
        return WarehouseDenied('unverified', reason='unverified_email')
    if isinstance(res, Allowed):
        mfa = _check_for_mfa(request, context)
        if mfa is not None:
            return mfa
    return res

def _check_for_mfa(request, context) -> WarehouseDenied | None:
    if False:
        print('Hello World!')
    assert isinstance(request.identity, User)
    if isinstance(context, TwoFactorRequireable):
        if request.registry.settings['warehouse.two_factor_requirement.enabled'] and context.owners_require_2fa and (not request.identity.has_two_factor):
            return WarehouseDenied('This project requires two factor authentication to be enabled for all contributors.', reason='owners_require_2fa')
        if request.registry.settings['warehouse.two_factor_mandate.enabled'] and context.pypi_mandates_2fa and (not request.identity.has_two_factor):
            return WarehouseDenied('PyPI requires two factor authentication to be enabled for all contributors to this project.', reason='pypi_mandates_2fa')
        if request.registry.settings['warehouse.two_factor_mandate.available'] and context.pypi_mandates_2fa and (not request.identity.has_two_factor):
            request.session.flash("This project is included in PyPI's two-factor mandate for critical projects. In the future, you will be unable to perform this action without enabling 2FA for your account", queue='warning')
    _exempt_routes = ['manage.account.recovery-codes', 'manage.account.totp-provision', 'manage.account.two-factor', 'manage.account.webauthn-provision']
    if request.identity.date_joined and request.identity.date_joined > datetime.datetime(2023, 8, 8):
        if request.matched_route.name.startswith('manage') and request.matched_route.name != 'manage.account' and (not any((request.matched_route.name.startswith(route) for route in _exempt_routes))) and (not request.identity.has_two_factor):
            return WarehouseDenied('You must enable two factor authentication to manage other settings', reason='manage_2fa_required')
        if request.matched_route.name == 'forklift.legacy.file_upload' and (not request.identity.has_two_factor):
            return WarehouseDenied('You must enable two factor authentication to upload', reason='upload_2fa_required')
    return None