from datetime import datetime, timezone
from django.db import router
from sentry import analytics
from sentry.coreapi import APIUnauthorized
from sentry.mediators.mediator import Mediator
from sentry.mediators.param import Param
from sentry.mediators.token_exchange.util import token_expiration
from sentry.mediators.token_exchange.validator import Validator
from sentry.models.apiapplication import ApiApplication
from sentry.models.apigrant import ApiGrant
from sentry.models.apitoken import ApiToken
from sentry.models.integrations.sentry_app import SentryApp
from sentry.models.integrations.sentry_app_installation import SentryAppInstallation
from sentry.models.user import User
from sentry.services.hybrid_cloud.app import RpcSentryAppInstallation
from sentry.silo import unguarded_write
from sentry.utils.cache import memoize

class GrantExchanger(Mediator):
    """
    Exchanges a Grant Code for an Access Token
    """
    install = Param(RpcSentryAppInstallation)
    code = Param(str)
    client_id = Param(str)
    user = Param(User)
    using = router.db_for_write(User)

    def call(self):
        if False:
            for i in range(10):
                print('nop')
        self._validate()
        self._create_token()
        self._delete_grant()
        return self.token

    def record_analytics(self):
        if False:
            while True:
                i = 10
        analytics.record('sentry_app.token_exchanged', sentry_app_installation_id=self.install.id, exchange_type='authorization')

    def _validate(self):
        if False:
            print('Hello World!')
        Validator.run(install=self.install, client_id=self.client_id, user=self.user)
        if not self._grant_belongs_to_install() or not self._sentry_app_user_owns_grant():
            raise APIUnauthorized
        if not self._grant_is_active():
            raise APIUnauthorized('Grant has already expired.')

    def _grant_belongs_to_install(self):
        if False:
            while True:
                i = 10
        return self.grant.sentry_app_installation.id == self.install.id

    def _sentry_app_user_owns_grant(self):
        if False:
            while True:
                i = 10
        return self.grant.application.owner == self.user

    def _grant_is_active(self):
        if False:
            return 10
        return self.grant.expires_at > datetime.now(timezone.utc)

    def _delete_grant(self):
        if False:
            i = 10
            return i + 15
        with unguarded_write(router.db_for_write(ApiGrant)):
            self.grant.delete()

    def _create_token(self):
        if False:
            while True:
                i = 10
        self.token = ApiToken.objects.create(user=self.user, application=self.application, scope_list=self.sentry_app.scope_list, expires_at=token_expiration())
        try:
            SentryAppInstallation.objects.get(id=self.install.id).update(api_token=self.token)
        except SentryAppInstallation.DoesNotExist:
            pass

    @memoize
    def grant(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return ApiGrant.objects.select_related('sentry_app_installation').select_related('application').select_related('application__sentry_app').get(code=self.code)
        except ApiGrant.DoesNotExist:
            raise APIUnauthorized

    @property
    def application(self):
        if False:
            return 10
        try:
            return self.grant.application
        except ApiApplication.DoesNotExist:
            raise APIUnauthorized

    @property
    def sentry_app(self):
        if False:
            print('Hello World!')
        try:
            return self.application.sentry_app
        except SentryApp.DoesNotExist:
            raise APIUnauthorized