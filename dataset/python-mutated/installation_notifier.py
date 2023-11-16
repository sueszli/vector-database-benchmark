from django.db import router
from sentry.api.serializers import AppPlatformEvent, SentryAppInstallationSerializer, serialize
from sentry.coreapi import APIUnauthorized
from sentry.mediators.mediator import Mediator
from sentry.mediators.param import Param
from sentry.models.integrations.sentry_app_installation import SentryAppInstallation
from sentry.services.hybrid_cloud.user.model import RpcUser
from sentry.utils.cache import memoize
from sentry.utils.sentry_apps import send_and_save_webhook_request

class InstallationNotifier(Mediator):
    install = Param(SentryAppInstallation)
    user = Param(RpcUser)
    action = Param(str)
    using = router.db_for_write(SentryAppInstallation)

    def call(self):
        if False:
            for i in range(10):
                print('nop')
        self._verify_action()
        self._send_webhook()

    def _verify_action(self):
        if False:
            i = 10
            return i + 15
        if self.action not in ['created', 'deleted']:
            raise APIUnauthorized(f"Invalid action '{self.action}'")

    def _send_webhook(self):
        if False:
            while True:
                i = 10
        return send_and_save_webhook_request(self.sentry_app, self.request)

    @property
    def request(self):
        if False:
            i = 10
            return i + 15
        data = serialize([self.install], user=self.user, serializer=SentryAppInstallationSerializer())[0]
        return AppPlatformEvent(resource='installation', action=self.action, install=self.install, data={'installation': data}, actor=self.user)

    @memoize
    def sentry_app(self):
        if False:
            while True:
                i = 10
        return self.install.sentry_app

    @memoize
    def api_grant(self):
        if False:
            for i in range(10):
                print('nop')
        return self.install.api_grant_id and self.install.api_grant