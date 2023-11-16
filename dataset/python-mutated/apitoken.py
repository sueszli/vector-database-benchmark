from __future__ import annotations
import secrets
from datetime import timedelta
from typing import ClassVar, Collection, Optional, Tuple
from django.db import models, router, transaction
from django.utils import timezone
from django.utils.encoding import force_str
from sentry.backup.dependencies import ImportKind
from sentry.backup.helpers import ImportFlags
from sentry.backup.scopes import ImportScope, RelocationScope
from sentry.constants import SentryAppStatus
from sentry.db.models import FlexibleForeignKey, control_silo_only_model, sane_repr
from sentry.db.models.outboxes import ControlOutboxProducingManager, ReplicatedControlModel
from sentry.models.apiscopes import HasApiScopes
from sentry.models.outbox import OutboxCategory
from sentry.types.region import find_all_region_names
DEFAULT_EXPIRATION = timedelta(days=30)

def default_expiration():
    if False:
        for i in range(10):
            print('nop')
    return timezone.now() + DEFAULT_EXPIRATION

def generate_token():
    if False:
        for i in range(10):
            print('nop')
    return secrets.token_hex(nbytes=32)

@control_silo_only_model
class ApiToken(ReplicatedControlModel, HasApiScopes):
    __relocation_scope__ = {RelocationScope.Global, RelocationScope.Config}
    category = OutboxCategory.API_TOKEN_UPDATE
    application = FlexibleForeignKey('sentry.ApiApplication', null=True)
    user = FlexibleForeignKey('sentry.User')
    name = models.CharField(max_length=255, null=True)
    token = models.CharField(max_length=64, unique=True, default=generate_token)
    token_last_characters = models.CharField(max_length=4, null=True)
    refresh_token = models.CharField(max_length=64, unique=True, null=True, default=generate_token)
    expires_at = models.DateTimeField(null=True, default=default_expiration)
    date_added = models.DateTimeField(default=timezone.now)
    objects: ClassVar[ControlOutboxProducingManager[ApiToken]] = ControlOutboxProducingManager(cache_fields=('token',))

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_apitoken'
    __repr__ = sane_repr('user_id', 'token', 'application_id')

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return force_str(self.token)

    def outbox_region_names(self) -> Collection[str]:
        if False:
            print('Hello World!')
        return list(find_all_region_names())

    def handle_async_replication(self, region_name: str, shard_identifier: int) -> None:
        if False:
            i = 10
            return i + 15
        from sentry.services.hybrid_cloud.auth.serial import serialize_api_token
        from sentry.services.hybrid_cloud.replica import region_replica_service
        region_replica_service.upsert_replicated_api_token(api_token=serialize_api_token(self), region_name=region_name)

    @classmethod
    def from_grant(cls, grant):
        if False:
            i = 10
            return i + 15
        with transaction.atomic(router.db_for_write(cls)):
            return cls.objects.create(application=grant.application, user=grant.user, scope_list=grant.get_scopes())

    def is_expired(self):
        if False:
            print('Hello World!')
        if not self.expires_at:
            return False
        return timezone.now() >= self.expires_at

    def get_audit_log_data(self):
        if False:
            for i in range(10):
                print('nop')
        return {'scopes': self.get_scopes()}

    def get_allowed_origins(self):
        if False:
            while True:
                i = 10
        if self.application:
            return self.application.get_allowed_origins()
        return ()

    def refresh(self, expires_at=None):
        if False:
            i = 10
            return i + 15
        if expires_at is None:
            expires_at = timezone.now() + DEFAULT_EXPIRATION
        self.update(token=generate_token(), refresh_token=generate_token(), expires_at=expires_at)

    def get_relocation_scope(self) -> RelocationScope:
        if False:
            while True:
                i = 10
        if self.application_id is not None:
            return RelocationScope.Global
        return RelocationScope.Config

    def write_relocation_import(self, scope: ImportScope, flags: ImportFlags) -> Optional[Tuple[int, ImportKind]]:
        if False:
            for i in range(10):
                print('nop')
        query = models.Q(token=self.token) | models.Q(refresh_token=self.refresh_token)
        existing = self.__class__.objects.filter(query).first()
        if existing:
            self.expires_at = timezone.now() + DEFAULT_EXPIRATION
            self.token = generate_token()
            self.refresh_token = generate_token()
        return super().write_relocation_import(scope, flags)

    @property
    def organization_id(self) -> int | None:
        if False:
            print('Hello World!')
        from sentry.models.integrations.sentry_app_installation import SentryAppInstallation
        from sentry.models.integrations.sentry_app_installation_token import SentryAppInstallationToken
        try:
            installation = SentryAppInstallation.objects.get_by_api_token(self.id).get()
        except SentryAppInstallation.DoesNotExist:
            installation = None
        if not installation or installation.sentry_app.status == SentryAppStatus.INTERNAL:
            try:
                install_token = SentryAppInstallationToken.objects.select_related('sentry_app_installation').get(api_token_id=self.id)
            except SentryAppInstallationToken.DoesNotExist:
                return None
            return install_token.sentry_app_installation.organization_id
        return installation.organization_id

def is_api_token_auth(auth: object) -> bool:
    if False:
        print('Hello World!')
    ':returns True when an API token is hitting the API.'
    from sentry.hybridcloud.models.apitokenreplica import ApiTokenReplica
    from sentry.services.hybrid_cloud.auth import AuthenticatedToken
    if isinstance(auth, AuthenticatedToken):
        return auth.kind == 'api_token'
    return isinstance(auth, ApiToken) or isinstance(auth, ApiTokenReplica)