from __future__ import annotations
from django.db import models
from django.utils import timezone
from sentry.backup.scopes import RelocationScope
from sentry.db.models import BoundedPositiveIntegerField, FlexibleForeignKey, Model, region_silo_only_model, sane_repr
from sentry.db.models.fields.hybrid_cloud_foreign_key import HybridCloudForeignKey
from sentry.db.models.fields.jsonfield import JSONField

@region_silo_only_model
class AuthProviderReplica(Model):
    __relocation_scope__ = RelocationScope.Excluded
    auth_provider_id = HybridCloudForeignKey('sentry.AuthProvider', on_delete='CASCADE', unique=True)
    organization = FlexibleForeignKey('sentry.Organization', on_delete=models.CASCADE, unique=True)
    provider = models.CharField(max_length=128)
    config = JSONField()
    default_role = BoundedPositiveIntegerField(default=50)
    default_global_access = models.BooleanField(default=True)
    date_added = models.DateTimeField(default=timezone.now)
    scim_enabled = models.BooleanField()
    allow_unlinked = models.BooleanField()

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_authproviderreplica'
    __repr__ = sane_repr('organization_id', 'provider')

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.provider

    def get_provider(self):
        if False:
            i = 10
            return i + 15
        from sentry.auth import manager
        return manager.get(self.provider, **self.config)

    @property
    def provider_name(self) -> str:
        if False:
            while True:
                i = 10
        return self.get_provider().name