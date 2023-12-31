from django.db import models
from django.utils import timezone

from sentry.backup.scopes import RelocationScope
from sentry.db.models import BoundedBigIntegerField, Model


class DeletedEntry(Model):
    __relocation_scope__ = RelocationScope.Excluded

    actor_label = models.CharField(max_length=64, null=True)
    # if the entry was created via a user
    actor_id = BoundedBigIntegerField(null=True)
    # if the entry was created via an api key
    actor_key = models.CharField(max_length=32, null=True)

    ip_address = models.GenericIPAddressField(null=True, unpack_ipv4=True)
    date_deleted = models.DateTimeField(default=timezone.now)

    date_created = models.DateTimeField(null=True)

    reason = models.TextField(blank=True, null=True)

    class Meta:
        abstract = True
