from typing import Any, List, Tuple
from django.db import models
from django.db.models import Q, UniqueConstraint
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from sentry.backup.scopes import RelocationScope
from sentry.db.models import FlexibleForeignKey, Model, region_silo_only_model, sane_repr
from sentry.db.models.fields.hybrid_cloud_foreign_key import HybridCloudForeignKey
from sentry.db.models.fields.text import CharField
from sentry.models.search_common import SearchType

class SortOptions:
    DATE = 'date'
    NEW = 'new'
    PRIORITY = 'priority'
    FREQ = 'freq'
    USER = 'user'
    INBOX = 'inbox'

    @classmethod
    def as_choices(cls):
        if False:
            for i in range(10):
                print('nop')
        return ((cls.DATE, _('Last Seen')), (cls.NEW, _('First Seen')), (cls.PRIORITY, _('Priority')), (cls.FREQ, _('Events')), (cls.USER, _('Users')), (cls.INBOX, _('Date Added')))

class Visibility:
    ORGANIZATION = 'organization'
    OWNER = 'owner'
    OWNER_PINNED = 'owner_pinned'

    @classmethod
    def as_choices(cls) -> List[Tuple[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        return [(cls.ORGANIZATION, _('Organization')), (cls.OWNER, _('Only for me')), (cls.OWNER_PINNED, _('My Pinned Search'))]

@region_silo_only_model
class SavedSearch(Model):
    """
    A saved search query.
    """
    __relocation_scope__ = RelocationScope.Organization
    organization = FlexibleForeignKey('sentry.Organization', null=True)
    type = models.PositiveSmallIntegerField(default=SearchType.ISSUE.value, null=True)
    name = models.CharField(max_length=128)
    query = models.TextField()
    sort = CharField(max_length=16, default=SortOptions.DATE, choices=SortOptions.as_choices(), null=True)
    date_added = models.DateTimeField(default=timezone.now)
    is_global = models.BooleanField(null=True, default=False, db_index=True)
    owner_id = HybridCloudForeignKey('sentry.User', on_delete='CASCADE', null=True)
    visibility = models.CharField(max_length=16, default=Visibility.OWNER, choices=Visibility.as_choices())

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_savedsearch'
        unique_together = ()
        constraints = [UniqueConstraint(fields=['organization', 'owner_id', 'type'], condition=Q(visibility=Visibility.OWNER_PINNED), name='sentry_savedsearch_pinning_constraint'), UniqueConstraint(fields=['is_global', 'name'], condition=Q(is_global=True), name='sentry_savedsearch_organization_id_313a24e907cdef99')]

    @property
    def is_pinned(self):
        if False:
            i = 10
            return i + 15
        return self.visibility == Visibility.OWNER_PINNED
    __repr__ = sane_repr('project_id', 'name')