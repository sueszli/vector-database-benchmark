from __future__ import annotations
from collections import defaultdict
from datetime import timedelta
from typing import TYPE_CHECKING, ClassVar, Iterable, List, Mapping, Optional, Tuple
from django.conf import settings
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from sentry.backup.dependencies import ImportKind, PrimaryKeyMap, get_model_name
from sentry.backup.helpers import ImportFlags
from sentry.backup.scopes import ImportScope, RelocationScope
from sentry.db.models import BaseManager, FlexibleForeignKey, control_silo_only_model, sane_repr
from sentry.db.models.outboxes import ControlOutboxProducingModel
from sentry.models.outbox import ControlOutboxBase, OutboxCategory
from sentry.services.hybrid_cloud.organization.model import RpcOrganization
from sentry.types.region import find_regions_for_user
from sentry.utils.security import get_secure_token
if TYPE_CHECKING:
    from sentry.models.user import User

class UserEmailManager(BaseManager['UserEmail']):

    def get_emails_by_user(self, organization: RpcOrganization) -> Mapping[User, Iterable[str]]:
        if False:
            return 10
        from sentry.models.organizationmembermapping import OrganizationMemberMapping
        emails_by_user = defaultdict(set)
        user_emails = self.filter(user_id__in=OrganizationMemberMapping.objects.filter(organization_id=organization.id).values_list('user_id', flat=True)).select_related('user')
        for entry in user_emails:
            emails_by_user[entry.user].add(entry.email)
        return emails_by_user

    def get_primary_email(self, user: User) -> UserEmail:
        if False:
            print('Hello World!')
        (user_email, _) = self.get_or_create(user_id=user.id, email=user.email)
        return user_email

@control_silo_only_model
class UserEmail(ControlOutboxProducingModel):
    __relocation_scope__ = RelocationScope.User
    __relocation_dependencies__ = {'sentry.Email'}
    user = FlexibleForeignKey(settings.AUTH_USER_MODEL, related_name='emails')
    email = models.EmailField(_('email address'), max_length=75)
    validation_hash = models.CharField(max_length=32, default=get_secure_token)
    date_hash_added = models.DateTimeField(default=timezone.now)
    is_verified = models.BooleanField(_('verified'), default=False, help_text=_('Designates whether this user has confirmed their email.'))
    objects: ClassVar[UserEmailManager] = UserEmailManager()

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_useremail'
        unique_together = (('user', 'email'),)
    __repr__ = sane_repr('user_id', 'email')

    def outboxes_for_update(self, shard_identifier: int | None=None) -> List[ControlOutboxBase]:
        if False:
            print('Hello World!')
        regions = find_regions_for_user(self.user_id)
        return [outbox for outbox in OutboxCategory.USER_UPDATE.as_control_outboxes(region_names=regions, shard_identifier=self.user_id, object_identifier=self.user_id)]

    def set_hash(self):
        if False:
            while True:
                i = 10
        self.date_hash_added = timezone.now()
        self.validation_hash = get_secure_token()

    def hash_is_valid(self):
        if False:
            print('Hello World!')
        return self.validation_hash and self.date_hash_added > timezone.now() - timedelta(hours=48)

    def is_primary(self):
        if False:
            while True:
                i = 10
        return self.user.email == self.email

    @classmethod
    def get_primary_email(cls, user: User) -> UserEmail:
        if False:
            for i in range(10):
                print('nop')
        '@deprecated'
        return cls.objects.get_primary_email(user)

    def normalize_before_relocation_import(self, pk_map: PrimaryKeyMap, scope: ImportScope, flags: ImportFlags) -> Optional[int]:
        if False:
            print('Hello World!')
        from sentry.models.user import User
        old_user_id = self.user_id
        old_pk = super().normalize_before_relocation_import(pk_map, scope, flags)
        if old_pk is None:
            return None
        if pk_map.get_kind(get_model_name(User), old_user_id) == ImportKind.Existing:
            return None
        if scope != ImportScope.Global:
            self.is_verified = False
            self.validation_hash = get_secure_token()
            self.date_hash_added = timezone.now()
        return old_pk

    def write_relocation_import(self, _s: ImportScope, _f: ImportFlags) -> Optional[Tuple[int, ImportKind]]:
        if False:
            i = 10
            return i + 15
        useremail = self.__class__.objects.get(user=self.user, email=self.email)
        for f in self._meta.fields:
            if f.name not in ['id', 'pk']:
                setattr(useremail, f.name, getattr(self, f.name))
        useremail.save()
        return (useremail.pk, ImportKind.Inserted)