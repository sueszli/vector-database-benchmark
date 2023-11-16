from functools import reduce
from operator import or_
from django.db import models
from django.utils import timezone
from sentry.backup.scopes import RelocationScope
from sentry.constants import MAX_EMAIL_FIELD_LENGTH
from sentry.db.models import BoundedBigIntegerField, Model, region_silo_only_model, sane_repr
from sentry.utils.datastructures import BidirectionalMapping
from sentry.utils.hashlib import md5_text
KEYWORD_MAP = BidirectionalMapping({'ident': 'id', 'username': 'username', 'email': 'email', 'ip_address': 'ip'})

@region_silo_only_model
class EventUser(Model):
    __relocation_scope__ = RelocationScope.Excluded
    project_id = BoundedBigIntegerField(db_index=True)
    hash = models.CharField(max_length=32)
    ident = models.CharField(max_length=128, null=True)
    email = models.EmailField(null=True, max_length=MAX_EMAIL_FIELD_LENGTH)
    username = models.CharField(max_length=128, null=True)
    name = models.CharField(max_length=128, null=True)
    ip_address = models.GenericIPAddressField(null=True)
    date_added = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_eventuser'
        unique_together = (('project_id', 'ident'), ('project_id', 'hash'))
        index_together = (('project_id', 'email'), ('project_id', 'ip_address'))
    __repr__ = sane_repr('project_id', 'ident', 'email', 'username', 'ip_address')

    @classmethod
    def attr_from_keyword(cls, keyword):
        if False:
            return 10
        return KEYWORD_MAP.get_key(keyword)

    @classmethod
    def hash_from_tag(cls, value):
        if False:
            while True:
                i = 10
        return md5_text(value.split(':', 1)[-1]).hexdigest()

    @classmethod
    def for_tags(cls, project_id, values):
        if False:
            print('Hello World!')
        '\n        Finds matching EventUser objects from a list of tag values.\n\n        Return a dictionary of {tag_value: event_user}.\n        '
        hashes = [cls.hash_from_tag(v) for v in values]
        return {e.tag_value: e for e in cls.objects.filter(project_id=project_id, hash__in=hashes)}

    def save(self, *args, **kwargs):
        if False:
            print('Hello World!')
        assert self.ident or self.username or self.email or self.ip_address, 'No identifying value found for user'
        if not self.hash:
            self.set_hash()
        super().save(*args, **kwargs)

    def set_hash(self):
        if False:
            for i in range(10):
                print('nop')
        self.hash = self.build_hash()

    def build_hash(self):
        if False:
            return 10
        for (key, value) in self.iter_attributes():
            if value:
                return md5_text(value).hexdigest()

    @property
    def tag_value(self):
        if False:
            print('Hello World!')
        '\n        Return the identifier used with tags to link this user.\n        '
        for (key, value) in self.iter_attributes():
            if value:
                return f'{KEYWORD_MAP[key]}:{value}'

    def iter_attributes(self):
        if False:
            i = 10
            return i + 15
        '\n        Iterate over key/value pairs for this EventUser in priority order.\n        '
        for key in KEYWORD_MAP.keys():
            yield (key, getattr(self, key))

    def get_label(self):
        if False:
            return 10
        return self.email or self.username or self.ident or self.ip_address

    def get_display_name(self):
        if False:
            i = 10
            return i + 15
        return self.name or self.email or self.username

    def find_similar_users(self, user):
        if False:
            for i in range(10):
                print('nop')
        from sentry.models.organizationmemberteam import OrganizationMemberTeam
        from sentry.models.project import Project
        project_ids = list(Project.objects.filter(teams__in=OrganizationMemberTeam.objects.filter(organizationmember__user=user, organizationmember__organization__project=self.project_id, is_active=True).values('team')).values_list('id', flat=True)[:1000])
        if not project_ids:
            return type(self).objects.none()
        filters = []
        if self.email:
            filters.append(models.Q(email=self.email))
        if self.ip_address:
            filters.append(models.Q(ip_address=self.ip_address))
        if not filters:
            return type(self).objects.none()
        return type(self).objects.exclude(id=self.id).filter(reduce(or_, filters), project_id__in=project_ids)