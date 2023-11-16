from __future__ import annotations
import re
import secrets
from typing import Any, ClassVar, Optional, Tuple
from urllib.parse import urlparse
import petname
from django.conf import settings
from django.db import ProgrammingError, models
from django.forms import model_to_dict
from django.urls import reverse
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from bitfield import TypedClassBitField
from sentry import features, options
from sentry.backup.dependencies import ImportKind
from sentry.backup.helpers import ImportFlags
from sentry.backup.scopes import ImportScope, RelocationScope
from sentry.db.models import BaseManager, BoundedPositiveIntegerField, FlexibleForeignKey, JSONField, Model, region_silo_only_model, sane_repr
from sentry.silo.base import SiloMode
from sentry.tasks.relay import schedule_invalidate_project_config
_token_re = re.compile('^[a-f0-9]{32}$')

class ProjectKeyStatus:
    ACTIVE = 0
    INACTIVE = 1

class ProjectKeyManager(BaseManager['ProjectKey']):

    def post_save(self, instance, **kwargs):
        if False:
            while True:
                i = 10
        schedule_invalidate_project_config(public_key=instance.public_key, trigger='projectkey.post_save')

    def post_delete(self, instance, **kwargs):
        if False:
            i = 10
            return i + 15
        schedule_invalidate_project_config(public_key=instance.public_key, trigger='projectkey.post_delete')

@region_silo_only_model
class ProjectKey(Model):
    __relocation_scope__ = RelocationScope.Organization
    project = FlexibleForeignKey('sentry.Project', related_name='key_set')
    label = models.CharField(max_length=64, blank=True, null=True)
    public_key = models.CharField(max_length=32, unique=True, null=True)
    secret_key = models.CharField(max_length=32, unique=True, null=True)

    class roles(TypedClassBitField):
        store: bool
        api: bool
        bitfield_default = ['store']
    status = BoundedPositiveIntegerField(default=0, choices=((ProjectKeyStatus.ACTIVE, _('Active')), (ProjectKeyStatus.INACTIVE, _('Inactive'))), db_index=True)
    date_added = models.DateTimeField(default=timezone.now, null=True)
    rate_limit_count = BoundedPositiveIntegerField(null=True)
    rate_limit_window = BoundedPositiveIntegerField(null=True)
    objects: ClassVar[ProjectKeyManager] = ProjectKeyManager(cache_fields=('public_key', 'secret_key'), cache_ttl=60 * 30)
    data: models.Field[dict[str, Any], dict[str, Any]] = JSONField()
    scopes = ('project:read', 'project:write', 'project:admin', 'project:releases', 'event:read', 'event:write', 'event:admin')

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_projectkey'
    __repr__ = sane_repr('project_id', 'public_key')

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self.public_key)

    @classmethod
    def generate_api_key(cls):
        if False:
            while True:
                i = 10
        return secrets.token_hex(nbytes=16)

    @classmethod
    def looks_like_api_key(cls, key):
        if False:
            i = 10
            return i + 15
        return bool(_token_re.match(key))

    @classmethod
    def from_dsn(cls, dsn):
        if False:
            return 10
        urlparts = urlparse(dsn)
        public_key = urlparts.username
        project_id = urlparts.path.rsplit('/', 1)[-1]
        try:
            return ProjectKey.objects.get(public_key=public_key, project=project_id)
        except ValueError:
            raise ProjectKey.DoesNotExist('ProjectKey matching query does not exist.')

    @classmethod
    def get_default(cls, project):
        if False:
            i = 10
            return i + 15
        return cls.objects.filter(project=project, roles=models.F('roles').bitor(cls.roles.store), status=ProjectKeyStatus.ACTIVE).first()

    @property
    def is_active(self):
        if False:
            i = 10
            return i + 15
        return self.status == ProjectKeyStatus.ACTIVE

    @property
    def rate_limit(self):
        if False:
            i = 10
            return i + 15
        if self.rate_limit_count and self.rate_limit_window:
            return (self.rate_limit_count, self.rate_limit_window)
        return (0, 0)

    def save(self, *args, **kwargs):
        if False:
            print('Hello World!')
        if not self.public_key:
            self.public_key = ProjectKey.generate_api_key()
        if not self.secret_key:
            self.secret_key = ProjectKey.generate_api_key()
        if not self.label:
            self.label = petname.generate(2, ' ', letters=10).title()
        super().save(*args, **kwargs)

    def get_dsn(self, domain=None, secure=True, public=False):
        if False:
            for i in range(10):
                print('nop')
        urlparts = urlparse(self.get_endpoint(public=public))
        if not public:
            key = f'{self.public_key}:{self.secret_key}'
        else:
            assert self.public_key is not None
            key = self.public_key
        if not urlparts.netloc or not urlparts.scheme:
            return ''
        return '{}://{}@{}/{}'.format(urlparts.scheme, key, urlparts.netloc + urlparts.path, self.project_id)

    @property
    def organization_id(self):
        if False:
            print('Hello World!')
        return self.project.organization_id

    @property
    def organization(self):
        if False:
            i = 10
            return i + 15
        return self.project.organization

    @property
    def dsn_private(self):
        if False:
            print('Hello World!')
        return self.get_dsn(public=False)

    @property
    def dsn_public(self):
        if False:
            return 10
        return self.get_dsn(public=True)

    @property
    def csp_endpoint(self):
        if False:
            i = 10
            return i + 15
        endpoint = self.get_endpoint()
        return f'{endpoint}/api/{self.project_id}/csp-report/?sentry_key={self.public_key}'

    @property
    def security_endpoint(self):
        if False:
            return 10
        endpoint = self.get_endpoint()
        return f'{endpoint}/api/{self.project_id}/security/?sentry_key={self.public_key}'

    @property
    def nel_endpoint(self):
        if False:
            print('Hello World!')
        endpoint = self.get_endpoint()
        return f'{endpoint}/api/{self.project_id}/nel/?sentry_key={self.public_key}'

    @property
    def minidump_endpoint(self):
        if False:
            while True:
                i = 10
        endpoint = self.get_endpoint()
        return f'{endpoint}/api/{self.project_id}/minidump/?sentry_key={self.public_key}'

    @property
    def unreal_endpoint(self):
        if False:
            print('Hello World!')
        return f'{self.get_endpoint()}/api/{self.project_id}/unreal/{self.public_key}/'

    @property
    def js_sdk_loader_cdn_url(self) -> str:
        if False:
            print('Hello World!')
        if settings.JS_SDK_LOADER_CDN_URL:
            return f'{settings.JS_SDK_LOADER_CDN_URL}{self.public_key}.min.js'
        else:
            endpoint = self.get_endpoint()
            return '{}{}'.format(endpoint, reverse('sentry-js-sdk-loader', args=[self.public_key, '.min']))

    def get_endpoint(self, public=True):
        if False:
            for i in range(10):
                print('nop')
        from sentry.api.utils import generate_region_url
        if public:
            endpoint = settings.SENTRY_PUBLIC_ENDPOINT or settings.SENTRY_ENDPOINT
        else:
            endpoint = settings.SENTRY_ENDPOINT
        if not endpoint and SiloMode.get_current_mode() == SiloMode.REGION:
            endpoint = generate_region_url()
        if not endpoint:
            endpoint = options.get('system.url-prefix')
        has_org_subdomain = False
        try:
            has_org_subdomain = features.has('organizations:org-subdomains', self.project.organization)
        except ProgrammingError:
            pass
        if has_org_subdomain:
            urlparts = urlparse(endpoint)
            if urlparts.scheme and urlparts.netloc:
                endpoint = '{}://{}.{}{}'.format(str(urlparts.scheme), settings.SENTRY_ORG_SUBDOMAIN_TEMPLATE.format(organization_id=self.project.organization_id), str(urlparts.netloc), str(urlparts.path))
        return endpoint

    def get_allowed_origins(self):
        if False:
            i = 10
            return i + 15
        from sentry.utils.http import get_origins
        return get_origins(self.project)

    def get_audit_log_data(self):
        if False:
            for i in range(10):
                print('nop')
        return {'label': self.label, 'public_key': self.public_key, 'secret_key': self.secret_key, 'roles': int(self.roles), 'status': self.status, 'rate_limit_count': self.rate_limit_count, 'rate_limit_window': self.rate_limit_window}

    def get_scopes(self):
        if False:
            while True:
                i = 10
        return self.scopes

    def write_relocation_import(self, _s: ImportScope, _f: ImportFlags) -> Optional[Tuple[int, ImportKind]]:
        if False:
            i = 10
            return i + 15
        matching_public_key = self.__class__.objects.filter(public_key=self.public_key).first()
        if not self.public_key or matching_public_key:
            self.public_key = self.generate_api_key()
        matching_secret_key = self.__class__.objects.filter(secret_key=self.secret_key).first()
        if not self.secret_key or matching_secret_key:
            self.secret_key = self.generate_api_key()
        (key, _) = ProjectKey.objects.get_or_create(project=self.project, defaults=model_to_dict(self))
        if key:
            self.pk = key.pk
            self.save()
        return (self.pk, ImportKind.Inserted)