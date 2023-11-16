import logging
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse
from django import forms
from django.conf import settings
from django.utils.translation import gettext as _
from netbox.registry import registry
from .choices import DataSourceTypeChoices
from .exceptions import SyncError
__all__ = ('LocalBackend', 'GitBackend', 'S3Backend')
logger = logging.getLogger('netbox.data_backends')

def register_backend(name):
    if False:
        return 10
    '\n    Decorator for registering a DataBackend class.\n    '

    def _wrapper(cls):
        if False:
            for i in range(10):
                print('nop')
        registry['data_backends'][name] = cls
        return cls
    return _wrapper

class DataBackend:
    parameters = {}
    sensitive_parameters = []
    do_not_call_in_templates = True

    def __init__(self, url, **kwargs):
        if False:
            while True:
                i = 10
        self.url = url
        self.params = kwargs
        self.config = self.init_config()

    def init_config(self):
        if False:
            return 10
        "\n        Hook to initialize the instance's configuration.\n        "
        return

    @property
    def url_scheme(self):
        if False:
            while True:
                i = 10
        return urlparse(self.url).scheme.lower()

    @contextmanager
    def fetch(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplemented()

@register_backend(DataSourceTypeChoices.LOCAL)
class LocalBackend(DataBackend):

    @contextmanager
    def fetch(self):
        if False:
            for i in range(10):
                print('nop')
        logger.debug(f'Data source type is local; skipping fetch')
        local_path = urlparse(self.url).path
        yield local_path

@register_backend(DataSourceTypeChoices.GIT)
class GitBackend(DataBackend):
    parameters = {'username': forms.CharField(required=False, label=_('Username'), widget=forms.TextInput(attrs={'class': 'form-control'}), help_text=_('Only used for cloning with HTTP(S)')), 'password': forms.CharField(required=False, label=_('Password'), widget=forms.TextInput(attrs={'class': 'form-control'}), help_text=_('Only used for cloning with HTTP(S)')), 'branch': forms.CharField(required=False, label=_('Branch'), widget=forms.TextInput(attrs={'class': 'form-control'}))}
    sensitive_parameters = ['password']

    def init_config(self):
        if False:
            i = 10
            return i + 15
        from dulwich.config import ConfigDict
        config = ConfigDict()
        if settings.HTTP_PROXIES and self.url_scheme in ('http', 'https'):
            if (proxy := settings.HTTP_PROXIES.get(self.url_scheme)):
                config.set('http', 'proxy', proxy)
        return config

    @contextmanager
    def fetch(self):
        if False:
            return 10
        from dulwich import porcelain
        local_path = tempfile.TemporaryDirectory()
        clone_args = {'branch': self.params.get('branch'), 'config': self.config, 'depth': 1, 'errstream': porcelain.NoneStream(), 'quiet': True}
        if self.url_scheme in ('http', 'https'):
            if self.params.get('username'):
                clone_args.update({'username': self.params.get('username'), 'password': self.params.get('password')})
        logger.debug(f'Cloning git repo: {self.url}')
        try:
            porcelain.clone(self.url, local_path.name, **clone_args)
        except BaseException as e:
            raise SyncError(f'Fetching remote data failed ({type(e).__name__}): {e}')
        yield local_path.name
        local_path.cleanup()

@register_backend(DataSourceTypeChoices.AMAZON_S3)
class S3Backend(DataBackend):
    parameters = {'aws_access_key_id': forms.CharField(label=_('AWS access key ID'), widget=forms.TextInput(attrs={'class': 'form-control'})), 'aws_secret_access_key': forms.CharField(label=_('AWS secret access key'), widget=forms.TextInput(attrs={'class': 'form-control'}))}
    sensitive_parameters = ['aws_secret_access_key']
    REGION_REGEX = 's3\\.([a-z0-9-]+)\\.amazonaws\\.com'

    def init_config(self):
        if False:
            print('Hello World!')
        from botocore.config import Config as Boto3Config
        return Boto3Config(proxies=settings.HTTP_PROXIES)

    @contextmanager
    def fetch(self):
        if False:
            print('Hello World!')
        import boto3
        local_path = tempfile.TemporaryDirectory()
        aws_access_key_id = self.params.get('aws_access_key_id')
        aws_secret_access_key = self.params.get('aws_secret_access_key')
        s3 = boto3.resource('s3', region_name=self._region_name, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, config=self.config)
        bucket = s3.Bucket(self._bucket_name)
        for obj in bucket.objects.filter(Prefix=self._remote_path):
            local_filename = os.path.join(local_path.name, obj.key)
            Path(os.path.dirname(local_filename)).mkdir(parents=True, exist_ok=True)
            bucket.download_file(obj.key, local_filename)
        yield local_path.name
        local_path.cleanup()

    @property
    def _region_name(self):
        if False:
            return 10
        domain = urlparse(self.url).netloc
        if (m := re.match(self.REGION_REGEX, domain)):
            return m.group(1)
        return None

    @property
    def _bucket_name(self):
        if False:
            i = 10
            return i + 15
        url_path = urlparse(self.url).path.lstrip('/')
        return url_path.split('/')[0]

    @property
    def _remote_path(self):
        if False:
            for i in range(10):
                print('nop')
        url_path = urlparse(self.url).path.lstrip('/')
        if '/' in url_path:
            return url_path.split('/', 1)[1]
        return ''