from __future__ import annotations
import errno
import logging
import os
import zipfile
from contextlib import contextmanager
from hashlib import sha1
from io import BytesIO
from tempfile import TemporaryDirectory
from typing import IO, ClassVar, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit
import sentry_sdk
from django.core.files.base import File as FileObj
from django.db import models, router
from typing_extensions import Self
from sentry import options
from sentry.backup.scopes import RelocationScope
from sentry.db.models import BaseManager, BoundedBigIntegerField, BoundedPositiveIntegerField, FlexibleForeignKey, Model, region_silo_only_model, sane_repr
from sentry.models.distribution import Distribution
from sentry.models.files.file import File
from sentry.models.files.utils import clear_cached_files
from sentry.models.release import Release
from sentry.utils import json, metrics
from sentry.utils.db import atomic_transaction
from sentry.utils.hashlib import sha1_text
from sentry.utils.zip import safe_extract_zip
logger = logging.getLogger(__name__)
ARTIFACT_INDEX_FILENAME = 'artifact-index.json'
ARTIFACT_INDEX_TYPE = 'release.artifact-index'

class PublicReleaseFileManager(models.Manager):
    """Manager for all release files that are not internal.

    Internal release files include:
    * Uploaded release archives
    * Artifact index mapping URLs to release archives

    This manager has the overhead of always joining the File table in order
    to filter release files.

    """

    def get_queryset(self):
        if False:
            print('Hello World!')
        return super().get_queryset().select_related('file').filter(file__type='release.file')

@region_silo_only_model
class ReleaseFile(Model):
    """
    A ReleaseFile is an association between a Release and a File.

    The ident of the file should be sha1(name) or
    sha1(name '\\x00\\x00' dist.name) and must be unique per release.
    """
    __relocation_scope__ = RelocationScope.Excluded
    organization_id = BoundedBigIntegerField()
    project_id = BoundedBigIntegerField(null=True)
    release_id = BoundedBigIntegerField()
    file = FlexibleForeignKey('sentry.File')
    ident = models.CharField(max_length=40)
    name = models.TextField()
    dist_id = BoundedBigIntegerField(null=True)
    artifact_count = BoundedPositiveIntegerField(null=True, default=1)
    __repr__ = sane_repr('release', 'ident')
    objects: ClassVar[BaseManager[Self]] = BaseManager()
    public_objects: ClassVar[PublicReleaseFileManager] = PublicReleaseFileManager()
    cache: ClassVar[ReleaseFileCache]

    class Meta:
        unique_together = (('release_id', 'ident'),)
        index_together = (('release_id', 'name'),)
        app_label = 'sentry'
        db_table = 'sentry_releasefile'

    def save(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        from sentry.models.distribution import Distribution
        if not self.ident and self.name:
            dist = None
            if self.dist_id:
                dist = Distribution.objects.get(pk=self.dist_id).name
            self.ident = type(self).get_ident(self.name, dist)
        return super().save(*args, **kwargs)

    def update(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if 'name' in kwargs and 'ident' not in kwargs:
            dist_name = None
            dist_id = kwargs.get('dist_id') or self.dist_id
            if dist_id:
                dist_name = Distribution.objects.filter(pk=dist_id).values_list('name', flat=True)[0]
            kwargs['ident'] = self.ident = type(self).get_ident(kwargs['name'], dist_name)
        return super().update(*args, **kwargs)

    @classmethod
    def get_ident(cls, name, dist=None):
        if False:
            print('Hello World!')
        if dist is not None:
            return sha1_text(name + '\x00\x00' + dist).hexdigest()
        return sha1_text(name).hexdigest()

    @classmethod
    def normalize(cls, url):
        if False:
            i = 10
            return i + 15
        'Transforms a full absolute url into 2 or 4 generalized options\n\n        * the original url as input\n        * (optional) original url without querystring\n        * the full url, but stripped of scheme and netloc\n        * (optional) full url without scheme and netloc or querystring\n        '
        (scheme, netloc, path, query, _) = urlsplit(url)
        uri_without_fragment = (scheme, netloc, path, query, '')
        uri_relative = ('', '', path, query, '')
        uri_without_query = (scheme, netloc, path, '', '')
        uri_relative_without_query = ('', '', path, '', '')
        urls = [urlunsplit(uri_without_fragment)]
        if query:
            urls.append(urlunsplit(uri_without_query))
        urls.append('~' + urlunsplit(uri_relative))
        if query:
            urls.append('~' + urlunsplit(uri_relative_without_query))
        return urls

class ReleaseFileCache:

    @property
    def cache_path(self):
        if False:
            i = 10
            return i + 15
        return options.get('releasefile.cache-path')

    def getfile(self, releasefile):
        if False:
            for i in range(10):
                print('nop')
        cutoff = options.get('releasefile.cache-limit')
        file_size = releasefile.file.size
        if file_size < cutoff:
            metrics.timing('release_file.cache.get.size', file_size, tags={'cutoff': True})
            return releasefile.file.getfile()
        file_id = str(releasefile.file.id)
        organization_id = str(releasefile.organization_id)
        file_path = os.path.join(self.cache_path, organization_id, file_id)
        hit = True
        try:
            os.stat(file_path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
            releasefile.file.save_to(file_path)
            hit = False
        metrics.timing('release_file.cache.get.size', file_size, tags={'hit': hit, 'cutoff': False})
        return FileObj(open(file_path, 'rb'))

    def clear_old_entries(self):
        if False:
            for i in range(10):
                print('nop')
        clear_cached_files(self.cache_path)
ReleaseFile.cache = ReleaseFileCache()

class ReleaseArchive:
    """Read-only view of uploaded ZIP-archive of release files"""

    def __init__(self, fileobj: IO):
        if False:
            while True:
                i = 10
        self._fileobj = fileobj
        self._zip_file = zipfile.ZipFile(self._fileobj)
        self.manifest = self._read_manifest()
        self.artifact_count = len(self.manifest.get('files', {}))
        files = self.manifest.get('files', {})
        self._entries_by_url = {entry['url']: (path, entry) for (path, entry) in files.items()}

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, exc, value, tb):
        if False:
            for i in range(10):
                print('nop')
        self.close()

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self._zip_file.close()
        self._fileobj.close()

    def info(self, filename: str) -> zipfile.ZipInfo:
        if False:
            return 10
        return self._zip_file.getinfo(filename)

    def read(self, filename: str) -> bytes:
        if False:
            return 10
        return self._zip_file.read(filename)

    def _read_manifest(self) -> dict:
        if False:
            while True:
                i = 10
        manifest_bytes = self.read('manifest.json')
        return json.loads(manifest_bytes.decode('utf-8'))

    def get_file_by_url(self, url: str) -> Tuple[IO[bytes], dict]:
        if False:
            for i in range(10):
                print('nop')
        'Return file-like object and headers.\n\n        The caller is responsible for closing the returned stream.\n\n        May raise ``KeyError``\n        '
        (filename, entry) = self._entries_by_url[url]
        return (self._zip_file.open(filename), entry.get('headers', {}))

    def extract(self) -> TemporaryDirectory:
        if False:
            return 10
        'Extract contents to a temporary directory.\n\n        The caller is responsible for cleanup of the temporary files.\n        '
        temp_dir = TemporaryDirectory()
        safe_extract_zip(self._fileobj, temp_dir.name, strip_toplevel=False)
        return temp_dir

class _ArtifactIndexData:
    """Holds data of artifact index and keeps track of changes"""

    def __init__(self, data: dict, fresh=False):
        if False:
            i = 10
            return i + 15
        self._data = data
        self.changed = fresh

    @property
    def data(self):
        if False:
            while True:
                i = 10
        'Meant to be read-only'
        return self._data

    @property
    def num_files(self):
        if False:
            while True:
                i = 10
        return len(self._data.get('files', {}))

    def get(self, filename: str):
        if False:
            while True:
                i = 10
        return self._data.get('files', {}).get(filename, None)

    def update_files(self, files: dict):
        if False:
            while True:
                i = 10
        if files:
            self._data.setdefault('files', {}).update(files)
            self.changed = True

    def delete(self, filename: str) -> bool:
        if False:
            return 10
        result = self._data.get('files', {}).pop(filename, None)
        deleted = result is not None
        if deleted:
            self.changed = True
        return deleted

class _ArtifactIndexGuard:
    """Ensures atomic write operations to the artifact index"""

    def __init__(self, release: Release, dist: Optional[Distribution], **filter_args):
        if False:
            for i in range(10):
                print('nop')
        self._release = release
        self._dist = dist
        self._ident = ReleaseFile.get_ident(ARTIFACT_INDEX_FILENAME, dist and dist.name)
        self._filter_args = filter_args

    def readable_data(self, use_cache: bool) -> Optional[dict]:
        if False:
            while True:
                i = 10
        'Simple read, no synchronization necessary'
        try:
            releasefile = self._releasefile_qs()[0]
        except IndexError:
            return None
        else:
            if use_cache:
                fp = ReleaseFile.cache.getfile(releasefile)
            else:
                fp = releasefile.file.getfile()
            with fp:
                return json.load(fp)

    @contextmanager
    def writable_data(self, create: bool, initial_artifact_count=None):
        if False:
            while True:
                i = 10
        'Context manager for editable artifact index'
        with atomic_transaction(using=(router.db_for_write(ReleaseFile), router.db_for_write(File))):
            created = False
            if create:
                (releasefile, created) = self._get_or_create_releasefile(initial_artifact_count)
            else:
                qs = self._releasefile_qs().select_for_update()
                try:
                    releasefile = qs[0]
                except IndexError:
                    releasefile = None
            if releasefile is None:
                index_data = None
            elif created:
                index_data = _ArtifactIndexData({}, fresh=True)
            else:
                source_file = releasefile.file
                if source_file.type != ARTIFACT_INDEX_TYPE:
                    raise RuntimeError('Unexpected file type for artifact index')
                raw_data = json.load(source_file.getfile())
                index_data = _ArtifactIndexData(raw_data)
            yield index_data
            if index_data is not None and index_data.changed:
                if created:
                    target_file = releasefile.file
                else:
                    target_file = File.objects.create(name=ARTIFACT_INDEX_FILENAME, type=ARTIFACT_INDEX_TYPE)
                target_file.putfile(BytesIO(json.dumps(index_data.data).encode()))
                artifact_count = index_data.num_files
                if not created:
                    old_file = releasefile.file
                    releasefile.update(file=target_file, artifact_count=artifact_count)
                    old_file.delete()

    def _get_or_create_releasefile(self, initial_artifact_count):
        if False:
            print('Hello World!')
        'Make sure that the release file exists'
        return ReleaseFile.objects.select_for_update().get_or_create(**self._key_fields(), defaults={'artifact_count': initial_artifact_count, 'file': lambda : File.objects.create(name=ARTIFACT_INDEX_FILENAME, type=ARTIFACT_INDEX_TYPE)})

    def _releasefile_qs(self):
        if False:
            while True:
                i = 10
        'QuerySet for selecting artifact index'
        return ReleaseFile.objects.filter(**self._key_fields(), **self._filter_args)

    def _key_fields(self):
        if False:
            for i in range(10):
                print('nop')
        'Columns needed to identify the artifact index in the db'
        return dict(organization_id=self._release.organization_id, release_id=self._release.id, dist_id=self._dist.id if self._dist else self._dist, name=ARTIFACT_INDEX_FILENAME, ident=self._ident)

@sentry_sdk.tracing.trace
def read_artifact_index(release: Release, dist: Optional[Distribution], use_cache: bool=False, **filter_args) -> Optional[dict]:
    if False:
        while True:
            i = 10
    'Get index data'
    guard = _ArtifactIndexGuard(release, dist, **filter_args)
    return guard.readable_data(use_cache)

def _compute_sha1(archive: ReleaseArchive, url: str) -> str:
    if False:
        return 10
    data = archive.read(url)
    return sha1(data).hexdigest()

@sentry_sdk.tracing.trace
def update_artifact_index(release: Release, dist: Optional[Distribution], archive_file: File, temp_file: Optional[IO]=None):
    if False:
        while True:
            i = 10
    'Add information from release archive to artifact index\n\n    :returns: The created ReleaseFile instance\n    '
    releasefile = ReleaseFile.objects.create(name=archive_file.name, release_id=release.id, organization_id=release.organization_id, dist_id=dist.id if dist else dist, file=archive_file, artifact_count=0)
    files_out = {}
    with ReleaseArchive(temp_file or archive_file.getfile()) as archive:
        manifest = archive.manifest
        files = manifest.get('files', {})
        if not files:
            return
        for (filename, info) in files.items():
            info = info.copy()
            url = info.pop('url')
            info['filename'] = filename
            info['archive_ident'] = releasefile.ident
            info['date_created'] = archive_file.timestamp
            info['sha1'] = _compute_sha1(archive, filename)
            info['size'] = archive.info(filename).file_size
            files_out[url] = info
    guard = _ArtifactIndexGuard(release, dist)
    with guard.writable_data(create=True, initial_artifact_count=len(files_out)) as index_data:
        index_data.update_files(files_out)
    return releasefile

@sentry_sdk.tracing.trace
def delete_from_artifact_index(release: Release, dist: Optional[Distribution], url: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Delete the file with the given url from the manifest.\n\n    Does *not* delete the file from the zip archive.\n\n    :returns: True if deleted\n    '
    guard = _ArtifactIndexGuard(release, dist)
    with guard.writable_data(create=False) as index_data:
        if index_data is not None:
            return index_data.delete(url)
    return False