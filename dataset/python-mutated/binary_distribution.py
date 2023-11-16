import codecs
import collections
import errno
import hashlib
import io
import itertools
import json
import os
import pathlib
import re
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import warnings
from contextlib import closing, contextmanager
from gzip import GzipFile
from typing import Dict, List, NamedTuple, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
import llnl.util.filesystem as fsys
import llnl.util.lang
import llnl.util.tty as tty
from llnl.util.filesystem import BaseDirectoryVisitor, mkdirp, visit_directory_tree
import spack.caches
import spack.cmd
import spack.config as config
import spack.database as spack_db
import spack.error
import spack.hooks
import spack.hooks.sbang
import spack.mirror
import spack.oci.image
import spack.oci.oci
import spack.oci.opener
import spack.platforms
import spack.relocate as relocate
import spack.repo
import spack.stage
import spack.store
import spack.traverse as traverse
import spack.util.crypto
import spack.util.file_cache as file_cache
import spack.util.gpg
import spack.util.path
import spack.util.spack_json as sjson
import spack.util.spack_yaml as syaml
import spack.util.timer as timer
import spack.util.url as url_util
import spack.util.web as web_util
from spack.caches import misc_cache_location
from spack.package_prefs import get_package_dir_permissions, get_package_group
from spack.relocate_text import utf8_paths_to_single_binary_regex
from spack.spec import Spec
from spack.stage import Stage
from spack.util.executable import which
BUILD_CACHE_RELATIVE_PATH = 'build_cache'
BUILD_CACHE_KEYS_RELATIVE_PATH = '_pgp'
CURRENT_BUILD_CACHE_LAYOUT_VERSION = 1

class BuildCacheDatabase(spack_db.Database):
    """A database for binary buildcaches.

    A database supports writing buildcache index files, in which case certain fields are not
    needed in each install record, and no locking is required. To use this feature, it provides
    ``lock_cfg=NO_LOCK``, and override the list of ``record_fields``.
    """
    record_fields = ('spec', 'ref_count', 'in_buildcache')

    def __init__(self, root):
        if False:
            i = 10
            return i + 15
        super().__init__(root, lock_cfg=spack_db.NO_LOCK)
        self._write_transaction_impl = llnl.util.lang.nullcontext
        self._read_transaction_impl = llnl.util.lang.nullcontext

class FetchCacheError(Exception):
    """Error thrown when fetching the cache failed, usually a composite error list."""

    def __init__(self, errors):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(errors, list):
            raise TypeError('Expected a list of errors')
        self.errors = errors
        if len(errors) > 1:
            msg = '        Error {0}: {1}: {2}'
            self.message = 'Multiple errors during fetching:\n'
            self.message += '\n'.join((msg.format(i + 1, err.__class__.__name__, str(err)) for (i, err) in enumerate(errors)))
        else:
            err = errors[0]
            self.message = '{0}: {1}'.format(err.__class__.__name__, str(err))
        super().__init__(self.message)

class ListMirrorSpecsError(spack.error.SpackError):
    """Raised when unable to retrieve list of specs from the mirror"""

class BinaryCacheIndex:
    """
    The BinaryCacheIndex tracks what specs are available on (usually remote)
    binary caches.

    This index is "best effort", in the sense that whenever we don't find
    what we're looking for here, we will attempt to fetch it directly from
    configured mirrors anyway.  Thus, it has the potential to speed things
    up, but cache misses shouldn't break any spack functionality.

    At the moment, everything in this class is initialized as lazily as
    possible, so that it avoids slowing anything in spack down until
    absolutely necessary.

    TODO: What's the cost if, e.g., we realize in the middle of a spack
    install that the cache is out of date, and we fetch directly?  Does it
    mean we should have paid the price to update the cache earlier?
    """

    def __init__(self, cache_root: Optional[str]=None):
        if False:
            print('Hello World!')
        self._index_cache_root: str = cache_root or binary_index_location()
        self._index_contents_key = 'contents.json'
        self._index_file_cache: Optional[file_cache.FileCache] = None
        self._local_index_cache: Optional[dict] = None
        self._specs_already_associated: Set[str] = set()
        self._last_fetch_times: Dict[str, float] = {}
        self._mirrors_for_spec: Dict[str, dict] = {}

    def _init_local_index_cache(self):
        if False:
            print('Hello World!')
        if not self._index_file_cache:
            self._index_file_cache = file_cache.FileCache(self._index_cache_root)
            cache_key = self._index_contents_key
            self._index_file_cache.init_entry(cache_key)
            cache_path = self._index_file_cache.cache_path(cache_key)
            self._local_index_cache = {}
            if os.path.isfile(cache_path):
                with self._index_file_cache.read_transaction(cache_key) as cache_file:
                    self._local_index_cache = json.load(cache_file)

    def clear(self):
        if False:
            print('Hello World!')
        'For testing purposes we need to be able to empty the cache and\n        clear associated data structures.'
        if self._index_file_cache:
            self._index_file_cache.destroy()
            self._index_file_cache = None
        self._local_index_cache = None
        self._specs_already_associated = set()
        self._last_fetch_times = {}
        self._mirrors_for_spec = {}

    def _write_local_index_cache(self):
        if False:
            while True:
                i = 10
        self._init_local_index_cache()
        cache_key = self._index_contents_key
        with self._index_file_cache.write_transaction(cache_key) as (old, new):
            json.dump(self._local_index_cache, new)

    def regenerate_spec_cache(self, clear_existing=False):
        if False:
            i = 10
            return i + 15
        'Populate the local cache of concrete specs (``_mirrors_for_spec``)\n        from the locally cached buildcache index files.  This is essentially a\n        no-op if it has already been done, as we keep track of the index\n        hashes for which we have already associated the built specs.'
        self._init_local_index_cache()
        if clear_existing:
            self._specs_already_associated = set()
            self._mirrors_for_spec = {}
        for mirror_url in self._local_index_cache:
            cache_entry = self._local_index_cache[mirror_url]
            cached_index_path = cache_entry['index_path']
            cached_index_hash = cache_entry['index_hash']
            if cached_index_hash not in self._specs_already_associated:
                self._associate_built_specs_with_mirror(cached_index_path, mirror_url)
                self._specs_already_associated.add(cached_index_hash)

    def _associate_built_specs_with_mirror(self, cache_key, mirror_url):
        if False:
            return 10
        tmpdir = tempfile.mkdtemp()
        try:
            db = BuildCacheDatabase(tmpdir)
            try:
                self._index_file_cache.init_entry(cache_key)
                cache_path = self._index_file_cache.cache_path(cache_key)
                with self._index_file_cache.read_transaction(cache_key):
                    db._read_from_file(cache_path)
            except spack_db.InvalidDatabaseVersionError as e:
                tty.warn(f"you need a newer Spack version to read the buildcache index for the following mirror: '{mirror_url}'. {e.database_version_message}")
                return
            spec_list = db.query_local(installed=False, in_buildcache=True)
            for indexed_spec in spec_list:
                dag_hash = indexed_spec.dag_hash()
                if dag_hash not in self._mirrors_for_spec:
                    self._mirrors_for_spec[dag_hash] = []
                for entry in self._mirrors_for_spec[dag_hash]:
                    if entry['mirror_url'] == mirror_url:
                        break
                else:
                    self._mirrors_for_spec[dag_hash].append({'mirror_url': mirror_url, 'spec': indexed_spec})
        finally:
            shutil.rmtree(tmpdir)

    def get_all_built_specs(self):
        if False:
            i = 10
            return i + 15
        spec_list = []
        for dag_hash in self._mirrors_for_spec:
            if len(self._mirrors_for_spec[dag_hash]) > 0:
                spec_list.append(self._mirrors_for_spec[dag_hash][0]['spec'])
        return spec_list

    def find_built_spec(self, spec, mirrors_to_check=None):
        if False:
            for i in range(10):
                print('nop')
        'Look in our cache for the built spec corresponding to ``spec``.\n\n        If the spec can be found among the configured binary mirrors, a\n        list is returned that contains the concrete spec and the mirror url\n        of each mirror where it can be found.  Otherwise, ``None`` is\n        returned.\n\n        This method does not trigger reading anything from remote mirrors, but\n        rather just checks if the concrete spec is found within the cache.\n\n        The cache can be updated by calling ``update()`` on the cache.\n\n        Args:\n            spec (spack.spec.Spec): Concrete spec to find\n            mirrors_to_check: Optional mapping containing mirrors to check.  If\n                None, just assumes all configured mirrors.\n\n        Returns:\n            An list of objects containing the found specs and mirror url where\n                each can be found, e.g.:\n\n                .. code-block:: python\n\n                    [\n                        {\n                            "spec": <concrete-spec>,\n                            "mirror_url": <mirror-root-url>\n                        }\n                    ]\n        '
        return self.find_by_hash(spec.dag_hash(), mirrors_to_check=mirrors_to_check)

    def find_by_hash(self, find_hash, mirrors_to_check=None):
        if False:
            for i in range(10):
                print('nop')
        'Same as find_built_spec but uses the hash of a spec.\n\n        Args:\n            find_hash (str): hash of the spec to search\n            mirrors_to_check: Optional mapping containing mirrors to check.  If\n                None, just assumes all configured mirrors.\n        '
        if find_hash not in self._mirrors_for_spec:
            return []
        results = self._mirrors_for_spec[find_hash]
        if not mirrors_to_check:
            return results
        mirror_urls = mirrors_to_check.values()
        return [r for r in results if r['mirror_url'] in mirror_urls]

    def update_spec(self, spec, found_list):
        if False:
            print('Hello World!')
        "\n        Take list of {'mirror_url': m, 'spec': s} objects and update the local\n        built_spec_cache\n        "
        spec_dag_hash = spec.dag_hash()
        if spec_dag_hash not in self._mirrors_for_spec:
            self._mirrors_for_spec[spec_dag_hash] = found_list
        else:
            current_list = self._mirrors_for_spec[spec_dag_hash]
            for new_entry in found_list:
                for cur_entry in current_list:
                    if new_entry['mirror_url'] == cur_entry['mirror_url']:
                        cur_entry['spec'] = new_entry['spec']
                        break
                else:
                    current_list.append({'mirror_url': new_entry['mirror_url'], 'spec': new_entry['spec']})

    def update(self, with_cooldown=False):
        if False:
            while True:
                i = 10
        'Make sure local cache of buildcache index files is up to date.\n        If the same mirrors are configured as the last time this was called\n        and none of the remote buildcache indices have changed, calling this\n        method will only result in fetching the index hash from each mirror\n        to confirm it is the same as what is stored locally.  Otherwise, the\n        buildcache ``index.json`` and ``index.json.hash`` files are retrieved\n        from each configured mirror and stored locally (both in memory and\n        on disk under ``_index_cache_root``).'
        self._init_local_index_cache()
        configured_mirror_urls = [m.fetch_url for m in spack.mirror.MirrorCollection(binary=True).values()]
        items_to_remove = []
        spec_cache_clear_needed = False
        spec_cache_regenerate_needed = not self._mirrors_for_spec
        fetch_errors = []
        all_methods_failed = True
        ttl = spack.config.get('config:binary_index_ttl', 600)
        now = time.time()
        for cached_mirror_url in self._local_index_cache:
            cache_entry = self._local_index_cache[cached_mirror_url]
            cached_index_path = cache_entry['index_path']
            if cached_mirror_url in configured_mirror_urls:
                if with_cooldown and ttl > 0 and (cached_mirror_url in self._last_fetch_times) and (now - self._last_fetch_times[cached_mirror_url][0] < ttl):
                    if self._last_fetch_times[cached_mirror_url][1]:
                        all_methods_failed = False
                else:
                    try:
                        needs_regen = self._fetch_and_cache_index(cached_mirror_url, cache_entry=cache_entry)
                        self._last_fetch_times[cached_mirror_url] = (now, True)
                        all_methods_failed = False
                    except FetchIndexError as e:
                        needs_regen = False
                        fetch_errors.append(e)
                        self._last_fetch_times[cached_mirror_url] = (now, False)
                    spec_cache_clear_needed |= needs_regen
                    spec_cache_regenerate_needed |= needs_regen
            else:
                items_to_remove.append({'url': cached_mirror_url, 'cache_key': os.path.join(self._index_cache_root, cached_index_path)})
                if cached_mirror_url in self._last_fetch_times:
                    del self._last_fetch_times[cached_mirror_url]
                spec_cache_clear_needed = True
                spec_cache_regenerate_needed = True
        for item in items_to_remove:
            url = item['url']
            cache_key = item['cache_key']
            self._index_file_cache.remove(cache_key)
            del self._local_index_cache[url]
        for mirror_url in configured_mirror_urls:
            if mirror_url in self._local_index_cache:
                continue
            try:
                needs_regen = self._fetch_and_cache_index(mirror_url)
                self._last_fetch_times[mirror_url] = (now, True)
                all_methods_failed = False
            except FetchIndexError as e:
                fetch_errors.append(e)
                needs_regen = False
                self._last_fetch_times[mirror_url] = (now, False)
            if needs_regen:
                spec_cache_regenerate_needed = True
        self._write_local_index_cache()
        if configured_mirror_urls and all_methods_failed:
            raise FetchCacheError(fetch_errors)
        if fetch_errors:
            tty.warn('The following issues were ignored while updating the indices of binary caches', FetchCacheError(fetch_errors))
        if spec_cache_regenerate_needed:
            self.regenerate_spec_cache(clear_existing=spec_cache_clear_needed)

    def _fetch_and_cache_index(self, mirror_url, cache_entry={}):
        if False:
            for i in range(10):
                print('nop')
        'Fetch a buildcache index file from a remote mirror and cache it.\n\n        If we already have a cached index from this mirror, then we first\n        check if the hash has changed, and we avoid fetching it if not.\n\n        Args:\n            mirror_url (str): Base url of mirror\n            cache_entry (dict): Old cache metadata with keys ``index_hash``, ``index_path``,\n                ``etag``\n\n        Returns:\n            True if the local index.json was updated.\n\n        Throws:\n            FetchIndexError\n        '
        scheme = urllib.parse.urlparse(mirror_url).scheme
        if scheme != 'oci' and (not web_util.url_exists(url_util.join(mirror_url, BUILD_CACHE_RELATIVE_PATH, 'index.json'))):
            return False
        if scheme == 'oci':
            fetcher = OCIIndexFetcher(mirror_url, cache_entry.get('index_hash', None))
        elif cache_entry.get('etag'):
            fetcher = EtagIndexFetcher(mirror_url, cache_entry['etag'])
        else:
            fetcher = DefaultIndexFetcher(mirror_url, local_hash=cache_entry.get('index_hash', None))
        result = fetcher.conditional_fetch()
        if result.fresh:
            return False
        url_hash = compute_hash(mirror_url)
        cache_key = '{}_{}.json'.format(url_hash[:10], result.hash[:10])
        self._index_file_cache.init_entry(cache_key)
        with self._index_file_cache.write_transaction(cache_key) as (old, new):
            new.write(result.data)
        self._local_index_cache[mirror_url] = {'index_hash': result.hash, 'index_path': cache_key, 'etag': result.etag}
        old_cache_key = cache_entry.get('index_path', None)
        if old_cache_key:
            self._index_file_cache.remove(old_cache_key)
        return True

def binary_index_location():
    if False:
        for i in range(10):
            print('nop')
    "Set up a BinaryCacheIndex for remote buildcache dbs in the user's homedir."
    cache_root = os.path.join(misc_cache_location(), 'indices')
    return spack.util.path.canonicalize_path(cache_root)
BINARY_INDEX: BinaryCacheIndex = llnl.util.lang.Singleton(BinaryCacheIndex)

class NoOverwriteException(spack.error.SpackError):
    """Raised when a file would be overwritten"""

    def __init__(self, file_path):
        if False:
            print('Hello World!')
        super().__init__(f'Refusing to overwrite the following file: {file_path}')

class NoGpgException(spack.error.SpackError):
    """
    Raised when gpg2 is not in PATH
    """

    def __init__(self, msg):
        if False:
            return 10
        super().__init__(msg)

class NoKeyException(spack.error.SpackError):
    """
    Raised when gpg has no default key added.
    """

    def __init__(self, msg):
        if False:
            while True:
                i = 10
        super().__init__(msg)

class PickKeyException(spack.error.SpackError):
    """
    Raised when multiple keys can be used to sign.
    """

    def __init__(self, keys):
        if False:
            i = 10
            return i + 15
        err_msg = 'Multiple keys available for signing\n%s\n' % keys
        err_msg += 'Use spack buildcache create -k <key hash> to pick a key.'
        super().__init__(err_msg)

class NoVerifyException(spack.error.SpackError):
    """
    Raised if file fails signature verification.
    """
    pass

class NoChecksumException(spack.error.SpackError):
    """
    Raised if file fails checksum verification.
    """

    def __init__(self, path, size, contents, algorithm, expected, computed):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(f'{algorithm} checksum failed for {path}', f'Expected {expected} but got {computed}. File size = {size} bytes. Contents = {contents!r}')

class NewLayoutException(spack.error.SpackError):
    """
    Raised if directory layout is different from buildcache.
    """

    def __init__(self, msg):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(msg)

class InvalidMetadataFile(spack.error.SpackError):
    pass

class UnsignedPackageException(spack.error.SpackError):
    """
    Raised if installation of unsigned package is attempted without
    the use of ``--no-check-signature``.
    """

def compute_hash(data):
    if False:
        while True:
            i = 10
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def build_cache_relative_path():
    if False:
        i = 10
        return i + 15
    return BUILD_CACHE_RELATIVE_PATH

def build_cache_keys_relative_path():
    if False:
        while True:
            i = 10
    return BUILD_CACHE_KEYS_RELATIVE_PATH

def build_cache_prefix(prefix):
    if False:
        while True:
            i = 10
    return os.path.join(prefix, build_cache_relative_path())

def buildinfo_file_name(prefix):
    if False:
        while True:
            i = 10
    'Filename of the binary package meta-data file'
    return os.path.join(prefix, '.spack', 'binary_distribution')

def read_buildinfo_file(prefix):
    if False:
        return 10
    'Read buildinfo file'
    with open(buildinfo_file_name(prefix), 'r') as f:
        return syaml.load(f)

class BuildManifestVisitor(BaseDirectoryVisitor):
    """Visitor that collects a list of files and symlinks
    that can be checked for need of relocation. It knows how
    to dedupe hardlinks and deal with symlinks to files and
    directories."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.visited = set()
        self.files = []
        self.symlinks = []

    def seen_before(self, root, rel_path):
        if False:
            while True:
                i = 10
        stat_result = os.lstat(os.path.join(root, rel_path))
        if stat_result.st_nlink == 1:
            return False
        identifier = (stat_result.st_dev, stat_result.st_ino)
        if identifier in self.visited:
            return True
        else:
            self.visited.add(identifier)
            return False

    def visit_file(self, root, rel_path, depth):
        if False:
            i = 10
            return i + 15
        if self.seen_before(root, rel_path):
            return
        self.files.append(rel_path)

    def visit_symlinked_file(self, root, rel_path, depth):
        if False:
            print('Hello World!')
        self.symlinks.append(rel_path)

    def before_visit_dir(self, root, rel_path, depth):
        if False:
            while True:
                i = 10
        return os.path.basename(rel_path) not in ('.spack', 'man')

    def before_visit_symlinked_dir(self, root, rel_path, depth):
        if False:
            for i in range(10):
                print('nop')
        self.visit_symlinked_file(root, rel_path, depth)
        return False

def file_matches(path, regex):
    if False:
        return 10
    with open(path, 'rb') as f:
        contents = f.read()
    return bool(regex.search(contents))

def get_buildfile_manifest(spec):
    if False:
        i = 10
        return i + 15
    "\n    Return a data structure with information about a build, including\n    text_to_relocate, binary_to_relocate, binary_to_relocate_fullpath\n    link_to_relocate, and other, which means it doesn't fit any of previous\n    checks (and should not be relocated). We exclude docs (man) and\n    metadata (.spack). This can be used to find a particular kind of file\n    in spack, or to generate the build metadata.\n    "
    data = {'text_to_relocate': [], 'binary_to_relocate': [], 'link_to_relocate': [], 'other': [], 'binary_to_relocate_fullpath': [], 'hardlinks_deduped': True}
    visitor = BuildManifestVisitor()
    root = spec.prefix
    visit_directory_tree(root, visitor)
    prefixes = [d.prefix for d in spec.traverse(root=True, deptype='all') if not d.external]
    prefixes.append(spack.hooks.sbang.sbang_install_path())
    prefixes.append(str(spack.store.STORE.layout.root))
    regex = utf8_paths_to_single_binary_regex(prefixes)
    for rel_path in visitor.symlinks:
        abs_path = os.path.join(root, rel_path)
        link = os.readlink(abs_path)
        if os.path.isabs(link) and link.startswith(spack.store.STORE.layout.root):
            data['link_to_relocate'].append(rel_path)
    for rel_path in visitor.files:
        abs_path = os.path.join(root, rel_path)
        (m_type, m_subtype) = fsys.mime_type(abs_path)
        if relocate.needs_binary_relocation(m_type, m_subtype):
            if m_subtype in ('x-executable', 'x-sharedlib', 'x-pie-executable') and sys.platform != 'darwin' or (m_subtype in 'x-mach-binary' and sys.platform == 'darwin') or (not rel_path.endswith('.o')):
                data['binary_to_relocate'].append(rel_path)
                data['binary_to_relocate_fullpath'].append(abs_path)
                continue
        elif relocate.needs_text_relocation(m_type, m_subtype) and file_matches(abs_path, regex):
            data['text_to_relocate'].append(rel_path)
            continue
        data['other'].append(abs_path)
    return data

def hashes_to_prefixes(spec):
    if False:
        while True:
            i = 10
    'Return a dictionary of hashes to prefixes for a spec and its deps, excluding externals'
    return {s.dag_hash(): str(s.prefix) for s in itertools.chain(spec.traverse(root=True, deptype='link'), spec.dependencies(deptype='run')) if not s.external}

def get_buildinfo_dict(spec):
    if False:
        return 10
    'Create metadata for a tarball'
    manifest = get_buildfile_manifest(spec)
    return {'sbang_install_path': spack.hooks.sbang.sbang_install_path(), 'buildpath': spack.store.STORE.layout.root, 'spackprefix': spack.paths.prefix, 'relative_prefix': os.path.relpath(spec.prefix, spack.store.STORE.layout.root), 'relocate_textfiles': manifest['text_to_relocate'], 'relocate_binaries': manifest['binary_to_relocate'], 'relocate_links': manifest['link_to_relocate'], 'hardlinks_deduped': manifest['hardlinks_deduped'], 'hash_to_prefix': hashes_to_prefixes(spec)}

def tarball_directory_name(spec):
    if False:
        while True:
            i = 10
    '\n    Return name of the tarball directory according to the convention\n    <os>-<architecture>/<compiler>/<package>-<version>/\n    '
    return spec.format_path('{architecture}/{compiler.name}-{compiler.version}/{name}-{version}')

def tarball_name(spec, ext):
    if False:
        print('Hello World!')
    '\n    Return the name of the tarfile according to the convention\n    <os>-<architecture>-<package>-<dag_hash><ext>\n    '
    spec_formatted = spec.format_path('{architecture}-{compiler.name}-{compiler.version}-{name}-{version}-{hash}')
    return f'{spec_formatted}{ext}'

def tarball_path_name(spec, ext):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the full path+name for a given spec according to the convention\n    <tarball_directory_name>/<tarball_name>\n    '
    return os.path.join(tarball_directory_name(spec), tarball_name(spec, ext))

def select_signing_key(key=None):
    if False:
        while True:
            i = 10
    if key is None:
        keys = spack.util.gpg.signing_keys()
        if len(keys) == 1:
            key = keys[0]
        if len(keys) > 1:
            raise PickKeyException(str(keys))
        if len(keys) == 0:
            raise NoKeyException('No default key available for signing.\nUse spack gpg init and spack gpg create to create a default key.')
    return key

def sign_specfile(key, force, specfile_path):
    if False:
        return 10
    signed_specfile_path = '%s.sig' % specfile_path
    if os.path.exists(signed_specfile_path):
        if force:
            os.remove(signed_specfile_path)
        else:
            raise NoOverwriteException(signed_specfile_path)
    key = select_signing_key(key)
    spack.util.gpg.sign(key, specfile_path, signed_specfile_path, clearsign=True)

def _read_specs_and_push_index(file_list, read_method, cache_prefix, db, temp_dir, concurrency):
    if False:
        i = 10
        return i + 15
    'Read all the specs listed in the provided list, using thread given thread parallelism,\n        generate the index, and push it to the mirror.\n\n    Args:\n        file_list (list(str)): List of urls or file paths pointing at spec files to read\n        read_method: A function taking a single argument, either a url or a file path,\n            and which reads the spec file at that location, and returns the spec.\n        cache_prefix (str): prefix of the build cache on s3 where index should be pushed.\n        db: A spack database used for adding specs and then writing the index.\n        temp_dir (str): Location to write index.json and hash for pushing\n        concurrency (int): Number of parallel processes to use when fetching\n    '
    for file in file_list:
        contents = read_method(file)
        if file.endswith('.json.sig'):
            specfile_json = Spec.extract_json_from_clearsig(contents)
            fetched_spec = Spec.from_dict(specfile_json)
        elif file.endswith('.json'):
            fetched_spec = Spec.from_json(contents)
        else:
            continue
        db.add(fetched_spec, None)
        db.mark(fetched_spec, 'in_buildcache', True)
    index_json_path = os.path.join(temp_dir, 'index.json')
    with open(index_json_path, 'w') as f:
        db._write_to_file(f)
    with open(index_json_path) as f:
        index_string = f.read()
        index_hash = compute_hash(index_string)
    index_hash_path = os.path.join(temp_dir, 'index.json.hash')
    with open(index_hash_path, 'w') as f:
        f.write(index_hash)
    web_util.push_to_url(index_json_path, url_util.join(cache_prefix, 'index.json'), keep_original=False, extra_args={'ContentType': 'application/json', 'CacheControl': 'no-cache'})
    web_util.push_to_url(index_hash_path, url_util.join(cache_prefix, 'index.json.hash'), keep_original=False, extra_args={'ContentType': 'text/plain', 'CacheControl': 'no-cache'})

def _specs_from_cache_aws_cli(cache_prefix):
    if False:
        i = 10
        return i + 15
    'Use aws cli to sync all the specs into a local temporary directory.\n\n    Args:\n        cache_prefix (str): prefix of the build cache on s3\n\n    Return:\n        List of the local file paths and a function that can read each one from the file system.\n    '
    read_fn = None
    file_list = None
    aws = which('aws')

    def file_read_method(file_path):
        if False:
            print('Hello World!')
        with open(file_path) as fd:
            return fd.read()
    tmpspecsdir = tempfile.mkdtemp()
    sync_command_args = ['s3', 'sync', '--exclude', '*', '--include', '*.spec.json.sig', '--include', '*.spec.json', cache_prefix, tmpspecsdir]
    try:
        tty.debug('Using aws s3 sync to download specs from {0} to {1}'.format(cache_prefix, tmpspecsdir))
        aws(*sync_command_args, output=os.devnull, error=os.devnull)
        file_list = fsys.find(tmpspecsdir, ['*.spec.json.sig', '*.spec.json'])
        read_fn = file_read_method
    except Exception:
        tty.warn('Failed to use aws s3 sync to retrieve specs, falling back to parallel fetch')
        shutil.rmtree(tmpspecsdir)
    return (file_list, read_fn)

def _specs_from_cache_fallback(cache_prefix):
    if False:
        while True:
            i = 10
    'Use spack.util.web module to get a list of all the specs at the remote url.\n\n    Args:\n        cache_prefix (str): Base url of mirror (location of spec files)\n\n    Return:\n        The list of complete spec file urls and a function that can read each one from its\n            remote location (also using the spack.util.web module).\n    '
    read_fn = None
    file_list = None

    def url_read_method(url):
        if False:
            i = 10
            return i + 15
        contents = None
        try:
            (_, _, spec_file) = web_util.read_from_url(url)
            contents = codecs.getreader('utf-8')(spec_file).read()
        except (URLError, web_util.SpackWebError) as url_err:
            tty.error('Error reading specfile: {0}'.format(url))
            tty.error(url_err)
        return contents
    try:
        file_list = [url_util.join(cache_prefix, entry) for entry in web_util.list_url(cache_prefix) if entry.endswith('spec.json') or entry.endswith('spec.json.sig')]
        read_fn = url_read_method
    except KeyError as inst:
        msg = 'No packages at {0}: {1}'.format(cache_prefix, inst)
        tty.warn(msg)
    except Exception as err:
        msg = 'Encountered problem listing packages at {0}: {1}'.format(cache_prefix, err)
        tty.warn(msg)
    return (file_list, read_fn)

def _spec_files_from_cache(cache_prefix):
    if False:
        print('Hello World!')
    'Get a list of all the spec files in the mirror and a function to\n    read them.\n\n    Args:\n        cache_prefix (str): Base url of mirror (location of spec files)\n\n    Return:\n        A tuple where the first item is a list of absolute file paths or\n        urls pointing to the specs that should be read from the mirror,\n        and the second item is a function taking a url or file path and\n        returning the spec read from that location.\n    '
    callbacks = []
    if cache_prefix.startswith('s3'):
        callbacks.append(_specs_from_cache_aws_cli)
    callbacks.append(_specs_from_cache_fallback)
    for specs_from_cache_fn in callbacks:
        (file_list, read_fn) = specs_from_cache_fn(cache_prefix)
        if file_list:
            return (file_list, read_fn)
    raise ListMirrorSpecsError('Failed to get list of specs from {0}'.format(cache_prefix))

def generate_package_index(cache_prefix, concurrency=32):
    if False:
        print('Hello World!')
    'Create or replace the build cache index on the given mirror.  The\n    buildcache index contains an entry for each binary package under the\n    cache_prefix.\n\n    Args:\n        cache_prefix(str): Base url of binary mirror.\n        concurrency: (int): The desired threading concurrency to use when\n            fetching the spec files from the mirror.\n\n    Return:\n        None\n    '
    try:
        (file_list, read_fn) = _spec_files_from_cache(cache_prefix)
    except ListMirrorSpecsError as err:
        tty.error('Unable to generate package index, {0}'.format(err))
        return
    tty.debug('Retrieving spec descriptor files from {0} to build index'.format(cache_prefix))
    tmpdir = tempfile.mkdtemp()
    db = BuildCacheDatabase(tmpdir)
    db.root = None
    db_root_dir = db.database_directory
    try:
        _read_specs_and_push_index(file_list, read_fn, cache_prefix, db, db_root_dir, concurrency)
    except Exception as err:
        msg = 'Encountered problem pushing package index to {0}: {1}'.format(cache_prefix, err)
        tty.warn(msg)
        tty.debug('\n' + traceback.format_exc())
    finally:
        shutil.rmtree(tmpdir)

def generate_key_index(key_prefix, tmpdir=None):
    if False:
        i = 10
        return i + 15
    'Create the key index page.\n\n    Creates (or replaces) the "index.json" page at the location given in\n    key_prefix.  This page contains an entry for each key (.pub) under\n    key_prefix.\n    '
    tty.debug(' '.join(('Retrieving key.pub files from', url_util.format(key_prefix), 'to build key index')))
    try:
        fingerprints = (entry[:-4] for entry in web_util.list_url(key_prefix, recursive=False) if entry.endswith('.pub'))
    except KeyError as inst:
        msg = 'No keys at {0}: {1}'.format(key_prefix, inst)
        tty.warn(msg)
        return
    except Exception as err:
        msg = 'Encountered problem listing keys at {0}: {1}'.format(key_prefix, err)
        tty.warn(msg)
        return
    remove_tmpdir = False
    keys_local = url_util.local_file_path(key_prefix)
    if keys_local:
        target = os.path.join(keys_local, 'index.json')
    else:
        if not tmpdir:
            tmpdir = tempfile.mkdtemp()
            remove_tmpdir = True
        target = os.path.join(tmpdir, 'index.json')
    index = {'keys': dict(((fingerprint, {}) for fingerprint in sorted(set(fingerprints))))}
    with open(target, 'w') as f:
        sjson.dump(index, f)
    if not keys_local:
        try:
            web_util.push_to_url(target, url_util.join(key_prefix, 'index.json'), keep_original=False, extra_args={'ContentType': 'application/json'})
        except Exception as err:
            msg = 'Encountered problem pushing key index to {0}: {1}'.format(key_prefix, err)
            tty.warn(msg)
        finally:
            if remove_tmpdir:
                shutil.rmtree(tmpdir)

@contextmanager
def gzip_compressed_tarfile(path):
    if False:
        print('Hello World!')
    'Create a reproducible, compressed tarfile'
    with open(path, 'wb') as f, ChecksumWriter(f) as inner_checksum, closing(GzipFile(filename='', mode='wb', compresslevel=6, mtime=0, fileobj=inner_checksum)) as gzip_file, ChecksumWriter(gzip_file) as outer_checksum, tarfile.TarFile(name='', mode='w', fileobj=outer_checksum) as tar:
        yield (tar, inner_checksum, outer_checksum)

def _tarinfo_name(absolute_path: str, *, _path=pathlib.PurePath) -> str:
    if False:
        while True:
            i = 10
    'Compute tarfile entry name as the relative path from the (system) root.'
    return _path(*_path(absolute_path).parts[1:]).as_posix()

def tarfile_of_spec_prefix(tar: tarfile.TarFile, prefix: str) -> None:
    if False:
        while True:
            i = 10
    'Create a tarfile of an install prefix of a spec. Skips existing buildinfo file.\n    Only adds regular files, symlinks and dirs. Skips devices, fifos. Preserves hardlinks.\n    Normalizes permissions like git. Tar entries are added in depth-first pre-order, with\n    dir entries partitioned by file | dir, and sorted alphabetically, for reproducibility.\n    Partitioning ensures only one dir is in memory at a time, and sorting improves compression.\n\n    Args:\n        tar: tarfile object to add files to\n        prefix: absolute install prefix of spec'
    if not os.path.isabs(prefix) or not os.path.isdir(prefix):
        raise ValueError(f"prefix '{prefix}' must be an absolute path to a directory")
    hardlink_to_tarinfo_name: Dict[Tuple[int, int], str] = dict()
    stat_key = lambda stat: (stat.st_dev, stat.st_ino)
    try:
        files_to_skip = [stat_key(os.lstat(buildinfo_file_name(prefix)))]
    except OSError:
        files_to_skip = []
    dir_stack = [prefix]
    while dir_stack:
        dir = dir_stack.pop()
        dir_info = tarfile.TarInfo(_tarinfo_name(dir))
        dir_info.type = tarfile.DIRTYPE
        dir_info.mode = 493
        tar.addfile(dir_info)
        with os.scandir(dir) as it:
            entries = sorted(it, key=lambda entry: entry.name)
        new_dirs = []
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                new_dirs.append(entry.path)
                continue
            file_info = tarfile.TarInfo(_tarinfo_name(entry.path))
            s = entry.stat(follow_symlinks=False)
            id = stat_key(s)
            if id in files_to_skip:
                continue
            file_info.mode = 420 if s.st_mode & 64 == 0 else 493
            if entry.is_symlink():
                file_info.type = tarfile.SYMTYPE
                file_info.linkname = os.readlink(entry.path)
                tar.addfile(file_info)
            elif entry.is_file(follow_symlinks=False):
                if s.st_nlink > 1:
                    if id in hardlink_to_tarinfo_name:
                        file_info.type = tarfile.LNKTYPE
                        file_info.linkname = hardlink_to_tarinfo_name[id]
                        tar.addfile(file_info)
                        continue
                    hardlink_to_tarinfo_name[id] = file_info.name
                file_info.type = tarfile.REGTYPE
                file_info.size = s.st_size
                with open(entry.path, 'rb') as f:
                    tar.addfile(file_info, f)
        dir_stack.extend(reversed(new_dirs))

class ChecksumWriter(io.BufferedIOBase):
    """Checksum writer computes a checksum while writing to a file."""
    myfileobj = None

    def __init__(self, fileobj, algorithm=hashlib.sha256):
        if False:
            while True:
                i = 10
        self.fileobj = fileobj
        self.hasher = algorithm()
        self.length = 0

    def hexdigest(self):
        if False:
            return 10
        return self.hasher.hexdigest()

    def write(self, data):
        if False:
            print('Hello World!')
        if isinstance(data, (bytes, bytearray)):
            length = len(data)
        else:
            data = memoryview(data)
            length = data.nbytes
        if length > 0:
            self.fileobj.write(data)
            self.hasher.update(data)
        self.length += length
        return length

    def read(self, size=-1):
        if False:
            i = 10
            return i + 15
        raise OSError(errno.EBADF, 'read() on write-only object')

    def read1(self, size=-1):
        if False:
            return 10
        raise OSError(errno.EBADF, 'read1() on write-only object')

    def peek(self, n):
        if False:
            return 10
        raise OSError(errno.EBADF, 'peek() on write-only object')

    @property
    def closed(self):
        if False:
            for i in range(10):
                print('nop')
        return self.fileobj is None

    def close(self):
        if False:
            i = 10
            return i + 15
        fileobj = self.fileobj
        if fileobj is None:
            return
        self.fileobj.close()
        self.fileobj = None

    def flush(self):
        if False:
            while True:
                i = 10
        self.fileobj.flush()

    def fileno(self):
        if False:
            print('Hello World!')
        return self.fileobj.fileno()

    def rewind(self):
        if False:
            print('Hello World!')
        raise OSError("Can't rewind while computing checksum")

    def readable(self):
        if False:
            print('Hello World!')
        return False

    def writable(self):
        if False:
            return 10
        return True

    def seekable(self):
        if False:
            print('Hello World!')
        return True

    def tell(self):
        if False:
            print('Hello World!')
        return self.fileobj.tell()

    def seek(self, offset, whence=io.SEEK_SET):
        if False:
            while True:
                i = 10
        if offset == 0 and whence == io.SEEK_CUR:
            return
        raise OSError("Can't seek while computing checksum")

    def readline(self, size=-1):
        if False:
            print('Hello World!')
        raise OSError(errno.EBADF, 'readline() on write-only object')

def _do_create_tarball(tarfile_path: str, binaries_dir: str, buildinfo: dict):
    if False:
        print('Hello World!')
    with gzip_compressed_tarfile(tarfile_path) as (tar, inner_checksum, outer_checksum):
        tarfile_of_spec_prefix(tar, binaries_dir)
        bstring = syaml.dump(buildinfo, default_flow_style=True).encode('utf-8')
        tarinfo = tarfile.TarInfo(name=_tarinfo_name(buildinfo_file_name(binaries_dir)))
        tarinfo.type = tarfile.REGTYPE
        tarinfo.size = len(bstring)
        tarinfo.mode = 420
        tar.addfile(tarinfo, io.BytesIO(bstring))
    return (inner_checksum.hexdigest(), outer_checksum.hexdigest())

class PushOptions(NamedTuple):
    force: bool = False
    regenerate_index: bool = False
    unsigned: bool = False
    key: Optional[str] = None

def push_or_raise(spec: Spec, out_url: str, options: PushOptions):
    if False:
        print('Hello World!')
    '\n    Build a tarball from given spec and put it into the directory structure\n    used at the mirror (following <tarball_directory_name>).\n\n    This method raises :py:class:`NoOverwriteException` when ``force=False`` and the tarball or\n    spec.json file already exist in the buildcache.\n    '
    if not spec.concrete:
        raise ValueError('spec must be concrete to build tarball')
    with tempfile.TemporaryDirectory(dir=spack.stage.get_stage_root()) as tmpdir:
        _build_tarball_in_stage_dir(spec, out_url, stage_dir=tmpdir, options=options)

def _build_tarball_in_stage_dir(spec: Spec, out_url: str, stage_dir: str, options: PushOptions):
    if False:
        return 10
    cache_prefix = build_cache_prefix(stage_dir)
    tarfile_name = tarball_name(spec, '.spack')
    tarfile_dir = os.path.join(cache_prefix, tarball_directory_name(spec))
    tarfile_path = os.path.join(tarfile_dir, tarfile_name)
    spackfile_path = os.path.join(cache_prefix, tarball_path_name(spec, '.spack'))
    remote_spackfile_path = url_util.join(out_url, os.path.relpath(spackfile_path, stage_dir))
    mkdirp(tarfile_dir)
    if web_util.url_exists(remote_spackfile_path):
        if options.force:
            web_util.remove_url(remote_spackfile_path)
        else:
            raise NoOverwriteException(url_util.format(remote_spackfile_path))
    spec_file = spack.store.STORE.layout.spec_file_path(spec)
    specfile_name = tarball_name(spec, '.spec.json')
    specfile_path = os.path.realpath(os.path.join(cache_prefix, specfile_name))
    signed_specfile_path = '{0}.sig'.format(specfile_path)
    remote_specfile_path = url_util.join(out_url, os.path.relpath(specfile_path, os.path.realpath(stage_dir)))
    remote_signed_specfile_path = '{0}.sig'.format(remote_specfile_path)
    if options.force:
        if web_util.url_exists(remote_specfile_path):
            web_util.remove_url(remote_specfile_path)
        if web_util.url_exists(remote_signed_specfile_path):
            web_util.remove_url(remote_signed_specfile_path)
    elif web_util.url_exists(remote_specfile_path) or web_util.url_exists(remote_signed_specfile_path):
        raise NoOverwriteException(url_util.format(remote_specfile_path))
    binaries_dir = spec.prefix
    buildinfo = get_buildinfo_dict(spec)
    (checksum, _) = _do_create_tarball(tarfile_path, binaries_dir, buildinfo)
    with open(spec_file, 'r') as inputfile:
        content = inputfile.read()
        if spec_file.endswith('.json'):
            spec_dict = sjson.load(content)
        else:
            raise ValueError('{0} not a valid spec file type'.format(spec_file))
    spec_dict['buildcache_layout_version'] = CURRENT_BUILD_CACHE_LAYOUT_VERSION
    spec_dict['binary_cache_checksum'] = {'hash_algorithm': 'sha256', 'hash': checksum}
    with open(specfile_path, 'w') as outfile:
        json.dump(spec_dict, outfile, indent=0, separators=(',', ':'))
    if not options.unsigned:
        key = select_signing_key(options.key)
        sign_specfile(key, options.force, specfile_path)
    web_util.push_to_url(spackfile_path, remote_spackfile_path, keep_original=False)
    web_util.push_to_url(signed_specfile_path if not options.unsigned else specfile_path, remote_signed_specfile_path if not options.unsigned else remote_specfile_path, keep_original=False)
    if not options.unsigned:
        push_keys(out_url, keys=[key], regenerate_index=options.regenerate_index, tmpdir=stage_dir)
    if options.regenerate_index:
        generate_package_index(url_util.join(out_url, os.path.relpath(cache_prefix, stage_dir)))
    return None

class NotInstalledError(spack.error.SpackError):
    """Raised when a spec is not installed but picked to be packaged."""

    def __init__(self, specs: List[Spec]):
        if False:
            while True:
                i = 10
        super().__init__('Cannot push non-installed packages', ', '.join((s.cformat('{name}{@version}{/hash:7}') for s in specs)))

def specs_to_be_packaged(specs: List[Spec], root: bool=True, dependencies: bool=True) -> List[Spec]:
    if False:
        while True:
            i = 10
    'Return the list of nodes to be packaged, given a list of specs.\n    Raises NotInstalledError if a spec is not installed but picked to be packaged.\n\n    Args:\n        specs: list of root specs to be processed\n        root: include the root of each spec in the nodes\n        dependencies: include the dependencies of each\n            spec in the nodes\n    '
    if not root and (not dependencies):
        return []
    with spack.store.STORE.db.read_transaction():
        if root:
            uninstalled_roots = list((s for s in specs if not s.installed))
            if uninstalled_roots:
                raise NotInstalledError(uninstalled_roots)
            roots = specs
        else:
            roots = []
        if dependencies:
            deps = list(traverse.traverse_nodes(specs, deptype='all', order='breadth', root=False, key=traverse.by_dag_hash))
            uninstalled_deps = list((s for s in deps if not s.installed))
            if uninstalled_deps:
                raise NotInstalledError(uninstalled_deps)
        else:
            deps = []
    return [s for s in itertools.chain(roots, deps) if not s.external]

def push(spec: Spec, mirror_url: str, options: PushOptions):
    if False:
        i = 10
        return i + 15
    'Create and push binary package for a single spec to the specified\n    mirror url.\n\n    Args:\n        spec: Spec to package and push\n        mirror_url: Desired destination url for binary package\n        options:\n\n    Returns:\n        True if package was pushed, False otherwise.\n\n    '
    try:
        push_or_raise(spec, mirror_url, options)
    except NoOverwriteException as e:
        warnings.warn(str(e))
        return False
    return True

def try_verify(specfile_path):
    if False:
        return 10
    'Utility function to attempt to verify a local file.  Assumes the\n    file is a clearsigned signature file.\n\n    Args:\n        specfile_path (str): Path to file to be verified.\n\n    Returns:\n        ``True`` if the signature could be verified, ``False`` otherwise.\n    '
    suppress = config.get('config:suppress_gpg_warnings', False)
    try:
        spack.util.gpg.verify(specfile_path, suppress_warnings=suppress)
    except Exception:
        return False
    return True

def try_fetch(url_to_fetch):
    if False:
        while True:
            i = 10
    'Utility function to try and fetch a file from a url, stage it\n    locally, and return the path to the staged file.\n\n    Args:\n        url_to_fetch (str): Url pointing to remote resource to fetch\n\n    Returns:\n        Path to locally staged resource or ``None`` if it could not be fetched.\n    '
    stage = Stage(url_to_fetch, keep=True)
    stage.create()
    try:
        stage.fetch()
    except spack.error.FetchError:
        stage.destroy()
        return None
    return stage

def _delete_staged_downloads(download_result):
    if False:
        print('Hello World!')
    'Clean up stages used to download tarball and specfile'
    download_result['tarball_stage'].destroy()
    download_result['specfile_stage'].destroy()

def _get_valid_spec_file(path: str, max_supported_layout: int) -> Tuple[Dict, int]:
    if False:
        return 10
    'Read and validate a spec file, returning the spec dict with its layout version, or raising\n    InvalidMetadataFile if invalid.'
    try:
        with open(path, 'rb') as f:
            binary_content = f.read()
    except OSError:
        raise InvalidMetadataFile(f'No such file: {path}')
    if binary_content[:2] == b'\x1f\x8b':
        raise InvalidMetadataFile('Compressed spec files are not supported')
    try:
        as_string = binary_content.decode('utf-8')
        if path.endswith('.json.sig'):
            spec_dict = Spec.extract_json_from_clearsig(as_string)
        else:
            spec_dict = json.loads(as_string)
    except Exception as e:
        raise InvalidMetadataFile(f'Could not parse {path} due to: {e}') from e
    try:
        layout_version = int(spec_dict.get('buildcache_layout_version', 0))
    except ValueError as e:
        raise InvalidMetadataFile('Could not parse layout version') from e
    if layout_version > max_supported_layout:
        raise InvalidMetadataFile(f'Layout version {layout_version} is too new for this version of Spack')
    return (spec_dict, layout_version)

def download_tarball(spec, unsigned=False, mirrors_for_spec=None):
    if False:
        while True:
            i = 10
    '\n    Download binary tarball for given package into stage area, returning\n    path to downloaded tarball if successful, None otherwise.\n\n    Args:\n        spec (spack.spec.Spec): Concrete spec\n        unsigned (bool): Whether or not to require signed binaries\n        mirrors_for_spec (list): Optional list of concrete specs and mirrors\n            obtained by calling binary_distribution.get_mirrors_for_spec().\n            These will be checked in order first before looking in other\n            configured mirrors.\n\n    Returns:\n        ``None`` if the tarball could not be downloaded (maybe also verified,\n        depending on whether new-style signed binary packages were found).\n        Otherwise, return an object indicating the path to the downloaded\n        tarball, the path to the downloaded specfile (in the case of new-style\n        buildcache), and whether or not the tarball is already verified.\n\n    .. code-block:: JSON\n\n       {\n           "tarball_path": "path-to-locally-saved-tarfile",\n           "specfile_path": "none-or-path-to-locally-saved-specfile",\n           "signature_verified": "true-if-binary-pkg-was-already-verified"\n       }\n    '
    configured_mirrors = spack.mirror.MirrorCollection(binary=True).values()
    if not configured_mirrors:
        tty.die('Please add a spack mirror to allow download of pre-compiled packages.')
    tarball = tarball_path_name(spec, '.spack')
    specfile_prefix = tarball_name(spec, '.spec')
    try_first = [i['mirror_url'] for i in mirrors_for_spec] if mirrors_for_spec else []
    try_next = [i.fetch_url for i in configured_mirrors if i.fetch_url not in try_first]
    mirrors = try_first + try_next
    tried_to_verify_sigs = []
    for try_signed in (True, False):
        for mirror in mirrors:
            parsed = urllib.parse.urlparse(mirror)
            if parsed.scheme == 'oci':
                ref = spack.oci.image.ImageReference.from_string(mirror[len('oci://'):]).with_tag(spack.oci.image.default_tag(spec))
                try:
                    response = spack.oci.opener.urlopen(urllib.request.Request(url=ref.manifest_url(), headers={'Accept': 'application/vnd.oci.image.manifest.v1+json'}))
                except Exception:
                    continue
                try:
                    manifest = json.loads(response.read())
                    spec_digest = spack.oci.image.Digest.from_string(manifest['config']['digest'])
                    tarball_digest = spack.oci.image.Digest.from_string(manifest['layers'][-1]['digest'])
                except Exception:
                    continue
                with spack.oci.oci.make_stage(ref.blob_url(spec_digest), spec_digest, keep=True) as local_specfile_stage:
                    try:
                        local_specfile_stage.fetch()
                        local_specfile_stage.check()
                        try:
                            _get_valid_spec_file(local_specfile_stage.save_filename, CURRENT_BUILD_CACHE_LAYOUT_VERSION)
                        except InvalidMetadataFile as e:
                            tty.warn(f'Ignoring binary package for {spec.name}/{spec.dag_hash()[:7]} from {mirror} due to invalid metadata file: {e}')
                            local_specfile_stage.destroy()
                            continue
                    except Exception:
                        continue
                    local_specfile_stage.cache_local()
                with spack.oci.oci.make_stage(ref.blob_url(tarball_digest), tarball_digest, keep=True) as tarball_stage:
                    try:
                        tarball_stage.fetch()
                        tarball_stage.check()
                    except Exception:
                        continue
                    tarball_stage.cache_local()
                return {'tarball_stage': tarball_stage, 'specfile_stage': local_specfile_stage, 'signature_verified': False}
            else:
                ext = 'json.sig' if try_signed else 'json'
                specfile_path = url_util.join(mirror, BUILD_CACHE_RELATIVE_PATH, specfile_prefix)
                specfile_url = f'{specfile_path}.{ext}'
                spackfile_url = url_util.join(mirror, BUILD_CACHE_RELATIVE_PATH, tarball)
                local_specfile_stage = try_fetch(specfile_url)
                if local_specfile_stage:
                    local_specfile_path = local_specfile_stage.save_filename
                    signature_verified = False
                    try:
                        _get_valid_spec_file(local_specfile_path, CURRENT_BUILD_CACHE_LAYOUT_VERSION)
                    except InvalidMetadataFile as e:
                        tty.warn(f'Ignoring binary package for {spec.name}/{spec.dag_hash()[:7]} from {mirror} due to invalid metadata file: {e}')
                        local_specfile_stage.destroy()
                        continue
                    if try_signed and (not unsigned):
                        tried_to_verify_sigs.append(specfile_url)
                        signature_verified = try_verify(local_specfile_path)
                        if not signature_verified:
                            tty.warn('Failed to verify: {0}'.format(specfile_url))
                    if unsigned or signature_verified or (not try_signed):
                        tarball_stage = try_fetch(spackfile_url)
                        if tarball_stage:
                            return {'tarball_stage': tarball_stage, 'specfile_stage': local_specfile_stage, 'signature_verified': signature_verified}
                    local_specfile_stage.destroy()
    if tried_to_verify_sigs:
        raise NoVerifyException('Spack found new style signed binary packages, but was unable to verify any of them.  Please obtain and trust the correct public key.  If these are public spack binaries, please see the spack docs for locations where keys can be found.')
    return None

def dedupe_hardlinks_if_necessary(root, buildinfo):
    if False:
        while True:
            i = 10
    'Updates a buildinfo dict for old archives that did\n    not dedupe hardlinks. De-duping hardlinks is necessary\n    when relocating files in parallel and in-place. This\n    means we must preserve inodes when relocating.'
    if buildinfo.get('hardlinks_deduped', False):
        return
    visited = set()
    for key in ('relocate_textfiles', 'relocate_binaries'):
        if key not in buildinfo:
            continue
        new_list = []
        for rel_path in buildinfo[key]:
            stat_result = os.lstat(os.path.join(root, rel_path))
            identifier = (stat_result.st_dev, stat_result.st_ino)
            if stat_result.st_nlink > 1:
                if identifier in visited:
                    continue
                visited.add(identifier)
            new_list.append(rel_path)
        buildinfo[key] = new_list

def relocate_package(spec):
    if False:
        return 10
    '\n    Relocate the given package\n    '
    workdir = str(spec.prefix)
    buildinfo = read_buildinfo_file(workdir)
    new_layout_root = str(spack.store.STORE.layout.root)
    new_prefix = str(spec.prefix)
    new_rel_prefix = str(os.path.relpath(new_prefix, new_layout_root))
    new_spack_prefix = str(spack.paths.prefix)
    old_sbang_install_path = None
    if 'sbang_install_path' in buildinfo:
        old_sbang_install_path = str(buildinfo['sbang_install_path'])
    old_layout_root = str(buildinfo['buildpath'])
    old_spack_prefix = str(buildinfo.get('spackprefix'))
    old_rel_prefix = buildinfo.get('relative_prefix')
    old_prefix = os.path.join(old_layout_root, old_rel_prefix)
    rel = buildinfo.get('relative_rpaths', False)
    if 'hash_to_prefix' in buildinfo:
        hash_to_old_prefix = buildinfo['hash_to_prefix']
    elif 'prefix_to_hash' in buildinfo:
        hash_to_old_prefix = dict(((v, k) for (k, v) in buildinfo['prefix_to_hash'].items()))
    else:
        hash_to_old_prefix = dict()
    if old_rel_prefix != new_rel_prefix and (not hash_to_old_prefix):
        msg = 'Package tarball was created from an install '
        msg += 'prefix with a different directory layout and an older '
        msg += 'buildcache create implementation. It cannot be relocated.'
        raise NewLayoutException(msg)
    prefix_to_prefix_text = collections.OrderedDict()
    prefix_to_prefix_bin = collections.OrderedDict()
    if old_sbang_install_path:
        install_path = spack.hooks.sbang.sbang_install_path()
        prefix_to_prefix_text[old_sbang_install_path] = install_path
    for (dag_hash, new_dep_prefix) in hashes_to_prefixes(spec).items():
        if dag_hash in hash_to_old_prefix:
            old_dep_prefix = hash_to_old_prefix[dag_hash]
            prefix_to_prefix_bin[old_dep_prefix] = new_dep_prefix
            prefix_to_prefix_text[old_dep_prefix] = new_dep_prefix
    prefix_to_prefix_text[old_prefix] = new_prefix
    prefix_to_prefix_bin[old_prefix] = new_prefix
    prefix_to_prefix_text[old_layout_root] = new_layout_root
    prefix_to_prefix_bin[old_layout_root] = new_layout_root
    orig_sbang = '#!/bin/bash {0}/bin/sbang'.format(old_spack_prefix)
    new_sbang = spack.hooks.sbang.sbang_shebang_line()
    prefix_to_prefix_text[orig_sbang] = new_sbang
    tty.debug('Relocating package from', '%s to %s.' % (old_layout_root, new_layout_root))
    dedupe_hardlinks_if_necessary(workdir, buildinfo)

    def is_backup_file(file):
        if False:
            print('Hello World!')
        return file.endswith('~')
    text_names = list()
    for filename in buildinfo['relocate_textfiles']:
        text_name = os.path.join(workdir, filename)
        if not is_backup_file(text_name):
            text_names.append(text_name)
    if old_prefix != new_prefix:
        files_to_relocate = [os.path.join(workdir, filename) for filename in buildinfo.get('relocate_binaries')]
        platform = spack.platforms.by_name(spec.platform)
        if 'macho' in platform.binary_formats:
            relocate.relocate_macho_binaries(files_to_relocate, old_layout_root, new_layout_root, prefix_to_prefix_bin, rel, old_prefix, new_prefix)
        elif 'elf' in platform.binary_formats and (not rel):
            relocate.new_relocate_elf_binaries(files_to_relocate, prefix_to_prefix_bin)
        elif 'elf' in platform.binary_formats and rel:
            relocate.relocate_elf_binaries(files_to_relocate, old_layout_root, new_layout_root, prefix_to_prefix_bin, rel, old_prefix, new_prefix)
        links = [os.path.join(workdir, f) for f in buildinfo.get('relocate_links', [])]
        relocate.relocate_links(links, prefix_to_prefix_bin)
        relocate.relocate_text(text_names, prefix_to_prefix_text)
        changed_files = relocate.relocate_text_bin(files_to_relocate, prefix_to_prefix_bin)
        if 'macho' in platform.binary_formats and sys.platform == 'darwin':
            codesign = which('codesign')
            if not codesign:
                return
            for binary in changed_files:
                codesign('-fs-', binary)
    elif old_spack_prefix != new_spack_prefix:
        relocate.relocate_text(text_names, prefix_to_prefix_text)

def _extract_inner_tarball(spec, filename, extract_to, unsigned, remote_checksum):
    if False:
        return 10
    stagepath = os.path.dirname(filename)
    spackfile_name = tarball_name(spec, '.spack')
    spackfile_path = os.path.join(stagepath, spackfile_name)
    tarfile_name = tarball_name(spec, '.tar.gz')
    tarfile_path = os.path.join(extract_to, tarfile_name)
    json_name = tarball_name(spec, '.spec.json')
    json_path = os.path.join(extract_to, json_name)
    with closing(tarfile.open(spackfile_path, 'r')) as tar:
        tar.extractall(extract_to)
    if not os.path.exists(tarfile_path):
        tarfile_name = tarball_name(spec, '.tar.bz2')
        tarfile_path = os.path.join(extract_to, tarfile_name)
    if os.path.exists(json_path):
        specfile_path = json_path
    else:
        raise ValueError('Cannot find spec file for {0}.'.format(extract_to))
    if not unsigned:
        if os.path.exists('%s.asc' % specfile_path):
            suppress = config.get('config:suppress_gpg_warnings', False)
            try:
                spack.util.gpg.verify('%s.asc' % specfile_path, specfile_path, suppress)
            except Exception:
                raise NoVerifyException('Spack was unable to verify package signature, please obtain and trust the correct public key.')
        else:
            raise UnsignedPackageException('To install unsigned packages, use the --no-check-signature option.')
    local_checksum = spack.util.crypto.checksum(hashlib.sha256, tarfile_path)
    expected = remote_checksum['hash']
    if local_checksum != expected:
        (size, contents) = fsys.filesummary(tarfile_path)
        raise NoChecksumException(tarfile_path, size, contents, 'sha256', expected, local_checksum)
    return tarfile_path

def _tar_strip_component(tar: tarfile.TarFile, prefix: str):
    if False:
        print('Hello World!')
    'Strip the top-level directory `prefix` from the member names in a tarfile.'
    regex = re.compile(re.escape(prefix) + '/*')
    for m in tar.getmembers():
        result = regex.match(m.name)
        assert result is not None
        m.name = m.name[result.end():]
        if m.linkname:
            result = regex.match(m.linkname)
            if result:
                m.linkname = m.linkname[result.end():]

def extract_tarball(spec, download_result, unsigned=False, force=False, timer=timer.NULL_TIMER):
    if False:
        for i in range(10):
            print('nop')
    '\n    extract binary tarball for given package into install area\n    '
    timer.start('extract')
    if os.path.exists(spec.prefix):
        if force:
            shutil.rmtree(spec.prefix)
        else:
            raise NoOverwriteException(str(spec.prefix))
    fsys.mkdirp(spec.prefix, mode=get_package_dir_permissions(spec), group=get_package_group(spec), default_perms='parents')
    specfile_path = download_result['specfile_stage'].save_filename
    (spec_dict, layout_version) = _get_valid_spec_file(specfile_path, CURRENT_BUILD_CACHE_LAYOUT_VERSION)
    bchecksum = spec_dict['binary_cache_checksum']
    filename = download_result['tarball_stage'].save_filename
    signature_verified = download_result['signature_verified']
    tmpdir = None
    if layout_version == 0:
        tmpdir = tempfile.mkdtemp()
        try:
            tarfile_path = _extract_inner_tarball(spec, filename, tmpdir, unsigned, bchecksum)
        except Exception as e:
            _delete_staged_downloads(download_result)
            shutil.rmtree(tmpdir)
            raise e
    elif layout_version == 1:
        tarfile_path = filename
        if not unsigned and (not signature_verified):
            raise UnsignedPackageException('To install unsigned packages, use the --no-check-signature option.')
        local_checksum = spack.util.crypto.checksum(hashlib.sha256, tarfile_path)
        expected = bchecksum['hash']
        if local_checksum != expected:
            (size, contents) = fsys.filesummary(tarfile_path)
            _delete_staged_downloads(download_result)
            raise NoChecksumException(tarfile_path, size, contents, 'sha256', expected, local_checksum)
    try:
        with closing(tarfile.open(tarfile_path, 'r')) as tar:
            _tar_strip_component(tar, prefix=_ensure_common_prefix(tar))
            tar.extractall(path=spec.prefix)
    except Exception:
        shutil.rmtree(spec.prefix, ignore_errors=True)
        _delete_staged_downloads(download_result)
        raise
    os.remove(tarfile_path)
    os.remove(specfile_path)
    timer.stop('extract')
    timer.start('relocate')
    try:
        relocate_package(spec)
    except Exception as e:
        shutil.rmtree(spec.prefix, ignore_errors=True)
        raise e
    else:
        manifest_file = os.path.join(spec.prefix, spack.store.STORE.layout.metadata_dir, spack.store.STORE.layout.manifest_file_name)
        if not os.path.exists(manifest_file):
            spec_id = spec.format('{name}/{hash:7}')
            tty.warn('No manifest file in tarball for spec %s' % spec_id)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        if os.path.exists(filename):
            os.remove(filename)
        _delete_staged_downloads(download_result)
    timer.stop('relocate')

def _ensure_common_prefix(tar: tarfile.TarFile) -> str:
    if False:
        print('Hello World!')
    common_prefix = min((e.name for e in tar.getmembers() if e.isdir()), key=len, default=None)
    if common_prefix is None:
        raise ValueError('Tarball does not contain a common prefix')
    for member in tar.getmembers():
        if not member.name.startswith(common_prefix):
            raise ValueError(f'Tarball contains file {member.name} outside of prefix {common_prefix}')
    return common_prefix

def install_root_node(spec, unsigned=False, force=False, sha256=None):
    if False:
        print('Hello World!')
    "Install the root node of a concrete spec from a buildcache.\n\n    Checking the sha256 sum of a node before installation is usually needed only\n    for software installed during Spack's bootstrapping (since we might not have\n    a proper signature verification mechanism available).\n\n    Args:\n        spec: spec to be installed (note that only the root node will be installed)\n        unsigned (bool): if True allows installing unsigned binaries\n        force (bool): force installation if the spec is already present in the\n            local store\n        sha256 (str): optional sha256 of the binary package, to be checked\n            before installation\n    "
    if spec.external or spec.virtual:
        warnings.warn('Skipping external or virtual package {0}'.format(spec.format()))
        return
    elif spec.concrete and spec.installed and (not force):
        warnings.warn('Package for spec {0} already installed.'.format(spec.format()))
        return
    download_result = download_tarball(spec, unsigned)
    if not download_result:
        msg = 'download of binary cache file for spec "{0}" failed'
        raise RuntimeError(msg.format(spec.format()))
    if sha256:
        checker = spack.util.crypto.Checker(sha256)
        msg = 'cannot verify checksum for "{0}" [expected={1}]'
        tarball_path = download_result['tarball_stage'].save_filename
        msg = msg.format(tarball_path, sha256)
        if not checker.check(tarball_path):
            (size, contents) = fsys.filesummary(tarball_path)
            _delete_staged_downloads(download_result)
            raise NoChecksumException(tarball_path, size, contents, checker.hash_name, sha256, checker.sum)
        tty.debug('Verified SHA256 checksum of the build cache')
    with spack.util.path.filter_padding():
        tty.msg('Installing "{0}" from a buildcache'.format(spec.format()))
        extract_tarball(spec, download_result, unsigned, force)
        spack.hooks.post_install(spec, False)
        spack.store.STORE.db.add(spec, spack.store.STORE.layout)

def install_single_spec(spec, unsigned=False, force=False):
    if False:
        for i in range(10):
            print('nop')
    'Install a single concrete spec from a buildcache.\n\n    Args:\n        spec (spack.spec.Spec): spec to be installed\n        unsigned (bool): if True allows installing unsigned binaries\n        force (bool): force installation if the spec is already present in the\n            local store\n    '
    for node in spec.traverse(root=True, order='post', deptype=('link', 'run')):
        install_root_node(node, unsigned=unsigned, force=force)

def try_direct_fetch(spec, mirrors=None):
    if False:
        while True:
            i = 10
    '\n    Try to find the spec directly on the configured mirrors\n    '
    specfile_name = tarball_name(spec, '.spec.json')
    signed_specfile_name = tarball_name(spec, '.spec.json.sig')
    specfile_is_signed = False
    found_specs = []
    binary_mirrors = spack.mirror.MirrorCollection(mirrors=mirrors, binary=True).values()
    for mirror in binary_mirrors:
        buildcache_fetch_url_json = url_util.join(mirror.fetch_url, BUILD_CACHE_RELATIVE_PATH, specfile_name)
        buildcache_fetch_url_signed_json = url_util.join(mirror.fetch_url, BUILD_CACHE_RELATIVE_PATH, signed_specfile_name)
        try:
            (_, _, fs) = web_util.read_from_url(buildcache_fetch_url_signed_json)
            specfile_is_signed = True
        except (URLError, web_util.SpackWebError, HTTPError) as url_err:
            try:
                (_, _, fs) = web_util.read_from_url(buildcache_fetch_url_json)
            except (URLError, web_util.SpackWebError, HTTPError) as url_err_x:
                tty.debug('Did not find {0} on {1}'.format(specfile_name, buildcache_fetch_url_signed_json), url_err, level=2)
                tty.debug('Did not find {0} on {1}'.format(specfile_name, buildcache_fetch_url_json), url_err_x, level=2)
                continue
        specfile_contents = codecs.getreader('utf-8')(fs).read()
        if specfile_is_signed:
            specfile_json = Spec.extract_json_from_clearsig(specfile_contents)
            fetched_spec = Spec.from_dict(specfile_json)
        else:
            fetched_spec = Spec.from_json(specfile_contents)
        fetched_spec._mark_concrete()
        found_specs.append({'mirror_url': mirror.fetch_url, 'spec': fetched_spec})
    return found_specs

def get_mirrors_for_spec(spec=None, mirrors_to_check=None, index_only=False):
    if False:
        while True:
            i = 10
    '\n    Check if concrete spec exists on mirrors and return a list\n    indicating the mirrors on which it can be found\n\n    Args:\n        spec (spack.spec.Spec): The spec to look for in binary mirrors\n        mirrors_to_check (dict): Optionally override the configured mirrors\n            with the mirrors in this dictionary.\n        index_only (bool): When ``index_only`` is set to ``True``, only the local\n            cache is checked, no requests are made.\n\n    Return:\n        A list of objects, each containing a ``mirror_url`` and ``spec`` key\n            indicating all mirrors where the spec can be found.\n    '
    if spec is None:
        return []
    if not spack.mirror.MirrorCollection(mirrors=mirrors_to_check, binary=True):
        tty.debug('No Spack mirrors are currently configured')
        return {}
    results = BINARY_INDEX.find_built_spec(spec, mirrors_to_check=mirrors_to_check)
    if not results and (not index_only):
        results = try_direct_fetch(spec, mirrors=mirrors_to_check)
        if results:
            BINARY_INDEX.update_spec(spec, results)
    return results

def update_cache_and_get_specs():
    if False:
        while True:
            i = 10
    '\n    Get all concrete specs for build caches available on configured mirrors.\n    Initialization of internal cache data structures is done as lazily as\n    possible, so this method will also attempt to initialize and update the\n    local index cache (essentially a no-op if it has been done already and\n    nothing has changed on the configured mirrors.)\n\n    Throws:\n        FetchCacheError\n    '
    BINARY_INDEX.update()
    return BINARY_INDEX.get_all_built_specs()

def clear_spec_cache():
    if False:
        print('Hello World!')
    BINARY_INDEX.clear()

def get_keys(install=False, trust=False, force=False, mirrors=None):
    if False:
        for i in range(10):
            print('nop')
    'Get pgp public keys available on mirror with suffix .pub'
    mirror_collection = mirrors or spack.mirror.MirrorCollection(binary=True)
    if not mirror_collection:
        tty.die('Please add a spack mirror to allow ' + 'download of build caches.')
    for mirror in mirror_collection.values():
        fetch_url = mirror.fetch_url
        keys_url = url_util.join(fetch_url, BUILD_CACHE_RELATIVE_PATH, BUILD_CACHE_KEYS_RELATIVE_PATH)
        keys_index = url_util.join(keys_url, 'index.json')
        tty.debug('Finding public keys in {0}'.format(url_util.format(fetch_url)))
        try:
            (_, _, json_file) = web_util.read_from_url(keys_index)
            json_index = sjson.load(codecs.getreader('utf-8')(json_file))
        except (URLError, web_util.SpackWebError) as url_err:
            if web_util.url_exists(keys_index):
                err_msg = ['Unable to find public keys in {0},', ' caught exception attempting to read from {1}.']
                tty.error(''.join(err_msg).format(url_util.format(fetch_url), url_util.format(keys_index)))
                tty.debug(url_err)
            continue
        for (fingerprint, key_attributes) in json_index['keys'].items():
            link = os.path.join(keys_url, fingerprint + '.pub')
            with Stage(link, name='build_cache', keep=True) as stage:
                if os.path.exists(stage.save_filename) and force:
                    os.remove(stage.save_filename)
                if not os.path.exists(stage.save_filename):
                    try:
                        stage.fetch()
                    except spack.error.FetchError:
                        continue
            tty.debug('Found key {0}'.format(fingerprint))
            if install:
                if trust:
                    spack.util.gpg.trust(stage.save_filename)
                    tty.debug('Added this key to trusted keys.')
                else:
                    tty.debug('Will not add this key to trusted keys.Use -t to install all downloaded keys')

def push_keys(*mirrors, **kwargs):
    if False:
        print('Hello World!')
    '\n    Upload pgp public keys to the given mirrors\n    '
    keys = kwargs.get('keys')
    regenerate_index = kwargs.get('regenerate_index', False)
    tmpdir = kwargs.get('tmpdir')
    remove_tmpdir = False
    keys = spack.util.gpg.public_keys(*(keys or []))
    try:
        for mirror in mirrors:
            push_url = getattr(mirror, 'push_url', mirror)
            keys_url = url_util.join(push_url, BUILD_CACHE_RELATIVE_PATH, BUILD_CACHE_KEYS_RELATIVE_PATH)
            keys_local = url_util.local_file_path(keys_url)
            verb = 'Writing' if keys_local else 'Uploading'
            tty.debug('{0} public keys to {1}'.format(verb, url_util.format(push_url)))
            if keys_local:
                prefix = keys_local
                mkdirp(keys_local)
            else:
                if tmpdir is None:
                    tmpdir = tempfile.mkdtemp()
                    remove_tmpdir = True
                prefix = tmpdir
            for fingerprint in keys:
                tty.debug('    ' + fingerprint)
                filename = fingerprint + '.pub'
                export_target = os.path.join(prefix, filename)
                spack.util.gpg.export_keys(export_target, [fingerprint])
                if not keys_local:
                    spack.util.web.push_to_url(export_target, url_util.join(keys_url, filename), keep_original=False)
            if regenerate_index:
                if keys_local:
                    generate_key_index(keys_url)
                else:
                    generate_key_index(keys_url, tmpdir)
    finally:
        if remove_tmpdir:
            shutil.rmtree(tmpdir)

def needs_rebuild(spec, mirror_url):
    if False:
        print('Hello World!')
    if not spec.concrete:
        raise ValueError('spec must be concrete to check against mirror')
    pkg_name = spec.name
    pkg_version = spec.version
    pkg_hash = spec.dag_hash()
    tty.debug('Checking {0}-{1}, dag_hash = {2}'.format(pkg_name, pkg_version, pkg_hash))
    tty.debug(spec.tree())
    cache_prefix = build_cache_prefix(mirror_url)
    specfile_name = tarball_name(spec, '.spec.json')
    specfile_path = os.path.join(cache_prefix, specfile_name)
    return not web_util.url_exists(specfile_path)

def check_specs_against_mirrors(mirrors, specs, output_file=None):
    if False:
        i = 10
        return i + 15
    "Check all the given specs against buildcaches on the given mirrors and\n    determine if any of the specs need to be rebuilt.  Specs need to be rebuilt\n    when their hash doesn't exist in the mirror.\n\n    Arguments:\n        mirrors (dict): Mirrors to check against\n        specs (typing.Iterable): Specs to check against mirrors\n        output_file (str): Path to output file to be written.  If provided,\n            mirrors with missing or out-of-date specs will be formatted as a\n            JSON object and written to this file.\n\n    Returns: 1 if any spec was out-of-date on any mirror, 0 otherwise.\n\n    "
    rebuilds = {}
    for mirror in spack.mirror.MirrorCollection(mirrors, binary=True).values():
        tty.debug('Checking for built specs at {0}'.format(mirror.fetch_url))
        rebuild_list = []
        for spec in specs:
            if needs_rebuild(spec, mirror.fetch_url):
                rebuild_list.append({'short_spec': spec.short_spec, 'hash': spec.dag_hash()})
        if rebuild_list:
            rebuilds[mirror.fetch_url] = {'mirrorName': mirror.name, 'mirrorUrl': mirror.fetch_url, 'rebuildSpecs': rebuild_list}
    if output_file:
        with open(output_file, 'w') as outf:
            outf.write(json.dumps(rebuilds))
    return 1 if rebuilds else 0

def _download_buildcache_entry(mirror_root, descriptions):
    if False:
        print('Hello World!')
    for description in descriptions:
        path = description['path']
        mkdirp(path)
        fail_if_missing = description['required']
        for url in description['url']:
            description_url = os.path.join(mirror_root, url)
            stage = Stage(description_url, name='build_cache', path=path, keep=True)
            try:
                stage.fetch()
                break
            except spack.error.FetchError as e:
                tty.debug(e)
        else:
            if fail_if_missing:
                tty.error('Failed to download required url {0}'.format(description_url))
                return False
    return True

def download_buildcache_entry(file_descriptions, mirror_url=None):
    if False:
        i = 10
        return i + 15
    if not mirror_url and (not spack.mirror.MirrorCollection(binary=True)):
        tty.die('Please provide or add a spack mirror to allow ' + 'download of buildcache entries.')
    if mirror_url:
        mirror_root = os.path.join(mirror_url, BUILD_CACHE_RELATIVE_PATH)
        return _download_buildcache_entry(mirror_root, file_descriptions)
    for mirror in spack.mirror.MirrorCollection(binary=True).values():
        mirror_root = os.path.join(mirror.fetch_url, BUILD_CACHE_RELATIVE_PATH)
        if _download_buildcache_entry(mirror_root, file_descriptions):
            return True
        else:
            continue
    return False

def download_single_spec(concrete_spec, destination, mirror_url=None):
    if False:
        while True:
            i = 10
    'Download the buildcache files for a single concrete spec.\n\n    Args:\n        concrete_spec: concrete spec to be downloaded\n        destination (str): path where to put the downloaded buildcache\n        mirror_url (str): url of the mirror from which to download\n    '
    tarfile_name = tarball_name(concrete_spec, '.spack')
    tarball_dir_name = tarball_directory_name(concrete_spec)
    tarball_path_name = os.path.join(tarball_dir_name, tarfile_name)
    local_tarball_path = os.path.join(destination, tarball_dir_name)
    files_to_fetch = [{'url': [tarball_path_name], 'path': local_tarball_path, 'required': True}, {'url': [tarball_name(concrete_spec, '.spec.json.sig'), tarball_name(concrete_spec, '.spec.json')], 'path': destination, 'required': True}]
    return download_buildcache_entry(files_to_fetch, mirror_url)

class BinaryCacheQuery:
    """Callable object to query if a spec is in a binary cache"""

    def __init__(self, all_architectures):
        if False:
            print('Hello World!')
        '\n        Args:\n            all_architectures (bool): if True consider all the spec for querying,\n                otherwise restrict to the current default architecture\n        '
        self.all_architectures = all_architectures
        specs = update_cache_and_get_specs()
        if not self.all_architectures:
            arch = spack.spec.Spec.default_arch()
            specs = [s for s in specs if s.satisfies(arch)]
        self.possible_specs = specs

    def __call__(self, spec: Spec, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            spec: The spec being searched for\n        '
        return [s for s in self.possible_specs if s.satisfies(spec)]

class FetchIndexError(Exception):

    def __str__(self):
        if False:
            while True:
                i = 10
        if len(self.args) == 1:
            return str(self.args[0])
        else:
            return '{}, due to: {}'.format(self.args[0], self.args[1])

class BuildcacheIndexError(spack.error.SpackError):
    """Raised when a buildcache cannot be read for any reason"""
FetchIndexResult = collections.namedtuple('FetchIndexResult', 'etag hash data fresh')

class DefaultIndexFetcher:
    """Fetcher for index.json, using separate index.json.hash as cache invalidation strategy"""

    def __init__(self, url, local_hash, urlopen=web_util.urlopen):
        if False:
            for i in range(10):
                print('nop')
        self.url = url
        self.local_hash = local_hash
        self.urlopen = urlopen
        self.headers = {'User-Agent': web_util.SPACK_USER_AGENT}

    def get_remote_hash(self):
        if False:
            for i in range(10):
                print('nop')
        url_index_hash = url_util.join(self.url, BUILD_CACHE_RELATIVE_PATH, 'index.json.hash')
        try:
            response = self.urlopen(urllib.request.Request(url_index_hash, headers=self.headers))
        except urllib.error.URLError:
            return None
        remote_hash = response.read(64)
        if not re.match(b'[a-f\\d]{64}$', remote_hash):
            return None
        return remote_hash.decode('utf-8')

    def conditional_fetch(self) -> FetchIndexResult:
        if False:
            return 10
        if self.local_hash and self.local_hash == self.get_remote_hash():
            return FetchIndexResult(etag=None, hash=None, data=None, fresh=True)
        url_index = url_util.join(self.url, BUILD_CACHE_RELATIVE_PATH, 'index.json')
        try:
            response = self.urlopen(urllib.request.Request(url_index, headers=self.headers))
        except urllib.error.URLError as e:
            raise FetchIndexError('Could not fetch index from {}'.format(url_index), e) from e
        try:
            result = codecs.getreader('utf-8')(response).read()
        except ValueError as e:
            raise FetchIndexError('Remote index {} is invalid'.format(url_index), e) from e
        computed_hash = compute_hash(result)
        if urllib.parse.urlparse(self.url).scheme not in ('http', 'https'):
            etag = None
        else:
            etag = web_util.parse_etag(response.headers.get('Etag', None) or response.headers.get('etag', None))
        return FetchIndexResult(etag=etag, hash=computed_hash, data=result, fresh=False)

class EtagIndexFetcher:
    """Fetcher for index.json, using ETags headers as cache invalidation strategy"""

    def __init__(self, url, etag, urlopen=web_util.urlopen):
        if False:
            while True:
                i = 10
        self.url = url
        self.etag = etag
        self.urlopen = urlopen

    def conditional_fetch(self) -> FetchIndexResult:
        if False:
            print('Hello World!')
        url = url_util.join(self.url, BUILD_CACHE_RELATIVE_PATH, 'index.json')
        headers = {'User-Agent': web_util.SPACK_USER_AGENT, 'If-None-Match': '"{}"'.format(self.etag)}
        try:
            response = self.urlopen(urllib.request.Request(url, headers=headers))
        except urllib.error.HTTPError as e:
            if e.getcode() == 304:
                return FetchIndexResult(etag=None, hash=None, data=None, fresh=True)
            raise FetchIndexError('Could not fetch index {}'.format(url), e) from e
        except urllib.error.URLError as e:
            raise FetchIndexError('Could not fetch index {}'.format(url), e) from e
        try:
            result = codecs.getreader('utf-8')(response).read()
        except ValueError as e:
            raise FetchIndexError('Remote index {} is invalid'.format(url), e) from e
        headers = response.headers
        etag_header_value = headers.get('Etag', None) or headers.get('etag', None)
        return FetchIndexResult(etag=web_util.parse_etag(etag_header_value), hash=compute_hash(result), data=result, fresh=False)

class OCIIndexFetcher:

    def __init__(self, url: str, local_hash, urlopen=None) -> None:
        if False:
            i = 10
            return i + 15
        self.local_hash = local_hash
        assert url.startswith('oci://')
        self.ref = spack.oci.image.ImageReference.from_string(url[6:])
        self.urlopen = urlopen or spack.oci.opener.urlopen

    def conditional_fetch(self) -> FetchIndexResult:
        if False:
            i = 10
            return i + 15
        'Download an index from an OCI registry type mirror.'
        url_manifest = self.ref.with_tag(spack.oci.image.default_index_tag).manifest_url()
        try:
            response = self.urlopen(urllib.request.Request(url=url_manifest, headers={'Accept': 'application/vnd.oci.image.manifest.v1+json'}))
        except urllib.error.URLError as e:
            raise FetchIndexError('Could not fetch manifest from {}'.format(url_manifest), e) from e
        try:
            manifest = json.loads(response.read())
        except Exception as e:
            raise FetchIndexError('Remote index {} is invalid'.format(url_manifest), e) from e
        try:
            index_digest = spack.oci.image.Digest.from_string(manifest['layers'][0]['digest'])
        except Exception as e:
            raise FetchIndexError('Remote index {} is invalid'.format(url_manifest), e) from e
        if index_digest.digest == self.local_hash:
            return FetchIndexResult(etag=None, hash=None, data=None, fresh=True)
        response = self.urlopen(urllib.request.Request(url=self.ref.blob_url(index_digest), headers={'Accept': 'application/vnd.oci.image.layer.v1.tar+gzip'}))
        result = codecs.getreader('utf-8')(response).read()
        if compute_hash(result) != index_digest.digest:
            raise FetchIndexError(f'Remote index {url_manifest} is invalid')
        return FetchIndexResult(etag=None, hash=index_digest.digest, data=result, fresh=False)