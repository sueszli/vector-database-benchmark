"""Concrete collection candidate management helper module."""
from __future__ import annotations
import json
import os
import tarfile
import subprocess
import typing as t
from contextlib import contextmanager
from hashlib import sha256
from urllib.error import URLError
from urllib.parse import urldefrag
from shutil import rmtree
from tempfile import mkdtemp
if t.TYPE_CHECKING:
    from ansible.galaxy.dependency_resolution.dataclasses import Candidate, Collection, Requirement
    from ansible.galaxy.token import GalaxyToken
from ansible.errors import AnsibleError
from ansible.galaxy import get_collections_galaxy_meta_info
from ansible.galaxy.api import should_retry_error
from ansible.galaxy.dependency_resolution.dataclasses import _GALAXY_YAML
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.api import retry_with_delays_and_condition
from ansible.module_utils.api import generate_jittered_backoff
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.yaml import yaml_load
from ansible.module_utils.urls import open_url
from ansible.utils.display import Display
from ansible.utils.sentinel import Sentinel
import yaml
display = Display()
MANIFEST_FILENAME = 'MANIFEST.json'

class ConcreteArtifactsManager:
    """Manager for on-disk collection artifacts.

    It is responsible for:
        * downloading remote collections from Galaxy-compatible servers and
          direct links to tarballs or SCM repositories
        * keeping track of local ones
        * keeping track of Galaxy API tokens for downloads from Galaxy'ish
          as well as the artifact hashes
        * keeping track of Galaxy API signatures for downloads from Galaxy'ish
        * caching all of above
        * retrieving the metadata out of the downloaded artifacts
    """

    def __init__(self, b_working_directory, validate_certs=True, keyring=None, timeout=60, required_signature_count=None, ignore_signature_errors=None):
        if False:
            return 10
        'Initialize ConcreteArtifactsManager caches and costraints.'
        self._validate_certs = validate_certs
        self._artifact_cache = {}
        self._galaxy_artifact_cache = {}
        self._artifact_meta_cache = {}
        self._galaxy_collection_cache = {}
        self._galaxy_collection_origin_cache = {}
        self._b_working_directory = b_working_directory
        self._supplemental_signature_cache = {}
        self._keyring = keyring
        self.timeout = timeout
        self._required_signature_count = required_signature_count
        self._ignore_signature_errors = ignore_signature_errors
        self._require_build_metadata = True

    @property
    def keyring(self):
        if False:
            print('Hello World!')
        return self._keyring

    @property
    def required_successful_signature_count(self):
        if False:
            print('Hello World!')
        return self._required_signature_count

    @property
    def ignore_signature_errors(self):
        if False:
            print('Hello World!')
        if self._ignore_signature_errors is None:
            return []
        return self._ignore_signature_errors

    @property
    def require_build_metadata(self):
        if False:
            return 10
        return self._require_build_metadata

    @require_build_metadata.setter
    def require_build_metadata(self, value):
        if False:
            return 10
        self._require_build_metadata = value

    def get_galaxy_artifact_source_info(self, collection):
        if False:
            i = 10
            return i + 15
        server = collection.src.api_server
        try:
            download_url = self._galaxy_collection_cache[collection][0]
            (signatures_url, signatures) = self._galaxy_collection_origin_cache[collection]
        except KeyError as key_err:
            raise RuntimeError('The is no known source for {coll!s}'.format(coll=collection)) from key_err
        return {'format_version': '1.0.0', 'namespace': collection.namespace, 'name': collection.name, 'version': collection.ver, 'server': server, 'version_url': signatures_url, 'download_url': download_url, 'signatures': signatures}

    def get_galaxy_artifact_path(self, collection):
        if False:
            print('Hello World!')
        "Given a Galaxy-stored collection, return a cached path.\n\n        If it's not yet on disk, this method downloads the artifact first.\n        "
        try:
            return self._galaxy_artifact_cache[collection]
        except KeyError:
            pass
        try:
            (url, sha256_hash, token) = self._galaxy_collection_cache[collection]
        except KeyError as key_err:
            raise RuntimeError('There is no known source for {coll!s}'.format(coll=collection)) from key_err
        display.vvvv("Fetching a collection tarball for '{collection!s}' from Ansible Galaxy".format(collection=collection))
        try:
            b_artifact_path = _download_file(url, self._b_working_directory, expected_hash=sha256_hash, validate_certs=self._validate_certs, token=token)
        except URLError as err:
            raise AnsibleError("Failed to download collection tar from '{coll_src!s}': {download_err!s}".format(coll_src=to_native(collection.src), download_err=to_native(err))) from err
        except Exception as err:
            raise AnsibleError("Failed to download collection tar from '{coll_src!s}' due to the following unforeseen error: {download_err!s}".format(coll_src=to_native(collection.src), download_err=to_native(err))) from err
        else:
            display.vvv("Collection '{coll!s}' obtained from server {server!s} {url!s}".format(coll=collection, server=collection.src or 'Galaxy', url=collection.src.api_server if collection.src is not None else ''))
        self._galaxy_artifact_cache[collection] = b_artifact_path
        return b_artifact_path

    def get_artifact_path(self, collection):
        if False:
            print('Hello World!')
        "Given a concrete collection pointer, return a cached path.\n\n        If it's not yet on disk, this method downloads the artifact first.\n        "
        try:
            return self._artifact_cache[collection.src]
        except KeyError:
            pass
        if collection.is_url:
            display.vvvv("Collection requirement '{collection!s}' is a URL to a tar artifact".format(collection=collection.fqcn))
            try:
                b_artifact_path = _download_file(collection.src, self._b_working_directory, expected_hash=None, validate_certs=self._validate_certs, timeout=self.timeout)
            except Exception as err:
                raise AnsibleError("Failed to download collection tar from '{coll_src!s}': {download_err!s}".format(coll_src=to_native(collection.src), download_err=to_native(err))) from err
        elif collection.is_scm:
            b_artifact_path = _extract_collection_from_git(collection.src, collection.ver, self._b_working_directory)
        elif collection.is_file or collection.is_dir or collection.is_subdirs:
            b_artifact_path = to_bytes(collection.src)
        else:
            raise RuntimeError('The artifact is of an unexpected type {art_type!s}'.format(art_type=collection.type))
        self._artifact_cache[collection.src] = b_artifact_path
        return b_artifact_path

    def get_artifact_path_from_unknown(self, collection):
        if False:
            for i in range(10):
                print('nop')
        if collection.is_concrete_artifact:
            return self.get_artifact_path(collection)
        return self.get_galaxy_artifact_path(collection)

    def _get_direct_collection_namespace(self, collection):
        if False:
            i = 10
            return i + 15
        return self.get_direct_collection_meta(collection)['namespace']

    def _get_direct_collection_name(self, collection):
        if False:
            for i in range(10):
                print('nop')
        return self.get_direct_collection_meta(collection)['name']

    def get_direct_collection_fqcn(self, collection):
        if False:
            return 10
        'Extract FQCN from the given on-disk collection artifact.\n\n        If the collection is virtual, ``None`` is returned instead\n        of a string.\n        '
        if collection.is_virtual:
            return None
        return '.'.join((self._get_direct_collection_namespace(collection), self._get_direct_collection_name(collection)))

    def get_direct_collection_version(self, collection):
        if False:
            return 10
        'Extract version from the given on-disk collection artifact.'
        return self.get_direct_collection_meta(collection)['version']

    def get_direct_collection_dependencies(self, collection):
        if False:
            while True:
                i = 10
        'Extract deps from the given on-disk collection artifact.'
        collection_dependencies = self.get_direct_collection_meta(collection)['dependencies']
        if collection_dependencies is None:
            collection_dependencies = {}
        return collection_dependencies

    def get_direct_collection_meta(self, collection):
        if False:
            i = 10
            return i + 15
        'Extract meta from the given on-disk collection artifact.'
        try:
            return self._artifact_meta_cache[collection.src]
        except KeyError:
            b_artifact_path = self.get_artifact_path(collection)
        if collection.is_url or collection.is_file:
            collection_meta = _get_meta_from_tar(b_artifact_path)
        elif collection.is_dir:
            try:
                collection_meta = _get_meta_from_dir(b_artifact_path, self.require_build_metadata)
            except LookupError as lookup_err:
                raise AnsibleError('Failed to find the collection dir deps: {err!s}'.format(err=to_native(lookup_err))) from lookup_err
        elif collection.is_scm:
            collection_meta = {'name': None, 'namespace': None, 'dependencies': {to_native(b_artifact_path): '*'}, 'version': '*'}
        elif collection.is_subdirs:
            collection_meta = {'name': None, 'namespace': None, 'dependencies': dict.fromkeys(map(to_native, collection.namespace_collection_paths), '*'), 'version': '*'}
        else:
            raise RuntimeError
        self._artifact_meta_cache[collection.src] = collection_meta
        return collection_meta

    def save_collection_source(self, collection, url, sha256_hash, token, signatures_url, signatures):
        if False:
            i = 10
            return i + 15
        'Store collection URL, SHA256 hash and Galaxy API token.\n\n        This is a hook that is supposed to be called before attempting to\n        download Galaxy-based collections with ``get_galaxy_artifact_path()``.\n        '
        self._galaxy_collection_cache[collection] = (url, sha256_hash, token)
        self._galaxy_collection_origin_cache[collection] = (signatures_url, signatures)

    @classmethod
    @contextmanager
    def under_tmpdir(cls, temp_dir_base, validate_certs=True, keyring=None, required_signature_count=None, ignore_signature_errors=None, require_build_metadata=True):
        if False:
            while True:
                i = 10
        'Custom ConcreteArtifactsManager constructor with temp dir.\n\n        This method returns a context manager that allocates and cleans\n        up a temporary directory for caching the collection artifacts\n        during the dependency resolution process.\n        '
        temp_path = mkdtemp(dir=to_bytes(temp_dir_base, errors='surrogate_or_strict'))
        b_temp_path = to_bytes(temp_path, errors='surrogate_or_strict')
        try:
            yield cls(b_temp_path, validate_certs, keyring=keyring, required_signature_count=required_signature_count, ignore_signature_errors=ignore_signature_errors)
        finally:
            rmtree(b_temp_path)

def parse_scm(collection, version):
    if False:
        for i in range(10):
            print('nop')
    'Extract name, version, path and subdir out of the SCM pointer.'
    if ',' in collection:
        (collection, version) = collection.split(',', 1)
    elif version == '*' or not version:
        version = 'HEAD'
    if collection.startswith('git+'):
        path = collection[4:]
    else:
        path = collection
    (path, fragment) = urldefrag(path)
    fragment = fragment.strip(os.path.sep)
    if path.endswith(os.path.sep + '.git'):
        name = path.split(os.path.sep)[-2]
    elif '://' not in path and '@' not in path:
        name = path
    else:
        name = path.split('/')[-1]
        if name.endswith('.git'):
            name = name[:-4]
    return (name, version, path, fragment)

def _extract_collection_from_git(repo_url, coll_ver, b_path):
    if False:
        print('Hello World!')
    (name, version, git_url, fragment) = parse_scm(repo_url, coll_ver)
    b_checkout_path = mkdtemp(dir=b_path, prefix=to_bytes(name, errors='surrogate_or_strict'))
    try:
        git_executable = get_bin_path('git')
    except ValueError as err:
        raise AnsibleError('Could not find git executable to extract the collection from the Git repository `{repo_url!s}`.'.format(repo_url=to_native(git_url))) from err
    if version == 'HEAD':
        git_clone_cmd = (git_executable, 'clone', '--depth=1', git_url, to_text(b_checkout_path))
    else:
        git_clone_cmd = (git_executable, 'clone', git_url, to_text(b_checkout_path))
    try:
        subprocess.check_call(git_clone_cmd)
    except subprocess.CalledProcessError as proc_err:
        raise AnsibleError('Failed to clone a Git repository from `{repo_url!s}`.'.format(repo_url=to_native(git_url))) from proc_err
    git_switch_cmd = (git_executable, 'checkout', to_text(version))
    try:
        subprocess.check_call(git_switch_cmd, cwd=b_checkout_path)
    except subprocess.CalledProcessError as proc_err:
        raise AnsibleError('Failed to switch a cloned Git repo `{repo_url!s}` to the requested revision `{commitish!s}`.'.format(commitish=to_native(version), repo_url=to_native(git_url))) from proc_err
    return os.path.join(b_checkout_path, to_bytes(fragment)) if fragment else b_checkout_path

@retry_with_delays_and_condition(backoff_iterator=generate_jittered_backoff(retries=6, delay_base=2, delay_threshold=40), should_retry_error=should_retry_error)
def _download_file(url, b_path, expected_hash, validate_certs, token=None, timeout=60):
    if False:
        return 10
    b_tarball_name = to_bytes(url.rsplit('/', 1)[1], errors='surrogate_or_strict')
    b_file_name = b_tarball_name[:-len('.tar.gz')]
    b_tarball_dir = mkdtemp(dir=b_path, prefix=b'-'.join((b_file_name, b'')))
    b_file_path = os.path.join(b_tarball_dir, b_tarball_name)
    display.display('Downloading %s to %s' % (url, to_text(b_tarball_dir)))
    try:
        resp = open_url(to_native(url, errors='surrogate_or_strict'), validate_certs=validate_certs, headers=None if token is None else token.headers(), unredirected_headers=['Authorization'], http_agent=user_agent(), timeout=timeout)
    except Exception as err:
        raise AnsibleError(to_native(err), orig_exc=err)
    with open(b_file_path, 'wb') as download_file:
        actual_hash = _consume_file(resp, write_to=download_file)
    if expected_hash:
        display.vvvv('Validating downloaded file hash {actual_hash!s} with expected hash {expected_hash!s}'.format(actual_hash=actual_hash, expected_hash=expected_hash))
        if expected_hash != actual_hash:
            raise AnsibleError('Mismatch artifact hash with downloaded file')
    return b_file_path

def _consume_file(read_from, write_to=None):
    if False:
        i = 10
        return i + 15
    bufsize = 65536
    sha256_digest = sha256()
    data = read_from.read(bufsize)
    while data:
        if write_to is not None:
            write_to.write(data)
            write_to.flush()
        sha256_digest.update(data)
        data = read_from.read(bufsize)
    return sha256_digest.hexdigest()

def _normalize_galaxy_yml_manifest(galaxy_yml, b_galaxy_yml_path, require_build_metadata=True):
    if False:
        i = 10
        return i + 15
    galaxy_yml_schema = get_collections_galaxy_meta_info()
    mandatory_keys = set()
    string_keys = set()
    list_keys = set()
    dict_keys = set()
    sentinel_keys = set()
    for info in galaxy_yml_schema:
        if info.get('required', False):
            mandatory_keys.add(info['key'])
        key_list_type = {'str': string_keys, 'list': list_keys, 'dict': dict_keys, 'sentinel': sentinel_keys}[info.get('type', 'str')]
        key_list_type.add(info['key'])
    all_keys = frozenset(mandatory_keys | string_keys | list_keys | dict_keys | sentinel_keys)
    set_keys = set(galaxy_yml.keys())
    missing_keys = mandatory_keys.difference(set_keys)
    if missing_keys:
        msg = "The collection galaxy.yml at '%s' is missing the following mandatory keys: %s" % (to_native(b_galaxy_yml_path), ', '.join(sorted(missing_keys)))
        if require_build_metadata:
            raise AnsibleError(msg)
        display.warning(msg)
        raise ValueError(msg)
    extra_keys = set_keys.difference(all_keys)
    if len(extra_keys) > 0:
        display.warning("Found unknown keys in collection galaxy.yml at '%s': %s" % (to_text(b_galaxy_yml_path), ', '.join(extra_keys)))
    for optional_string in string_keys:
        if optional_string not in galaxy_yml:
            galaxy_yml[optional_string] = None
    for optional_list in list_keys:
        list_val = galaxy_yml.get(optional_list, None)
        if list_val is None:
            galaxy_yml[optional_list] = []
        elif not isinstance(list_val, list):
            galaxy_yml[optional_list] = [list_val]
    for optional_dict in dict_keys:
        if optional_dict not in galaxy_yml:
            galaxy_yml[optional_dict] = {}
    for optional_sentinel in sentinel_keys:
        if optional_sentinel not in galaxy_yml:
            galaxy_yml[optional_sentinel] = Sentinel
    if not galaxy_yml.get('version'):
        galaxy_yml['version'] = '*'
    return galaxy_yml

def _get_meta_from_dir(b_path, require_build_metadata=True):
    if False:
        while True:
            i = 10
    try:
        return _get_meta_from_installed_dir(b_path)
    except LookupError:
        return _get_meta_from_src_dir(b_path, require_build_metadata)

def _get_meta_from_src_dir(b_path, require_build_metadata=True):
    if False:
        i = 10
        return i + 15
    galaxy_yml = os.path.join(b_path, _GALAXY_YAML)
    if not os.path.isfile(galaxy_yml):
        raise LookupError("The collection galaxy.yml path '{path!s}' does not exist.".format(path=to_native(galaxy_yml)))
    with open(galaxy_yml, 'rb') as manifest_file_obj:
        try:
            manifest = yaml_load(manifest_file_obj)
        except yaml.error.YAMLError as yaml_err:
            raise AnsibleError("Failed to parse the galaxy.yml at '{path!s}' with the following error:\n{err_txt!s}".format(path=to_native(galaxy_yml), err_txt=to_native(yaml_err))) from yaml_err
    if not isinstance(manifest, dict):
        if require_build_metadata:
            raise AnsibleError(f"The collection galaxy.yml at '{to_native(galaxy_yml)}' is incorrectly formatted.")
        display.warning(f"The collection galaxy.yml at '{to_native(galaxy_yml)}' is incorrectly formatted.")
        raise ValueError(f"The collection galaxy.yml at '{to_native(galaxy_yml)}' is incorrectly formatted.")
    return _normalize_galaxy_yml_manifest(manifest, galaxy_yml, require_build_metadata)

def _get_json_from_installed_dir(b_path, filename):
    if False:
        for i in range(10):
            print('nop')
    b_json_filepath = os.path.join(b_path, to_bytes(filename, errors='surrogate_or_strict'))
    try:
        with open(b_json_filepath, 'rb') as manifest_fd:
            b_json_text = manifest_fd.read()
    except (IOError, OSError):
        raise LookupError("The collection {manifest!s} path '{path!s}' does not exist.".format(manifest=filename, path=to_native(b_json_filepath)))
    manifest_txt = to_text(b_json_text, errors='surrogate_or_strict')
    try:
        manifest = json.loads(manifest_txt)
    except ValueError:
        raise AnsibleError('Collection tar file member {member!s} does not contain a valid json string.'.format(member=filename))
    return manifest

def _get_meta_from_installed_dir(b_path):
    if False:
        return 10
    manifest = _get_json_from_installed_dir(b_path, MANIFEST_FILENAME)
    collection_info = manifest['collection_info']
    version = collection_info.get('version')
    if not version:
        raise AnsibleError(u'Collection metadata file `{manifest_filename!s}` at `{meta_file!s}` is expected to have a valid SemVer version value but got {version!s}'.format(manifest_filename=MANIFEST_FILENAME, meta_file=to_text(b_path), version=to_text(repr(version))))
    return collection_info

def _get_meta_from_tar(b_path):
    if False:
        return 10
    if not os.path.exists(b_path):
        raise AnsibleError(f"Unable to find collection artifact file at '{to_native(b_path)}'.")
    if not tarfile.is_tarfile(b_path):
        raise AnsibleError("Collection artifact at '{path!s}' is not a valid tar file.".format(path=to_native(b_path)))
    with tarfile.open(b_path, mode='r') as collection_tar:
        try:
            member = collection_tar.getmember(MANIFEST_FILENAME)
        except KeyError:
            raise AnsibleError("Collection at '{path!s}' does not contain the required file {manifest_file!s}.".format(path=to_native(b_path), manifest_file=MANIFEST_FILENAME))
        with _tarfile_extract(collection_tar, member) as (_member, member_obj):
            if member_obj is None:
                raise AnsibleError('Collection tar file does not contain member {member!s}'.format(member=MANIFEST_FILENAME))
            text_content = to_text(member_obj.read(), errors='surrogate_or_strict')
            try:
                manifest = json.loads(text_content)
            except ValueError:
                raise AnsibleError('Collection tar file member {member!s} does not contain a valid json string.'.format(member=MANIFEST_FILENAME))
            return manifest['collection_info']

@contextmanager
def _tarfile_extract(tar, member):
    if False:
        i = 10
        return i + 15
    tar_obj = tar.extractfile(member)
    try:
        yield (member, tar_obj)
    finally:
        if tar_obj is not None:
            tar_obj.close()