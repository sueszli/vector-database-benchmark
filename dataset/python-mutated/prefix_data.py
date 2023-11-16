"""Tools for managing the packages installed within an environment."""
from __future__ import annotations
import json
import os
import re
from logging import getLogger
from os.path import basename, lexists
from pathlib import Path
from ..auxlib.exceptions import ValidationError
from ..base.constants import CONDA_ENV_VARS_UNSET_VAR, CONDA_PACKAGE_EXTENSIONS, PREFIX_MAGIC_FILE, PREFIX_STATE_FILE
from ..base.context import context
from ..common.constants import NULL
from ..common.io import time_recorder
from ..common.path import get_python_site_packages_short_path, win_path_ok
from ..common.pkg_formats.python import get_site_packages_anchor_files
from ..common.serialize import json_load
from ..deprecations import deprecated
from ..exceptions import BasicClobberError, CondaDependencyError, CorruptedEnvironmentError, maybe_raise
from ..gateways.disk.create import write_as_json_to_file
from ..gateways.disk.delete import rm_rf
from ..gateways.disk.read import read_python_record
from ..gateways.disk.test import file_path_is_writable
from ..models.match_spec import MatchSpec
from ..models.prefix_graph import PrefixGraph
from ..models.records import PackageRecord, PrefixRecord
log = getLogger(__name__)

class PrefixDataType(type):
    """Basic caching of PrefixData instance objects."""

    def __call__(cls, prefix_path: str | os.PathLike | Path, pip_interop_enabled: bool | None=None):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(prefix_path, PrefixData):
            return prefix_path
        elif (prefix_path := Path(prefix_path)) in PrefixData._cache_:
            return PrefixData._cache_[prefix_path]
        else:
            prefix_data_instance = super().__call__(prefix_path, pip_interop_enabled)
            PrefixData._cache_[prefix_path] = prefix_data_instance
            return prefix_data_instance

class PrefixData(metaclass=PrefixDataType):
    _cache_: dict[Path, PrefixData] = {}

    def __init__(self, prefix_path: Path, pip_interop_enabled: bool | None=None):
        if False:
            return 10
        self.prefix_path = prefix_path
        self.__prefix_records = None
        self.__is_writable = NULL
        self._pip_interop_enabled = pip_interop_enabled if pip_interop_enabled is not None else context.pip_interop_enabled

    @time_recorder(module_name=__name__)
    def load(self):
        if False:
            for i in range(10):
                print('nop')
        self.__prefix_records = {}
        _conda_meta_dir = self.prefix_path / 'conda-meta'
        if lexists(_conda_meta_dir):
            conda_meta_json_paths = (p for p in (entry.path for entry in os.scandir(_conda_meta_dir)) if p[-5:] == '.json')
            for meta_file in conda_meta_json_paths:
                self._load_single_record(meta_file)
        if self._pip_interop_enabled:
            self._load_site_packages()

    def reload(self):
        if False:
            i = 10
            return i + 15
        self.load()
        return self

    def _get_json_fn(self, prefix_record):
        if False:
            for i in range(10):
                print('nop')
        fn = prefix_record.fn
        known_ext = False
        for ext in CONDA_PACKAGE_EXTENSIONS + ('.dist-info',):
            if fn.endswith(ext):
                fn = fn.replace(ext, '')
                known_ext = True
        if not known_ext:
            raise ValueError('Attempted to make prefix record for unknown package type: %s' % fn)
        return fn + '.json'

    def insert(self, prefix_record):
        if False:
            i = 10
            return i + 15
        assert prefix_record.name not in self._prefix_records, 'Prefix record insertion error: a record with name %s already exists in the prefix. This is a bug in conda. Please report it at https://github.com/conda/conda/issues' % prefix_record.name
        prefix_record_json_path = self.prefix_path / 'conda-meta' / self._get_json_fn(prefix_record)
        if lexists(prefix_record_json_path):
            maybe_raise(BasicClobberError(source_path=None, target_path=prefix_record_json_path, context=context), context)
            rm_rf(prefix_record_json_path)
        write_as_json_to_file(prefix_record_json_path, prefix_record)
        self._prefix_records[prefix_record.name] = prefix_record

    def remove(self, package_name):
        if False:
            return 10
        assert package_name in self._prefix_records
        prefix_record = self._prefix_records[package_name]
        prefix_record_json_path = self.prefix_path / 'conda-meta' / self._get_json_fn(prefix_record)
        if self.is_writable:
            rm_rf(prefix_record_json_path)
        del self._prefix_records[package_name]

    def get(self, package_name, default=NULL):
        if False:
            print('Hello World!')
        try:
            return self._prefix_records[package_name]
        except KeyError:
            if default is not NULL:
                return default
            else:
                raise

    def iter_records(self):
        if False:
            return 10
        return iter(self._prefix_records.values())

    def iter_records_sorted(self):
        if False:
            return 10
        prefix_graph = PrefixGraph(self.iter_records())
        return iter(prefix_graph.graph)

    def all_subdir_urls(self):
        if False:
            while True:
                i = 10
        subdir_urls = set()
        for prefix_record in self.iter_records():
            subdir_url = prefix_record.channel.subdir_url
            if subdir_url and subdir_url not in subdir_urls:
                log.debug('adding subdir url %s for %s', subdir_url, prefix_record)
                subdir_urls.add(subdir_url)
        return subdir_urls

    def query(self, package_ref_or_match_spec):
        if False:
            while True:
                i = 10
        param = package_ref_or_match_spec
        if isinstance(param, str):
            param = MatchSpec(param)
        if isinstance(param, MatchSpec):
            return (prefix_rec for prefix_rec in self.iter_records() if param.match(prefix_rec))
        else:
            assert isinstance(param, PackageRecord)
            return (prefix_rec for prefix_rec in self.iter_records() if prefix_rec == param)

    @property
    def _prefix_records(self):
        if False:
            print('Hello World!')
        return self.__prefix_records or self.load() or self.__prefix_records

    def _load_single_record(self, prefix_record_json_path):
        if False:
            for i in range(10):
                print('nop')
        log.debug('loading prefix record %s', prefix_record_json_path)
        with open(prefix_record_json_path) as fh:
            try:
                json_data = json_load(fh.read())
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise CorruptedEnvironmentError(self.prefix_path, prefix_record_json_path)
            prefix_record = PrefixRecord(**json_data)
            try:
                (n, v, b) = basename(prefix_record_json_path)[:-5].rsplit('-', 2)
                if (n, v, b) != (prefix_record.name, prefix_record.version, prefix_record.build):
                    raise ValueError()
            except ValueError:
                log.warn('Ignoring malformed prefix record at: %s', prefix_record_json_path)
                return
            self.__prefix_records[prefix_record.name] = prefix_record

    @property
    def is_writable(self):
        if False:
            while True:
                i = 10
        if self.__is_writable == NULL:
            test_path = self.prefix_path / PREFIX_MAGIC_FILE
            if not test_path.is_file():
                is_writable = None
            else:
                is_writable = file_path_is_writable(test_path)
            self.__is_writable = is_writable
        return self.__is_writable

    @deprecated('24.3', '24.9')
    def _has_python(self):
        if False:
            print('Hello World!')
        return 'python' in self._prefix_records

    @property
    def _python_pkg_record(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the prefix record for the package python.'
        return next((prefix_record for prefix_record in self.__prefix_records.values() if prefix_record.name == 'python'), None)

    def _load_site_packages(self):
        if False:
            while True:
                i = 10
        '\n        Load non-conda-installed python packages in the site-packages of the prefix.\n\n        Python packages not handled by conda are installed via other means,\n        like using pip or using python setup.py develop for local development.\n\n        Packages found that are not handled by conda are converted into a\n        prefix record and handled in memory.\n\n        Packages clobbering conda packages (i.e. the conda-meta record) are\n        removed from the in memory representation.\n        '
        python_pkg_record = self._python_pkg_record
        if not python_pkg_record:
            return {}
        site_packages_dir = get_python_site_packages_short_path(python_pkg_record.version)
        site_packages_path = self.prefix_path / win_path_ok(site_packages_dir)
        if not site_packages_path.is_dir():
            return {}
        prefix_graph = PrefixGraph(self.iter_records())
        python_records = prefix_graph.all_descendants(python_pkg_record)
        conda_python_packages = get_conda_anchor_files_and_records(site_packages_dir, python_records)
        sp_anchor_files = get_site_packages_anchor_files(site_packages_path, site_packages_dir)
        conda_anchor_files = set(conda_python_packages)
        clobbered_conda_anchor_files = conda_anchor_files - sp_anchor_files
        non_conda_anchor_files = sp_anchor_files - conda_anchor_files
        for conda_anchor_file in clobbered_conda_anchor_files:
            prefix_rec = self._prefix_records.pop(conda_python_packages[conda_anchor_file].name)
            try:
                extracted_package_dir = basename(prefix_rec.extracted_package_dir)
            except AttributeError:
                extracted_package_dir = '-'.join((prefix_rec.name, prefix_rec.version, prefix_rec.build))
            prefix_rec_json_path = self.prefix_path / 'conda-meta' / f'{extracted_package_dir}.json'
            try:
                rm_rf(prefix_rec_json_path)
            except OSError:
                log.debug("stale information, but couldn't remove: %s", prefix_rec_json_path)
            else:
                log.debug('removed due to stale information: %s', prefix_rec_json_path)
        new_packages = {}
        for af in non_conda_anchor_files:
            try:
                python_record = read_python_record(self.prefix_path, af, python_pkg_record.version)
            except OSError as e:
                log.info("Python record ignored for anchor path '%s'\n  due to %s", af, e)
                continue
            except ValidationError:
                import sys
                (exc_type, exc_value, exc_traceback) = sys.exc_info()
                import traceback
                tb = traceback.format_exception(exc_type, exc_value, exc_traceback)
                log.warn('Problem reading non-conda package record at %s. Please verify that you still need this, and if so, that this is still installed correctly. Reinstalling this package may help.', af)
                log.debug('ValidationError: \n%s\n', '\n'.join(tb))
                continue
            if not python_record:
                continue
            self.__prefix_records[python_record.name] = python_record
            new_packages[python_record.name] = python_record
        return new_packages

    def _get_environment_state_file(self):
        if False:
            return 10
        env_vars_file = self.prefix_path / PREFIX_STATE_FILE
        if lexists(env_vars_file):
            with open(env_vars_file) as f:
                prefix_state = json.loads(f.read())
        else:
            prefix_state = {}
        return prefix_state

    def _write_environment_state_file(self, state):
        if False:
            i = 10
            return i + 15
        env_vars_file = self.prefix_path / PREFIX_STATE_FILE
        env_vars_file.write_text(json.dumps(state, ensure_ascii=False, default=lambda x: x.__dict__))

    def get_environment_env_vars(self):
        if False:
            i = 10
            return i + 15
        prefix_state = self._get_environment_state_file()
        env_vars_all = dict(prefix_state.get('env_vars', {}))
        env_vars = {k: v for (k, v) in env_vars_all.items() if v != CONDA_ENV_VARS_UNSET_VAR}
        return env_vars

    def set_environment_env_vars(self, env_vars):
        if False:
            for i in range(10):
                print('nop')
        env_state_file = self._get_environment_state_file()
        current_env_vars = env_state_file.get('env_vars')
        if current_env_vars:
            current_env_vars.update(env_vars)
        else:
            env_state_file['env_vars'] = env_vars
        self._write_environment_state_file(env_state_file)
        return env_state_file.get('env_vars')

    def unset_environment_env_vars(self, env_vars):
        if False:
            i = 10
            return i + 15
        env_state_file = self._get_environment_state_file()
        current_env_vars = env_state_file.get('env_vars')
        if current_env_vars:
            for env_var in env_vars:
                if env_var in current_env_vars.keys():
                    current_env_vars[env_var] = CONDA_ENV_VARS_UNSET_VAR
            self._write_environment_state_file(env_state_file)
        return env_state_file.get('env_vars')

def get_conda_anchor_files_and_records(site_packages_short_path, python_records):
    if False:
        return 10
    'Return the anchor files for the conda records of python packages.'
    anchor_file_endings = ('.egg-info/PKG-INFO', '.dist-info/RECORD', '.egg-info')
    conda_python_packages = {}
    matcher = re.compile('^{}/[^/]+(?:{})$'.format(re.escape(site_packages_short_path), '|'.join((re.escape(fn) for fn in anchor_file_endings)))).match
    for prefix_record in python_records:
        anchor_paths = tuple((fpath for fpath in prefix_record.files if matcher(fpath)))
        if len(anchor_paths) > 1:
            anchor_path = sorted(anchor_paths, key=len)[0]
            log.info('Package %s has multiple python anchor files.\n  Using %s', prefix_record.record_id(), anchor_path)
            conda_python_packages[anchor_path] = prefix_record
        elif anchor_paths:
            conda_python_packages[anchor_paths[0]] = prefix_record
    return conda_python_packages

def get_python_version_for_prefix(prefix):
    if False:
        return 10
    py_record_iter = (rcrd for rcrd in PrefixData(prefix).iter_records() if rcrd.name == 'python')
    record = next(py_record_iter, None)
    if record is None:
        return None
    next_record = next(py_record_iter, None)
    if next_record is not None:
        raise CondaDependencyError('multiple python records found in prefix %s' % prefix)
    elif record.version[3].isdigit():
        return record.version[:4]
    else:
        return record.version[:3]

def delete_prefix_from_linked_data(path: str | os.PathLike | Path) -> bool:
    if False:
        i = 10
        return i + 15
    'Here, path may be a complete prefix or a dist inside a prefix'
    path = Path(path)
    for prefix in sorted(PrefixData._cache_, reverse=True):
        try:
            path.relative_to(prefix)
            del PrefixData._cache_[prefix]
            return True
        except ValueError:
            continue
    return False