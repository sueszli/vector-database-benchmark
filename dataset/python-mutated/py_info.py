"""
The PythonInfo contains information about a concrete instance of a Python interpreter.

Note: this file is also used to query target interpreters, so can only use standard library methods
"""
from __future__ import annotations
import json
import logging
import os
import platform
import re
import sys
import sysconfig
import warnings
from collections import OrderedDict, namedtuple
from string import digits
VersionInfo = namedtuple('VersionInfo', ['major', 'minor', 'micro', 'releaselevel', 'serial'])

def _get_path_extensions():
    if False:
        return 10
    return list(OrderedDict.fromkeys(['', *os.environ.get('PATHEXT', '').lower().split(os.pathsep)]))
EXTENSIONS = _get_path_extensions()
_CONF_VAR_RE = re.compile('\\{\\w+\\}')

class PythonInfo:
    """Contains information for a Python interpreter."""

    def __init__(self) -> None:
        if False:
            while True:
                i = 10

        def abs_path(v):
            if False:
                print('Hello World!')
            return None if v is None else os.path.abspath(v)
        self.platform = sys.platform
        self.implementation = platform.python_implementation()
        if self.implementation == 'PyPy':
            self.pypy_version_info = tuple(sys.pypy_version_info)
        self.version_info = VersionInfo(*sys.version_info)
        self.architecture = 64 if sys.maxsize > 2 ** 32 else 32
        self.version_nodot = sysconfig.get_config_var('py_version_nodot')
        self.version = sys.version
        self.os = os.name
        self.prefix = abs_path(getattr(sys, 'prefix', None))
        self.base_prefix = abs_path(getattr(sys, 'base_prefix', None))
        self.real_prefix = abs_path(getattr(sys, 'real_prefix', None))
        self.base_exec_prefix = abs_path(getattr(sys, 'base_exec_prefix', None))
        self.exec_prefix = abs_path(getattr(sys, 'exec_prefix', None))
        self.executable = abs_path(sys.executable)
        self.original_executable = abs_path(self.executable)
        self.system_executable = self._fast_get_system_executable()
        try:
            __import__('venv')
            has = True
        except ImportError:
            has = False
        self.has_venv = has
        self.path = sys.path
        self.file_system_encoding = sys.getfilesystemencoding()
        self.stdout_encoding = getattr(sys.stdout, 'encoding', None)
        scheme_names = sysconfig.get_scheme_names()
        if 'venv' in scheme_names:
            self.sysconfig_scheme = 'venv'
            self.sysconfig_paths = {i: sysconfig.get_path(i, expand=False, scheme=self.sysconfig_scheme) for i in sysconfig.get_path_names()}
            self.distutils_install = {}
        elif sys.version_info[:2] == (3, 10) and 'deb_system' in scheme_names:
            self.sysconfig_scheme = 'posix_prefix'
            self.sysconfig_paths = {i: sysconfig.get_path(i, expand=False, scheme=self.sysconfig_scheme) for i in sysconfig.get_path_names()}
            self.distutils_install = {}
        else:
            self.sysconfig_scheme = None
            self.sysconfig_paths = {i: sysconfig.get_path(i, expand=False) for i in sysconfig.get_path_names()}
            self.distutils_install = self._distutils_install().copy()
        makefile = getattr(sysconfig, 'get_makefile_filename', getattr(sysconfig, '_get_makefile_filename', None))
        self.sysconfig = {k: v for (k, v) in [('makefile_filename', makefile())] if k is not None}
        config_var_keys = set()
        for element in self.sysconfig_paths.values():
            for k in _CONF_VAR_RE.findall(element):
                config_var_keys.add(k[1:-1])
        config_var_keys.add('PYTHONFRAMEWORK')
        self.sysconfig_vars = {i: sysconfig.get_config_var(i or '') for i in config_var_keys}
        confs = {k: self.system_prefix if v is not None and v.startswith(self.prefix) else v for (k, v) in self.sysconfig_vars.items()}
        self.system_stdlib = self.sysconfig_path('stdlib', confs)
        self.system_stdlib_platform = self.sysconfig_path('platstdlib', confs)
        self.max_size = getattr(sys, 'maxsize', getattr(sys, 'maxint', None))
        self._creators = None

    def _fast_get_system_executable(self):
        if False:
            while True:
                i = 10
        'Try to get the system executable by just looking at properties.'
        if self.real_prefix or (self.base_prefix is not None and self.base_prefix != self.prefix):
            if self.real_prefix is None:
                base_executable = getattr(sys, '_base_executable', None)
                if base_executable is not None:
                    if sys.executable != base_executable:
                        if os.path.exists(base_executable):
                            return base_executable
                        (major, minor) = (self.version_info.major, self.version_info.minor)
                        if self.os == 'posix' and (major, minor) >= (3, 11):
                            base_dir = os.path.dirname(base_executable)
                            for base_executable in [os.path.join(base_dir, exe) for exe in (f'python{major}', f'python{major}.{minor}')]:
                                if os.path.exists(base_executable):
                                    return base_executable
            return None
        return self.original_executable

    def install_path(self, key):
        if False:
            for i in range(10):
                print('nop')
        result = self.distutils_install.get(key)
        if result is None:
            prefixes = (self.prefix, self.exec_prefix, self.base_prefix, self.base_exec_prefix)
            config_var = {k: '' if v in prefixes else v for (k, v) in self.sysconfig_vars.items()}
            result = self.sysconfig_path(key, config_var=config_var).lstrip(os.sep)
        return result

    @staticmethod
    def _distutils_install():
        if False:
            i = 10
            return i + 15
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                from distutils import dist
                from distutils.command.install import SCHEME_KEYS
            except ImportError:
                return {}
        d = dist.Distribution({'script_args': '--no-user-cfg'})
        if hasattr(sys, '_framework'):
            sys._framework = None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            i = d.get_command_obj('install', create=True)
        i.prefix = os.sep
        i.finalize_options()
        return {key: getattr(i, f'install_{key}')[1:].lstrip(os.sep) for key in SCHEME_KEYS}

    @property
    def version_str(self):
        if False:
            i = 10
            return i + 15
        return '.'.join((str(i) for i in self.version_info[0:3]))

    @property
    def version_release_str(self):
        if False:
            print('Hello World!')
        return '.'.join((str(i) for i in self.version_info[0:2]))

    @property
    def python_name(self):
        if False:
            i = 10
            return i + 15
        version_info = self.version_info
        return f'python{version_info.major}.{version_info.minor}'

    @property
    def is_old_virtualenv(self):
        if False:
            i = 10
            return i + 15
        return self.real_prefix is not None

    @property
    def is_venv(self):
        if False:
            while True:
                i = 10
        return self.base_prefix is not None

    def sysconfig_path(self, key, config_var=None, sep=os.sep):
        if False:
            for i in range(10):
                print('nop')
        pattern = self.sysconfig_paths[key]
        if config_var is None:
            config_var = self.sysconfig_vars
        else:
            base = self.sysconfig_vars.copy()
            base.update(config_var)
            config_var = base
        return pattern.format(**config_var).replace('/', sep)

    def creators(self, refresh=False):
        if False:
            for i in range(10):
                print('nop')
        if self._creators is None or refresh is True:
            from virtualenv.run.plugin.creators import CreatorSelector
            self._creators = CreatorSelector.for_interpreter(self)
        return self._creators

    @property
    def system_include(self):
        if False:
            print('Hello World!')
        path = self.sysconfig_path('include', {k: self.system_prefix if v is not None and v.startswith(self.prefix) else v for (k, v) in self.sysconfig_vars.items()})
        if not os.path.exists(path):
            fallback = os.path.join(self.prefix, os.path.dirname(self.install_path('headers')))
            if os.path.exists(fallback):
                path = fallback
        return path

    @property
    def system_prefix(self):
        if False:
            i = 10
            return i + 15
        return self.real_prefix or self.base_prefix or self.prefix

    @property
    def system_exec_prefix(self):
        if False:
            print('Hello World!')
        return self.real_prefix or self.base_exec_prefix or self.exec_prefix

    def __unicode__(self):
        if False:
            i = 10
            return i + 15
        return repr(self)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return '{}({!r})'.format(self.__class__.__name__, {k: v for (k, v) in self.__dict__.items() if not k.startswith('_')})

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return '{}({})'.format(self.__class__.__name__, ', '.join((f'{k}={v}' for (k, v) in (('spec', self.spec), ('system' if self.system_executable is not None and self.system_executable != self.executable else None, self.system_executable), ('original' if self.original_executable not in {self.system_executable, self.executable} else None, self.original_executable), ('exe', self.executable), ('platform', self.platform), ('version', repr(self.version)), ('encoding_fs_io', f'{self.file_system_encoding}-{self.stdout_encoding}')) if k is not None)))

    @property
    def spec(self):
        if False:
            while True:
                i = 10
        return '{}{}-{}'.format(self.implementation, '.'.join((str(i) for i in self.version_info)), self.architecture)

    @classmethod
    def clear_cache(cls, app_data):
        if False:
            while True:
                i = 10
        from virtualenv.discovery.cached_py_info import clear
        clear(app_data)
        cls._cache_exe_discovery.clear()

    def satisfies(self, spec, impl_must_match):
        if False:
            i = 10
            return i + 15
        'Check if a given specification can be satisfied by the this python interpreter instance.'
        if spec.path:
            if self.executable == os.path.abspath(spec.path):
                return True
            if not spec.is_abs:
                basename = os.path.basename(self.original_executable)
                spec_path = spec.path
                if sys.platform == 'win32':
                    (basename, suffix) = os.path.splitext(basename)
                    if spec_path.endswith(suffix):
                        spec_path = spec_path[:-len(suffix)]
                if basename != spec_path:
                    return False
        if impl_must_match and spec.implementation is not None and (spec.implementation.lower() != self.implementation.lower()):
            return False
        if spec.architecture is not None and spec.architecture != self.architecture:
            return False
        for (our, req) in zip(self.version_info[0:3], (spec.major, spec.minor, spec.micro)):
            if req is not None and our is not None and (our != req):
                return False
        return True
    _current_system = None
    _current = None

    @classmethod
    def current(cls, app_data=None):
        if False:
            return 10
        '\n        This locates the current host interpreter information. This might be different than what we run into in case\n        the host python has been upgraded from underneath us.\n        '
        if cls._current is None:
            cls._current = cls.from_exe(sys.executable, app_data, raise_on_error=True, resolve_to_host=False)
        return cls._current

    @classmethod
    def current_system(cls, app_data=None):
        if False:
            i = 10
            return i + 15
        '\n        This locates the current host interpreter information. This might be different than what we run into in case\n        the host python has been upgraded from underneath us.\n        '
        if cls._current_system is None:
            cls._current_system = cls.from_exe(sys.executable, app_data, raise_on_error=True, resolve_to_host=True)
        return cls._current_system

    def _to_json(self):
        if False:
            for i in range(10):
                print('nop')
        return json.dumps(self._to_dict(), indent=2)

    def _to_dict(self):
        if False:
            while True:
                i = 10
        data = {var: getattr(self, var) if var not in ('_creators',) else None for var in vars(self)}
        data['version_info'] = data['version_info']._asdict()
        return data

    @classmethod
    def from_exe(cls, exe, app_data=None, raise_on_error=True, ignore_cache=False, resolve_to_host=True, env=None):
        if False:
            i = 10
            return i + 15
        'Given a path to an executable get the python information.'
        from virtualenv.discovery.cached_py_info import from_exe
        env = os.environ if env is None else env
        proposed = from_exe(cls, app_data, exe, env=env, raise_on_error=raise_on_error, ignore_cache=ignore_cache)
        if isinstance(proposed, PythonInfo) and resolve_to_host:
            try:
                proposed = proposed._resolve_to_system(app_data, proposed)
            except Exception as exception:
                if raise_on_error:
                    raise
                logging.info('ignore %s due cannot resolve system due to %r', proposed.original_executable, exception)
                proposed = None
        return proposed

    @classmethod
    def _from_json(cls, payload):
        if False:
            return 10
        raw = json.loads(payload)
        return cls._from_dict(raw.copy())

    @classmethod
    def _from_dict(cls, data):
        if False:
            while True:
                i = 10
        data['version_info'] = VersionInfo(**data['version_info'])
        result = cls()
        result.__dict__ = data.copy()
        return result

    @classmethod
    def _resolve_to_system(cls, app_data, target):
        if False:
            return 10
        start_executable = target.executable
        prefixes = OrderedDict()
        while target.system_executable is None:
            prefix = target.real_prefix or target.base_prefix or target.prefix
            if prefix in prefixes:
                if len(prefixes) == 1:
                    logging.info('%r links back to itself via prefixes', target)
                    target.system_executable = target.executable
                    break
                for (at, (p, t)) in enumerate(prefixes.items(), start=1):
                    logging.error('%d: prefix=%s, info=%r', at, p, t)
                logging.error('%d: prefix=%s, info=%r', len(prefixes) + 1, prefix, target)
                msg = 'prefixes are causing a circle {}'.format('|'.join(prefixes.keys()))
                raise RuntimeError(msg)
            prefixes[prefix] = target
            target = target.discover_exe(app_data, prefix=prefix, exact=False)
        if target.executable != target.system_executable:
            target = cls.from_exe(target.system_executable, app_data)
        target.executable = start_executable
        return target
    _cache_exe_discovery = {}

    def discover_exe(self, app_data, prefix, exact=True, env=None):
        if False:
            i = 10
            return i + 15
        key = (prefix, exact)
        if key in self._cache_exe_discovery and prefix:
            logging.debug('discover exe from cache %s - exact %s: %r', prefix, exact, self._cache_exe_discovery[key])
            return self._cache_exe_discovery[key]
        logging.debug('discover exe for %s in %s', self, prefix)
        possible_names = self._find_possible_exe_names()
        possible_folders = self._find_possible_folders(prefix)
        discovered = []
        env = os.environ if env is None else env
        for folder in possible_folders:
            for name in possible_names:
                info = self._check_exe(app_data, folder, name, exact, discovered, env)
                if info is not None:
                    self._cache_exe_discovery[key] = info
                    return info
        if exact is False and discovered:
            info = self._select_most_likely(discovered, self)
            folders = os.pathsep.join(possible_folders)
            self._cache_exe_discovery[key] = info
            logging.debug('no exact match found, chosen most similar of %s within base folders %s', info, folders)
            return info
        msg = 'failed to detect {} in {}'.format('|'.join(possible_names), os.pathsep.join(possible_folders))
        raise RuntimeError(msg)

    def _check_exe(self, app_data, folder, name, exact, discovered, env):
        if False:
            for i in range(10):
                print('nop')
        exe_path = os.path.join(folder, name)
        if not os.path.exists(exe_path):
            return None
        info = self.from_exe(exe_path, app_data, resolve_to_host=False, raise_on_error=False, env=env)
        if info is None:
            return None
        for item in ['implementation', 'architecture', 'version_info']:
            found = getattr(info, item)
            searched = getattr(self, item)
            if found != searched:
                if item == 'version_info':
                    (found, searched) = ('.'.join((str(i) for i in found)), '.'.join((str(i) for i in searched)))
                executable = info.executable
                logging.debug('refused interpreter %s because %s differs %s != %s', executable, item, found, searched)
                if exact is False:
                    discovered.append(info)
                break
        else:
            return info
        return None

    @staticmethod
    def _select_most_likely(discovered, target):
        if False:
            while True:
                i = 10

        def sort_by(info):
            if False:
                return 10
            matches = [info.implementation == target.implementation, info.version_info.major == target.version_info.major, info.version_info.minor == target.version_info.minor, info.architecture == target.architecture, info.version_info.micro == target.version_info.micro, info.version_info.releaselevel == target.version_info.releaselevel, info.version_info.serial == target.version_info.serial]
            return sum((1 << pos if match else 0 for (pos, match) in enumerate(reversed(matches))))
        sorted_discovered = sorted(discovered, key=sort_by, reverse=True)
        return sorted_discovered[0]

    def _find_possible_folders(self, inside_folder):
        if False:
            return 10
        candidate_folder = OrderedDict()
        executables = OrderedDict()
        executables[os.path.realpath(self.executable)] = None
        executables[self.executable] = None
        executables[os.path.realpath(self.original_executable)] = None
        executables[self.original_executable] = None
        for exe in executables:
            base = os.path.dirname(exe)
            if base.startswith(self.prefix):
                relative = base[len(self.prefix):]
                candidate_folder[f'{inside_folder}{relative}'] = None
        candidate_folder[inside_folder] = None
        return [i for i in candidate_folder if os.path.exists(i)]

    def _find_possible_exe_names(self):
        if False:
            for i in range(10):
                print('nop')
        name_candidate = OrderedDict()
        for name in self._possible_base():
            for at in (3, 2, 1, 0):
                version = '.'.join((str(i) for i in self.version_info[:at]))
                for arch in [f'-{self.architecture}', '']:
                    for ext in EXTENSIONS:
                        candidate = f'{name}{version}{arch}{ext}'
                        name_candidate[candidate] = None
        return list(name_candidate.keys())

    def _possible_base(self):
        if False:
            return 10
        possible_base = OrderedDict()
        basename = os.path.splitext(os.path.basename(self.executable))[0].rstrip(digits)
        possible_base[basename] = None
        possible_base[self.implementation] = None
        if 'python' in possible_base:
            del possible_base['python']
        possible_base['python'] = None
        for base in possible_base:
            lower = base.lower()
            yield lower
            from virtualenv.info import fs_is_case_sensitive
            if fs_is_case_sensitive():
                if base != lower:
                    yield base
                upper = base.upper()
                if upper != base:
                    yield upper
if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) >= 1:
        start_cookie = argv[0]
        argv = argv[1:]
    else:
        start_cookie = ''
    if len(argv) >= 1:
        end_cookie = argv[0]
        argv = argv[1:]
    else:
        end_cookie = ''
    sys.argv = sys.argv[:1] + argv
    info = PythonInfo()._to_json()
    sys.stdout.write(''.join((start_cookie[::-1], info, end_cookie[::-1])))