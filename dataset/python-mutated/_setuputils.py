"""
gevent build utilities.
"""
from __future__ import print_function, absolute_import, division
import re
import os
import os.path
import sys
import sysconfig
from distutils import sysconfig as dist_sysconfig
from subprocess import check_call
from glob import glob
from setuptools import Extension as _Extension
from setuptools.command.build_ext import build_ext
THIS_DIR = os.path.dirname(__file__)
PYPY = hasattr(sys, 'pypy_version_info')
WIN = sys.platform.startswith('win')
PY311 = sys.version_info[:2] >= (3, 11)
PY312 = sys.version_info[:2] >= (3, 12)
RUNNING_ON_TRAVIS = os.environ.get('TRAVIS')
RUNNING_ON_APPVEYOR = os.environ.get('APPVEYOR')
RUNNING_ON_GITHUB_ACTIONS = os.environ.get('GITHUB_ACTIONS')
RUNNING_ON_CI = RUNNING_ON_TRAVIS or RUNNING_ON_APPVEYOR or RUNNING_ON_GITHUB_ACTIONS
RUNNING_FROM_CHECKOUT = os.path.isdir(os.path.join(THIS_DIR, '.git'))
LIBRARIES = []
DEFINE_MACROS = []
if WIN:
    LIBRARIES += ['ws2_32']
    DEFINE_MACROS += [('FD_SETSIZE', '1024'), ('_WIN32', '1')]

def quoted_abspath(*segments):
    if False:
        print('Hello World!')
    return '"' + os.path.abspath(os.path.join(*segments)) + '"'

def read(*names):
    if False:
        print('Hello World!')
    'Read a file path relative to this file.'
    with open(os.path.join(THIS_DIR, *names)) as f:
        return f.read()

def read_version(name='src/gevent/__init__.py'):
    if False:
        for i in range(10):
            print('nop')
    contents = read(name)
    version = re.search("__version__\\s*=\\s*'(.*)'", contents, re.M).group(1)
    assert version, 'could not read version'
    return version

def dep_abspath(depname, *extra):
    if False:
        print('Hello World!')
    return os.path.abspath(os.path.join('deps', depname, *extra))

def quoted_dep_abspath(depname):
    if False:
        for i in range(10):
            print('nop')
    return quoted_abspath(dep_abspath(depname))

def glob_many(*globs):
    if False:
        i = 10
        return i + 15
    '\n    Return a list of all the glob patterns expanded.\n    '
    result = []
    for pattern in globs:
        result.extend(glob(pattern))
    return sorted(result)

def bool_from_environ(key):
    if False:
        i = 10
        return i + 15
    value = os.environ.get(key)
    if not value:
        return
    value = value.lower().strip()
    if value in ('1', 'true', 'on', 'yes'):
        return True
    if value in ('0', 'false', 'off', 'no'):
        return False
    raise ValueError('Environment variable %r has invalid value %r. Please set it to 1, 0 or an empty string' % (key, value))

def _check_embed(key, defkey, path=None, warn=False):
    if False:
        i = 10
        return i + 15
    "\n    Find a boolean value, configured in the environment at *key* or\n    *defkey* (typically, *defkey* will be shared by several calls). If\n    those don't exist, then check for the existence of *path* and return\n    that (if path is given)\n    "
    value = bool_from_environ(key)
    if value is None:
        value = bool_from_environ(defkey)
    if value is not None:
        if warn:
            print('Warning: gevent setup: legacy environment key %s or %s found' % (key, defkey))
        return value
    return os.path.exists(path) if path is not None else None

def should_embed(dep_name):
    if False:
        return 10
    '\n    Check the configuration for the dep_name and see if it should be\n    embedded. Environment keys are derived from the dep name: libev\n    becomes GEVENTSETUP_EMBED_LIBEV and c-ares becomes\n    GEVENTSETUP_EMBED_CARES.\n    '
    path = dep_abspath(dep_name)
    normal_dep_key = dep_name.replace('-', '').upper()
    default_key = 'GEVENTSETUP_EMBED'
    dep_key = default_key + '_' + normal_dep_key
    result = _check_embed(dep_key, default_key)
    if result is not None:
        return result
    legacy_default_key = 'EMBED'
    legacy_dep_key = normal_dep_key + '_' + legacy_default_key
    return _check_embed(legacy_dep_key, legacy_default_key, path, warn=True)

def get_include_dirs(*extra_paths):
    if False:
        return 10
    '\n    Return additional include directories that might be needed to\n    compile extensions. Specifically, we need the greenlet.h header\n    in many of our extensions.\n    '
    dist_inc_dir = os.path.abspath(dist_sysconfig.get_python_inc())
    sys_inc_dir = os.path.abspath(sysconfig.get_path('include'))
    venv_include_dir = os.path.join(sys.prefix, 'include', 'site', 'python' + sysconfig.get_python_version())
    venv_include_dir = os.path.abspath(venv_include_dir)
    dep_inc_dir = os.path.abspath('deps')
    return [p for p in (dist_inc_dir, sys_inc_dir, dep_inc_dir) + extra_paths if os.path.exists(p)]

def _system(cmd, cwd=None, env=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    sys.stdout.write('Running %r in %s\n' % (cmd, cwd or os.getcwd()))
    sys.stdout.flush()
    if 'shell' not in kwargs:
        kwargs['shell'] = True
    env = env or os.environ.copy()
    return check_call(cmd, cwd=cwd, env=env, **kwargs)

def system(cmd, cwd=None, env=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    if _system(cmd, cwd=cwd, env=env, **kwargs):
        sys.exit(1)
COMMON_UTILITY_INCLUDE_DIR = 'src/gevent/_generated_include'

def _dummy_cythonize(extensions, **_kwargs):
    if False:
        print('Hello World!')
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            (path, ext) = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                ext = '.c'
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions
try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = _dummy_cythonize

def cythonize1(ext):
    if False:
        return 10
    standard_include_paths = ['src/gevent', 'src/gevent/libev', 'src/gevent/resolver', '.']
    if PY311:
        ext.define_macros.append(('CYTHON_FAST_THREAD_STATE', '0'))
    try:
        new_ext = cythonize([ext], include_path=standard_include_paths, annotate=True, compiler_directives={'language_level': '3str', 'always_allow_keywords': False, 'infer_types': True, 'nonecheck': False}, common_utility_include_dir=COMMON_UTILITY_INCLUDE_DIR)[0]
    except ValueError:
        import traceback
        traceback.print_exc()
        new_ext = _dummy_cythonize([ext])[0]
    for optional_attr in ('configure', 'optional'):
        if hasattr(ext, optional_attr):
            setattr(new_ext, optional_attr, getattr(ext, optional_attr))
    new_ext.extra_compile_args.extend(IGNORE_THIRD_PARTY_WARNINGS)
    new_ext.include_dirs.extend(standard_include_paths)
    return new_ext
IGNORE_THIRD_PARTY_WARNINGS = ()
if sys.platform == 'darwin':
    IGNORE_THIRD_PARTY_WARNINGS += ('-Wno-unreachable-code', '-Wno-deprecated-declarations', '-Wno-incompatible-sysroot', '-Wno-tautological-compare', '-Wno-implicit-function-declaration', '-Wno-unused-value', '-Wno-macro-redefined')

class BuildFailed(Exception):
    pass
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError, IOError)

class ConfiguringBuildExt(build_ext):
    gevent_pre_run_actions = ()

    @classmethod
    def gevent_add_pre_run_action(cls, action):
        if False:
            while True:
                i = 10
        cls.gevent_pre_run_actions += (action,)

    def finalize_options(self):
        if False:
            print('Hello World!')
        build_ext.finalize_options(self)

    def gevent_prepare(self, ext):
        if False:
            for i in range(10):
                print('nop')
        configure = getattr(ext, 'configure', None)
        if configure:
            configure(self, ext)

    def build_extension(self, ext):
        if False:
            print('Hello World!')
        self.gevent_prepare(ext)
        try:
            return build_ext.build_extension(self, ext)
        except ext_errors:
            if getattr(ext, 'optional', False):
                raise BuildFailed()
            raise

    def pre_run(self, *_args):
        if False:
            while True:
                i = 10
        for action in self.gevent_pre_run_actions:
            action()

class Extension(_Extension):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.libraries = []
        self.define_macros = []
        _Extension.__init__(self, *args, **kwargs)
from distutils.command.clean import clean
from distutils import log
from distutils.dir_util import remove_tree

class GeventClean(clean):
    BASE_GEVENT_SRC = os.path.join('src', 'gevent')

    def __find_directories_in(self, top, named=None):
        if False:
            i = 10
            return i + 15
        "\n        Iterate directories, beneath and including *top* ignoring '.'\n        entries.\n        "
        for (dirpath, dirnames, _) in os.walk(top):
            dirnames[:] = [x for x in dirnames if not x.startswith('.')]
            for dirname in dirnames:
                if named is None or named == dirname:
                    yield os.path.join(dirpath, dirname)

    def __glob_under(self, base, file_pat):
        if False:
            i = 10
            return i + 15
        return glob_many(os.path.join(base, file_pat), *(os.path.join(x, file_pat) for x in self.__find_directories_in(base)))

    def __remove_dirs(self, remove_file):
        if False:
            while True:
                i = 10
        dirs_to_remove = ['htmlcov', '.eggs', COMMON_UTILITY_INCLUDE_DIR]
        if self.all:
            dirs_to_remove += ['.tox', '.runtimes', 'wheelhouse', os.path.join('.', 'docs', '_build')]
        dir_finders = [(self.__find_directories_in, '.', '__pycache__')]
        for finder in dir_finders:
            func = finder[0]
            args = finder[1:]
            dirs_to_remove.extend(func(*args))
        for f in sorted(dirs_to_remove):
            remove_file(f)

    def run(self):
        if False:
            print('Hello World!')
        clean.run(self)
        if self.dry_run:

            def remove_file(f):
                if False:
                    i = 10
                    return i + 15
                if os.path.isdir(f):
                    remove_tree(f, dry_run=self.dry_run)
                elif os.path.exists(f):
                    log.info("Would remove '%s'", f)
        else:

            def remove_file(f):
                if False:
                    for i in range(10):
                        print('nop')
                if os.path.isdir(f):
                    remove_tree(f, dry_run=self.dry_run)
                elif os.path.exists(f):
                    log.info("Removing '%s'", f)
                    os.remove(f)
        self.__remove_dirs(remove_file)

        def glob_gevent(file_path):
            if False:
                return 10
            return glob(os.path.join(self.BASE_GEVENT_SRC, file_path))

        def glob_gevent_and_under(file_pat):
            if False:
                return 10
            return self.__glob_under(self.BASE_GEVENT_SRC, file_pat)

        def glob_root_and_under(file_pat):
            if False:
                return 10
            return self.__glob_under('.', file_pat)
        files_to_remove = ['.coverage', os.path.join(self.BASE_GEVENT_SRC, 'libev', 'corecext.c'), os.path.join(self.BASE_GEVENT_SRC, 'libev', 'corecext.h'), os.path.join(self.BASE_GEVENT_SRC, 'resolver', 'cares.c'), os.path.join(self.BASE_GEVENT_SRC, 'resolver', 'cares.c')]

        def dep_configure_artifacts(dep):
            if False:
                for i in range(10):
                    print('nop')
            for f in ('config.h', 'config.log', 'config.status', 'config.cache', 'configure-output.txt', '.libs'):
                yield os.path.join('deps', dep, f)
        file_finders = [(glob_gevent, '*.c'), (glob_gevent_and_under, '*.html'), (glob_gevent_and_under, '*.so'), (glob_gevent_and_under, '*.pyd'), (glob_root_and_under, '*.o'), (glob_gevent_and_under, '*.pyc'), (glob_gevent_and_under, '*.pyo'), (dep_configure_artifacts, 'libev'), (dep_configure_artifacts, 'libuv'), (dep_configure_artifacts, 'c-ares')]
        for (func, pat) in file_finders:
            files_to_remove.extend(func(pat))
        for f in sorted(files_to_remove):
            remove_file(f)