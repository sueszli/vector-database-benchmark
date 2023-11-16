from __future__ import annotations
import contextlib
import functools
import json
import os.path
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Generator
from collections.abc import Sequence
from typing import ContextManager
from typing import IO
from typing import Protocol
import pre_commit.constants as C
from pre_commit import lang_base
from pre_commit.envcontext import envcontext
from pre_commit.envcontext import PatchesT
from pre_commit.envcontext import Var
from pre_commit.prefix import Prefix
from pre_commit.util import cmd_output
from pre_commit.util import rmtree
ENVIRONMENT_DIR = 'golangenv'
health_check = lang_base.basic_health_check
run_hook = lang_base.basic_run_hook
_ARCH_ALIASES = {'x86_64': 'amd64', 'i386': '386', 'aarch64': 'arm64', 'armv8': 'arm64', 'armv7l': 'armv6l'}
_ARCH = platform.machine().lower()
_ARCH = _ARCH_ALIASES.get(_ARCH, _ARCH)

class ExtractAll(Protocol):

    def extractall(self, path: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        ...
if sys.platform == 'win32':
    _EXT = 'zip'

    def _open_archive(bio: IO[bytes]) -> ContextManager[ExtractAll]:
        if False:
            i = 10
            return i + 15
        return zipfile.ZipFile(bio)
else:
    _EXT = 'tar.gz'

    def _open_archive(bio: IO[bytes]) -> ContextManager[ExtractAll]:
        if False:
            for i in range(10):
                print('nop')
        return tarfile.open(fileobj=bio)

@functools.lru_cache(maxsize=1)
def get_default_version() -> str:
    if False:
        i = 10
        return i + 15
    if lang_base.exe_exists('go'):
        return 'system'
    else:
        return C.DEFAULT

def get_env_patch(venv: str, version: str) -> PatchesT:
    if False:
        for i in range(10):
            print('nop')
    if version == 'system':
        return (('PATH', (os.path.join(venv, 'bin'), os.pathsep, Var('PATH'))),)
    return (('GOROOT', os.path.join(venv, '.go')), ('PATH', (os.path.join(venv, 'bin'), os.pathsep, os.path.join(venv, '.go', 'bin'), os.pathsep, Var('PATH'))))

@functools.lru_cache
def _infer_go_version(version: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    if version != C.DEFAULT:
        return version
    resp = urllib.request.urlopen('https://go.dev/dl/?mode=json')
    return json.load(resp)[0]['version'][2:]

def _get_url(version: str) -> str:
    if False:
        return 10
    os_name = platform.system().lower()
    version = _infer_go_version(version)
    return f'https://dl.google.com/go/go{version}.{os_name}-{_ARCH}.{_EXT}'

def _install_go(version: str, dest: str) -> None:
    if False:
        print('Hello World!')
    try:
        resp = urllib.request.urlopen(_get_url(version))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(f'Could not find a version matching your system requirements (os={platform.system().lower()}; arch={_ARCH})') from e
        else:
            raise
    else:
        with tempfile.TemporaryFile() as f:
            shutil.copyfileobj(resp, f)
            f.seek(0)
            with _open_archive(f) as archive:
                archive.extractall(dest)
        shutil.move(os.path.join(dest, 'go'), os.path.join(dest, '.go'))

@contextlib.contextmanager
def in_env(prefix: Prefix, version: str) -> Generator[None, None, None]:
    if False:
        for i in range(10):
            print('nop')
    envdir = lang_base.environment_dir(prefix, ENVIRONMENT_DIR, version)
    with envcontext(get_env_patch(envdir, version)):
        yield

def install_environment(prefix: Prefix, version: str, additional_dependencies: Sequence[str]) -> None:
    if False:
        return 10
    env_dir = lang_base.environment_dir(prefix, ENVIRONMENT_DIR, version)
    if version != 'system':
        _install_go(version, env_dir)
    if sys.platform == 'cygwin':
        gopath = cmd_output('cygpath', '-w', env_dir)[1].strip()
    else:
        gopath = env_dir
    env = dict(os.environ, GOPATH=gopath)
    env.pop('GOBIN', None)
    if version != 'system':
        env['GOROOT'] = os.path.join(env_dir, '.go')
        env['PATH'] = os.pathsep.join((os.path.join(env_dir, '.go', 'bin'), os.environ['PATH']))
    lang_base.setup_cmd(prefix, ('go', 'install', './...'), env=env)
    for dependency in additional_dependencies:
        lang_base.setup_cmd(prefix, ('go', 'install', dependency), env=env)
    pkgdir = os.path.join(env_dir, 'pkg')
    if os.path.exists(pkgdir):
        rmtree(pkgdir)