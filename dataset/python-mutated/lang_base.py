from __future__ import annotations
import contextlib
import os
import random
import re
import shlex
from collections.abc import Generator
from collections.abc import Sequence
from typing import Any
from typing import ContextManager
from typing import NoReturn
from typing import Protocol
import pre_commit.constants as C
from pre_commit import parse_shebang
from pre_commit import xargs
from pre_commit.prefix import Prefix
from pre_commit.util import cmd_output_b
FIXED_RANDOM_SEED = 1542676187
SHIMS_RE = re.compile('[/\\\\]shims[/\\\\]')

class Language(Protocol):

    @property
    def ENVIRONMENT_DIR(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        ...

    def get_default_version(self) -> str:
        if False:
            i = 10
            return i + 15
        ...

    def health_check(self, prefix: Prefix, version: str) -> str | None:
        if False:
            return 10
        ...

    def install_environment(self, prefix: Prefix, version: str, additional_dependencies: Sequence[str]) -> None:
        if False:
            i = 10
            return i + 15
        ...

    def in_env(self, prefix: Prefix, version: str) -> ContextManager[None]:
        if False:
            return 10
        ...

    def run_hook(self, prefix: Prefix, entry: str, args: Sequence[str], file_args: Sequence[str], *, is_local: bool, require_serial: bool, color: bool) -> tuple[int, bytes]:
        if False:
            i = 10
            return i + 15
        ...

def exe_exists(exe: str) -> bool:
    if False:
        print('Hello World!')
    found = parse_shebang.find_executable(exe)
    if found is None:
        return False
    homedir = os.path.expanduser('~')
    try:
        common: str | None = os.path.commonpath((found, homedir))
    except ValueError:
        common = None
    return not SHIMS_RE.search(found) and (os.path.dirname(homedir) == homedir or common != homedir)

def setup_cmd(prefix: Prefix, cmd: tuple[str, ...], **kwargs: Any) -> None:
    if False:
        while True:
            i = 10
    cmd_output_b(*cmd, cwd=prefix.prefix_dir, **kwargs)

def environment_dir(prefix: Prefix, d: str, language_version: str) -> str:
    if False:
        while True:
            i = 10
    return prefix.path(f'{d}-{language_version}')

def assert_version_default(binary: str, version: str) -> None:
    if False:
        i = 10
        return i + 15
    if version != C.DEFAULT:
        raise AssertionError(f'for now, pre-commit requires system-installed {binary} -- you selected `language_version: {version}`')

def assert_no_additional_deps(lang: str, additional_deps: Sequence[str]) -> None:
    if False:
        print('Hello World!')
    if additional_deps:
        raise AssertionError(f'for now, pre-commit does not support additional_dependencies for {lang} -- you selected `additional_dependencies: {additional_deps}`')

def basic_get_default_version() -> str:
    if False:
        while True:
            i = 10
    return C.DEFAULT

def basic_health_check(prefix: Prefix, language_version: str) -> str | None:
    if False:
        return 10
    return None

def no_install(prefix: Prefix, version: str, additional_dependencies: Sequence[str]) -> NoReturn:
    if False:
        i = 10
        return i + 15
    raise AssertionError('This language is not installable')

@contextlib.contextmanager
def no_env(prefix: Prefix, version: str) -> Generator[None, None, None]:
    if False:
        while True:
            i = 10
    yield

def target_concurrency() -> int:
    if False:
        while True:
            i = 10
    if 'PRE_COMMIT_NO_CONCURRENCY' in os.environ:
        return 1
    elif 'TRAVIS' in os.environ:
        return 2
    else:
        return xargs.cpu_count()

def _shuffled(seq: Sequence[str]) -> list[str]:
    if False:
        print('Hello World!')
    'Deterministically shuffle'
    fixed_random = random.Random()
    fixed_random.seed(FIXED_RANDOM_SEED, version=1)
    seq = list(seq)
    fixed_random.shuffle(seq)
    return seq

def run_xargs(cmd: tuple[str, ...], file_args: Sequence[str], *, require_serial: bool, color: bool) -> tuple[int, bytes]:
    if False:
        print('Hello World!')
    if require_serial:
        jobs = 1
    else:
        file_args = _shuffled(file_args)
        jobs = target_concurrency()
    return xargs.xargs(cmd, file_args, target_concurrency=jobs, color=color)

def hook_cmd(entry: str, args: Sequence[str]) -> tuple[str, ...]:
    if False:
        i = 10
        return i + 15
    return (*shlex.split(entry), *args)

def basic_run_hook(prefix: Prefix, entry: str, args: Sequence[str], file_args: Sequence[str], *, is_local: bool, require_serial: bool, color: bool) -> tuple[int, bytes]:
    if False:
        return 10
    return run_xargs(hook_cmd(entry, args), file_args, require_serial=require_serial, color=color)