import contextlib
import os
import subprocess
import sys
import traceback
import warnings
from collections.abc import Iterator
from typing import NoReturn
from build import BuildBackendException, BuildException, FailedProcessError, ProjectBuilder
from build.env import DefaultIsolatedEnv
_COLORS = {'red': '\x1b[91m', 'green': '\x1b[92m', 'yellow': '\x1b[93m', 'bold': '\x1b[1m', 'dim': '\x1b[2m', 'underline': '\x1b[4m', 'reset': '\x1b[0m'}
_NO_COLORS = {color: '' for color in _COLORS}

def _init_colors() -> dict[str, str]:
    if False:
        while True:
            i = 10
    if 'NO_COLOR' in os.environ:
        if 'FORCE_COLOR' in os.environ:
            warnings.warn('Both NO_COLOR and FORCE_COLOR environment variables are set, disabling color', stacklevel=2)
        return _NO_COLORS
    elif 'FORCE_COLOR' in os.environ or sys.stdout.isatty():
        return _COLORS
    return _NO_COLORS
_STYLES = _init_colors()

def _cprint(fmt: str='', msg: str='') -> None:
    if False:
        print('Hello World!')
    print(fmt.format(msg, **_STYLES), flush=True)

def _error(msg: str, code: int=1) -> NoReturn:
    if False:
        while True:
            i = 10
    '\n    Print an error message and exit. Will color the output when writing to a TTY.\n\n    :param msg: Error message\n    :param code: Error code\n    '
    _cprint('{red}ERROR{reset} {}', msg)
    raise SystemExit(code)

class _ProjectBuilder(ProjectBuilder):

    @staticmethod
    def log(message: str) -> None:
        if False:
            print('Hello World!')
        _cprint('{bold}* {}{reset}', message)

class _DefaultIsolatedEnv(DefaultIsolatedEnv):

    @staticmethod
    def log(message: str) -> None:
        if False:
            i = 10
            return i + 15
        _cprint('{bold}* {}{reset}', message)

@contextlib.contextmanager
def _handle_build_error() -> Iterator[None]:
    if False:
        for i in range(10):
            print('nop')
    try:
        yield
    except (BuildException, FailedProcessError) as e:
        _error(str(e))
    except BuildBackendException as e:
        if isinstance(e.exception, subprocess.CalledProcessError):
            _cprint()
            _error(str(e))
        if e.exc_info:
            tb_lines = traceback.format_exception(e.exc_info[0], e.exc_info[1], e.exc_info[2], limit=-1)
            tb = ''.join(tb_lines)
        else:
            tb = traceback.format_exc(-1)
        _cprint('\n{dim}{}{reset}\n', tb.strip('\n'))
        _error(str(e))