from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from os import fspath
from subprocess import Popen
from . import __version__
from .core.types import PathLike
from .settings import settings
from .util.compiler import _nodejs_path
__all__ = ('init', 'build')

def init(base_dir: PathLike, *, interactive: bool=False, bokehjs_version: str | None=None, debug: bool=False) -> bool:
    if False:
        i = 10
        return i + 15
    ' Initialize a directory as a new bokeh extension.\n\n    Arguments:\n        base_dir (str) : The location of the extension.\n\n        interactive (bool) : Guide the user step-by-step.\n\n        bokehjs_version (str) : Use a specific version of bokehjs.\n\n        debug (bool) : Allow for remote debugging.\n\n    Returns:\n        bool\n\n    '
    args: list[str] = []
    if interactive:
        args.append('--interactive')
    if bokehjs_version:
        args.extend(['--bokehjs-version', bokehjs_version])
    proc = _run_command('init', base_dir, args, debug)
    return proc.returncode == 0

def build(base_dir: PathLike, *, rebuild: bool=False, debug: bool=False) -> bool:
    if False:
        print('Hello World!')
    ' Build a bokeh extension in the given directory.\n\n    Arguments:\n        base_dir (str) : The location of the extension.\n\n        rebuild (bool) : Ignore caches and rebuild from scratch.\n\n        debug (bool) : Allow for remote debugging.\n\n    Returns:\n        bool\n\n    '
    args: list[str] = []
    if rebuild:
        args.append('--rebuild')
    proc = _run_command('build', base_dir, args, debug)
    return proc.returncode == 0

def _run_command(command: str, base_dir: PathLike, args: list[str], debug: bool=False) -> Popen[bytes]:
    if False:
        print('Hello World!')
    bokehjs_dir = settings.bokehjs_path()
    if debug:
        compiler_script = bokehjs_dir / 'js' / 'compiler' / 'main.js'
    else:
        compiler_script = bokehjs_dir / 'js' / 'compiler.js'
    cmd = ['--no-deprecation', fspath(compiler_script), command, '--base-dir', fspath(base_dir), '--bokehjs-dir', fspath(bokehjs_dir), '--bokeh-version', __version__]
    if debug:
        cmd.insert(0, '--inspect-brk')
    cmd.extend(args)
    proc = Popen([_nodejs_path(), *cmd])
    proc.communicate()
    return proc