"""Automatically restart the server when a source file is modified.

Most applications should not access this module directly.  Instead,
pass the keyword argument ``autoreload=True`` to the
`tornado.web.Application` constructor (or ``debug=True``, which
enables this setting and several others).  This will enable autoreload
mode as well as checking for changes to templates and static
resources.  Note that restarting is a destructive operation and any
requests in progress will be aborted when the process restarts.  (If
you want to disable autoreload while using other debug-mode features,
pass both ``debug=True`` and ``autoreload=False``).

This module can also be used as a command-line wrapper around scripts
such as unit test runners.  See the `main` method for details.

The command-line wrapper and Application debug modes can be used together.
This combination is encouraged as the wrapper catches syntax errors and
other import-time failures, while debug mode catches changes once
the server has started.

This module will not work correctly when `.HTTPServer`'s multi-process
mode is used.

Reloading loses any Python interpreter command-line arguments (e.g. ``-u``)
because it re-executes Python using ``sys.executable`` and ``sys.argv``.
Additionally, modifying these variables will cause reloading to behave
incorrectly.

"""
import os
import sys
if __name__ == '__main__':
    if sys.path[0] == os.path.dirname(__file__):
        del sys.path[0]
import functools
import importlib.abc
import os
import pkgutil
import sys
import traceback
import types
import subprocess
import weakref
from tornado import ioloop
from tornado.log import gen_log
from tornado import process
try:
    import signal
except ImportError:
    signal = None
from typing import Callable, Dict, Optional, List, Union
_has_execv = sys.platform != 'win32'
_watched_files = set()
_reload_hooks = []
_reload_attempted = False
_io_loops: 'weakref.WeakKeyDictionary[ioloop.IOLoop, bool]' = weakref.WeakKeyDictionary()
_autoreload_is_main = False
_original_argv: Optional[List[str]] = None
_original_spec = None

def start(check_time: int=500) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Begins watching source files for changes.\n\n    .. versionchanged:: 5.0\n       The ``io_loop`` argument (deprecated since version 4.1) has been removed.\n    '
    io_loop = ioloop.IOLoop.current()
    if io_loop in _io_loops:
        return
    _io_loops[io_loop] = True
    if len(_io_loops) > 1:
        gen_log.warning('tornado.autoreload started more than once in the same process')
    modify_times: Dict[str, float] = {}
    callback = functools.partial(_reload_on_update, modify_times)
    scheduler = ioloop.PeriodicCallback(callback, check_time)
    scheduler.start()

def wait() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Wait for a watched file to change, then restart the process.\n\n    Intended to be used at the end of scripts like unit test runners,\n    to run the tests again after any source file changes (but see also\n    the command-line interface in `main`)\n    '
    io_loop = ioloop.IOLoop()
    io_loop.add_callback(start)
    io_loop.start()

def watch(filename: str) -> None:
    if False:
        return 10
    'Add a file to the watch list.\n\n    All imported modules are watched by default.\n    '
    _watched_files.add(filename)

def add_reload_hook(fn: Callable[[], None]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Add a function to be called before reloading the process.\n\n    Note that for open file and socket handles it is generally\n    preferable to set the ``FD_CLOEXEC`` flag (using `fcntl` or\n    `os.set_inheritable`) instead of using a reload hook to close them.\n    '
    _reload_hooks.append(fn)

def _reload_on_update(modify_times: Dict[str, float]) -> None:
    if False:
        print('Hello World!')
    if _reload_attempted:
        return
    if process.task_id() is not None:
        return
    for module in list(sys.modules.values()):
        if not isinstance(module, types.ModuleType):
            continue
        path = getattr(module, '__file__', None)
        if not path:
            continue
        if path.endswith('.pyc') or path.endswith('.pyo'):
            path = path[:-1]
        _check_file(modify_times, path)
    for path in _watched_files:
        _check_file(modify_times, path)

def _check_file(modify_times: Dict[str, float], path: str) -> None:
    if False:
        return 10
    try:
        modified = os.stat(path).st_mtime
    except Exception:
        return
    if path not in modify_times:
        modify_times[path] = modified
        return
    if modify_times[path] != modified:
        gen_log.info('%s modified; restarting server', path)
        _reload()

def _reload() -> None:
    if False:
        i = 10
        return i + 15
    global _reload_attempted
    _reload_attempted = True
    for fn in _reload_hooks:
        fn()
    if sys.platform != 'win32':
        signal.setitimer(signal.ITIMER_REAL, 0, 0)
    if _autoreload_is_main:
        assert _original_argv is not None
        spec = _original_spec
        argv = _original_argv
    else:
        spec = getattr(sys.modules['__main__'], '__spec__', None)
        argv = sys.argv
    if spec and spec.name != '__main__':
        argv = ['-m', spec.name] + argv[1:]
    if not _has_execv:
        subprocess.Popen([sys.executable] + argv)
        os._exit(0)
    else:
        os.execv(sys.executable, [sys.executable] + argv)
_USAGE = '\n  python -m tornado.autoreload -m module.to.run [args...]\n  python -m tornado.autoreload path/to/script.py [args...]\n'

def main() -> None:
    if False:
        while True:
            i = 10
    'Command-line wrapper to re-run a script whenever its source changes.\n\n    Scripts may be specified by filename or module name::\n\n        python -m tornado.autoreload -m tornado.test.runtests\n        python -m tornado.autoreload tornado/test/runtests.py\n\n    Running a script with this wrapper is similar to calling\n    `tornado.autoreload.wait` at the end of the script, but this wrapper\n    can catch import-time problems like syntax errors that would otherwise\n    prevent the script from reaching its call to `wait`.\n    '
    import optparse
    import tornado.autoreload
    global _autoreload_is_main
    global _original_argv, _original_spec
    tornado.autoreload._autoreload_is_main = _autoreload_is_main = True
    original_argv = sys.argv
    tornado.autoreload._original_argv = _original_argv = original_argv
    original_spec = getattr(sys.modules['__main__'], '__spec__', None)
    tornado.autoreload._original_spec = _original_spec = original_spec
    parser = optparse.OptionParser(prog='python -m tornado.autoreload', usage=_USAGE, epilog='Either -m or a path must be specified, but not both')
    parser.disable_interspersed_args()
    parser.add_option('-m', dest='module', metavar='module', help='module to run')
    parser.add_option('--until-success', action='store_true', help='stop reloading after the program exist successfully (status code 0)')
    (opts, rest) = parser.parse_args()
    if opts.module is None:
        if not rest:
            print('Either -m or a path must be specified', file=sys.stderr)
            sys.exit(1)
        path = rest[0]
        sys.argv = rest[:]
    else:
        path = None
        sys.argv = [sys.argv[0]] + rest
    exit_status: Union[int, str, None] = 1
    try:
        import runpy
        if opts.module is not None:
            runpy.run_module(opts.module, run_name='__main__', alter_sys=True)
        else:
            assert path is not None
            runpy.run_path(path, run_name='__main__')
    except SystemExit as e:
        exit_status = e.code
        gen_log.info('Script exited with status %s', e.code)
    except Exception as e:
        gen_log.warning('Script exited with uncaught exception', exc_info=True)
        for (filename, lineno, name, line) in traceback.extract_tb(sys.exc_info()[2]):
            watch(filename)
        if isinstance(e, SyntaxError):
            if e.filename is not None:
                watch(e.filename)
    else:
        exit_status = 0
        gen_log.info('Script exited normally')
    sys.argv = original_argv
    if opts.module is not None:
        assert opts.module is not None
        loader = pkgutil.get_loader(opts.module)
        if loader is not None and isinstance(loader, importlib.abc.FileLoader):
            watch(loader.get_filename())
    if opts.until_success and (not exit_status):
        return
    wait()
if __name__ == '__main__':
    main()