"""Scalene: a CPU+memory+GPU (and more) profiler for Python.

    https://github.com/plasma-umass/scalene

    See the paper "docs/scalene-paper.pdf" in this repository for technical
    details on an earlier version of Scalene's design; note that a
    number of these details have changed.

    by Emery Berger
    https://emeryberger.com

    usage: scalene test/testme.py
    usage help: scalene --help

"""
import argparse
import atexit
import builtins
import contextlib
import functools
import gc
import importlib
import inspect
import json
import math
import multiprocessing
import os
import pathlib
import platform
import re
import signal
import stat
import sys
import sysconfig
import tempfile
import threading
import time
import traceback
import webbrowser
from rich.console import Console
from scalene.find_browser import find_browser
console = Console(style='white on blue')

def nada(*args):
    if False:
        while True:
            i = 10
    pass
console.log = nada
from collections import defaultdict
from importlib.abc import SourceLoader
from importlib.machinery import ModuleSpec
from jinja2 import Environment, FileSystemLoader
from types import CodeType, FrameType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast
from scalene.scalene_arguments import ScaleneArguments
from scalene.scalene_client_timer import ScaleneClientTimer
from scalene.scalene_funcutils import ScaleneFuncUtils
from scalene.scalene_json import ScaleneJSON
from scalene.scalene_mapfile import ScaleneMapFile
from scalene.scalene_output import ScaleneOutput
from scalene.scalene_preload import ScalenePreload
from scalene.scalene_signals import ScaleneSignals
from scalene.scalene_statistics import Address, ByteCodeIndex, Filename, LineNumber, ScaleneStatistics
from scalene.scalene_version import scalene_version, scalene_date
if sys.platform != 'win32':
    import resource
if platform.system() == 'Darwin':
    from scalene.scalene_apple_gpu import ScaleneAppleGPU as ScaleneGPU
else:
    from scalene.scalene_gpu import ScaleneGPU
from scalene.scalene_parseargs import ScaleneParseArgs, StopJupyterExecution
from scalene.scalene_sigqueue import ScaleneSigQueue
MINIMUM_PYTHON_VERSION_MAJOR = 3
MINIMUM_PYTHON_VERSION_MINOR = 8

def require_python(version: Tuple[int, int]) -> None:
    if False:
        i = 10
        return i + 15
    assert sys.version_info >= version, f'Scalene requires Python version {version[0]}.{version[1]} or above.'
require_python((MINIMUM_PYTHON_VERSION_MAJOR, MINIMUM_PYTHON_VERSION_MINOR))

class LineNo:

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        frame = inspect.currentframe()
        assert frame
        assert frame.f_back
        return str(frame.f_back.f_lineno)

class FileName:

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        frame = inspect.currentframe()
        assert frame
        assert frame.f_back
        assert frame.f_back.f_code
        return str(frame.f_back.f_code.co_filename)
__LINE__ = LineNo()
__FILE__ = FileName()

def scalene_redirect_profile(func: Any) -> Any:
    if False:
        return 10
    'Handle @profile decorators.\n\n    If Scalene encounters any functions decorated by @profile, it will\n    only report stats for those functions.\n\n    '
    return Scalene.profile(func)
builtins.profile = scalene_redirect_profile
NEWLINE_TRIGGER_LENGTH = 98820

def start() -> None:
    if False:
        i = 10
        return i + 15
    'Start profiling.'
    Scalene.start()

def stop() -> None:
    if False:
        while True:
            i = 10
    'Stop profiling.'
    Scalene.stop()

def _get_module_details(mod_name: str, error: Type[Exception]=ImportError) -> Tuple[str, ModuleSpec, CodeType]:
    if False:
        print('Hello World!')
    'Copy of `runpy._get_module_details`, but not private.'
    if mod_name.startswith('.'):
        raise error('Relative module names not supported')
    (pkg_name, _, _) = mod_name.rpartition('.')
    if pkg_name:
        try:
            __import__(pkg_name)
        except ImportError as e:
            if e.name is None or (e.name != pkg_name and (not pkg_name.startswith(e.name + '.'))):
                raise
        existing = sys.modules.get(mod_name)
        if existing is not None and (not hasattr(existing, '__path__')):
            from warnings import warn
            msg = '{mod_name!r} found in sys.modules after import of package {pkg_name!r}, but prior to execution of {mod_name!r}; this may result in unpredictable behaviour'.format(mod_name=mod_name, pkg_name=pkg_name)
            warn(RuntimeWarning(msg))
    try:
        spec = importlib.util.find_spec(mod_name)
    except (ImportError, AttributeError, TypeError, ValueError) as ex:
        msg = 'Error while finding module specification for {!r} ({}: {})'
        if mod_name.endswith('.py'):
            msg += f". Try using '{mod_name[:-3]}' instead of '{mod_name}' as the module name."
        raise error(msg.format(mod_name, type(ex).__name__, ex)) from ex
    if spec is None:
        raise error('No module named %s' % mod_name)
    if spec.submodule_search_locations is not None:
        if mod_name == '__main__' or mod_name.endswith('.__main__'):
            raise error('Cannot use package as __main__ module')
        try:
            pkg_main_name = mod_name + '.__main__'
            return _get_module_details(pkg_main_name, error)
        except error as e:
            if mod_name not in sys.modules:
                raise
            raise error(('%s; %r is a package and cannot ' + 'be directly executed') % (e, mod_name))
    loader = spec.loader
    if not isinstance(loader, SourceLoader):
        raise error('%r is a namespace package and cannot be executed' % mod_name)
    try:
        code = loader.get_code(mod_name)
    except ImportError as e:
        raise error(format(e)) from e
    if code is None:
        raise error('No code object available for %s' % mod_name)
    return (mod_name, spec, code)

class Scalene:
    """The Scalene profiler itself."""
    __availableCPUs: int
    try:
        __availableCPUs = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()
        __availableCPUs = cpu_count if cpu_count else 1
    __in_jupyter = False
    __start_time = 0
    __sigterm_exit_code = 143
    __is_child = -1
    __parent_pid = -1
    __initialized: bool = False
    __last_profiled = [Filename('NADA'), LineNumber(0), ByteCodeIndex(0)]
    __last_profiled_invalidated = False
    __gui_dir = 'scalene-gui'
    __profile_filename = Filename('profile.json')
    __profiler_html = Filename('profile.html')
    __error_message = 'Error in program being profiled'
    BYTES_PER_MB = 1024 * 1024
    MALLOC_ACTION = 'M'
    FREE_ACTION = 'F'
    FREE_ACTION_SAMPLED = 'f'
    __files_to_profile: Set[Filename] = set()
    __functions_to_profile: Dict[Filename, Set[Any]] = defaultdict(set)
    __original_thread_join = threading.Thread.join
    __original_lock = threading.Lock
    __args = ScaleneArguments()
    __signals = ScaleneSignals()
    __stats = ScaleneStatistics()
    __output = ScaleneOutput()
    __json = ScaleneJSON()
    __gpu = ScaleneGPU()
    __output.gpu = __gpu.has_gpu()
    __json.gpu = __gpu.has_gpu()
    __invalidate_queue: List[Tuple[Filename, LineNumber]] = []
    __invalidate_mutex: threading.Lock
    __profiler_base: str

    @staticmethod
    def get_original_lock() -> threading.Lock:
        if False:
            for i in range(10):
                print('nop')
        'Return the true lock, which we shim in replacement_lock.py.'
        return Scalene.__original_lock()
    __all_python_names = [os.path.basename(sys.executable), os.path.basename(sys.executable) + str(sys.version_info.major), os.path.basename(sys.executable) + str(sys.version_info.major) + '.' + str(sys.version_info.minor)]
    __last_signal_time_virtual: float = 0
    __last_signal_time_wallclock: float = 0
    __last_signal_time_sys: float = 0
    __last_signal_time_user: float = 0
    __program_path = Filename('')
    __entrypoint_dir = Filename('')
    __python_alias_dir: pathlib.Path
    __next_output_time: float = float('inf')
    __pid: int = 0
    __malloc_mapfile: ScaleneMapFile
    __memcpy_mapfile: ScaleneMapFile
    __program_being_profiled = Filename('')
    __is_thread_sleeping: Dict[int, bool] = defaultdict(bool)
    child_pids: Set[int] = set()
    __alloc_sigq: ScaleneSigQueue[Any]
    __memcpy_sigq: ScaleneSigQueue[Any]
    __sigqueues: List[ScaleneSigQueue[Any]]
    client_timer: ScaleneClientTimer = ScaleneClientTimer()
    __orig_signal = signal.signal
    __orig_exit = os._exit
    __orig_raise_signal = signal.raise_signal
    __orig_kill = os.kill
    if sys.platform != 'win32':
        __orig_setitimer = signal.setitimer
        __orig_siginterrupt = signal.siginterrupt

    @staticmethod
    def get_all_signals_set() -> Set[int]:
        if False:
            print('Hello World!')
        'Return the set of all signals currently set.\n\n        Used by replacement_signal_fns.py to shim signals used by the client program.\n        '
        return set(Scalene.__signals.get_all_signals())

    @staticmethod
    def get_timer_signals() -> Tuple[int, signal.Signals]:
        if False:
            i = 10
            return i + 15
        'Return the set of all TIMER signals currently set.\n\n        Used by replacement_signal_fns.py to shim timers used by the client program.\n        '
        return Scalene.__signals.get_timer_signals()

    @staticmethod
    def set_in_jupyter() -> None:
        if False:
            print('Hello World!')
        'Tell Scalene that it is running inside a Jupyter notebook.'
        Scalene.__in_jupyter = True

    @staticmethod
    def in_jupyter() -> bool:
        if False:
            i = 10
            return i + 15
        'Return whether Scalene is running inside a Jupyter notebook.'
        return Scalene.__in_jupyter

    @staticmethod
    def interruption_handler(signum: Union[Callable[[signal.Signals, FrameType], None], int, signal.Handlers, None], this_frame: Optional[FrameType]) -> None:
        if False:
            i = 10
            return i + 15
        'Handle keyboard interrupts (e.g., Ctrl-C).'
        raise KeyboardInterrupt

    @staticmethod
    def on_stack(frame: FrameType, fname: Filename, lineno: LineNumber) -> Optional[FrameType]:
        if False:
            for i in range(10):
                print('nop')
        'Find a frame matching the given filename and line number, if any.\n\n        Used for checking whether we are still executing the same line\n        of code or not in invalidate_lines (for per-line memory\n        accounting).\n        '
        f = frame
        current_file_and_line = (fname, lineno)
        while f:
            if (f.f_code.co_filename, f.f_lineno) == current_file_and_line:
                return f
            f = cast(FrameType, f.f_back)
        return None

    @staticmethod
    def update_line() -> None:
        if False:
            print('Hello World!')
        'Mark a new line by allocating the trigger number of bytes.'
        bytearray(NEWLINE_TRIGGER_LENGTH)

    @staticmethod
    def invalidate_lines_python(frame: FrameType, _event: str, _arg: str) -> Any:
        if False:
            while True:
                i = 10
        'Mark the last_profiled information as invalid as soon as we execute a different line of code.'
        try:
            ff = frame.f_code.co_filename
            fl = frame.f_lineno
            (fname, lineno, lasti) = Scalene.__last_profiled
            if ff == fname and fl == lineno:
                return Scalene.invalidate_lines_python
            frame.f_trace = None
            frame.f_trace_lines = False
            if Scalene.on_stack(frame, fname, lineno):
                return None
            sys.settrace(None)
            with Scalene.__invalidate_mutex:
                Scalene.__invalidate_queue.append((Scalene.__last_profiled[0], Scalene.__last_profiled[1]))
                Scalene.update_line()
            Scalene.__last_profiled_invalidated = True
            Scalene.__last_profiled = [Filename('NADA'), LineNumber(0), ByteCodeIndex(0)]
            return None
        except AttributeError:
            return None
        except Exception as e:
            print(f'{Scalene.__error_message}:\n', e)
            traceback.print_exc()
            return None

    @classmethod
    def clear_metrics(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Clear the various states for forked processes.'
        cls.__stats.clear()
        cls.child_pids.clear()

    @classmethod
    def add_child_pid(cls, pid: int) -> None:
        if False:
            i = 10
            return i + 15
        'Add this pid to the set of children. Used when forking.'
        cls.child_pids.add(pid)

    @classmethod
    def remove_child_pid(cls, pid: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Remove a child once we have joined with it (used by replacement_pjoin.py).'
        with contextlib.suppress(KeyError):
            cls.child_pids.remove(pid)

    @staticmethod
    def profile(func: Any) -> Any:
        if False:
            return 10
        'Record the file and function name.\n\n        Replacement @profile decorator function.  Scalene tracks which\n        functions - in which files - have been decorated; if any have,\n        it and only reports stats for those.\n\n        '
        Scalene.__files_to_profile.add(func.__code__.co_filename)
        Scalene.__functions_to_profile[func.__code__.co_filename].add(func)
        if Scalene.__args.memory:
            from scalene import pywhere
            pywhere.register_files_to_profile(list(Scalene.__files_to_profile), Scalene.__program_path, Scalene.__args.profile_all)
        return func

    @staticmethod
    def shim(func: Callable[[Any], Any]) -> Any:
        if False:
            return 10
        'Provide a decorator that calls the wrapped function with the\n        Scalene variant.\n\n                Wrapped function must be of type (s: Scalene) -> Any.\n\n                This decorator allows for marking a function in a separate\n                file as a drop-in replacement for an existing library\n                function. The intention is for these functions to replace a\n                function that indefinitely blocks (which interferes with\n                Scalene) with a function that awakens periodically to allow\n                for signals to be delivered.\n\n        '
        func(Scalene)

        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if False:
                return 10
            return func(*args, **kwargs)
        return wrapped

    @staticmethod
    def set_thread_sleeping(tid: int) -> None:
        if False:
            i = 10
            return i + 15
        'Indicate the given thread is sleeping.\n\n        Used to attribute CPU time.\n        '
        Scalene.__is_thread_sleeping[tid] = True

    @staticmethod
    def reset_thread_sleeping(tid: int) -> None:
        if False:
            print('Hello World!')
        'Indicate the given thread is not sleeping.\n\n        Used to attribute CPU time.'
        Scalene.__is_thread_sleeping[tid] = False
    timer_signals = True

    @staticmethod
    def windows_timer_loop() -> None:
        if False:
            print('Hello World!')
        'For Windows, send periodic timer signals; launch as a background thread.'
        Scalene.timer_signals = True
        while Scalene.timer_signals:
            Scalene.__windows_queue.get()
            time.sleep(Scalene.__args.cpu_sampling_rate)
            Scalene.__orig_raise_signal(Scalene.__signals.cpu_signal)

    @staticmethod
    def start_signal_queues() -> None:
        if False:
            return 10
        'Start the signal processing queues (i.e., their threads).'
        for sigq in Scalene.__sigqueues:
            sigq.start()

    @staticmethod
    def stop_signal_queues() -> None:
        if False:
            return 10
        'Stop the signal processing queues (i.e., their threads).'
        for sigq in Scalene.__sigqueues:
            sigq.stop()

    @staticmethod
    def term_signal_handler(signum: Union[Callable[[signal.Signals, FrameType], None], int, signal.Handlers, None], this_frame: Optional[FrameType]) -> None:
        if False:
            return 10
        'Handle terminate signals.'
        Scalene.stop()
        Scalene.output_profile()
        Scalene.__orig_exit(Scalene.__sigterm_exit_code)

    @staticmethod
    def malloc_signal_handler(signum: Union[Callable[[signal.Signals, FrameType], None], int, signal.Handlers, None], this_frame: Optional[FrameType]) -> None:
        if False:
            i = 10
            return i + 15
        'Handle allocation signals.'
        if not Scalene.__args.memory:
            return
        from scalene import pywhere
        if this_frame:
            Scalene.enter_function_meta(this_frame, Scalene.__stats)
        found_frame = False
        f = this_frame
        while f:
            if (found_frame := Scalene.should_trace(f.f_code.co_filename, f.f_code.co_name)):
                break
            f = cast(FrameType, f.f_back)
        if not found_frame:
            return
        assert f
        invalidated = pywhere.get_last_profiled_invalidated()
        (fname, lineno, lasti) = Scalene.__last_profiled
        if not invalidated and this_frame and (not Scalene.on_stack(this_frame, fname, lineno)):
            with Scalene.__invalidate_mutex:
                Scalene.__invalidate_queue.append((Scalene.__last_profiled[0], Scalene.__last_profiled[1]))
                Scalene.update_line()
        pywhere.set_last_profiled_invalidated_false()
        Scalene.__last_profiled = [Filename(f.f_code.co_filename), LineNumber(f.f_lineno), ByteCodeIndex(f.f_lasti)]
        Scalene.__alloc_sigq.put([0])
        pywhere.enable_settrace()
        del this_frame

    @staticmethod
    def free_signal_handler(signum: Union[Callable[[signal.Signals, FrameType], None], int, signal.Handlers, None], this_frame: Optional[FrameType]) -> None:
        if False:
            i = 10
            return i + 15
        'Handle free signals.'
        if this_frame:
            Scalene.enter_function_meta(this_frame, Scalene.__stats)
        Scalene.__alloc_sigq.put([0])
        del this_frame

    @staticmethod
    def memcpy_signal_handler(signum: Union[Callable[[signal.Signals, FrameType], None], int, signal.Handlers, None], this_frame: Optional[FrameType]) -> None:
        if False:
            print('Hello World!')
        'Handle memcpy signals.'
        Scalene.__memcpy_sigq.put((signum, this_frame))
        del this_frame

    @staticmethod
    def enable_signals() -> None:
        if False:
            i = 10
            return i + 15
        'Set up the signal handlers to handle interrupts for profiling and start the\n        timer interrupts.'
        if sys.platform == 'win32':
            Scalene.timer_signals = True
            Scalene.__orig_signal(Scalene.__signals.cpu_signal, Scalene.cpu_signal_handler)
            Scalene.timer_signals = True
            t = threading.Thread(target=Scalene.windows_timer_loop)
            t.start()
            Scalene.__windows_queue.put(None)
            Scalene.start_signal_queues()
            return
        Scalene.start_signal_queues()
        Scalene.__orig_signal(Scalene.__signals.malloc_signal, Scalene.malloc_signal_handler)
        Scalene.__orig_signal(Scalene.__signals.free_signal, Scalene.free_signal_handler)
        Scalene.__orig_signal(Scalene.__signals.memcpy_signal, Scalene.memcpy_signal_handler)
        Scalene.__orig_signal(signal.SIGTERM, Scalene.term_signal_handler)
        for s in Scalene.__signals.get_all_signals():
            Scalene.__orig_siginterrupt(s, False)
        Scalene.__orig_signal(Scalene.__signals.cpu_signal, Scalene.cpu_signal_handler)
        if sys.platform != 'win32':
            Scalene.__orig_setitimer(Scalene.__signals.cpu_timer_signal, Scalene.__args.cpu_sampling_rate)

    def __init__(self, arguments: argparse.Namespace, program_being_profiled: Optional[Filename]=None) -> None:
        if False:
            while True:
                i = 10
        import scalene.replacement_exit
        import scalene.replacement_get_context
        import scalene.replacement_lock
        import scalene.replacement_mp_lock
        import scalene.replacement_pjoin
        import scalene.replacement_signal_fns
        import scalene.replacement_thread_join
        if sys.platform != 'win32':
            import scalene.replacement_fork
            import scalene.replacement_poll_selector
        Scalene.__args = cast(ScaleneArguments, arguments)
        Scalene.__alloc_sigq = ScaleneSigQueue(Scalene.alloc_sigqueue_processor)
        Scalene.__memcpy_sigq = ScaleneSigQueue(Scalene.memcpy_sigqueue_processor)
        Scalene.__sigqueues = [Scalene.__alloc_sigq, Scalene.__memcpy_sigq]
        Scalene.__invalidate_mutex = Scalene.get_original_lock()
        if sys.platform == 'win32':
            import queue
            Scalene.__windows_queue = queue.Queue()
            if arguments.memory:
                print(f'Scalene warning: Memory profiling is not currently supported for Windows.')
                arguments.memory = False
        try:
            Scalene.__malloc_mapfile = ScaleneMapFile('malloc')
            Scalene.__memcpy_mapfile = ScaleneMapFile('memcpy')
        except Exception:
            if arguments.memory:
                sys.exit(1)
        Scalene.__signals.set_timer_signals(arguments.use_virtual_time)
        Scalene.__profiler_base = str(os.path.dirname(__file__))
        if arguments.pid:
            dirname = os.environ['PATH'].split(os.pathsep)[0]
            Scalene.__python_alias_dir = pathlib.Path(dirname)
            Scalene.__pid = arguments.pid
        else:
            Scalene.__python_alias_dir = pathlib.Path(tempfile.mkdtemp(prefix='scalene'))
            Scalene.__pid = 0
            cmdline = ''
            cmdline += f' --cpu-sampling-rate={arguments.cpu_sampling_rate}'
            if arguments.use_virtual_time:
                cmdline += ' --use-virtual-time'
            if 'off' in arguments and arguments.off:
                cmdline += ' --off'
            if arguments.cpu:
                cmdline += ' --cpu'
            if arguments.gpu:
                cmdline += ' --gpu'
            if arguments.memory:
                cmdline += ' --memory'
            if arguments.cli:
                cmdline += ' --cli'
            if arguments.web:
                cmdline += ' --web'
            if arguments.no_browser:
                cmdline += ' --no-browser'
            environ = ScalenePreload.get_preload_environ(arguments)
            if sys.platform == 'win32':
                preface = '\n'.join((f'set {k}={str(v)}\n' for (k, v) in environ.items()))
            else:
                preface = ' '.join(('='.join((k, str(v))) for (k, v) in environ.items()))
            shebang = '@echo off' if sys.platform == 'win32' else '#!/bin/bash'
            executable = sys.executable
            cmdline += f' --pid={os.getpid()} ---'
            all_args = '%* & exit 0' if sys.platform == 'win32' else '"$@"'
            payload = f'{shebang}\n{preface} {executable} -m scalene {cmdline} {all_args}\n'
            for name in Scalene.__all_python_names:
                fname = os.path.join(Scalene.__python_alias_dir, name)
                if sys.platform == 'win32':
                    fname = re.sub('\\.exe$', '.bat', fname)
                with open(fname, 'w') as file:
                    file.write(payload)
                os.chmod(fname, stat.S_IXUSR | stat.S_IRUSR | stat.S_IWUSR)
            sys.path.insert(0, str(Scalene.__python_alias_dir))
            os.environ['PATH'] = str(Scalene.__python_alias_dir) + os.pathsep + os.environ['PATH']
            sys.executable = os.path.join(Scalene.__python_alias_dir, Scalene.__all_python_names[0])
            if sys.platform == 'win32' and sys.executable.endswith('.exe'):
                sys.executable = re.sub('\\.exe$', '.bat', sys.executable)
        atexit.register(Scalene.exit_handler)
        if program_being_profiled:
            Scalene.__program_being_profiled = Filename(program_being_profiled)

    @staticmethod
    def cpu_signal_handler(signum: Union[Callable[[signal.Signals, FrameType], None], int, signal.Handlers, None], this_frame: Optional[FrameType]) -> None:
        if False:
            return 10
        'Handle CPU signals.'
        try:
            now_sys: float = 0
            now_user: float = 0
            if sys.platform != 'win32':
                ru = resource.getrusage(resource.RUSAGE_SELF)
                now_sys = ru.ru_stime
                now_user = ru.ru_utime
            else:
                time_info = os.times()
                now_sys = time_info.system
                now_user = time_info.user
            now_virtual = time.process_time()
            now_wallclock = time.perf_counter()
            if Scalene.__last_signal_time_virtual == 0 or Scalene.__last_signal_time_wallclock == 0:
                Scalene.__last_signal_time_virtual = now_virtual
                Scalene.__last_signal_time_wallclock = now_wallclock
                Scalene.__last_signal_time_sys = now_sys
                Scalene.__last_signal_time_user = now_user
                if sys.platform != 'win32':
                    Scalene.__orig_setitimer(Scalene.__signals.cpu_timer_signal, Scalene.__args.cpu_sampling_rate)
                return
            (gpu_load, gpu_mem_used) = Scalene.__gpu.get_stats()
            Scalene.process_cpu_sample(signum, Scalene.compute_frames_to_record(), now_virtual, now_wallclock, now_sys, now_user, gpu_load, gpu_mem_used, Scalene.__last_signal_time_virtual, Scalene.__last_signal_time_wallclock, Scalene.__last_signal_time_sys, Scalene.__last_signal_time_user, Scalene.__is_thread_sleeping)
            elapsed = now_wallclock - Scalene.__last_signal_time_wallclock
            Scalene.__last_signal_time_virtual = now_virtual
            Scalene.__last_signal_time_wallclock = now_wallclock
            Scalene.__last_signal_time_sys = now_sys
            Scalene.__last_signal_time_user = now_user
            if sys.platform != 'win32':
                if Scalene.client_timer.is_set:
                    (should_raise, remaining_time) = Scalene.client_timer.yield_next_delay(elapsed)
                    if should_raise:
                        Scalene.__orig_raise_signal(signal.SIGUSR1)
                    to_wait: float
                    if remaining_time > 0:
                        to_wait = min(remaining_time, Scalene.__args.cpu_sampling_rate)
                    else:
                        to_wait = Scalene.__args.cpu_sampling_rate
                        Scalene.client_timer.reset()
                    Scalene.__orig_setitimer(Scalene.__signals.cpu_timer_signal, to_wait)
                else:
                    Scalene.__orig_setitimer(Scalene.__signals.cpu_timer_signal, Scalene.__args.cpu_sampling_rate)
        finally:
            if sys.platform == 'win32':
                Scalene.__windows_queue.put(None)

    @staticmethod
    def flamegraph_format() -> str:
        if False:
            for i in range(10):
                print('nop')
        "Converts stacks to a string suitable for input to Brendan Gregg's flamegraph.pl script."
        output = ''
        for stk in Scalene.__stats.stacks.keys():
            for item in stk:
                (fname, fn_name, lineno) = item
                output += f'{fname} {fn_name}:{lineno};'
            output += ' ' + str(Scalene.__stats.stacks[stk])
            output += '\n'
        return output

    @staticmethod
    def output_profile(program_args: Optional[List[str]]=None) -> bool:
        if False:
            return 10
        'Output the profile. Returns true iff there was any info reported the profile.'
        if Scalene.__args.json:
            json_output = Scalene.__json.output_profiles(Scalene.__program_being_profiled, Scalene.__stats, Scalene.__pid, Scalene.profile_this_code, Scalene.__python_alias_dir, Scalene.__program_path, Scalene.__entrypoint_dir, program_args, profile_memory=Scalene.__args.memory, reduced_profile=Scalene.__args.reduced_profile)
            if 'is_child' in json_output:
                return True
            outfile = Scalene.__output.output_file
            if not outfile:
                if sys.platform == 'win32':
                    outfile = 'CON'
                else:
                    outfile = '/dev/stdout'
            with open(outfile, 'w') as f:
                f.write(json.dumps(json_output, sort_keys=True, indent=4) + '\n')
            return json_output != {}
        else:
            output = Scalene.__output
            column_width = Scalene.__args.column_width
            if not Scalene.__args.html:
                with contextlib.suppress(Exception):
                    if 'ipykernel' in sys.modules:
                        column_width = 132
                    else:
                        import shutil
                        column_width = shutil.get_terminal_size().columns
            did_output: bool = output.output_profiles(column_width, Scalene.__stats, Scalene.__pid, Scalene.profile_this_code, Scalene.__python_alias_dir, Scalene.__program_path, program_args, profile_memory=Scalene.__args.memory, reduced_profile=Scalene.__args.reduced_profile)
            return did_output

    @staticmethod
    def profile_this_code(fname: Filename, lineno: LineNumber) -> bool:
        if False:
            return 10
        'When using @profile, only profile files & lines that have been decorated.'
        if not Scalene.__files_to_profile:
            return True
        if fname not in Scalene.__files_to_profile:
            return False
        line_info = (inspect.getsourcelines(fn) for fn in Scalene.__functions_to_profile[fname])
        found_function = any((line_start <= lineno < line_start + len(lines) for (lines, line_start) in line_info))
        return found_function

    @staticmethod
    def add_stack(frame: FrameType) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add one to the stack starting from this frame.'
        stk = list()
        f: Optional[FrameType] = frame
        while f:
            if Scalene.should_trace(f.f_code.co_filename, f.f_code.co_name):
                stk.insert(0, (f.f_code.co_filename, f.f_code.co_name, f.f_lineno))
            f = f.f_back
        Scalene.__stats.stacks[tuple(stk)] += 1

    @staticmethod
    def print_stacks() -> None:
        if False:
            print('Hello World!')
        print(Scalene.__stats.stacks)

    @staticmethod
    def process_cpu_sample(_signum: Union[Callable[[signal.Signals, FrameType], None], int, signal.Handlers, None], new_frames: List[Tuple[FrameType, int, FrameType]], now_virtual: float, now_wallclock: float, now_sys: float, now_user: float, gpu_load: float, gpu_mem_used: float, prev_virtual: float, prev_wallclock: float, _prev_sys: float, prev_user: float, is_thread_sleeping: Dict[int, bool]) -> None:
        if False:
            i = 10
            return i + 15
        'Handle interrupts for CPU profiling.'
        if now_wallclock >= Scalene.__next_output_time:
            Scalene.__next_output_time += Scalene.__args.profile_interval
            stats = Scalene.__stats
            with contextlib.ExitStack() as stack:
                _ = [stack.enter_context(s.lock) for s in Scalene.__sigqueues]
                stats.stop_clock()
                Scalene.output_profile()
                stats.start_clock()
        if not new_frames:
            return
        elapsed_virtual = now_virtual - prev_virtual
        elapsed_wallclock = now_wallclock - prev_wallclock
        elapsed_user = now_user - prev_user
        if any([elapsed_virtual < 0, elapsed_wallclock < 0, elapsed_user < 0]):
            return
        cpu_utilization = 0.0
        if elapsed_wallclock != 0:
            cpu_utilization = elapsed_user / elapsed_wallclock
        core_utilization = cpu_utilization / Scalene.__availableCPUs
        if cpu_utilization > 1.0:
            cpu_utilization = 1.0
            elapsed_wallclock = elapsed_user
        if math.isnan(gpu_load):
            gpu_load = 0.0
        gpu_time = gpu_load * Scalene.__args.cpu_sampling_rate
        Scalene.__stats.total_gpu_samples += gpu_time
        python_time = Scalene.__args.cpu_sampling_rate
        c_time = elapsed_virtual - python_time
        c_time = max(c_time, 0)
        total_time = python_time + c_time
        total_frames = sum((not is_thread_sleeping[tident] for (frame, tident, orig_frame) in new_frames))
        if total_frames == 0:
            total_frames = 1
        normalized_time = total_time / total_frames
        main_thread_frame = new_frames[0][0]
        if Scalene.__args.stacks:
            Scalene.add_stack(main_thread_frame)
        average_python_time = python_time / total_frames
        average_c_time = c_time / total_frames
        average_gpu_time = gpu_time / total_frames
        average_cpu_time = (python_time + c_time) / total_frames
        Scalene.enter_function_meta(main_thread_frame, Scalene.__stats)
        fname = Filename(main_thread_frame.f_code.co_filename)
        lineno = LineNumber(main_thread_frame.f_lineno)
        main_tid = cast(int, threading.main_thread().ident)
        if not is_thread_sleeping[main_tid]:
            Scalene.__stats.cpu_samples_python[fname][lineno] += average_python_time
            Scalene.__stats.cpu_samples_c[fname][lineno] += average_c_time
            Scalene.__stats.cpu_samples[fname] += average_cpu_time
            Scalene.__stats.cpu_utilization[fname][lineno].push(cpu_utilization)
            Scalene.__stats.core_utilization[fname][lineno].push(core_utilization)
            Scalene.__stats.gpu_samples[fname][lineno] += average_gpu_time
            Scalene.__stats.gpu_mem_samples[fname][lineno].push(gpu_mem_used)
        for (frame, tident, orig_frame) in new_frames:
            if frame == main_thread_frame:
                continue
            Scalene.add_stack(frame)
            fname = Filename(frame.f_code.co_filename)
            lineno = LineNumber(frame.f_lineno)
            Scalene.enter_function_meta(frame, Scalene.__stats)
            if is_thread_sleeping[tident]:
                continue
            if ScaleneFuncUtils.is_call_function(orig_frame.f_code, ByteCodeIndex(orig_frame.f_lasti)):
                Scalene.__stats.cpu_samples_c[fname][lineno] += normalized_time
            else:
                Scalene.__stats.cpu_samples_python[fname][lineno] += normalized_time
            Scalene.__stats.cpu_samples[fname] += normalized_time
            Scalene.__stats.cpu_utilization[fname][lineno].push(cpu_utilization)
            Scalene.__stats.core_utilization[fname][lineno].push(core_utilization)
        del new_frames[:]
        del new_frames
        del is_thread_sleeping
        Scalene.__stats.total_cpu_samples += total_time

    @staticmethod
    def compute_frames_to_record() -> List[Tuple[FrameType, int, FrameType]]:
        if False:
            i = 10
            return i + 15
        'Collect all stack frames that Scalene actually processes.'
        frames: List[Tuple[FrameType, int]] = [(cast(FrameType, sys._current_frames().get(cast(int, t.ident), None)), cast(int, t.ident)) for t in threading.enumerate() if t != threading.main_thread()]
        tid = cast(int, threading.main_thread().ident)
        frames.insert(0, (sys._current_frames().get(tid, cast(FrameType, None)), tid))
        new_frames: List[Tuple[FrameType, int, FrameType]] = []
        for (frame, tident) in frames:
            orig_frame = frame
            if not frame:
                continue
            fname = frame.f_code.co_filename
            func = frame.f_code.co_name
            if not fname:
                back = cast(FrameType, frame.f_back)
                fname = Filename(back.f_code.co_filename)
                func = back.f_code.co_name
            while not Scalene.should_trace(fname, func):
                if frame:
                    frame = cast(FrameType, frame.f_back)
                else:
                    break
                if frame:
                    fname = frame.f_code.co_filename
                    func = frame.f_code.co_name
            if frame:
                new_frames.append((frame, tident, orig_frame))
        del frames[:]
        return new_frames

    @staticmethod
    def get_fully_qualified_name(frame: FrameType) -> Filename:
        if False:
            return 10
        version = sys.version_info
        if version.major >= 3 and version.minor >= 11:
            fn_name = Filename(frame.f_code.co_qualname)
            return fn_name
        f = frame
        fn_name = Filename(f.f_code.co_name)
        while f and f.f_back and f.f_back.f_code:
            if 'self' in f.f_locals:
                prepend_name = f.f_locals['self'].__class__.__name__
                if 'Scalene' not in prepend_name:
                    fn_name = Filename(f'{prepend_name}.{fn_name}')
                break
            if 'cls' in f.f_locals:
                prepend_name = getattr(f.f_locals['cls'], '__name__', None)
                if not prepend_name or 'Scalene' in prepend_name:
                    break
                fn_name = Filename(f'{prepend_name}.{fn_name}')
                break
            f = f.f_back
        return fn_name

    @staticmethod
    def enter_function_meta(frame: FrameType, stats: ScaleneStatistics) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Update tracking info so we can correctly report line number info later.'
        fname = Filename(frame.f_code.co_filename)
        lineno = LineNumber(frame.f_lineno)
        f = frame
        try:
            while '<' in Filename(f.f_code.co_name):
                f = cast(FrameType, f.f_back)
                if f is None:
                    return
        except Exception:
            return
        if not Scalene.should_trace(f.f_code.co_filename, f.f_code.co_name):
            return
        fn_name = Scalene.get_fully_qualified_name(f)
        firstline = f.f_code.co_firstlineno
        stats.function_map[fname][lineno] = fn_name
        stats.firstline_map[fn_name] = LineNumber(firstline)

    @staticmethod
    def alloc_sigqueue_processor(x: Optional[List[int]]) -> None:
        if False:
            return 10
        'Handle interrupts for memory profiling (mallocs and frees).'
        stats = Scalene.__stats
        curr_pid = os.getpid()
        arr: List[Tuple[int, str, float, float, str, Filename, LineNumber, ByteCodeIndex]] = []
        with contextlib.suppress(FileNotFoundError):
            while Scalene.__malloc_mapfile.read():
                count_str = Scalene.__malloc_mapfile.get_str()
                if count_str.strip() == '':
                    break
                (action, alloc_time_str, count_str, python_fraction_str, pid, pointer, reported_fname, reported_lineno, bytei_str) = count_str.split(',')
                if int(curr_pid) != int(pid):
                    continue
                arr.append((int(alloc_time_str), action, float(count_str), float(python_fraction_str), pointer, Filename(reported_fname), LineNumber(int(reported_lineno)), ByteCodeIndex(int(bytei_str))))
        stats.alloc_samples += len(arr)
        before = max(stats.current_footprint, 0)
        prevmax = stats.max_footprint
        freed_last_trigger = 0
        for item in arr:
            (_alloc_time, action, count, _python_fraction, pointer, fname, lineno, bytei) = item
            is_malloc = action == Scalene.MALLOC_ACTION
            count /= Scalene.BYTES_PER_MB
            if is_malloc:
                stats.current_footprint += count
                if stats.current_footprint > stats.max_footprint:
                    stats.max_footprint = stats.current_footprint
                    stats.max_footprint_loc = (fname, lineno)
            else:
                assert action in [Scalene.FREE_ACTION, Scalene.FREE_ACTION_SAMPLED]
                stats.current_footprint -= count
                stats.current_footprint = max(0, stats.current_footprint)
                if action == Scalene.FREE_ACTION_SAMPLED and stats.last_malloc_triggered[2] == pointer:
                    freed_last_trigger += 1
            timestamp = time.monotonic_ns() - Scalene.__start_time
            stats.memory_footprint_samples.append([timestamp, stats.current_footprint])
        after = stats.current_footprint
        if freed_last_trigger:
            if freed_last_trigger <= 1:
                (this_fn, this_ln, _this_ptr) = stats.last_malloc_triggered
                if this_ln != 0:
                    (mallocs, frees) = stats.leak_score[this_fn][this_ln]
                    stats.leak_score[this_fn][this_ln] = (mallocs, frees + 1)
            stats.last_malloc_triggered = (Filename(''), LineNumber(0), Address('0x0'))
        allocs = 0.0
        last_malloc = (Filename(''), LineNumber(0), Address('0x0'))
        malloc_pointer = '0x0'
        curr = before
        for item in arr:
            (_alloc_time, action, count, python_fraction, pointer, fname, lineno, bytei) = item
            is_malloc = action == Scalene.MALLOC_ACTION
            if is_malloc and count == NEWLINE_TRIGGER_LENGTH + 1:
                with Scalene.__invalidate_mutex:
                    (last_file, last_line) = Scalene.__invalidate_queue.pop(0)
                stats.memory_malloc_count[last_file][last_line] += 1
                stats.memory_aggregate_footprint[last_file][last_line] += stats.memory_current_highwater_mark[last_file][last_line]
                stats.memory_current_footprint[last_file][last_line] = 0
                stats.memory_current_highwater_mark[last_file][last_line] = 0
                continue
            stats.bytei_map[fname][lineno].add(bytei)
            count /= Scalene.BYTES_PER_MB
            if is_malloc:
                allocs += count
                curr += count
                malloc_pointer = pointer
                stats.memory_malloc_samples[fname][lineno] += count
                stats.memory_python_samples[fname][lineno] += python_fraction * count
                stats.malloc_samples[fname] += 1
                stats.total_memory_malloc_samples += count
                stats.memory_current_footprint[fname][lineno] += count
                if stats.memory_current_footprint[fname][lineno] > stats.memory_current_highwater_mark[fname][lineno]:
                    stats.memory_current_highwater_mark[fname][lineno] = stats.memory_current_footprint[fname][lineno]
                stats.memory_current_highwater_mark[fname][lineno] = max(stats.memory_current_highwater_mark[fname][lineno], stats.memory_current_footprint[fname][lineno])
                stats.memory_max_footprint[fname][lineno] = max(stats.memory_current_footprint[fname][lineno], stats.memory_max_footprint[fname][lineno])
            else:
                assert action in [Scalene.FREE_ACTION, Scalene.FREE_ACTION_SAMPLED]
                curr -= count
                stats.memory_free_samples[fname][lineno] += count
                stats.memory_free_count[fname][lineno] += 1
                stats.total_memory_free_samples += count
                stats.memory_current_footprint[fname][lineno] -= count
                stats.memory_current_footprint[fname][lineno] = max(0, stats.memory_current_footprint[fname][lineno])
            stats.per_line_footprint_samples[fname][lineno].append([time.monotonic_ns() - Scalene.__start_time, max(0, curr)])
            if allocs > 0:
                last_malloc = (Filename(fname), LineNumber(lineno), Address(malloc_pointer))
        stats.allocation_velocity = (stats.allocation_velocity[0] + (after - before), stats.allocation_velocity[1] + allocs)
        if Scalene.__args.memory_leak_detector and prevmax < stats.max_footprint and (stats.max_footprint > 100):
            stats.last_malloc_triggered = last_malloc
            (fname, lineno, _) = last_malloc
            (mallocs, frees) = stats.leak_score[fname][lineno]
            stats.leak_score[fname][lineno] = (mallocs + 1, frees)

    @staticmethod
    def before_fork() -> None:
        if False:
            while True:
                i = 10
        'The parent process should invoke this function just before a fork.\n\n        Invoked by replacement_fork.py.\n        '
        Scalene.stop_signal_queues()

    @staticmethod
    def after_fork_in_parent(child_pid: int) -> None:
        if False:
            i = 10
            return i + 15
        'The parent process should invoke this function after a fork.\n\n        Invoked by replacement_fork.py.\n        '
        Scalene.add_child_pid(child_pid)
        Scalene.start_signal_queues()

    @staticmethod
    def after_fork_in_child() -> None:
        if False:
            return 10
        '\n        Executed by a child process after a fork; mutates the\n        current profiler into a child.\n\n        Invoked by replacement_fork.py.\n        '
        Scalene.__is_child = True
        Scalene.clear_metrics()
        if Scalene.__gpu.has_gpu():
            Scalene.__gpu.nvml_reinit()
        Scalene.__pid = Scalene.__parent_pid
        if 'off' not in Scalene.__args or not Scalene.__args.off:
            Scalene.enable_signals()

    @staticmethod
    def memcpy_sigqueue_processor(_signum: Union[Callable[[signal.Signals, FrameType], None], int, signal.Handlers, None], frame: FrameType) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Process memcpy signals (used in a ScaleneSigQueue).'
        curr_pid = os.getpid()
        arr: List[Tuple[str, int, int, int, int]] = []
        with contextlib.suppress(ValueError):
            while Scalene.__memcpy_mapfile.read():
                count_str = Scalene.__memcpy_mapfile.get_str()
                (memcpy_time_str, count_str2, pid, filename, lineno, bytei) = count_str.split(',')
                if int(curr_pid) != int(pid):
                    continue
                arr.append((filename, int(lineno), int(bytei), int(memcpy_time_str), int(count_str2)))
        arr.sort()
        for item in arr:
            (filename, linenum, byteindex, _memcpy_time, count) = item
            fname = Filename(filename)
            line_no = LineNumber(linenum)
            byteidx = ByteCodeIndex(byteindex)
            Scalene.__stats.bytei_map[fname][line_no].add(byteidx)
            Scalene.__stats.memcpy_samples[fname][line_no] += int(count)

    @staticmethod
    @functools.lru_cache(None)
    def should_trace(filename: Filename, func: str) -> bool:
        if False:
            return 10
        'Return true if we should trace this filename and function.'
        if not filename:
            return False
        if Scalene.__profiler_base in filename:
            return False
        if Scalene.__functions_to_profile:
            if filename in Scalene.__functions_to_profile:
                if func in {fn.__code__.co_name for fn in Scalene.__functions_to_profile[filename]}:
                    return True
            return False
        try:
            resolved_filename = str(pathlib.Path(filename).resolve())
        except OSError:
            return False
        if not Scalene.__args.profile_all:
            for n in sysconfig.get_scheme_names():
                for p in sysconfig.get_path_names():
                    libdir = str(pathlib.Path(sysconfig.get_path(p, n)).resolve())
                    if libdir in resolved_filename:
                        return False
        profile_exclude_list = Scalene.__args.profile_exclude.split(',')
        if any((prof in filename for prof in profile_exclude_list if prof != '')):
            return False
        if filename.startswith('_ipython-input-'):
            import IPython
            if (result := re.match('_ipython-input-([0-9]+)-.*', filename)):
                cell_contents = IPython.get_ipython().history_manager.input_hist_raw[int(result[1])]
                with open(filename, 'w+') as f:
                    f.write(cell_contents)
                return True
        profile_only_set = set(Scalene.__args.profile_only.split(','))
        if profile_only_set and all((prof not in filename for prof in profile_only_set)):
            return False
        if filename[0] == '<' and filename[-1] == '>':
            return False
        if Scalene.__args.profile_all:
            return True
        filename = Filename(os.path.normpath(os.path.join(Scalene.__program_path, filename)))
        return Scalene.__program_path in filename
    __done = False

    @staticmethod
    def start() -> None:
        if False:
            i = 10
            return i + 15
        'Initiate profiling.'
        if not Scalene.__initialized:
            print('ERROR: Do not try to invoke `start` if you have not called Scalene using one of the methods\nin https://github.com/plasma-umass/scalene#using-scalene\n(The most likely issue is that you need to run your code with `scalene`, not `python`).')
            sys.exit(1)
        Scalene.__stats.start_clock()
        Scalene.enable_signals()
        Scalene.__start_time = time.monotonic_ns()
        Scalene.__done = False

    @staticmethod
    def stop() -> None:
        if False:
            for i in range(10):
                print('nop')
        'Complete profiling.'
        Scalene.__done = True
        Scalene.disable_signals()
        Scalene.__stats.stop_clock()
        if Scalene.__args.outfile:
            Scalene.__profile_filename = os.path.join(os.path.dirname(Scalene.__args.outfile), os.path.basename(Scalene.__profile_filename))
        if Scalene.__args.web and (not Scalene.__args.cli) and (not Scalene.__is_child):
            try:
                if not find_browser():
                    Scalene.__args.web = False
                else:
                    Scalene.__args.json = True
                    Scalene.__output.html = False
                    Scalene.__output.output_file = Scalene.__profile_filename
            except Exception:
                Scalene.__args.web = False
            if Scalene.__args.web and Scalene.in_jupyter():
                Scalene.__args.json = True
                Scalene.__output.html = False
                Scalene.__output.output_file = Scalene.__profile_filename

    @staticmethod
    def is_done() -> bool:
        if False:
            i = 10
            return i + 15
        'Return true if Scalene has stopped profiling.'
        return Scalene.__done

    @staticmethod
    def start_signal_handler(_signum: Union[Callable[[signal.Signals, FrameType], None], int, signal.Handlers, None], _this_frame: Optional[FrameType]) -> None:
        if False:
            print('Hello World!')
        'Respond to a signal to start or resume profiling (--on).\n\n        See scalene_parseargs.py.\n        '
        for pid in Scalene.child_pids:
            Scalene.__orig_kill(pid, Scalene.__signals.start_profiling_signal)
        Scalene.start()

    @staticmethod
    def stop_signal_handler(_signum: Union[Callable[[signal.Signals, FrameType], None], int, signal.Handlers, None], _this_frame: Optional[FrameType]) -> None:
        if False:
            while True:
                i = 10
        'Respond to a signal to suspend profiling (--off).\n\n        See scalene_parseargs.py.\n        '
        for pid in Scalene.child_pids:
            Scalene.__orig_kill(pid, Scalene.__signals.stop_profiling_signal)
        Scalene.stop()
        if Scalene.__output.output_file:
            Scalene.output_profile(sys.argv)

    @staticmethod
    def disable_signals(retry: bool=True) -> None:
        if False:
            print('Hello World!')
        'Turn off the profiling signals.'
        if sys.platform == 'win32':
            Scalene.timer_signals = False
            return
        try:
            Scalene.__orig_setitimer(Scalene.__signals.cpu_timer_signal, 0)
            Scalene.__orig_signal(Scalene.__signals.malloc_signal, signal.SIG_IGN)
            Scalene.__orig_signal(Scalene.__signals.free_signal, signal.SIG_IGN)
            Scalene.__orig_signal(Scalene.__signals.memcpy_signal, signal.SIG_IGN)
            Scalene.stop_signal_queues()
        except Exception:
            if retry:
                Scalene.disable_signals(retry=False)

    @staticmethod
    def exit_handler() -> None:
        if False:
            while True:
                i = 10
        'When we exit, disable all signals.'
        Scalene.disable_signals()
        with contextlib.suppress(Exception):
            if not Scalene.__pid:
                Scalene.__python_alias_dir.cleanup()
        with contextlib.suppress(Exception):
            os.remove(f'/tmp/scalene-malloc-lock{os.getpid()}')

    @staticmethod
    def generate_html(profile_fname: Filename, output_fname: Filename) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Apply a template to generate a single HTML payload containing the current profile.'
        try:
            profile_file = pathlib.Path(profile_fname)
            profile = profile_file.read_text()
        except FileNotFoundError:
            return
        scalene_dir = os.path.dirname(__file__)
        gui_fname = os.path.join(scalene_dir, 'scalene-gui', 'scalene-gui.js')
        gui_file = pathlib.Path(gui_fname)
        gui_js = gui_file.read_text()
        environment = Environment(loader=FileSystemLoader(os.path.join(scalene_dir, 'scalene-gui')))
        template = environment.get_template('index.html.template')
        rendered_content = template.render(profile=profile, gui_js=gui_js, scalene_version=scalene_version, scalene_date=scalene_date)
        try:
            with open(output_fname, 'w', encoding='utf-8') as f:
                f.write(rendered_content)
        except OSError:
            pass

    def profile_code(self, code: str, the_globals: Dict[str, str], the_locals: Dict[str, str], left: List[str]) -> int:
        if False:
            while True:
                i = 10
        'Initiate execution and profiling.'
        if Scalene.__args.memory:
            from scalene import pywhere
            pywhere.populate_struct()
        if 'off' not in Scalene.__args or not Scalene.__args.off:
            self.start()
        exit_status = 0
        try:
            exec(code, the_globals, the_locals)
        except SystemExit as se:
            exit_status = se.code
        except KeyboardInterrupt:
            print('Scalene execution interrupted.')
        except Exception as e:
            print(f'{Scalene.__error_message}:\n', e)
            traceback.print_exc()
            exit_status = 1
        finally:
            self.stop()
            if Scalene.__args.memory:
                pywhere.disable_settrace()
                pywhere.depopulate_struct()
            stats = Scalene.__stats
            (last_file, last_line, _) = Scalene.__last_profiled
            stats.memory_malloc_count[last_file][last_line] += 1
            stats.memory_aggregate_footprint[last_file][last_line] += stats.memory_current_highwater_mark[last_file][last_line]
            did_output = Scalene.output_profile(left)
            if not did_output:
                print('Scalene: Program did not run for long enough to profile.')
            if not (did_output and Scalene.__args.web and (not Scalene.__args.cli) and (not Scalene.__is_child)):
                return exit_status
            Scalene.generate_html(profile_fname=Scalene.__profile_filename, output_fname=Scalene.__args.outfile if Scalene.__args.outfile else Scalene.__profiler_html)
            if Scalene.in_jupyter():
                from scalene.scalene_jupyter import ScaleneJupyter
                port = ScaleneJupyter.find_available_port(8181, 9000)
                if not port:
                    print('Scalene error: could not find an available port.')
                else:
                    ScaleneJupyter.display_profile(port, Scalene.__profiler_html)
            elif not Scalene.__args.no_browser:
                old_dyld = os.environ.pop('DYLD_INSERT_LIBRARIES', '')
                old_ld = os.environ.pop('LD_PRELOAD', '')
                if Scalene.__args.outfile:
                    output_fname = Scalene.__args.outfile
                else:
                    output_fname = f'{os.getcwd()}/{Scalene.__profiler_html}'
                if Scalene.__pid == 0:
                    webbrowser.open(f'file:///{output_fname}')
                os.environ.update({'DYLD_INSERT_LIBRARIES': old_dyld, 'LD_PRELOAD': old_ld})
        return exit_status

    @staticmethod
    def process_args(args: argparse.Namespace) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Process all arguments.'
        Scalene.__args = cast(ScaleneArguments, args)
        Scalene.__next_output_time = time.perf_counter() + Scalene.__args.profile_interval
        Scalene.__output.html = args.html
        if args.outfile:
            Scalene.__output.output_file = os.path.abspath(os.path.expanduser(args.outfile))
        Scalene.__is_child = args.pid != 0
        Scalene.__parent_pid = args.pid if Scalene.__is_child else os.getpid()
        if not Scalene.__args.gpu:
            Scalene.__output.gpu = False
            Scalene.__json.gpu = False

    @staticmethod
    def set_initialized() -> None:
        if False:
            print('Hello World!')
        'Indicate that Scalene has been initialized and is ready to begin profiling.'
        Scalene.__initialized = True

    @staticmethod
    def main() -> None:
        if False:
            i = 10
            return i + 15
        'Initialize and profile.'
        (args, left) = ScaleneParseArgs.parse_args()
        Scalene.set_initialized()
        Scalene.run_profiler(args, left)

    @staticmethod
    def run_profiler(args: argparse.Namespace, left: List[str], is_jupyter: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Set up and initiate profiling.'
        if is_jupyter:
            Scalene.set_in_jupyter()
        if not Scalene.__initialized:
            print('ERROR: Do not try to manually invoke `run_profiler`.\nTo invoke Scalene programmatically, see the usage noted in https://github.com/plasma-umass/scalene#using-scalene')
            sys.exit(1)
        if sys.platform != 'win32':
            Scalene.__orig_signal(Scalene.__signals.start_profiling_signal, Scalene.start_signal_handler)
            Scalene.__orig_signal(Scalene.__signals.stop_profiling_signal, Scalene.stop_signal_handler)
            Scalene.__orig_siginterrupt(Scalene.__signals.start_profiling_signal, False)
            Scalene.__orig_siginterrupt(Scalene.__signals.stop_profiling_signal, False)
        Scalene.__orig_signal(signal.SIGINT, Scalene.interruption_handler)
        did_preload = False if is_jupyter else ScalenePreload.setup_preload(args)
        if not did_preload:
            with contextlib.suppress(Exception):
                if os.getpgrp() != os.tcgetpgrp(sys.stdout.fileno()):
                    print(f'Scalene now profiling process {os.getpid()}')
                    print(f'  to disable profiling: python3 -m scalene.profile --off --pid {os.getpid()}')
                    print(f'  to resume profiling:  python3 -m scalene.profile --on  --pid {os.getpid()}')
        Scalene.__stats.clear_all()
        sys.argv = left
        with contextlib.suppress(Exception):
            if not is_jupyter:
                multiprocessing.set_start_method('fork')
        spec = None
        try:
            Scalene.process_args(args)
            progs = None
            exit_status = 0
            try:
                if len(sys.argv) >= 2 and sys.argv[0] == '-m':
                    module = True
                    (_, mod_name, *sys.argv) = sys.argv
                    (_, spec, _) = _get_module_details(mod_name)
                    if not spec.origin:
                        raise FileNotFoundError
                    sys.argv.insert(0, spec.origin)
                else:
                    module = False
                progs = [x for x in sys.argv if re.match('.*\\.py$', x)]
                with contextlib.suppress(Exception):
                    progs.extend((sys.argv[0], __file__))
                if not progs:
                    raise FileNotFoundError
                prog_name = os.path.abspath(os.path.expanduser(progs[0]))
                with open(prog_name, 'r', encoding='utf-8') as prog_being_profiled:
                    code: Any = ''
                    try:
                        code = compile(prog_being_profiled.read(), prog_name, 'exec')
                    except SyntaxError:
                        traceback.print_exc()
                        sys.exit(1)
                    program_path = Filename(os.path.dirname(prog_name))
                    if not module:
                        sys.path.insert(0, program_path)
                        Scalene.__entrypoint_dir = program_path
                    if len(args.program_path) > 0:
                        Scalene.__program_path = Filename(os.path.abspath(args.program_path))
                    else:
                        Scalene.__program_path = program_path
                    if Scalene.__args.memory:
                        from scalene import pywhere
                        pywhere.register_files_to_profile(list(Scalene.__files_to_profile), Scalene.__program_path, Scalene.__args.profile_all)
                    import __main__
                    the_locals = __main__.__dict__
                    the_globals = __main__.__dict__
                    the_globals['__file__'] = prog_name
                    the_globals['__spec__'] = None
                    if spec is not None:
                        name = spec.name
                        the_globals['__package__'] = name.split('.')[0]
                    gc.collect()
                    profiler = Scalene(args, Filename(prog_name))
                    try:
                        exit_status = profiler.profile_code(code, the_locals, the_globals, left)
                        if not is_jupyter:
                            sys.exit(exit_status)
                    except StopJupyterExecution:
                        pass
                    except AttributeError:
                        raise
                    except Exception as ex:
                        template = 'Scalene: An exception of type {0} occurred. Arguments:\n{1!r}'
                        message = template.format(type(ex).__name__, ex.args)
                        print(message)
                        print(traceback.format_exc())
            except (FileNotFoundError, IOError):
                if progs:
                    print(f'Scalene: could not find input file {prog_name}')
                else:
                    print('Scalene: no input file specified.')
                sys.exit(1)
        except SystemExit as e:
            exit_status = e.code
        except StopJupyterExecution:
            pass
        except Exception:
            print('Scalene failed to initialize.\n' + traceback.format_exc())
            sys.exit(1)
        finally:
            with contextlib.suppress(Exception):
                Scalene.__malloc_mapfile.close()
                Scalene.__memcpy_mapfile.close()
                if not Scalene.__is_child:
                    Scalene.__malloc_mapfile.cleanup()
                    Scalene.__memcpy_mapfile.cleanup()
            if not is_jupyter:
                sys.exit(exit_status)
if __name__ == '__main__':
    Scalene.main()