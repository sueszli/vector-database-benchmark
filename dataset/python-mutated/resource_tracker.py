import os
import shutil
import sys
import signal
import warnings
import threading
from _multiprocessing import sem_unlink
from multiprocessing import util
from . import spawn
if sys.platform == 'win32':
    import _winapi
    import msvcrt
    from multiprocessing.reduction import duplicate
__all__ = ['ensure_running', 'register', 'unregister']
_HAVE_SIGMASK = hasattr(signal, 'pthread_sigmask')
_IGNORED_SIGNALS = (signal.SIGINT, signal.SIGTERM)
_CLEANUP_FUNCS = {'folder': shutil.rmtree, 'file': os.unlink}
if os.name == 'posix':
    _CLEANUP_FUNCS['semlock'] = sem_unlink
VERBOSE = False

class ResourceTracker:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._lock = threading.Lock()
        self._fd = None
        self._pid = None

    def getfd(self):
        if False:
            return 10
        self.ensure_running()
        return self._fd

    def ensure_running(self):
        if False:
            for i in range(10):
                print('nop')
        'Make sure that resource tracker process is running.\n\n        This can be run from any process.  Usually a child process will use\n        the resource created by its parent.'
        with self._lock:
            if self._fd is not None:
                if self._check_alive():
                    return
                os.close(self._fd)
                if os.name == 'posix':
                    try:
                        os.waitpid(self._pid, 0)
                    except OSError:
                        pass
                self._fd = None
                self._pid = None
                warnings.warn('resource_tracker: process died unexpectedly, relaunching.  Some folders/sempahores might leak.')
            fds_to_pass = []
            try:
                fds_to_pass.append(sys.stderr.fileno())
            except Exception:
                pass
            (r, w) = os.pipe()
            if sys.platform == 'win32':
                _r = duplicate(msvcrt.get_osfhandle(r), inheritable=True)
                os.close(r)
                r = _r
            cmd = f'from {main.__module__} import main; main({r}, {VERBOSE})'
            try:
                fds_to_pass.append(r)
                exe = spawn.get_executable()
                args = [exe, *util._args_from_interpreter_flags(), '-c', cmd]
                util.debug(f'launching resource tracker: {args}')
                try:
                    if _HAVE_SIGMASK:
                        signal.pthread_sigmask(signal.SIG_BLOCK, _IGNORED_SIGNALS)
                    pid = spawnv_passfds(exe, args, fds_to_pass)
                finally:
                    if _HAVE_SIGMASK:
                        signal.pthread_sigmask(signal.SIG_UNBLOCK, _IGNORED_SIGNALS)
            except BaseException:
                os.close(w)
                raise
            else:
                self._fd = w
                self._pid = pid
            finally:
                if sys.platform == 'win32':
                    _winapi.CloseHandle(r)
                else:
                    os.close(r)

    def _check_alive(self):
        if False:
            for i in range(10):
                print('nop')
        'Check for the existence of the resource tracker process.'
        try:
            self._send('PROBE', '', '')
        except BrokenPipeError:
            return False
        else:
            return True

    def register(self, name, rtype):
        if False:
            return 10
        'Register a named resource, and increment its refcount.'
        self.ensure_running()
        self._send('REGISTER', name, rtype)

    def unregister(self, name, rtype):
        if False:
            i = 10
            return i + 15
        'Unregister a named resource with resource tracker.'
        self.ensure_running()
        self._send('UNREGISTER', name, rtype)

    def maybe_unlink(self, name, rtype):
        if False:
            for i in range(10):
                print('nop')
        'Decrement the refcount of a resource, and delete it if it hits 0'
        self.ensure_running()
        self._send('MAYBE_UNLINK', name, rtype)

    def _send(self, cmd, name, rtype):
        if False:
            i = 10
            return i + 15
        if len(name) > 512:
            raise ValueError('name too long')
        msg = f'{cmd}:{name}:{rtype}\n'.encode('ascii')
        nbytes = os.write(self._fd, msg)
        assert nbytes == len(msg)
_resource_tracker = ResourceTracker()
ensure_running = _resource_tracker.ensure_running
register = _resource_tracker.register
maybe_unlink = _resource_tracker.maybe_unlink
unregister = _resource_tracker.unregister
getfd = _resource_tracker.getfd

def main(fd, verbose=0):
    if False:
        return 10
    'Run resource tracker.'
    if verbose:
        util.log_to_stderr(level=util.DEBUG)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if _HAVE_SIGMASK:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _IGNORED_SIGNALS)
    for f in (sys.stdin, sys.stdout):
        try:
            f.close()
        except Exception:
            pass
    if verbose:
        util.debug('Main resource tracker is running')
    registry = {rtype: {} for rtype in _CLEANUP_FUNCS.keys()}
    try:
        if sys.platform == 'win32':
            fd = msvcrt.open_osfhandle(fd, os.O_RDONLY)
        with open(fd, 'rb') as f:
            while True:
                line = f.readline()
                if line == b'':
                    break
                try:
                    splitted = line.strip().decode('ascii').split(':')
                    (cmd, name, rtype) = (splitted[0], ':'.join(splitted[1:-1]), splitted[-1])
                    if cmd == 'PROBE':
                        continue
                    if rtype not in _CLEANUP_FUNCS:
                        raise ValueError(f'Cannot register {name} for automatic cleanup: unknown resource type ({rtype}). Resource type should be one of the following: {list(_CLEANUP_FUNCS.keys())}')
                    if cmd == 'REGISTER':
                        if name not in registry[rtype]:
                            registry[rtype][name] = 1
                        else:
                            registry[rtype][name] += 1
                        if verbose:
                            util.debug(f'[ResourceTracker] incremented refcount of {rtype} {name} (current {registry[rtype][name]})')
                    elif cmd == 'UNREGISTER':
                        del registry[rtype][name]
                        if verbose:
                            util.debug(f'[ResourceTracker] unregister {name} {rtype}: registry({len(registry)})')
                    elif cmd == 'MAYBE_UNLINK':
                        registry[rtype][name] -= 1
                        if verbose:
                            util.debug(f'[ResourceTracker] decremented refcount of {rtype} {name} (current {registry[rtype][name]})')
                        if registry[rtype][name] == 0:
                            del registry[rtype][name]
                            try:
                                if verbose:
                                    util.debug(f'[ResourceTracker] unlink {name}')
                                _CLEANUP_FUNCS[rtype](name)
                            except Exception as e:
                                warnings.warn(f'resource_tracker: {name}: {e!r}')
                    else:
                        raise RuntimeError(f'unrecognized command {cmd!r}')
                except BaseException:
                    try:
                        sys.excepthook(*sys.exc_info())
                    except BaseException:
                        pass
    finally:

        def _unlink_resources(rtype_registry, rtype):
            if False:
                while True:
                    i = 10
            if rtype_registry:
                try:
                    warnings.warn(f'resource_tracker: There appear to be {len(rtype_registry)} leaked {rtype} objects to clean up at shutdown')
                except Exception:
                    pass
            for name in rtype_registry:
                try:
                    _CLEANUP_FUNCS[rtype](name)
                    if verbose:
                        util.debug(f'[ResourceTracker] unlink {name}')
                except Exception as e:
                    warnings.warn(f'resource_tracker: {name}: {e!r}')
        for (rtype, rtype_registry) in registry.items():
            if rtype == 'folder':
                continue
            else:
                _unlink_resources(rtype_registry, rtype)
        if 'folder' in registry:
            _unlink_resources(registry['folder'], 'folder')
    if verbose:
        util.debug('resource tracker shut down')

def spawnv_passfds(path, args, passfds):
    if False:
        return 10
    passfds = sorted(passfds)
    if sys.platform != 'win32':
        (errpipe_read, errpipe_write) = os.pipe()
        try:
            from .reduction import _mk_inheritable
            from .fork_exec import fork_exec
            _pass = [_mk_inheritable(fd) for fd in passfds]
            return fork_exec(args, _pass)
        finally:
            os.close(errpipe_read)
            os.close(errpipe_write)
    else:
        cmd = ' '.join((f'"{x}"' for x in args))
        try:
            (_, ht, pid, _) = _winapi.CreateProcess(path, cmd, None, None, True, 0, None, None, None)
            _winapi.CloseHandle(ht)
        except BaseException:
            pass
        return pid