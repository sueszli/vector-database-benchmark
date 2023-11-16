__license__ = 'GPL v3'
__copyright__ = '2012, Kovid Goyal <kovid@kovidgoyal.net>'
__docformat__ = 'restructuredtext en'
import importlib
import os
import time
import traceback
from multiprocessing import Pipe
from threading import Thread
from calibre.constants import iswindows
from calibre.utils.ipc import eintr_retry_call
from calibre.utils.ipc.launch import Worker
from calibre.utils.monotonic import monotonic
from polyglot.builtins import environ_item, string_or_bytes
if iswindows:
    from multiprocessing.connection import PipeConnection as Connection
else:
    from multiprocessing.connection import Connection

class WorkerError(Exception):

    def __init__(self, msg, orig_tb='', log_path=None):
        if False:
            while True:
                i = 10
        Exception.__init__(self, msg)
        self.orig_tb = orig_tb
        self.log_path = log_path

class ConnectedWorker(Thread):

    def __init__(self, conn, args):
        if False:
            return 10
        Thread.__init__(self)
        self.daemon = True
        self.conn = conn
        self.args = args
        self.accepted = False
        self.tb = None
        self.res = None

    def run(self):
        if False:
            i = 10
            return i + 15
        self.accepted = True
        conn = self.conn
        with conn:
            try:
                eintr_retry_call(conn.send, self.args)
                self.res = eintr_retry_call(conn.recv)
            except BaseException:
                self.tb = traceback.format_exc()

class OffloadWorker:

    def __init__(self, conn, worker):
        if False:
            while True:
                i = 10
        self.conn = conn
        self.worker = worker
        self.kill_thread = t = Thread(target=self.worker.kill)
        t.daemon = True

    def __call__(self, module, func, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        eintr_retry_call(self.conn.send, (module, func, args, kwargs))
        return eintr_retry_call(self.conn.recv)

    def shutdown(self):
        if False:
            return 10
        try:
            eintr_retry_call(self.conn.send, None)
        except OSError:
            pass
        except:
            import traceback
            traceback.print_exc()
        finally:
            self.conn = None
            try:
                os.remove(self.worker.log_path)
            except:
                pass
            self.kill_thread.start()

    def is_alive(self):
        if False:
            return 10
        return self.worker.is_alive or self.kill_thread.is_alive()

def communicate(ans, worker, conn, args, timeout=300, heartbeat=None, abort=None):
    if False:
        while True:
            i = 10
    cw = ConnectedWorker(conn, args)
    cw.start()
    st = monotonic()
    check_heartbeat = callable(heartbeat)
    while worker.is_alive and cw.is_alive():
        cw.join(0.01)
        delta = monotonic() - st
        if not cw.accepted and delta > min(10, timeout):
            break
        hung = not heartbeat() if check_heartbeat else delta > timeout
        if hung:
            raise WorkerError('Worker appears to have hung')
        if abort is not None and abort.is_set():
            return
    if not cw.accepted:
        if not cw.tb:
            raise WorkerError('Failed to connect to worker process')
        raise WorkerError('Failed to connect to worker process', cw.tb)
    if cw.tb:
        raise WorkerError('Failed to communicate with worker process', cw.tb)
    if cw.res is None:
        raise WorkerError('Something strange happened. The worker process was aborted without an exception.')
    if cw.res.get('tb', None):
        raise WorkerError('Worker failed', cw.res['tb'])
    ans['result'] = cw.res['result']

def create_worker(env, priority='normal', cwd=None, func='main'):
    if False:
        while True:
            i = 10
    env = dict(env)
    (a, b) = Pipe()
    with a:
        env.update({'CALIBRE_WORKER_FD': str(a.fileno()), 'CALIBRE_SIMPLE_WORKER': environ_item('calibre.utils.ipc.simple_worker:%s' % func)})
        w = Worker(env)
        w(cwd=cwd, priority=priority, pass_fds=(a.fileno(),))
    return (b, w)

def start_pipe_worker(command, env=None, priority='normal', **process_args):
    if False:
        return 10
    import subprocess
    w = Worker(env or {})
    args = {'stdout': subprocess.PIPE, 'stdin': subprocess.PIPE, 'env': w.env, 'close_fds': True}
    args.update(process_args)
    pass_fds = None
    try:
        if iswindows:
            priority = {'high': subprocess.HIGH_PRIORITY_CLASS, 'normal': subprocess.NORMAL_PRIORITY_CLASS, 'low': subprocess.IDLE_PRIORITY_CLASS}[priority]
            args['creationflags'] = subprocess.CREATE_NO_WINDOW | priority
            pass_fds = args.pop('pass_fds', None)
            if pass_fds:
                for fd in pass_fds:
                    os.set_handle_inheritable(fd, True)
                args['startupinfo'] = subprocess.STARTUPINFO(lpAttributeList={'handle_list': pass_fds})
        else:
            niceness = {'normal': 0, 'low': 10, 'high': 20}[priority]
            args['env']['CALIBRE_WORKER_NICENESS'] = str(niceness)
        exe = w.executable
        cmd = [exe] if isinstance(exe, string_or_bytes) else exe
        p = subprocess.Popen(cmd + ['--pipe-worker', command], **args)
    finally:
        if iswindows and pass_fds:
            for fd in pass_fds:
                os.set_handle_inheritable(fd, False)
    return p

def two_part_fork_job(env=None, priority='normal', cwd=None):
    if False:
        return 10
    env = env or {}
    (conn, w) = create_worker(env, priority, cwd)

    def run_job(mod_name, func_name, args=(), kwargs=None, timeout=300, no_output=False, heartbeat=None, abort=None, module_is_source_code=False):
        if False:
            for i in range(10):
                print('nop')
        ans = {'result': None, 'stdout_stderr': None}
        kwargs = kwargs or {}
        try:
            communicate(ans, w, conn, (mod_name, func_name, args, kwargs, module_is_source_code), timeout=timeout, heartbeat=heartbeat, abort=abort)
        except WorkerError as e:
            if not no_output:
                e.log_path = w.log_path
            raise
        finally:
            t = Thread(target=w.kill)
            t.daemon = True
            t.start()
            if no_output:
                try:
                    os.remove(w.log_path)
                except:
                    pass
        if not no_output:
            ans['stdout_stderr'] = w.log_path
        return ans
    run_job.worker = w
    return run_job

def fork_job(mod_name, func_name, args=(), kwargs=None, timeout=300, cwd=None, priority='normal', env={}, no_output=False, heartbeat=None, abort=None, module_is_source_code=False):
    if False:
        print('Hello World!')
    '\n    Run a job in a worker process. A job is simply a function that will be\n    called with the supplied arguments, in the worker process.\n    The result of the function will be returned.\n    If an error occurs a WorkerError is raised.\n\n    :param mod_name: Module to import in the worker process\n\n    :param func_name: Function to call in the worker process from the imported\n    module\n\n    :param args: Positional arguments to pass to the function\n\n    :param kwargs: Keyword arguments to pass to the function\n\n    :param timeout: The time in seconds to wait for the worker process to\n    complete. If it takes longer a WorkerError is raised and the process is\n    killed.\n\n    :param cwd: The working directory for the worker process. I recommend\n    against using this, unless you are sure the path is pure ASCII.\n\n    :param priority: The process priority for the worker process\n\n    :param env: Extra environment variables to set for the worker process\n\n    :param no_output: If True, the stdout and stderr of the worker process are\n    discarded\n\n    :param heartbeat: If not None, it is used to check if the worker has hung,\n    instead of a simple timeout. It must be a callable that takes no\n    arguments and returns True or False. The worker will be assumed to have\n    hung if this function returns False. At that point, the process will be\n    killed and a WorkerError will be raised.\n\n    :param abort: If not None, it must be an Event. As soon as abort.is_set()\n    returns True, the worker process is killed. No error is raised.\n\n    :param module_is_source_code: If True, the ``mod`` is treated as python\n    source rather than a module name to import. The source is executed as a\n    module. Useful if you want to use fork_job from within a script to run some\n    dynamically generated python.\n\n    :return: A dictionary with the keys result and stdout_stderr. result is the\n    return value of the function (it must be picklable). stdout_stderr is the\n    path to a file that contains the stdout and stderr of the worker process.\n    If you set no_output=True, then this will not be present.\n    '
    return two_part_fork_job(env, priority, cwd)(mod_name, func_name, args=args, kwargs=kwargs, timeout=timeout, no_output=no_output, heartbeat=heartbeat, abort=abort, module_is_source_code=module_is_source_code)

def offload_worker(env={}, priority='normal', cwd=None):
    if False:
        while True:
            i = 10
    (conn, w) = create_worker(env=env, priority=priority, cwd=cwd, func='offload')
    return OffloadWorker(conn, w)

def compile_code(src):
    if False:
        while True:
            i = 10
    import io
    import re
    if not isinstance(src, str):
        match = re.search(b'coding[:=]\\s*([-\\w.]+)', src[:200])
        enc = match.group(1).decode('utf-8') if match else 'utf-8'
        src = src.decode(enc)
    src = re.sub('^#.*coding\\s*[:=]\\s*([-\\w.]+)', '#', src, flags=re.MULTILINE)
    src = io.StringIO(src, newline=None).getvalue()
    namespace = {'time': time, 're': re, 'os': os, 'io': io}
    exec(src, namespace)
    return namespace

def main():
    if False:
        while True:
            i = 10
    with Connection(int(os.environ['CALIBRE_WORKER_FD'])) as conn:
        args = eintr_retry_call(conn.recv)
        try:
            (mod, func, args, kwargs, module_is_source_code) = args
            if module_is_source_code:
                importlib.import_module('calibre.customize.ui')
                mod = compile_code(mod)
                func = mod[func]
            else:
                try:
                    mod = importlib.import_module(mod)
                except ImportError:
                    importlib.import_module('calibre.customize.ui')
                    mod = importlib.import_module(mod)
                func = getattr(mod, func)
            res = {'result': func(*args, **kwargs)}
        except:
            res = {'tb': traceback.format_exc()}
        try:
            conn.send(res)
        except:
            conn.send(res)

def offload():
    if False:
        while True:
            i = 10
    func_cache = {}
    with Connection(int(os.environ['CALIBRE_WORKER_FD'])) as conn:
        while True:
            args = eintr_retry_call(conn.recv)
            if args is None:
                break
            res = {'result': None, 'tb': None}
            try:
                (mod, func, args, kwargs) = args
                if mod is None:
                    eintr_retry_call(conn.send, res)
                    continue
                f = func_cache.get((mod, func), None)
                if f is None:
                    try:
                        m = importlib.import_module(mod)
                    except ImportError:
                        importlib.import_module('calibre.customize.ui')
                        m = importlib.import_module(mod)
                    func_cache[mod, func] = f = getattr(m, func)
                res['result'] = f(*args, **kwargs)
            except:
                import traceback
                res['tb'] = traceback.format_exc()
            eintr_retry_call(conn.send, res)