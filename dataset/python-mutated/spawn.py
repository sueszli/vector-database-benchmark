import multiprocessing
import multiprocessing.connection
import signal
import sys
import warnings
from typing import Optional
from . import _prctl_pr_set_pdeathsig

class ProcessException(Exception):
    __slots__ = ['error_index', 'error_pid']

    def __init__(self, msg: str, error_index: int, pid: int):
        if False:
            print('Hello World!')
        super().__init__(msg)
        self.msg = msg
        self.error_index = error_index
        self.pid = pid

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
        return (type(self), (self.msg, self.error_index, self.pid))

class ProcessRaisedException(ProcessException):
    """Exception raised when a process failed due to an exception raised by the code."""

    def __init__(self, msg: str, error_index: int, error_pid: int):
        if False:
            while True:
                i = 10
        super().__init__(msg, error_index, error_pid)

class ProcessExitedException(ProcessException):
    """Exception raised when a process failed due to signal or exited with a specific code."""
    __slots__ = ['exit_code']

    def __init__(self, msg: str, error_index: int, error_pid: int, exit_code: int, signal_name: Optional[str]=None):
        if False:
            while True:
                i = 10
        super().__init__(msg, error_index, error_pid)
        self.exit_code = exit_code
        self.signal_name = signal_name

    def __reduce__(self):
        if False:
            print('Hello World!')
        return (type(self), (self.msg, self.error_index, self.pid, self.exit_code, self.signal_name))

def _wrap(fn, i, args, error_queue):
    if False:
        while True:
            i = 10
    _prctl_pr_set_pdeathsig(signal.SIGINT)
    try:
        fn(i, *args)
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback
        error_queue.put(traceback.format_exc())
        sys.exit(1)

class ProcessContext:

    def __init__(self, processes, error_queues):
        if False:
            print('Hello World!')
        self.error_queues = error_queues
        self.processes = processes
        self.sentinels = {process.sentinel: index for (index, process) in enumerate(processes)}

    def pids(self):
        if False:
            for i in range(10):
                print('nop')
        return [int(process.pid) for process in self.processes]

    def join(self, timeout=None):
        if False:
            return 10
        'Join one or more processes within spawn context.\n\n        Attempt to join one or more processes in this spawn context.\n        If one of them exited with a non-zero exit status, this function\n        kills the remaining processes and raises an exception with the cause\n        of the first process exiting.\n\n        Returns ``True`` if all processes have been joined successfully,\n        ``False`` if there are more processes that need to be joined.\n\n        Args:\n            timeout (float): Wait this long before giving up on waiting.\n        '
        if len(self.sentinels) == 0:
            return True
        ready = multiprocessing.connection.wait(self.sentinels.keys(), timeout=timeout)
        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break
        if error_index is None:
            return len(self.sentinels) == 0
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join()
        failed_process = self.processes[error_index]
        if self.error_queues[error_index].empty():
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                name = signal.Signals(-exitcode).name
                raise ProcessExitedException('process %d terminated with signal %s' % (error_index, name), error_index=error_index, error_pid=failed_process.pid, exit_code=exitcode, signal_name=name)
            else:
                raise ProcessExitedException('process %d terminated with exit code %d' % (error_index, exitcode), error_index=error_index, error_pid=failed_process.pid, exit_code=exitcode)
        original_trace = self.error_queues[error_index].get()
        msg = '\n\n-- Process %d terminated with the following error:\n' % error_index
        msg += original_trace
        raise ProcessRaisedException(msg, error_index, failed_process.pid)

class SpawnContext(ProcessContext):

    def __init__(self, processes, error_queues):
        if False:
            print('Hello World!')
        warnings.warn('SpawnContext is renamed to ProcessContext since 1.4 release.')
        super().__init__(processes, error_queues)

def start_processes(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    if False:
        for i in range(10):
            print('nop')
    mp = multiprocessing.get_context(start_method)
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(target=_wrap, args=(fn, i, args, error_queue), daemon=daemon)
        process.start()
        error_queues.append(error_queue)
        processes.append(process)
    context = ProcessContext(processes, error_queues)
    if not join:
        return context
    while not context.join():
        pass

def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    if False:
        print('Hello World!')
    "Spawns ``nprocs`` processes that run ``fn`` with ``args``.\n\n    If one of the processes exits with a non-zero exit status, the\n    remaining processes are killed and an exception is raised with the\n    cause of termination. In the case an exception was caught in the\n    child process, it is forwarded and its traceback is included in\n    the exception raised in the parent process.\n\n    Args:\n        fn (function): Function is called as the entrypoint of the\n            spawned process. This function must be defined at the top\n            level of a module so it can be pickled and spawned. This\n            is a requirement imposed by multiprocessing.\n\n            The function is called as ``fn(i, *args)``, where ``i`` is\n            the process index and ``args`` is the passed through tuple\n            of arguments.\n\n        args (tuple): Arguments passed to ``fn``.\n        nprocs (int): Number of processes to spawn.\n        join (bool): Perform a blocking join on all processes.\n        daemon (bool): The spawned processes' daemon flag. If set to True,\n                       daemonic processes will be created.\n        start_method (str): (deprecated) this method will always use ``spawn``\n                               as the start method. To use a different start method\n                               use ``start_processes()``.\n\n    Returns:\n        None if ``join`` is ``True``,\n        :class:`~ProcessContext` if ``join`` is ``False``\n\n    "
    if start_method != 'spawn':
        msg = 'This method only supports start_method=spawn (got: %s).\nTo use a different start_method use:\n\t\t torch.multiprocessing.start_processes(...)' % start_method
        warnings.warn(msg)
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')