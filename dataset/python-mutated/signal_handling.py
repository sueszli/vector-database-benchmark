"""Signal handling for multiprocessing data loading.

NOTE [ Signal handling in multiprocessing data loading ]

In cases like DataLoader, if a worker process dies due to bus error/segfault
or just hang, the main process will hang waiting for data. This is difficult
to avoid on PyTorch side as it can be caused by limited shm, or other
libraries users call in the workers. In this file and `DataLoader.cpp`, we make
our best effort to provide some error message to users when such unfortunate
events happen.

When a _BaseDataLoaderIter starts worker processes, their pids are registered in a
defined in `DataLoader.cpp`: id(_BaseDataLoaderIter) => Collection[ Worker pids ]
via `_set_worker_pids`.

When an error happens in a worker process, the main process received a SIGCHLD,
and Python will eventually call the handler registered below
(in `_set_SIGCHLD_handler`). In the handler, the `_error_if_any_worker_fails`
call checks all registered worker pids and raise proper error message to
prevent main process from hanging waiting for data from worker.

Additionally, at the beginning of each worker's `_utils.worker._worker_loop`,
`_set_worker_signal_handlers` is called to register critical signal handlers
(e.g., for SIGSEGV, SIGBUS, SIGFPE, SIGTERM) in C, which just prints an error
message to stderr before triggering the default handler. So a message will also
be printed from the worker process when it is killed by such signals.

See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for the reasoning of
this signal handling design and other mechanism we implement to make our
multiprocessing data loading robust to errors.
"""
import signal
import threading
from . import IS_WINDOWS
from torch._C import _set_worker_pids, _remove_worker_pids
from torch._C import _error_if_any_worker_fails, _set_worker_signal_handlers
_SIGCHLD_handler_set = False
'Whether SIGCHLD handler is set for DataLoader worker failures. Only one\nhandler needs to be set for all DataLoaders in a process.'

def _set_SIGCHLD_handler():
    if False:
        i = 10
        return i + 15
    if IS_WINDOWS:
        return
    if not isinstance(threading.current_thread(), threading._MainThread):
        return
    global _SIGCHLD_handler_set
    if _SIGCHLD_handler_set:
        return
    previous_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(previous_handler):
        previous_handler = None

    def handler(signum, frame):
        if False:
            return 10
        _error_if_any_worker_fails()
        if previous_handler is not None:
            assert callable(previous_handler)
            previous_handler(signum, frame)
    signal.signal(signal.SIGCHLD, handler)
    _SIGCHLD_handler_set = True