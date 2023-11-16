from __future__ import annotations
import ctypes
import ctypes.util
import sys
import traceback
from functools import partial
from itertools import count
from threading import Lock, Thread
from typing import Any, Callable, Generic, TypeVar
import outcome
RetT = TypeVar('RetT')

def _to_os_thread_name(name: str) -> bytes:
    if False:
        print('Hello World!')
    return name.encode('ascii', errors='replace')[:15]

def get_os_thread_name_func() -> Callable[[int | None, str], None] | None:
    if False:
        print('Hello World!')

    def namefunc(setname: Callable[[int, bytes], int], ident: int | None, name: str) -> None:
        if False:
            i = 10
            return i + 15
        if ident is not None:
            setname(ident, _to_os_thread_name(name))

    def darwin_namefunc(setname: Callable[[bytes], int], ident: int | None, name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        if ident is not None:
            setname(_to_os_thread_name(name))
    libpthread_path = ctypes.util.find_library('pthread')
    if not libpthread_path:
        return None
    try:
        libpthread = ctypes.CDLL(libpthread_path)
    except Exception:
        return None
    pthread_setname_np = getattr(libpthread, 'pthread_setname_np', None)
    if pthread_setname_np is None:
        return None
    pthread_setname_np.restype = ctypes.c_int
    if sys.platform == 'darwin':
        pthread_setname_np.argtypes = [ctypes.c_char_p]
        return partial(darwin_namefunc, pthread_setname_np)
    pthread_setname_np.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    return partial(namefunc, pthread_setname_np)
set_os_thread_name = get_os_thread_name_func()
IDLE_TIMEOUT = 10
name_counter = count()

class WorkerThread(Generic[RetT]):

    def __init__(self, thread_cache: ThreadCache) -> None:
        if False:
            return 10
        self._job: tuple[Callable[[], RetT], Callable[[outcome.Outcome[RetT]], object], str | None] | None = None
        self._thread_cache = thread_cache
        self._worker_lock = Lock()
        self._worker_lock.acquire()
        self._default_name = f'Trio thread {next(name_counter)}'
        self._thread = Thread(target=self._work, name=self._default_name, daemon=True)
        if set_os_thread_name:
            set_os_thread_name(self._thread.ident, self._default_name)
        self._thread.start()

    def _handle_job(self) -> None:
        if False:
            while True:
                i = 10
        assert self._job is not None
        (fn, deliver, name) = self._job
        self._job = None
        if name is not None:
            self._thread.name = name
            if set_os_thread_name:
                set_os_thread_name(self._thread.ident, name)
        result = outcome.capture(fn)
        if name is not None:
            self._thread.name = self._default_name
            if set_os_thread_name:
                set_os_thread_name(self._thread.ident, self._default_name)
        self._thread_cache._idle_workers[self] = None
        try:
            deliver(result)
        except BaseException as e:
            print('Exception while delivering result of thread', file=sys.stderr)
            traceback.print_exception(type(e), e, e.__traceback__)

    def _work(self) -> None:
        if False:
            while True:
                i = 10
        while True:
            if self._worker_lock.acquire(timeout=IDLE_TIMEOUT):
                self._handle_job()
            else:
                try:
                    del self._thread_cache._idle_workers[self]
                except KeyError:
                    continue
                else:
                    return

class ThreadCache:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self._idle_workers: dict[WorkerThread[Any], None] = {}

    def start_thread_soon(self, fn: Callable[[], RetT], deliver: Callable[[outcome.Outcome[RetT]], object], name: str | None=None) -> None:
        if False:
            while True:
                i = 10
        worker: WorkerThread[RetT]
        try:
            (worker, _) = self._idle_workers.popitem()
        except KeyError:
            worker = WorkerThread(self)
        worker._job = (fn, deliver, name)
        worker._worker_lock.release()
THREAD_CACHE = ThreadCache()

def start_thread_soon(fn: Callable[[], RetT], deliver: Callable[[outcome.Outcome[RetT]], object], name: str | None=None) -> None:
    if False:
        i = 10
        return i + 15
    "Runs ``deliver(outcome.capture(fn))`` in a worker thread.\n\n    Generally ``fn`` does some blocking work, and ``deliver`` delivers the\n    result back to whoever is interested.\n\n    This is a low-level, no-frills interface, very similar to using\n    `threading.Thread` to spawn a thread directly. The main difference is\n    that this function tries to reuse threads when possible, so it can be\n    a bit faster than `threading.Thread`.\n\n    Worker threads have the `~threading.Thread.daemon` flag set, which means\n    that if your main thread exits, worker threads will automatically be\n    killed. If you want to make sure that your ``fn`` runs to completion, then\n    you should make sure that the main thread remains alive until ``deliver``\n    is called.\n\n    It is safe to call this function simultaneously from multiple threads.\n\n    Args:\n\n        fn (sync function): Performs arbitrary blocking work.\n\n        deliver (sync function): Takes the `outcome.Outcome` of ``fn``, and\n          delivers it. *Must not block.*\n\n    Because worker threads are cached and reused for multiple calls, neither\n    function should mutate thread-level state, like `threading.local` objects\n    – or if they do, they should be careful to revert their changes before\n    returning.\n\n    Note:\n\n        The split between ``fn`` and ``deliver`` serves two purposes. First,\n        it's convenient, since most callers need something like this anyway.\n\n        Second, it avoids a small race condition that could cause too many\n        threads to be spawned. Consider a program that wants to run several\n        jobs sequentially on a thread, so the main thread submits a job, waits\n        for it to finish, submits another job, etc. In theory, this program\n        should only need one worker thread. But what could happen is:\n\n        1. Worker thread: First job finishes, and calls ``deliver``.\n\n        2. Main thread: receives notification that the job finished, and calls\n           ``start_thread_soon``.\n\n        3. Main thread: sees that no worker threads are marked idle, so spawns\n           a second worker thread.\n\n        4. Original worker thread: marks itself as idle.\n\n        To avoid this, threads mark themselves as idle *before* calling\n        ``deliver``.\n\n        Is this potential extra thread a major problem? Maybe not, but it's\n        easy enough to avoid, and we figure that if the user is trying to\n        limit how many threads they're using then it's polite to respect that.\n\n    "
    THREAD_CACHE.start_thread_soon(fn, deliver, name)