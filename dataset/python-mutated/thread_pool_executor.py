import queue
import threading
import weakref
from concurrent.futures import _base

class _WorkItem(object):

    def __init__(self, future, fn, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._future = future
        self._fn = fn
        self._fn_args = args
        self._fn_kwargs = kwargs

    def run(self):
        if False:
            while True:
                i = 10
        if self._future.set_running_or_notify_cancel():
            try:
                self._future.set_result(self._fn(*self._fn_args, **self._fn_kwargs))
            except BaseException as exc:
                self._future.set_exception(exc)

class _Worker(threading.Thread):

    def __init__(self, idle_worker_queue, work_item):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._idle_worker_queue = idle_worker_queue
        self._work_item = work_item
        self._wake_semaphore = threading.Semaphore(0)
        self._lock = threading.Lock()
        self._shutdown = False

    def run(self):
        if False:
            print('Hello World!')
        while True:
            self._work_item.run()
            self._work_item = None
            self._idle_worker_queue.put(self)
            self._wake_semaphore.acquire()
            if self._work_item is None:
                return

    def assign_work(self, work_item):
        if False:
            i = 10
            return i + 15
        'Assigns the work item and wakes up the thread.\n\n    This method must only be called while the worker is idle.\n    '
        self._work_item = work_item
        self._wake_semaphore.release()

    def shutdown(self):
        if False:
            while True:
                i = 10
        "Wakes up this thread with a 'None' work item signalling to shutdown."
        self._wake_semaphore.release()

class UnboundedThreadPoolExecutor(_base.Executor):

    def __init__(self):
        if False:
            while True:
                i = 10
        self._idle_worker_queue = queue.Queue()
        self._max_idle_threads = 16
        self._workers = weakref.WeakSet()
        self._shutdown = False
        self._lock = threading.Lock()

    def submit(self, fn, *args, **kwargs):
        if False:
            return 10
        'Attempts to submit the work item.\n\n    A runtime error is raised if the pool has been shutdown.\n    '
        future = _base.Future()
        work_item = _WorkItem(future, fn, args, kwargs)
        with self._lock:
            if self._shutdown:
                raise RuntimeError('Cannot schedule new tasks after thread pool has been shutdown.')
            try:
                self._idle_worker_queue.get(block=False).assign_work(work_item)
                if self._idle_worker_queue.qsize() > self._max_idle_threads:
                    try:
                        self._idle_worker_queue.get(block=False).shutdown()
                    except queue.Empty:
                        pass
            except queue.Empty:
                worker = _Worker(self._idle_worker_queue, work_item)
                worker.daemon = True
                worker.start()
                self._workers.add(worker)
        return future

    def shutdown(self, wait=True):
        if False:
            return 10
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            for worker in self._workers:
                worker.shutdown()
            if wait:
                for worker in self._workers:
                    worker.join()

class _SharedUnboundedThreadPoolExecutor(UnboundedThreadPoolExecutor):

    def shutdown(self, wait=True):
        if False:
            return 10
        pass
_SHARED_UNBOUNDED_THREAD_POOL_EXECUTOR = _SharedUnboundedThreadPoolExecutor()

def shared_unbounded_instance():
    if False:
        for i in range(10):
            print('nop')
    return _SHARED_UNBOUNDED_THREAD_POOL_EXECUTOR