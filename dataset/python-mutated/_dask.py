from __future__ import print_function, division, absolute_import
import asyncio
import concurrent.futures
import contextlib
import time
from uuid import uuid4
import weakref
from .parallel import parallel_config
from .parallel import AutoBatchingMixin, ParallelBackendBase
from ._utils import _TracebackCapturingWrapper, _retrieve_traceback_capturing_wrapped_call
try:
    import dask
    import distributed
except ImportError:
    dask = None
    distributed = None
if dask is not None and distributed is not None:
    from dask.utils import funcname
    from dask.sizeof import sizeof
    from dask.distributed import Client, as_completed, get_client, secede, rejoin
    from distributed.utils import thread_state
    try:
        from distributed.utils import TimeoutError as _TimeoutError
    except ImportError:
        from tornado.gen import TimeoutError as _TimeoutError

def is_weakrefable(obj):
    if False:
        print('Hello World!')
    try:
        weakref.ref(obj)
        return True
    except TypeError:
        return False

class _WeakKeyDictionary:
    """A variant of weakref.WeakKeyDictionary for unhashable objects.

    This datastructure is used to store futures for broadcasted data objects
    such as large numpy arrays or pandas dataframes that are not hashable and
    therefore cannot be used as keys of traditional python dicts.

    Furthermore using a dict with id(array) as key is not safe because the
    Python is likely to reuse id of recently collected arrays.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._data = {}

    def __getitem__(self, obj):
        if False:
            return 10
        (ref, val) = self._data[id(obj)]
        if ref() is not obj:
            raise KeyError(obj)
        return val

    def __setitem__(self, obj, value):
        if False:
            for i in range(10):
                print('nop')
        key = id(obj)
        try:
            (ref, _) = self._data[key]
            if ref() is not obj:
                raise KeyError(obj)
        except KeyError:

            def on_destroy(_):
                if False:
                    print('Hello World!')
                del self._data[key]
            ref = weakref.ref(obj, on_destroy)
        self._data[key] = (ref, value)

    def __len__(self):
        if False:
            return 10
        return len(self._data)

    def clear(self):
        if False:
            while True:
                i = 10
        self._data.clear()

def _funcname(x):
    if False:
        i = 10
        return i + 15
    try:
        if isinstance(x, list):
            x = x[0][0]
    except Exception:
        pass
    return funcname(x)

def _make_tasks_summary(tasks):
    if False:
        while True:
            i = 10
    'Summarize of list of (func, args, kwargs) function calls'
    unique_funcs = {func for (func, args, kwargs) in tasks}
    if len(unique_funcs) == 1:
        mixed = False
    else:
        mixed = True
    return (len(tasks), mixed, _funcname(tasks))

class Batch:
    """dask-compatible wrapper that executes a batch of tasks"""

    def __init__(self, tasks):
        if False:
            while True:
                i = 10
        (self._num_tasks, self._mixed, self._funcname) = _make_tasks_summary(tasks)

    def __call__(self, tasks=None):
        if False:
            for i in range(10):
                print('nop')
        results = []
        with parallel_config(backend='dask'):
            for (func, args, kwargs) in tasks:
                results.append(func(*args, **kwargs))
            return results

    def __repr__(self):
        if False:
            return 10
        descr = f'batch_of_{self._funcname}_{self._num_tasks}_calls'
        if self._mixed:
            descr = 'mixed_' + descr
        return descr

def _joblib_probe_task():
    if False:
        return 10
    pass

class DaskDistributedBackend(AutoBatchingMixin, ParallelBackendBase):
    MIN_IDEAL_BATCH_DURATION = 0.2
    MAX_IDEAL_BATCH_DURATION = 1.0
    supports_retrieve_callback = True
    default_n_jobs = -1

    def __init__(self, scheduler_host=None, scatter=None, client=None, loop=None, wait_for_workers_timeout=10, **submit_kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        if distributed is None:
            msg = "You are trying to use 'dask' as a joblib parallel backend but dask is not installed. Please install dask to fix this error."
            raise ValueError(msg)
        if client is None:
            if scheduler_host:
                client = Client(scheduler_host, loop=loop, set_as_default=False)
            else:
                try:
                    client = get_client()
                except ValueError as e:
                    msg = "To use Joblib with Dask first create a Dask Client\n\n    from dask.distributed import Client\n    client = Client()\nor\n    client = Client('scheduler-address:8786')"
                    raise ValueError(msg) from e
        self.client = client
        if scatter is not None and (not isinstance(scatter, (list, tuple))):
            raise TypeError('scatter must be a list/tuple, got `%s`' % type(scatter).__name__)
        if scatter is not None and len(scatter) > 0:
            self._scatter = list(scatter)
            scattered = self.client.scatter(scatter, broadcast=True)
            self.data_futures = {id(x): f for (x, f) in zip(scatter, scattered)}
        else:
            self._scatter = []
            self.data_futures = {}
        self.wait_for_workers_timeout = wait_for_workers_timeout
        self.submit_kwargs = submit_kwargs
        self.waiting_futures = as_completed([], loop=client.loop, with_results=True, raise_errors=False)
        self._results = {}
        self._callbacks = {}

    async def _collect(self):
        while self._continue:
            async for (future, result) in self.waiting_futures:
                cf_future = self._results.pop(future)
                callback = self._callbacks.pop(future)
                if future.status == 'error':
                    (typ, exc, tb) = result
                    cf_future.set_exception(exc)
                else:
                    cf_future.set_result(result)
                    callback(result)
            await asyncio.sleep(0.01)

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
        return (DaskDistributedBackend, ())

    def get_nested_backend(self):
        if False:
            i = 10
            return i + 15
        return (DaskDistributedBackend(client=self.client), -1)

    def configure(self, n_jobs=1, parallel=None, **backend_args):
        if False:
            for i in range(10):
                print('nop')
        self.parallel = parallel
        return self.effective_n_jobs(n_jobs)

    def start_call(self):
        if False:
            return 10
        self._continue = True
        self.client.loop.add_callback(self._collect)
        self.call_data_futures = _WeakKeyDictionary()

    def stop_call(self):
        if False:
            return 10
        self._continue = False
        time.sleep(0.01)
        self.call_data_futures.clear()

    def effective_n_jobs(self, n_jobs):
        if False:
            print('Hello World!')
        effective_n_jobs = sum(self.client.ncores().values())
        if effective_n_jobs != 0 or not self.wait_for_workers_timeout:
            return effective_n_jobs
        try:
            self.client.submit(_joblib_probe_task).result(timeout=self.wait_for_workers_timeout)
        except _TimeoutError as e:
            error_msg = "DaskDistributedBackend has no worker after {} seconds. Make sure that workers are started and can properly connect to the scheduler and increase the joblib/dask connection timeout with:\n\nparallel_config(backend='dask', wait_for_workers_timeout={})".format(self.wait_for_workers_timeout, max(10, 2 * self.wait_for_workers_timeout))
            raise TimeoutError(error_msg) from e
        return sum(self.client.ncores().values())

    async def _to_func_args(self, func):
        itemgetters = dict()
        call_data_futures = getattr(self, 'call_data_futures', None)

        async def maybe_to_futures(args):
            out = []
            for arg in args:
                arg_id = id(arg)
                if arg_id in itemgetters:
                    out.append(itemgetters[arg_id])
                    continue
                f = self.data_futures.get(arg_id, None)
                if f is None and call_data_futures is not None:
                    try:
                        f = await call_data_futures[arg]
                    except KeyError:
                        pass
                    if f is None:
                        if is_weakrefable(arg) and sizeof(arg) > 1000.0:
                            _coro = self.client.scatter(arg, asynchronous=True, hash=False)
                            t = asyncio.Task(_coro)
                            call_data_futures[arg] = t
                            f = await t
                if f is not None:
                    out.append(f)
                else:
                    out.append(arg)
            return out
        tasks = []
        for (f, args, kwargs) in func.items:
            args = list(await maybe_to_futures(args))
            kwargs = dict(zip(kwargs.keys(), await maybe_to_futures(kwargs.values())))
            tasks.append((f, args, kwargs))
        return (Batch(tasks), tasks)

    def apply_async(self, func, callback=None):
        if False:
            return 10
        cf_future = concurrent.futures.Future()
        cf_future.get = cf_future.result

        async def f(func, callback):
            (batch, tasks) = await self._to_func_args(func)
            key = f'{repr(batch)}-{uuid4().hex}'
            dask_future = self.client.submit(_TracebackCapturingWrapper(batch), tasks=tasks, key=key, **self.submit_kwargs)
            self.waiting_futures.add(dask_future)
            self._callbacks[dask_future] = callback
            self._results[dask_future] = cf_future
        self.client.loop.add_callback(f, func, callback)
        return cf_future

    def retrieve_result_callback(self, out):
        if False:
            print('Hello World!')
        return _retrieve_traceback_capturing_wrapped_call(out)

    def abort_everything(self, ensure_ready=True):
        if False:
            for i in range(10):
                print('nop')
        ' Tell the client to cancel any task submitted via this instance\n\n        joblib.Parallel will never access those results\n        '
        with self.waiting_futures.lock:
            self.waiting_futures.futures.clear()
            while not self.waiting_futures.queue.empty():
                self.waiting_futures.queue.get()

    @contextlib.contextmanager
    def retrieval_context(self):
        if False:
            while True:
                i = 10
        "Override ParallelBackendBase.retrieval_context to avoid deadlocks.\n\n        This removes thread from the worker's thread pool (using 'secede').\n        Seceding avoids deadlock in nested parallelism settings.\n        "
        if hasattr(thread_state, 'execution_state'):
            secede()
        yield
        if hasattr(thread_state, 'execution_state'):
            rejoin()