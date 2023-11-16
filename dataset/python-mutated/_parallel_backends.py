"""
Backends for embarrassingly parallel code.
"""
import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._utils import _TracebackCapturingWrapper, _retrieve_traceback_capturing_wrapped_call
from ._multiprocessing_helpers import mp
if mp is not None:
    from .pool import MemmappingPool
    from multiprocessing.pool import ThreadPool
    from .executor import get_memmapping_executor
    from .externals.loky import process_executor, cpu_count
    from .externals.loky.process_executor import ShutdownExecutorError

class ParallelBackendBase(metaclass=ABCMeta):
    """Helper abc which defines all methods a ParallelBackend must implement"""
    supports_inner_max_num_threads = False
    supports_retrieve_callback = False
    default_n_jobs = 1

    @property
    def supports_return_generator(self):
        if False:
            i = 10
            return i + 15
        return self.supports_retrieve_callback

    @property
    def supports_timeout(self):
        if False:
            i = 10
            return i + 15
        return self.supports_retrieve_callback
    nesting_level = None

    def __init__(self, nesting_level=None, inner_max_num_threads=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.nesting_level = nesting_level
        self.inner_max_num_threads = inner_max_num_threads
    MAX_NUM_THREADS_VARS = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS']
    TBB_ENABLE_IPC_VAR = 'ENABLE_IPC'

    @abstractmethod
    def effective_n_jobs(self, n_jobs):
        if False:
            print('Hello World!')
        'Determine the number of jobs that can actually run in parallel\n\n        n_jobs is the number of workers requested by the callers. Passing\n        n_jobs=-1 means requesting all available workers for instance matching\n        the number of CPU cores on the worker host(s).\n\n        This method should return a guesstimate of the number of workers that\n        can actually perform work concurrently. The primary use case is to make\n        it possible for the caller to know in how many chunks to slice the\n        work.\n\n        In general working on larger data chunks is more efficient (less\n        scheduling overhead and better use of CPU cache prefetching heuristics)\n        as long as all the workers have enough work to do.\n        '

    @abstractmethod
    def apply_async(self, func, callback=None):
        if False:
            for i in range(10):
                print('nop')
        'Schedule a func to be run'

    def retrieve_result_callback(self, out):
        if False:
            return 10
        'Called within the callback function passed in apply_async.\n\n        The argument of this function is the argument given to a callback in\n        the considered backend. It is supposed to return the outcome of a task\n        if it succeeded or raise the exception if it failed.\n        '

    def configure(self, n_jobs=1, parallel=None, prefer=None, require=None, **backend_args):
        if False:
            i = 10
            return i + 15
        'Reconfigure the backend and return the number of workers.\n\n        This makes it possible to reuse an existing backend instance for\n        successive independent calls to Parallel with different parameters.\n        '
        self.parallel = parallel
        return self.effective_n_jobs(n_jobs)

    def start_call(self):
        if False:
            return 10
        'Call-back method called at the beginning of a Parallel call'

    def stop_call(self):
        if False:
            i = 10
            return i + 15
        'Call-back method called at the end of a Parallel call'

    def terminate(self):
        if False:
            for i in range(10):
                print('nop')
        'Shutdown the workers and free the shared memory.'

    def compute_batch_size(self):
        if False:
            for i in range(10):
                print('nop')
        'Determine the optimal batch size'
        return 1

    def batch_completed(self, batch_size, duration):
        if False:
            while True:
                i = 10
        'Callback indicate how long it took to run a batch'

    def get_exceptions(self):
        if False:
            print('Hello World!')
        'List of exception types to be captured.'
        return []

    def abort_everything(self, ensure_ready=True):
        if False:
            i = 10
            return i + 15
        'Abort any running tasks\n\n        This is called when an exception has been raised when executing a task\n        and all the remaining tasks will be ignored and can therefore be\n        aborted to spare computation resources.\n\n        If ensure_ready is True, the backend should be left in an operating\n        state as future tasks might be re-submitted via that same backend\n        instance.\n\n        If ensure_ready is False, the implementer of this method can decide\n        to leave the backend in a closed / terminated state as no new task\n        are expected to be submitted to this backend.\n\n        Setting ensure_ready to False is an optimization that can be leveraged\n        when aborting tasks via killing processes from a local process pool\n        managed by the backend it-self: if we expect no new tasks, there is no\n        point in re-creating new workers.\n        '
        pass

    def get_nested_backend(self):
        if False:
            i = 10
            return i + 15
        'Backend instance to be used by nested Parallel calls.\n\n        By default a thread-based backend is used for the first level of\n        nesting. Beyond, switch to sequential backend to avoid spawning too\n        many threads on the host.\n        '
        nesting_level = getattr(self, 'nesting_level', 0) + 1
        if nesting_level > 1:
            return (SequentialBackend(nesting_level=nesting_level), None)
        else:
            return (ThreadingBackend(nesting_level=nesting_level), None)

    @contextlib.contextmanager
    def retrieval_context(self):
        if False:
            return 10
        'Context manager to manage an execution context.\n\n        Calls to Parallel.retrieve will be made inside this context.\n\n        By default, this does nothing. It may be useful for subclasses to\n        handle nested parallelism. In particular, it may be required to avoid\n        deadlocks if a backend manages a fixed number of workers, when those\n        workers may be asked to do nested Parallel calls. Without\n        \'retrieval_context\' this could lead to deadlock, as all the workers\n        managed by the backend may be "busy" waiting for the nested parallel\n        calls to finish, but the backend has no free workers to execute those\n        tasks.\n        '
        yield

    def _prepare_worker_env(self, n_jobs):
        if False:
            for i in range(10):
                print('nop')
        'Return environment variables limiting threadpools in external libs.\n\n        This function return a dict containing environment variables to pass\n        when creating a pool of process. These environment variables limit the\n        number of threads to `n_threads` for OpenMP, MKL, Accelerated and\n        OpenBLAS libraries in the child processes.\n        '
        explicit_n_threads = self.inner_max_num_threads
        default_n_threads = max(cpu_count() // n_jobs, 1)
        env = {}
        for var in self.MAX_NUM_THREADS_VARS:
            if explicit_n_threads is None:
                var_value = os.environ.get(var, default_n_threads)
            else:
                var_value = explicit_n_threads
            env[var] = str(var_value)
        if self.TBB_ENABLE_IPC_VAR not in os.environ:
            env[self.TBB_ENABLE_IPC_VAR] = '1'
        return env

    @staticmethod
    def in_main_thread():
        if False:
            while True:
                i = 10
        return isinstance(threading.current_thread(), threading._MainThread)

class SequentialBackend(ParallelBackendBase):
    """A ParallelBackend which will execute all batches sequentially.

    Does not use/create any threading objects, and hence has minimal
    overhead. Used when n_jobs == 1.
    """
    uses_threads = True
    supports_timeout = False
    supports_retrieve_callback = False
    supports_sharedmem = True

    def effective_n_jobs(self, n_jobs):
        if False:
            return 10
        'Determine the number of jobs which are going to run in parallel'
        if n_jobs == 0:
            raise ValueError('n_jobs == 0 in Parallel has no meaning')
        return 1

    def apply_async(self, func, callback=None):
        if False:
            print('Hello World!')
        'Schedule a func to be run'
        raise RuntimeError('Should never be called for SequentialBackend.')

    def retrieve_result_callback(self, out):
        if False:
            print('Hello World!')
        raise RuntimeError('Should never be called for SequentialBackend.')

    def get_nested_backend(self):
        if False:
            i = 10
            return i + 15
        from .parallel import get_active_backend
        return get_active_backend()

class PoolManagerMixin(object):
    """A helper class for managing pool of workers."""
    _pool = None

    def effective_n_jobs(self, n_jobs):
        if False:
            i = 10
            return i + 15
        'Determine the number of jobs which are going to run in parallel'
        if n_jobs == 0:
            raise ValueError('n_jobs == 0 in Parallel has no meaning')
        elif mp is None or n_jobs is None:
            return 1
        elif n_jobs < 0:
            n_jobs = max(cpu_count() + 1 + n_jobs, 1)
        return n_jobs

    def terminate(self):
        if False:
            print('Hello World!')
        'Shutdown the process or thread pool'
        if self._pool is not None:
            self._pool.close()
            self._pool.terminate()
            self._pool = None

    def _get_pool(self):
        if False:
            i = 10
            return i + 15
        'Used by apply_async to make it possible to implement lazy init'
        return self._pool

    def apply_async(self, func, callback=None):
        if False:
            while True:
                i = 10
        'Schedule a func to be run'
        return self._get_pool().apply_async(_TracebackCapturingWrapper(func), (), callback=callback, error_callback=callback)

    def retrieve_result_callback(self, out):
        if False:
            i = 10
            return i + 15
        'Mimic concurrent.futures results, raising an error if needed.'
        return _retrieve_traceback_capturing_wrapped_call(out)

    def abort_everything(self, ensure_ready=True):
        if False:
            while True:
                i = 10
        'Shutdown the pool and restart a new one with the same parameters'
        self.terminate()
        if ensure_ready:
            self.configure(n_jobs=self.parallel.n_jobs, parallel=self.parallel, **self.parallel._backend_args)

class AutoBatchingMixin(object):
    """A helper class for automagically batching jobs."""
    MIN_IDEAL_BATCH_DURATION = 0.2
    MAX_IDEAL_BATCH_DURATION = 2
    _DEFAULT_EFFECTIVE_BATCH_SIZE = 1
    _DEFAULT_SMOOTHED_BATCH_DURATION = 0.0

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self._effective_batch_size = self._DEFAULT_EFFECTIVE_BATCH_SIZE
        self._smoothed_batch_duration = self._DEFAULT_SMOOTHED_BATCH_DURATION

    def compute_batch_size(self):
        if False:
            for i in range(10):
                print('nop')
        'Determine the optimal batch size'
        old_batch_size = self._effective_batch_size
        batch_duration = self._smoothed_batch_duration
        if batch_duration > 0 and batch_duration < self.MIN_IDEAL_BATCH_DURATION:
            ideal_batch_size = int(old_batch_size * self.MIN_IDEAL_BATCH_DURATION / batch_duration)
            ideal_batch_size *= 2
            batch_size = min(2 * old_batch_size, ideal_batch_size)
            batch_size = max(batch_size, 1)
            self._effective_batch_size = batch_size
            if self.parallel.verbose >= 10:
                self.parallel._print(f'Batch computation too fast ({batch_duration}s.) Setting batch_size={batch_size}.')
        elif batch_duration > self.MAX_IDEAL_BATCH_DURATION and old_batch_size >= 2:
            ideal_batch_size = int(old_batch_size * self.MIN_IDEAL_BATCH_DURATION / batch_duration)
            batch_size = max(2 * ideal_batch_size, 1)
            self._effective_batch_size = batch_size
            if self.parallel.verbose >= 10:
                self.parallel._print(f'Batch computation too slow ({batch_duration}s.) Setting batch_size={batch_size}.')
        else:
            batch_size = old_batch_size
        if batch_size != old_batch_size:
            self._smoothed_batch_duration = self._DEFAULT_SMOOTHED_BATCH_DURATION
        return batch_size

    def batch_completed(self, batch_size, duration):
        if False:
            print('Hello World!')
        'Callback indicate how long it took to run a batch'
        if batch_size == self._effective_batch_size:
            old_duration = self._smoothed_batch_duration
            if old_duration == self._DEFAULT_SMOOTHED_BATCH_DURATION:
                new_duration = duration
            else:
                new_duration = 0.8 * old_duration + 0.2 * duration
            self._smoothed_batch_duration = new_duration

    def reset_batch_stats(self):
        if False:
            return 10
        'Reset batch statistics to default values.\n\n        This avoids interferences with future jobs.\n        '
        self._effective_batch_size = self._DEFAULT_EFFECTIVE_BATCH_SIZE
        self._smoothed_batch_duration = self._DEFAULT_SMOOTHED_BATCH_DURATION

class ThreadingBackend(PoolManagerMixin, ParallelBackendBase):
    """A ParallelBackend which will use a thread pool to execute batches in.

    This is a low-overhead backend but it suffers from the Python Global
    Interpreter Lock if the called function relies a lot on Python objects.
    Mostly useful when the execution bottleneck is a compiled extension that
    explicitly releases the GIL (for instance a Cython loop wrapped in a "with
    nogil" block or an expensive call to a library such as NumPy).

    The actual thread pool is lazily initialized: the actual thread pool
    construction is delayed to the first call to apply_async.

    ThreadingBackend is used as the default backend for nested calls.
    """
    supports_retrieve_callback = True
    uses_threads = True
    supports_sharedmem = True

    def configure(self, n_jobs=1, parallel=None, **backend_args):
        if False:
            for i in range(10):
                print('nop')
        'Build a process or thread pool and return the number of workers'
        n_jobs = self.effective_n_jobs(n_jobs)
        if n_jobs == 1:
            raise FallbackToBackend(SequentialBackend(nesting_level=self.nesting_level))
        self.parallel = parallel
        self._n_jobs = n_jobs
        return n_jobs

    def _get_pool(self):
        if False:
            for i in range(10):
                print('nop')
        'Lazily initialize the thread pool\n\n        The actual pool of worker threads is only initialized at the first\n        call to apply_async.\n        '
        if self._pool is None:
            self._pool = ThreadPool(self._n_jobs)
        return self._pool

class MultiprocessingBackend(PoolManagerMixin, AutoBatchingMixin, ParallelBackendBase):
    """A ParallelBackend which will use a multiprocessing.Pool.

    Will introduce some communication and memory overhead when exchanging
    input and output data with the with the worker Python processes.
    However, does not suffer from the Python Global Interpreter Lock.
    """
    supports_retrieve_callback = True
    supports_return_generator = False

    def effective_n_jobs(self, n_jobs):
        if False:
            for i in range(10):
                print('nop')
        'Determine the number of jobs which are going to run in parallel.\n\n        This also checks if we are attempting to create a nested parallel\n        loop.\n        '
        if mp is None:
            return 1
        if mp.current_process().daemon:
            if n_jobs != 1:
                if inside_dask_worker():
                    msg = "Inside a Dask worker with daemon=True, setting n_jobs=1.\nPossible work-arounds:\n- dask.config.set({'distributed.worker.daemon': False})- set the environment variable DASK_DISTRIBUTED__WORKER__DAEMON=False\nbefore creating your Dask cluster."
                else:
                    msg = 'Multiprocessing-backed parallel loops cannot be nested, setting n_jobs=1'
                warnings.warn(msg, stacklevel=3)
            return 1
        if process_executor._CURRENT_DEPTH > 0:
            if n_jobs != 1:
                warnings.warn('Multiprocessing-backed parallel loops cannot be nested, below loky, setting n_jobs=1', stacklevel=3)
            return 1
        elif not (self.in_main_thread() or self.nesting_level == 0):
            if n_jobs != 1:
                warnings.warn('Multiprocessing-backed parallel loops cannot be nested below threads, setting n_jobs=1', stacklevel=3)
            return 1
        return super(MultiprocessingBackend, self).effective_n_jobs(n_jobs)

    def configure(self, n_jobs=1, parallel=None, prefer=None, require=None, **memmappingpool_args):
        if False:
            while True:
                i = 10
        'Build a process or thread pool and return the number of workers'
        n_jobs = self.effective_n_jobs(n_jobs)
        if n_jobs == 1:
            raise FallbackToBackend(SequentialBackend(nesting_level=self.nesting_level))
        gc.collect()
        self._pool = MemmappingPool(n_jobs, **memmappingpool_args)
        self.parallel = parallel
        return n_jobs

    def terminate(self):
        if False:
            for i in range(10):
                print('nop')
        'Shutdown the process or thread pool'
        super(MultiprocessingBackend, self).terminate()
        self.reset_batch_stats()

class LokyBackend(AutoBatchingMixin, ParallelBackendBase):
    """Managing pool of workers with loky instead of multiprocessing."""
    supports_retrieve_callback = True
    supports_inner_max_num_threads = True

    def configure(self, n_jobs=1, parallel=None, prefer=None, require=None, idle_worker_timeout=300, **memmappingexecutor_args):
        if False:
            i = 10
            return i + 15
        'Build a process executor and return the number of workers'
        n_jobs = self.effective_n_jobs(n_jobs)
        if n_jobs == 1:
            raise FallbackToBackend(SequentialBackend(nesting_level=self.nesting_level))
        self._workers = get_memmapping_executor(n_jobs, timeout=idle_worker_timeout, env=self._prepare_worker_env(n_jobs=n_jobs), context_id=parallel._id, **memmappingexecutor_args)
        self.parallel = parallel
        return n_jobs

    def effective_n_jobs(self, n_jobs):
        if False:
            while True:
                i = 10
        'Determine the number of jobs which are going to run in parallel'
        if n_jobs == 0:
            raise ValueError('n_jobs == 0 in Parallel has no meaning')
        elif mp is None or n_jobs is None:
            return 1
        elif mp.current_process().daemon:
            if n_jobs != 1:
                if inside_dask_worker():
                    msg = "Inside a Dask worker with daemon=True, setting n_jobs=1.\nPossible work-arounds:\n- dask.config.set({'distributed.worker.daemon': False})\n- set the environment variable DASK_DISTRIBUTED__WORKER__DAEMON=False\nbefore creating your Dask cluster."
                else:
                    msg = 'Loky-backed parallel loops cannot be called in a multiprocessing, setting n_jobs=1'
                warnings.warn(msg, stacklevel=3)
            return 1
        elif not (self.in_main_thread() or self.nesting_level == 0):
            if n_jobs != 1:
                warnings.warn('Loky-backed parallel loops cannot be nested below threads, setting n_jobs=1', stacklevel=3)
            return 1
        elif n_jobs < 0:
            n_jobs = max(cpu_count() + 1 + n_jobs, 1)
        return n_jobs

    def apply_async(self, func, callback=None):
        if False:
            for i in range(10):
                print('nop')
        'Schedule a func to be run'
        future = self._workers.submit(func)
        if callback is not None:
            future.add_done_callback(callback)
        return future

    def retrieve_result_callback(self, out):
        if False:
            return 10
        try:
            return out.result()
        except ShutdownExecutorError:
            raise RuntimeError("The executor underlying Parallel has been shutdown. This is likely due to the garbage collection of a previous generator from a call to Parallel with return_as='generator'. Make sure the generator is not garbage collected when submitting a new job or that it is first properly exhausted.")

    def terminate(self):
        if False:
            return 10
        if self._workers is not None:
            self._workers._temp_folder_manager._clean_temporary_resources(context_id=self.parallel._id, force=False)
            self._workers = None
        self.reset_batch_stats()

    def abort_everything(self, ensure_ready=True):
        if False:
            for i in range(10):
                print('nop')
        'Shutdown the workers and restart a new one with the same parameters\n        '
        self._workers.terminate(kill_workers=True)
        self._workers = None
        if ensure_ready:
            self.configure(n_jobs=self.parallel.n_jobs, parallel=self.parallel)

class FallbackToBackend(Exception):
    """Raised when configuration should fallback to another backend"""

    def __init__(self, backend):
        if False:
            return 10
        self.backend = backend

def inside_dask_worker():
    if False:
        for i in range(10):
            print('nop')
    'Check whether the current function is executed inside a Dask worker.\n    '
    try:
        from distributed import get_worker
    except ImportError:
        return False
    try:
        get_worker()
        return True
    except ValueError:
        return False