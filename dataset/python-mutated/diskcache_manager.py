import traceback
from contextvars import copy_context
from . import BaseLongCallbackManager
from ..._callback_context import context_value
from ..._utils import AttributeDict
from ...exceptions import PreventUpdate
_pending_value = '__$pending__'

class DiskcacheManager(BaseLongCallbackManager):
    """Manage the background execution of callbacks with subprocesses and a diskcache result backend."""

    def __init__(self, cache=None, cache_by=None, expire=None):
        if False:
            print('Hello World!')
        "\n        Long callback manager that runs callback logic in a subprocess and stores\n        results on disk using diskcache\n\n        :param cache:\n            A diskcache.Cache or diskcache.FanoutCache instance. See the diskcache\n            documentation for information on configuration options. If not provided,\n            a diskcache.Cache instance will be created with default values.\n        :param cache_by:\n            A list of zero-argument functions.  When provided, caching is enabled and\n            the return values of these functions are combined with the callback\n            function's input arguments and source code to generate cache keys.\n        :param expire:\n            If provided, a cache entry will be removed when it has not been accessed\n            for ``expire`` seconds.  If not provided, the lifetime of cache entries\n            is determined by the default behavior of the ``cache`` instance.\n        "
        try:
            import diskcache
            import psutil
            import multiprocess
        except ImportError as missing_imports:
            raise ImportError('DiskcacheLongCallbackManager requires extra dependencies which can be installed doing\n\n    $ pip install "dash[diskcache]"\n') from missing_imports
        if cache is None:
            self.handle = diskcache.Cache()
        else:
            if not isinstance(cache, (diskcache.Cache, diskcache.FanoutCache)):
                raise ValueError('First argument must be a diskcache.Cache or diskcache.FanoutCache object')
            self.handle = cache
        self.expire = expire
        super().__init__(cache_by)

    def terminate_job(self, job):
        if False:
            print('Hello World!')
        import psutil
        if job is None:
            return
        job = int(job)
        with self.handle.transact():
            if psutil.pid_exists(job):
                process = psutil.Process(job)
                for proc in process.children(recursive=True):
                    try:
                        proc.kill()
                    except psutil.NoSuchProcess:
                        pass
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
                try:
                    process.wait(1)
                except (psutil.TimeoutExpired, psutil.NoSuchProcess):
                    pass

    def terminate_unhealthy_job(self, job):
        if False:
            while True:
                i = 10
        import psutil
        job = int(job)
        if job and psutil.pid_exists(job):
            if not self.job_running(job):
                self.terminate_job(job)
                return True
        return False

    def job_running(self, job):
        if False:
            while True:
                i = 10
        import psutil
        job = int(job)
        if job and psutil.pid_exists(job):
            proc = psutil.Process(job)
            return proc.status() != psutil.STATUS_ZOMBIE
        return False

    def make_job_fn(self, fn, progress, key=None):
        if False:
            i = 10
            return i + 15
        return _make_job_fn(fn, self.handle, progress)

    def clear_cache_entry(self, key):
        if False:
            i = 10
            return i + 15
        self.handle.delete(key)

    def call_job_fn(self, key, job_fn, args, context):
        if False:
            return 10
        from multiprocess import Process
        proc = Process(target=job_fn, args=(key, self._make_progress_key(key), args, context))
        proc.start()
        return proc.pid

    def get_progress(self, key):
        if False:
            i = 10
            return i + 15
        progress_key = self._make_progress_key(key)
        progress_data = self.handle.get(progress_key)
        if progress_data:
            self.handle.delete(progress_key)
        return progress_data

    def result_ready(self, key):
        if False:
            for i in range(10):
                print('nop')
        return self.handle.get(key) is not None

    def get_result(self, key, job):
        if False:
            while True:
                i = 10
        result = self.handle.get(key, self.UNDEFINED)
        if result is self.UNDEFINED:
            return self.UNDEFINED
        if self.cache_by is None:
            self.clear_cache_entry(key)
        elif self.expire:
            self.handle.touch(key, expire=self.expire)
        self.clear_cache_entry(self._make_progress_key(key))
        if job:
            self.terminate_job(job)
        return result

def _make_job_fn(fn, cache, progress):
    if False:
        return 10

    def job_fn(result_key, progress_key, user_callback_args, context):
        if False:
            i = 10
            return i + 15

        def _set_progress(progress_value):
            if False:
                for i in range(10):
                    print('nop')
            if not isinstance(progress_value, (list, tuple)):
                progress_value = [progress_value]
            cache.set(progress_key, progress_value)
        maybe_progress = [_set_progress] if progress else []
        ctx = copy_context()

        def run():
            if False:
                print('Hello World!')
            c = AttributeDict(**context)
            c.ignore_register_page = False
            context_value.set(c)
            try:
                if isinstance(user_callback_args, dict):
                    user_callback_output = fn(*maybe_progress, **user_callback_args)
                elif isinstance(user_callback_args, (list, tuple)):
                    user_callback_output = fn(*maybe_progress, *user_callback_args)
                else:
                    user_callback_output = fn(*maybe_progress, user_callback_args)
            except PreventUpdate:
                cache.set(result_key, {'_dash_no_update': '_dash_no_update'})
            except Exception as err:
                cache.set(result_key, {'long_callback_error': {'msg': str(err), 'tb': traceback.format_exc()}})
            else:
                cache.set(result_key, user_callback_output)
        ctx.run(run)
    return job_fn

class DiskcacheLongCallbackManager(DiskcacheManager):
    """Deprecated: use `from dash import DiskcacheManager` instead."""