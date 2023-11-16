import json
import traceback
from contextvars import copy_context
from _plotly_utils.utils import PlotlyJSONEncoder
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.exceptions import PreventUpdate
from dash.long_callback.managers import BaseLongCallbackManager

class CeleryManager(BaseLongCallbackManager):
    """Manage background execution of callbacks with a celery queue."""

    def __init__(self, celery_app, cache_by=None, expire=None):
        if False:
            while True:
                i = 10
        "\n        Long callback manager that runs callback logic on a celery task queue,\n        and stores results using a celery result backend.\n\n        :param celery_app:\n            A celery.Celery application instance that must be configured with a\n            result backend. See the celery documentation for information on\n            configuration options.\n        :param cache_by:\n            A list of zero-argument functions.  When provided, caching is enabled and\n            the return values of these functions are combined with the callback\n            function's input arguments and source code to generate cache keys.\n        :param expire:\n            If provided, a cache entry will be removed when it has not been accessed\n            for ``expire`` seconds.  If not provided, the lifetime of cache entries\n            is determined by the default behavior of the celery result backend.\n        "
        try:
            import celery
            from celery.backends.base import DisabledBackend
        except ImportError as missing_imports:
            raise ImportError('CeleryLongCallbackManager requires extra dependencies which can be installed doing\n\n    $ pip install "dash[celery]"\n') from missing_imports
        if not isinstance(celery_app, celery.Celery):
            raise ValueError('First argument must be a celery.Celery object')
        if isinstance(celery_app.backend, DisabledBackend):
            raise ValueError('Celery instance must be configured with a result backend')
        self.handle = celery_app
        self.expire = expire
        super().__init__(cache_by)

    def terminate_job(self, job):
        if False:
            i = 10
            return i + 15
        if job is None:
            return
        self.handle.control.terminate(job)

    def terminate_unhealthy_job(self, job):
        if False:
            while True:
                i = 10
        task = self.get_task(job)
        if task and task.status in ('FAILURE', 'REVOKED'):
            return self.terminate_job(job)
        return False

    def job_running(self, job):
        if False:
            i = 10
            return i + 15
        future = self.get_task(job)
        return future and future.status in ('PENDING', 'RECEIVED', 'STARTED', 'RETRY', 'PROGRESS')

    def make_job_fn(self, fn, progress, key=None):
        if False:
            for i in range(10):
                print('nop')
        return _make_job_fn(fn, self.handle, progress, key)

    def get_task(self, job):
        if False:
            for i in range(10):
                print('nop')
        if job:
            return self.handle.AsyncResult(job)
        return None

    def clear_cache_entry(self, key):
        if False:
            while True:
                i = 10
        self.handle.backend.delete(key)

    def call_job_fn(self, key, job_fn, args, context):
        if False:
            while True:
                i = 10
        task = job_fn.delay(key, self._make_progress_key(key), args, context)
        return task.task_id

    def get_progress(self, key):
        if False:
            return 10
        progress_key = self._make_progress_key(key)
        progress_data = self.handle.backend.get(progress_key)
        if progress_data:
            self.handle.backend.delete(progress_key)
            return json.loads(progress_data)
        return None

    def result_ready(self, key):
        if False:
            return 10
        return self.handle.backend.get(key) is not None

    def get_result(self, key, job):
        if False:
            print('Hello World!')
        result = self.handle.backend.get(key)
        if result is None:
            return self.UNDEFINED
        result = json.loads(result)
        if self.cache_by is None:
            self.clear_cache_entry(key)
        elif self.expire:
            self.handle.backend.expire(key, self.expire)
        self.clear_cache_entry(self._make_progress_key(key))
        self.terminate_job(job)
        return result

def _make_job_fn(fn, celery_app, progress, key):
    if False:
        for i in range(10):
            print('nop')
    cache = celery_app.backend

    @celery_app.task(name=f'long_callback_{key}')
    def job_fn(result_key, progress_key, user_callback_args, context=None):
        if False:
            while True:
                i = 10

        def _set_progress(progress_value):
            if False:
                return 10
            if not isinstance(progress_value, (list, tuple)):
                progress_value = [progress_value]
            cache.set(progress_key, json.dumps(progress_value, cls=PlotlyJSONEncoder))
        maybe_progress = [_set_progress] if progress else []
        ctx = copy_context()

        def run():
            if False:
                return 10
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
                cache.set(result_key, json.dumps({'_dash_no_update': '_dash_no_update'}, cls=PlotlyJSONEncoder))
            except Exception as err:
                cache.set(result_key, json.dumps({'long_callback_error': {'msg': str(err), 'tb': traceback.format_exc()}}))
            else:
                cache.set(result_key, json.dumps(user_callback_output, cls=PlotlyJSONEncoder))
        ctx.run(run)
    return job_fn

class CeleryLongCallbackManager(CeleryManager):
    """Deprecated: use `from dash import CeleryManager` instead."""