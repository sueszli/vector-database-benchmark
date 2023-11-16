import logging
import threading
from contextlib import nullcontext
from functools import wraps
from types import TracebackType
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Iterable, Optional, Set, Type, TypeVar, Union
from prometheus_client import Metric
from prometheus_client.core import REGISTRY, Counter, Gauge
from typing_extensions import ParamSpec
from twisted.internet import defer
from synapse.logging.context import ContextResourceUsage, LoggingContext, PreserveLoggingContext
from synapse.logging.opentracing import SynapseTags, start_active_span
from synapse.metrics._types import Collector
if TYPE_CHECKING:
    import resource
    from typing_extensions import LiteralString
logger = logging.getLogger(__name__)
_background_process_start_count = Counter('synapse_background_process_start_count', 'Number of background processes started', ['name'])
_background_process_in_flight_count = Gauge('synapse_background_process_in_flight_count', 'Number of background processes in flight', labelnames=['name'])
_background_process_ru_utime = Counter('synapse_background_process_ru_utime_seconds', 'User CPU time used by background processes, in seconds', ['name'], registry=None)
_background_process_ru_stime = Counter('synapse_background_process_ru_stime_seconds', 'System CPU time used by background processes, in seconds', ['name'], registry=None)
_background_process_db_txn_count = Counter('synapse_background_process_db_txn_count', 'Number of database transactions done by background processes', ['name'], registry=None)
_background_process_db_txn_duration = Counter('synapse_background_process_db_txn_duration_seconds', 'Seconds spent by background processes waiting for database transactions, excluding scheduling time', ['name'], registry=None)
_background_process_db_sched_duration = Counter('synapse_background_process_db_sched_duration_seconds', 'Seconds spent by background processes waiting for database connections', ['name'], registry=None)
_background_process_counts: Dict[str, int] = {}
_background_processes_active_since_last_scrape: 'Set[_BackgroundProcess]' = set()
_bg_metrics_lock = threading.Lock()

class _Collector(Collector):
    """A custom metrics collector for the background process metrics.

    Ensures that all of the metrics are up-to-date with any in-flight processes
    before they are returned.
    """

    def collect(self) -> Iterable[Metric]:
        if False:
            while True:
                i = 10
        global _background_processes_active_since_last_scrape
        with _bg_metrics_lock:
            _background_processes_copy = _background_processes_active_since_last_scrape
            _background_processes_active_since_last_scrape = set()
        for process in _background_processes_copy:
            process.update_metrics()
        for m in (_background_process_ru_utime, _background_process_ru_stime, _background_process_db_txn_count, _background_process_db_txn_duration, _background_process_db_sched_duration):
            yield from m.collect()
REGISTRY.register(_Collector())

class _BackgroundProcess:

    def __init__(self, desc: str, ctx: LoggingContext):
        if False:
            i = 10
            return i + 15
        self.desc = desc
        self._context = ctx
        self._reported_stats: Optional[ContextResourceUsage] = None

    def update_metrics(self) -> None:
        if False:
            return 10
        'Updates the metrics with values from this process.'
        new_stats = self._context.get_resource_usage()
        if self._reported_stats is None:
            diff = new_stats
        else:
            diff = new_stats - self._reported_stats
        self._reported_stats = new_stats
        _background_process_ru_utime.labels(self.desc).inc(max(diff.ru_utime, 0))
        _background_process_ru_stime.labels(self.desc).inc(max(diff.ru_stime, 0))
        _background_process_db_txn_count.labels(self.desc).inc(diff.db_txn_count)
        _background_process_db_txn_duration.labels(self.desc).inc(diff.db_txn_duration_sec)
        _background_process_db_sched_duration.labels(self.desc).inc(diff.db_sched_duration_sec)
R = TypeVar('R')

def run_as_background_process(desc: 'LiteralString', func: Callable[..., Awaitable[Optional[R]]], *args: Any, bg_start_span: bool=True, **kwargs: Any) -> 'defer.Deferred[Optional[R]]':
    if False:
        return 10
    "Run the given function in its own logcontext, with resource metrics\n\n    This should be used to wrap processes which are fired off to run in the\n    background, instead of being associated with a particular request.\n\n    It returns a Deferred which completes when the function completes, but it doesn't\n    follow the synapse logcontext rules, which makes it appropriate for passing to\n    clock.looping_call and friends (or for firing-and-forgetting in the middle of a\n    normal synapse async function).\n\n    Args:\n        desc: a description for this background process type\n        func: a function, which may return a Deferred or a coroutine\n        bg_start_span: Whether to start an opentracing span. Defaults to True.\n            Should only be disabled for processes that will not log to or tag\n            a span.\n        args: positional args for func\n        kwargs: keyword args for func\n\n    Returns:\n        Deferred which returns the result of func, or `None` if func raises.\n        Note that the returned Deferred does not follow the synapse logcontext\n        rules.\n    "

    async def run() -> Optional[R]:
        with _bg_metrics_lock:
            count = _background_process_counts.get(desc, 0)
            _background_process_counts[desc] = count + 1
        _background_process_start_count.labels(desc).inc()
        _background_process_in_flight_count.labels(desc).inc()
        with BackgroundProcessLoggingContext(desc, count) as context:
            try:
                if bg_start_span:
                    ctx = start_active_span(f'bgproc.{desc}', tags={SynapseTags.REQUEST_ID: str(context)})
                else:
                    ctx = nullcontext()
                with ctx:
                    return await func(*args, **kwargs)
            except Exception:
                logger.exception("Background process '%s' threw an exception", desc)
                return None
            finally:
                _background_process_in_flight_count.labels(desc).dec()
    with PreserveLoggingContext():
        return defer.ensureDeferred(run())
P = ParamSpec('P')

def wrap_as_background_process(desc: 'LiteralString') -> Callable[[Callable[P, Awaitable[Optional[R]]]], Callable[P, 'defer.Deferred[Optional[R]]']]:
    if False:
        while True:
            i = 10
    'Decorator that wraps an asynchronous function `func`, returning a synchronous\n    decorated function. Calling the decorated version runs `func` as a background\n    process, forwarding all arguments verbatim.\n\n    That is,\n\n        @wrap_as_background_process\n        def func(*args): ...\n        func(1, 2, third=3)\n\n    is equivalent to:\n\n        def func(*args): ...\n        run_as_background_process(func, 1, 2, third=3)\n\n    The former can be convenient if `func` needs to be run as a background process in\n    multiple places.\n    '

    def wrap_as_background_process_inner(func: Callable[P, Awaitable[Optional[R]]]) -> Callable[P, 'defer.Deferred[Optional[R]]']:
        if False:
            i = 10
            return i + 15

        @wraps(func)
        def wrap_as_background_process_inner_2(*args: P.args, **kwargs: P.kwargs) -> 'defer.Deferred[Optional[R]]':
            if False:
                while True:
                    i = 10
            return run_as_background_process(desc, func, *args, **kwargs)
        return wrap_as_background_process_inner_2
    return wrap_as_background_process_inner

class BackgroundProcessLoggingContext(LoggingContext):
    """A logging context that tracks in flight metrics for background
    processes.
    """
    __slots__ = ['_proc']

    def __init__(self, name: str, instance_id: Optional[Union[int, str]]=None):
        if False:
            return 10
        '\n\n        Args:\n            name: The name of the background process. Each distinct `name` gets a\n                separate prometheus time series.\n\n            instance_id: an identifer to add to `name` to distinguish this instance of\n                the named background process in the logs. If this is `None`, one is\n                made up based on id(self).\n        '
        if instance_id is None:
            instance_id = id(self)
        super().__init__('%s-%s' % (name, instance_id))
        self._proc: Optional[_BackgroundProcess] = _BackgroundProcess(name, self)

    def start(self, rusage: 'Optional[resource.struct_rusage]') -> None:
        if False:
            i = 10
            return i + 15
        'Log context has started running (again).'
        super().start(rusage)
        if self._proc is None:
            logger.error('Background process re-entered without a proc: %s', self.name, stack_info=True)
            return
        with _bg_metrics_lock:
            _background_processes_active_since_last_scrape.add(self._proc)

    def __exit__(self, type: Optional[Type[BaseException]], value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        if False:
            while True:
                i = 10
        'Log context has finished.'
        super().__exit__(type, value, traceback)
        if self._proc is None:
            logger.error('Background process exited without a proc: %s', self.name, stack_info=True)
            return
        with _bg_metrics_lock:
            _background_processes_active_since_last_scrape.discard(self._proc)
        self._proc.update_metrics()
        self._proc = None