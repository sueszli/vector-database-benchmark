import collections
import contextlib
import logging
import threading
import typing
from typing import Any, Callable, ContextManager, DefaultDict, Dict, Iterator, List, Mapping, MutableSet, Optional, Set, Tuple
from weakref import WeakSet
from prometheus_client.core import Counter
from twisted.internet import defer
from synapse.api.errors import LimitExceededError
from synapse.config.ratelimiting import FederationRatelimitSettings
from synapse.logging.context import PreserveLoggingContext, make_deferred_yieldable, run_in_background
from synapse.logging.opentracing import start_active_span
from synapse.metrics import Histogram, LaterGauge
from synapse.util import Clock
if typing.TYPE_CHECKING:
    from contextlib import _GeneratorContextManager
logger = logging.getLogger(__name__)
rate_limit_sleep_counter = Counter('synapse_rate_limit_sleep', 'Number of requests slept by the rate limiter', ['rate_limiter_name'])
rate_limit_reject_counter = Counter('synapse_rate_limit_reject', 'Number of requests rejected by the rate limiter', ['rate_limiter_name'])
queue_wait_timer = Histogram('synapse_rate_limit_queue_wait_time_seconds', 'Amount of time spent waiting for the rate limiter to let our request through.', ['rate_limiter_name'], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0, 20.0, '+Inf'))
_rate_limiter_instances: MutableSet['FederationRateLimiter'] = WeakSet()
_rate_limiter_instances_lock = threading.Lock()

def _get_counts_from_rate_limiter_instance(count_func: Callable[['FederationRateLimiter'], int]) -> Mapping[Tuple[str, ...], int]:
    if False:
        i = 10
        return i + 15
    'Returns a count of something (slept/rejected hosts) by (metrics_name)'
    with _rate_limiter_instances_lock:
        rate_limiter_instances = list(_rate_limiter_instances)
    counts: Dict[Tuple[str, ...], int] = {}
    for rate_limiter_instance in rate_limiter_instances:
        if rate_limiter_instance.metrics_name:
            key = (rate_limiter_instance.metrics_name,)
            counts[key] = count_func(rate_limiter_instance)
    return counts
LaterGauge('synapse_rate_limit_sleep_affected_hosts', 'Number of hosts that had requests put to sleep', ['rate_limiter_name'], lambda : _get_counts_from_rate_limiter_instance(lambda rate_limiter_instance: sum((ratelimiter.should_sleep() for ratelimiter in rate_limiter_instance.ratelimiters.values()))))
LaterGauge('synapse_rate_limit_reject_affected_hosts', 'Number of hosts that had requests rejected', ['rate_limiter_name'], lambda : _get_counts_from_rate_limiter_instance(lambda rate_limiter_instance: sum((ratelimiter.should_reject() for ratelimiter in rate_limiter_instance.ratelimiters.values()))))

class FederationRateLimiter:
    """Used to rate limit request per-host."""

    def __init__(self, clock: Clock, config: FederationRatelimitSettings, metrics_name: Optional[str]=None):
        if False:
            return 10
        "\n        Args:\n            clock\n            config\n            metrics_name: The name of the rate limiter so we can differentiate it\n                from the rest in the metrics. If `None`, we don't track metrics\n                for this rate limiter.\n\n        "
        self.metrics_name = metrics_name

        def new_limiter() -> '_PerHostRatelimiter':
            if False:
                print('Hello World!')
            return _PerHostRatelimiter(clock=clock, config=config, metrics_name=metrics_name)
        self.ratelimiters: DefaultDict[str, '_PerHostRatelimiter'] = collections.defaultdict(new_limiter)
        with _rate_limiter_instances_lock:
            _rate_limiter_instances.add(self)

    def ratelimit(self, host: str) -> '_GeneratorContextManager[defer.Deferred[None]]':
        if False:
            while True:
                i = 10
        'Used to ratelimit an incoming request from a given host\n\n        Example usage:\n\n            with rate_limiter.ratelimit(origin) as wait_deferred:\n                yield wait_deferred\n                # Handle request ...\n\n        Args:\n            host: Origin of incoming request.\n\n        Returns:\n            context manager which returns a deferred.\n        '
        return self.ratelimiters[host].ratelimit(host)

class _PerHostRatelimiter:

    def __init__(self, clock: Clock, config: FederationRatelimitSettings, metrics_name: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        "\n        Args:\n            clock\n            config\n            metrics_name: The name of the rate limiter so we can differentiate it\n                from the rest in the metrics. If `None`, we don't track metrics\n                for this rate limiter.\n                from the rest in the metrics\n        "
        self.clock = clock
        self.metrics_name = metrics_name
        self.window_size = config.window_size
        self.sleep_limit = config.sleep_limit
        self.sleep_sec = config.sleep_delay / 1000.0
        self.reject_limit = config.reject_limit
        self.concurrent_requests = config.concurrent
        self.sleeping_requests: Set[object] = set()
        self.ready_request_queue: collections.OrderedDict[object, defer.Deferred[None]] = collections.OrderedDict()
        self.current_processing: Set[object] = set()
        self.request_times: List[int] = []

    @contextlib.contextmanager
    def ratelimit(self, host: str) -> 'Iterator[defer.Deferred[None]]':
        if False:
            while True:
                i = 10
        self.host = host
        request_id = object()
        ret = defer.ensureDeferred(self._on_enter_with_tracing(request_id))
        try:
            yield ret
        finally:
            self._on_exit(request_id)

    def should_reject(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Whether to reject the request if we already have too many queued up\n        (either sleeping or in the ready queue).\n        '
        queue_size = len(self.ready_request_queue) + len(self.sleeping_requests)
        return queue_size > self.reject_limit

    def should_sleep(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Whether to sleep the request if we already have too many requests coming\n        through within the window.\n        '
        return len(self.request_times) > self.sleep_limit

    async def _on_enter_with_tracing(self, request_id: object) -> None:
        maybe_metrics_cm: ContextManager = contextlib.nullcontext()
        if self.metrics_name:
            maybe_metrics_cm = queue_wait_timer.labels(self.metrics_name).time()
        with start_active_span('ratelimit wait'), maybe_metrics_cm:
            await self._on_enter(request_id)

    def _on_enter(self, request_id: object) -> 'defer.Deferred[None]':
        if False:
            while True:
                i = 10
        time_now = self.clock.time_msec()
        self.request_times[:] = [r for r in self.request_times if time_now - r < self.window_size]
        if self.should_reject():
            logger.debug('Ratelimiter(%s): rejecting request', self.host)
            if self.metrics_name:
                rate_limit_reject_counter.labels(self.metrics_name).inc()
            raise LimitExceededError(limiter_name='rc_federation', retry_after_ms=int(self.window_size / self.sleep_limit))
        self.request_times.append(time_now)

        def queue_request() -> 'defer.Deferred[None]':
            if False:
                i = 10
                return i + 15
            if len(self.current_processing) >= self.concurrent_requests:
                queue_defer: defer.Deferred[None] = defer.Deferred()
                self.ready_request_queue[request_id] = queue_defer
                logger.info('Ratelimiter(%s): queueing request (queue now %i items)', self.host, len(self.ready_request_queue))
                return queue_defer
            else:
                return defer.succeed(None)
        logger.debug('Ratelimit(%s) [%s]: len(self.request_times)=%d', self.host, id(request_id), len(self.request_times))
        if self.should_sleep():
            logger.debug('Ratelimiter(%s) [%s]: sleeping request for %f sec', self.host, id(request_id), self.sleep_sec)
            if self.metrics_name:
                rate_limit_sleep_counter.labels(self.metrics_name).inc()
            ret_defer = run_in_background(self.clock.sleep, self.sleep_sec)
            self.sleeping_requests.add(request_id)

            def on_wait_finished(_: Any) -> 'defer.Deferred[None]':
                if False:
                    print('Hello World!')
                logger.debug('Ratelimit(%s) [%s]: Finished sleeping', self.host, id(request_id))
                self.sleeping_requests.discard(request_id)
                queue_defer = queue_request()
                return queue_defer
            ret_defer.addBoth(on_wait_finished)
        else:
            ret_defer = queue_request()

        def on_start(r: object) -> object:
            if False:
                while True:
                    i = 10
            logger.debug('Ratelimit(%s) [%s]: Processing req', self.host, id(request_id))
            self.current_processing.add(request_id)
            return r

        def on_err(r: object) -> object:
            if False:
                print('Hello World!')
            self.current_processing.discard(request_id)
            return r

        def on_both(r: object) -> object:
            if False:
                i = 10
                return i + 15
            self.sleeping_requests.discard(request_id)
            self.ready_request_queue.pop(request_id, None)
            return r
        ret_defer.addCallbacks(on_start, on_err)
        ret_defer.addBoth(on_both)
        return make_deferred_yieldable(ret_defer)

    def _on_exit(self, request_id: object) -> None:
        if False:
            return 10
        logger.debug('Ratelimit(%s) [%s]: Processed req', self.host, id(request_id))

        def start_next_request() -> None:
            if False:
                print('Hello World!')
            self.current_processing.discard(request_id)
            try:
                (_, deferred) = self.ready_request_queue.popitem(last=False)
                with PreserveLoggingContext():
                    deferred.callback(None)
            except KeyError:
                pass
        self.clock.call_later(0.0, start_next_request)