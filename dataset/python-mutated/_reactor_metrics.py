import logging
import time
from selectors import SelectSelector, _PollLikeSelector
from typing import Any, Callable, Iterable
from prometheus_client import Histogram, Metric
from prometheus_client.core import REGISTRY, GaugeMetricFamily
from twisted.internet import reactor, selectreactor
from twisted.internet.asyncioreactor import AsyncioSelectorReactor
from synapse.metrics._types import Collector
try:
    from selectors import KqueueSelector
except ImportError:

    class KqueueSelector:
        pass
try:
    from twisted.internet.epollreactor import EPollReactor
except ImportError:

    class EPollReactor:
        pass
try:
    from twisted.internet.pollreactor import PollReactor
except ImportError:

    class PollReactor:
        pass
logger = logging.getLogger(__name__)
tick_time = Histogram('python_twisted_reactor_tick_time', 'Tick time of the Twisted reactor (sec)', buckets=[0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])

class CallWrapper:
    """A wrapper for a callable which records the time between calls"""

    def __init__(self, wrapped: Callable[..., Any]):
        if False:
            i = 10
            return i + 15
        self.last_polled = time.time()
        self._wrapped = wrapped

    def __call__(self, *args, **kwargs) -> Any:
        if False:
            while True:
                i = 10
        tick_time.observe(time.time() - self.last_polled)
        ret = self._wrapped(*args, **kwargs)
        self.last_polled = time.time()
        return ret

class ObjWrapper:
    """A wrapper for an object which wraps a specified method in CallWrapper.

    Other methods/attributes are passed to the original object.

    This is necessary when the wrapped object does not allow the attribute to be
    overwritten.
    """

    def __init__(self, wrapped: Any, method_name: str):
        if False:
            for i in range(10):
                print('nop')
        self._wrapped = wrapped
        self._method_name = method_name
        self._wrapped_method = CallWrapper(getattr(wrapped, method_name))

    def __getattr__(self, item: str) -> Any:
        if False:
            return 10
        if item == self._method_name:
            return self._wrapped_method
        return getattr(self._wrapped, item)

class ReactorLastSeenMetric(Collector):

    def __init__(self, call_wrapper: CallWrapper):
        if False:
            return 10
        self._call_wrapper = call_wrapper

    def collect(self) -> Iterable[Metric]:
        if False:
            print('Hello World!')
        cm = GaugeMetricFamily('python_twisted_reactor_last_seen', 'Seconds since the Twisted reactor was last seen')
        cm.add_metric([], time.time() - self._call_wrapper.last_polled)
        yield cm
wrapper = None
try:
    if isinstance(reactor, (PollReactor, EPollReactor)):
        reactor._poller = ObjWrapper(reactor._poller, 'poll')
        wrapper = reactor._poller._wrapped_method
    elif isinstance(reactor, selectreactor.SelectReactor):
        wrapper = selectreactor._select = CallWrapper(selectreactor._select)
    elif isinstance(reactor, AsyncioSelectorReactor):
        asyncio_loop = reactor._asyncioEventloop
        selector = asyncio_loop._selector
        if isinstance(selector, SelectSelector):
            wrapper = selector._select = CallWrapper(selector._select)
        elif isinstance(selector, _PollLikeSelector):
            selector._selector = ObjWrapper(selector._selector, 'poll')
            wrapper = selector._selector._wrapped_method
        elif isinstance(selector, KqueueSelector):
            selector._selector = ObjWrapper(selector._selector, 'control')
            wrapper = selector._selector._wrapped_method
        else:
            logger.warning('Skipping configuring ReactorLastSeenMetric: unexpected asyncio loop selector: %r via %r', selector, asyncio_loop)
except Exception as e:
    logger.warning('Configuring ReactorLastSeenMetric failed: %r', e)
if wrapper:
    REGISTRY.register(ReactorLastSeenMetric(wrapper))