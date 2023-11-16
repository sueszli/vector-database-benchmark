from typing import Optional
from twisted.internet import defer
from twisted.internet.defer import Deferred
from synapse.config.homeserver import HomeServerConfig
from synapse.config.ratelimiting import FederationRatelimitSettings
from synapse.util.ratelimitutils import FederationRateLimiter
from tests.server import ThreadedMemoryReactorClock, get_clock
from tests.unittest import TestCase
from tests.utils import default_config

class FederationRateLimiterTestCase(TestCase):

    def test_ratelimit(self) -> None:
        if False:
            return 10
        'A simple test with the default values'
        (reactor, clock) = get_clock()
        rc_config = build_rc_config()
        ratelimiter = FederationRateLimiter(clock, rc_config)
        with ratelimiter.ratelimit('testhost') as d1:
            self.successResultOf(d1)

    def test_concurrent_limit(self) -> None:
        if False:
            return 10
        'Test what happens when we hit the concurrent limit'
        (reactor, clock) = get_clock()
        rc_config = build_rc_config({'rc_federation': {'concurrent': 2}})
        ratelimiter = FederationRateLimiter(clock, rc_config)
        with ratelimiter.ratelimit('testhost') as d1:
            self.successResultOf(d1)
            cm2 = ratelimiter.ratelimit('testhost')
            d2 = cm2.__enter__()
            self.successResultOf(d2)
            cm3 = ratelimiter.ratelimit('testhost')
            d3 = cm3.__enter__()
            self.assertNoResult(d3)
            cm2.__exit__(None, None, None)
            reactor.advance(0.0)
            self.successResultOf(d3)

    def test_sleep_limit(self) -> None:
        if False:
            return 10
        'Test what happens when we hit the sleep limit'
        (reactor, clock) = get_clock()
        rc_config = build_rc_config({'rc_federation': {'sleep_limit': 2, 'sleep_delay': 500}})
        ratelimiter = FederationRateLimiter(clock, rc_config)
        with ratelimiter.ratelimit('testhost') as d1:
            self.successResultOf(d1)
        with ratelimiter.ratelimit('testhost') as d2:
            self.successResultOf(d2)
        with ratelimiter.ratelimit('testhost') as d3:
            self.assertNoResult(d3)
            sleep_time = _await_resolution(reactor, d3)
            self.assertAlmostEqual(sleep_time, 500, places=3)

    def test_lots_of_queued_things(self) -> None:
        if False:
            print('Hello World!')
        'Tests lots of synchronous things queued up behind a slow thing.\n\n        The stack should *not* explode when the slow thing completes.\n        '
        (reactor, clock) = get_clock()
        rc_config = build_rc_config({'rc_federation': {'sleep_limit': 1000000000, 'reject_limit': 1000000000, 'concurrent': 1}})
        ratelimiter = FederationRateLimiter(clock, rc_config)
        with ratelimiter.ratelimit('testhost') as d:
            self.successResultOf(d)

            async def task() -> None:
                with ratelimiter.ratelimit('testhost') as d:
                    await d
            for _ in range(1, 100):
                defer.ensureDeferred(task())
            last_task = defer.ensureDeferred(task())
        reactor.advance(0.0)
        self.successResultOf(last_task)

def _await_resolution(reactor: ThreadedMemoryReactorClock, d: Deferred) -> float:
    if False:
        return 10
    'advance the clock until the deferred completes.\n\n    Returns the number of milliseconds it took to complete.\n    '
    start_time = reactor.seconds()
    while not d.called:
        reactor.advance(0.01)
    return (reactor.seconds() - start_time) * 1000

def build_rc_config(settings: Optional[dict]=None) -> FederationRatelimitSettings:
    if False:
        while True:
            i = 10
    config_dict = default_config('test')
    config_dict.update(settings or {})
    config = HomeServerConfig()
    config.parse_config_dict(config_dict, '', '')
    return config.ratelimiting.rc_federation