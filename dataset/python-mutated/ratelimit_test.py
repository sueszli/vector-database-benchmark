import unittest
from mock import patch
import pylibmc
from pylons import app_globals as g
from r2.lib import ratelimit
from r2.lib.cache import LocalCache

class RateLimitStandaloneFunctionsTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.patch('ratelimit.time.time', lambda : self.now)
        self.cache = LocalCache()
        self.patch('ratelimit.g.ratelimitcache', self.cache)

    def patch(self, *a, **kw):
        if False:
            i = 10
            return i + 15
        p = patch(*a, **kw)
        p.start()
        self.addCleanup(p.stop)

    def test_get_timeslice(self):
        if False:
            i = 10
            return i + 15
        self.now = 125
        ts = ratelimit.get_timeslice(60)
        self.assertEquals(120, ts.beginning)
        self.assertEquals(180, ts.end)
        self.assertEquals(55, ts.remaining)

    def test_make_ratelimit_cache_key_1s(self):
        if False:
            for i in range(10):
                print('nop')
        self.now = 14
        ts = ratelimit.get_timeslice(1)
        key = ratelimit._make_ratelimit_cache_key('a', ts)
        self.assertEquals('rl:a-000014', key)

    def test_make_ratelimit_cache_key_1m(self):
        if False:
            i = 10
            return i + 15
        self.now = 65
        ts = ratelimit.get_timeslice(60)
        key = ratelimit._make_ratelimit_cache_key('a', ts)
        self.assertEquals('rl:a-000100', key)

    def test_make_ratelimit_cache_key_1h(self):
        if False:
            i = 10
            return i + 15
        self.now = 3650
        ts = ratelimit.get_timeslice(3600)
        key = ratelimit._make_ratelimit_cache_key('a', ts)
        self.assertEquals('rl:a-010000', key)

    def test_make_ratelimit_cache_key_1d(self):
        if False:
            return 10
        self.now = 24 * 3600 + 5
        ts = ratelimit.get_timeslice(24 * 3600)
        key = ratelimit._make_ratelimit_cache_key('a', ts)
        self.assertEquals('rl:a-@86400', key)

    def test_make_ratelimit_cache_key_1w(self):
        if False:
            print('Hello World!')
        self.now = 7 * 24 * 3600 + 5
        ts = ratelimit.get_timeslice(24 * 3600)
        key = ratelimit._make_ratelimit_cache_key('a', ts)
        self.assertEquals('rl:a-@604800', key)

    def test_record_usage(self):
        if False:
            return 10
        self.now = 24 * 3600 + 5
        ts = ratelimit.get_timeslice(3600)
        ratelimit.record_usage('a', ts)
        self.assertEquals(1, self.cache['rl:a-000000'])
        ratelimit.record_usage('a', ts)
        self.assertEquals(2, self.cache['rl:a-000000'])
        self.now = 24 * 3600 + 5 * 3600
        ts = ratelimit.get_timeslice(3600)
        ratelimit.record_usage('a', ts)
        self.assertEquals(1, self.cache['rl:a-050000'])

    def test_record_usage_across_slice_expiration(self):
        if False:
            i = 10
            return i + 15
        self.now = 24 * 3600 + 5
        ts = ratelimit.get_timeslice(3600)
        real_incr = self.cache.incr
        evicted = False

        def fake_incr(key):
            if False:
                while True:
                    i = 10
            if evicted:
                del self.cache[key]
                raise pylibmc.NotFound()
            return real_incr(key)
        with patch.object(self.cache, 'incr', fake_incr):
            evicted = True
            ratelimit.record_usage('a', ts)
            self.assertEquals(1, self.cache['rl:a-000000'])

    def test_get_usage(self):
        if False:
            i = 10
            return i + 15
        self.now = 24 * 3600 + 5 * 3600
        ts = ratelimit.get_timeslice(3600)
        self.assertEquals(None, ratelimit.get_usage('a', ts))
        ratelimit.record_usage('a', ts)
        self.assertEquals(1, ratelimit.get_usage('a', ts))

class RateLimitTest(unittest.TestCase):

    class TestRateLimit(ratelimit.RateLimit):
        event_name = 'TestRateLimit'
        event_type = 'tests'
        key = 'tests'
        limit = 1
        seconds = 3600

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.patch('ratelimit.time.time', lambda : self.now)
        self.cache = LocalCache()
        self.patch('ratelimit.g.ratelimitcache', self.cache)

    def patch(self, *a, **kw):
        if False:
            i = 10
            return i + 15
        p = patch(*a, **kw)
        p.start()
        self.addCleanup(p.stop)

    def test_record_usage(self):
        if False:
            return 10
        rl = self.TestRateLimit()
        self.now = 24 * 3600 + 5
        rl.record_usage()
        self.assertEquals(1, self.cache['rl:tests-000000'])
        rl.record_usage()
        self.assertEquals(2, self.cache['rl:tests-000000'])
        self.now = 24 * 3600 + 5 * 3600
        rl.record_usage()
        self.assertEquals(1, self.cache['rl:tests-050000'])

    def test_get_usage(self):
        if False:
            print('Hello World!')
        rl = self.TestRateLimit()
        self.now = 24 * 3600 + 5 * 3600
        self.assertTrue(rl.check())
        rl.record_usage()
        self.assertFalse(rl.check())

class LiveConfigRateLimitTest(unittest.TestCase):

    class TestRateLimit(ratelimit.LiveConfigRateLimit):
        event_name = 'TestRateLimit'
        event_type = 'tests'
        key = 'rl-tests'
        limit_live_key = 'RL_TESTS'
        seconds_live_key = 'RL_TESTS_RESET_SECS'

    def patch_liveconfig(self, k, v):
        if False:
            for i in range(10):
                print('nop')
        'Helper method to patch g.live_config (with cleanup).'

        def cleanup(orig=g.live_config.get(k), has=k in g.live_config):
            if False:
                i = 10
                return i + 15
            if has:
                g.live_config[k] = orig
            else:
                del g.live_config[k]
        g.live_config[k] = v
        self.addCleanup(cleanup)

    def configure_rate_limit(self, num, per_unit):
        if False:
            return 10
        self.patch_liveconfig('RL_TESTS', num)
        self.patch_liveconfig('RL_TESTS_RESET_SECS', per_unit)

    def test_limit(self):
        if False:
            while True:
                i = 10
        self.configure_rate_limit(1, 3600)
        rl = self.TestRateLimit()
        self.assertEquals(1, rl.limit)
        self.configure_rate_limit(2, 3600)
        self.assertEquals(2, rl.limit)

    def test_seconds(self):
        if False:
            print('Hello World!')
        self.configure_rate_limit(1, 3600)
        rl = self.TestRateLimit()
        self.assertEquals(3600, rl.seconds)
        self.configure_rate_limit(1, 300)
        self.assertEquals(300, rl.seconds)