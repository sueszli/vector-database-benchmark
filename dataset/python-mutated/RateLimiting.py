from datetime import datetime, timezone
from . import Framework

class RateLimiting(Framework.TestCase):

    def testRateLimiting(self):
        if False:
            return 10
        self.assertEqual(self.g.rate_limiting, (4904, 5000))
        self.g.get_user('yurinnick')
        self.assertEqual(self.g.rate_limiting, (4903, 5000))
        self.assertEqual(self.g.rate_limiting_resettime, 1684195041)

    def testResetTime(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.g.rate_limiting_resettime, 1684195041)

    def testGetRateLimit(self):
        if False:
            return 10
        rateLimit = self.g.get_rate_limit()
        self.assertEqual(repr(rateLimit), 'RateLimit(core=Rate(reset=2023-05-15 23:57:21+00:00, remaining=4904, limit=5000))')
        self.assertEqual(repr(rateLimit.core), 'Rate(reset=2023-05-15 23:57:21+00:00, remaining=4904, limit=5000)')
        self.assertEqual(rateLimit.core.limit, 5000)
        self.assertEqual(rateLimit.core.remaining, 4904)
        self.assertEqual(rateLimit.core.used, 96)
        self.assertEqual(rateLimit.core.reset, datetime(2023, 5, 15, 23, 57, 21, tzinfo=timezone.utc))