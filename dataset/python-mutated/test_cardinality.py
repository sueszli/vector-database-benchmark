from typing import Collection, Optional, Sequence
import pytest
from sentry.ratelimits.cardinality import GrantedQuota, Quota, RedisCardinalityLimiter, RequestedQuota

@pytest.fixture
def limiter():
    if False:
        while True:
            i = 10
    return RedisCardinalityLimiter()

class LimiterHelper:
    """
    Wrapper interface around the rate limiter, with specialized, stateful and
    primitive interface for more readable tests.
    """

    def __init__(self, limiter: RedisCardinalityLimiter):
        if False:
            i = 10
            return i + 15
        self.limiter = limiter
        self.quota = Quota(window_seconds=3600, granularity_seconds=60, limit=10)
        self.timestamp = 3600

    def add_value(self, value: int) -> Optional[int]:
        if False:
            while True:
                i = 10
        values = self.add_values([value])
        if values:
            (value,) = values
            return value
        else:
            return None

    def add_values(self, values: Sequence[int]) -> Collection[int]:
        if False:
            while True:
                i = 10
        request = RequestedQuota(prefix='hello', unit_hashes=values, quota=self.quota)
        (new_timestamp, grants) = self.limiter.check_within_quotas([request], timestamp=self.timestamp)
        self.limiter.use_quotas(grants, new_timestamp)
        (grant,) = grants
        return grant.granted_unit_hashes

def test_basic(limiter: RedisCardinalityLimiter):
    if False:
        for i in range(10):
            print('nop')
    helper = LimiterHelper(limiter)
    for _ in range(20):
        assert helper.add_value(1) == 1
    for _ in range(20):
        assert helper.add_value(2) == 2
    assert [helper.add_value(10 + i) for i in range(100)] == list(range(10, 18)) + [None] * 92
    helper.timestamp += 3600
    assert [helper.add_value(10 + i) for i in range(100)] == list(range(10, 20)) + [None] * 90

def test_multiple_prefixes(limiter: RedisCardinalityLimiter):
    if False:
        for i in range(10):
            print('nop')
    "\n    Test multiple prefixes/organizations and just make sure we're not leaking\n    state between prefixes.\n\n    * `a` only consumes 5 of the quota first and runs out of quota in the\n      second `check_within_quotas` call\n    * `b` immediately exceeds the quota.\n    * `c` fits comfortably into the quota at first (fills out the limit exactly)\n    "
    quota = Quota(window_seconds=3600, granularity_seconds=60, limit=10)
    requests = [RequestedQuota(prefix='a', unit_hashes={1, 2, 3, 4, 5}, quota=quota), RequestedQuota(prefix='b', unit_hashes={1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, quota=quota), RequestedQuota(prefix='c', unit_hashes={11, 12, 13, 14, 15, 16, 17, 18, 19, 20}, quota=quota)]
    (new_timestamp, grants) = limiter.check_within_quotas(requests)
    assert grants == [GrantedQuota(request=requests[0], granted_unit_hashes=[1, 2, 3, 4, 5], reached_quota=None), GrantedQuota(request=requests[1], granted_unit_hashes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], reached_quota=quota), GrantedQuota(request=requests[2], granted_unit_hashes=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20], reached_quota=None)]
    limiter.use_quotas(grants, new_timestamp)
    requests = [RequestedQuota(prefix='a', unit_hashes={6, 7, 8, 9, 10, 11}, quota=quota), RequestedQuota(prefix='b', unit_hashes={1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, quota=quota), RequestedQuota(prefix='c', unit_hashes={11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}, quota=quota)]
    (new_timestamp, grants) = limiter.check_within_quotas(requests)
    assert grants == [GrantedQuota(request=requests[0], granted_unit_hashes=[6, 7, 8, 9, 10], reached_quota=quota), GrantedQuota(request=requests[1], granted_unit_hashes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], reached_quota=quota), GrantedQuota(request=requests[2], granted_unit_hashes=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20], reached_quota=quota)]
    limiter.use_quotas(grants, new_timestamp)

def test_sliding(limiter: RedisCardinalityLimiter):
    if False:
        i = 10
        return i + 15
    '\n    Our rate limiter has a sliding window of [now - 1 hour ; now], with a\n    granularity of 1 hour.\n\n    What that means is that, as time moves on, old hashes should be forgotten\n    _one by one_, and the quota budget they occupy should become _gradually_\n    available to newer, never-seen-before items.\n    '
    helper = LimiterHelper(limiter)
    admissions = []
    for i in range(100):
        admissions.append(helper.add_value(i))
        helper.timestamp += 360
    assert admissions == list(range(100))
    admissions = []
    expected = []
    for i in range(100, 200):
        admissions.append(helper.add_value(i))
        expected.append(i if i % 10 == 0 else None)
        helper.timestamp += 36
    assert admissions == expected

def test_sampling(limiter: RedisCardinalityLimiter) -> None:
    if False:
        print('Hello World!')
    '\n    demonstrate behavior when "shard sampling" is active. If one out of 10\n    shards for an organization are stored, it is still possible to limit the\n    exactly correct amount of hashes, for certain hash values.\n    '
    limiter.impl.num_physical_shards = 1
    limiter.impl.num_shards = 10
    helper = LimiterHelper(limiter)
    admissions = [helper.add_value(i) for i in reversed(range(10))]
    assert admissions == list(reversed(range(10)))
    admissions = [helper.add_value(i) for i in range(100, 110)]
    assert admissions == [None] * 10

def test_sampling_going_bad(limiter: RedisCardinalityLimiter):
    if False:
        print('Hello World!')
    '\n    test an edgecase of set sampling in the cardinality limiter. it is not\n    exactly desired behavior but a known sampling artifact\n    '
    limiter.impl.num_physical_shards = 1
    limiter.impl.num_shards = 10
    helper = LimiterHelper(limiter)
    admissions = [helper.add_value(i) for i in range(10)]
    assert admissions == [0] + [None] * 9

def test_regression_mixed_order(limiter: RedisCardinalityLimiter):
    if False:
        i = 10
        return i + 15
    '\n    Regression test to assert we still accept hashes after dropping some\n    within the same request, regardless of set order.\n    '
    helper = LimiterHelper(limiter)
    assert helper.add_value(5) == 5
    assert helper.add_values([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 5]) == [0, 1, 2, 3, 4, 6, 7, 8, 9, 5]