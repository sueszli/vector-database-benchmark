from typing import Any, Optional, Sequence, Tuple
from sentry_redis_tools.clients import RedisCluster, StrictRedis
from sentry_redis_tools.sliding_windows_rate_limiter import GrantedQuota, Quota
from sentry_redis_tools.sliding_windows_rate_limiter import RedisSlidingWindowRateLimiter as RedisSlidingWindowRateLimiterImpl
from sentry_redis_tools.sliding_windows_rate_limiter import RequestedQuota, Timestamp
from sentry.exceptions import InvalidConfiguration
from sentry.utils import redis
from sentry.utils.services import Service
__all__ = ['Quota', 'GrantedQuota', 'RequestedQuota', 'Timestamp']

class SlidingWindowRateLimiter(Service):

    def __init__(self, **options: Any) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def check_within_quotas(self, requests: Sequence[RequestedQuota], timestamp: Optional[Timestamp]=None) -> Tuple[Timestamp, Sequence[GrantedQuota]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Given a set of quotas requests and limits, compute how much quota could\n        be consumed.\n\n        :param requests: The requests to return "grants" for.\n        :param timestamp: The timestamp of the incoming request. Defaults to\n            the current timestamp.\n\n            Providing a too old timestamp here _can_ effectively disable rate\n            limits, as the older request counts may no longer be stored.\n            However, consistently providing old timestamps here will work\n            correctly.\n        '
        raise NotImplementedError()

    def use_quotas(self, requests: Sequence[RequestedQuota], grants: Sequence[GrantedQuota], timestamp: Timestamp) -> None:
        if False:
            return 10
        '\n        Given a set of requests and the corresponding return values from\n        `check_within_quotas`, consume the quotas.\n\n        :param requests: The requests that have previously been passed to\n            `check_within_quotas`.\n        :param timestamp: The request timestamp that has previously been passed\n            to `check_within_quotas`.\n        :param grants: The return value of `check_within_quotas` which\n            indicates how much quota should actually be consumed.\n\n        Why is checking quotas and using quotas two separate implementations?\n        Isn\'t that a time-of-check-time-of-use bug, and allows me to over-spend\n        quota when requests are happening concurrently?\n\n        1) It\'s desirable to first check quotas, then do a potentially fallible\n           operation, then consume quotas. This rate limiter is primarily going to\n           be used inside of the metrics string indexer to rate-limit database\n           writes. What we want to do there is: read DB, check rate limits,\n           write to DB, use rate limits.\n\n           If we did it any other way (the obvious one being to read DB,\n           check-and-use rate limits, write DB), crashes while writing to the\n           database can over-consume quotas. This is not a big problem if those\n           crashes are flukes, and especially not a problem if the crashes are\n           a result of an overloaded DB.\n\n           It is however a problem in case the consumer is crash-looping, or\n           crashing (quickly) for 100% of requests (e.g. due to schema\n           mismatches between code and DB that somehow don\'t surface during the\n           DB read). In that case the quotas would be consumed immediately and\n           incident recovery would require us to reset all quotas manually (or\n           disable rate limiting via some killswitch)\n\n        3) The redis backend (really the only backend we care about) already\n           has some consistency problems.\n\n           a) Redis only provides strong consistency and ability to\n              check-and-increment counters when all involved keys hit the same\n              Redis node. That means that a quota with prefix="org_id:123" can\n              only run on a single redis node. It also means that a global\n              quota (`prefix="global"`) would have to run on a single\n              (unchangeable) redis node to be strongly consistent. That\'s\n              however a problem for scalability.\n\n              There\'s no obvious way to make global quotas consistent with\n              per-org quotas this way, so right now it means that requests\n              containing multiple quotas with different `prefixes` cannot be\n              checked-and-incremented atomically even if we were to change the\n              rate-limiter\'s interface.\n\n           b) This is easily fixable, but because of the above, we\n              currently don\'t control Redis sharding at all, meaning that even\n              keys within a single quota\'s window will hit different Redis\n              node. This also helps further distribute the load internally.\n\n              Since we have given up on atomic check-and-increments in general\n              anyway, there\'s no reason to explicitly control sharding.\n\n        '
        raise NotImplementedError()

    def check_and_use_quotas(self, requests: Sequence[RequestedQuota], timestamp: Optional[Timestamp]=None) -> Sequence[GrantedQuota]:
        if False:
            return 10
        '\n        Check the quota requests in Redis and consume the quota in one go. See\n        `check_within_quotas` for parameters.\n        '
        (timestamp, grants) = self.check_within_quotas(requests, timestamp)
        self.use_quotas(requests, grants, timestamp)
        return grants

class RedisSlidingWindowRateLimiter(SlidingWindowRateLimiter):

    def __init__(self, **options: Any) -> None:
        if False:
            i = 10
            return i + 15
        cluster_key = options.get('cluster', 'default')
        client = redis.redis_clusters.get(cluster_key)
        assert isinstance(client, (StrictRedis, RedisCluster)), client
        self.client = client
        self.impl = RedisSlidingWindowRateLimiterImpl(self.client)
        super().__init__(**options)

    def validate(self) -> None:
        if False:
            i = 10
            return i + 15
        try:
            self.client.ping()
            self.client.connection_pool.disconnect()
        except Exception as e:
            raise InvalidConfiguration(str(e))

    def check_within_quotas(self, requests: Sequence[RequestedQuota], timestamp: Optional[Timestamp]=None) -> Tuple[Timestamp, Sequence[GrantedQuota]]:
        if False:
            i = 10
            return i + 15
        return self.impl.check_within_quotas(requests, timestamp)

    def use_quotas(self, requests: Sequence[RequestedQuota], grants: Sequence[GrantedQuota], timestamp: Timestamp) -> None:
        if False:
            for i in range(10):
                print('nop')
        return self.impl.use_quotas(requests, grants, timestamp)