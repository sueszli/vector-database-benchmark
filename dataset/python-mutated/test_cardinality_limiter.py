import time
from typing import Optional, Sequence, Tuple
import pytest
from arroyo import Partition, Topic
from sentry.ratelimits.cardinality import CardinalityLimiter, GrantedQuota, RequestedQuota, Timestamp
from sentry.sentry_metrics.configuration import UseCaseKey
from sentry.sentry_metrics.consumers.indexer.common import BrokerMeta
from sentry.sentry_metrics.indexer.limiters.cardinality import TimeseriesCardinalityLimiter, _build_quota_key
from sentry.sentry_metrics.use_case_id_registry import UseCaseID
from sentry.testutils.helpers.options import override_options

@pytest.fixture(autouse=True)
def rollout_all_orgs_generic_metrics(set_sentry_option):
    if False:
        i = 10
        return i + 15
    with set_sentry_option('sentry-metrics.cardinality-limiter.orgs-rollout-rate', 1.0):
        yield

class MockCardinalityLimiter(CardinalityLimiter):

    def __init__(self):
        if False:
            return 10
        self.grant_hashes = {}
        self.assert_requests: Optional[Sequence[RequestedQuota]] = None

    def check_within_quotas(self, requests: Sequence[RequestedQuota], timestamp: Optional[Timestamp]=None) -> Tuple[Timestamp, Sequence[GrantedQuota]]:
        if False:
            i = 10
            return i + 15
        if timestamp is None:
            timestamp = int(time.time())
        else:
            timestamp = int(timestamp)
        if self.assert_requests is not None:
            assert requests == self.assert_requests
        grants = []
        granted = {request.prefix: 0 for request in requests}
        for request in requests:
            prefix = request.prefix
            granted_hashes = set()
            for hash in request.unit_hashes:
                if granted[prefix] < self.grant_hashes[prefix]:
                    granted[prefix] += 1
                    granted_hashes.add(hash)
            grants.append(GrantedQuota(request=request, granted_unit_hashes=granted_hashes, reached_quota=None))
        return (timestamp, grants)

    def use_quotas(self, grants: Sequence[GrantedQuota], timestamp: Timestamp) -> None:
        if False:
            while True:
                i = 10
        pass

def test_reject_all():
    if False:
        i = 10
        return i + 15
    with override_options({'sentry-metrics.cardinality-limiter.limits.performance.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 0}], 'sentry-metrics.cardinality-limiter.limits.spans.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 0}], 'sentry-metrics.cardinality-limiter.limits.custom.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 0}]}):
        backend = MockCardinalityLimiter()
        backend.grant_hashes = {_build_quota_key(UseCaseID.TRANSACTIONS, 1): 0, _build_quota_key(UseCaseID.SPANS, 1): 0, _build_quota_key(UseCaseID.CUSTOM, 1): 0}
        limiter = TimeseriesCardinalityLimiter('', backend)
        result = limiter.check_cardinality_limits(UseCaseKey.PERFORMANCE, {BrokerMeta(Partition(Topic('topic'), 0), 0): {'org_id': 1, 'name': 'foo', 'tags': {}, 'use_case_id': UseCaseID.TRANSACTIONS}, BrokerMeta(Partition(Topic('topic'), 0), 1): {'org_id': 1, 'name': 'bar', 'tags': {}, 'use_case_id': UseCaseID.SPANS}})
        assert result.keys_to_remove == [BrokerMeta(Partition(Topic('topic'), 0), 0), BrokerMeta(Partition(Topic('topic'), 0), 1)]

def test_reject_all_with_default():
    if False:
        for i in range(10):
            print('nop')
    with override_options({'sentry-metrics.cardinality-limiter.limits.performance.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 0}], 'sentry-metrics.cardinality-limiter.limits.spans.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 0}], 'sentry-metrics.cardinality-limiter.limits.custom.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 0}], 'sentry-metrics.cardinality-limiter.limits.generic-metrics.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 0}]}):
        backend = MockCardinalityLimiter()
        backend.grant_hashes = {_build_quota_key(UseCaseID.TRANSACTIONS, 1): 0, _build_quota_key(UseCaseID.SPANS, 1): 0, _build_quota_key(UseCaseID.CUSTOM, 1): 0, _build_quota_key(UseCaseID.ESCALATING_ISSUES, 1): 0}
        limiter = TimeseriesCardinalityLimiter('', backend)
        result = limiter.check_cardinality_limits(UseCaseKey.PERFORMANCE, {BrokerMeta(Partition(Topic('topic'), 0), 0): {'org_id': 1, 'name': 'foo', 'tags': {}, 'use_case_id': UseCaseID.TRANSACTIONS}, BrokerMeta(Partition(Topic('topic'), 0), 1): {'org_id': 1, 'name': 'bar', 'tags': {}, 'use_case_id': UseCaseID.SPANS}, BrokerMeta(Partition(Topic('topic'), 0), 2): {'org_id': 1, 'name': 'boo', 'tags': {}, 'use_case_id': UseCaseID.ESCALATING_ISSUES}})
        assert result.keys_to_remove == [BrokerMeta(Partition(Topic('topic'), 0), 0), BrokerMeta(Partition(Topic('topic'), 0), 1), BrokerMeta(Partition(Topic('topic'), 0), 2)]

def test_reject_partial():
    if False:
        for i in range(10):
            print('nop')
    with override_options({'sentry-metrics.cardinality-limiter.limits.performance.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 2}], 'sentry-metrics.cardinality-limiter.limits.spans.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 0}], 'sentry-metrics.cardinality-limiter.limits.custom.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 0}]}):
        backend = MockCardinalityLimiter()
        backend.grant_hashes = {_build_quota_key(UseCaseID.TRANSACTIONS, 1): 2, _build_quota_key(UseCaseID.SPANS, 1): 0, _build_quota_key(UseCaseID.CUSTOM, 1): 0}
        limiter = TimeseriesCardinalityLimiter('', backend)
        result = limiter.check_cardinality_limits(UseCaseKey.PERFORMANCE, {BrokerMeta(Partition(Topic('topic'), 0), 0): {'org_id': 1, 'name': 'foo', 'tags': {}, 'use_case_id': UseCaseID.TRANSACTIONS}, BrokerMeta(Partition(Topic('topic'), 0), 1): {'org_id': 1, 'name': 'bar', 'tags': {}, 'use_case_id': UseCaseID.TRANSACTIONS}, BrokerMeta(Partition(Topic('topic'), 0), 2): {'org_id': 1, 'name': 'baz', 'tags': {}, 'use_case_id': UseCaseID.SPANS}})
        assert result.keys_to_remove == [BrokerMeta(Partition(Topic('topic'), 0), 2)]

def test_reject_partial_again():
    if False:
        return 10
    with override_options({'sentry-metrics.cardinality-limiter.limits.performance.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 2}], 'sentry-metrics.cardinality-limiter.limits.spans.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 2}], 'sentry-metrics.cardinality-limiter.limits.custom.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 0}]}):
        backend = MockCardinalityLimiter()
        backend.grant_hashes = {_build_quota_key(UseCaseID.TRANSACTIONS, 1): 2, _build_quota_key(UseCaseID.SPANS, 1): 2, _build_quota_key(UseCaseID.CUSTOM, 1): 0}
        limiter = TimeseriesCardinalityLimiter('', backend)
        result = limiter.check_cardinality_limits(UseCaseKey.PERFORMANCE, {BrokerMeta(Partition(Topic('topic'), 0), 0): {'org_id': 1, 'name': 'foo', 'tags': {}, 'use_case_id': UseCaseID.TRANSACTIONS}, BrokerMeta(Partition(Topic('topic'), 0), 1): {'org_id': 1, 'name': 'bar', 'tags': {}, 'use_case_id': UseCaseID.TRANSACTIONS}, BrokerMeta(Partition(Topic('topic'), 0), 2): {'org_id': 1, 'name': 'baz', 'tags': {}, 'use_case_id': UseCaseID.SPANS}, BrokerMeta(Partition(Topic('topic'), 0), 3): {'org_id': 1, 'name': 'boo', 'tags': {}, 'use_case_id': UseCaseID.CUSTOM}, BrokerMeta(Partition(Topic('topic'), 0), 4): {'org_id': 1, 'name': 'bye', 'tags': {}, 'use_case_id': UseCaseID.SPANS}})
        assert result.keys_to_remove == [BrokerMeta(Partition(Topic('topic'), 0), 3)]

def test_accept_all():
    if False:
        while True:
            i = 10
    with override_options({'sentry-metrics.cardinality-limiter.limits.performance.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 100}], 'sentry-metrics.cardinality-limiter.limits.spans.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 100}], 'sentry-metrics.cardinality-limiter.limits.custom.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 100}], 'sentry-metrics.cardinality-limiter.limits.generic-metrics.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 100}]}):
        backend = MockCardinalityLimiter()
        backend.grant_hashes = {_build_quota_key(UseCaseID.TRANSACTIONS, 1): 100, _build_quota_key(UseCaseID.SPANS, 1): 100, _build_quota_key(UseCaseID.CUSTOM, 1): 100, _build_quota_key(UseCaseID.ESCALATING_ISSUES, 1): 100}
        limiter = TimeseriesCardinalityLimiter('', backend)
        result = limiter.check_cardinality_limits(UseCaseKey.PERFORMANCE, {BrokerMeta(Partition(Topic('topic'), 0), 0): {'org_id': 1, 'name': 'foo', 'tags': {}, 'use_case_id': UseCaseID.TRANSACTIONS}, BrokerMeta(Partition(Topic('topic'), 0), 1): {'org_id': 1, 'name': 'bar', 'tags': {}, 'use_case_id': UseCaseID.TRANSACTIONS}, BrokerMeta(Partition(Topic('topic'), 0), 2): {'org_id': 1, 'name': 'baz', 'tags': {}, 'use_case_id': UseCaseID.SPANS}, BrokerMeta(Partition(Topic('topic'), 0), 3): {'org_id': 1, 'name': 'bazz', 'tags': {}, 'use_case_id': UseCaseID.ESCALATING_ISSUES}, BrokerMeta(Partition(Topic('topic'), 0), 4): {'org_id': 1, 'name': 'bye', 'tags': {}, 'use_case_id': UseCaseID.CUSTOM}})
        assert not result.keys_to_remove

def test_sample_rate_zero(set_sentry_option):
    if False:
        while True:
            i = 10
    '\n    Assert that with a rollout rate of zero, no quotas are applied.\n    '
    with override_options({'sentry-metrics.cardinality-limiter.limits.performance.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 10}], 'sentry-metrics.cardinality-limiter.limits.spans.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 10}], 'sentry-metrics.cardinality-limiter.limits.custom.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 10}]}), set_sentry_option('sentry-metrics.cardinality-limiter.orgs-rollout-rate', 0.0):
        backend = MockCardinalityLimiter()
        backend.grant_hashes = {_build_quota_key(UseCaseID.TRANSACTIONS, 1): 10, _build_quota_key(UseCaseID.SPANS, 1): 10, _build_quota_key(UseCaseID.CUSTOM, 1): 10}
        limiter = TimeseriesCardinalityLimiter('', backend)
        result = limiter.check_cardinality_limits(UseCaseKey.PERFORMANCE, {BrokerMeta(Partition(Topic('topic'), 0), 0): {'org_id': 1, 'name': 'foo', 'tags': {}, 'use_case_id': UseCaseID.TRANSACTIONS}, BrokerMeta(Partition(Topic('topic'), 0), 1): {'org_id': 1, 'name': 'bar', 'tags': {}, 'use_case_id': UseCaseID.SPANS}, BrokerMeta(Partition(Topic('topic'), 0), 2): {'org_id': 1, 'name': 'baz', 'tags': {}, 'use_case_id': UseCaseID.CUSTOM}})
        assert not result.keys_to_remove
        assert result._grants is None

def test_sample_rate_half(set_sentry_option):
    if False:
        i = 10
        return i + 15
    with override_options({'sentry-metrics.cardinality-limiter.limits.performance.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 2}], 'sentry-metrics.cardinality-limiter.limits.spans.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 1}], 'sentry-metrics.cardinality-limiter.limits.custom.per-org': [{'window_seconds': 3600, 'granularity_seconds': 60, 'limit': 0}]}), set_sentry_option('sentry-metrics.cardinality-limiter.orgs-rollout-rate', 0.5):
        backend = MockCardinalityLimiter()
        backend.grant_hashes = {_build_quota_key(UseCaseID.TRANSACTIONS, 1): 0, _build_quota_key(UseCaseID.SPANS, 1): 0, _build_quota_key(UseCaseID.CUSTOM, 1): 0}
        limiter = TimeseriesCardinalityLimiter('', backend)
        result = limiter.check_cardinality_limits(UseCaseKey.PERFORMANCE, {BrokerMeta(Partition(Topic('topic'), 0), 0): {'org_id': 1, 'name': 'foo', 'tags': {}, 'use_case_id': UseCaseID.TRANSACTIONS}, BrokerMeta(Partition(Topic('topic'), 0), 1): {'org_id': 99, 'name': 'bar', 'tags': {}, 'use_case_id': UseCaseID.TRANSACTIONS}})
        assert result.keys_to_remove == [BrokerMeta(Partition(Topic('topic'), 0), 0)]