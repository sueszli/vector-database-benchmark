import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from unittest import mock
from sentry.ratelimits.concurrent import DEFAULT_MAX_TTL_SECONDS, ConcurrentRateLimiter
from sentry.testutils.cases import TestCase
from sentry.testutils.helpers.datetime import freeze_time

class ConcurrentLimiterTest(TestCase):

    def setUp(self):
        if False:
            return 10
        self.backend = ConcurrentRateLimiter()

    def test_add_and_remove(self):
        if False:
            print('Hello World!')
        'Test the basic adding and removal of requests to the concurrent\n        rate limiter, no concurrency testing done here'
        limit = 8
        with freeze_time('2000-01-01'):
            for i in range(1, limit + 1):
                assert self.backend.start_request('foo', limit, f'request_id{i}').current_executions == i
            info = self.backend.start_request('foo', limit, 'request_id_over_the_limit')
            assert info.current_executions == limit
            assert info.limit_exceeded
            assert self.backend.get_concurrent_requests('foo') == limit
            self.backend.finish_request('foo', 'request_id1')
            assert self.backend.get_concurrent_requests('foo') == limit - 1

    def test_fails_open(self):
        if False:
            print('Hello World!')

        class FakeClient:

            def __init__(self, real_client):
                if False:
                    while True:
                        i = 10
                self._client = real_client

            def __getattr__(self, name):
                if False:
                    i = 10
                    return i + 15

                def fail(*args, **kwargs):
                    if False:
                        print('Hello World!')
                    raise Exception('OH NO')
                return fail
        limiter = ConcurrentRateLimiter()
        with mock.patch.object(limiter, 'client', FakeClient(limiter.client)):
            failed_request = limiter.start_request('key', 100, 'some_uid')
            assert failed_request.current_executions == -1
            assert failed_request.limit_exceeded is False
            limiter.finish_request('key', 'some_uid')

    def test_cleanup_stale(self):
        if False:
            i = 10
            return i + 15
        limit = 10
        num_stale = 5
        request_date = datetime(2000, 1, 1)
        with freeze_time(request_date):
            for i in range(1, num_stale + 1):
                assert self.backend.start_request('foo', limit, f'request_id{i}').current_executions == i
            assert self.backend.get_concurrent_requests('foo') == num_stale
        with freeze_time(request_date + timedelta(seconds=DEFAULT_MAX_TTL_SECONDS + 1)):
            assert self.backend.start_request('foo', limit, 'updated_request').current_executions == 1

    def test_finish_non_existent(self):
        if False:
            i = 10
            return i + 15
        self.backend.finish_request('fasdlfkdsalfkjlasdkjlasdkjflsakj', 'fsdlkajflsdakjsda')

    def test_concurrent(self):
        if False:
            for i in range(10):
                print('nop')

        def do_request():
            if False:
                print('Hello World!')
            uid = uuid.uuid4().hex
            meta = self.backend.start_request('foo', 3, uid)
            time.sleep(0.2)
            self.backend.finish_request('foo', uid)
            return meta
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for _ in range(4):
                futures.append(executor.submit(do_request))
            results = []
            for f in futures:
                results.append(f.result())
            assert len([r for r in results if r.limit_exceeded]) == 1
            time.sleep(0.3)
            assert not do_request().limit_exceeded