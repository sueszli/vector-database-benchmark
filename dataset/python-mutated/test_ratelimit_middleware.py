from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from time import sleep, time
from unittest.mock import patch, sentinel
from django.contrib.auth.models import AnonymousUser
from django.http.request import HttpRequest
from django.test import RequestFactory, override_settings
from django.urls import re_path, reverse
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from sentry.api.base import Endpoint
from sentry.api.endpoints.organization_group_index import OrganizationGroupIndexEndpoint
from sentry.middleware.ratelimit import RatelimitMiddleware
from sentry.models.apikey import ApiKey
from sentry.models.integrations.sentry_app_installation import SentryAppInstallation
from sentry.models.user import User
from sentry.ratelimits.config import RateLimitConfig, get_default_rate_limits_for_group
from sentry.ratelimits.utils import get_rate_limit_config, get_rate_limit_key, get_rate_limit_value
from sentry.silo.base import SiloMode
from sentry.testutils.cases import APITestCase, BaseTestCase, TestCase
from sentry.testutils.helpers.datetime import freeze_time
from sentry.testutils.silo import all_silo_test, assume_test_silo_mode
from sentry.types.ratelimit import RateLimit, RateLimitCategory

@all_silo_test(stable=True)
@override_settings(SENTRY_SELF_HOSTED=False)
class RatelimitMiddlewareTest(TestCase, BaseTestCase):
    middleware = RatelimitMiddleware(lambda request: sentinel.response)

    @cached_property
    def factory(self):
        if False:
            for i in range(10):
                print('nop')
        return RequestFactory()

    class TestEndpoint(Endpoint):

        def get(self):
            if False:
                while True:
                    i = 10
            return Response({'ok': True})
    _test_endpoint = TestEndpoint.as_view()

    def populate_sentry_app_request(self, request):
        if False:
            print('Hello World!')
        install = self.create_sentry_app_installation(organization=self.organization)
        token = install.api_token
        with assume_test_silo_mode(SiloMode.CONTROL):
            request.user = User.objects.get(id=install.sentry_app.proxy_user_id)
        request.auth = token

    def populate_internal_integration_request(self, request):
        if False:
            for i in range(10):
                print('nop')
        internal_integration = self.create_internal_integration(name='my_app', organization=self.organization, scopes=('project:read',), webhook_url='http://example.com')
        token = None
        with assume_test_silo_mode(SiloMode.CONTROL):
            install = SentryAppInstallation.objects.get(sentry_app=internal_integration.id, organization_id=self.organization.id)
            token = install.api_token
        assert token is not None
        with assume_test_silo_mode(SiloMode.CONTROL):
            request.user = User.objects.get(id=internal_integration.proxy_user_id)
        request.auth = token

    @patch('sentry.middleware.ratelimit.get_rate_limit_value', side_effect=Exception)
    def test_fails_open(self, default_rate_limit_mock):
        if False:
            return 10
        'Test that if something goes wrong in the rate limit middleware,\n        the request still goes through'
        request = self.factory.get('/')
        with freeze_time('2000-01-01'):
            default_rate_limit_mock.return_value = RateLimit(0, 100)
            self.middleware.process_view(request, self._test_endpoint, [], {})

    def test_process_response_fails_open(self):
        if False:
            while True:
                i = 10
        request = self.factory.get('/')
        bad_response = sentinel.response
        assert self.middleware.process_response(request, bad_response) is bad_response

        class BadRequest(HttpRequest):

            def __getattr__(self, attr):
                if False:
                    i = 10
                    return i + 15
                raise Exception('nope')
        bad_request = BadRequest()
        assert self.middleware.process_response(bad_request, bad_response) is bad_response

    @patch('sentry.middleware.ratelimit.get_rate_limit_value')
    def test_positive_rate_limit_check(self, default_rate_limit_mock):
        if False:
            i = 10
            return i + 15
        request = self.factory.get('/')
        with freeze_time('2000-01-01'):
            default_rate_limit_mock.return_value = RateLimit(0, 100)
            self.middleware.process_view(request, self._test_endpoint, [], {})
            assert request.will_be_rate_limited
        with freeze_time('2000-01-02'):
            default_rate_limit_mock.return_value = RateLimit(10, 100)
            for _ in range(10):
                self.middleware.process_view(request, self._test_endpoint, [], {})
                assert not request.will_be_rate_limited
            self.middleware.process_view(request, self._test_endpoint, [], {})
            assert request.will_be_rate_limited

    @patch('sentry.middleware.ratelimit.get_rate_limit_value')
    def test_positive_rate_limit_response_headers(self, default_rate_limit_mock):
        if False:
            while True:
                i = 10
        request = self.factory.get('/')
        with freeze_time('2000-01-01'), patch.object(RatelimitMiddlewareTest.TestEndpoint, 'enforce_rate_limit', True):
            default_rate_limit_mock.return_value = RateLimit(0, 100)
            response = self.middleware.process_view(request, self._test_endpoint, [], {})
            assert request.will_be_rate_limited
            assert response
            assert response['Access-Control-Allow-Methods'] == 'GET'
            assert response['Access-Control-Allow-Origin'] == '*'
            assert response['Access-Control-Allow-Headers']
            assert response['Access-Control-Expose-Headers']

    @patch('sentry.middleware.ratelimit.get_rate_limit_value')
    def test_negative_rate_limit_check(self, default_rate_limit_mock):
        if False:
            for i in range(10):
                print('nop')
        request = self.factory.get('/')
        default_rate_limit_mock.return_value = RateLimit(10, 100)
        self.middleware.process_view(request, self._test_endpoint, [], {})
        assert not request.will_be_rate_limited
        default_rate_limit_mock.return_value = RateLimit(1, 1)
        with freeze_time('2000-01-01') as frozen_time:
            self.middleware.process_view(request, self._test_endpoint, [], {})
            assert not request.will_be_rate_limited
            frozen_time.shift(1)
            self.middleware.process_view(request, self._test_endpoint, [], {})
            assert not request.will_be_rate_limited

    @patch('sentry.middleware.ratelimit.get_rate_limit_value')
    @override_settings(SENTRY_SELF_HOSTED=True)
    def test_self_hosted_rate_limit_check(self, default_rate_limit_mock):
        if False:
            return 10
        "Check that for self hosted installs we don't rate limit"
        request = self.factory.get('/')
        default_rate_limit_mock.return_value = RateLimit(10, 100)
        self.middleware.process_view(request, self._test_endpoint, [], {})
        assert not request.will_be_rate_limited
        default_rate_limit_mock.return_value = RateLimit(1, 1)
        with freeze_time('2000-01-01') as frozen_time:
            self.middleware.process_view(request, self._test_endpoint, [], {})
            assert not request.will_be_rate_limited
            frozen_time.shift(1)
            self.middleware.process_view(request, self._test_endpoint, [], {})
            assert not request.will_be_rate_limited

    def test_rate_limit_category(self):
        if False:
            while True:
                i = 10
        request = self.factory.get('/')
        request.META['REMOTE_ADDR'] = None
        self.middleware.process_view(request, self._test_endpoint, [], {})
        assert request.rate_limit_category is None
        request = self.factory.get('/')
        self.middleware.process_view(request, self._test_endpoint, [], {})
        assert request.rate_limit_category == RateLimitCategory.IP
        request.session = {}
        request.user = self.user
        self.middleware.process_view(request, self._test_endpoint, [], {})
        assert request.rate_limit_category == RateLimitCategory.USER
        self.populate_sentry_app_request(request)
        self.middleware.process_view(request, self._test_endpoint, [], {})
        assert request.rate_limit_category == RateLimitCategory.ORGANIZATION
        self.populate_internal_integration_request(request)
        self.middleware.process_view(request, self._test_endpoint, [], {})
        assert request.rate_limit_category == RateLimitCategory.ORGANIZATION

    def test_get_rate_limit_key(self):
        if False:
            return 10
        view = OrganizationGroupIndexEndpoint.as_view()
        rate_limit_config = get_rate_limit_config(view.view_class)
        rate_limit_group = rate_limit_config.group if rate_limit_config else RateLimitConfig().group
        request = self.factory.get('/')
        assert get_rate_limit_key(view, request, rate_limit_group, rate_limit_config) == 'ip:default:OrganizationGroupIndexEndpoint:GET:127.0.0.1'
        request.META['REMOTE_ADDR'] = None
        assert get_rate_limit_key(view, request, rate_limit_group, rate_limit_config) is None
        request.META['REMOTE_ADDR'] = '684D:1111:222:3333:4444:5555:6:77'
        assert get_rate_limit_key(view, request, rate_limit_group, rate_limit_config) == 'ip:default:OrganizationGroupIndexEndpoint:GET:684D:1111:222:3333:4444:5555:6:77'
        request.session = {}
        request.user = self.user
        assert get_rate_limit_key(view, request, rate_limit_group, rate_limit_config) == f'user:default:OrganizationGroupIndexEndpoint:GET:{self.user.id}'
        token = self.create_user_auth_token(user=self.user, scope_list=['event:read', 'org:read'])
        request.auth = token
        request.user = self.user
        assert get_rate_limit_key(view, request, rate_limit_group, rate_limit_config) == f'user:default:OrganizationGroupIndexEndpoint:GET:{self.user.id}'
        self.populate_sentry_app_request(request)
        assert get_rate_limit_key(view, request, rate_limit_group, rate_limit_config) == f'org:default:OrganizationGroupIndexEndpoint:GET:{self.organization.id}'
        self.populate_internal_integration_request(request)
        assert get_rate_limit_key(view, request, rate_limit_group, rate_limit_config) == f'org:default:OrganizationGroupIndexEndpoint:GET:{self.organization.id}'
        request.user = AnonymousUser()
        api_key = None
        with assume_test_silo_mode(SiloMode.CONTROL):
            api_key = ApiKey.objects.create(organization_id=self.organization.id, scope_list=['project:write'])
        request.auth = api_key
        assert get_rate_limit_key(view, request, rate_limit_group, rate_limit_config) == 'ip:default:OrganizationGroupIndexEndpoint:GET:684D:1111:222:3333:4444:5555:6:77'

@override_settings(SENTRY_SELF_HOSTED=False)
class TestGetRateLimitValue(TestCase):

    def test_default_rate_limit_values(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that the default rate limits are called for endpoints without overrides'

        class TestEndpoint(Endpoint):
            pass
        view = TestEndpoint.as_view()
        rate_limit_config = get_rate_limit_config(view.view_class)
        assert get_rate_limit_value('GET', RateLimitCategory.IP, rate_limit_config) == get_default_rate_limits_for_group('default', RateLimitCategory.IP)
        assert get_rate_limit_value('POST', RateLimitCategory.ORGANIZATION, rate_limit_config) == get_default_rate_limits_for_group('default', RateLimitCategory.ORGANIZATION)
        assert get_rate_limit_value('DELETE', RateLimitCategory.USER, rate_limit_config) == get_default_rate_limits_for_group('default', RateLimitCategory.USER)

    def test_override_rate_limit(self):
        if False:
            print('Hello World!')
        'Override one or more of the default rate limits'

        class TestEndpoint(Endpoint):
            rate_limits = {'GET': {RateLimitCategory.IP: RateLimit(100, 5)}, 'POST': {RateLimitCategory.USER: RateLimit(20, 4)}}
        view = TestEndpoint.as_view()
        rate_limit_config = get_rate_limit_config(view.view_class)
        assert get_rate_limit_value('GET', RateLimitCategory.IP, rate_limit_config) == RateLimit(100, 5)
        assert get_rate_limit_value('GET', RateLimitCategory.USER, rate_limit_config) == get_default_rate_limits_for_group('default', category=RateLimitCategory.USER)
        assert get_rate_limit_value('POST', RateLimitCategory.IP, rate_limit_config) == get_default_rate_limits_for_group('default', category=RateLimitCategory.IP)
        assert get_rate_limit_value('POST', RateLimitCategory.USER, rate_limit_config) == RateLimit(20, 4)

class RateLimitHeaderTestEndpoint(Endpoint):
    permission_classes = (AllowAny,)
    enforce_rate_limit = True
    rate_limits = {'GET': {RateLimitCategory.IP: RateLimit(2, 100)}}

    def inject_call(self):
        if False:
            i = 10
            return i + 15
        return

    def get(self, request):
        if False:
            return 10
        self.inject_call()
        return Response({'ok': True})

class RaceConditionEndpoint(Endpoint):
    permission_classes = (AllowAny,)
    enforce_rate_limit = False
    rate_limits = {'GET': {RateLimitCategory.IP: RateLimit(40, 100)}}

    def get(self, request):
        if False:
            return 10
        return Response({'ok': True})
CONCURRENT_RATE_LIMIT = 3
CONCURRENT_ENDPOINT_DURATION = 0.2

class ConcurrentRateLimitedEndpoint(Endpoint):
    permission_classes = (AllowAny,)
    enforce_rate_limit = True
    rate_limits = RateLimitConfig(group='foo', limit_overrides={'GET': {RateLimitCategory.IP: RateLimit(20, 1, CONCURRENT_RATE_LIMIT), RateLimitCategory.USER: RateLimit(20, 1, CONCURRENT_RATE_LIMIT), RateLimitCategory.ORGANIZATION: RateLimit(20, 1, CONCURRENT_RATE_LIMIT)}})

    def get(self, request):
        if False:
            print('Hello World!')
        sleep(CONCURRENT_ENDPOINT_DURATION)
        return Response({'ok': True})

class CallableRateLimitConfigEndpoint(Endpoint):
    permission_classes = (AllowAny,)
    enforce_rate_limit = True

    def rate_limits(request):
        if False:
            for i in range(10):
                print('nop')
        return {'GET': {RateLimitCategory.IP: RateLimit(20, 1), RateLimitCategory.USER: RateLimit(20, 1), RateLimitCategory.ORGANIZATION: RateLimit(20, 1)}}

    def get(self, request):
        if False:
            for i in range(10):
                print('nop')
        return Response({'ok': True})
urlpatterns = [re_path('^/ratelimit$', RateLimitHeaderTestEndpoint.as_view(), name='ratelimit-header-endpoint'), re_path('^/race-condition$', RaceConditionEndpoint.as_view(), name='race-condition-endpoint'), re_path('^/concurrent$', ConcurrentRateLimitedEndpoint.as_view(), name='concurrent-endpoint'), re_path('^/callable-config$', CallableRateLimitConfigEndpoint.as_view(), name='callable-config-endpoint')]

@override_settings(ROOT_URLCONF='tests.sentry.middleware.test_ratelimit_middleware', SENTRY_SELF_HOSTED=False)
class TestRatelimitHeader(APITestCase):
    endpoint = 'ratelimit-header-endpoint'

    def test_header_counts(self):
        if False:
            print('Hello World!')
        'Ensure that the header remainder counts decrease properly'
        with freeze_time('2000-01-01'):
            expected_reset_time = int(time() + 100)
            response = self.get_success_response()
            assert int(response['X-Sentry-Rate-Limit-Remaining']) == 1
            assert int(response['X-Sentry-Rate-Limit-Limit']) == 2
            assert int(response['X-Sentry-Rate-Limit-Reset']) == expected_reset_time
            response = self.get_success_response()
            assert int(response['X-Sentry-Rate-Limit-Remaining']) == 0
            assert int(response['X-Sentry-Rate-Limit-Limit']) == 2
            assert int(response['X-Sentry-Rate-Limit-Reset']) == expected_reset_time
            response = self.get_error_response()
            assert int(response['X-Sentry-Rate-Limit-Remaining']) == 0
            assert int(response['X-Sentry-Rate-Limit-Limit']) == 2
            assert int(response['X-Sentry-Rate-Limit-Reset']) == expected_reset_time
            response = self.get_error_response()
            assert int(response['X-Sentry-Rate-Limit-Remaining']) == 0
            assert int(response['X-Sentry-Rate-Limit-Limit']) == 2
            assert int(response['X-Sentry-Rate-Limit-Reset']) == expected_reset_time

    @patch('sentry.middleware.ratelimit.get_rate_limit_key')
    def test_omit_header(self, can_be_ratelimited_patch):
        if False:
            i = 10
            return i + 15
        "\n        Ensure that functions that can't be rate limited don't have rate limit headers\n\n        These functions include, but are not limited to:\n            - UI Statistics Endpoints\n            - Endpoints that don't inherit api.base.Endpoint\n        "
        can_be_ratelimited_patch.return_value = None
        response = self.get_response()
        assert not response.has_header('X-Sentry-Rate-Limit-Remaining')
        assert not response.has_header('X-Sentry-Rate-Limit-Limit')
        assert not response.has_header('X-Sentry-Rate-Limit-Reset')

    def test_header_race_condition(self):
        if False:
            while True:
                i = 10
        "Make sure concurrent requests don't affect each other's rate limit"

        def parallel_request(*args, **kwargs):
            if False:
                while True:
                    i = 10
            self.client.get(reverse('race-condition-endpoint'))
        with patch.object(RateLimitHeaderTestEndpoint, 'inject_call', parallel_request):
            response = self.get_success_response()
        assert int(response['X-Sentry-Rate-Limit-Remaining']) == 1
        assert int(response['X-Sentry-Rate-Limit-Limit']) == 2

@override_settings(ROOT_URLCONF='tests.sentry.middleware.test_ratelimit_middleware', SENTRY_SELF_HOSTED=False)
class TestConcurrentRateLimiter(APITestCase):
    endpoint = 'concurrent-endpoint'

    def test_request_finishes(self):
        if False:
            return 10
        for _ in range(2):
            response = self.get_success_response()
            assert int(response['X-Sentry-Rate-Limit-ConcurrentRemaining']) == CONCURRENT_RATE_LIMIT - 1
            assert int(response['X-Sentry-Rate-Limit-ConcurrentLimit']) == CONCURRENT_RATE_LIMIT

    def test_concurrent_request_rate_limiting(self):
        if False:
            for i in range(10):
                print('nop')
        'test the concurrent rate limiter end to-end'
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for _ in range(CONCURRENT_RATE_LIMIT + 1):
                sleep(0.01)
                futures.append(executor.submit(self.get_response))
            results = []
            for f in futures:
                results.append(f.result())
            limits = sorted((int(r['X-Sentry-Rate-Limit-ConcurrentRemaining']) for r in results))
            assert limits == [0, 0, *range(1, CONCURRENT_RATE_LIMIT)]
            sleep(CONCURRENT_ENDPOINT_DURATION + 0.1)
            response = self.get_success_response()
            assert int(response['X-Sentry-Rate-Limit-ConcurrentRemaining']) == CONCURRENT_RATE_LIMIT - 1

@override_settings(ROOT_URLCONF='tests.sentry.middleware.test_ratelimit_middleware', SENTRY_SELF_HOSTED=False)
class TestCallableRateLimitConfig(APITestCase):
    endpoint = 'callable-config-endpoint'

    def test_request_finishes(self):
        if False:
            return 10
        response = self.get_success_response()
        assert int(response['X-Sentry-Rate-Limit-Remaining']) == 19
        assert int(response['X-Sentry-Rate-Limit-Limit']) == 20