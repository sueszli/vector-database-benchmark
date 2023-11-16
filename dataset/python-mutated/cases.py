from __future__ import annotations
import hashlib
import inspect
import os.path
import re
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union
from unittest import mock
from urllib.parse import urlencode
from uuid import uuid4
from zlib import compress
import pytest
import requests
import responses
import sentry_kafka_schemas
from click.testing import CliRunner
from django.conf import settings
from django.contrib.auth import login
from django.contrib.auth.models import AnonymousUser
from django.core import signing
from django.core.cache import cache
from django.db import DEFAULT_DB_ALIAS, connection, connections
from django.db.migrations.executor import MigrationExecutor
from django.http import HttpRequest
from django.test import TestCase as DjangoTestCase
from django.test import TransactionTestCase as DjangoTransactionTestCase
from django.test import override_settings
from django.test.utils import CaptureQueriesContext
from django.urls import resolve, reverse
from django.utils import timezone as django_timezone
from django.utils.functional import cached_property
from pkg_resources import iter_entry_points
from requests.utils import CaseInsensitiveDict, get_encoding_from_headers
from rest_framework import status
from rest_framework.test import APITestCase as BaseAPITestCase
from sentry_relay.consts import SPAN_STATUS_NAME_TO_CODE
from snuba_sdk import Granularity, Limit, Offset
from snuba_sdk.conditions import BooleanCondition, Condition, ConditionGroup
from sentry import auth, eventstore
from sentry.auth.authenticators.totp import TotpInterface
from sentry.auth.provider import Provider
from sentry.auth.providers.dummy import DummyProvider
from sentry.auth.providers.saml2.activedirectory.apps import ACTIVE_DIRECTORY_PROVIDER_NAME
from sentry.auth.superuser import COOKIE_DOMAIN as SU_COOKIE_DOMAIN
from sentry.auth.superuser import COOKIE_NAME as SU_COOKIE_NAME
from sentry.auth.superuser import COOKIE_PATH as SU_COOKIE_PATH
from sentry.auth.superuser import COOKIE_SALT as SU_COOKIE_SALT
from sentry.auth.superuser import COOKIE_SECURE as SU_COOKIE_SECURE
from sentry.auth.superuser import ORG_ID as SU_ORG_ID
from sentry.auth.superuser import Superuser
from sentry.event_manager import EventManager
from sentry.eventstore.models import Event
from sentry.eventstream.snuba import SnubaEventStream
from sentry.issues.grouptype import NoiseConfig, PerformanceNPlusOneGroupType
from sentry.issues.ingest import send_issue_occurrence_to_eventstream
from sentry.mail import mail_adapter
from sentry.mediators.project_rules.creator import Creator
from sentry.models.apitoken import ApiToken
from sentry.models.authprovider import AuthProvider as AuthProviderModel
from sentry.models.commit import Commit
from sentry.models.commitauthor import CommitAuthor
from sentry.models.dashboard import Dashboard
from sentry.models.dashboard_widget import DashboardWidget, DashboardWidgetDisplayTypes, DashboardWidgetQuery
from sentry.models.deletedorganization import DeletedOrganization
from sentry.models.deploy import Deploy
from sentry.models.environment import Environment
from sentry.models.files.file import File
from sentry.models.groupmeta import GroupMeta
from sentry.models.identity import Identity, IdentityProvider, IdentityStatus
from sentry.models.notificationsetting import NotificationSetting
from sentry.models.options.project_option import ProjectOption
from sentry.models.options.user_option import UserOption
from sentry.models.organization import Organization
from sentry.models.organizationmember import OrganizationMember
from sentry.models.project import Project
from sentry.models.release import Release
from sentry.models.releasecommit import ReleaseCommit
from sentry.models.repository import Repository
from sentry.models.rule import RuleSource
from sentry.models.user import User
from sentry.models.useremail import UserEmail
from sentry.monitors.models import Monitor, MonitorEnvironment, MonitorType, ScheduleType
from sentry.notifications.types import NotificationSettingOptionValues, NotificationSettingTypes
from sentry.plugins.base import plugins
from sentry.replays.lib.event_linking import transform_event_for_linking_payload
from sentry.replays.models import ReplayRecordingSegment
from sentry.rules.base import RuleBase
from sentry.search.events.constants import METRIC_FRUSTRATED_TAG_VALUE, METRIC_SATISFACTION_TAG_KEY, METRIC_SATISFIED_TAG_VALUE, METRIC_TOLERATED_TAG_VALUE, METRICS_MAP, SPAN_METRICS_MAP
from sentry.sentry_metrics import indexer
from sentry.sentry_metrics.aggregation_option_registry import AggregationOption
from sentry.sentry_metrics.configuration import UseCaseKey
from sentry.sentry_metrics.use_case_id_registry import METRIC_PATH_MAPPING, UseCaseID
from sentry.silo import SiloMode
from sentry.snuba.dataset import EntityKey
from sentry.snuba.metrics.datasource import get_series
from sentry.snuba.metrics.extraction import OnDemandMetricSpec
from sentry.snuba.metrics.naming_layer.public import TransactionMetricKey
from sentry.tagstore.snuba.backend import SnubaTagStorage
from sentry.testutils.factories import get_fixture_path
from sentry.testutils.helpers.datetime import before_now, iso_format
from sentry.testutils.helpers.notifications import TEST_ISSUE_OCCURRENCE
from sentry.testutils.helpers.slack import install_slack
from sentry.testutils.pytest.fixtures import default_project
from sentry.testutils.pytest.selenium import Browser
from sentry.types.condition_activity import ConditionActivity, ConditionActivityType
from sentry.types.integrations import ExternalProviders
from sentry.utils import json
from sentry.utils.auth import SsoSession
from sentry.utils.dates import to_timestamp
from sentry.utils.json import dumps_htmlsafe
from sentry.utils.performance_issues.performance_detection import detect_performance_problems
from sentry.utils.retries import TimedRetryPolicy
from sentry.utils.samples import load_data
from sentry.utils.snuba import _snuba_pool
from ..services.hybrid_cloud.organization.serial import serialize_rpc_organization
from ..shared_integrations.client.proxy import IntegrationProxyClient
from ..snuba.metrics import MetricConditionField, MetricField, MetricGroupByField, MetricOrderByField, MetricsQuery, get_date_range
from ..snuba.metrics.naming_layer.mri import SessionMRI, TransactionMRI, parse_mri
from .asserts import assert_status_code
from .factories import Factories
from .fixtures import Fixtures
from .helpers import AuthProvider, Feature, TaskRunner, override_options, parse_queries
from .silo import assume_test_silo_mode
from .skips import requires_snuba
__all__ = ('TestCase', 'TransactionTestCase', 'APITestCase', 'TwoFactorAPITestCase', 'AuthProviderTestCase', 'RuleTestCase', 'PermissionTestCase', 'PluginTestCase', 'CliTestCase', 'AcceptanceTestCase', 'IntegrationTestCase', 'SnubaTestCase', 'BaseMetricsTestCase', 'BaseMetricsLayerTestCase', 'BaseIncidentsTest', 'IntegrationRepositoryTestCase', 'ReleaseCommitPatchTest', 'SetRefsTestCase', 'OrganizationDashboardWidgetTestCase', 'SCIMTestCase', 'SCIMAzureTestCase', 'MetricsEnhancedPerformanceTestCase', 'MetricsAPIBaseTestCase', 'OrganizationMetricMetaIntegrationTestCase', 'ProfilesSnubaTestCase', 'ReplaysAcceptanceTestCase', 'ReplaysSnubaTestCase', 'MonitorTestCase', 'MonitorIngestTestCase')
from ..types.region import get_region_by_name
DEFAULT_USER_AGENT = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'
DETECT_TESTCASE_MISUSE = os.environ.get('SENTRY_DETECT_TESTCASE_MISUSE') == '1'
SILENCE_MIXED_TESTCASE_MISUSE = os.environ.get('SENTRY_SILENCE_MIXED_TESTCASE_MISUSE') == '1'
SessionOrTransactionMRI = Union[SessionMRI, TransactionMRI]

class BaseTestCase(Fixtures):

    def assertRequiresAuthentication(self, path, method='GET'):
        if False:
            for i in range(10):
                print('nop')
        resp = getattr(self.client, method.lower())(path)
        assert resp.status_code == 302
        assert resp['Location'].startswith('http://testserver' + reverse('sentry-login'))

    @pytest.fixture(autouse=True)
    def setup_dummy_auth_provider(self):
        if False:
            while True:
                i = 10
        auth.register('dummy', DummyProvider)
        self.addCleanup(auth.unregister, 'dummy', DummyProvider)

    def tasks(self):
        if False:
            i = 10
            return i + 15
        return TaskRunner()

    @pytest.fixture(autouse=True)
    def polyfill_capture_on_commit_callbacks(self, django_capture_on_commit_callbacks):
        if False:
            i = 10
            return i + 15
        "\n        https://pytest-django.readthedocs.io/en/latest/helpers.html#django_capture_on_commit_callbacks\n\n        pytest-django comes with its own polyfill of this Django helper for\n        older Django versions, so we're using that.\n        "
        self.capture_on_commit_callbacks = django_capture_on_commit_callbacks

    @pytest.fixture(autouse=True)
    def expose_stale_database_reads(self, stale_database_reads):
        if False:
            return 10
        self.stale_database_reads = stale_database_reads

    def feature(self, names):
        if False:
            for i in range(10):
                print('nop')
        "\n        >>> with self.feature({'feature:name': True})\n        >>>     # ...\n        "
        return Feature(names)

    def auth_provider(self, name, cls):
        if False:
            for i in range(10):
                print('nop')
        "\n        >>> with self.auth_provider('name', Provider)\n        >>>     # ...\n        "
        return AuthProvider(name, cls)

    def save_session(self):
        if False:
            print('Hello World!')
        self.session.save()
        self.save_cookie(name=settings.SESSION_COOKIE_NAME, value=self.session.session_key, max_age=None, path='/', domain=settings.SESSION_COOKIE_DOMAIN, secure=settings.SESSION_COOKIE_SECURE or None, expires=None)

    def save_cookie(self, name, value, **params):
        if False:
            while True:
                i = 10
        self.client.cookies[name] = value
        self.client.cookies[name].update({k.replace('_', '-'): v for (k, v) in params.items()})

    def make_request(self, user=None, auth=None, method=None, is_superuser=False, path='/', secure_scheme=False, subdomain=None, *, GET: dict[str, str] | None=None) -> HttpRequest:
        if False:
            while True:
                i = 10
        request = HttpRequest()
        if subdomain:
            setattr(request, 'subdomain', subdomain)
        if method:
            request.method = method
        request.path = path
        request.META['REMOTE_ADDR'] = '127.0.0.1'
        request.META['SERVER_NAME'] = 'testserver'
        request.META['SERVER_PORT'] = 80
        if GET is not None:
            for (k, v) in GET.items():
                request.GET[k] = v
        if secure_scheme:
            secure_header = settings.SECURE_PROXY_SSL_HEADER
            request.META[secure_header[0]] = secure_header[1]
        request.session = self.session
        request.auth = auth
        request.user = user or AnonymousUser()
        request.superuser = Superuser(request)
        if is_superuser:
            request.superuser.set_logged_in(user)
        request.is_superuser = lambda : request.superuser.is_active
        request.successful_authenticator = None
        return request

    @TimedRetryPolicy.wrap(timeout=5)
    def login_as(self, user, organization_id=None, organization_ids=None, superuser=False, superuser_sso=True):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(user, OrganizationMember):
            with assume_test_silo_mode(SiloMode.CONTROL):
                user = User.objects.get(id=user.user_id)
        user.backend = settings.AUTHENTICATION_BACKENDS[0]
        request = self.make_request()
        with assume_test_silo_mode(SiloMode.CONTROL):
            login(request, user)
        request.user = user
        if organization_ids is None:
            organization_ids = set()
        else:
            organization_ids = set(organization_ids)
        if superuser and superuser_sso is not False:
            if SU_ORG_ID:
                organization_ids.add(SU_ORG_ID)
        if organization_id:
            organization_ids.add(organization_id)
        if organization_ids:
            for o in organization_ids:
                sso_session = SsoSession.create(o)
                self.session[sso_session.session_key] = sso_session.to_dict()
        if not superuser:
            request.superuser._set_logged_out()
        elif request.user.is_superuser and superuser:
            request.superuser.set_logged_in(request.user)
            self.save_cookie(name=SU_COOKIE_NAME, value=signing.get_cookie_signer(salt=SU_COOKIE_NAME + SU_COOKIE_SALT).sign(request.superuser.token), max_age=None, path=SU_COOKIE_PATH, domain=SU_COOKIE_DOMAIN, secure=SU_COOKIE_SECURE or None, expires=None)
        self.save_session()

    def load_fixture(self, filepath):
        if False:
            i = 10
            return i + 15
        with open(get_fixture_path(filepath), 'rb') as fp:
            return fp.read()

    def _pre_setup(self):
        if False:
            while True:
                i = 10
        super()._pre_setup()
        cache.clear()
        ProjectOption.objects.clear_local_cache()
        GroupMeta.objects.clear_local_cache()

    def _post_teardown(self):
        if False:
            i = 10
            return i + 15
        super()._post_teardown()

    def options(self, options):
        if False:
            return 10
        '\n        A context manager that temporarily sets a global option and reverts\n        back to the original value when exiting the context.\n        '
        return override_options(options)

    def assert_valid_deleted_log(self, deleted_log, original_object):
        if False:
            i = 10
            return i + 15
        assert deleted_log is not None
        assert original_object.name == deleted_log.name
        assert deleted_log.name == original_object.name
        assert deleted_log.slug == original_object.slug
        if not isinstance(deleted_log, DeletedOrganization):
            assert deleted_log.organization_id == original_object.organization.id
            assert deleted_log.organization_name == original_object.organization.name
            assert deleted_log.organization_slug == original_object.organization.slug
        assert deleted_log.date_created == original_object.date_added
        assert deleted_log.date_deleted >= deleted_log.date_created

    def assertWriteQueries(self, queries, debug=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        func = kwargs.pop('func', None)
        using = kwargs.pop('using', DEFAULT_DB_ALIAS)
        conn = connections[using]
        context = _AssertQueriesContext(self, queries, debug, conn)
        if func is None:
            return context
        with context:
            func(*args, **kwargs)

    def get_mock_uuid(self):
        if False:
            while True:
                i = 10

        class uuid:
            hex = 'abc123'
            bytes = b'\x00\x01\x02'
        return uuid

class _AssertQueriesContext(CaptureQueriesContext):

    def __init__(self, test_case, queries, debug, connection):
        if False:
            for i in range(10):
                print('nop')
        self.test_case = test_case
        self.queries = queries
        self.debug = debug
        super().__init__(connection)

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            i = 10
            return i + 15
        super().__exit__(exc_type, exc_value, traceback)
        if exc_type is not None:
            return
        parsed_queries = parse_queries(self.captured_queries)
        if self.debug:
            import pprint
            pprint.pprint('====================== Raw Queries ======================')
            pprint.pprint(self.captured_queries)
            pprint.pprint('====================== Table writes ======================')
            pprint.pprint(parsed_queries)
        for (table, num) in parsed_queries.items():
            expected = self.queries.get(table, 0)
            if expected == 0:
                import pprint
                pprint.pprint('WARNING: no query against %s emitted, add debug=True to see all the queries' % table)
            else:
                self.test_case.assertTrue(num == expected, '%d write queries expected on `%s`, got %d, add debug=True to see all the queries' % (expected, table, num))
        for (table, num) in self.queries.items():
            executed = parsed_queries.get(table, None)
            self.test_case.assertFalse(executed is None, 'no query against %s emitted, add debug=True to see all the queries' % table)

@override_settings(ROOT_URLCONF='sentry.web.urls')
class TestCase(BaseTestCase, DjangoTestCase):
    databases: set[str] | str = '__all__'

    @contextmanager
    def auto_select_silo_mode_on_redirects(self):
        if False:
            while True:
                i = 10
        "\n        Tests that utilize follow=True may follow redirects between silo modes.  This isn't ideal but convenient for\n        testing certain work flows.  Using this context manager, the silo mode in the test will swap automatically\n        for each view's decorator in order to prevent otherwise unavoidable SiloAvailability errors.\n        "
        old_request = self.client.request

        def request(**request: Any) -> Any:
            if False:
                for i in range(10):
                    print('nop')
            resolved = resolve(request['PATH_INFO'])
            view_class = getattr(resolved.func, 'view_class', None)
            if view_class is not None:
                endpoint_silo_limit = getattr(view_class, 'silo_limit', None)
                if endpoint_silo_limit:
                    for mode in endpoint_silo_limit.modes:
                        if mode is SiloMode.MONOLITH or mode is SiloMode.get_current_mode():
                            continue
                        region = None
                        if mode is SiloMode.REGION:
                            region = get_region_by_name(settings.SENTRY_MONOLITH_REGION)
                        with SiloMode.exit_single_process_silo_context(), SiloMode.enter_single_process_silo_context(mode, region):
                            return old_request(**request)
            return old_request(**request)
        with mock.patch.object(self.client, 'request', new=request):
            yield
    if DETECT_TESTCASE_MISUSE:

        @pytest.fixture(autouse=True, scope='class')
        def _require_db_usage(self, request):
            if False:
                return 10

            class State:
                used_db = {}
                base = request.cls
            state = State()
            yield state
            did_not_use = set()
            did_use = set()
            for (name, used) in state.used_db.items():
                if used:
                    did_use.add(name)
                else:
                    did_not_use.add(name)
            if did_not_use and (not did_use):
                pytest.fail(f'none of the test functions in {state.base} used the DB! Use `unittest.TestCase` instead of `sentry.testutils.TestCase` for those kinds of tests.')
            elif did_not_use and did_use and (not SILENCE_MIXED_TESTCASE_MISUSE):
                pytest.fail(f'Some of the test functions in {state.base} used the DB and some did not! test functions using the db: {did_use}\nUse `unittest.TestCase` instead of `sentry.testutils.TestCase` for the tests not using the db.')

        @pytest.fixture(autouse=True, scope='function')
        def _check_function_for_db(self, request, monkeypatch, _require_db_usage):
            if False:
                return 10
            from django.db.backends.base.base import BaseDatabaseWrapper
            real_ensure_connection = BaseDatabaseWrapper.ensure_connection
            state = _require_db_usage

            def ensure_connection(*args, **kwargs):
                if False:
                    return 10
                for info in inspect.stack():
                    frame = info.frame
                    try:
                        first_arg_name = frame.f_code.co_varnames[0]
                        first_arg = frame.f_locals[first_arg_name]
                    except LookupError:
                        continue
                    if type(first_arg) is state.base and info.function in state.used_db:
                        state.used_db[info.function] = True
                        break
                return real_ensure_connection(*args, **kwargs)
            monkeypatch.setattr(BaseDatabaseWrapper, 'ensure_connection', ensure_connection)
            state.used_db[request.function.__name__] = False
            yield

class TransactionTestCase(BaseTestCase, DjangoTransactionTestCase):
    databases: set[str] | str = '__all__'
    pass

class PerformanceIssueTestCase(BaseTestCase):
    databases: set[str] | str = '__all__'

    def create_performance_issue(self, tags=None, contexts=None, fingerprint=None, transaction=None, event_data=None, issue_type=None, noise_limit=0, project_id=None, detector_option='performance.issues.n_plus_one_db.problem-creation', user_data=None):
        if False:
            i = 10
            return i + 15
        if issue_type is None:
            issue_type = PerformanceNPlusOneGroupType
        if event_data is None:
            event_data = load_data('transaction-n-plus-one', timestamp=before_now(minutes=10))
        if tags is not None:
            event_data['tags'] = tags
        if contexts is not None:
            event_data['contexts'] = contexts
        if transaction:
            event_data['transaction'] = transaction
        if project_id is None:
            project_id = self.project.id
        if user_data:
            event_data['user'] = user_data
        perf_event_manager = EventManager(event_data)
        perf_event_manager.normalize()

        def detect_performance_problems_interceptor(data: Event, project: Project):
            if False:
                for i in range(10):
                    print('nop')
            perf_problems = detect_performance_problems(data, project)
            if fingerprint:
                for perf_problem in perf_problems:
                    perf_problem.fingerprint = fingerprint
            return perf_problems
        with mock.patch('sentry.issues.ingest.send_issue_occurrence_to_eventstream', side_effect=send_issue_occurrence_to_eventstream) as mock_eventstream, mock.patch('sentry.event_manager.detect_performance_problems', side_effect=detect_performance_problems_interceptor), mock.patch.object(issue_type, 'noise_config', new=NoiseConfig(noise_limit, timedelta(minutes=1))), override_options({'performance.issues.all.problem-detection': 1.0, detector_option: 1.0}):
            event = perf_event_manager.save(project_id)
            if mock_eventstream.call_args:
                event = event.for_group(mock_eventstream.call_args[0][2].group)
                event.occurrence = mock_eventstream.call_args[0][1]
            return event

class APITestCase(BaseTestCase, BaseAPITestCase):
    """
    Extend APITestCase to inherit access to `client`, an object with methods
    that simulate API calls to Sentry, and the helper `get_response`, which
    combines and simplifies a lot of tedious parts of making API calls in tests.
    When creating API tests, use a new class per endpoint-method pair. The class
    must set the string `endpoint`.
    """
    databases: set[str] | str = '__all__'
    method = 'get'

    @property
    def endpoint(self):
        if False:
            print('Hello World!')
        raise NotImplementedError(f'implement for {type(self).__module__}.{type(self).__name__}')

    def get_response(self, *args, **params):
        if False:
            print('Hello World!')
        "\n        Simulate an API call to the test case's URI and method.\n\n        :param params:\n            Note: These names are intentionally a little funny to prevent name\n             collisions with real API arguments.\n            * extra_headers: (Optional) Dict mapping keys to values that will be\n             passed as request headers.\n            * qs_params: (Optional) Dict mapping keys to values that will be\n             url-encoded into a API call's query string.\n            * raw_data: (Optional) Sometimes we want to precompute the JSON body.\n        :returns Response object\n        "
        url = reverse(self.endpoint, args=args)
        if 'qs_params' in params:
            query_string = urlencode(params.pop('qs_params'), doseq=True)
            url = f'{url}?{query_string}'
        headers = params.pop('extra_headers', {})
        raw_data = params.pop('raw_data', None)
        if raw_data and isinstance(raw_data, bytes):
            raw_data = raw_data.decode('utf-8')
        if raw_data and isinstance(raw_data, str):
            raw_data = json.loads(raw_data)
        data = raw_data or params
        method = params.pop('method', self.method).lower()
        return getattr(self.client, method)(url, format='json', data=data, **headers)

    def get_success_response(self, *args, **params):
        if False:
            while True:
                i = 10
        "\n        Call `get_response` (see above) and assert the response's status code.\n\n        :param params:\n            * status_code: (Optional) Assert that the response's status code is\n            a specific code. Omit to assert any successful status_code.\n        :returns Response object\n        "
        status_code = params.pop('status_code', None)
        if status_code and status_code >= 400:
            raise Exception('status_code must be < 400')
        method = params.pop('method', self.method).lower()
        response = self.get_response(*args, method=method, **params)
        if status_code:
            assert_status_code(response, status_code)
        elif method == 'get':
            assert_status_code(response, status.HTTP_200_OK)
        elif method == 'put':
            assert_status_code(response, status.HTTP_200_OK)
        elif method == 'delete':
            assert_status_code(response, status.HTTP_204_NO_CONTENT)
        else:
            assert_status_code(response, 200, 300)
        return response

    def get_error_response(self, *args, **params):
        if False:
            for i in range(10):
                print('nop')
        "\n        Call `get_response` (see above) and assert that the response's status\n        code is an error code. Basically it's syntactic sugar.\n\n        :param params:\n            * status_code: (Optional) Assert that the response's status code is\n            a specific error code. Omit to assert any error status_code.\n        :returns Response object\n        "
        status_code = params.pop('status_code', None)
        if status_code and status_code < 400:
            raise Exception('status_code must be >= 400 (an error status code)')
        response = self.get_response(*args, **params)
        if status_code:
            assert_status_code(response, status_code)
        else:
            assert_status_code(response, 400, 600)
        return response

    def get_cursor_headers(self, response):
        if False:
            i = 10
            return i + 15
        return [link['cursor'] for link in requests.utils.parse_header_links(response.get('link').rstrip('>').replace('>,<', ',<'))]

    def analytics_called_with_args(self, fn, name, **kwargs):
        if False:
            return 10
        for (call_args, call_kwargs) in fn.call_args_list:
            event_name = call_args[0]
            if event_name == name:
                assert all((call_kwargs.get(key, None) == val for (key, val) in kwargs.items()))
                return True
        return False

    @contextmanager
    def api_gateway_proxy_stubbed(self):
        if False:
            print('Hello World!')
        'Mocks a fake api gateway proxy that redirects via Client objects'

        def proxy_raw_request(method: str, url: str, headers: Mapping[str, str], params: Mapping[str, str] | None, data: Any, **kwds: Any) -> requests.Response:
            if False:
                print('Hello World!')
            from django.test.client import Client
            client = Client()
            extra: Mapping[str, Any] = {f"HTTP_{k.replace('-', '_').upper()}": v for (k, v) in headers.items()}
            if params:
                url += '?' + urlencode(params)
            with assume_test_silo_mode(SiloMode.REGION):
                resp = getattr(client, method.lower())(url, b''.join(data), headers['Content-Type'], **extra)
            response = requests.Response()
            response.status_code = resp.status_code
            response.headers = CaseInsensitiveDict(resp.headers)
            response.encoding = get_encoding_from_headers(response.headers)
            response.raw = BytesIO(resp.content)
            return response
        with mock.patch('sentry.api_gateway.proxy.external_request', new=proxy_raw_request):
            yield

class TwoFactorAPITestCase(APITestCase):

    @cached_property
    def path_2fa(self):
        if False:
            i = 10
            return i + 15
        return reverse('sentry-account-settings-security')

    def enable_org_2fa(self, organization):
        if False:
            while True:
                i = 10
        organization.flags.require_2fa = True
        organization.save()

    def api_enable_org_2fa(self, organization, user):
        if False:
            while True:
                i = 10
        self.login_as(user)
        url = reverse('sentry-api-0-organization-details', kwargs={'organization_slug': organization.slug})
        return self.client.put(url, data={'require2FA': True})

    def api_disable_org_2fa(self, organization, user):
        if False:
            i = 10
            return i + 15
        url = reverse('sentry-api-0-organization-details', kwargs={'organization_slug': organization.slug})
        return self.client.put(url, data={'require2FA': False})

    def assert_can_enable_org_2fa(self, organization, user, status_code=200):
        if False:
            i = 10
            return i + 15
        self.__helper_enable_organization_2fa(organization, user, status_code)

    def assert_cannot_enable_org_2fa(self, organization, user, status_code, err_msg=None):
        if False:
            return 10
        self.__helper_enable_organization_2fa(organization, user, status_code, err_msg)

    def __helper_enable_organization_2fa(self, organization, user, status_code, err_msg=None):
        if False:
            print('Hello World!')
        response = self.api_enable_org_2fa(organization, user)
        assert response.status_code == status_code
        if err_msg:
            assert err_msg.encode('utf-8') in response.content
        organization = Organization.objects.get(id=organization.id)
        if 200 <= status_code < 300:
            assert organization.flags.require_2fa
        else:
            assert not organization.flags.require_2fa

    def add_2fa_users_to_org(self, organization, num_of_users=10, num_with_2fa=5):
        if False:
            return 10
        non_compliant_members = []
        for num in range(0, num_of_users):
            user = self.create_user('foo_%s@example.com' % num)
            self.create_member(organization=organization, user=user)
            if num_with_2fa:
                TotpInterface().enroll(user)
                num_with_2fa -= 1
            else:
                non_compliant_members.append(user.email)
        return non_compliant_members

class AuthProviderTestCase(TestCase):
    provider: type[Provider] = DummyProvider
    provider_name = 'dummy'

    def setUp(self):
        if False:
            return 10
        super().setUp()
        if self.provider_name != 'dummy' or self.provider != DummyProvider:
            auth.register(self.provider_name, self.provider)
            self.addCleanup(auth.unregister, self.provider_name, self.provider)

class RuleTestCase(TestCase):

    @property
    def rule_cls(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError(f'implement for {type(self).__module__}.{type(self).__name__}')

    def get_event(self):
        if False:
            return 10
        return self.event

    def get_rule(self, **kwargs):
        if False:
            print('Hello World!')
        kwargs.setdefault('project', self.project)
        kwargs.setdefault('data', {})
        return self.rule_cls(**kwargs)

    def get_state(self, **kwargs):
        if False:
            return 10
        from sentry.rules import EventState
        kwargs.setdefault('is_new', True)
        kwargs.setdefault('is_regression', True)
        kwargs.setdefault('is_new_group_environment', True)
        kwargs.setdefault('has_reappeared', True)
        return EventState(**kwargs)

    def get_condition_activity(self, **kwargs) -> ConditionActivity:
        if False:
            i = 10
            return i + 15
        kwargs.setdefault('group_id', self.event.group.id)
        kwargs.setdefault('type', ConditionActivityType.CREATE_ISSUE)
        kwargs.setdefault('timestamp', self.event.datetime)
        return ConditionActivity(**kwargs)

    def passes_activity(self, rule: RuleBase, condition_activity: Optional[ConditionActivity]=None, event_map: Optional[Dict[str, Any]]=None):
        if False:
            for i in range(10):
                print('nop')
        if condition_activity is None:
            condition_activity = self.get_condition_activity()
        if event_map is None:
            event_map = {}
        return rule.passes_activity(condition_activity, event_map)

    def assertPasses(self, rule, event=None, **kwargs):
        if False:
            while True:
                i = 10
        if event is None:
            event = self.event
        state = self.get_state(**kwargs)
        assert rule.passes(event, state) is True

    def assertDoesNotPass(self, rule, event=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if event is None:
            event = self.event
        state = self.get_state(**kwargs)
        assert rule.passes(event, state) is False

class PermissionTestCase(TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.owner = self.create_user(is_superuser=False)
        self.organization = self.create_organization(owner=self.owner, flags=0)
        self.team = self.create_team(organization=self.organization)

    def assert_can_access(self, user, path, method='GET', **kwargs):
        if False:
            while True:
                i = 10
        self.login_as(user, superuser=user.is_superuser)
        resp = getattr(self.client, method.lower())(path, **kwargs)
        assert resp.status_code >= 200 and resp.status_code < 300
        return resp

    def assert_cannot_access(self, user, path, method='GET', **kwargs):
        if False:
            return 10
        self.login_as(user, superuser=user.is_superuser)
        resp = getattr(self.client, method.lower())(path, **kwargs)
        assert resp.status_code >= 300

    def assert_member_can_access(self, path, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.assert_role_can_access(path, 'member', **kwargs)

    def assert_manager_can_access(self, path, **kwargs):
        if False:
            return 10
        return self.assert_role_can_access(path, 'manager', **kwargs)

    def assert_teamless_member_can_access(self, path, **kwargs):
        if False:
            i = 10
            return i + 15
        user = self.create_user(is_superuser=False)
        self.create_member(user=user, organization=self.organization, role='member', teams=[])
        self.assert_can_access(user, path, **kwargs)

    def assert_member_cannot_access(self, path, **kwargs):
        if False:
            while True:
                i = 10
        return self.assert_role_cannot_access(path, 'member', **kwargs)

    def assert_manager_cannot_access(self, path, **kwargs):
        if False:
            print('Hello World!')
        return self.assert_role_cannot_access(path, 'manager', **kwargs)

    def assert_teamless_member_cannot_access(self, path, **kwargs):
        if False:
            i = 10
            return i + 15
        user = self.create_user(is_superuser=False)
        self.create_member(user=user, organization=self.organization, role='member', teams=[])
        self.assert_cannot_access(user, path, **kwargs)

    def assert_team_admin_can_access(self, path, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.assert_role_can_access(path, 'admin', **kwargs)

    def assert_teamless_admin_can_access(self, path, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        user = self.create_user(is_superuser=False)
        self.create_member(user=user, organization=self.organization, role='admin', teams=[])
        self.assert_can_access(user, path, **kwargs)

    def assert_team_admin_cannot_access(self, path, **kwargs):
        if False:
            while True:
                i = 10
        return self.assert_role_cannot_access(path, 'admin', **kwargs)

    def assert_teamless_admin_cannot_access(self, path, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        user = self.create_user(is_superuser=False)
        self.create_member(user=user, organization=self.organization, role='admin', teams=[])
        self.assert_cannot_access(user, path, **kwargs)

    def assert_team_owner_can_access(self, path, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.assert_role_can_access(path, 'owner', **kwargs)

    def assert_owner_can_access(self, path, **kwargs):
        if False:
            i = 10
            return i + 15
        return self.assert_role_can_access(path, 'owner', **kwargs)

    def assert_owner_cannot_access(self, path, **kwargs):
        if False:
            i = 10
            return i + 15
        return self.assert_role_cannot_access(path, 'owner', **kwargs)

    def assert_non_member_cannot_access(self, path, **kwargs):
        if False:
            return 10
        user = self.create_user(is_superuser=False)
        self.assert_cannot_access(user, path, **kwargs)

    def assert_role_can_access(self, path, role, **kwargs):
        if False:
            print('Hello World!')
        user = self.create_user(is_superuser=False)
        self.create_member(user=user, organization=self.organization, role=role, teams=[self.team])
        return self.assert_can_access(user, path, **kwargs)

    def assert_role_cannot_access(self, path, role, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        user = self.create_user(is_superuser=False)
        self.create_member(user=user, organization=self.organization, role=role, teams=[self.team])
        self.assert_cannot_access(user, path, **kwargs)

@requires_snuba
class PluginTestCase(TestCase):

    @property
    def plugin(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError(f'implement for {type(self).__module__}.{type(self).__name__}')

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        if inspect.isclass(self.plugin):
            plugins.register(self.plugin)
            self.addCleanup(plugins.unregister, self.plugin)

    def assertAppInstalled(self, name, path):
        if False:
            while True:
                i = 10
        for ep in iter_entry_points('sentry.apps'):
            if ep.name == name:
                ep_path = ep.module_name
                if ep_path == path:
                    return
                self.fail('Found app in entry_points, but wrong class. Got %r, expected %r' % (ep_path, path))
        self.fail(f'Missing app from entry_points: {name!r}')

    def assertPluginInstalled(self, name, plugin):
        if False:
            for i in range(10):
                print('nop')
        path = type(plugin).__module__ + ':' + type(plugin).__name__
        for ep in iter_entry_points('sentry.plugins'):
            if ep.name == name:
                ep_path = ep.module_name + ':' + '.'.join(ep.attrs)
                if ep_path == path:
                    return
                self.fail('Found plugin in entry_points, but wrong class. Got %r, expected %r' % (ep_path, path))
        self.fail(f'Missing plugin from entry_points: {name!r}')

class CliTestCase(TestCase):

    @cached_property
    def runner(self) -> CliRunner:
        if False:
            while True:
                i = 10
        return CliRunner()

    @property
    def command(self):
        if False:
            return 10
        raise NotImplementedError(f'implement for {type(self).__module__}.{type(self).__name__}')
    default_args = []

    def invoke(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        args += tuple(self.default_args)
        return self.runner.invoke(self.command, args, obj={}, **kwargs)

@pytest.mark.usefixtures('browser')
class AcceptanceTestCase(TransactionTestCase):
    browser: Browser

    @pytest.fixture(autouse=True)
    def _setup_today(self):
        if False:
            i = 10
            return i + 15
        with mock.patch('django.utils.timezone.now', return_value=datetime(2013, 5, 18, 15, 13, 58, 132928, tzinfo=timezone.utc)):
            yield

    def wait_for_loading(self):
        if False:
            while True:
                i = 10
        self.browser.wait_until_not('[data-test-id="events-request-loading"]')
        self.browser.wait_until_not('[data-test-id="loading-indicator"]')
        self.browser.wait_until_not('.loading')

    def tearDown(self):
        if False:
            return 10
        self.wait_for_loading()
        super().tearDown()

    def save_cookie(self, name, value, **params):
        if False:
            return 10
        self.browser.save_cookie(name=name, value=value, **params)

    def save_session(self):
        if False:
            return 10
        self.session.save()
        self.save_cookie(name=settings.SESSION_COOKIE_NAME, value=self.session.session_key)
        self.client.cookies[settings.SESSION_COOKIE_NAME] = self.session.session_key

    def dismiss_assistant(self, which=None):
        if False:
            for i in range(10):
                print('nop')
        if which is None:
            which = ('issue', 'issue_stream')
        if isinstance(which, str):
            which = [which]
        for item in which:
            res = self.client.put('/api/0/assistant/', content_type='application/json', data=json.dumps({'guide': item, 'status': 'viewed', 'useful': True}))
            assert res.status_code == 201, res.content

class IntegrationTestCase(TestCase):

    @property
    def provider(self):
        if False:
            print('Hello World!')
        raise NotImplementedError(f'implement for {type(self).__module__}.{type(self).__name__}')

    def setUp(self):
        if False:
            while True:
                i = 10
        from sentry.integrations.pipeline import IntegrationPipeline
        super().setUp()
        self.organization = self.create_organization(name='foo', owner=self.user)
        with assume_test_silo_mode(SiloMode.REGION):
            rpc_organization = serialize_rpc_organization(self.organization)
        self.login_as(self.user)
        self.request = self.make_request(self.user)
        self.pipeline = IntegrationPipeline(request=self.request, organization=rpc_organization, provider_key=self.provider.key)
        self.init_path = reverse('sentry-organization-integrations-setup', kwargs={'organization_slug': self.organization.slug, 'provider_id': self.provider.key})
        self.setup_path = reverse('sentry-extension-setup', kwargs={'provider_id': self.provider.key})
        self.configure_path = f'/extensions/{self.provider.key}/configure/'
        self.pipeline.initialize()
        self.save_session()

    def assertDialogSuccess(self, resp):
        if False:
            print('Hello World!')
        assert b'window.opener.postMessage({"success":true' in resp.content

@pytest.mark.snuba
@requires_snuba
class SnubaTestCase(BaseTestCase):
    """
    Mixin for enabling test case classes to talk to snuba
    Useful when you are working on acceptance tests or integration
    tests that require snuba.
    """
    databases: set[str] | str = '__all__'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.init_snuba()

    @pytest.fixture(autouse=True)
    def initialize(self, reset_snuba, call_snuba):
        if False:
            for i in range(10):
                print('nop')
        self.call_snuba = call_snuba

    @contextmanager
    def disable_snuba_query_cache(self):
        if False:
            i = 10
            return i + 15
        self.snuba_update_config({'use_readthrough_query_cache': 0, 'use_cache': 0})
        yield
        self.snuba_update_config({'use_readthrough_query_cache': None, 'use_cache': None})

    @classmethod
    def snuba_get_config(cls):
        if False:
            print('Hello World!')
        return _snuba_pool.request('GET', '/config.json').data

    @classmethod
    def snuba_update_config(cls, config_vals):
        if False:
            print('Hello World!')
        return _snuba_pool.request('POST', '/config.json', body=json.dumps(config_vals))

    def init_snuba(self):
        if False:
            return 10
        self.snuba_eventstream = SnubaEventStream()
        self.snuba_tagstore = SnubaTagStorage()

    def store_event(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Simulates storing an event for testing.\n\n        To set event title:\n        - use "message": "{title}" field for errors\n        - use "transaction": "{title}" field for transactions\n        More info on event payloads: https://develop.sentry.dev/sdk/event-payloads/\n        '
        with mock.patch('sentry.eventstream.insert', self.snuba_eventstream.insert):
            stored_event = Factories.store_event(*args, **kwargs)
            stored_group = stored_event.group
            if stored_group is not None:
                self.store_group(stored_group)
            stored_groups = stored_event.groups
            if stored_groups is not None:
                for group in stored_groups:
                    self.store_group(group)
            return stored_event

    def wait_for_event_count(self, project_id, total, attempts=2):
        if False:
            i = 10
            return i + 15
        "\n        Wait until the event count reaches the provided value or until attempts is reached.\n\n        Useful when you're storing several events and need to ensure that snuba/clickhouse\n        state has settled.\n        "
        attempt = 0
        snuba_filter = eventstore.Filter(project_ids=[project_id])
        last_events_seen = 0
        while attempt < attempts:
            events = eventstore.backend.get_events(snuba_filter, referrer='test.wait_for_event_count')
            last_events_seen = len(events)
            if len(events) >= total:
                break
            attempt += 1
            time.sleep(0.05)
        if attempt == attempts:
            assert False, f'Could not ensure that {total} event(s) were persisted within {attempt} attempt(s). Event count is instead currently {last_events_seen}.'

    def bulk_store_sessions(self, sessions):
        if False:
            i = 10
            return i + 15
        assert requests.post(settings.SENTRY_SNUBA + '/tests/entities/sessions/insert', data=json.dumps(sessions)).status_code == 200

    def build_session(self, **kwargs):
        if False:
            return 10
        session = {'session_id': str(uuid4()), 'distinct_id': str(uuid4()), 'status': 'ok', 'seq': 0, 'retention_days': 90, 'duration': 60.0, 'errors': 0, 'started': time.time() // 60 * 60, 'received': time.time()}
        translators = [('release', 'version', 'release'), ('environment', 'name', 'environment'), ('project_id', 'id', 'project'), ('org_id', 'id', 'organization')]
        for (key, attr, default_attr) in translators:
            if key not in kwargs:
                kwargs[key] = getattr(self, default_attr)
            val = kwargs[key]
            kwargs[key] = getattr(val, attr, val)
        session.update(kwargs)
        return session

    def store_session(self, session):
        if False:
            return 10
        self.bulk_store_sessions([session])

    def store_group(self, group):
        if False:
            while True:
                i = 10
        data = [self.__wrap_group(group)]
        assert requests.post(settings.SENTRY_SNUBA + '/tests/entities/groupedmessage/insert', data=json.dumps(data)).status_code == 200

    def store_outcome(self, group):
        if False:
            i = 10
            return i + 15
        data = [self.__wrap_group(group)]
        assert requests.post(settings.SENTRY_SNUBA + '/tests/entities/outcomes/insert', data=json.dumps(data)).status_code == 200

    def to_snuba_time_format(self, datetime_value):
        if False:
            for i in range(10):
                print('nop')
        date_format = '%Y-%m-%d %H:%M:%S%z'
        return datetime_value.strftime(date_format)

    def __wrap_group(self, group):
        if False:
            return 10
        return {'event': 'change', 'kind': 'insert', 'table': 'sentry_groupedmessage', 'columnnames': ['id', 'logger', 'level', 'message', 'status', 'times_seen', 'last_seen', 'first_seen', 'data', 'score', 'project_id', 'time_spent_total', 'time_spent_count', 'resolved_at', 'active_at', 'is_public', 'platform', 'num_comments', 'first_release_id', 'short_id'], 'columnvalues': [group.id, group.logger, group.level, group.message, group.status, group.times_seen, self.to_snuba_time_format(group.last_seen), self.to_snuba_time_format(group.first_seen), group.data, group.score, group.project.id, group.time_spent_total, group.time_spent_count, group.resolved_at, self.to_snuba_time_format(group.active_at), group.is_public, group.platform, group.num_comments, group.first_release.id if group.first_release else None, group.short_id]}

    def snuba_insert(self, events):
        if False:
            return 10
        'Write a (wrapped) event (or events) to Snuba.'
        if not isinstance(events, list):
            events = [events]
        assert requests.post(settings.SENTRY_SNUBA + '/tests/entities/events/insert', data=json.dumps(events)).status_code == 200

class BaseMetricsTestCase(SnubaTestCase):
    snuba_endpoint = '/tests/entities/{entity}/insert'

    def store_session(self, session):
        if False:
            i = 10
            return i + 15
        'Mimic relays behavior of always emitting a metric for a started session,\n        and emitting an additional one if the session is fatal\n        https://github.com/getsentry/relay/blob/e3c064e213281c36bde5d2b6f3032c6d36e22520/relay-server/src/actors/envelopes.rs#L357\n        '
        user = session.get('distinct_id')
        org_id = session['org_id']
        project_id = session['project_id']
        base_tags = {}
        if session.get('release') is not None:
            base_tags['release'] = session['release']
        if session.get('environment') is not None:
            base_tags['environment'] = session['environment']
        if session.get('abnormal_mechanism') is not None:
            base_tags['abnormal_mechanism'] = session['abnormal_mechanism']
        user_is_nil = user is None or user == '00000000-0000-0000-0000-000000000000'

        def push(type, mri: str, tags, value):
            if False:
                while True:
                    i = 10
            self.store_metric(org_id, project_id, type, mri, {**tags, **base_tags}, int(session['started'] if isinstance(session['started'], (int, float)) else to_timestamp(session['started'])), value, use_case_id=UseCaseID.SESSIONS)
        if session['seq'] == 0:
            push('counter', SessionMRI.RAW_SESSION.value, {'session.status': 'init'}, +1)
        status = session['status']
        if session.get('errors', 0) > 0 or status not in ('ok', 'exited'):
            push('set', SessionMRI.RAW_ERROR.value, {}, session['session_id'])
            if not user_is_nil:
                push('set', SessionMRI.RAW_USER.value, {'session.status': 'errored'}, user)
        elif not user_is_nil:
            push('set', SessionMRI.RAW_USER.value, {}, user)
        if status in ('abnormal', 'crashed'):
            push('counter', SessionMRI.RAW_SESSION.value, {'session.status': status}, +1)
            if not user_is_nil:
                push('set', SessionMRI.RAW_USER.value, {'session.status': status}, user)
        if status == 'exited':
            if session['duration'] is not None:
                push('distribution', SessionMRI.RAW_DURATION.value, {'session.status': status}, session['duration'])

    def bulk_store_sessions(self, sessions):
        if False:
            i = 10
            return i + 15
        for session in sessions:
            self.store_session(session)

    @classmethod
    def store_metric(cls, org_id: int, project_id: int, type: Literal['counter', 'set', 'distribution', 'gauge'], name: str, tags: Dict[str, str], timestamp: int, value: Any, use_case_id: UseCaseID, aggregation_option: Optional[AggregationOption]=None) -> None:
        if False:
            print('Hello World!')
        mapping_meta = {}

        def metric_id(key: str):
            if False:
                i = 10
                return i + 15
            assert isinstance(key, str)
            res = indexer.record(use_case_id=use_case_id, org_id=org_id, string=key)
            assert res is not None, key
            mapping_meta[str(res)] = key
            return res

        def tag_key(name):
            if False:
                while True:
                    i = 10
            assert isinstance(name, str)
            res = indexer.record(use_case_id=use_case_id, org_id=org_id, string=name)
            assert res is not None, name
            mapping_meta[str(res)] = name
            return str(res)

        def tag_value(name):
            if False:
                i = 10
                return i + 15
            assert isinstance(name, str)
            if METRIC_PATH_MAPPING[use_case_id] == UseCaseKey.PERFORMANCE:
                return name
            res = indexer.record(use_case_id=use_case_id, org_id=org_id, string=name)
            assert res is not None, name
            mapping_meta[str(res)] = name
            return res
        assert not isinstance(value, list)
        if type == 'set':
            value = [int.from_bytes(hashlib.md5(str(value).encode()).digest()[:8], 'big')]
        elif type == 'distribution':
            value = [value]
        elif type == 'gauge':
            if not isinstance(value, Dict):
                value = {'min': value, 'max': value, 'sum': value, 'count': int(value), 'last': value}
        msg = {'org_id': org_id, 'project_id': project_id, 'metric_id': metric_id(name), 'timestamp': timestamp, 'tags': {tag_key(key): tag_value(value) for (key, value) in tags.items()}, 'type': {'counter': 'c', 'set': 's', 'distribution': 'd', 'gauge': 'g'}[type], 'value': value, 'retention_days': 90, 'use_case_id': use_case_id.value, 'sentry_received_timestamp': timestamp + 10, 'version': 2 if METRIC_PATH_MAPPING[use_case_id] == UseCaseKey.PERFORMANCE else 1}
        msg['mapping_meta'] = {}
        msg['mapping_meta'][msg['type']] = mapping_meta
        if aggregation_option:
            msg['aggregation_option'] = aggregation_option.value
        if METRIC_PATH_MAPPING[use_case_id] == UseCaseKey.PERFORMANCE:
            entity = f'generic_metrics_{type}s'
        else:
            entity = f'metrics_{type}s'
        cls.__send_buckets([msg], entity)

    @classmethod
    def __send_buckets(cls, buckets, entity):
        if False:
            print('Hello World!')
        if entity.startswith('generic_'):
            codec = sentry_kafka_schemas.get_codec('snuba-generic-metrics')
        else:
            codec = sentry_kafka_schemas.get_codec('snuba-metrics')
        for bucket in buckets:
            codec.validate(bucket)
        assert requests.post(settings.SENTRY_SNUBA + cls.snuba_endpoint.format(entity=entity), data=json.dumps(buckets)).status_code == 200

class BaseMetricsLayerTestCase(BaseMetricsTestCase):
    ENTITY_SHORTHANDS = {'c': 'counter', 's': 'set', 'd': 'distribution', 'g': 'gauge'}
    MOCK_DATETIME = (django_timezone.now() - timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)

    @property
    def now(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the current time instance that will be used throughout the tests of the metrics layer.\n\n        This method has to be implemented in all the children classes because it serves as a way to standardize\n        access to time.\n        '
        raise NotImplementedError

    def _extract_entity_from_mri(self, mri_string: str) -> Optional[str]:
        if False:
            return 10
        '\n        Extracts the entity name from the MRI given a map of shorthands used to represent that entity in the MRI.\n        '
        if (parsed_mri := parse_mri(mri_string)) is not None:
            return self.ENTITY_SHORTHANDS[parsed_mri.entity]
        else:
            return None

    def _store_metric(self, name: str, tags: Dict[str, str], value: int | float | Dict[str, int | float], use_case_id: UseCaseID, type: Optional[str]=None, org_id: Optional[int]=None, project_id: Optional[int]=None, days_before_now: int=0, hours_before_now: int=0, minutes_before_now: int=0, seconds_before_now: int=0, aggregation_option: Optional[AggregationOption]=None):
        if False:
            while True:
                i = 10
        self.store_metric(org_id=self.organization.id if org_id is None else org_id, project_id=self.project.id if project_id is None else project_id, type=self._extract_entity_from_mri(name) if type is None else type, name=name, tags=tags, timestamp=int(self.adjust_timestamp(self.now - timedelta(days=days_before_now, hours=hours_before_now, minutes=minutes_before_now, seconds=seconds_before_now)).timestamp()), value=value, use_case_id=use_case_id, aggregation_option=aggregation_option)

    @staticmethod
    def adjust_timestamp(time: datetime) -> datetime:
        if False:
            print('Hello World!')
        return time - timedelta(seconds=1)

    def store_performance_metric(self, name: str, tags: Dict[str, str], value: int | float | Dict[str, int | float], type: Optional[str]=None, org_id: Optional[int]=None, project_id: Optional[int]=None, days_before_now: int=0, hours_before_now: int=0, minutes_before_now: int=0, seconds_before_now: int=0, aggregation_option: Optional[AggregationOption]=None):
        if False:
            for i in range(10):
                print('nop')
        self._store_metric(type=type, name=name, tags=tags, value=value, org_id=org_id, project_id=project_id, use_case_id=UseCaseID.TRANSACTIONS, days_before_now=days_before_now, hours_before_now=hours_before_now, minutes_before_now=minutes_before_now, seconds_before_now=seconds_before_now, aggregation_option=aggregation_option)

    def store_release_health_metric(self, name: str, tags: Dict[str, str], value: int, type: Optional[str]=None, org_id: Optional[int]=None, project_id: Optional[int]=None, days_before_now: int=0, hours_before_now: int=0, minutes_before_now: int=0, seconds_before_now: int=0):
        if False:
            print('Hello World!')
        self._store_metric(type=type, name=name, tags=tags, value=value, org_id=org_id, project_id=project_id, use_case_id=UseCaseID.SESSIONS, days_before_now=days_before_now, hours_before_now=hours_before_now, minutes_before_now=minutes_before_now, seconds_before_now=seconds_before_now)

    def store_custom_metric(self, name: str, tags: Dict[str, str], value: int | float | Dict[str, int | float], type: Optional[str]=None, org_id: Optional[int]=None, project_id: Optional[int]=None, days_before_now: int=0, hours_before_now: int=0, minutes_before_now: int=0, seconds_before_now: int=0, aggregation_option: Optional[AggregationOption]=None):
        if False:
            i = 10
            return i + 15
        self._store_metric(type=type, name=name, tags=tags, value=value, org_id=org_id, project_id=project_id, use_case_id=UseCaseID.CUSTOM, days_before_now=days_before_now, hours_before_now=hours_before_now, minutes_before_now=minutes_before_now, seconds_before_now=seconds_before_now, aggregation_option=aggregation_option)

    def build_metrics_query(self, select: Sequence[MetricField], project_ids: Optional[Sequence[int]]=None, where: Optional[Sequence[Union[BooleanCondition, Condition, MetricConditionField]]]=None, having: Optional[ConditionGroup]=None, groupby: Optional[Sequence[MetricGroupByField]]=None, orderby: Optional[Sequence[MetricOrderByField]]=None, limit: Optional[Limit]=None, offset: Optional[Offset]=None, include_totals: bool=True, include_series: bool=True, before_now: Optional[str]=None, granularity: Optional[str]=None):
        if False:
            return 10
        (start, end, granularity_in_seconds) = get_date_range({'statsPeriod': before_now, 'interval': granularity})
        return MetricsQuery(org_id=self.organization.id, project_ids=[self.project.id] + (project_ids if project_ids is not None else []), select=select, start=start, end=end, granularity=Granularity(granularity=granularity_in_seconds), where=where, having=having, groupby=groupby, orderby=orderby, limit=limit, offset=offset, include_totals=include_totals, include_series=include_series)

class MetricsEnhancedPerformanceTestCase(BaseMetricsLayerTestCase, TestCase):
    TYPE_MAP = {'metrics_distributions': 'distribution', 'metrics_sets': 'set', 'metrics_counters': 'counter', 'metrics_gauges': 'gauge'}
    ENTITY_MAP = {'transaction.duration': 'metrics_distributions', 'span.duration': 'metrics_distributions', 'span.self_time': 'metrics_distributions', 'http.response_content_length': 'metrics_distributions', 'http.decoded_response_content_length': 'metrics_distributions', 'http.response_transfer_size': 'metrics_distributions', 'measurements.lcp': 'metrics_distributions', 'measurements.fp': 'metrics_distributions', 'measurements.fcp': 'metrics_distributions', 'measurements.fid': 'metrics_distributions', 'measurements.cls': 'metrics_distributions', 'measurements.frames_frozen_rate': 'metrics_distributions', 'measurements.time_to_initial_display': 'metrics_distributions', 'spans.http': 'metrics_distributions', 'user': 'metrics_sets'}
    ON_DEMAND_KEY_MAP = {'c': TransactionMetricKey.COUNT_ON_DEMAND.value, 'd': TransactionMetricKey.DIST_ON_DEMAND.value, 's': TransactionMetricKey.SET_ON_DEMAND.value}
    ON_DEMAND_MRI_MAP = {'c': TransactionMRI.COUNT_ON_DEMAND.value, 'd': TransactionMRI.DIST_ON_DEMAND.value, 's': TransactionMRI.SET_ON_DEMAND.value}
    ON_DEMAND_ENTITY_MAP = {'c': EntityKey.MetricsCounters.value, 'd': EntityKey.MetricsDistributions.value, 's': EntityKey.MetricsSets.value}
    METRIC_STRINGS = []
    DEFAULT_METRIC_TIMESTAMP = datetime(2015, 1, 1, 10, 15, 0, tzinfo=timezone.utc)

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self._index_metric_strings()

    def _index_metric_strings(self):
        if False:
            for i in range(10):
                print('nop')
        strings = ['transaction', 'environment', 'http.status', 'transaction.status', METRIC_TOLERATED_TAG_VALUE, METRIC_SATISFIED_TAG_VALUE, METRIC_FRUSTRATED_TAG_VALUE, METRIC_SATISFACTION_TAG_KEY, *self.METRIC_STRINGS, *list(SPAN_STATUS_NAME_TO_CODE.keys()), *list(METRICS_MAP.values())]
        org_strings = {self.organization.id: set(strings)}
        indexer.bulk_record({UseCaseID.TRANSACTIONS: org_strings})

    def store_transaction_metric(self, value: list[Any] | Any, metric: str='transaction.duration', internal_metric: Optional[str]=None, entity: Optional[str]=None, tags: Optional[Dict[str, str]]=None, timestamp: Optional[datetime]=None, project: Optional[int]=None, use_case_id: UseCaseID=UseCaseID.TRANSACTIONS, aggregation_option: Optional[AggregationOption]=None):
        if False:
            i = 10
            return i + 15
        internal_metric = METRICS_MAP[metric] if internal_metric is None else internal_metric
        entity = self.ENTITY_MAP[metric] if entity is None else entity
        org_id = self.organization.id
        if tags is None:
            tags = {}
        if timestamp is None:
            metric_timestamp = self.DEFAULT_METRIC_TIMESTAMP.timestamp()
        else:
            metric_timestamp = timestamp.timestamp()
        if project is None:
            project = self.project.id
        if not isinstance(value, list):
            value = [value]
        for subvalue in value:
            self.store_metric(org_id, project, self.TYPE_MAP[entity], internal_metric, tags, int(metric_timestamp), subvalue, use_case_id=use_case_id, aggregation_option=aggregation_option)

    def store_on_demand_metric(self, value: list[Any] | Any, spec: OnDemandMetricSpec, additional_tags: Optional[Dict[str, str]]=None, timestamp: Optional[datetime]=None):
        if False:
            return 10
        project: Project = default_project
        metric_spec = spec.to_metric_spec(project)
        metric_spec_tags = metric_spec['tags'] or [] if metric_spec else []
        spec_tags = {i['key']: i.get('value') or i.get('field') for i in metric_spec_tags}
        metric_type = spec._metric_type
        self.store_transaction_metric(value, metric=self.ON_DEMAND_KEY_MAP[metric_type], internal_metric=self.ON_DEMAND_MRI_MAP[metric_type], entity=self.ON_DEMAND_ENTITY_MAP[metric_type], tags={**spec_tags, **additional_tags}, timestamp=timestamp)
        return spec

    def store_span_metric(self, value: List[int] | int, metric: str='span.self_time', internal_metric: Optional[str]=None, entity: Optional[str]=None, tags: Optional[Dict[str, str]]=None, timestamp: Optional[datetime]=None, project: Optional[int]=None, use_case_id: UseCaseID=UseCaseID.SPANS):
        if False:
            print('Hello World!')
        internal_metric = SPAN_METRICS_MAP[metric] if internal_metric is None else internal_metric
        entity = self.ENTITY_MAP[metric] if entity is None else entity
        org_id = self.organization.id
        if tags is None:
            tags = {}
        if timestamp is None:
            metric_timestamp = self.DEFAULT_METRIC_TIMESTAMP.timestamp()
        else:
            metric_timestamp = timestamp.timestamp()
        if project is None:
            project = self.project.id
        if not isinstance(value, list):
            value = [value]
        for subvalue in value:
            self.store_metric(org_id, project, self.TYPE_MAP[entity], internal_metric, tags, int(metric_timestamp), subvalue, use_case_id=use_case_id)

    def wait_for_metric_count(self, project, total, metric='transaction.duration', mri=TransactionMRI.DURATION.value, attempts=2):
        if False:
            i = 10
            return i + 15
        attempt = 0
        metrics_query = self.build_metrics_query(before_now='1d', granularity='1d', select=[MetricField(op='count', metric_mri=mri)], include_series=False)
        while attempt < attempts:
            data = get_series([project], metrics_query=metrics_query, use_case_id=UseCaseID.TRANSACTIONS)
            count = data['groups'][0]['totals'][f'count({metric})']
            if count >= total:
                break
            attempt += 1
            time.sleep(0.05)
        if attempt == attempts:
            assert False, f'Could not ensure that {total} metric(s) were persisted within {attempt} attempt(s).'

class BaseIncidentsTest(SnubaTestCase):

    def create_event(self, timestamp, fingerprint=None, user=None):
        if False:
            while True:
                i = 10
        event_id = uuid4().hex
        if fingerprint is None:
            fingerprint = event_id
        data = {'event_id': event_id, 'fingerprint': [fingerprint], 'timestamp': iso_format(timestamp), 'type': 'error', 'exception': [{'type': 'Foo'}]}
        if user:
            data['user'] = user
        return self.store_event(data=data, project_id=self.project.id)

    @cached_property
    def now(self):
        if False:
            while True:
                i = 10
        return django_timezone.now().replace(minute=0, second=0, microsecond=0)

@pytest.mark.snuba
@requires_snuba
class OutcomesSnubaTest(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        assert requests.post(settings.SENTRY_SNUBA + '/tests/outcomes/drop').status_code == 200

    def store_outcomes(self, outcome, num_times=1):
        if False:
            while True:
                i = 10
        outcomes = []
        for _ in range(num_times):
            outcome_copy = outcome.copy()
            outcome_copy['timestamp'] = outcome_copy['timestamp'].strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            outcomes.append(outcome_copy)
        assert requests.post(settings.SENTRY_SNUBA + '/tests/entities/outcomes/insert', data=json.dumps(outcomes)).status_code == 200

@pytest.mark.snuba
@requires_snuba
class ProfilesSnubaTestCase(TestCase, BaseTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        assert requests.post(settings.SENTRY_SNUBA + '/tests/functions/drop').status_code == 200

    def store_functions(self, functions, project, transaction=None, extras=None, timestamp=None):
        if False:
            while True:
                i = 10
        if timestamp is None:
            timestamp = before_now(minutes=10)
        if transaction is None:
            transaction = load_data('transaction', timestamp=timestamp)
        profile_context = transaction.setdefault('contexts', {}).setdefault('profile', {})
        if profile_context.get('profile_id') is None:
            profile_context['profile_id'] = uuid4().hex
        profile_id = profile_context.get('profile_id')
        timestamp = transaction['timestamp']
        self.store_event(transaction, project_id=project.id)
        functions = [{**function, 'fingerprint': self.function_fingerprint(function)} for function in functions]
        functions_payload = {'project_id': project.id, 'profile_id': profile_id, 'transaction_name': transaction['transaction'], 'platform': transaction['platform'], 'functions': functions, 'timestamp': timestamp, 'retention_days': 90}
        if extras is not None:
            functions_payload.update(extras)
        response = requests.post(settings.SENTRY_SNUBA + '/tests/entities/functions/insert', json=[functions_payload])
        assert response.status_code == 200
        return {'transaction': transaction, 'functions': functions}

    def function_fingerprint(self, function):
        if False:
            while True:
                i = 10
        hasher = hashlib.md5()
        if function.get('package') is not None:
            hasher.update(function['package'].encode())
        else:
            hasher.update(b'')
        hasher.update(b':')
        hasher.update(function['function'].encode())
        return int(hasher.hexdigest()[:8], 16)

@pytest.mark.snuba
@requires_snuba
class ReplaysSnubaTestCase(TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        assert requests.post(settings.SENTRY_SNUBA + '/tests/replays/drop').status_code == 200

    def store_replays(self, replay):
        if False:
            return 10
        response = requests.post(settings.SENTRY_SNUBA + '/tests/entities/replays/insert', json=[replay])
        assert response.status_code == 200

    def mock_event_links(self, timestamp, project_id, level, replay_id, event_id):
        if False:
            for i in range(10):
                print('nop')
        event = self.store_event(data={'timestamp': int(timestamp.timestamp()), 'event_id': event_id, 'level': level, 'message': 'testing', 'contexts': {'replay': {'replay_id': replay_id}}}, project_id=project_id)
        return transform_event_for_linking_payload(replay_id, event)

class ReplaysAcceptanceTestCase(AcceptanceTestCase, SnubaTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.now = datetime.utcnow().replace(tzinfo=timezone.utc)
        super().setUp()
        self.drop_replays()
        patcher = mock.patch('django.utils.timezone.now', return_value=self.now)
        patcher.start()
        self.addCleanup(patcher.stop)

    def drop_replays(self):
        if False:
            print('Hello World!')
        assert requests.post(settings.SENTRY_SNUBA + '/tests/replays/drop').status_code == 200

    def store_replays(self, replays):
        if False:
            for i in range(10):
                print('nop')
        assert len(replays) >= 2, 'You need to store at least 2 replay events for the replay to be considered valid'
        response = requests.post(settings.SENTRY_SNUBA + '/tests/entities/replays/insert', json=replays)
        assert response.status_code == 200

    def store_replay_segments(self, replay_id: str, project_id: str, segment_id: int, segment):
        if False:
            i = 10
            return i + 15
        f = File.objects.create(name='rr:{segment_id}', type='replay.recording')
        f.putfile(BytesIO(compress(dumps_htmlsafe(segment).encode())))
        ReplayRecordingSegment.objects.create(replay_id=replay_id, project_id=project_id, segment_id=segment_id, file_id=f.id)

class IntegrationRepositoryTestCase(APITestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.login_as(self.user)

    @pytest.fixture(autouse=True)
    def responses_context(self):
        if False:
            return 10
        with responses.mock:
            yield

    def add_create_repository_responses(self, repository_config):
        if False:
            while True:
                i = 10
        raise NotImplementedError(f'implement for {type(self).__module__}.{type(self).__name__}')

    @assume_test_silo_mode(SiloMode.REGION)
    def create_repository(self, repository_config, integration_id, organization_slug=None, add_responses=True):
        if False:
            i = 10
            return i + 15
        if add_responses:
            self.add_create_repository_responses(repository_config)
        if not integration_id:
            data = {'provider': self.provider_name, 'identifier': repository_config['id']}
        else:
            data = {'provider': self.provider_name, 'installation': integration_id, 'identifier': repository_config['id']}
        response = self.client.post(path=reverse('sentry-api-0-organization-repositories', args=[organization_slug or self.organization.slug]), data=data)
        return response

    def assert_error_message(self, response, error_type, error_message):
        if False:
            while True:
                i = 10
        assert response.data['error_type'] == error_type
        assert error_message in response.data['errors']['__all__']

class ReleaseCommitPatchTest(APITestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        user = self.create_user(is_staff=False, is_superuser=False)
        self.org = self.create_organization()
        self.org.save()
        team = self.create_team(organization=self.org)
        self.project = self.create_project(name='foo', organization=self.org, teams=[team])
        self.create_member(teams=[team], user=user, organization=self.org)
        self.login_as(user=user)

    @cached_property
    def url(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError(f'implement for {type(self).__module__}.{type(self).__name__}')

    def assert_commit(self, commit, repo_id, key, author_id, message):
        if False:
            for i in range(10):
                print('nop')
        assert commit.organization_id == self.org.id
        assert commit.repository_id == repo_id
        assert commit.key == key
        assert commit.author_id == author_id
        assert commit.message == message

    def assert_file_change(self, file_change, type, filename, commit_id):
        if False:
            i = 10
            return i + 15
        assert file_change.type == type
        assert file_change.filename == filename
        assert file_change.commit_id == commit_id

class SetRefsTestCase(APITestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.user = self.create_user(is_staff=False, is_superuser=False)
        self.org = self.create_organization()
        self.team = self.create_team(organization=self.org)
        self.project = self.create_project(name='foo', organization=self.org, teams=[self.team])
        self.create_member(teams=[self.team], user=self.user, organization=self.org)
        self.login_as(user=self.user)
        self.group = self.create_group(project=self.project)
        self.repo = Repository.objects.create(organization_id=self.org.id, name='test/repo')

    def assert_fetch_commits(self, mock_fetch_commit, prev_release_id, release_id, refs):
        if False:
            while True:
                i = 10
        assert len(mock_fetch_commit.method_calls) == 1
        kwargs = mock_fetch_commit.method_calls[0][2]['kwargs']
        assert kwargs == {'prev_release_id': prev_release_id, 'refs': refs, 'release_id': release_id, 'user_id': self.user.id}

    def assert_head_commit(self, head_commit, commit_key, release_id=None):
        if False:
            return 10
        assert self.org.id == head_commit.organization_id
        assert self.repo.id == head_commit.repository_id
        if release_id:
            assert release_id == head_commit.release_id
        else:
            assert self.release.id == head_commit.release_id
        self.assert_commit(head_commit.commit, commit_key)

    def assert_commit(self, commit, key):
        if False:
            print('Hello World!')
        assert self.org.id == commit.organization_id
        assert self.repo.id == commit.repository_id
        assert commit.key == key

class OrganizationDashboardWidgetTestCase(APITestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.login_as(self.user)
        self.dashboard = Dashboard.objects.create(title='Dashboard 1', created_by_id=self.user.id, organization=self.organization)
        self.anon_users_query = {'name': 'Anonymous Users', 'fields': ['count()'], 'aggregates': ['count()'], 'columns': [], 'fieldAliases': ['Count Alias'], 'conditions': '!has:user.email'}
        self.known_users_query = {'name': 'Known Users', 'fields': ['count_unique(user.email)'], 'aggregates': ['count_unique(user.email)'], 'columns': [], 'fieldAliases': [], 'conditions': 'has:user.email'}
        self.geo_errors_query = {'name': 'Errors by Geo', 'fields': ['count()', 'geo.country_code'], 'aggregates': ['count()'], 'columns': ['geo.country_code'], 'fieldAliases': [], 'conditions': 'has:geo.country_code'}

    def do_request(self, method, url, data=None):
        if False:
            for i in range(10):
                print('nop')
        func = getattr(self.client, method)
        return func(url, data=data)

    def assert_widget_queries(self, widget_id, data):
        if False:
            return 10
        result_queries = DashboardWidgetQuery.objects.filter(widget_id=widget_id).order_by('order')
        for (ds, expected_ds) in zip(result_queries, data):
            assert ds.name == expected_ds['name']
            assert ds.fields == expected_ds['fields']
            assert ds.conditions == expected_ds['conditions']

    def assert_widget(self, widget, order, title, display_type, queries=None):
        if False:
            i = 10
            return i + 15
        assert widget.order == order
        assert widget.display_type == display_type
        assert widget.title == title
        if not queries:
            return
        self.assert_widget_queries(widget.id, queries)

    def assert_widget_data(self, data, title, display_type, queries=None):
        if False:
            print('Hello World!')
        assert data['displayType'] == display_type
        assert data['title'] == title
        if not queries:
            return
        self.assert_widget_queries(data['id'], queries)

    def assert_serialized_widget_query(self, data, widget_data_source):
        if False:
            return 10
        if 'id' in data:
            assert data['id'] == str(widget_data_source.id)
        if 'name' in data:
            assert data['name'] == widget_data_source.name
        if 'fields' in data:
            assert data['fields'] == widget_data_source.fields
        if 'conditions' in data:
            assert data['conditions'] == widget_data_source.conditions
        if 'orderby' in data:
            assert data['orderby'] == widget_data_source.orderby
        if 'aggregates' in data:
            assert data['aggregates'] == widget_data_source.aggregates
        if 'columns' in data:
            assert data['columns'] == widget_data_source.columns
        if 'fieldAliases' in data:
            assert data['fieldAliases'] == widget_data_source.field_aliases

    def get_widgets(self, dashboard_id):
        if False:
            return 10
        return DashboardWidget.objects.filter(dashboard_id=dashboard_id).order_by('order')

    def assert_serialized_widget(self, data, expected_widget):
        if False:
            i = 10
            return i + 15
        if 'id' in data:
            assert data['id'] == str(expected_widget.id)
        if 'title' in data:
            assert data['title'] == expected_widget.title
        if 'interval' in data:
            assert data['interval'] == expected_widget.interval
        if 'limit' in data:
            assert data['limit'] == expected_widget.limit
        if 'displayType' in data:
            assert data['displayType'] == DashboardWidgetDisplayTypes.get_type_name(expected_widget.display_type)
        if 'layout' in data:
            assert data['layout'] == expected_widget.detail['layout']

    def create_user_member_role(self):
        if False:
            for i in range(10):
                print('nop')
        self.user = self.create_user(is_superuser=False)
        self.create_member(user=self.user, organization=self.organization, role='member', teams=[self.team])
        self.login_as(self.user)

@pytest.mark.migrations
class TestMigrations(TransactionTestCase):
    """
    From https://www.caktusgroup.com/blog/2016/02/02/writing-unit-tests-django-migrations/

    Note that when running these tests locally you will need to set the `MIGRATIONS_TEST_MIGRATE=1`
    environmental variable for these to pass.
    """

    @property
    def app(self):
        if False:
            print('Hello World!')
        return 'sentry'

    @property
    def migrate_from(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError(f'implement for {type(self).__module__}.{type(self).__name__}')

    @property
    def migrate_to(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError(f'implement for {type(self).__module__}.{type(self).__name__}')

    @property
    def connection(self):
        if False:
            return 10
        return 'default'

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        migrate_from = [(self.app, self.migrate_from)]
        migrate_to = [(self.app, self.migrate_to)]
        connection = connections[self.connection]
        self.setup_initial_state()
        executor = MigrationExecutor(connection)
        matching_migrations = [m for m in executor.loader.applied_migrations if m[0] == self.app]
        if not matching_migrations:
            raise AssertionError('no migrations detected!\n\ntry running this test with `MIGRATIONS_TEST_MIGRATE=1 pytest ...`')
        self.current_migration = [max(matching_migrations)]
        old_apps = executor.loader.project_state(migrate_from).apps
        executor.migrate(migrate_from)
        self.setup_before_migration(old_apps)
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(migrate_to)
        self.apps = executor.loader.project_state(migrate_to).apps

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(self.current_migration)

    def setup_initial_state(self):
        if False:
            print('Hello World!')
        pass

    def setup_before_migration(self, apps):
        if False:
            while True:
                i = 10
        pass

class SCIMTestCase(APITestCase):

    def setUp(self, provider='dummy'):
        if False:
            while True:
                i = 10
        super().setUp()
        with assume_test_silo_mode(SiloMode.CONTROL):
            self.auth_provider_inst = AuthProviderModel(organization_id=self.organization.id, provider=provider)
            self.auth_provider_inst.enable_scim(self.user)
            self.auth_provider_inst.save()
            self.scim_user = ApiToken.objects.get(token=self.auth_provider_inst.get_scim_token()).user
        self.login_as(user=self.scim_user)

class SCIMAzureTestCase(SCIMTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        auth.register(ACTIVE_DIRECTORY_PROVIDER_NAME, DummyProvider)
        super().setUp(provider=ACTIVE_DIRECTORY_PROVIDER_NAME)
        self.addCleanup(auth.unregister, ACTIVE_DIRECTORY_PROVIDER_NAME, DummyProvider)

class ActivityTestCase(TestCase):

    @assume_test_silo_mode(SiloMode.CONTROL)
    def another_user(self, email_string, team=None, alt_email_string=None):
        if False:
            while True:
                i = 10
        user = self.create_user(email_string)
        if alt_email_string:
            UserEmail.objects.create(email=alt_email_string, user=user)
            assert UserEmail.objects.filter(user=user, email=alt_email_string).update(is_verified=True)
        assert UserEmail.objects.filter(user=user, email=user.email).update(is_verified=True)
        self.create_member(user=user, organization=self.org, teams=[team] if team else None)
        return user

    def another_commit(self, order, name, user, repository, alt_email_string=None):
        if False:
            i = 10
            return i + 15
        commit = Commit.objects.create(key=name * 40, repository_id=repository.id, organization_id=self.org.id, author=CommitAuthor.objects.create(organization_id=self.org.id, name=user.name, email=alt_email_string or user.email))
        ReleaseCommit.objects.create(organization_id=self.org.id, release=self.release, commit=commit, order=order)
        return commit

    def another_release(self, name):
        if False:
            print('Hello World!')
        release = Release.objects.create(version=name * 40, organization_id=self.project.organization_id, date_released=django_timezone.now())
        release.add_project(self.project)
        release.add_project(self.project2)
        deploy = Deploy.objects.create(release=release, organization_id=self.org.id, environment_id=self.environment.id)
        return (release, deploy)

    def get_notification_uuid(self, text: str) -> str:
        if False:
            print('Hello World!')
        result = re.search('notification.*_uuid=([a-zA-Z0-9-]+)', text)
        assert result is not None
        return result[1]

class SlackActivityNotificationTest(ActivityTestCase):

    @cached_property
    def adapter(self):
        if False:
            while True:
                i = 10
        return mail_adapter

    def setUp(self):
        if False:
            return 10
        with assume_test_silo_mode(SiloMode.CONTROL):
            NotificationSetting.objects.update_settings(ExternalProviders.SLACK, NotificationSettingTypes.WORKFLOW, NotificationSettingOptionValues.ALWAYS, user_id=self.user.id)
            NotificationSetting.objects.update_settings(ExternalProviders.SLACK, NotificationSettingTypes.DEPLOY, NotificationSettingOptionValues.ALWAYS, user_id=self.user.id)
            NotificationSetting.objects.update_settings(ExternalProviders.SLACK, NotificationSettingTypes.ISSUE_ALERTS, NotificationSettingOptionValues.ALWAYS, user_id=self.user.id)
            UserOption.objects.create(user=self.user, key='self_notifications', value='1')
            self.integration = install_slack(self.organization)
            self.idp = IdentityProvider.objects.create(type='slack', external_id='TXXXXXXX1', config={})
            self.identity = Identity.objects.create(external_id='UXXXXXXX1', idp=self.idp, user=self.user, status=IdentityStatus.VALID, scopes=[])
        responses.add(method=responses.POST, url='https://slack.com/api/chat.postMessage', body='{"ok": true}', status=200, content_type='application/json')
        self.name = self.user.get_display_name()
        self.short_id = self.group.qualified_short_id

    @pytest.fixture(autouse=True)
    def responses_context(self):
        if False:
            for i in range(10):
                print('nop')
        with responses.mock:
            yield

    def assert_performance_issue_attachments(self, attachment, project_slug, referrer, alert_type='workflow'):
        if False:
            for i in range(10):
                print('nop')
        assert attachment['title'] == 'N+1 Query'
        assert attachment['text'] == 'db - SELECT `books_author`.`id`, `books_author`.`name` FROM `books_author` WHERE `books_author`.`id` = %s LIMIT 21'
        notification_uuid = self.get_notification_uuid(attachment['title_link'])
        assert attachment['footer'] == f'{project_slug} | production | <http://testserver/settings/account/notifications/{alert_type}/?referrer={referrer}&notification_uuid={notification_uuid}|Notification Settings>'

    def assert_generic_issue_attachments(self, attachment, project_slug, referrer, alert_type='workflow'):
        if False:
            while True:
                i = 10
        assert attachment['title'] == TEST_ISSUE_OCCURRENCE.issue_title
        assert attachment['text'] == TEST_ISSUE_OCCURRENCE.evidence_display[0].value
        notification_uuid = self.get_notification_uuid(attachment['title_link'])
        assert attachment['footer'] == f'{project_slug} | <http://testserver/settings/account/notifications/{alert_type}/?referrer={referrer}&notification_uuid={notification_uuid}|Notification Settings>'

class MSTeamsActivityNotificationTest(ActivityTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        with assume_test_silo_mode(SiloMode.CONTROL):
            NotificationSetting.objects.update_settings(ExternalProviders.MSTEAMS, NotificationSettingTypes.WORKFLOW, NotificationSettingOptionValues.ALWAYS, user_id=self.user.id)
            NotificationSetting.objects.update_settings(ExternalProviders.MSTEAMS, NotificationSettingTypes.ISSUE_ALERTS, NotificationSettingOptionValues.ALWAYS, user_id=self.user.id)
            NotificationSetting.objects.update_settings(ExternalProviders.MSTEAMS, NotificationSettingTypes.DEPLOY, NotificationSettingOptionValues.ALWAYS, user_id=self.user.id)
            UserOption.objects.create(user=self.user, key='self_notifications', value='1')
        self.tenant_id = '50cccd00-7c9c-4b32-8cda-58a084f9334a'
        self.integration = self.create_integration(self.organization, self.tenant_id, metadata={'access_token': 'xoxb-xxxxxxxxx-xxxxxxxxxx-xxxxxxxxxxxx', 'service_url': 'https://testserviceurl.com/testendpoint/', 'installation_type': 'tenant', 'expires_at': 1234567890, 'tenant_id': self.tenant_id}, name='Personal Installation', provider='msteams')
        self.idp = self.create_identity_provider(integration=self.integration, type='msteams', external_id=self.tenant_id, config={})
        self.user_id_1 = '29:1XJKJMvc5GBtc2JwZq0oj8tHZmzrQgFmB39ATiQWA85gQtHieVkKilBZ9XHoq9j7Zaqt7CZ-NJWi7me2kHTL3Bw'
        self.user_1 = self.user
        self.identity_1 = self.create_identity(user=self.user_1, identity_provider=self.idp, external_id=self.user_id_1)

@pytest.mark.usefixtures('reset_snuba')
class MetricsAPIBaseTestCase(BaseMetricsLayerTestCase, APITestCase):

    def build_and_store_session(self, days_before_now: int=0, hours_before_now: int=0, minutes_before_now: int=0, seconds_before_now: int=0, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs['started'] = self.adjust_timestamp(self.now - timedelta(days=days_before_now, hours=hours_before_now, minutes=minutes_before_now, seconds=seconds_before_now)).timestamp()
        self.store_session(self.build_session(**kwargs))

class OrganizationMetricMetaIntegrationTestCase(MetricsAPIBaseTestCase):

    def __indexer_record(self, org_id: int, value: str) -> int:
        if False:
            i = 10
            return i + 15
        return indexer.record(use_case_id=UseCaseID.SESSIONS, org_id=org_id, string=value)

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.login_as(user=self.user)
        now = int(time.time())
        org_id = self.organization.id
        self.store_metric(org_id=org_id, project_id=self.project.id, name='metric1', timestamp=now, tags={'tag1': 'value1', 'tag2': 'value2'}, type='counter', value=1, use_case_id=UseCaseID.SESSIONS)
        self.store_metric(org_id=org_id, project_id=self.project.id, name='metric1', timestamp=now, tags={'tag3': 'value3'}, type='counter', value=1, use_case_id=UseCaseID.SESSIONS)
        self.store_metric(org_id=org_id, project_id=self.project.id, name='metric2', timestamp=now, tags={'tag4': 'value3', 'tag1': 'value2', 'tag2': 'value1'}, type='set', value=123, use_case_id=UseCaseID.SESSIONS)
        self.store_metric(org_id=org_id, project_id=self.project.id, name='metric3', timestamp=now, tags={}, type='set', value=123, use_case_id=UseCaseID.SESSIONS)

class MonitorTestCase(APITestCase):

    def _create_monitor(self, **kwargs):
        if False:
            return 10
        return Monitor.objects.create(organization_id=self.organization.id, project_id=self.project.id, type=MonitorType.CRON_JOB, config={'schedule': '* * * * *', 'schedule_type': ScheduleType.CRONTAB, 'checkin_margin': None, 'max_runtime': None}, **kwargs)

    def _create_monitor_environment(self, monitor, name='production', **kwargs):
        if False:
            i = 10
            return i + 15
        environment = Environment.get_or_create(project=self.project, name=name)
        monitorenvironment_defaults = {'status': monitor.status, **kwargs}
        return MonitorEnvironment.objects.create(monitor=monitor, environment=environment, **monitorenvironment_defaults)

    def _create_alert_rule(self, monitor):
        if False:
            return 10
        conditions = [{'id': 'sentry.rules.conditions.first_seen_event.FirstSeenEventCondition'}, {'id': 'sentry.rules.conditions.regression_event.RegressionEventCondition'}, {'id': 'sentry.rules.filters.tagged_event.TaggedEventFilter', 'key': 'monitor.slug', 'match': 'eq', 'value': monitor.slug}]
        rule = Creator(name='New Cool Rule', owner=None, project=self.project, action_match='any', filter_match='all', conditions=conditions, actions=[], frequency=5, environment=self.environment.id).call()
        rule.update(source=RuleSource.CRON_MONITOR)
        config = monitor.config
        config['alert_rule_id'] = rule.id
        monitor.config = config
        monitor.save()
        return rule

class MonitorIngestTestCase(MonitorTestCase):
    """
    Base test case which provides support for both styles of legacy ingestion
    endpoints, as well as sets up token and DSN authentication helpers
    """

    @property
    def endpoint_with_org(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError(f'implement for {type(self).__module__}.{type(self).__name__}')

    @property
    def dsn_auth_headers(self):
        if False:
            while True:
                i = 10
        return {'HTTP_AUTHORIZATION': f'DSN {self.project_key.dsn_public}'}

    @property
    def token_auth_headers(self):
        if False:
            return 10
        return {'HTTP_AUTHORIZATION': f'Bearer {self.token.token}'}

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.project_key = self.create_project_key()
        sentry_app = self.create_sentry_app(organization=self.organization, scopes=['project:write'])
        app = self.create_sentry_app_installation(slug=sentry_app.slug, organization=self.organization)
        self.token = self.create_internal_integration_token(app, user=self.user)

    def _get_path_functions(self):
        if False:
            return 10
        return (lambda monitor_slug: reverse(self.endpoint, args=[monitor_slug]), lambda monitor_slug: reverse(self.endpoint_with_org, args=[self.organization.slug, monitor_slug]))

class IntegratedApiTestCase(BaseTestCase):

    def should_call_api_without_proxying(self) -> bool:
        if False:
            i = 10
            return i + 15
        return not IntegrationProxyClient.determine_whether_should_proxy_to_control()