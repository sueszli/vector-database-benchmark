from datetime import datetime, timedelta
from typing import Any, List, Literal, Optional, Type
from zoneinfo import ZoneInfo
from dateutil.parser import isoparse
from freezegun import freeze_time
from pydantic import BaseModel
from posthog.hogql_queries.query_runner import QueryResponse, QueryRunner, RunnableQueryNode
from posthog.models.team.team import Team
from posthog.test.base import BaseTest

class TestQuery(BaseModel):
    kind: Literal['TestQuery'] = 'TestQuery'
    some_attr: str
    other_attr: Optional[List[Any]] = []

class TestQueryRunner(BaseTest):

    def setup_test_query_runner_class(self, query_class: Type[RunnableQueryNode]=TestQuery):
        if False:
            i = 10
            return i + 15
        'Setup required methods and attributes of the abstract base class.'

        class TestQueryRunner(QueryRunner):
            query_type = query_class

            def calculate(self) -> QueryResponse:
                if False:
                    for i in range(10):
                        print('nop')
                return QueryResponse(results=list())

            def _refresh_frequency(self) -> timedelta:
                if False:
                    print('Hello World!')
                return timedelta(minutes=4)

            def _is_stale(self, cached_result_package) -> bool:
                if False:
                    return 10
                return isoparse(cached_result_package.last_refresh) + timedelta(minutes=10) <= datetime.now(tz=ZoneInfo('UTC'))
        TestQueryRunner.__abstractmethods__ = frozenset()
        return TestQueryRunner

    def test_init_with_query_instance(self):
        if False:
            while True:
                i = 10
        TestQueryRunner = self.setup_test_query_runner_class()
        runner = TestQueryRunner(query=TestQuery(some_attr='bla'), team=self.team)
        self.assertEqual(runner.query, TestQuery(some_attr='bla'))

    def test_init_with_query_dict(self):
        if False:
            i = 10
            return i + 15
        TestQueryRunner = self.setup_test_query_runner_class()
        runner = TestQueryRunner(query={'some_attr': 'bla'}, team=self.team)
        self.assertEqual(runner.query, TestQuery(some_attr='bla'))

    def test_serializes_to_json(self):
        if False:
            for i in range(10):
                print('nop')
        TestQueryRunner = self.setup_test_query_runner_class()
        runner = TestQueryRunner(query={'some_attr': 'bla'}, team=self.team)
        json = runner.toJSON()
        self.assertEqual(json, '{"some_attr":"bla"}')

    def test_serializes_to_json_ignores_empty_dict(self):
        if False:
            for i in range(10):
                print('nop')
        TestQueryRunner = self.setup_test_query_runner_class()
        runner = TestQueryRunner(query={'some_attr': 'bla', 'other_attr': []}, team=self.team)
        json = runner.toJSON()
        self.assertEqual(json, '{"some_attr":"bla"}')

    def test_cache_key(self):
        if False:
            return 10
        TestQueryRunner = self.setup_test_query_runner_class()
        team = Team.objects.create(pk=42, organization=self.organization)
        runner = TestQueryRunner(query={'some_attr': 'bla'}, team=team)
        cache_key = runner._cache_key()
        self.assertEqual(cache_key, 'cache_33c9ea3098895d5a363a75feefafef06')

    def test_cache_key_runner_subclass(self):
        if False:
            while True:
                i = 10
        TestQueryRunner = self.setup_test_query_runner_class()

        class TestSubclassQueryRunner(TestQueryRunner):
            pass
        team = Team.objects.create(pk=42, organization=self.organization)
        runner = TestSubclassQueryRunner(query={'some_attr': 'bla'}, team=team)
        cache_key = runner._cache_key()
        self.assertEqual(cache_key, 'cache_d626615de8ad0df73c1d8610ca586597')

    def test_cache_key_different_timezone(self):
        if False:
            i = 10
            return i + 15
        TestQueryRunner = self.setup_test_query_runner_class()
        team = Team.objects.create(pk=42, organization=self.organization)
        team.timezone = 'Europe/Vienna'
        team.save()
        runner = TestQueryRunner(query={'some_attr': 'bla'}, team=team)
        cache_key = runner._cache_key()
        self.assertEqual(cache_key, 'cache_aeb23ec9e8de56dd8499f99f2e976d5a')

    def test_cache_response(self):
        if False:
            while True:
                i = 10
        TestQueryRunner = self.setup_test_query_runner_class()
        runner = TestQueryRunner(query={'some_attr': 'bla'}, team=self.team)
        with freeze_time(datetime(2023, 2, 4, 13, 37, 42)):
            response = runner.run(refresh_requested=False)
            self.assertEqual(response.is_cached, False)
            self.assertEqual(response.last_refresh, '2023-02-04T13:37:42Z')
            self.assertEqual(response.next_allowed_client_refresh, '2023-02-04T13:41:42Z')
            response = runner.run(refresh_requested=False)
            self.assertEqual(response.is_cached, True)
            response = runner.run(refresh_requested=True)
            self.assertEqual(response.is_cached, False)
        with freeze_time(datetime(2023, 2, 4, 13, 37 + 11, 42)):
            response = runner.run(refresh_requested=False)
            self.assertEqual(response.is_cached, False)