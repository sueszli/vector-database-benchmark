import pytest
from sentry.exceptions import InvalidSearchQuery
from sentry.snuba import discover
from sentry.testutils.cases import SnubaTestCase, TestCase
from sentry.testutils.helpers.datetime import before_now, iso_format

class GetFacetsTest(SnubaTestCase, TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.project = self.create_project()
        self.min_ago = before_now(minutes=1)
        self.day_ago = before_now(days=1)

    def test_invalid_query(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(InvalidSearchQuery):
            discover.get_facets('\n', {'project_id': [self.project.id], 'end': self.min_ago, 'start': self.day_ago}, 'testing.get-facets-test')

    def test_no_results(self):
        if False:
            i = 10
            return i + 15
        results = discover.get_facets('', {'project_id': [self.project.id], 'end': self.min_ago, 'start': self.day_ago}, 'testing.get-facets-test')
        assert results == []

    def test_single_project(self):
        if False:
            i = 10
            return i + 15
        self.store_event(data={'message': 'very bad', 'type': 'default', 'timestamp': iso_format(before_now(minutes=2)), 'tags': {'color': 'red', 'paying': '1'}}, project_id=self.project.id)
        self.store_event(data={'message': 'very bad', 'type': 'default', 'timestamp': iso_format(before_now(minutes=2)), 'tags': {'color': 'blue', 'paying': '0'}}, project_id=self.project.id)
        params = {'project_id': [self.project.id], 'start': self.day_ago, 'end': self.min_ago}
        result = discover.get_facets('', params, 'testing.get-facets-test')
        assert len(result) == 5
        assert {r.key for r in result} == {'color', 'paying', 'level'}
        assert {r.value for r in result} == {'red', 'blue', '1', '0', 'error'}
        assert {r.count for r in result} == {1, 2}

    def test_project_filter(self):
        if False:
            while True:
                i = 10
        self.store_event(data={'message': 'very bad', 'type': 'default', 'timestamp': iso_format(before_now(minutes=2)), 'tags': {'color': 'red'}}, project_id=self.project.id)
        other_project = self.create_project()
        self.store_event(data={'message': 'very bad', 'type': 'default', 'timestamp': iso_format(before_now(minutes=2)), 'tags': {'toy': 'train'}}, project_id=other_project.id)
        params = {'project_id': [self.project.id], 'start': self.day_ago, 'end': self.min_ago}
        result = discover.get_facets('', params, 'testing.get-facets-test')
        keys = {r.key for r in result}
        assert keys == {'color', 'level'}
        params = {'project_id': [self.project.id, other_project.id], 'start': self.day_ago, 'end': self.min_ago}
        result = discover.get_facets('', params, 'testing.get-facets-test')
        keys = {r.key for r in result}
        assert keys == {'level', 'toy', 'color', 'project'}
        projects = [f for f in result if f.key == 'project']
        assert [p.count for p in projects] == [1, 1]

    def test_environment_promoted_tag(self):
        if False:
            for i in range(10):
                print('nop')
        for env in ('prod', 'staging', None):
            self.store_event(data={'message': 'very bad', 'type': 'default', 'environment': env, 'timestamp': iso_format(before_now(minutes=2))}, project_id=self.project.id)
        params = {'project_id': [self.project.id], 'start': self.day_ago, 'end': self.min_ago}
        result = discover.get_facets('', params, 'testing.get-facets-test')
        keys = {r.key for r in result}
        assert keys == {'environment', 'level'}
        assert {None, 'prod', 'staging'} == {f.value for f in result if f.key == 'environment'}
        assert {1} == {f.count for f in result if f.key == 'environment'}

    def test_query_string(self):
        if False:
            for i in range(10):
                print('nop')
        self.store_event(data={'message': 'very bad', 'type': 'default', 'timestamp': iso_format(before_now(minutes=2)), 'tags': {'color': 'red'}}, project_id=self.project.id)
        self.store_event(data={'message': 'oh my', 'type': 'default', 'timestamp': iso_format(before_now(minutes=2)), 'tags': {'toy': 'train'}}, project_id=self.project.id)
        params = {'project_id': [self.project.id], 'start': self.day_ago, 'end': self.min_ago}
        result = discover.get_facets('bad', params, 'testing.get-facets-test')
        keys = {r.key for r in result}
        assert 'color' in keys
        assert 'toy' not in keys
        result = discover.get_facets('color:red', params, 'testing.get-facets-test')
        keys = {r.key for r in result}
        assert 'color' in keys
        assert 'toy' not in keys

    def test_query_string_with_aggregate_condition(self):
        if False:
            for i in range(10):
                print('nop')
        self.store_event(data={'message': 'very bad', 'type': 'default', 'timestamp': iso_format(before_now(minutes=2)), 'tags': {'color': 'red'}}, project_id=self.project.id)
        self.store_event(data={'message': 'oh my', 'type': 'default', 'timestamp': iso_format(before_now(minutes=2)), 'tags': {'toy': 'train'}}, project_id=self.project.id)
        params = {'project_id': [self.project.id], 'start': self.day_ago, 'end': self.min_ago}
        result = discover.get_facets('bad', params, 'testing.get-facets-test')
        keys = {r.key for r in result}
        assert 'color' in keys
        assert 'toy' not in keys
        result = discover.get_facets('color:red p95():>1', params, 'testing.get-facets-test')
        keys = {r.key for r in result}
        assert 'color' in keys
        assert 'toy' not in keys

    def test_date_params(self):
        if False:
            i = 10
            return i + 15
        self.store_event(data={'message': 'very bad', 'type': 'default', 'timestamp': iso_format(before_now(minutes=2)), 'tags': {'color': 'red'}}, project_id=self.project.id)
        self.store_event(data={'message': 'oh my', 'type': 'default', 'timestamp': iso_format(before_now(days=2)), 'tags': {'toy': 'train'}}, project_id=self.project.id)
        params = {'project_id': [self.project.id], 'start': self.day_ago, 'end': self.min_ago}
        result = discover.get_facets('', params, 'testing.get-facets-test')
        keys = {r.key for r in result}
        assert 'color' in keys
        assert 'toy' not in keys

    def test_count_sorting(self):
        if False:
            i = 10
            return i + 15
        for _ in range(5):
            self.store_event(data={'message': 'very bad', 'type': 'default', 'timestamp': iso_format(before_now(minutes=2)), 'tags': {'color': 'zzz'}}, project_id=self.project.id)
        self.store_event(data={'message': 'oh my', 'type': 'default', 'timestamp': iso_format(before_now(minutes=2)), 'tags': {'color': 'aaa'}}, project_id=self.project.id)
        params = {'project_id': [self.project.id], 'start': self.day_ago, 'end': self.min_ago}
        result = discover.get_facets('', params, 'testing.get-facets-test')
        first = result[0]
        assert first.key == 'color'
        assert first.value == 'zzz'
        second = result[1]
        assert second.key == 'color'
        assert second.value == 'aaa'