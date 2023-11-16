import mock
from mock import patch
from redash.models import DataSource, Query, QueryResult
from redash.utils.configuration import ConfigurationContainer
from tests import BaseTestCase

class DataSourceTest(BaseTestCase):

    def test_get_schema(self):
        if False:
            i = 10
            return i + 15
        return_value = [{'name': 'table', 'columns': []}]
        with mock.patch('redash.query_runner.pg.PostgreSQL.get_schema') as patched_get_schema:
            patched_get_schema.return_value = return_value
            schema = self.factory.data_source.get_schema()
            self.assertEqual(return_value, schema)

    def test_get_schema_uses_cache(self):
        if False:
            print('Hello World!')
        return_value = [{'name': 'table', 'columns': []}]
        with mock.patch('redash.query_runner.pg.PostgreSQL.get_schema') as patched_get_schema:
            patched_get_schema.return_value = return_value
            self.factory.data_source.get_schema()
            schema = self.factory.data_source.get_schema()
            self.assertEqual(return_value, schema)
            self.assertEqual(patched_get_schema.call_count, 1)

    def test_get_schema_skips_cache_with_refresh_true(self):
        if False:
            for i in range(10):
                print('nop')
        return_value = [{'name': 'table', 'columns': []}]
        with mock.patch('redash.query_runner.pg.PostgreSQL.get_schema') as patched_get_schema:
            patched_get_schema.return_value = return_value
            self.factory.data_source.get_schema()
            new_return_value = [{'name': 'new_table', 'columns': []}]
            patched_get_schema.return_value = new_return_value
            schema = self.factory.data_source.get_schema(refresh=True)
            self.assertEqual(new_return_value, schema)
            self.assertEqual(patched_get_schema.call_count, 2)

    def test_schema_sorter(self):
        if False:
            print('Hello World!')
        input_data = [{'name': 'zoo', 'columns': ['is_zebra', 'is_snake', 'is_cow']}, {'name': 'all_terain_vehicle', 'columns': ['has_wheels', 'has_engine', 'has_all_wheel_drive']}]
        expected_output = [{'name': 'all_terain_vehicle', 'columns': ['has_all_wheel_drive', 'has_engine', 'has_wheels']}, {'name': 'zoo', 'columns': ['is_cow', 'is_snake', 'is_zebra']}]
        real_output = self.factory.data_source._sort_schema(input_data)
        self.assertEqual(real_output, expected_output)

    def test_model_uses_schema_sorter(self):
        if False:
            print('Hello World!')
        orig_schema = [{'name': 'zoo', 'columns': ['is_zebra', 'is_snake', 'is_cow']}, {'name': 'all_terain_vehicle', 'columns': ['has_wheels', 'has_engine', 'has_all_wheel_drive']}]
        sorted_schema = [{'name': 'all_terain_vehicle', 'columns': ['has_all_wheel_drive', 'has_engine', 'has_wheels']}, {'name': 'zoo', 'columns': ['is_cow', 'is_snake', 'is_zebra']}]
        with mock.patch('redash.query_runner.pg.PostgreSQL.get_schema') as patched_get_schema:
            patched_get_schema.return_value = orig_schema
            out_schema = self.factory.data_source.get_schema()
            self.assertEqual(out_schema, sorted_schema)

    @patch('redash.redis_connection.set')
    def test_expires_schema(self, mock_redis):
        if False:
            i = 10
            return i + 15
        expected_ttl = 606600
        with mock.patch('redash.query_runner.pg.PostgreSQL.get_schema') as patched_get_schema:
            patched_get_schema.return_value = None
            self.factory.data_source.get_schema(refresh=True)
        mock_redis.assert_called_with('data_source:schema:1', 'null', ex=expected_ttl)

class TestDataSourceCreate(BaseTestCase):

    def test_adds_data_source_to_default_group(self):
        if False:
            i = 10
            return i + 15
        data_source = DataSource.create_with_group(org=self.factory.org, name='test', options=ConfigurationContainer.from_json('{"dbname": "test"}'), type='pg')
        self.assertIn(self.factory.org.default_group.id, data_source.groups)

class TestDataSourceIsPaused(BaseTestCase):

    def test_returns_false_by_default(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(self.factory.data_source.paused)

    def test_persists_selection(self):
        if False:
            return 10
        self.factory.data_source.pause()
        self.assertTrue(self.factory.data_source.paused)
        self.factory.data_source.resume()
        self.assertFalse(self.factory.data_source.paused)

    def test_allows_setting_reason(self):
        if False:
            return 10
        reason = 'Some good reason.'
        self.factory.data_source.pause(reason)
        self.assertTrue(self.factory.data_source.paused)
        self.assertEqual(self.factory.data_source.pause_reason, reason)

    def test_resume_clears_reason(self):
        if False:
            for i in range(10):
                print('nop')
        self.factory.data_source.pause('Reason')
        self.factory.data_source.resume()
        self.assertEqual(self.factory.data_source.pause_reason, None)

    def test_reason_is_none_by_default(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.factory.data_source.pause_reason, None)

class TestDataSourceDelete(BaseTestCase):

    def test_deletes_the_data_source(self):
        if False:
            for i in range(10):
                print('nop')
        data_source = self.factory.create_data_source()
        data_source.delete()
        self.assertIsNone(DataSource.query.get(data_source.id))

    def test_sets_queries_data_source_to_null(self):
        if False:
            while True:
                i = 10
        data_source = self.factory.create_data_source()
        query = self.factory.create_query(data_source=data_source)
        data_source.delete()
        self.assertIsNone(DataSource.query.get(data_source.id))
        self.assertIsNone(Query.query.get(query.id).data_source_id)

    def test_deletes_child_models(self):
        if False:
            i = 10
            return i + 15
        data_source = self.factory.create_data_source()
        self.factory.create_query_result(data_source=data_source)
        self.factory.create_query(data_source=data_source, latest_query_data=self.factory.create_query_result(data_source=data_source))
        data_source.delete()
        self.assertIsNone(DataSource.query.get(data_source.id))
        self.assertEqual(0, QueryResult.query.filter(QueryResult.data_source == data_source).count())

    @patch('redash.redis_connection.delete')
    def test_deletes_schema(self, mock_redis):
        if False:
            while True:
                i = 10
        data_source = self.factory.create_data_source()
        data_source.delete()
        mock_redis.assert_called_with(data_source._schema_key)