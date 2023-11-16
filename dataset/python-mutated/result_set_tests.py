from datetime import datetime
import tests.integration_tests.test_app
from superset.dataframe import df_to_records
from superset.db_engine_specs import BaseEngineSpec
from superset.result_set import dedup, SupersetResultSet
from .base_tests import SupersetTestCase

class TestSupersetResultSet(SupersetTestCase):

    def test_dedup(self):
        if False:
            return 10
        self.assertEqual(dedup(['foo', 'bar']), ['foo', 'bar'])
        self.assertEqual(dedup(['foo', 'bar', 'foo', 'bar', 'Foo']), ['foo', 'bar', 'foo__1', 'bar__1', 'Foo'])
        self.assertEqual(dedup(['foo', 'bar', 'bar', 'bar', 'Bar']), ['foo', 'bar', 'bar__1', 'bar__2', 'Bar'])
        self.assertEqual(dedup(['foo', 'bar', 'bar', 'bar', 'Bar'], case_sensitive=False), ['foo', 'bar', 'bar__1', 'bar__2', 'Bar__3'])

    def test_get_columns_basic(self):
        if False:
            while True:
                i = 10
        data = [('a1', 'b1', 'c1'), ('a2', 'b2', 'c2')]
        cursor_descr = (('a', 'string'), ('b', 'string'), ('c', 'string'))
        results = SupersetResultSet(data, cursor_descr, BaseEngineSpec)
        self.assertEqual(results.columns, [{'is_dttm': False, 'type': 'STRING', 'column_name': 'a', 'name': 'a'}, {'is_dttm': False, 'type': 'STRING', 'column_name': 'b', 'name': 'b'}, {'is_dttm': False, 'type': 'STRING', 'column_name': 'c', 'name': 'c'}])

    def test_get_columns_with_int(self):
        if False:
            print('Hello World!')
        data = [('a1', 1), ('a2', 2)]
        cursor_descr = (('a', 'string'), ('b', 'int'))
        results = SupersetResultSet(data, cursor_descr, BaseEngineSpec)
        self.assertEqual(results.columns, [{'is_dttm': False, 'type': 'STRING', 'column_name': 'a', 'name': 'a'}, {'is_dttm': False, 'type': 'INT', 'column_name': 'b', 'name': 'b'}])

    def test_get_columns_type_inference(self):
        if False:
            while True:
                i = 10
        data = [(1.2, 1, 'foo', datetime(2018, 10, 19, 23, 39, 16, 660000), True), (3.14, 2, 'bar', datetime(2019, 10, 19, 23, 39, 16, 660000), False)]
        cursor_descr = (('a', None), ('b', None), ('c', None), ('d', None), ('e', None))
        results = SupersetResultSet(data, cursor_descr, BaseEngineSpec)
        self.assertEqual(results.columns, [{'is_dttm': False, 'type': 'FLOAT', 'column_name': 'a', 'name': 'a'}, {'is_dttm': False, 'type': 'INT', 'column_name': 'b', 'name': 'b'}, {'is_dttm': False, 'type': 'STRING', 'column_name': 'c', 'name': 'c'}, {'is_dttm': True, 'type': 'DATETIME', 'column_name': 'd', 'name': 'd'}, {'is_dttm': False, 'type': 'BOOL', 'column_name': 'e', 'name': 'e'}])

    def test_is_date(self):
        if False:
            i = 10
            return i + 15
        data = [('a', 1), ('a', 2)]
        cursor_descr = (('a', 'string'), ('a', 'string'))
        results = SupersetResultSet(data, cursor_descr, BaseEngineSpec)
        self.assertEqual(results.is_temporal('DATE'), True)
        self.assertEqual(results.is_temporal('DATETIME'), True)
        self.assertEqual(results.is_temporal('TIME'), True)
        self.assertEqual(results.is_temporal('TIMESTAMP'), True)
        self.assertEqual(results.is_temporal('STRING'), False)
        self.assertEqual(results.is_temporal(''), False)
        self.assertEqual(results.is_temporal(None), False)

    def test_dedup_with_data(self):
        if False:
            while True:
                i = 10
        data = [('a', 1), ('a', 2)]
        cursor_descr = (('a', 'string'), ('a', 'string'))
        results = SupersetResultSet(data, cursor_descr, BaseEngineSpec)
        column_names = [col['column_name'] for col in results.columns]
        self.assertListEqual(column_names, ['a', 'a__1'])

    def test_int64_with_missing_data(self):
        if False:
            return 10
        data = [(None,), (1239162456494753670,), (None,), (None,), (None,), (None,)]
        cursor_descr = [('user_id', 'bigint', None, None, None, None, True)]
        results = SupersetResultSet(data, cursor_descr, BaseEngineSpec)
        self.assertEqual(results.columns[0]['type'], 'BIGINT')

    def test_data_as_list_of_lists(self):
        if False:
            while True:
                i = 10
        data = [[1, 'a'], [2, 'b']]
        cursor_descr = [('user_id', 'INT', None, None, None, None, True), ('username', 'STRING', None, None, None, None, True)]
        results = SupersetResultSet(data, cursor_descr, BaseEngineSpec)
        df = results.to_pandas_df()
        self.assertEqual(df_to_records(df), [{'user_id': 1, 'username': 'a'}, {'user_id': 2, 'username': 'b'}])

    def test_nullable_bool(self):
        if False:
            while True:
                i = 10
        data = [(None,), (True,), (None,), (None,), (None,), (None,)]
        cursor_descr = [('is_test', 'bool', None, None, None, None, True)]
        results = SupersetResultSet(data, cursor_descr, BaseEngineSpec)
        self.assertEqual(results.columns[0]['type'], 'BOOL')
        df = results.to_pandas_df()
        self.assertEqual(df_to_records(df), [{'is_test': None}, {'is_test': True}, {'is_test': None}, {'is_test': None}, {'is_test': None}, {'is_test': None}])

    def test_nested_types(self):
        if False:
            i = 10
            return i + 15
        data = [(4, [{'table_name': 'unicode_test', 'database_id': 1}], [1, 2, 3], {'chart_name': 'scatter'}), (3, [{'table_name': 'birth_names', 'database_id': 1}], [4, 5, 6], {'chart_name': 'plot'})]
        cursor_descr = [('id',), ('dict_arr',), ('num_arr',), ('map_col',)]
        results = SupersetResultSet(data, cursor_descr, BaseEngineSpec)
        self.assertEqual(results.columns[0]['type'], 'INT')
        self.assertEqual(results.columns[1]['type'], 'STRING')
        self.assertEqual(results.columns[2]['type'], 'STRING')
        self.assertEqual(results.columns[3]['type'], 'STRING')
        df = results.to_pandas_df()
        self.assertEqual(df_to_records(df), [{'id': 4, 'dict_arr': '[{"table_name": "unicode_test", "database_id": 1}]', 'num_arr': '[1, 2, 3]', 'map_col': "{'chart_name': 'scatter'}"}, {'id': 3, 'dict_arr': '[{"table_name": "birth_names", "database_id": 1}]', 'num_arr': '[4, 5, 6]', 'map_col': "{'chart_name': 'plot'}"}])

    def test_single_column_multidim_nested_types(self):
        if False:
            for i in range(10):
                print('nop')
        data = [(['test', [['foo', 123456, [[['test'], 3432546, 7657658766], [['fake'], 656756765, 324324324324]]]], ['test2', 43, 765765765], None, None],)]
        cursor_descr = [('metadata',)]
        results = SupersetResultSet(data, cursor_descr, BaseEngineSpec)
        self.assertEqual(results.columns[0]['type'], 'STRING')
        df = results.to_pandas_df()
        self.assertEqual(df_to_records(df), [{'metadata': '["test", [["foo", 123456, [[["test"], 3432546, 7657658766], [["fake"], 656756765, 324324324324]]]], ["test2", 43, 765765765], null, null]'}])

    def test_nested_list_types(self):
        if False:
            i = 10
            return i + 15
        data = [([{'TestKey': [123456, 'foo']}],)]
        cursor_descr = [('metadata',)]
        results = SupersetResultSet(data, cursor_descr, BaseEngineSpec)
        self.assertEqual(results.columns[0]['type'], 'STRING')
        df = results.to_pandas_df()
        self.assertEqual(df_to_records(df), [{'metadata': '[{"TestKey": [123456, "foo"]}]'}])

    def test_empty_datetime(self):
        if False:
            while True:
                i = 10
        data = [(None,)]
        cursor_descr = [('ds', 'timestamp', None, None, None, None, True)]
        results = SupersetResultSet(data, cursor_descr, BaseEngineSpec)
        self.assertEqual(results.columns[0]['type'], 'TIMESTAMP')

    def test_no_type_coercion(self):
        if False:
            return 10
        data = [('a', 1), ('b', 2)]
        cursor_descr = [('one', 'varchar', None, None, None, None, True), ('two', 'int', None, None, None, None, True)]
        results = SupersetResultSet(data, cursor_descr, BaseEngineSpec)
        self.assertEqual(results.columns[0]['type'], 'VARCHAR')
        self.assertEqual(results.columns[1]['type'], 'INT')

    def test_empty_data(self):
        if False:
            return 10
        data = []
        cursor_descr = [('emptyone', 'varchar', None, None, None, None, True), ('emptytwo', 'int', None, None, None, None, True)]
        results = SupersetResultSet(data, cursor_descr, BaseEngineSpec)
        self.assertEqual(results.columns, [])