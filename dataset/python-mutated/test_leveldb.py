from __future__ import annotations
from unittest import mock
import pytest
from airflow.exceptions import AirflowOptionalProviderFeatureException
try:
    from airflow.providers.google.leveldb.hooks.leveldb import LevelDBHook, LevelDBHookException
except AirflowOptionalProviderFeatureException:
    pytest.skip('LevelDB not available', allow_module_level=True)

class TestLevelDBHook:

    @mock.patch.dict('os.environ', AIRFLOW_CONN_LEVELDB_DEFAULT='test')
    def test_get_conn_db_is_not_none(self):
        if False:
            i = 10
            return i + 15
        'Test get_conn method of hook'
        hook = LevelDBHook(leveldb_conn_id='leveldb_default')
        hook.get_conn(name='/tmp/testdb/', create_if_missing=True)
        assert hook.db is not None, 'Check existence of DB object in connection creation'
        hook.close_conn()

    @mock.patch.dict('os.environ', AIRFLOW_CONN_LEVELDB_DEFAULT='test')
    def test_run(self):
        if False:
            print('Hello World!')
        'Test run method of hook'
        hook = LevelDBHook(leveldb_conn_id='leveldb_default')
        hook.get_conn(name='/tmp/testdb/', create_if_missing=True)
        assert hook.run('get', b'test_key0') is None, 'Initially, this key in LevelDB is empty'
        hook.run('put', b'test_key0', b'test_value0')
        assert hook.run('get', b'test_key0') == b'test_value0', 'Connection to LevelDB with PUT and GET works.'
        hook.run('delete', b'test_key0')
        assert hook.run('get', b'test_key0') is None, 'Connection to LevelDB with DELETE works.'
        hook.close_conn()

    @mock.patch.dict('os.environ', AIRFLOW_CONN_LEVELDB_DEFAULT='test')
    def test_get(self):
        if False:
            while True:
                i = 10
        'Test get method of hook'
        hook = LevelDBHook(leveldb_conn_id='leveldb_default')
        db = hook.get_conn(name='/tmp/testdb/', create_if_missing=True)
        db.put(b'test_key', b'test_value')
        assert hook.get(b'test_key') == b'test_value'
        hook.close_conn()

    @mock.patch.dict('os.environ', AIRFLOW_CONN_LEVELDB_DEFAULT='test')
    def test_put(self):
        if False:
            while True:
                i = 10
        'Test put method of hook'
        hook = LevelDBHook(leveldb_conn_id='leveldb_default')
        db = hook.get_conn(name='/tmp/testdb/', create_if_missing=True)
        hook.put(b'test_key2', b'test_value2')
        assert db.get(b'test_key2') == b'test_value2'
        hook.close_conn()

    @mock.patch.dict('os.environ', AIRFLOW_CONN_LEVELDB_DEFAULT='test')
    def test_delete(self):
        if False:
            print('Hello World!')
        'Test delete method of hook'
        hook = LevelDBHook(leveldb_conn_id='leveldb_default')
        db = hook.get_conn(name='/tmp/testdb/', create_if_missing=True)
        db.put(b'test_key3', b'test_value3')
        hook.delete(b'test_key3')
        assert db.get(b'test_key3') is None
        hook.close_conn()

    @mock.patch.dict('os.environ', AIRFLOW_CONN_LEVELDB_DEFAULT='test')
    def test_write_batch(self):
        if False:
            print('Hello World!')
        'Test write batch method of hook'
        hook = LevelDBHook(leveldb_conn_id='leveldb_default')
        db = hook.get_conn(name='/tmp/testdb/', create_if_missing=True)
        keys = [b'key', b'another-key']
        values = [b'value', b'another-value']
        hook.write_batch(keys, values)
        assert db.get(b'key') == b'value'
        assert db.get(b'another-key') == b'another-value'
        hook.close_conn()

    @mock.patch.dict('os.environ', AIRFLOW_CONN_LEVELDB_DEFAULT='test')
    def test_exception(self):
        if False:
            for i in range(10):
                print('nop')
        'Test raising exception of hook in run method if we have unknown command in input'
        hook = LevelDBHook(leveldb_conn_id='leveldb_default')
        hook.get_conn(name='/tmp/testdb/', create_if_missing=True)
        with pytest.raises(LevelDBHookException):
            hook.run(command='other_command', key=b'key', value=b'value')

    @mock.patch.dict('os.environ', AIRFLOW_CONN_LEVELDB_DEFAULT='test')
    def test_comparator(self):
        if False:
            print('Hello World!')
        'Test comparator'

        def comparator(a, b):
            if False:
                for i in range(10):
                    print('nop')
            a = a.lower()
            b = b.lower()
            if a < b:
                return -1
            if a > b:
                return 1
            return 0
        hook = LevelDBHook(leveldb_conn_id='leveldb_default')
        hook.get_conn(name='/tmp/testdb2/', create_if_missing=True, comparator=comparator, comparator_name=b'CaseInsensitiveComparator')
        assert hook.db is not None, 'Check existence of DB object(with comparator) in connection creation'
        hook.close_conn()