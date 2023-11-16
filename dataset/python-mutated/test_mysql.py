from __future__ import annotations
import json
import os
import uuid
from contextlib import closing
from unittest import mock
import pytest
from airflow.models import Connection
from airflow.models.dag import DAG
try:
    import MySQLdb.cursors
    from airflow.providers.mysql.hooks.mysql import MySqlHook
except ImportError:
    pytest.skip('MySQL not available', allow_module_level=True)
from airflow.utils import timezone
from tests.test_utils.asserts import assert_equal_ignore_multiple_spaces
SSL_DICT = {'cert': '/tmp/client-cert.pem', 'ca': '/tmp/server-ca.pem', 'key': '/tmp/client-key.pem'}

class TestMySqlHookConn:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.connection = Connection(conn_type='mysql', login='login', password='password', host='host', schema='schema')
        self.db_hook = MySqlHook()
        self.db_hook.get_connection = mock.Mock()
        self.db_hook.get_connection.return_value = self.connection

    @mock.patch('MySQLdb.connect')
    def test_get_conn(self, mock_connect):
        if False:
            i = 10
            return i + 15
        self.db_hook.get_conn()
        assert mock_connect.call_count == 1
        (args, kwargs) = mock_connect.call_args
        assert args == ()
        assert kwargs['user'] == 'login'
        assert kwargs['passwd'] == 'password'
        assert kwargs['host'] == 'host'
        assert kwargs['db'] == 'schema'

    @mock.patch('MySQLdb.connect')
    def test_get_uri(self, mock_connect):
        if False:
            print('Hello World!')
        self.connection.extra = json.dumps({'charset': 'utf-8'})
        self.db_hook.get_conn()
        assert mock_connect.call_count == 1
        (args, kwargs) = mock_connect.call_args
        assert self.db_hook.get_uri() == 'mysql://login:password@host/schema?charset=utf-8'

    @mock.patch('MySQLdb.connect')
    def test_get_conn_from_connection(self, mock_connect):
        if False:
            print('Hello World!')
        conn = Connection(login='login-conn', password='password-conn', host='host', schema='schema')
        hook = MySqlHook(connection=conn)
        hook.get_conn()
        mock_connect.assert_called_once_with(user='login-conn', passwd='password-conn', host='host', db='schema', port=3306)

    @mock.patch('MySQLdb.connect')
    def test_get_conn_from_connection_with_schema(self, mock_connect):
        if False:
            for i in range(10):
                print('nop')
        conn = Connection(login='login-conn', password='password-conn', host='host', schema='schema')
        hook = MySqlHook(connection=conn, schema='schema-override')
        hook.get_conn()
        mock_connect.assert_called_once_with(user='login-conn', passwd='password-conn', host='host', db='schema-override', port=3306)

    @mock.patch('MySQLdb.connect')
    def test_get_conn_port(self, mock_connect):
        if False:
            print('Hello World!')
        self.connection.port = 3307
        self.db_hook.get_conn()
        assert mock_connect.call_count == 1
        (args, kwargs) = mock_connect.call_args
        assert args == ()
        assert kwargs['port'] == 3307

    @mock.patch('MySQLdb.connect')
    def test_get_conn_charset(self, mock_connect):
        if False:
            i = 10
            return i + 15
        self.connection.extra = json.dumps({'charset': 'utf-8'})
        self.db_hook.get_conn()
        assert mock_connect.call_count == 1
        (args, kwargs) = mock_connect.call_args
        assert args == ()
        assert kwargs['charset'] == 'utf-8'
        assert kwargs['use_unicode'] is True

    @mock.patch('MySQLdb.connect')
    def test_get_conn_cursor(self, mock_connect):
        if False:
            for i in range(10):
                print('nop')
        self.connection.extra = json.dumps({'cursor': 'sscursor'})
        self.db_hook.get_conn()
        assert mock_connect.call_count == 1
        (args, kwargs) = mock_connect.call_args
        assert args == ()
        assert kwargs['cursorclass'] == MySQLdb.cursors.SSCursor

    @mock.patch('MySQLdb.connect')
    def test_get_conn_local_infile(self, mock_connect):
        if False:
            print('Hello World!')
        self.db_hook.local_infile = True
        self.db_hook.get_conn()
        assert mock_connect.call_count == 1
        (args, kwargs) = mock_connect.call_args
        assert args == ()
        assert kwargs['local_infile'] == 1

    @mock.patch('MySQLdb.connect')
    def test_get_con_unix_socket(self, mock_connect):
        if False:
            print('Hello World!')
        self.connection.extra = json.dumps({'unix_socket': '/tmp/socket'})
        self.db_hook.get_conn()
        assert mock_connect.call_count == 1
        (args, kwargs) = mock_connect.call_args
        assert args == ()
        assert kwargs['unix_socket'] == '/tmp/socket'

    @mock.patch('MySQLdb.connect')
    def test_get_conn_ssl_as_dictionary(self, mock_connect):
        if False:
            for i in range(10):
                print('nop')
        self.connection.extra = json.dumps({'ssl': SSL_DICT})
        self.db_hook.get_conn()
        assert mock_connect.call_count == 1
        (args, kwargs) = mock_connect.call_args
        assert args == ()
        assert kwargs['ssl'] == SSL_DICT

    @mock.patch('MySQLdb.connect')
    def test_get_conn_ssl_as_string(self, mock_connect):
        if False:
            for i in range(10):
                print('nop')
        self.connection.extra = json.dumps({'ssl': json.dumps(SSL_DICT)})
        self.db_hook.get_conn()
        assert mock_connect.call_count == 1
        (args, kwargs) = mock_connect.call_args
        assert args == ()
        assert kwargs['ssl'] == SSL_DICT

    @mock.patch('MySQLdb.connect')
    def test_get_ssl_mode(self, mock_connect):
        if False:
            return 10
        self.connection.extra = json.dumps({'ssl_mode': 'DISABLED'})
        self.db_hook.get_conn()
        assert mock_connect.call_count == 1
        (args, kwargs) = mock_connect.call_args
        assert args == ()
        assert kwargs['ssl_mode'] == 'DISABLED'

    @mock.patch('MySQLdb.connect')
    @mock.patch('airflow.providers.amazon.aws.hooks.base_aws.AwsBaseHook.get_client_type')
    def test_get_conn_rds_iam(self, mock_client, mock_connect):
        if False:
            print('Hello World!')
        self.connection.extra = '{"iam":true}'
        mock_client.return_value.generate_db_auth_token.return_value = 'aws_token'
        self.db_hook.get_conn()
        mock_connect.assert_called_once_with(user='login', passwd='aws_token', host='host', db='schema', port=3306, read_default_group='enable-cleartext-plugin')

    @mock.patch('MySQLdb.connect')
    def test_get_conn_init_command(self, mock_connect):
        if False:
            i = 10
            return i + 15
        self.db_hook.init_command = "SET time_zone = '+00:00';"
        self.db_hook.get_conn()
        assert mock_connect.call_count == 1
        (args, kwargs) = mock_connect.call_args
        assert args == ()
        assert kwargs['init_command'] == "SET time_zone = '+00:00';"

class MockMySQLConnectorConnection:
    DEFAULT_AUTOCOMMIT = 'default'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._autocommit = self.DEFAULT_AUTOCOMMIT

    @property
    def autocommit(self):
        if False:
            return 10
        return self._autocommit

    @autocommit.setter
    def autocommit(self, autocommit):
        if False:
            return 10
        self._autocommit = autocommit

class TestMySqlHook:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.cur = mock.MagicMock(rowcount=0)
        self.conn = mock.MagicMock()
        self.conn.cursor.return_value = self.cur
        conn = self.conn

        class SubMySqlHook(MySqlHook):
            conn_name_attr = 'test_conn_id'

            def get_conn(self):
                if False:
                    i = 10
                    return i + 15
                return conn
        self.db_hook = SubMySqlHook()

    @pytest.mark.parametrize('autocommit', [True, False])
    def test_set_autocommit_mysql_connector(self, autocommit):
        if False:
            print('Hello World!')
        conn = MockMySQLConnectorConnection()
        self.db_hook.set_autocommit(conn, autocommit)
        assert conn.autocommit is autocommit

    def test_get_autocommit_mysql_connector(self):
        if False:
            print('Hello World!')
        conn = MockMySQLConnectorConnection()
        assert self.db_hook.get_autocommit(conn) == MockMySQLConnectorConnection.DEFAULT_AUTOCOMMIT

    def test_set_autocommit_mysqldb(self):
        if False:
            for i in range(10):
                print('nop')
        autocommit = False
        self.db_hook.set_autocommit(self.conn, autocommit)
        self.conn.autocommit.assert_called_once_with(autocommit)

    def test_get_autocommit_mysqldb(self):
        if False:
            while True:
                i = 10
        self.db_hook.get_autocommit(self.conn)
        self.conn.get_autocommit.assert_called_once()

    def test_run_without_autocommit(self):
        if False:
            print('Hello World!')
        sql = 'SQL'
        self.conn.get_autocommit.return_value = False
        self.db_hook.run(sql, autocommit=False)
        self.conn.autocommit.assert_called_once_with(False)
        self.cur.execute.assert_called_once_with(sql)
        assert self.conn.commit.call_count == 1

    def test_run_with_autocommit(self):
        if False:
            i = 10
            return i + 15
        sql = 'SQL'
        self.db_hook.run(sql, autocommit=True)
        self.conn.autocommit.assert_called_once_with(True)
        self.cur.execute.assert_called_once_with(sql)
        self.conn.commit.assert_not_called()

    def test_run_with_parameters(self):
        if False:
            return 10
        sql = 'SQL'
        parameters = ('param1', 'param2')
        self.db_hook.run(sql, autocommit=True, parameters=parameters)
        self.conn.autocommit.assert_called_once_with(True)
        self.cur.execute.assert_called_once_with(sql, parameters)
        self.conn.commit.assert_not_called()

    def test_run_multi_queries(self):
        if False:
            while True:
                i = 10
        sql = ['SQL1', 'SQL2']
        self.db_hook.run(sql, autocommit=True)
        self.conn.autocommit.assert_called_once_with(True)
        for (i, item) in enumerate(self.cur.execute.call_args_list):
            (args, kwargs) = item
            assert len(args) == 1
            assert args[0] == sql[i]
            assert kwargs == {}
        calls = [mock.call(sql[0]), mock.call(sql[1])]
        self.cur.execute.assert_has_calls(calls, any_order=True)
        self.conn.commit.assert_not_called()

    def test_bulk_load(self):
        if False:
            print('Hello World!')
        self.db_hook.bulk_load('table', '/tmp/file')
        self.cur.execute.assert_called_once_with("\n            LOAD DATA LOCAL INFILE '/tmp/file'\n            INTO TABLE table\n            ")

    def test_bulk_dump(self):
        if False:
            for i in range(10):
                print('nop')
        self.db_hook.bulk_dump('table', '/tmp/file')
        self.cur.execute.assert_called_once_with("\n            SELECT * INTO OUTFILE '/tmp/file'\n            FROM table\n            ")

    def test_serialize_cell(self):
        if False:
            print('Hello World!')
        assert 'foo' == self.db_hook._serialize_cell('foo', None)

    def test_bulk_load_custom(self):
        if False:
            while True:
                i = 10
        self.db_hook.bulk_load_custom('table', '/tmp/file', 'IGNORE', 'FIELDS TERMINATED BY \';\'\n            OPTIONALLY ENCLOSED BY \'"\'\n            IGNORE 1 LINES')
        self.cur.execute.assert_called_once_with('\n            LOAD DATA LOCAL INFILE \'/tmp/file\'\n            IGNORE\n            INTO TABLE table\n            FIELDS TERMINATED BY \';\'\n            OPTIONALLY ENCLOSED BY \'"\'\n            IGNORE 1 LINES\n            ')
DEFAULT_DATE = timezone.datetime(2015, 1, 1)
DEFAULT_DATE_ISO = DEFAULT_DATE.isoformat()
DEFAULT_DATE_DS = DEFAULT_DATE_ISO[:10]
TEST_DAG_ID = 'unit_test_dag'

class MySqlContext:

    def __init__(self, client):
        if False:
            return 10
        self.client = client
        self.connection = MySqlHook.get_connection(MySqlHook.default_conn_name)
        self.init_client = self.connection.extra_dejson.get('client', 'mysqlclient')

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.connection.set_extra(f'{{"client": "{self.client}"}}')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        self.connection.set_extra(f'{{"client": "{self.init_client}"}}')

@pytest.mark.backend('mysql')
class TestMySql:

    def setup_method(self):
        if False:
            print('Hello World!')
        args = {'owner': 'airflow', 'start_date': DEFAULT_DATE}
        dag = DAG(TEST_DAG_ID, default_args=args)
        self.dag = dag

    def teardown_method(self):
        if False:
            for i in range(10):
                print('nop')
        drop_tables = {'test_mysql_to_mysql', 'test_airflow'}
        with closing(MySqlHook().get_conn()) as conn:
            with closing(conn.cursor()) as cursor:
                for table in drop_tables:
                    cursor.execute(f'DROP TABLE IF EXISTS {table}')

    @pytest.mark.parametrize('client', ['mysqlclient', 'mysql-connector-python'])
    @mock.patch.dict('os.environ', {'AIRFLOW_CONN_AIRFLOW_DB': 'mysql://root@mysql/airflow?charset=utf8mb4'})
    def test_mysql_hook_test_bulk_load(self, client, tmp_path):
        if False:
            while True:
                i = 10
        with MySqlContext(client):
            records = ('foo', 'bar', 'baz')
            path = tmp_path / 'testfile'
            path.write_text('\n'.join(records))
            hook = MySqlHook('airflow_db', local_infile=True)
            with closing(hook.get_conn()) as conn, closing(conn.cursor()) as cursor:
                cursor.execute('\n                    CREATE TABLE IF NOT EXISTS test_airflow (\n                        dummy VARCHAR(50)\n                    )\n                ')
                cursor.execute('TRUNCATE TABLE test_airflow')
                hook.bulk_load('test_airflow', os.fspath(path))
                cursor.execute('SELECT dummy FROM test_airflow')
                results = tuple((result[0] for result in cursor.fetchall()))
                assert sorted(results) == sorted(records)

    @pytest.mark.parametrize('client', ['mysqlclient', 'mysql-connector-python'])
    def test_mysql_hook_test_bulk_dump(self, client):
        if False:
            i = 10
            return i + 15
        with MySqlContext(client):
            hook = MySqlHook('airflow_db')
            priv = hook.get_first('SELECT @@global.secure_file_priv')
            if priv and priv[0]:
                hook.bulk_dump('INFORMATION_SCHEMA.TABLES', os.path.join(priv[0], f'TABLES_{client}-{uuid.uuid1()}'))
            elif priv == ('',):
                hook.bulk_dump('INFORMATION_SCHEMA.TABLES', f'TABLES_{client}_{uuid.uuid1()}')
            else:
                raise pytest.skip('Skip test_mysql_hook_test_bulk_load since file output is not permitted')

    @pytest.mark.parametrize('client', ['mysqlclient', 'mysql-connector-python'])
    @mock.patch('airflow.providers.mysql.hooks.mysql.MySqlHook.get_conn')
    def test_mysql_hook_test_bulk_dump_mock(self, mock_get_conn, client):
        if False:
            print('Hello World!')
        with MySqlContext(client):
            mock_execute = mock.MagicMock()
            mock_get_conn.return_value.cursor.return_value.execute = mock_execute
            hook = MySqlHook('airflow_db')
            table = 'INFORMATION_SCHEMA.TABLES'
            tmp_file = '/path/to/output/file'
            hook.bulk_dump(table, tmp_file)
            assert mock_execute.call_count == 1
            query = f"\n                SELECT * INTO OUTFILE '{tmp_file}'\n                FROM {table}\n            "
            assert_equal_ignore_multiple_spaces(mock_execute.call_args.args[0], query)