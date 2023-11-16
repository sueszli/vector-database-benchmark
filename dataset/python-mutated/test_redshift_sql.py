from __future__ import annotations
import json
from unittest import mock
import pytest
from airflow.exceptions import AirflowException
from airflow.models import Connection
from airflow.providers.amazon.aws.hooks.redshift_sql import RedshiftSQLHook
from airflow.utils.types import NOTSET
LOGIN_USER = 'login'
LOGIN_PASSWORD = 'password'
LOGIN_HOST = 'host'
LOGIN_PORT = 5439
LOGIN_SCHEMA = 'dev'

class TestRedshiftSQLHookConn:

    def setup_method(self):
        if False:
            return 10
        self.connection = Connection(conn_type='redshift', login=LOGIN_USER, password=LOGIN_PASSWORD, host=LOGIN_HOST, port=LOGIN_PORT, schema=LOGIN_SCHEMA)
        self.db_hook = RedshiftSQLHook()
        self.db_hook.get_connection = mock.Mock()
        self.db_hook.get_connection.return_value = self.connection

    def test_get_uri(self):
        if False:
            for i in range(10):
                print('nop')
        expected = 'redshift+redshift_connector://login:password@host:5439/dev'
        x = self.db_hook.get_uri()
        assert x == expected

    @mock.patch('airflow.providers.amazon.aws.hooks.redshift_sql.redshift_connector.connect')
    def test_get_conn(self, mock_connect):
        if False:
            while True:
                i = 10
        self.db_hook.get_conn()
        mock_connect.assert_called_once_with(user='login', password='password', host='host', port=5439, database='dev')

    @mock.patch('airflow.providers.amazon.aws.hooks.redshift_sql.redshift_connector.connect')
    def test_get_conn_extra(self, mock_connect):
        if False:
            print('Hello World!')
        self.connection.extra = json.dumps({'iam': False, 'cluster_identifier': 'my-test-cluster', 'profile': 'default'})
        self.db_hook.get_conn()
        mock_connect.assert_called_once_with(user=LOGIN_USER, password=LOGIN_PASSWORD, host=LOGIN_HOST, port=LOGIN_PORT, cluster_identifier='my-test-cluster', profile='default', database=LOGIN_SCHEMA, iam=False)

    @mock.patch('airflow.providers.amazon.aws.hooks.base_aws.AwsBaseHook.conn')
    @mock.patch('airflow.providers.amazon.aws.hooks.redshift_sql.redshift_connector.connect')
    @pytest.mark.parametrize('aws_conn_id', [NOTSET, None, 'mock_aws_conn'])
    def test_get_conn_iam(self, mock_connect, mock_aws_hook_conn, aws_conn_id):
        if False:
            print('Hello World!')
        mock_conn_extra = {'iam': True, 'profile': 'default', 'cluster_identifier': 'my-test-cluster'}
        if aws_conn_id is not NOTSET:
            self.db_hook.aws_conn_id = aws_conn_id
        self.connection.extra = json.dumps(mock_conn_extra)
        mock_db_user = f'IAM:{self.connection.login}'
        mock_db_pass = 'aws_token'
        mock_aws_hook_conn.get_cluster_credentials.return_value = {'DbPassword': mock_db_pass, 'DbUser': mock_db_user}
        self.db_hook.get_conn()
        mock_aws_hook_conn.get_cluster_credentials.assert_called_once_with(DbUser=LOGIN_USER, DbName=LOGIN_SCHEMA, ClusterIdentifier='my-test-cluster', AutoCreate=False)
        mock_connect.assert_called_once_with(user=mock_db_user, password=mock_db_pass, host=LOGIN_HOST, port=LOGIN_PORT, cluster_identifier='my-test-cluster', profile='default', database=LOGIN_SCHEMA, iam=True)

    @pytest.mark.parametrize('conn_params, conn_extra, expected_call_args', [({}, {}, {}), ({'login': 'test'}, {}, {'user': 'test'}), ({}, {'user': 'test'}, {'user': 'test'}), ({'login': 'original'}, {'user': 'overridden'}, {'user': 'overridden'}), ({'login': 'test1'}, {'password': 'test2'}, {'user': 'test1', 'password': 'test2'})])
    @mock.patch('airflow.providers.amazon.aws.hooks.redshift_sql.redshift_connector.connect')
    def test_get_conn_overrides_correctly(self, mock_connect, conn_params, conn_extra, expected_call_args):
        if False:
            while True:
                i = 10
        with mock.patch('airflow.providers.amazon.aws.hooks.redshift_sql.RedshiftSQLHook.conn', Connection(conn_type='redshift', extra=conn_extra, **conn_params)):
            self.db_hook.get_conn()
            mock_connect.assert_called_once_with(**expected_call_args)

    @pytest.mark.parametrize('connection_host, connection_extra, expected_cluster_identifier, expected_exception_msg', [(None, {'iam': True}, None, 'Please set cluster_identifier or host in redshift connection.'), (None, {'iam': True, 'cluster_identifier': 'cluster_identifier_from_extra'}, 'cluster_identifier_from_extra', None), ('cluster_identifier_from_host.x.y', {'iam': True}, 'cluster_identifier_from_host', None), ('cluster_identifier_from_host.x.y', {'iam': True, 'cluster_identifier': 'cluster_identifier_from_extra'}, 'cluster_identifier_from_extra', None)])
    @mock.patch('airflow.providers.amazon.aws.hooks.base_aws.AwsBaseHook.conn')
    @mock.patch('airflow.providers.amazon.aws.hooks.redshift_sql.redshift_connector.connect')
    def test_get_iam_token(self, mock_connect, mock_aws_hook_conn, connection_host, connection_extra, expected_cluster_identifier, expected_exception_msg):
        if False:
            i = 10
            return i + 15
        self.connection.host = connection_host
        self.connection.extra = json.dumps(connection_extra)
        mock_db_user = f'IAM:{self.connection.login}'
        mock_db_pass = 'aws_token'
        mock_aws_hook_conn.get_cluster_credentials.return_value = {'DbPassword': mock_db_pass, 'DbUser': mock_db_user}
        if expected_exception_msg is not None:
            with pytest.raises(AirflowException, match=expected_exception_msg):
                self.db_hook.get_conn()
        else:
            self.db_hook.get_conn()
            mock_aws_hook_conn.get_cluster_credentials.assert_called_once_with(DbUser=LOGIN_USER, DbName=LOGIN_SCHEMA, ClusterIdentifier=expected_cluster_identifier, AutoCreate=False)