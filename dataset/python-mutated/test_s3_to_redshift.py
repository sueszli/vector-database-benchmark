from __future__ import annotations
from copy import deepcopy
from unittest import mock
import pytest
from boto3.session import Session
from airflow.exceptions import AirflowException
from airflow.models.connection import Connection
from airflow.providers.amazon.aws.transfers.s3_to_redshift import S3ToRedshiftOperator
from tests.test_utils.asserts import assert_equal_ignore_multiple_spaces

class TestS3ToRedshiftTransfer:

    @mock.patch('airflow.providers.amazon.aws.hooks.s3.S3Hook.get_connection')
    @mock.patch('airflow.models.connection.Connection')
    @mock.patch('boto3.session.Session')
    @mock.patch('airflow.providers.amazon.aws.hooks.redshift_sql.RedshiftSQLHook.run')
    def test_execute(self, mock_run, mock_session, mock_connection, mock_hook):
        if False:
            for i in range(10):
                print('nop')
        access_key = 'aws_access_key_id'
        secret_key = 'aws_secret_access_key'
        mock_session.return_value = Session(access_key, secret_key)
        mock_session.return_value.access_key = access_key
        mock_session.return_value.secret_key = secret_key
        mock_session.return_value.token = None
        mock_connection.return_value = Connection()
        mock_hook.return_value = Connection()
        schema = 'schema'
        table = 'table'
        s3_bucket = 'bucket'
        s3_key = 'key'
        copy_options = ''
        op = S3ToRedshiftOperator(schema=schema, table=table, s3_bucket=s3_bucket, s3_key=s3_key, copy_options=copy_options, redshift_conn_id='redshift_conn_id', aws_conn_id='aws_conn_id', task_id='task_id', dag=None)
        op.execute(None)
        expected_copy_query = "\n                        COPY schema.table\n                        FROM 's3://bucket/key'\n                        credentials\n                        'aws_access_key_id=aws_access_key_id;aws_secret_access_key=aws_secret_access_key'\n                        ;\n                     "
        actual_copy_query = mock_run.call_args.args[0]
        assert mock_run.call_count == 1
        assert access_key in actual_copy_query
        assert secret_key in actual_copy_query
        assert_equal_ignore_multiple_spaces(actual_copy_query, expected_copy_query)

    @mock.patch('airflow.providers.amazon.aws.hooks.s3.S3Hook.get_connection')
    @mock.patch('airflow.models.connection.Connection')
    @mock.patch('boto3.session.Session')
    @mock.patch('airflow.providers.amazon.aws.hooks.redshift_sql.RedshiftSQLHook.run')
    def test_execute_with_column_list(self, mock_run, mock_session, mock_connection, mock_hook):
        if False:
            for i in range(10):
                print('nop')
        access_key = 'aws_access_key_id'
        secret_key = 'aws_secret_access_key'
        mock_session.return_value = Session(access_key, secret_key)
        mock_session.return_value.access_key = access_key
        mock_session.return_value.secret_key = secret_key
        mock_session.return_value.token = None
        mock_connection.return_value = Connection()
        mock_hook.return_value = Connection()
        schema = 'schema'
        table = 'table'
        s3_bucket = 'bucket'
        s3_key = 'key'
        column_list = ['column_1', 'column_2']
        copy_options = ''
        op = S3ToRedshiftOperator(schema=schema, table=table, s3_bucket=s3_bucket, s3_key=s3_key, column_list=column_list, copy_options=copy_options, redshift_conn_id='redshift_conn_id', aws_conn_id='aws_conn_id', task_id='task_id', dag=None)
        op.execute(None)
        expected_copy_query = "\n                        COPY schema.table (column_1, column_2)\n                        FROM 's3://bucket/key'\n                        credentials\n                        'aws_access_key_id=aws_access_key_id;aws_secret_access_key=aws_secret_access_key'\n                        ;\n                     "
        actual_copy_query = mock_run.call_args.args[0]
        assert mock_run.call_count == 1
        assert access_key in actual_copy_query
        assert secret_key in actual_copy_query
        assert_equal_ignore_multiple_spaces(actual_copy_query, expected_copy_query)

    @mock.patch('airflow.providers.amazon.aws.hooks.s3.S3Hook.get_connection')
    @mock.patch('airflow.models.connection.Connection')
    @mock.patch('boto3.session.Session')
    @mock.patch('airflow.providers.amazon.aws.hooks.redshift_sql.RedshiftSQLHook.run')
    def test_replace(self, mock_run, mock_session, mock_connection, mock_hook):
        if False:
            for i in range(10):
                print('nop')
        access_key = 'aws_access_key_id'
        secret_key = 'aws_secret_access_key'
        mock_session.return_value = Session(access_key, secret_key)
        mock_session.return_value.access_key = access_key
        mock_session.return_value.secret_key = secret_key
        mock_session.return_value.token = None
        mock_connection.return_value = Connection()
        mock_hook.return_value = Connection()
        schema = 'schema'
        table = 'table'
        s3_bucket = 'bucket'
        s3_key = 'key'
        copy_options = ''
        op = S3ToRedshiftOperator(schema=schema, table=table, s3_bucket=s3_bucket, s3_key=s3_key, copy_options=copy_options, method='REPLACE', redshift_conn_id='redshift_conn_id', aws_conn_id='aws_conn_id', task_id='task_id', dag=None)
        op.execute(None)
        copy_statement = "\n                        COPY schema.table\n                        FROM 's3://bucket/key'\n                        credentials\n                        'aws_access_key_id=aws_access_key_id;aws_secret_access_key=aws_secret_access_key'\n                        ;\n                     "
        delete_statement = f'DELETE FROM {schema}.{table};'
        transaction = f'\n                    BEGIN;\n                    {delete_statement}\n                    {copy_statement}\n                    COMMIT\n                    '
        assert_equal_ignore_multiple_spaces('\n'.join(mock_run.call_args.args[0]), transaction)
        assert mock_run.call_count == 1

    @mock.patch('airflow.providers.amazon.aws.hooks.s3.S3Hook.get_connection')
    @mock.patch('airflow.models.connection.Connection')
    @mock.patch('boto3.session.Session')
    @mock.patch('airflow.providers.amazon.aws.hooks.redshift_sql.RedshiftSQLHook.run')
    def test_upsert(self, mock_run, mock_session, mock_connection, mock_hook):
        if False:
            i = 10
            return i + 15
        access_key = 'aws_access_key_id'
        secret_key = 'aws_secret_access_key'
        mock_session.return_value = Session(access_key, secret_key)
        mock_session.return_value.access_key = access_key
        mock_session.return_value.secret_key = secret_key
        mock_session.return_value.token = None
        mock_connection.return_value = Connection()
        mock_hook.return_value = Connection()
        schema = 'schema'
        table = 'table'
        s3_bucket = 'bucket'
        s3_key = 'key'
        copy_options = ''
        op = S3ToRedshiftOperator(schema=schema, table=table, s3_bucket=s3_bucket, s3_key=s3_key, copy_options=copy_options, method='UPSERT', upsert_keys=['id'], redshift_conn_id='redshift_conn_id', aws_conn_id='aws_conn_id', task_id='task_id', dag=None)
        op.execute(None)
        copy_statement = f"\n                        COPY #{table}\n                        FROM 's3://bucket/key'\n                        credentials\n                        'aws_access_key_id=aws_access_key_id;aws_secret_access_key=aws_secret_access_key'\n                        ;\n                     "
        transaction = f'\n                    CREATE TABLE #{table} (LIKE {schema}.{table} INCLUDING DEFAULTS);\n                    {copy_statement}\n                    BEGIN;\n                    DELETE FROM {schema}.{table} USING #{table} WHERE {table}.id = #{table}.id;\n                    INSERT INTO {schema}.{table} SELECT * FROM #{table};\n                    COMMIT\n                    '
        assert_equal_ignore_multiple_spaces('\n'.join(mock_run.call_args.args[0]), transaction)
        assert mock_run.call_count == 1

    @mock.patch('airflow.providers.amazon.aws.hooks.s3.S3Hook.get_connection')
    @mock.patch('airflow.models.connection.Connection')
    @mock.patch('boto3.session.Session')
    @mock.patch('airflow.providers.amazon.aws.hooks.redshift_sql.RedshiftSQLHook.run')
    def test_execute_sts_token(self, mock_run, mock_session, mock_connection, mock_hook):
        if False:
            i = 10
            return i + 15
        access_key = 'ASIA_aws_access_key_id'
        secret_key = 'aws_secret_access_key'
        token = 'aws_secret_token'
        mock_session.return_value = Session(access_key, secret_key, token)
        mock_session.return_value.access_key = access_key
        mock_session.return_value.secret_key = secret_key
        mock_session.return_value.token = token
        mock_connection.return_value = Connection()
        mock_hook.return_value = Connection()
        schema = 'schema'
        table = 'table'
        s3_bucket = 'bucket'
        s3_key = 'key'
        copy_options = ''
        op = S3ToRedshiftOperator(schema=schema, table=table, s3_bucket=s3_bucket, s3_key=s3_key, copy_options=copy_options, redshift_conn_id='redshift_conn_id', aws_conn_id='aws_conn_id', task_id='task_id', dag=None)
        op.execute(None)
        expected_copy_query = "\n                            COPY schema.table\n                            FROM 's3://bucket/key'\n                            credentials\n                            'aws_access_key_id=ASIA_aws_access_key_id;aws_secret_access_key=aws_secret_access_key;token=aws_secret_token'\n                            ;\n                         "
        actual_copy_query = mock_run.call_args.args[0]
        assert access_key in actual_copy_query
        assert secret_key in actual_copy_query
        assert token in actual_copy_query
        assert mock_run.call_count == 1
        assert_equal_ignore_multiple_spaces(actual_copy_query, expected_copy_query)

    @mock.patch('airflow.providers.amazon.aws.hooks.s3.S3Hook.get_connection')
    @mock.patch('airflow.models.connection.Connection')
    @mock.patch('boto3.session.Session')
    @mock.patch('airflow.providers.amazon.aws.hooks.redshift_sql.RedshiftSQLHook.run')
    def test_execute_role_arn(self, mock_run, mock_session, mock_connection, mock_hook):
        if False:
            i = 10
            return i + 15
        access_key = 'ASIA_aws_access_key_id'
        secret_key = 'aws_secret_access_key'
        token = 'aws_secret_token'
        extra = {'role_arn': 'arn:aws:iam::112233445566:role/myRole'}
        mock_session.return_value = Session(access_key, secret_key, token)
        mock_session.return_value.access_key = access_key
        mock_session.return_value.secret_key = secret_key
        mock_session.return_value.token = token
        mock_connection.return_value = Connection(extra=extra)
        mock_hook.return_value = Connection(extra=extra)
        schema = 'schema'
        table = 'table'
        s3_bucket = 'bucket'
        s3_key = 'key'
        copy_options = ''
        op = S3ToRedshiftOperator(schema=schema, table=table, s3_bucket=s3_bucket, s3_key=s3_key, copy_options=copy_options, redshift_conn_id='redshift_conn_id', aws_conn_id='aws_conn_id', task_id='task_id', dag=None)
        op.execute(None)
        expected_copy_query = "\n                            COPY schema.table\n                            FROM 's3://bucket/key'\n                            credentials\n                            'aws_iam_role=arn:aws:iam::112233445566:role/myRole'\n                            ;\n                         "
        actual_copy_query = mock_run.call_args.args[0]
        assert extra['role_arn'] in actual_copy_query
        assert mock_run.call_count == 1
        assert_equal_ignore_multiple_spaces(actual_copy_query, expected_copy_query)

    @mock.patch('airflow.providers.amazon.aws.hooks.s3.S3Hook.get_connection')
    @mock.patch('airflow.models.connection.Connection')
    @mock.patch('boto3.session.Session')
    @mock.patch('airflow.providers.amazon.aws.hooks.redshift_sql.RedshiftSQLHook.run')
    def test_different_region(self, mock_run, mock_session, mock_connection, mock_hook):
        if False:
            print('Hello World!')
        access_key = 'aws_access_key_id'
        secret_key = 'aws_secret_access_key'
        extra = {'region': 'eu-central-1'}
        mock_session.return_value = Session(access_key, secret_key)
        mock_session.return_value.access_key = access_key
        mock_session.return_value.secret_key = secret_key
        mock_session.return_value.token = None
        mock_connection.return_value = Connection(extra=extra)
        mock_hook.return_value = Connection(extra=extra)
        schema = 'schema'
        table = 'table'
        s3_bucket = 'bucket'
        s3_key = 'key'
        copy_options = ''
        op = S3ToRedshiftOperator(schema=schema, table=table, s3_bucket=s3_bucket, s3_key=s3_key, copy_options=copy_options, redshift_conn_id='redshift_conn_id', aws_conn_id='aws_conn_id', task_id='task_id', dag=None)
        op.execute(None)
        expected_copy_query = "\n                        COPY schema.table\n                        FROM 's3://bucket/key'\n                        credentials\n                        'aws_access_key_id=aws_access_key_id;aws_secret_access_key=aws_secret_access_key'\n                        region 'eu-central-1'\n                        ;\n                     "
        actual_copy_query = mock_run.call_args.args[0]
        assert access_key in actual_copy_query
        assert secret_key in actual_copy_query
        assert extra['region'] in actual_copy_query
        assert mock_run.call_count == 1
        assert_equal_ignore_multiple_spaces(actual_copy_query, expected_copy_query)

    def test_template_fields_overrides(self):
        if False:
            i = 10
            return i + 15
        assert S3ToRedshiftOperator.template_fields == ('s3_bucket', 's3_key', 'schema', 'table', 'column_list', 'copy_options', 'redshift_conn_id', 'method')

    def test_execute_unavailable_method(self):
        if False:
            i = 10
            return i + 15
        '\n        Test execute unavailable method\n        '
        with pytest.raises(AirflowException):
            S3ToRedshiftOperator(schema='schema', table='table', s3_bucket='bucket', s3_key='key', method='unavailable_method', task_id='task_id', dag=None).execute({})

    @pytest.mark.parametrize('param', ['sql', 'parameters'])
    def test_invalid_param_in_redshift_data_api_kwargs(self, param):
        if False:
            print('Hello World!')
        '\n        Test passing invalid param in RS Data API kwargs raises an error\n        '
        with pytest.raises(AirflowException):
            S3ToRedshiftOperator(schema='schema', table='table', s3_bucket='bucket', s3_key='key', task_id='task_id', dag=None, redshift_data_api_kwargs={param: 'param'})

    @mock.patch('airflow.providers.amazon.aws.hooks.s3.S3Hook.get_connection')
    @mock.patch('airflow.models.connection.Connection')
    @mock.patch('boto3.session.Session')
    @mock.patch('airflow.providers.amazon.aws.hooks.redshift_sql.RedshiftSQLHook.run')
    @mock.patch('airflow.providers.amazon.aws.hooks.redshift_data.RedshiftDataHook.conn')
    def test_using_redshift_data_api(self, mock_rs, mock_run, mock_session, mock_connection, mock_hook):
        if False:
            for i in range(10):
                print('nop')
        '\n        Using the Redshift Data API instead of the SQL-based connection\n        '
        access_key = 'aws_access_key_id'
        secret_key = 'aws_secret_access_key'
        mock_session.return_value = Session(access_key, secret_key)
        mock_session.return_value.access_key = access_key
        mock_session.return_value.secret_key = secret_key
        mock_session.return_value.token = None
        mock_connection.return_value = Connection()
        mock_hook.return_value = Connection()
        mock_rs.execute_statement.return_value = {'Id': 'STATEMENT_ID'}
        mock_rs.describe_statement.return_value = {'Status': 'FINISHED'}
        schema = 'schema'
        table = 'table'
        s3_bucket = 'bucket'
        s3_key = 'key'
        copy_options = ''
        database = 'database'
        cluster_identifier = 'cluster_identifier'
        db_user = 'db_user'
        secret_arn = 'secret_arn'
        statement_name = 'statement_name'
        op = S3ToRedshiftOperator(schema=schema, table=table, s3_bucket=s3_bucket, s3_key=s3_key, copy_options=copy_options, redshift_conn_id='redshift_conn_id', aws_conn_id='aws_conn_id', task_id='task_id', dag=None, redshift_data_api_kwargs=dict(database=database, cluster_identifier=cluster_identifier, db_user=db_user, secret_arn=secret_arn, statement_name=statement_name))
        op.execute(None)
        mock_run.assert_not_called()
        mock_rs.execute_statement.assert_called_once()
        _call = deepcopy(mock_rs.execute_statement.call_args.kwargs)
        _call.pop('Sql')
        assert _call == dict(Database=database, ClusterIdentifier=cluster_identifier, DbUser=db_user, SecretArn=secret_arn, StatementName=statement_name, WithEvent=False)
        expected_copy_query = "\n                        COPY schema.table\n                        FROM 's3://bucket/key'\n                        credentials\n                        'aws_access_key_id=aws_access_key_id;aws_secret_access_key=aws_secret_access_key'\n                        ;\n                     "
        actual_copy_query = mock_rs.execute_statement.call_args.kwargs['Sql']
        mock_rs.describe_statement.assert_called_once_with(Id='STATEMENT_ID')
        assert access_key in actual_copy_query
        assert secret_key in actual_copy_query
        assert_equal_ignore_multiple_spaces(actual_copy_query, expected_copy_query)