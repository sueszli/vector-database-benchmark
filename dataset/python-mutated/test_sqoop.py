from __future__ import annotations
import json
from io import StringIO
from unittest import mock
from unittest.mock import call, patch
import pytest
from airflow.exceptions import AirflowException
from airflow.models import Connection
from airflow.providers.apache.sqoop.hooks.sqoop import SqoopHook
from airflow.utils import db
pytestmark = pytest.mark.db_test

class TestSqoopHook:
    _config = {'conn_id': 'sqoop_test', 'num_mappers': 22, 'verbose': True, 'properties': {'mapred.map.max.attempts': '1'}, 'hcatalog_database': 'hive_database', 'hcatalog_table': 'hive_table'}
    _config_export_extra_options = {'extra_options': {'update-key': 'id', 'update-mode': 'allowinsert', 'fetch-size': 1}}
    _config_export = {'table': 'export_data_to', 'export_dir': '/hdfs/data/to/be/exported', 'input_null_string': '\\n', 'input_null_non_string': '\\t', 'staging_table': 'database.staging', 'clear_staging_table': True, 'enclosed_by': '"', 'escaped_by': '\\', 'input_fields_terminated_by': '|', 'input_lines_terminated_by': '\n', 'input_optionally_enclosed_by': '"', 'batch': True, 'relaxed_isolation': True, 'schema': 'domino'}
    _config_import_extra_options = {'extra_options': {'hcatalog-storage-stanza': '"stored as orcfile"', 'show': '', 'fetch-size': 1}}
    _config_import = {'target_dir': '/hdfs/data/target/location', 'append': True, 'file_type': 'parquet', 'split_by': '\n', 'direct': True, 'driver': 'com.microsoft.jdbc.sqlserver.SQLServerDriver'}
    _config_json = {'namenode': 'http://0.0.0.0:50070/', 'job_tracker': 'http://0.0.0.0:50030/', 'files': '/path/to/files', 'archives': '/path/to/archives'}

    def setup_method(self):
        if False:
            print('Hello World!')
        db.merge_conn(Connection(conn_id='sqoop_test', conn_type='sqoop', schema='schema', host='rmdbs', port=5050, extra=json.dumps(self._config_json)))
        db.merge_conn(Connection(conn_id='sqoop_test_mssql', conn_type='mssql', schema='schema', host='rmdbs', port=5050, extra=None))
        db.merge_conn(Connection(conn_id='invalid_host_conn', conn_type='mssql', schema='schema', host='rmdbs?query_param1=value1', port=5050, extra=None))
        db.merge_conn(Connection(conn_id='invalid_schema_conn', conn_type='mssql', schema='schema?query_param1=value1', host='rmdbs', port=5050, extra=None))

    @patch('subprocess.Popen')
    def test_popen(self, mock_popen):
        if False:
            while True:
                i = 10
        mock_proc = mock.MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = StringIO('stdout')
        mock_proc.stderr = StringIO('stderr')
        mock_proc.communicate.return_value = [StringIO('stdout\nstdout'), StringIO('stderr\nstderr')]
        mock_popen.return_value.__enter__.return_value = mock_proc
        hook = SqoopHook(conn_id='sqoop_test', libjars='/path/to/jars', **self._config_export_extra_options)
        hook.export_table(**self._config_export)
        assert mock_popen.mock_calls[0] == call(['sqoop', 'export', '-fs', self._config_json['namenode'], '-jt', self._config_json['job_tracker'], '-libjars', '/path/to/jars', '-files', self._config_json['files'], '-archives', self._config_json['archives'], '--connect', 'rmdbs:5050/schema', '--input-null-string', self._config_export['input_null_string'], '--input-null-non-string', self._config_export['input_null_non_string'], '--staging-table', self._config_export['staging_table'], '--clear-staging-table', '--enclosed-by', self._config_export['enclosed_by'], '--escaped-by', self._config_export['escaped_by'], '--input-fields-terminated-by', self._config_export['input_fields_terminated_by'], '--input-lines-terminated-by', self._config_export['input_lines_terminated_by'], '--input-optionally-enclosed-by', self._config_export['input_optionally_enclosed_by'], '--batch', '--relaxed-isolation', '--export-dir', self._config_export['export_dir'], '--update-key', 'id', '--update-mode', 'allowinsert', '--fetch-size', str(self._config_export_extra_options['extra_options'].get('fetch-size')), '--table', self._config_export['table'], '--', '--schema', self._config_export['schema']], stderr=-2, stdout=-1)

    def test_submit_none_mappers(self):
        if False:
            i = 10
            return i + 15
        "\n        Test to check that if value of num_mappers is None, then it shouldn't be in the cmd built.\n        "
        _config_without_mappers = self._config.copy()
        _config_without_mappers['num_mappers'] = None
        hook = SqoopHook(**_config_without_mappers)
        cmd = ' '.join(hook._prepare_command())
        assert '--num-mappers' not in cmd

    def test_submit(self):
        if False:
            print('Hello World!')
        '\n        Tests to verify that from connection extra option the options are added to the Sqoop command.\n        '
        hook = SqoopHook(**self._config)
        cmd = ' '.join(hook._prepare_command())
        if self._config_json['namenode']:
            assert f"-fs {self._config_json['namenode']}" in cmd
        if self._config_json['job_tracker']:
            assert f"-jt {self._config_json['job_tracker']}" in cmd
        if self._config_json['files']:
            assert f"-files {self._config_json['files']}" in cmd
        if self._config_json['archives']:
            assert f"-archives {self._config_json['archives']}" in cmd
        assert f"--hcatalog-database {self._config['hcatalog_database']}" in cmd
        assert f"--hcatalog-table {self._config['hcatalog_table']}" in cmd
        if self._config['verbose']:
            assert '--verbose' in cmd
        if self._config['num_mappers']:
            assert f"--num-mappers {self._config['num_mappers']}" in cmd
        for (key, value) in self._config['properties'].items():
            assert f'-D {key}={value}' in cmd
        with pytest.raises(OSError):
            hook.export_table(**self._config_export)
        with pytest.raises(OSError):
            hook.import_table(table='table', target_dir='/sqoop/example/path', schema='schema')
        with pytest.raises(OSError):
            hook.import_query(query='SELECT * FROM sometable', target_dir='/sqoop/example/path')

    def test_export_cmd(self):
        if False:
            while True:
                i = 10
        '\n        Tests to verify the hook export command is building correct Sqoop export command.\n        '
        hook = SqoopHook(**self._config_export_extra_options)
        cmd = ' '.join(hook._export_cmd(self._config_export['table'], self._config_export['export_dir'], input_null_string=self._config_export['input_null_string'], input_null_non_string=self._config_export['input_null_non_string'], staging_table=self._config_export['staging_table'], clear_staging_table=self._config_export['clear_staging_table'], enclosed_by=self._config_export['enclosed_by'], escaped_by=self._config_export['escaped_by'], input_fields_terminated_by=self._config_export['input_fields_terminated_by'], input_lines_terminated_by=self._config_export['input_lines_terminated_by'], input_optionally_enclosed_by=self._config_export['input_optionally_enclosed_by'], batch=self._config_export['batch'], relaxed_isolation=self._config_export['relaxed_isolation'], schema=self._config_export['schema']))
        assert f"--input-null-string {self._config_export['input_null_string']}" in cmd
        assert f"--input-null-non-string {self._config_export['input_null_non_string']}" in cmd
        assert f"--staging-table {self._config_export['staging_table']}" in cmd
        assert f"--enclosed-by {self._config_export['enclosed_by']}" in cmd
        assert f"--escaped-by {self._config_export['escaped_by']}" in cmd
        assert f"--input-fields-terminated-by {self._config_export['input_fields_terminated_by']}" in cmd
        assert f"--input-lines-terminated-by {self._config_export['input_lines_terminated_by']}" in cmd
        assert f"--input-optionally-enclosed-by {self._config_export['input_optionally_enclosed_by']}" in cmd
        assert '--update-key id' in cmd
        assert '--update-mode allowinsert' in cmd
        if self._config_export['clear_staging_table']:
            assert '--clear-staging-table' in cmd
        if self._config_export['batch']:
            assert '--batch' in cmd
        if self._config_export['relaxed_isolation']:
            assert '--relaxed-isolation' in cmd
        if self._config_export_extra_options['extra_options']:
            assert '--update-key' in cmd
            assert '--update-mode' in cmd
            assert '--fetch-size' in cmd
        if self._config_export['schema']:
            assert '-- --schema' in cmd

    def test_import_cmd(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests to verify the hook import command is building correct Sqoop import command.\n        '
        hook = SqoopHook()
        cmd = ' '.join(hook._import_cmd(self._config_import['target_dir'], append=self._config_import['append'], file_type=self._config_import['file_type'], split_by=self._config_import['split_by'], direct=self._config_import['direct'], driver=self._config_import['driver']))
        if self._config_import['append']:
            assert '--append' in cmd
        if self._config_import['direct']:
            assert '--direct' in cmd
        assert f"--target-dir {self._config_import['target_dir']}" in cmd
        assert f"--driver {self._config_import['driver']}" in cmd
        assert f"--split-by {self._config_import['split_by']}" in cmd
        assert '--show' not in cmd
        assert 'hcatalog-storage-stanza "stored as orcfile"' not in cmd
        hook = SqoopHook(**self._config_import_extra_options)
        cmd = ' '.join(hook._import_cmd(target_dir=None, append=self._config_import['append'], file_type=self._config_import['file_type'], split_by=self._config_import['split_by'], direct=self._config_import['direct'], driver=self._config_import['driver']))
        assert '--target-dir' not in cmd
        assert '--show' in cmd
        assert 'hcatalog-storage-stanza "stored as orcfile"' in cmd
        assert '--fetch-size' in cmd

    def test_get_export_format_argument(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests to verify the hook get format function is building\n        correct Sqoop command with correct format type.\n        '
        hook = SqoopHook()
        assert '--as-avrodatafile' in hook._get_export_format_argument('avro')
        assert '--as-parquetfile' in hook._get_export_format_argument('parquet')
        assert '--as-sequencefile' in hook._get_export_format_argument('sequence')
        assert '--as-textfile' in hook._get_export_format_argument('text')
        with pytest.raises(AirflowException):
            hook._get_export_format_argument('unknown')

    def test_cmd_mask_password(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests to verify the hook masking function will correctly mask a user password in Sqoop command.\n        '
        hook = SqoopHook()
        assert hook.cmd_mask_password(['--password', 'supersecret']) == ['--password', 'MASKED']
        cmd = ['--target', 'targettable']
        assert hook.cmd_mask_password(cmd) == cmd

    def test_connection_string_preparation(self):
        if False:
            print('Hello World!')
        '\n        Tests to verify the hook creates the connection string correctly for mssql and not DB connections.\n        '
        hook = SqoopHook(conn_id='sqoop_test_mssql')
        assert f'{hook.conn.host}:{hook.conn.port};databaseName={hook.conn.schema}' in hook._prepare_command()
        hook = SqoopHook(conn_id='sqoop_test')
        assert f'{hook.conn.host}:{hook.conn.port}/{hook.conn.schema}' in hook._prepare_command()

    def test_invalid_host(self):
        if False:
            return 10
        hook = SqoopHook(conn_id='invalid_host_conn')
        with pytest.raises(ValueError, match='should not contain a'):
            hook._prepare_command()

    def test_invalid_schema(self):
        if False:
            while True:
                i = 10
        hook = SqoopHook(conn_id='invalid_schema_conn')
        with pytest.raises(ValueError, match='should not contain a'):
            hook._prepare_command()