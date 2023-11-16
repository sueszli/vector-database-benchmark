import json
import os
import tempfile
import unittest
from unittest import mock
import pytest
from airflow import __version__ as airflow_version
from airflow.models import Connection
from dagster_airflow import make_dagster_definitions_from_airflow_dags_path
from dagster_airflow_tests.marks import requires_local_db
LOAD_CONNECTION_DAG_FILE_AIRFLOW_2_CONTENTS = '\nimport pendulum\nfrom airflow import DAG\nfrom dagster_airflow import DagsterCloudOperator\n\nwith DAG(\n    "example_connections",\n    schedule="@once",\n    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),\n    catchup=False,\n) as dag:\n    ## never succeeds outside of mocks\n    connection_test = DagsterCloudOperator(\n        task_id="connection_test",\n        job_name="connection_test",\n        run_config={"foo": "bar"},\n        dagster_conn_id="dagster_connection_test",\n    )\n\nwith DAG(\n    "example_connections_duplicate",\n    schedule="@once",\n    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),\n    catchup=False,\n) as dag:\n    ## never succeeds outside of mocks\n    connection_test = DagsterCloudOperator(\n        task_id="connection_test_duplicate",\n        job_name="connection_test",\n        run_config={"foo": "bar"},\n        dagster_conn_id="dagster_connection_test",\n    )\n\n'

@pytest.mark.skipif(airflow_version < '2.0.0', reason='requires airflow 2')
@requires_local_db
class TestConnectionsAirflow2(unittest.TestCase):

    @mock.patch('dagster_airflow.hooks.dagster_hook.DagsterHook.launch_run', return_value='run_id')
    @mock.patch('dagster_airflow.hooks.dagster_hook.DagsterHook.wait_for_run')
    def test_ingest_airflow_dags_with_connections(self, launch_run, wait_for_run):
        if False:
            print('Hello World!')
        connections = [Connection(conn_id='dagster_connection_test', conn_type='dagster', host='prod', password='test_token', description='test-org', port='test-port', schema='test-port', extra={'foo': 'bar'})]
        with tempfile.TemporaryDirectory() as tmpdir_path:
            with open(os.path.join(tmpdir_path, 'test_connection_dag.py'), 'wb') as f:
                f.write(bytes(LOAD_CONNECTION_DAG_FILE_AIRFLOW_2_CONTENTS.encode('utf-8')))
            definitions = make_dagster_definitions_from_airflow_dags_path(tmpdir_path, connections=connections)
            repo = definitions.get_repository_def()
            assert repo.has_job('example_connections')
            job = repo.get_job('example_connections')
            result = job.execute_in_process()
            assert result.success
            for event in result.all_events:
                assert event.event_type_value != 'STEP_FAILURE'
            launch_run.assert_called_once()
            wait_for_run.assert_called_once()
LOAD_CONNECTION_DAG_AIRFLOW_1_FILE_CONTENTS = '\nimport pendulum\nfrom airflow import DAG\nfrom dagster_airflow import DagsterCloudOperator\nfrom airflow.utils.dates import days_ago\n\nwith DAG(\n    "example_connections",\n    schedule_interval="@once",\n    start_date=days_ago(1),\n    catchup=False,\n) as dag:\n    ## never succeeds outside of mocks\n    connection_test = DagsterCloudOperator(\n        task_id="connection_test",\n        job_name="connection_test",\n        repository_name="test-repo",\n        repostitory_location_name="test-location",\n        user_token="test-token",\n        organization_id="test-org",\n        run_config={"foo": "bar"},\n        dagster_conn_id="dagster_connection_test",\n    )\n'

@pytest.mark.skipif(airflow_version >= '2.0.0', reason='requires airflow 1')
@requires_local_db
class TestConnectionsAirflow1(unittest.TestCase):

    @mock.patch('dagster_airflow.hooks.dagster_hook.DagsterHook.launch_run', return_value='run_id')
    @mock.patch('dagster_airflow.hooks.dagster_hook.DagsterHook.wait_for_run')
    def test_ingest_airflow_dags_with_connections(self, launch_run, wait_for_run):
        if False:
            return 10
        connections = [Connection(conn_id='dagster_connection_test', conn_type='dagster', host='prod', password='test_token', port='test-port', schema='test-port', extra=json.dumps({'foo': 'bar'}))]
        with tempfile.TemporaryDirectory() as tmpdir_path:
            with open(os.path.join(tmpdir_path, 'test_connection_dag.py'), 'wb') as f:
                f.write(bytes(LOAD_CONNECTION_DAG_AIRFLOW_1_FILE_CONTENTS.encode('utf-8')))
            definitions = make_dagster_definitions_from_airflow_dags_path(tmpdir_path, connections=connections)
            repo = definitions.get_repository_def()
            assert repo.has_job('example_connections')
            job = repo.get_job('example_connections')
            result = job.execute_in_process()
            assert result.success
            for event in result.all_events:
                assert event.event_type_value != 'STEP_FAILURE'
            launch_run.assert_called_once()
            wait_for_run.assert_called_once()