from __future__ import annotations
import importlib
import random
import string
import subprocess
from unittest import mock
import pytest
from airflow import settings
from airflow.example_dags import example_complex
from airflow.models import DagBag, TaskInstance
from airflow.utils.log.log_reader import TaskLogReader
from airflow.utils.session import provide_session
from tests.providers.google.cloud.utils.gcp_authenticator import GCP_GCS_KEY
from tests.test_utils.config import conf_vars
from tests.test_utils.db import clear_db_connections, clear_db_runs
from tests.test_utils.gcp_system_helpers import GoogleSystemTest, provide_gcp_context, resolve_full_gcp_key_path

@pytest.mark.system('google')
@pytest.mark.credential_file(GCP_GCS_KEY)
class TestGCSTaskHandlerSystem(GoogleSystemTest):

    @classmethod
    def setup_class(cls) -> None:
        if False:
            return 10
        unique_suffix = ''.join(random.sample(string.ascii_lowercase, 16))
        cls.bucket_name = f'airflow-gcs-task-handler-tests-{unique_suffix}'
        cls.create_gcs_bucket(cls.bucket_name)
        clear_db_connections()

    @classmethod
    def teardown_class(cls) -> None:
        if False:
            return 10
        cls.delete_gcs_bucket(cls.bucket_name)

    def setup_method(self) -> None:
        if False:
            i = 10
            return i + 15
        clear_db_runs()

    def teardown_method(self) -> None:
        if False:
            return 10
        from airflow.config_templates import airflow_local_settings
        importlib.reload(airflow_local_settings)
        settings.configure_logging()
        clear_db_runs()

    @provide_session
    def test_should_read_logs(self, session):
        if False:
            print('Hello World!')
        with mock.patch.dict('os.environ', AIRFLOW__LOGGING__REMOTE_LOGGING='true', AIRFLOW__LOGGING__REMOTE_BASE_LOG_FOLDER=f'gs://{self.bucket_name}/path/to/logs', AIRFLOW__LOGGING__REMOTE_LOG_CONN_ID='google_cloud_default', AIRFLOW__CORE__LOAD_EXAMPLES='false', AIRFLOW__CORE__DAGS_FOLDER=example_complex.__file__, GOOGLE_APPLICATION_CREDENTIALS=resolve_full_gcp_key_path(GCP_GCS_KEY)):
            assert 0 == subprocess.Popen(['airflow', 'dags', 'trigger', 'example_complex']).wait()
            assert 0 == subprocess.Popen(['airflow', 'scheduler', '--num-runs', '1']).wait()
        ti = session.query(TaskInstance).filter(TaskInstance.task_id == 'create_entry_group').first()
        dag = DagBag(dag_folder=example_complex.__file__).dags['example_complex']
        task = dag.task_dict['create_entry_group']
        ti.task = task
        self.assert_remote_logs('INFO - Task exited with return code 0', ti)

    def assert_remote_logs(self, expected_message, ti):
        if False:
            for i in range(10):
                print('nop')
        with provide_gcp_context(GCP_GCS_KEY), conf_vars({('logging', 'remote_logging'): 'True', ('logging', 'remote_base_log_folder'): f'gs://{self.bucket_name}/path/to/logs', ('logging', 'remote_log_conn_id'): 'google_cloud_default'}):
            from airflow.config_templates import airflow_local_settings
            importlib.reload(airflow_local_settings)
            settings.configure_logging()
            task_log_reader = TaskLogReader()
            logs = '\n'.join(task_log_reader.read_log_stream(ti, try_number=None, metadata={}))
            assert expected_message in logs