from __future__ import annotations
import importlib
import random
import string
import subprocess
from unittest import mock
import pytest
from airflow import settings
from airflow.example_dags import example_complex
from airflow.models import TaskInstance
from airflow.utils.log.log_reader import TaskLogReader
from airflow.utils.session import provide_session
from tests.providers.google.cloud.utils.gcp_authenticator import GCP_STACKDRIVER
from tests.test_utils.config import conf_vars
from tests.test_utils.db import clear_db_runs
from tests.test_utils.gcp_system_helpers import GoogleSystemTest, provide_gcp_context, resolve_full_gcp_key_path

@pytest.mark.system('google')
@pytest.mark.credential_file(GCP_STACKDRIVER)
class TestStackdriverLoggingHandlerSystem(GoogleSystemTest):

    def setup_method(self) -> None:
        if False:
            while True:
                i = 10
        clear_db_runs()
        self.log_name = 'stackdriver-tests-'.join(random.sample(string.ascii_lowercase, 16))

    def teardown_method(self) -> None:
        if False:
            while True:
                i = 10
        from airflow.config_templates import airflow_local_settings
        importlib.reload(airflow_local_settings)
        settings.configure_logging()
        clear_db_runs()

    @provide_session
    def test_should_support_key_auth(self, session):
        if False:
            while True:
                i = 10
        with mock.patch.dict('os.environ', AIRFLOW__LOGGING__REMOTE_LOGGING='true', AIRFLOW__LOGGING__REMOTE_BASE_LOG_FOLDER=f'stackdriver://{self.log_name}', AIRFLOW__LOGGING__GOOGLE_KEY_PATH=resolve_full_gcp_key_path(GCP_STACKDRIVER), AIRFLOW__CORE__LOAD_EXAMPLES='false', AIRFLOW__CORE__DAGS_FOLDER=example_complex.__file__):
            assert 0 == subprocess.Popen(['airflow', 'dags', 'trigger', 'example_complex']).wait()
            assert 0 == subprocess.Popen(['airflow', 'scheduler', '--num-runs', '1']).wait()
        ti = session.query(TaskInstance).filter(TaskInstance.task_id == 'create_entry_group').first()
        self.assert_remote_logs('terminated with exit code 0', ti)

    @provide_session
    def test_should_support_adc(self, session):
        if False:
            i = 10
            return i + 15
        with mock.patch.dict('os.environ', AIRFLOW__LOGGING__REMOTE_LOGGING='true', AIRFLOW__LOGGING__REMOTE_BASE_LOG_FOLDER=f'stackdriver://{self.log_name}', AIRFLOW__CORE__LOAD_EXAMPLES='false', AIRFLOW__CORE__DAGS_FOLDER=example_complex.__file__, GOOGLE_APPLICATION_CREDENTIALS=resolve_full_gcp_key_path(GCP_STACKDRIVER)):
            assert 0 == subprocess.Popen(['airflow', 'dags', 'trigger', 'example_complex']).wait()
            assert 0 == subprocess.Popen(['airflow', 'scheduler', '--num-runs', '1']).wait()
        ti = session.query(TaskInstance).filter(TaskInstance.task_id == 'create_entry_group').first()
        self.assert_remote_logs('terminated with exit code 0', ti)

    def assert_remote_logs(self, expected_message, ti):
        if False:
            i = 10
            return i + 15
        with provide_gcp_context(GCP_STACKDRIVER), conf_vars({('logging', 'remote_logging'): 'True', ('logging', 'remote_base_log_folder'): f'stackdriver://{self.log_name}'}):
            from airflow.config_templates import airflow_local_settings
            importlib.reload(airflow_local_settings)
            settings.configure_logging()
            task_log_reader = TaskLogReader()
            logs = '\n'.join(task_log_reader.read_log_stream(ti, try_number=None, metadata={}))
            print('=' * 80)
            print(logs)
            print('=' * 80)
            assert expected_message in logs