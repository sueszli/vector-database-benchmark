from __future__ import annotations
import os
import tempfile
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Sequence
from unittest import mock
import pytest
from google.auth.environment_vars import CLOUD_SDK_CONFIG_DIR, CREDENTIALS
from airflow.providers.google.cloud.utils.credentials_provider import provide_gcp_conn_and_credentials
from tests.providers.google.cloud.utils.gcp_authenticator import GCP_GCS_KEY, GCP_SECRET_MANAGER_KEY
from tests.test_utils import AIRFLOW_MAIN_FOLDER
from tests.test_utils.logging_command_executor import CommandExecutor
from tests.test_utils.system_tests_class import SystemTest
CLOUD_DAG_FOLDER = os.path.join(AIRFLOW_MAIN_FOLDER, 'airflow', 'providers', 'google', 'cloud', 'example_dags')
MARKETING_DAG_FOLDER = os.path.join(AIRFLOW_MAIN_FOLDER, 'airflow', 'providers', 'google', 'marketing_platform', 'example_dags')
GSUITE_DAG_FOLDER = os.path.join(AIRFLOW_MAIN_FOLDER, 'airflow', 'providers', 'google', 'suite', 'example_dags')
FIREBASE_DAG_FOLDER = os.path.join(AIRFLOW_MAIN_FOLDER, 'airflow', 'providers', 'google', 'firebase', 'example_dags')
LEVELDB_DAG_FOLDER = os.path.join(AIRFLOW_MAIN_FOLDER, 'airflow', 'providers', 'google', 'leveldb', 'example_dags')
POSTGRES_LOCAL_EXECUTOR = os.path.join(AIRFLOW_MAIN_FOLDER, 'tests', 'test_utils', 'postgres_local_executor.cfg')

def resolve_full_gcp_key_path(key: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Returns path full path to provided GCP key.\n\n    :param key: Name of the GCP key, for example ``my_service.json``\n    :returns: Full path to the key\n    '
    path = os.environ.get('CREDENTIALS_DIR', '/files/airflow-breeze-config/keys')
    key = os.path.join(path, key)
    return key

@contextmanager
def provide_gcp_context(key_file_path: str | None=None, scopes: Sequence | None=None, project_id: str | None=None):
    if False:
        return 10
    '\n    Context manager that provides:\n\n    - GCP credentials for application supporting `Application Default Credentials (ADC)\n    strategy <https://cloud.google.com/docs/authentication/production>`__.\n    - temporary value of :envvar:`AIRFLOW_CONN_GOOGLE_CLOUD_DEFAULT` variable\n    - the ``gcloud`` config directory isolated from user configuration\n\n    Moreover it resolves full path to service keys so user can pass ``myservice.json``\n    as ``key_file_path``.\n\n    :param key_file_path: Path to file with GCP credentials .json file.\n    :param scopes: OAuth scopes for the connection\n    :param project_id: The id of GCP project for the connection.\n        Default: ``os.environ["GCP_PROJECT_ID"]`` or None\n    '
    key_file_path = resolve_full_gcp_key_path(key_file_path)
    if project_id is None:
        project_id = os.environ.get('GCP_PROJECT_ID')
    with provide_gcp_conn_and_credentials(key_file_path, scopes, project_id), tempfile.TemporaryDirectory() as gcloud_config_tmp, mock.patch.dict('os.environ', {CLOUD_SDK_CONFIG_DIR: gcloud_config_tmp}):
        executor = CommandExecutor()
        if key_file_path:
            executor.execute_cmd(['gcloud', 'auth', 'activate-service-account', f'--key-file={key_file_path}'])
        if project_id:
            executor.execute_cmd(['gcloud', 'config', 'set', 'core/project', project_id])
        yield

@contextmanager
@provide_gcp_context(GCP_GCS_KEY)
def provide_gcs_bucket(bucket_name: str):
    if False:
        print('Hello World!')
    GoogleSystemTest.create_gcs_bucket(bucket_name)
    yield
    GoogleSystemTest.delete_gcs_bucket(bucket_name)

@pytest.mark.system('google')
class GoogleSystemTest(SystemTest):

    @staticmethod
    def execute_cmd(*args, **kwargs):
        if False:
            return 10
        executor = CommandExecutor()
        return executor.execute_cmd(*args, **kwargs)

    @staticmethod
    def _project_id():
        if False:
            return 10
        return os.environ.get('GCP_PROJECT_ID')

    @staticmethod
    def _service_key():
        if False:
            return 10
        return os.environ.get(CREDENTIALS)

    @classmethod
    def execute_with_ctx(cls, cmd: list[str], key: str=GCP_GCS_KEY, project_id=None, scopes=None, silent: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        Executes command with context created by provide_gcp_context and activated\n        service key.\n        '
        current_project_id = project_id or cls._project_id()
        with provide_gcp_context(key, project_id=current_project_id, scopes=scopes):
            cls.execute_cmd(cmd=cmd, silent=silent)

    @classmethod
    def create_gcs_bucket(cls, name: str, location: str | None=None) -> None:
        if False:
            print('Hello World!')
        bucket_name = f'gs://{name}' if not name.startswith('gs://') else name
        cmd = ['gsutil', 'mb']
        if location:
            cmd += ['-c', 'regional', '-l', location]
        cmd += [bucket_name]
        cls.execute_with_ctx(cmd, key=GCP_GCS_KEY)

    @classmethod
    def delete_gcs_bucket(cls, name: str):
        if False:
            i = 10
            return i + 15
        bucket_name = f'gs://{name}' if not name.startswith('gs://') else name
        cmd = ['gsutil', '-m', 'rm', '-r', bucket_name]
        cls.execute_with_ctx(cmd, key=GCP_GCS_KEY)

    @classmethod
    def upload_to_gcs(cls, source_uri: str, target_uri: str):
        if False:
            print('Hello World!')
        cls.execute_with_ctx(['gsutil', 'cp', source_uri, target_uri], key=GCP_GCS_KEY)

    @classmethod
    def upload_content_to_gcs(cls, lines: str, bucket: str, filename: str):
        if False:
            while True:
                i = 10
        bucket_name = f'gs://{bucket}' if not bucket.startswith('gs://') else bucket
        with TemporaryDirectory(prefix='airflow-gcp') as tmp_dir:
            tmp_path = os.path.join(tmp_dir, filename)
            tmp_dir_path = os.path.dirname(tmp_path)
            if tmp_dir_path:
                os.makedirs(tmp_dir_path, exist_ok=True)
            with open(tmp_path, 'w') as file:
                file.writelines(lines)
                file.flush()
            os.chmod(tmp_path, 777)
            cls.upload_to_gcs(tmp_path, bucket_name)

    @classmethod
    def get_project_number(cls, project_id: str) -> str:
        if False:
            return 10
        cmd = ['gcloud', 'projects', 'describe', project_id, '--format', 'value(projectNumber)']
        return cls.check_output(cmd).decode('utf-8').strip()

    @classmethod
    def grant_bucket_access(cls, bucket: str, account_email: str):
        if False:
            print('Hello World!')
        bucket_name = f'gs://{bucket}' if not bucket.startswith('gs://') else bucket
        cls.execute_cmd(['gsutil', 'iam', 'ch', f'serviceAccount:{account_email}:admin', bucket_name])

    @classmethod
    def delete_secret(cls, name: str, silent: bool=False):
        if False:
            return 10
        cmd = ['gcloud', 'secrets', 'delete', name, '--project', GoogleSystemTest._project_id(), '--quiet']
        cls.execute_with_ctx(cmd, key=GCP_SECRET_MANAGER_KEY, silent=silent)

    @classmethod
    def create_secret(cls, name: str, value: str):
        if False:
            while True:
                i = 10
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(value.encode('UTF-8'))
            tmp.flush()
            cmd = ['gcloud', 'secrets', 'create', name, '--replication-policy', 'automatic', '--project', GoogleSystemTest._project_id(), '--data-file', tmp.name]
            cls.execute_with_ctx(cmd, key=GCP_SECRET_MANAGER_KEY)

    @classmethod
    def update_secret(cls, name: str, value: str):
        if False:
            return 10
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(value.encode('UTF-8'))
            tmp.flush()
            cmd = ['gcloud', 'secrets', 'versions', 'add', name, '--project', GoogleSystemTest._project_id(), '--data-file', tmp.name]
            cls.execute_with_ctx(cmd, key=GCP_SECRET_MANAGER_KEY)