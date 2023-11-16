from __future__ import annotations
import json
import os
import subprocess
from airflow import settings
from airflow.exceptions import AirflowException
from airflow.models import Connection
from tests.test_utils import AIRFLOW_MAIN_FOLDER
from tests.test_utils.logging_command_executor import CommandExecutor
GCP_AI_KEY = 'gcp_ai.json'
GCP_BIGQUERY_KEY = 'gcp_bigquery.json'
GCP_BIGTABLE_KEY = 'gcp_bigtable.json'
GCP_CLOUD_BUILD_KEY = 'gcp_cloud_build.json'
GCP_CLOUD_COMPOSER = 'gcp_cloud_composer.json'
GCP_CLOUDSQL_KEY = 'gcp_cloudsql.json'
GCP_COMPUTE_KEY = 'gcp_compute.json'
GCP_COMPUTE_SSH_KEY = 'gcp_compute_ssh.json'
GCP_DATACATALOG_KEY = 'gcp_datacatalog.json'
GCP_DATAFLOW_KEY = 'gcp_dataflow.json'
GCP_DATAFUSION_KEY = 'gcp_datafusion.json'
GCP_DATAPROC_KEY = 'gcp_dataproc.json'
GCP_DATASTORE_KEY = 'gcp_datastore.json'
GCP_DLP_KEY = 'gcp_dlp.json'
GCP_FUNCTION_KEY = 'gcp_function.json'
GCP_GCS_KEY = 'gcp_gcs.json'
GCP_GCS_TRANSFER_KEY = 'gcp_gcs_transfer.json'
GCP_GKE_KEY = 'gcp_gke.json'
GCP_KMS_KEY = 'gcp_kms.json'
GCP_MEMORYSTORE = 'gcp_memorystore.json'
GCP_PUBSUB_KEY = 'gcp_pubsub.json'
GCP_SECRET_MANAGER_KEY = 'gcp_secret_manager.json'
GCP_SPANNER_KEY = 'gcp_spanner.json'
GCP_STACKDRIVER = 'gcp_stackdriver.json'
GCP_TASKS_KEY = 'gcp_tasks.json'
GCP_VERTEX_AI_KEY = 'gcp_vertex_ai.json'
GCP_WORKFLOWS_KEY = 'gcp_workflows.json'
GMP_KEY = 'gmp.json'
G_FIREBASE_KEY = 'g_firebase.json'
GCP_AWS_KEY = 'gcp_aws.json'
KEYPATH_EXTRA = 'extra__google_cloud_platform__key_path'
KEYFILE_DICT_EXTRA = 'extra__google_cloud_platform__keyfile_dict'
SCOPE_EXTRA = 'extra__google_cloud_platform__scope'
PROJECT_EXTRA = 'extra__google_cloud_platform__project'

class GcpAuthenticator(CommandExecutor):
    """
    Initialises the authenticator.

    :param gcp_key: name of the key to use for authentication (see GCP_*_KEY values)
    :param project_extra: optional extra project parameter passed to google cloud
           connection
    """
    original_account: str | None = None

    def __init__(self, gcp_key: str, project_extra: str | None=None):
        if False:
            print('Hello World!')
        super().__init__()
        self.gcp_key = gcp_key
        self.project_extra = project_extra
        self.project_id = self.get_project_id()
        self.full_key_path = None
        self._set_key_path()

    @staticmethod
    def get_project_id():
        if False:
            for i in range(10):
                print('nop')
        return os.environ.get('GCP_PROJECT_ID')

    def set_key_path_in_airflow_connection(self):
        if False:
            while True:
                i = 10
        "\n        Set key path in 'google_cloud_default' connection to point to the full\n        key path\n        :return: None\n        "
        with settings.Session() as session:
            conn = session.query(Connection).filter(Connection.conn_id == 'google_cloud_default')[0]
            extras = conn.extra_dejson
            extras[KEYPATH_EXTRA] = self.full_key_path
            if extras.get(KEYFILE_DICT_EXTRA):
                del extras[KEYFILE_DICT_EXTRA]
            extras[SCOPE_EXTRA] = 'https://www.googleapis.com/auth/cloud-platform'
            extras[PROJECT_EXTRA] = self.project_extra if self.project_extra else self.project_id
            conn.extra = json.dumps(extras)

    def set_dictionary_in_airflow_connection(self):
        if False:
            while True:
                i = 10
        "\n        Set dictionary in 'google_cloud_default' connection to contain content\n        of the json service account file.\n        :return: None\n        "
        with settings.Session() as session:
            conn = session.query(Connection).filter(Connection.conn_id == 'google_cloud_default')[0]
            extras = conn.extra_dejson
            with open(self.full_key_path) as path_file:
                content = json.load(path_file)
            extras[KEYFILE_DICT_EXTRA] = json.dumps(content)
            if extras.get(KEYPATH_EXTRA):
                del extras[KEYPATH_EXTRA]
            extras[SCOPE_EXTRA] = 'https://www.googleapis.com/auth/cloud-platform'
            extras[PROJECT_EXTRA] = self.project_extra
            conn.extra = json.dumps(extras)

    def _set_key_path(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets full key path - if GCP_CONFIG_DIR points to absolute\n            directory, it tries to find the key in this directory. Otherwise it assumes\n            that Airflow is running from the directory where configuration is checked\n            out next to airflow directory in config directory\n            it tries to find the key folder in the workspace's config\n            directory.\n        :param : name of the key file to find.\n        "
        if 'GCP_CONFIG_DIR' in os.environ:
            gcp_config_dir = os.environ['GCP_CONFIG_DIR']
        else:
            gcp_config_dir = os.path.join(AIRFLOW_MAIN_FOLDER, os.pardir, 'config')
        if not os.path.isdir(gcp_config_dir):
            self.log.info('The %s is not a directory', gcp_config_dir)
        key_dir = os.path.join(gcp_config_dir, 'keys')
        if not os.path.isdir(key_dir):
            self.log.error('The %s is not a directory', key_dir)
            return
        key_path = os.path.join(key_dir, self.gcp_key)
        if not os.path.isfile(key_path):
            self.log.error('The %s file is missing', key_path)
        self.full_key_path = key_path

    def _validate_key_set(self):
        if False:
            for i in range(10):
                print('nop')
        if self.full_key_path is None:
            raise AirflowException('The gcp_key is not set!')
        if not os.path.isfile(self.full_key_path):
            raise AirflowException(f'The key {self.gcp_key} could not be found. Please copy it to the {self.full_key_path} path.')

    def gcp_authenticate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Authenticate with service account specified via key name.\n        '
        self._validate_key_set()
        self.log.info('Setting the Google Cloud key to %s', self.full_key_path)
        self.execute_cmd(['gcloud', 'auth', 'activate-service-account', f'--key-file={self.full_key_path}', f'--project={self.project_id}'])
        self.set_key_path_in_airflow_connection()

    def gcp_revoke_authentication(self):
        if False:
            i = 10
            return i + 15
        '\n        Change default authentication to none - which is not existing one.\n        '
        self._validate_key_set()
        self.log.info('Revoking authentication - setting it to none')
        self.execute_cmd(['gcloud', 'config', 'get-value', 'account', f'--project={self.project_id}'])
        self.execute_cmd(['gcloud', 'config', 'set', 'account', 'none', f'--project={self.project_id}'])

    def gcp_store_authentication(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Store authentication as it was originally so it can be restored and revoke\n        authentication.\n        '
        self._validate_key_set()
        if not GcpAuthenticator.original_account:
            GcpAuthenticator.original_account = self.check_output(['gcloud', 'config', 'get-value', 'account', f'--project={self.project_id}']).decode('utf-8')
            self.log.info('Storing account: to restore it later %s', GcpAuthenticator.original_account)

    def gcp_restore_authentication(self):
        if False:
            i = 10
            return i + 15
        '\n        Restore authentication to the original one.\n        '
        self._validate_key_set()
        if GcpAuthenticator.original_account:
            self.log.info('Restoring original account stored: %s', GcpAuthenticator.original_account)
            subprocess.call(['gcloud', 'config', 'set', 'account', GcpAuthenticator.original_account, f'--project={self.project_id}'])
        else:
            self.log.info('Not restoring the original Google Cloud account: it is not set')