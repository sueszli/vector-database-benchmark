from __future__ import annotations
import pytest
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
from airflow.providers.google.marketing_platform.example_dags.example_display_video import BUCKET
from tests.providers.google.cloud.utils.gcp_authenticator import GCP_BIGQUERY_KEY, GMP_KEY
from tests.test_utils.gcp_system_helpers import MARKETING_DAG_FOLDER, GoogleSystemTest, provide_gcp_context
SCOPES = ['https://www.googleapis.com/auth/doubleclickbidmanager', 'https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/display-video']

@pytest.mark.system('google.marketing_platform')
@pytest.mark.credential_file(GMP_KEY)
class TestDisplayVideoSystem(GoogleSystemTest):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.create_gcs_bucket(BUCKET)

    def teardown_method(self):
        if False:
            print('Hello World!')
        self.delete_gcs_bucket(BUCKET)
        with provide_gcp_context(GCP_BIGQUERY_KEY, scopes=SCOPES):
            hook = BigQueryHook()
            hook.delete_dataset(dataset_id='airflow_test', delete_contents=True)

    @provide_gcp_context(GMP_KEY, scopes=SCOPES)
    def test_run_example_dag(self):
        if False:
            return 10
        self.run_dag('example_display_video', MARKETING_DAG_FOLDER)

    @provide_gcp_context(GMP_KEY, scopes=SCOPES)
    def test_run_example_dag_misc(self):
        if False:
            return 10
        self.run_dag('example_display_video_misc', MARKETING_DAG_FOLDER)

    @provide_gcp_context(GMP_KEY, scopes=SCOPES)
    def test_run_example_dag_sdf(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_dag('example_display_video_sdf', MARKETING_DAG_FOLDER)