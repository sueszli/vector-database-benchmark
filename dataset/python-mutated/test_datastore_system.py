from __future__ import annotations
import os
import pytest
from tests.providers.google.cloud.utils.gcp_authenticator import GCP_DATASTORE_KEY
from tests.test_utils.gcp_system_helpers import CLOUD_DAG_FOLDER, GoogleSystemTest, provide_gcp_context
BUCKET = os.environ.get('GCP_DATASTORE_BUCKET', 'datastore-system-test')

@pytest.mark.backend('mysql', 'postgres')
@pytest.mark.credential_file(GCP_DATASTORE_KEY)
class TestGcpDatastoreSystem(GoogleSystemTest):

    @provide_gcp_context(GCP_DATASTORE_KEY)
    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.create_gcs_bucket(BUCKET, location='europe-central2')

    @provide_gcp_context(GCP_DATASTORE_KEY)
    def teardown_method(self):
        if False:
            return 10
        self.delete_gcs_bucket(BUCKET)

    @provide_gcp_context(GCP_DATASTORE_KEY)
    def test_run_example_dag(self):
        if False:
            while True:
                i = 10
        self.run_dag('example_gcp_datastore', CLOUD_DAG_FOLDER)

    @provide_gcp_context(GCP_DATASTORE_KEY)
    def test_run_example_dag_operations(self):
        if False:
            return 10
        self.run_dag('example_gcp_datastore_operations', CLOUD_DAG_FOLDER)