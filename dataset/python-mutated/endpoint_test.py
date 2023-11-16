import os
from google.api_core.retry import Retry
import automl_tables_set_endpoint
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

@Retry()
def test_client_creation(capsys):
    if False:
        i = 10
        return i + 15
    automl_tables_set_endpoint.create_client_with_endpoint(PROJECT)
    (out, _) = capsys.readouterr()
    assert 'ListDatasetsPager' in out