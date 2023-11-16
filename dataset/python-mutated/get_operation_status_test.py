import os
from google.api_core.retry import Retry
from google.cloud import automl
import pytest
import get_operation_status
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']

@Retry()
@pytest.fixture(scope='function')
def operation_id():
    if False:
        i = 10
        return i + 15
    client = automl.AutoMlClient()
    project_location = f'projects/{PROJECT_ID}/locations/us-central1'
    generator = client._transport.operations_client.list_operations(project_location, filter_='').pages
    page = next(generator)
    operation = next(page)
    yield operation.name

@Retry()
def test_get_operation_status(capsys, operation_id):
    if False:
        for i in range(10):
            print('nop')
    get_operation_status.get_operation_status(operation_id)
    (out, _) = capsys.readouterr()
    assert 'Operation details' in out