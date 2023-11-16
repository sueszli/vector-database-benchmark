import os
import uuid
from google.api_core.retry import Retry
from google.cloud import automl_v1beta1 as automl
import pytest
import video_object_tracking_create_dataset
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
DATASET_ID = None

@pytest.fixture(scope='function', autouse=True)
def teardown():
    if False:
        while True:
            i = 10
    yield
    client = automl.AutoMlClient()
    dataset_full_id = client.dataset_path(PROJECT_ID, 'us-central1', DATASET_ID)
    response = client.delete_dataset(name=dataset_full_id)
    response.result()

@Retry()
def test_video_classification_create_dataset(capsys):
    if False:
        i = 10
        return i + 15
    dataset_name = f'test_{uuid.uuid4()}'.replace('-', '')[:32]
    video_object_tracking_create_dataset.create_dataset(PROJECT_ID, dataset_name)
    (out, _) = capsys.readouterr()
    assert 'Dataset id: ' in out
    global DATASET_ID
    DATASET_ID = out.splitlines()[1].split()[2]