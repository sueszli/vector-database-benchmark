import datetime
import os
from google.api_core.retry import Retry
from google.cloud import automl
import translate_create_dataset
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']

@Retry()
def test_translate_create_dataset(capsys):
    if False:
        return 10
    dataset_name = 'test_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    translate_create_dataset.create_dataset(PROJECT_ID, dataset_name)
    (out, _) = capsys.readouterr()
    assert 'Dataset id: ' in out
    dataset_id = out.splitlines()[1].split()[2]
    client = automl.AutoMlClient()
    dataset_full_id = client.dataset_path(PROJECT_ID, 'us-central1', dataset_id)
    response = client.delete_dataset(name=dataset_full_id)
    response.result()