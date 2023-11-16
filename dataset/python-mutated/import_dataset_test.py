import os
from google.api_core.retry import Retry
import import_dataset
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
BUCKET_ID = f'{PROJECT_ID}-lcm'
DATASET_ID = 'TEN0000000000000000000'

@Retry()
def test_import_dataset(capsys):
    if False:
        while True:
            i = 10
    try:
        data = f'gs://{BUCKET_ID}/sentiment-analysis/dataset.csv'
        import_dataset.import_dataset(PROJECT_ID, DATASET_ID, data)
        (out, _) = capsys.readouterr()
        assert "The Dataset doesn't exist or is inaccessible for use with AutoMl." in out
    except Exception as e:
        assert "The Dataset doesn't exist or is inaccessible for use with AutoMl." in e.message