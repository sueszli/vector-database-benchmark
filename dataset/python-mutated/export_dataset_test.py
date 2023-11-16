import datetime
import os
from google.api_core.retry import Retry
import export_dataset
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
BUCKET_ID = f'{PROJECT_ID}-lcm'
PREFIX = 'TEST_EXPORT_OUTPUT_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
DATASET_ID = 'TEN0000000000000000000'

@Retry()
def test_export_dataset(capsys):
    if False:
        while True:
            i = 10
    try:
        export_dataset.export_dataset(PROJECT_ID, DATASET_ID, f'gs://{BUCKET_ID}/{PREFIX}/')
        (out, _) = capsys.readouterr()
        assert "The Dataset doesn't exist or is inaccessible for use with AutoMl." in out
    except Exception as e:
        assert "The Dataset doesn't exist or is inaccessible for use with AutoMl." in e.message