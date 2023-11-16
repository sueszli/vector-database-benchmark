import os
from google.api_core.retry import Retry
import vision_object_detection_create_model
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
DATASET_ID = 'IOD0000000000000000'

@Retry()
def test_vision_object_detection_create_model(capsys):
    if False:
        print('Hello World!')
    try:
        vision_object_detection_create_model.create_model(PROJECT_ID, DATASET_ID, 'object_test_create_model')
        (out, _) = capsys.readouterr()
        assert 'Dataset does not exist.' in out
    except Exception as e:
        assert 'Dataset does not exist.' in e.message