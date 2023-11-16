import os
from google.api_core.retry import Retry
import video_object_tracking_create_model
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
DATASET_ID = 'VOT00000000000000000000'
OPERATION_ID = None

@Retry()
def test_video_classification_create_model(capsys):
    if False:
        i = 10
        return i + 15
    try:
        video_object_tracking_create_model.create_model(PROJECT_ID, DATASET_ID, 'video_object_test_create_model')
        (out, _) = capsys.readouterr()
        assert 'Dataset does not exist.' in out
    except Exception as e:
        assert 'Dataset does not exist.' in e.message