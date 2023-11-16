import os
from google.api_core.retry import Retry
import pytest
import vision_object_detection_deploy_model_node_count
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
MODEL_ID = '0000000000000000000000'

@Retry()
@pytest.mark.slow
def test_object_detection_deploy_model_with_node_count(capsys):
    if False:
        i = 10
        return i + 15
    try:
        vision_object_detection_deploy_model_node_count.deploy_model(PROJECT_ID, MODEL_ID)
        (out, _) = capsys.readouterr()
        assert 'The model does not exist' in out
    except Exception as e:
        assert 'The model does not exist' in e.message