import os
from google.api_core.retry import Retry
import pytest
import vision_classification_deploy_model_node_count
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
MODEL_ID = 'ICN0000000000000000000'

@Retry()
@pytest.mark.slow
def test_classification_deploy_model_with_node_count(capsys):
    if False:
        while True:
            i = 10
    try:
        vision_classification_deploy_model_node_count.deploy_model(PROJECT_ID, MODEL_ID)
        (out, _) = capsys.readouterr()
        assert 'The model does not exist' in out
    except Exception as e:
        assert 'The model does not exist' in e.message