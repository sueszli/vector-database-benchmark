import os
from google.api_core.retry import Retry
from google.cloud import automl
import pytest
import vision_classification_predict
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
MODEL_ID = os.environ['VISION_CLASSIFICATION_MODEL_ID']

@pytest.fixture(scope='function', autouse=True)
def setup():
    if False:
        i = 10
        return i + 15
    client = automl.AutoMlClient()
    model_full_id = client.model_path(PROJECT_ID, 'us-central1', MODEL_ID)
    model = client.get_model(name=model_full_id)
    if model.deployment_state == automl.Model.DeploymentState.UNDEPLOYED:
        response = client.deploy_model(name=model_full_id)
        response.result()

@Retry()
def test_vision_classification_predict(capsys):
    if False:
        print('Hello World!')
    file_path = 'resources/test.png'
    vision_classification_predict.predict(PROJECT_ID, MODEL_ID, file_path)
    (out, _) = capsys.readouterr()
    assert 'Predicted class name:' in out