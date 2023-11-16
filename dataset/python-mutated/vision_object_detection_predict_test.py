import os
import backoff
from google.api_core.exceptions import DeadlineExceeded
from google.cloud import automl
import pytest
import vision_object_detection_predict
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
MODEL_ID = os.environ['OBJECT_DETECTION_MODEL_ID']

@pytest.fixture(scope='function')
def verify_model_state():
    if False:
        while True:
            i = 10
    client = automl.AutoMlClient()
    model_full_id = client.model_path(PROJECT_ID, 'us-central1', MODEL_ID)
    model = client.get_model(name=model_full_id)
    if model.deployment_state == automl.Model.DeploymentState.UNDEPLOYED:
        response = client.deploy_model(name=model_full_id)
        response.result(600)

def test_vision_object_detection_predict(capsys, verify_model_state):
    if False:
        while True:
            i = 10
    file_path = 'resources/salad.jpg'

    @backoff.on_exception(backoff.expo, DeadlineExceeded, max_time=300)
    def run_sample():
        if False:
            while True:
                i = 10
        vision_object_detection_predict.predict(PROJECT_ID, MODEL_ID, file_path)
    run_sample()
    (out, _) = capsys.readouterr()
    assert 'Predicted class name:' in out