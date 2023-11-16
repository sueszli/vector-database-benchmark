import os
from google.api_core.retry import Retry
from google.cloud import automl
import pytest
import language_sentiment_analysis_predict
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
MODEL_ID = os.environ['SENTIMENT_ANALYSIS_MODEL_ID']

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
def test_sentiment_analysis_predict(capsys):
    if False:
        while True:
            i = 10
    text = 'Hopefully this Claritin kicks in soon'
    language_sentiment_analysis_predict.predict(PROJECT_ID, MODEL_ID, text)
    (out, _) = capsys.readouterr()
    assert 'Predicted sentiment score: ' in out