import os
from google.api_core.retry import Retry
import pytest
import deploy_model
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
MODEL_ID = 'TRL0000000000000000000'

@Retry()
@pytest.mark.slow
def test_deploy_model(capsys):
    if False:
        print('Hello World!')
    try:
        deploy_model.deploy_model(PROJECT_ID, MODEL_ID)
        (out, _) = capsys.readouterr()
        assert 'The model does not exist' in out
    except Exception as e:
        assert 'The model does not exist' in e.message