import os
from google.api_core.retry import Retry
import list_models
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']

@Retry()
def test_list_models(capsys):
    if False:
        while True:
            i = 10
    list_models.list_models(PROJECT_ID)
    (out, _) = capsys.readouterr()
    assert 'Model id: ' in out