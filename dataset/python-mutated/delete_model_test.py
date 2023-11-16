import os
from google.api_core.retry import Retry
import delete_model
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']

@Retry()
def test_delete_model(capsys):
    if False:
        while True:
            i = 10
    try:
        delete_model.delete_model(PROJECT_ID, 'TRL0000000000000000000')
        (out, _) = capsys.readouterr()
        assert 'The model does not exist' in out
    except Exception as e:
        assert 'The model does not exist' in e.message