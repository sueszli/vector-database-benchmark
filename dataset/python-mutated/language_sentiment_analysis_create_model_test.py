import os
from google.api_core.retry import Retry
import language_sentiment_analysis_create_model
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
DATASET_ID = 'TST00000000000000000'

@Retry()
def test_sentiment_analysis_create_model(capsys):
    if False:
        while True:
            i = 10
    try:
        language_sentiment_analysis_create_model.create_model(PROJECT_ID, DATASET_ID, 'lang_sent_test_create_model')
        (out, _) = capsys.readouterr()
        assert 'Dataset does not exist.' in out
    except Exception as e:
        assert 'Dataset does not exist.' in e.message