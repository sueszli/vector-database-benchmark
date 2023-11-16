import os
import google.auth
import list_training_phrases
(_, PROJECT_ID) = google.auth.default()
INTENT_ID = os.getenv('INTENT_ID')
LOCATION = 'global'
AGENT_ID = os.getenv('AGENT_ID')

def test_list_training_phrases(capsys):
    if False:
        print('Hello World!')
    training_phrases = list_training_phrases.list_training_phrases(PROJECT_ID, AGENT_ID, INTENT_ID, LOCATION)
    assert len(training_phrases) >= 15