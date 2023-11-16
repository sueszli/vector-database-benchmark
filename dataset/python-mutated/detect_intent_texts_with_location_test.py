from __future__ import absolute_import
import os
import uuid
from detect_intent_texts_with_location import detect_intent_texts_with_location
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
LOCATION_ID = 'europe-west2'
SESSION_ID = 'test_{}'.format(uuid.uuid4())
TEXTS = ['hello', 'book a meeting room', 'Mountain View', 'tomorrow', '10 AM', '2 hours', '10 people', 'A', 'yes']

def test_detect_intent_texts_with_location(capsys):
    if False:
        return 10
    detect_intent_texts_with_location(PROJECT_ID, LOCATION_ID, SESSION_ID, TEXTS, 'en-GB')
    (out, _) = capsys.readouterr()
    assert 'Fulfillment text: All set!' in out