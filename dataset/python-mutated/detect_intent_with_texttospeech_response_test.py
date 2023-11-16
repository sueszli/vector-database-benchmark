from __future__ import absolute_import
import os
import uuid
from detect_intent_with_texttospeech_response import detect_intent_with_texttospeech_response
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
SESSION_ID = 'test_{}'.format(uuid.uuid4())
TEXTS = ['hello']

def test_detect_intent_with_sentiment_analysis(capsys):
    if False:
        for i in range(10):
            print('nop')
    detect_intent_with_texttospeech_response(PROJECT_ID, SESSION_ID, TEXTS, 'en-US')
    (out, _) = capsys.readouterr()
    assert 'Audio content written to file' in out
    statinfo = os.stat('output.wav')
    assert statinfo.st_size > 0
    os.remove('output.wav')
    assert not os.path.isfile('output.wav')