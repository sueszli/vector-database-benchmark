"""Tests for detect_intent_texts."""
from __future__ import absolute_import
import os
from streaming_detect_intent_partial_response import streaming_detect_intent_partial_response
DIRNAME = os.path.realpath(os.path.dirname(__file__))
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
AGENT_ID = os.getenv('AGENT_ID')
AUDIO_PATH = os.getenv('AUDIO_PATH')
AUDIO = f'{DIRNAME}/{AUDIO_PATH}'

def test_streaming_detect_intent_partial_response(capsys):
    if False:
        print('Hello World!')
    encoding = 'AUDIO_ENCODING_LINEAR_16'
    sample_rate_hertz = 24000
    streaming_detect_intent_partial_response(PROJECT_ID, 'global', AGENT_ID, AUDIO, encoding, sample_rate_hertz, 'en-US')
    (out, _) = capsys.readouterr()
    assert 'Intermediate transcript:' in out
    assert "Response text: Hi! I'm the virtual flights agent." in out