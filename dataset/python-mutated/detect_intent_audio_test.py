"""Tests for detect_intent_texts."""
from __future__ import absolute_import
import os
import uuid
from detect_intent_audio import detect_intent_audio
DIRNAME = os.path.realpath(os.path.dirname(__file__))
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
AGENT_ID = os.getenv('AGENT_ID')
AGENT_ID_US_CENTRAL1 = os.getenv('AGENT_ID_US_CENTRAL1')
AGENT = f'projects/{PROJECT_ID}/locations/global/agents/{AGENT_ID}'
AGENT_US_CENTRAL1 = f'projects/{PROJECT_ID}/locations/us-central1/agents/{AGENT_ID_US_CENTRAL1}'
SESSION_ID = uuid.uuid4()
AUDIO_PATH = os.getenv('AUDIO_PATH')
AUDIO = f'{DIRNAME}/{AUDIO_PATH}'

def test_detect_intent_texts(capsys):
    if False:
        while True:
            i = 10
    detect_intent_audio(AGENT, SESSION_ID, AUDIO, 'en-US')
    (out, _) = capsys.readouterr()
    assert "Response text: Hi! I'm the virtual flights agent." in out

def test_detect_intent_texts_regional(capsys):
    if False:
        for i in range(10):
            print('nop')
    detect_intent_audio(AGENT_US_CENTRAL1, SESSION_ID, AUDIO, 'en-US')
    (out, _) = capsys.readouterr()
    assert "Response text: Hi! I'm the virtual flights agent." in out