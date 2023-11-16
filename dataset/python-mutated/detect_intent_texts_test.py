"""Tests for detect_intent_texts."""
from __future__ import absolute_import
import os
import uuid
from detect_intent_texts import detect_intent_texts
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
AGENT_ID = os.getenv('AGENT_ID')
AGENT = f'projects/{PROJECT_ID}/locations/global/agents/{AGENT_ID}'
SESSION_ID = uuid.uuid4()
TEXTS = ['hello', 'book a flight']
AGENT_ID_US_CENTRAL1 = os.getenv('AGENT_ID_US_CENTRAL1')
AGENT_US_CENTRAL1 = f'projects/{PROJECT_ID}/locations/us-central1/agents/{AGENT_ID_US_CENTRAL1}'

def test_detect_intent_texts(capsys):
    if False:
        while True:
            i = 10
    detect_intent_texts(AGENT, SESSION_ID, TEXTS, 'en-US')
    (out, _) = capsys.readouterr()
    assert 'Response text: I can help you find a ticket' in out

def test_detect_intent_texts_regional(capsys):
    if False:
        return 10
    detect_intent_texts(AGENT_US_CENTRAL1, SESSION_ID, TEXTS, 'en-US')
    (out, _) = capsys.readouterr()
    assert 'Response text: I can help you find a ticket' in out