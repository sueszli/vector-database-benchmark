from __future__ import absolute_import
import os
import uuid
import detect_intent_knowledge
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
SESSION_ID = 'session_{}'.format(uuid.uuid4())
KNOWLEDGE_BASE_ID = 'MjEwMjE4MDQ3MDQwMDc0NTQ3Mg'
TEXTS = ['Where is my data stored?']

def test_detect_intent_knowledge(capsys):
    if False:
        return 10
    detect_intent_knowledge.detect_intent_knowledge(PROJECT_ID, SESSION_ID, 'en-us', KNOWLEDGE_BASE_ID, TEXTS)
    (out, _) = capsys.readouterr()
    assert 'Knowledge results' in out