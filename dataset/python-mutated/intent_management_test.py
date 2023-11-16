from __future__ import absolute_import
import os
import uuid
import intent_management
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
INTENT_DISPLAY_NAME = 'test_{}'.format(uuid.uuid4())
MESSAGE_TEXTS = ['fake_message_text_for_testing_1', 'fake_message_text_for_testing_2']
TRAINING_PHRASE_PARTS = ['fake_training_phrase_part_1', 'fake_training_phease_part_2']

def test_create_intent(capsys):
    if False:
        i = 10
        return i + 15
    intent_management.create_intent(PROJECT_ID, INTENT_DISPLAY_NAME, TRAINING_PHRASE_PARTS, MESSAGE_TEXTS)
    intent_ids = intent_management._get_intent_ids(PROJECT_ID, INTENT_DISPLAY_NAME)
    assert len(intent_ids) == 1
    intent_management.list_intents(PROJECT_ID)
    (out, _) = capsys.readouterr()
    assert INTENT_DISPLAY_NAME in out
    for message_text in MESSAGE_TEXTS:
        assert message_text in out

def test_delete_session_entity_type(capsys):
    if False:
        while True:
            i = 10
    intent_ids = intent_management._get_intent_ids(PROJECT_ID, INTENT_DISPLAY_NAME)
    for intent_id in intent_ids:
        intent_management.delete_intent(PROJECT_ID, intent_id)
    intent_management.list_intents(PROJECT_ID)
    (out, _) = capsys.readouterr()
    assert INTENT_DISPLAY_NAME not in out
    intent_ids = intent_management._get_intent_ids(PROJECT_ID, INTENT_DISPLAY_NAME)
    assert len(intent_ids) == 0