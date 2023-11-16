from __future__ import absolute_import
import os
from unittest import mock
from google.cloud import dialogflow_v2beta1 as dialogflow
import pytest
import conversation_profile_management
import test_utils
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
CONVERSATION_PROFILE_DISPLAY_NAME = 'fake_conversation_profile_name'
CONVERSATION_PROFILE_NAME = f'conversationProfiles/{CONVERSATION_PROFILE_DISPLAY_NAME}'

@pytest.fixture(scope='function')
def mock_conversation():
    if False:
        for i in range(10):
            print('nop')
    yield mock.MagicMock(return_value=test_utils.create_mock_conversation(CONVERSATION_PROFILE_DISPLAY_NAME, CONVERSATION_PROFILE_NAME))

@pytest.fixture(scope='function')
def mock_conversation_list():
    if False:
        return 10
    yield mock.MagicMock(return_value=[test_utils.create_mock_conversation(CONVERSATION_PROFILE_DISPLAY_NAME, CONVERSATION_PROFILE_NAME)])

def test_create_conversation_profile(capsys, mock_conversation, mock_conversation_list):
    if False:
        print('Hello World!')
    with mock.patch('conversation_profile_management.dialogflow.ConversationProfilesClient.list_conversation_profiles', mock.MagicMock(spec=dialogflow.ListConversationProfilesResponse)):
        response = conversation_profile_management.list_conversation_profiles(PROJECT_ID)
        assert not any((x.display_name == CONVERSATION_PROFILE_DISPLAY_NAME for x in response))
    with mock.patch('conversation_profile_management.dialogflow.ConversationProfilesClient.create_conversation_profile', mock_conversation):
        response = conversation_profile_management.create_conversation_profile_article_faq(project_id=PROJECT_ID, display_name=CONVERSATION_PROFILE_DISPLAY_NAME, article_suggestion_knowledge_base_id='abc')
        (out, _) = capsys.readouterr()
        assert response.display_name == CONVERSATION_PROFILE_DISPLAY_NAME
    conversation_profile_id = out.split('conversationProfiles/')[1].rstrip()
    with mock.patch('conversation_profile_management.dialogflow.ConversationProfilesClient.list_conversation_profiles', mock_conversation_list):
        response = conversation_profile_management.list_conversation_profiles(PROJECT_ID)
        assert any((x.display_name == CONVERSATION_PROFILE_DISPLAY_NAME for x in response))
    with mock.patch('conversation_profile_management.dialogflow.ConversationProfilesClient.get_conversation_profile', mock_conversation):
        conversation_profile_management.get_conversation_profile(PROJECT_ID, conversation_profile_id)
        (out, _) = capsys.readouterr()
        assert f'Display Name: {CONVERSATION_PROFILE_DISPLAY_NAME}' in out
    with mock.patch('conversation_profile_management.dialogflow.ConversationProfilesClient.list_conversation_profiles', mock.MagicMock(return_value=None)):
        conversation_profile_management.delete_conversation_profile(PROJECT_ID, conversation_profile_id)
    with mock.patch('conversation_profile_management.dialogflow.ConversationProfilesClient.list_conversation_profiles', mock.MagicMock(spec=dialogflow.ListConversationProfilesResponse)):
        conversation_profile_management.list_conversation_profiles(PROJECT_ID)
        (out, _) = capsys.readouterr()
        assert f'Display Name: {CONVERSATION_PROFILE_DISPLAY_NAME}' not in out