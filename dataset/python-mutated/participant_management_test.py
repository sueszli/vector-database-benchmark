import os
from unittest import mock
from google.api_core.operation import Operation
from google.cloud import dialogflow_v2beta1 as dialogflow
import pytest
import conversation_management
import conversation_profile_management
import document_management
import knowledge_base_management
import participant_management
import test_utils
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
CONTENT_URI = 'gs://cloud-samples-data/dialogflow/participant_test.html'
CONVERSATION_PROFILE_DISPLAY_NAME = 'fake_conversation_profile'
DOCUMENT_DISPLAY_NAME = 'Cancel an order'
MIME_TYPE = 'text/html'
KNOWLEDGE_BASE_DISPLAY_NAME = 'fake_KNOWLEDGE_BASE_DISPLAY_NAME'
KNOWLEDGE_BASE_ID = 'documents/123'
KNOWLEDGE_TYPE = 'ARTICLE_SUGGESTION'

@pytest.fixture(scope='function')
def mock_create_document_operation():
    if False:
        i = 10
        return i + 15
    return test_utils.create_mock_create_document_operation(DOCUMENT_DISPLAY_NAME, KNOWLEDGE_BASE_ID, MIME_TYPE, [getattr(dialogflow.Document.KnowledgeType, KNOWLEDGE_TYPE)], CONTENT_URI)

@pytest.fixture(scope='function')
def mock_document():
    if False:
        i = 10
        return i + 15
    return test_utils.create_mock_document(DOCUMENT_DISPLAY_NAME, KNOWLEDGE_BASE_ID, MIME_TYPE, [getattr(dialogflow.Document.KnowledgeType, KNOWLEDGE_TYPE)], CONTENT_URI)

def test_analyze_content_text(capsys, mock_create_document_operation, mock_document):
    if False:
        i = 10
        return i + 15
    'Test analyze content api with text only messages.'
    knowledge_base_management.create_knowledge_base(PROJECT_ID, KNOWLEDGE_BASE_DISPLAY_NAME)
    (out, _) = capsys.readouterr()
    knowledge_base_id = out.split('knowledgeBases/')[1].rstrip()
    knowledge_base_management.get_knowledge_base(PROJECT_ID, knowledge_base_id)
    (out, _) = capsys.readouterr()
    assert f'Display Name: {KNOWLEDGE_BASE_DISPLAY_NAME}' in out
    with mock.patch('google.cloud.dialogflow_v2beta1.DocumentsClient.create_document', mock_create_document_operation):
        document_management.create_document(PROJECT_ID, knowledge_base_id, DOCUMENT_DISPLAY_NAME, MIME_TYPE, KNOWLEDGE_TYPE, CONTENT_URI)
        (out, _) = capsys.readouterr()
        document_id = out.split('documents/')[1].split(' - MIME Type:')[0].rstrip()
        assert document_id == '123'
    with mock.patch('google.cloud.dialogflow_v2beta1.DocumentsClient.get_document', mock_document):
        document_management.get_document(PROJECT_ID, knowledge_base_id, document_id)
        (out, _) = capsys.readouterr()
        assert f'Display Name: {DOCUMENT_DISPLAY_NAME}' in out
    conversation_profile_management.create_conversation_profile_article_faq(project_id=PROJECT_ID, display_name=CONVERSATION_PROFILE_DISPLAY_NAME, article_suggestion_knowledge_base_id=knowledge_base_id)
    (out, _) = capsys.readouterr()
    assert 'Display Name: {}'.format(CONVERSATION_PROFILE_DISPLAY_NAME) in out
    conversation_profile_id = out.split('conversationProfiles/')[1].rstrip()
    conversation_management.create_conversation(project_id=PROJECT_ID, conversation_profile_id=conversation_profile_id)
    (out, _) = capsys.readouterr()
    conversation_id = out.split('conversations/')[1].rstrip()
    participant_management.create_participant(project_id=PROJECT_ID, conversation_id=conversation_id, role='END_USER')
    (out, _) = capsys.readouterr()
    end_user_id = out.split('participants/')[1].rstrip()
    participant_management.create_participant(project_id=PROJECT_ID, conversation_id=conversation_id, role='HUMAN_AGENT')
    (out, _) = capsys.readouterr()
    human_agent_id = out.split('participants/')[1].rstrip()
    with mock.patch('google.cloud.dialogflow_v2beta1.ParticipantsClient.analyze_content', mock.MagicMock(spec=dialogflow.AnalyzeContentResponse)):
        participant_management.analyze_content_text(project_id=PROJECT_ID, conversation_id=conversation_id, participant_id=human_agent_id, text='Hi, how are you?')
        (out, _) = capsys.readouterr()
        participant_management.analyze_content_text(project_id=PROJECT_ID, conversation_id=conversation_id, participant_id=end_user_id, text='Hi, I am doing well, how about you?')
        (out, _) = capsys.readouterr()
        participant_management.analyze_content_text(project_id=PROJECT_ID, conversation_id=conversation_id, participant_id=human_agent_id, text='Great. How can I help you?')
        (out, _) = capsys.readouterr()
        participant_management.analyze_content_text(project_id=PROJECT_ID, conversation_id=conversation_id, participant_id=end_user_id, text='So I ordered something, but I do not like it.')
        (out, _) = capsys.readouterr()
        participant_management.analyze_content_text(project_id=PROJECT_ID, conversation_id=conversation_id, participant_id=end_user_id, text='Thinking if I can cancel that order')
        (suggestion_out, _) = capsys.readouterr()
    conversation_management.complete_conversation(project_id=PROJECT_ID, conversation_id=conversation_id)
    conversation_profile_management.delete_conversation_profile(project_id=PROJECT_ID, conversation_profile_id=conversation_profile_id)
    with mock.patch('google.cloud.dialogflow_v2beta1.DocumentsClient.delete_document', mock.MagicMock(spec=Operation)):
        document_management.delete_document(PROJECT_ID, knowledge_base_id, document_id)
    knowledge_base_management.delete_knowledge_base(PROJECT_ID, knowledge_base_id)