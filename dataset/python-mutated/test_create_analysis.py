import google.auth
from google.cloud import contact_center_insights_v1
import pytest
import create_analysis
TRANSCRIPT_URI = 'gs://cloud-samples-data/ccai/chat_sample.json'
AUDIO_URI = 'gs://cloud-samples-data/ccai/voice_6912.txt'

@pytest.fixture
def project_id():
    if False:
        i = 10
        return i + 15
    (_, project_id) = google.auth.default()
    return project_id

@pytest.fixture
def conversation_resource(project_id):
    if False:
        while True:
            i = 10
    insights_client = contact_center_insights_v1.ContactCenterInsightsClient()
    parent = contact_center_insights_v1.ContactCenterInsightsClient.common_location_path(project_id, 'us-central1')
    conversation = contact_center_insights_v1.Conversation()
    conversation.data_source.gcs_source.transcript_uri = TRANSCRIPT_URI
    conversation.data_source.gcs_source.audio_uri = AUDIO_URI
    conversation.medium = contact_center_insights_v1.Conversation.Medium.CHAT
    conversation = insights_client.create_conversation(parent=parent, conversation=conversation)
    yield conversation
    delete_request = contact_center_insights_v1.DeleteConversationRequest()
    delete_request.name = conversation.name
    delete_request.force = True
    insights_client.delete_conversation(request=delete_request)

@pytest.fixture
def analysis_resource(conversation_resource):
    if False:
        while True:
            i = 10
    conversation_name = conversation_resource.name
    yield create_analysis.create_analysis(conversation_name)

def test_create_analysis(capsys, analysis_resource):
    if False:
        while True:
            i = 10
    analysis = analysis_resource
    (out, err) = capsys.readouterr()
    assert f'Created {analysis.name}' in out