import google.auth
from google.cloud import contact_center_insights_v1
import pytest
import get_operation
TRANSCRIPT_URI = 'gs://cloud-samples-data/ccai/chat_sample.json'
AUDIO_URI = 'gs://cloud-samples-data/ccai/voice_6912.txt'

@pytest.fixture
def project_id():
    if False:
        return 10
    (_, project_id) = google.auth.default()
    return project_id

@pytest.fixture
def insights_client():
    if False:
        while True:
            i = 10
    return contact_center_insights_v1.ContactCenterInsightsClient()

@pytest.fixture
def conversation_resource(project_id, insights_client):
    if False:
        return 10
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
def analysis_operation(conversation_resource, insights_client):
    if False:
        while True:
            i = 10
    conversation_name = conversation_resource.name
    analysis = contact_center_insights_v1.Analysis()
    analysis_operation = insights_client.create_analysis(parent=conversation_name, analysis=analysis)
    analysis_operation.result(timeout=600)
    yield analysis_operation

def test_get_operation(capsys, analysis_operation):
    if False:
        return 10
    operation_name = analysis_operation.operation.name
    get_operation.get_operation(operation_name)
    (out, err) = capsys.readouterr()
    assert 'Operation is done' in out