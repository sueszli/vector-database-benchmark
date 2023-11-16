import google.auth
from google.cloud import contact_center_insights_v1
import pytest
import create_conversation

@pytest.fixture
def project_id():
    if False:
        return 10
    (_, project_id) = google.auth.default()
    return project_id

@pytest.fixture
def conversation_resource(project_id):
    if False:
        while True:
            i = 10
    conversation = create_conversation.create_conversation(project_id)
    yield conversation
    insights_client = contact_center_insights_v1.ContactCenterInsightsClient()
    insights_client.delete_conversation(name=conversation.name)

def test_create_conversation(capsys, conversation_resource):
    if False:
        i = 10
        return i + 15
    conversation = conversation_resource
    (out, err) = capsys.readouterr()
    assert f'Created {conversation.name}' in out