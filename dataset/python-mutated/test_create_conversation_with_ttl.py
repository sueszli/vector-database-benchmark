import google.auth
from google.cloud import contact_center_insights_v1
import pytest
import create_conversation_with_ttl

@pytest.fixture
def project_id():
    if False:
        for i in range(10):
            print('nop')
    (_, project_id) = google.auth.default()
    return project_id

@pytest.fixture
def conversation_resource(project_id):
    if False:
        print('Hello World!')
    conversation = create_conversation_with_ttl.create_conversation_with_ttl(project_id)
    yield conversation
    insights_client = contact_center_insights_v1.ContactCenterInsightsClient()
    insights_client.delete_conversation(name=conversation.name)

def test_create_conversation_with_ttl(capsys, conversation_resource):
    if False:
        while True:
            i = 10
    conversation = conversation_resource
    (out, err) = capsys.readouterr()
    assert f'Created {conversation.name}' in out