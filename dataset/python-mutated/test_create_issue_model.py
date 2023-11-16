import google.auth
from google.cloud import contact_center_insights_v1
import pytest
import create_issue_model
MIN_CONVERSATION_COUNT = 10000

@pytest.fixture
def project_id():
    if False:
        while True:
            i = 10
    (_, project_id) = google.auth.default()
    return project_id

@pytest.fixture
def insights_client():
    if False:
        for i in range(10):
            print('nop')
    return contact_center_insights_v1.ContactCenterInsightsClient()

@pytest.fixture
def count_conversations(project_id, insights_client):
    if False:
        return 10
    list_request = contact_center_insights_v1.ListConversationsRequest()
    list_request.page_size = 1000
    list_request.parent = contact_center_insights_v1.ContactCenterInsightsClient.common_location_path(project_id, 'us-central1')
    conversations = insights_client.list_conversations(request=list_request)
    conversation_count = len(list(conversations))
    yield conversation_count

@pytest.fixture
def issue_model_resource(project_id, insights_client, count_conversations):
    if False:
        i = 10
        return i + 15
    conversation_count = count_conversations
    if conversation_count >= MIN_CONVERSATION_COUNT:
        issue_model = create_issue_model.create_issue_model(project_id)
        yield issue_model
        insights_client.delete_issue_model(name=issue_model.name)
    else:
        yield None

def test_create_issue_model(capsys, issue_model_resource):
    if False:
        return 10
    issue_model = issue_model_resource
    if issue_model:
        (out, err) = capsys.readouterr()
        assert f'Created {issue_model.name}' in out