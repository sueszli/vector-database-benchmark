import google.auth
from google.cloud import contact_center_insights_v1
from google.protobuf import field_mask_pb2
import pytest
import set_project_ttl

@pytest.fixture
def project_id():
    if False:
        return 10
    (_, project_id) = google.auth.default()
    return project_id

@pytest.fixture
def clear_project_ttl(project_id):
    if False:
        for i in range(10):
            print('nop')
    yield
    settings = contact_center_insights_v1.Settings()
    settings.name = contact_center_insights_v1.ContactCenterInsightsClient.settings_path(project_id, 'us-central1')
    settings.conversation_ttl = None
    update_mask = field_mask_pb2.FieldMask()
    update_mask.paths.append('conversation_ttl')
    insights_client = contact_center_insights_v1.ContactCenterInsightsClient()
    insights_client.update_settings(settings=settings, update_mask=update_mask)

def test_set_project_ttl(capsys, project_id, clear_project_ttl):
    if False:
        while True:
            i = 10
    set_project_ttl.set_project_ttl(project_id)
    (out, err) = capsys.readouterr()
    assert 'Set TTL for all incoming conversations to 1 day' in out