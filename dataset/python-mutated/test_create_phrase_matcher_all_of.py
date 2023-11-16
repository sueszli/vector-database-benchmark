import google.auth
from google.cloud import contact_center_insights_v1
import pytest
import create_phrase_matcher_all_of

@pytest.fixture
def project_id():
    if False:
        for i in range(10):
            print('nop')
    (_, project_id) = google.auth.default()
    return project_id

@pytest.fixture
def insights_client():
    if False:
        while True:
            i = 10
    return contact_center_insights_v1.ContactCenterInsightsClient()

@pytest.fixture
def phrase_matcher_all_of(project_id, insights_client):
    if False:
        return 10
    phrase_matcher = create_phrase_matcher_all_of.create_phrase_matcher_all_of(project_id)
    yield phrase_matcher
    insights_client.delete_phrase_matcher(name=phrase_matcher.name)

def test_create_phrase_matcher_all_of(capsys, phrase_matcher_all_of):
    if False:
        return 10
    phrase_matcher = phrase_matcher_all_of
    (out, err) = capsys.readouterr()
    assert f'Created {phrase_matcher.name}' in out