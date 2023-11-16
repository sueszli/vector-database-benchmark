import uuid
import google.auth
from google.cloud import contact_center_insights_v1, pubsub_v1
from google.protobuf import field_mask_pb2
import pytest
import enable_pubsub_notifications
UUID = uuid.uuid4().hex[:8]
CONVERSATION_TOPIC_ID = 'create-conversation-' + UUID
ANALYSIS_TOPIC_ID = 'create-analysis-' + UUID

@pytest.fixture
def project_id():
    if False:
        return 10
    (_, project_id) = google.auth.default()
    return project_id

@pytest.fixture
def pubsub_topics(project_id):
    if False:
        return 10
    pubsub_client = pubsub_v1.PublisherClient()
    conversation_topic_path = pubsub_client.topic_path(project_id, CONVERSATION_TOPIC_ID)
    conversation_topic = pubsub_client.create_topic(request={'name': conversation_topic_path})
    analysis_topic_path = pubsub_client.topic_path(project_id, ANALYSIS_TOPIC_ID)
    analysis_topic = pubsub_client.create_topic(request={'name': analysis_topic_path})
    yield (conversation_topic.name, analysis_topic.name)
    pubsub_client.delete_topic(request={'topic': conversation_topic.name})
    pubsub_client.delete_topic(request={'topic': analysis_topic.name})

@pytest.fixture
def disable_pubsub_notifications(project_id):
    if False:
        print('Hello World!')
    yield
    settings = contact_center_insights_v1.Settings()
    settings.name = contact_center_insights_v1.ContactCenterInsightsClient.settings_path(project_id, 'us-central1')
    settings.pubsub_notification_settings = {}
    update_mask = field_mask_pb2.FieldMask()
    update_mask.paths.append('pubsub_notification_settings')
    insights_client = contact_center_insights_v1.ContactCenterInsightsClient()
    insights_client.update_settings(settings=settings, update_mask=update_mask)

def test_enable_pubsub_notifications(capsys, project_id, pubsub_topics, disable_pubsub_notifications):
    if False:
        i = 10
        return i + 15
    (conversation_topic, analysis_topic) = pubsub_topics
    enable_pubsub_notifications.enable_pubsub_notifications(project_id, conversation_topic, analysis_topic)
    (out, err) = capsys.readouterr()
    assert 'Enabled Pub/Sub notifications' in out