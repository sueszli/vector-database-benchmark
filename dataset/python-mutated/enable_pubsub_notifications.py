from google.api_core import protobuf_helpers
from google.cloud import contact_center_insights_v1

def enable_pubsub_notifications(project_id: str, topic_create_conversation: str, topic_create_analysis: str) -> None:
    if False:
        print('Hello World!')
    "Enables Cloud Pub/Sub notifications for specified events.\n\n    Args:\n        project_id:\n            The project identifier. For example, 'my-project'.\n        topic_create_conversation:\n            The Cloud Pub/Sub topic to notify of conversation creation events.\n            Format is 'projects/{project_id}/topics/{topic_id}'.\n            For example, 'projects/my-project/topics/my-topic'.\n        topic_create_analysis:\n            The Cloud Pub/Sub topic to notify of analysis creation events.\n            Format is 'projects/{project_id}/topics/{topic_id}'.\n            For example, 'projects/my-project/topics/my-topic'.\n\n    Returns:\n        None.\n    "
    settings = contact_center_insights_v1.Settings()
    settings.name = contact_center_insights_v1.ContactCenterInsightsClient.settings_path(project_id, 'us-central1')
    settings.pubsub_notification_settings = {'create-conversation': topic_create_conversation, 'create-analysis': topic_create_analysis}
    update_mask = protobuf_helpers.field_mask(None, type(settings).pb(settings))
    insights_client = contact_center_insights_v1.ContactCenterInsightsClient()
    insights_client.update_settings(settings=settings, update_mask=update_mask)
    print('Enabled Pub/Sub notifications')