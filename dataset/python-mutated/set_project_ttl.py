from google.api_core import protobuf_helpers
from google.cloud import contact_center_insights_v1
from google.protobuf import duration_pb2

def set_project_ttl(project_id: str) -> None:
    if False:
        print('Hello World!')
    "Sets a project-level TTL for all incoming conversations.\n\n    Args:\n        project_id:\n            The project identifier. For example, 'my-project'.\n\n    Returns:\n        None.\n    "
    settings = contact_center_insights_v1.Settings()
    settings.name = contact_center_insights_v1.ContactCenterInsightsClient.settings_path(project_id, 'us-central1')
    conversation_ttl = duration_pb2.Duration()
    conversation_ttl.seconds = 86400
    settings.conversation_ttl = conversation_ttl
    update_mask = protobuf_helpers.field_mask(None, type(settings).pb(settings))
    insights_client = contact_center_insights_v1.ContactCenterInsightsClient()
    insights_client.update_settings(settings=settings, update_mask=update_mask)
    new_conversation_ttl = insights_client.get_settings(name=settings.name).conversation_ttl
    print('Set TTL for all incoming conversations to {} day'.format(new_conversation_ttl.days))