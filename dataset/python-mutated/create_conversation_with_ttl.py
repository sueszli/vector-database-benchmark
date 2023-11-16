from google.cloud import contact_center_insights_v1
from google.protobuf import duration_pb2

def create_conversation_with_ttl(project_id: str, transcript_uri: str='gs://cloud-samples-data/ccai/chat_sample.json', audio_uri: str='gs://cloud-samples-data/ccai/voice_6912.txt') -> contact_center_insights_v1.Conversation:
    if False:
        for i in range(10):
            print('nop')
    "Creates a conversation with a TTL value.\n\n    Args:\n        project_id:\n            The project identifier. For example, 'my-project'.\n        transcript_uri:\n            The Cloud Storage URI that points to a file that contains the\n            conversation transcript. Format is 'gs://{bucket_name}/{file.json}'.\n            For example, 'gs://cloud-samples-data/ccai/chat_sample.json'.\n        audio_uri:\n            The Cloud Storage URI that points to a file that contains the\n            conversation audio. Format is 'gs://{bucket_name}/{file.json}'.\n            For example, 'gs://cloud-samples-data/ccai/voice_6912.txt'.\n\n    Returns:\n        A conversation.\n    "
    parent = contact_center_insights_v1.ContactCenterInsightsClient.common_location_path(project_id, 'us-central1')
    conversation = contact_center_insights_v1.Conversation()
    conversation.data_source.gcs_source.transcript_uri = transcript_uri
    conversation.data_source.gcs_source.audio_uri = audio_uri
    conversation.medium = contact_center_insights_v1.Conversation.Medium.CHAT
    ttl = duration_pb2.Duration()
    ttl.seconds = 86400
    conversation.ttl = ttl
    insights_client = contact_center_insights_v1.ContactCenterInsightsClient()
    conversation = insights_client.create_conversation(parent=parent, conversation=conversation)
    print(f'Created {conversation.name}')
    return conversation