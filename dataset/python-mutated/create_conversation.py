from google.cloud import contact_center_insights_v1

def create_conversation(project_id: str, transcript_uri: str='gs://cloud-samples-data/ccai/chat_sample.json', audio_uri: str='gs://cloud-samples-data/ccai/voice_6912.txt') -> contact_center_insights_v1.Conversation:
    if False:
        i = 10
        return i + 15
    "Creates a conversation.\n\n    Args:\n        project_id:\n            The project identifier. For example, 'my-project'.\n        transcript_uri:\n            The Cloud Storage URI that points to a file that contains the\n            conversation transcript. Format is 'gs://{bucket_name}/{file.json}'.\n            For example, 'gs://cloud-samples-data/ccai/chat_sample.json'.\n        audio_uri:\n            The Cloud Storage URI that points to a file that contains the\n            conversation audio. Format is 'gs://{bucket_name}/{file.json}'.\n            For example, 'gs://cloud-samples-data/ccai/voice_6912.txt'.\n\n    Returns:\n        A conversation.\n    "
    parent = contact_center_insights_v1.ContactCenterInsightsClient.common_location_path(project_id, 'us-central1')
    conversation = contact_center_insights_v1.Conversation()
    conversation.data_source.gcs_source.transcript_uri = transcript_uri
    conversation.data_source.gcs_source.audio_uri = audio_uri
    conversation.medium = contact_center_insights_v1.Conversation.Medium.CHAT
    insights_client = contact_center_insights_v1.ContactCenterInsightsClient()
    conversation = insights_client.create_conversation(parent=parent, conversation=conversation)
    print(f'Created {conversation.name}')
    return conversation