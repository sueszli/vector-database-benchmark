from google.cloud import contact_center_insights_v1

def sample_ingest_conversations():
    if False:
        for i in range(10):
            print('nop')
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    gcs_source = contact_center_insights_v1.GcsSource()
    gcs_source.bucket_uri = 'bucket_uri_value'
    transcript_object_config = contact_center_insights_v1.TranscriptObjectConfig()
    transcript_object_config.medium = 'CHAT'
    request = contact_center_insights_v1.IngestConversationsRequest(gcs_source=gcs_source, transcript_object_config=transcript_object_config, parent='parent_value')
    operation = client.ingest_conversations(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)