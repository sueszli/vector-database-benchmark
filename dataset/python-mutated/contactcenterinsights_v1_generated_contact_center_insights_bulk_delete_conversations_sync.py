from google.cloud import contact_center_insights_v1

def sample_bulk_delete_conversations():
    if False:
        while True:
            i = 10
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.BulkDeleteConversationsRequest(parent='parent_value')
    operation = client.bulk_delete_conversations(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)