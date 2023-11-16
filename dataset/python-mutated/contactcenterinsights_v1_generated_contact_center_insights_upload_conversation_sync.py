from google.cloud import contact_center_insights_v1

def sample_upload_conversation():
    if False:
        for i in range(10):
            print('nop')
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.UploadConversationRequest(parent='parent_value')
    operation = client.upload_conversation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)