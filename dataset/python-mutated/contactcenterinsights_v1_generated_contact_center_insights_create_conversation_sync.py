from google.cloud import contact_center_insights_v1

def sample_create_conversation():
    if False:
        for i in range(10):
            print('nop')
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.CreateConversationRequest(parent='parent_value')
    response = client.create_conversation(request=request)
    print(response)