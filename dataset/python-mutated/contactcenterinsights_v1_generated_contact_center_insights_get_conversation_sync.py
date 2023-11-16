from google.cloud import contact_center_insights_v1

def sample_get_conversation():
    if False:
        print('Hello World!')
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.GetConversationRequest(name='name_value')
    response = client.get_conversation(request=request)
    print(response)