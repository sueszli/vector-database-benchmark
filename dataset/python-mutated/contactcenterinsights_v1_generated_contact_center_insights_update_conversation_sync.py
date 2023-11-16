from google.cloud import contact_center_insights_v1

def sample_update_conversation():
    if False:
        while True:
            i = 10
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.UpdateConversationRequest()
    response = client.update_conversation(request=request)
    print(response)