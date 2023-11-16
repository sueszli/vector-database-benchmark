from google.cloud import contact_center_insights_v1

def sample_list_conversations():
    if False:
        return 10
    client = contact_center_insights_v1.ContactCenterInsightsClient()
    request = contact_center_insights_v1.ListConversationsRequest(parent='parent_value')
    page_result = client.list_conversations(request=request)
    for response in page_result:
        print(response)