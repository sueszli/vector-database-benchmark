from google.cloud import dialogflow_v2

def sample_list_conversation_profiles():
    if False:
        return 10
    client = dialogflow_v2.ConversationProfilesClient()
    request = dialogflow_v2.ListConversationProfilesRequest(parent='parent_value')
    page_result = client.list_conversation_profiles(request=request)
    for response in page_result:
        print(response)