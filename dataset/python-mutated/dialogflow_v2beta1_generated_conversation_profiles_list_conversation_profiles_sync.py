from google.cloud import dialogflow_v2beta1

def sample_list_conversation_profiles():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.ConversationProfilesClient()
    request = dialogflow_v2beta1.ListConversationProfilesRequest(parent='parent_value')
    page_result = client.list_conversation_profiles(request=request)
    for response in page_result:
        print(response)