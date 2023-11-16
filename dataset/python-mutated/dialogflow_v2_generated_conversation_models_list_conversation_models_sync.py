from google.cloud import dialogflow_v2

def sample_list_conversation_models():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.ConversationModelsClient()
    request = dialogflow_v2.ListConversationModelsRequest(parent='parent_value')
    page_result = client.list_conversation_models(request=request)
    for response in page_result:
        print(response)