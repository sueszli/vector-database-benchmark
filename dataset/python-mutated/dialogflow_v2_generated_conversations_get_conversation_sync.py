from google.cloud import dialogflow_v2

def sample_get_conversation():
    if False:
        return 10
    client = dialogflow_v2.ConversationsClient()
    request = dialogflow_v2.GetConversationRequest(name='name_value')
    response = client.get_conversation(request=request)
    print(response)