from google.cloud import dialogflow_v2beta1

def sample_complete_conversation():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2beta1.ConversationsClient()
    request = dialogflow_v2beta1.CompleteConversationRequest(name='name_value')
    response = client.complete_conversation(request=request)
    print(response)