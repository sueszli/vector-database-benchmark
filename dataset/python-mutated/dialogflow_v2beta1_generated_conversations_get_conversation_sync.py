from google.cloud import dialogflow_v2beta1

def sample_get_conversation():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.ConversationsClient()
    request = dialogflow_v2beta1.GetConversationRequest(name='name_value')
    response = client.get_conversation(request=request)
    print(response)