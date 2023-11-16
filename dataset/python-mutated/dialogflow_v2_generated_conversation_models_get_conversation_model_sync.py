from google.cloud import dialogflow_v2

def sample_get_conversation_model():
    if False:
        print('Hello World!')
    client = dialogflow_v2.ConversationModelsClient()
    request = dialogflow_v2.GetConversationModelRequest(name='name_value')
    response = client.get_conversation_model(request=request)
    print(response)