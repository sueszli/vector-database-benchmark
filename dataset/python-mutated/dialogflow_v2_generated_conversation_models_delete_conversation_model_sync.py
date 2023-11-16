from google.cloud import dialogflow_v2

def sample_delete_conversation_model():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.ConversationModelsClient()
    request = dialogflow_v2.DeleteConversationModelRequest(name='name_value')
    operation = client.delete_conversation_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)