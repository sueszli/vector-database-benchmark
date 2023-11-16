from google.cloud import dialogflow_v2

def sample_undeploy_conversation_model():
    if False:
        print('Hello World!')
    client = dialogflow_v2.ConversationModelsClient()
    request = dialogflow_v2.UndeployConversationModelRequest(name='name_value')
    operation = client.undeploy_conversation_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)