from google.cloud import dialogflow_v2

def sample_deploy_conversation_model():
    if False:
        return 10
    client = dialogflow_v2.ConversationModelsClient()
    request = dialogflow_v2.DeployConversationModelRequest(name='name_value')
    operation = client.deploy_conversation_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)