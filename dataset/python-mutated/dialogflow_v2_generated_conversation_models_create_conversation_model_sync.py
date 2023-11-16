from google.cloud import dialogflow_v2

def sample_create_conversation_model():
    if False:
        print('Hello World!')
    client = dialogflow_v2.ConversationModelsClient()
    conversation_model = dialogflow_v2.ConversationModel()
    conversation_model.display_name = 'display_name_value'
    conversation_model.datasets.dataset = 'dataset_value'
    request = dialogflow_v2.CreateConversationModelRequest(conversation_model=conversation_model)
    operation = client.create_conversation_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)