from google.cloud import dialogflow_v2

def sample_create_conversation_model_evaluation():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.ConversationModelsClient()
    request = dialogflow_v2.CreateConversationModelEvaluationRequest(parent='parent_value')
    operation = client.create_conversation_model_evaluation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)