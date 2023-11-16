from google.cloud import dialogflow_v2

def sample_get_conversation_model_evaluation():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.ConversationModelsClient()
    request = dialogflow_v2.GetConversationModelEvaluationRequest(name='name_value')
    response = client.get_conversation_model_evaluation(request=request)
    print(response)