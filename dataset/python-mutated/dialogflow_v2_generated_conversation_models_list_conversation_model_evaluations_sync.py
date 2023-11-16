from google.cloud import dialogflow_v2

def sample_list_conversation_model_evaluations():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.ConversationModelsClient()
    request = dialogflow_v2.ListConversationModelEvaluationsRequest(parent='parent_value')
    page_result = client.list_conversation_model_evaluations(request=request)
    for response in page_result:
        print(response)