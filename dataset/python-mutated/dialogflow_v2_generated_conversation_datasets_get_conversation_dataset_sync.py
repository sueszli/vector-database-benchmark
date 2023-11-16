from google.cloud import dialogflow_v2

def sample_get_conversation_dataset():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.ConversationDatasetsClient()
    request = dialogflow_v2.GetConversationDatasetRequest(name='name_value')
    response = client.get_conversation_dataset(request=request)
    print(response)