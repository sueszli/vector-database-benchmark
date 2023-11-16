from google.cloud import dialogflow_v2

def sample_delete_conversation_dataset():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.ConversationDatasetsClient()
    request = dialogflow_v2.DeleteConversationDatasetRequest(name='name_value')
    operation = client.delete_conversation_dataset(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)