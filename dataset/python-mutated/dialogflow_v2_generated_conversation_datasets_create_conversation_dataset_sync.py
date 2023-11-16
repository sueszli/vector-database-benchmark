from google.cloud import dialogflow_v2

def sample_create_conversation_dataset():
    if False:
        print('Hello World!')
    client = dialogflow_v2.ConversationDatasetsClient()
    conversation_dataset = dialogflow_v2.ConversationDataset()
    conversation_dataset.display_name = 'display_name_value'
    request = dialogflow_v2.CreateConversationDatasetRequest(parent='parent_value', conversation_dataset=conversation_dataset)
    operation = client.create_conversation_dataset(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)