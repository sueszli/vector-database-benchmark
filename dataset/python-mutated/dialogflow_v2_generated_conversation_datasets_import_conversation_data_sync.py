from google.cloud import dialogflow_v2

def sample_import_conversation_data():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.ConversationDatasetsClient()
    input_config = dialogflow_v2.InputConfig()
    input_config.gcs_source.uris = ['uris_value1', 'uris_value2']
    request = dialogflow_v2.ImportConversationDataRequest(name='name_value', input_config=input_config)
    operation = client.import_conversation_data(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)