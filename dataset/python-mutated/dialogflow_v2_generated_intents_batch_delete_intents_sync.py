from google.cloud import dialogflow_v2

def sample_batch_delete_intents():
    if False:
        return 10
    client = dialogflow_v2.IntentsClient()
    intents = dialogflow_v2.Intent()
    intents.display_name = 'display_name_value'
    request = dialogflow_v2.BatchDeleteIntentsRequest(parent='parent_value', intents=intents)
    operation = client.batch_delete_intents(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)