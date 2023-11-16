from google.cloud import dialogflow_v2beta1

def sample_batch_delete_intents():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.IntentsClient()
    intents = dialogflow_v2beta1.Intent()
    intents.display_name = 'display_name_value'
    request = dialogflow_v2beta1.BatchDeleteIntentsRequest(parent='parent_value', intents=intents)
    operation = client.batch_delete_intents(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)