from google.cloud import dialogflow_v2beta1

def sample_batch_update_intents():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.IntentsClient()
    request = dialogflow_v2beta1.BatchUpdateIntentsRequest(intent_batch_uri='intent_batch_uri_value', parent='parent_value')
    operation = client.batch_update_intents(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)