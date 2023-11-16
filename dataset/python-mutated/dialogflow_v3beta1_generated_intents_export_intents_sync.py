from google.cloud import dialogflowcx_v3beta1

def sample_export_intents():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3beta1.IntentsClient()
    request = dialogflowcx_v3beta1.ExportIntentsRequest(intents_uri='intents_uri_value', parent='parent_value', intents=['intents_value1', 'intents_value2'])
    operation = client.export_intents(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)