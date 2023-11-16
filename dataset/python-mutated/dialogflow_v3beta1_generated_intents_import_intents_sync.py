from google.cloud import dialogflowcx_v3beta1

def sample_import_intents():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.IntentsClient()
    request = dialogflowcx_v3beta1.ImportIntentsRequest(intents_uri='intents_uri_value', parent='parent_value')
    operation = client.import_intents(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)