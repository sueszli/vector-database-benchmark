from google.cloud import dialogflowcx_v3

def sample_import_intents():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3.IntentsClient()
    request = dialogflowcx_v3.ImportIntentsRequest(intents_uri='intents_uri_value', parent='parent_value')
    operation = client.import_intents(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)