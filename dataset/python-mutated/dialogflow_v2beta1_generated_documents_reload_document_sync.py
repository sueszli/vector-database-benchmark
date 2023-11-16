from google.cloud import dialogflow_v2beta1

def sample_reload_document():
    if False:
        return 10
    client = dialogflow_v2beta1.DocumentsClient()
    request = dialogflow_v2beta1.ReloadDocumentRequest(name='name_value')
    operation = client.reload_document(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)