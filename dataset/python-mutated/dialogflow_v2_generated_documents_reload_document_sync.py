from google.cloud import dialogflow_v2

def sample_reload_document():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.DocumentsClient()
    request = dialogflow_v2.ReloadDocumentRequest(content_uri='content_uri_value', name='name_value')
    operation = client.reload_document(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)