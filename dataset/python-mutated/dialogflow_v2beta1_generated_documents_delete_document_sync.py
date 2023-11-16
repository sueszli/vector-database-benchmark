from google.cloud import dialogflow_v2beta1

def sample_delete_document():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.DocumentsClient()
    request = dialogflow_v2beta1.DeleteDocumentRequest(name='name_value')
    operation = client.delete_document(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)