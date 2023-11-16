from google.cloud import dialogflow_v2

def sample_export_document():
    if False:
        print('Hello World!')
    client = dialogflow_v2.DocumentsClient()
    request = dialogflow_v2.ExportDocumentRequest(name='name_value')
    operation = client.export_document(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)