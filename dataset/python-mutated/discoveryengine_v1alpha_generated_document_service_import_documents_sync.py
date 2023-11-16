from google.cloud import discoveryengine_v1alpha

def sample_import_documents():
    if False:
        while True:
            i = 10
    client = discoveryengine_v1alpha.DocumentServiceClient()
    request = discoveryengine_v1alpha.ImportDocumentsRequest(parent='parent_value')
    operation = client.import_documents(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)