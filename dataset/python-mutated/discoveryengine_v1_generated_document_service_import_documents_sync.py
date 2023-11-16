from google.cloud import discoveryengine_v1

def sample_import_documents():
    if False:
        print('Hello World!')
    client = discoveryengine_v1.DocumentServiceClient()
    request = discoveryengine_v1.ImportDocumentsRequest(parent='parent_value')
    operation = client.import_documents(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)