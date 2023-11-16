from google.cloud import discoveryengine_v1

def sample_purge_documents():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1.DocumentServiceClient()
    request = discoveryengine_v1.PurgeDocumentsRequest(parent='parent_value', filter='filter_value')
    operation = client.purge_documents(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)