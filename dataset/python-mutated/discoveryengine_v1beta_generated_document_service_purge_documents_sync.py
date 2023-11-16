from google.cloud import discoveryengine_v1beta

def sample_purge_documents():
    if False:
        while True:
            i = 10
    client = discoveryengine_v1beta.DocumentServiceClient()
    request = discoveryengine_v1beta.PurgeDocumentsRequest(parent='parent_value', filter='filter_value')
    operation = client.purge_documents(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)