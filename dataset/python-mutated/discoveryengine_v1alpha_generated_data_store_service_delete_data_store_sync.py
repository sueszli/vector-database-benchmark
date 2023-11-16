from google.cloud import discoveryengine_v1alpha

def sample_delete_data_store():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1alpha.DataStoreServiceClient()
    request = discoveryengine_v1alpha.DeleteDataStoreRequest(name='name_value')
    operation = client.delete_data_store(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)