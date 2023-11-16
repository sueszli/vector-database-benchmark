from google.cloud import discoveryengine_v1alpha

def sample_create_data_store():
    if False:
        return 10
    client = discoveryengine_v1alpha.DataStoreServiceClient()
    data_store = discoveryengine_v1alpha.DataStore()
    data_store.display_name = 'display_name_value'
    request = discoveryengine_v1alpha.CreateDataStoreRequest(parent='parent_value', data_store=data_store, data_store_id='data_store_id_value')
    operation = client.create_data_store(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)