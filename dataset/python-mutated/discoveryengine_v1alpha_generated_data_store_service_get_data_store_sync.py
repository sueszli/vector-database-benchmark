from google.cloud import discoveryengine_v1alpha

def sample_get_data_store():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1alpha.DataStoreServiceClient()
    request = discoveryengine_v1alpha.GetDataStoreRequest(name='name_value')
    response = client.get_data_store(request=request)
    print(response)