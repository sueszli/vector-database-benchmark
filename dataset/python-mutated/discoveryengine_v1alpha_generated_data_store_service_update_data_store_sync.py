from google.cloud import discoveryengine_v1alpha

def sample_update_data_store():
    if False:
        i = 10
        return i + 15
    client = discoveryengine_v1alpha.DataStoreServiceClient()
    data_store = discoveryengine_v1alpha.DataStore()
    data_store.display_name = 'display_name_value'
    request = discoveryengine_v1alpha.UpdateDataStoreRequest(data_store=data_store)
    response = client.update_data_store(request=request)
    print(response)