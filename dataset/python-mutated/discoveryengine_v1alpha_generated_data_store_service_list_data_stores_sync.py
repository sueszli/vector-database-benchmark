from google.cloud import discoveryengine_v1alpha

def sample_list_data_stores():
    if False:
        while True:
            i = 10
    client = discoveryengine_v1alpha.DataStoreServiceClient()
    request = discoveryengine_v1alpha.ListDataStoresRequest(parent='parent_value')
    page_result = client.list_data_stores(request=request)
    for response in page_result:
        print(response)