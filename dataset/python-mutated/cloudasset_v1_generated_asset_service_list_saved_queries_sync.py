from google.cloud import asset_v1

def sample_list_saved_queries():
    if False:
        print('Hello World!')
    client = asset_v1.AssetServiceClient()
    request = asset_v1.ListSavedQueriesRequest(parent='parent_value')
    page_result = client.list_saved_queries(request=request)
    for response in page_result:
        print(response)