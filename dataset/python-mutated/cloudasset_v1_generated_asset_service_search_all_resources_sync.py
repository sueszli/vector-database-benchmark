from google.cloud import asset_v1

def sample_search_all_resources():
    if False:
        print('Hello World!')
    client = asset_v1.AssetServiceClient()
    request = asset_v1.SearchAllResourcesRequest(scope='scope_value')
    page_result = client.search_all_resources(request=request)
    for response in page_result:
        print(response)