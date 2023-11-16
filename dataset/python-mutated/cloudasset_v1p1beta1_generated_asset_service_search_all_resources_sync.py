from google.cloud import asset_v1p1beta1

def sample_search_all_resources():
    if False:
        i = 10
        return i + 15
    client = asset_v1p1beta1.AssetServiceClient()
    request = asset_v1p1beta1.SearchAllResourcesRequest(scope='scope_value')
    page_result = client.search_all_resources(request=request)
    for response in page_result:
        print(response)