from google.cloud import asset_v1

def sample_list_assets():
    if False:
        print('Hello World!')
    client = asset_v1.AssetServiceClient()
    request = asset_v1.ListAssetsRequest(parent='parent_value')
    page_result = client.list_assets(request=request)
    for response in page_result:
        print(response)