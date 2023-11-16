from google.cloud import asset_v1p5beta1

def sample_list_assets():
    if False:
        for i in range(10):
            print('nop')
    client = asset_v1p5beta1.AssetServiceClient()
    request = asset_v1p5beta1.ListAssetsRequest(parent='parent_value')
    page_result = client.list_assets(request=request)
    for response in page_result:
        print(response)