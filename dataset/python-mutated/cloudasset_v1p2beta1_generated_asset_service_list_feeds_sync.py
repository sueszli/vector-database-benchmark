from google.cloud import asset_v1p2beta1

def sample_list_feeds():
    if False:
        return 10
    client = asset_v1p2beta1.AssetServiceClient()
    request = asset_v1p2beta1.ListFeedsRequest(parent='parent_value')
    response = client.list_feeds(request=request)
    print(response)