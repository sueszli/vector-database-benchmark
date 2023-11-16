from google.cloud import asset_v1

def sample_list_feeds():
    if False:
        while True:
            i = 10
    client = asset_v1.AssetServiceClient()
    request = asset_v1.ListFeedsRequest(parent='parent_value')
    response = client.list_feeds(request=request)
    print(response)