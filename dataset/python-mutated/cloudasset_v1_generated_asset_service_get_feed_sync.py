from google.cloud import asset_v1

def sample_get_feed():
    if False:
        return 10
    client = asset_v1.AssetServiceClient()
    request = asset_v1.GetFeedRequest(name='name_value')
    response = client.get_feed(request=request)
    print(response)