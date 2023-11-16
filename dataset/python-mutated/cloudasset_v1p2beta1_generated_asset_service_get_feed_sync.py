from google.cloud import asset_v1p2beta1

def sample_get_feed():
    if False:
        while True:
            i = 10
    client = asset_v1p2beta1.AssetServiceClient()
    request = asset_v1p2beta1.GetFeedRequest(name='name_value')
    response = client.get_feed(request=request)
    print(response)