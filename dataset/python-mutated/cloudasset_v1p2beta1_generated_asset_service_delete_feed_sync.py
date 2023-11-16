from google.cloud import asset_v1p2beta1

def sample_delete_feed():
    if False:
        i = 10
        return i + 15
    client = asset_v1p2beta1.AssetServiceClient()
    request = asset_v1p2beta1.DeleteFeedRequest(name='name_value')
    client.delete_feed(request=request)