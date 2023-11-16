from google.cloud import asset_v1p2beta1

def sample_update_feed():
    if False:
        i = 10
        return i + 15
    client = asset_v1p2beta1.AssetServiceClient()
    feed = asset_v1p2beta1.Feed()
    feed.name = 'name_value'
    request = asset_v1p2beta1.UpdateFeedRequest(feed=feed)
    response = client.update_feed(request=request)
    print(response)