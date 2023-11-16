from google.cloud import asset_v1

def sample_update_feed():
    if False:
        print('Hello World!')
    client = asset_v1.AssetServiceClient()
    feed = asset_v1.Feed()
    feed.name = 'name_value'
    request = asset_v1.UpdateFeedRequest(feed=feed)
    response = client.update_feed(request=request)
    print(response)