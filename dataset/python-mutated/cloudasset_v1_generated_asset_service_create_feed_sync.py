from google.cloud import asset_v1

def sample_create_feed():
    if False:
        return 10
    client = asset_v1.AssetServiceClient()
    feed = asset_v1.Feed()
    feed.name = 'name_value'
    request = asset_v1.CreateFeedRequest(parent='parent_value', feed_id='feed_id_value', feed=feed)
    response = client.create_feed(request=request)
    print(response)