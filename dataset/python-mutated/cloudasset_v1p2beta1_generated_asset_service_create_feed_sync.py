from google.cloud import asset_v1p2beta1

def sample_create_feed():
    if False:
        print('Hello World!')
    client = asset_v1p2beta1.AssetServiceClient()
    feed = asset_v1p2beta1.Feed()
    feed.name = 'name_value'
    request = asset_v1p2beta1.CreateFeedRequest(parent='parent_value', feed_id='feed_id_value', feed=feed)
    response = client.create_feed(request=request)
    print(response)