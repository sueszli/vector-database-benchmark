from google.cloud.video import live_stream_v1

def sample_get_asset():
    if False:
        while True:
            i = 10
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.GetAssetRequest(name='name_value')
    response = client.get_asset(request=request)
    print(response)