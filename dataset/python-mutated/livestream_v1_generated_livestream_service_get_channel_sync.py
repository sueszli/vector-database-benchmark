from google.cloud.video import live_stream_v1

def sample_get_channel():
    if False:
        return 10
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.GetChannelRequest(name='name_value')
    response = client.get_channel(request=request)
    print(response)