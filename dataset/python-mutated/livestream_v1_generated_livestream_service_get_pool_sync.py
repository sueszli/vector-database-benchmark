from google.cloud.video import live_stream_v1

def sample_get_pool():
    if False:
        i = 10
        return i + 15
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.GetPoolRequest(name='name_value')
    response = client.get_pool(request=request)
    print(response)