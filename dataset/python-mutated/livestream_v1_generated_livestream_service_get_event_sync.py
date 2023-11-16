from google.cloud.video import live_stream_v1

def sample_get_event():
    if False:
        i = 10
        return i + 15
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.GetEventRequest(name='name_value')
    response = client.get_event(request=request)
    print(response)