from google.cloud.video import live_stream_v1

def sample_delete_event():
    if False:
        return 10
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.DeleteEventRequest(name='name_value')
    client.delete_event(request=request)