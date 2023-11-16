from google.cloud.video import live_stream_v1

def sample_create_event():
    if False:
        while True:
            i = 10
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.CreateEventRequest(parent='parent_value', event_id='event_id_value')
    response = client.create_event(request=request)
    print(response)