from google.cloud.video import live_stream_v1

def sample_list_events():
    if False:
        i = 10
        return i + 15
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.ListEventsRequest(parent='parent_value')
    page_result = client.list_events(request=request)
    for response in page_result:
        print(response)