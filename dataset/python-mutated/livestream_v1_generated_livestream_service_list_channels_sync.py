from google.cloud.video import live_stream_v1

def sample_list_channels():
    if False:
        i = 10
        return i + 15
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.ListChannelsRequest(parent='parent_value')
    page_result = client.list_channels(request=request)
    for response in page_result:
        print(response)