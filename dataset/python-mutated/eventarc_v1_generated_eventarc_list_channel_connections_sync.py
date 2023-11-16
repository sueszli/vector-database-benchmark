from google.cloud import eventarc_v1

def sample_list_channel_connections():
    if False:
        return 10
    client = eventarc_v1.EventarcClient()
    request = eventarc_v1.ListChannelConnectionsRequest(parent='parent_value')
    page_result = client.list_channel_connections(request=request)
    for response in page_result:
        print(response)