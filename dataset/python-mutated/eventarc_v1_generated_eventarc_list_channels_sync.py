from google.cloud import eventarc_v1

def sample_list_channels():
    if False:
        while True:
            i = 10
    client = eventarc_v1.EventarcClient()
    request = eventarc_v1.ListChannelsRequest(parent='parent_value')
    page_result = client.list_channels(request=request)
    for response in page_result:
        print(response)