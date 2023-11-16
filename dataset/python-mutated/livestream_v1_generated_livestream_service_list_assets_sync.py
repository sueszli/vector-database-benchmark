from google.cloud.video import live_stream_v1

def sample_list_assets():
    if False:
        print('Hello World!')
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.ListAssetsRequest(parent='parent_value')
    page_result = client.list_assets(request=request)
    for response in page_result:
        print(response)