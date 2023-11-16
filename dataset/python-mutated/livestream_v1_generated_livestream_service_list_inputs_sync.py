from google.cloud.video import live_stream_v1

def sample_list_inputs():
    if False:
        print('Hello World!')
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.ListInputsRequest(parent='parent_value')
    page_result = client.list_inputs(request=request)
    for response in page_result:
        print(response)