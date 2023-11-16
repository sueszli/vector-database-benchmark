from google.cloud import datastream_v1

def sample_list_streams():
    if False:
        i = 10
        return i + 15
    client = datastream_v1.DatastreamClient()
    request = datastream_v1.ListStreamsRequest(parent='parent_value')
    page_result = client.list_streams(request=request)
    for response in page_result:
        print(response)