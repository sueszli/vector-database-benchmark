from google.cloud import datastream_v1

def sample_list_stream_objects():
    if False:
        while True:
            i = 10
    client = datastream_v1.DatastreamClient()
    request = datastream_v1.ListStreamObjectsRequest(parent='parent_value')
    page_result = client.list_stream_objects(request=request)
    for response in page_result:
        print(response)