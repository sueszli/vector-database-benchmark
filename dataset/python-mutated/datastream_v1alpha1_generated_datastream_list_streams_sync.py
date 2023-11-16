from google.cloud import datastream_v1alpha1

def sample_list_streams():
    if False:
        for i in range(10):
            print('nop')
    client = datastream_v1alpha1.DatastreamClient()
    request = datastream_v1alpha1.ListStreamsRequest(parent='parent_value')
    page_result = client.list_streams(request=request)
    for response in page_result:
        print(response)