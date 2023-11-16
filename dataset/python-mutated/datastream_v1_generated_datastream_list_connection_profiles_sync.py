from google.cloud import datastream_v1

def sample_list_connection_profiles():
    if False:
        while True:
            i = 10
    client = datastream_v1.DatastreamClient()
    request = datastream_v1.ListConnectionProfilesRequest(parent='parent_value')
    page_result = client.list_connection_profiles(request=request)
    for response in page_result:
        print(response)