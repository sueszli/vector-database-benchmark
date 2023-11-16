from google.cloud import datastream_v1

def sample_list_private_connections():
    if False:
        while True:
            i = 10
    client = datastream_v1.DatastreamClient()
    request = datastream_v1.ListPrivateConnectionsRequest(parent='parent_value')
    page_result = client.list_private_connections(request=request)
    for response in page_result:
        print(response)