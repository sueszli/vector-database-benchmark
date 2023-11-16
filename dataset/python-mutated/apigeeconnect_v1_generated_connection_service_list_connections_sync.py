from google.cloud import apigeeconnect_v1

def sample_list_connections():
    if False:
        i = 10
        return i + 15
    client = apigeeconnect_v1.ConnectionServiceClient()
    request = apigeeconnect_v1.ListConnectionsRequest(parent='parent_value')
    page_result = client.list_connections(request=request)
    for response in page_result:
        print(response)