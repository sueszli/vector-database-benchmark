from google.cloud import ids_v1

def sample_list_endpoints():
    if False:
        while True:
            i = 10
    client = ids_v1.IDSClient()
    request = ids_v1.ListEndpointsRequest(parent='parent_value')
    page_result = client.list_endpoints(request=request)
    for response in page_result:
        print(response)