from google.cloud import edgenetwork_v1

def sample_list_interconnects():
    if False:
        for i in range(10):
            print('nop')
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.ListInterconnectsRequest(parent='parent_value')
    page_result = client.list_interconnects(request=request)
    for response in page_result:
        print(response)