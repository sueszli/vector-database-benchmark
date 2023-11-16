from google.cloud import edgenetwork_v1

def sample_list_routers():
    if False:
        i = 10
        return i + 15
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.ListRoutersRequest(parent='parent_value')
    page_result = client.list_routers(request=request)
    for response in page_result:
        print(response)