from google.cloud import edgenetwork_v1

def sample_list_subnets():
    if False:
        while True:
            i = 10
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.ListSubnetsRequest(parent='parent_value')
    page_result = client.list_subnets(request=request)
    for response in page_result:
        print(response)