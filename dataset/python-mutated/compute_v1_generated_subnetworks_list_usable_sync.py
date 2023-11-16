from google.cloud import compute_v1

def sample_list_usable():
    if False:
        print('Hello World!')
    client = compute_v1.SubnetworksClient()
    request = compute_v1.ListUsableSubnetworksRequest(project='project_value')
    page_result = client.list_usable(request=request)
    for response in page_result:
        print(response)