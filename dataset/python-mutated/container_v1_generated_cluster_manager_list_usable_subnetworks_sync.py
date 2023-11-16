from google.cloud import container_v1

def sample_list_usable_subnetworks():
    if False:
        print('Hello World!')
    client = container_v1.ClusterManagerClient()
    request = container_v1.ListUsableSubnetworksRequest()
    page_result = client.list_usable_subnetworks(request=request)
    for response in page_result:
        print(response)