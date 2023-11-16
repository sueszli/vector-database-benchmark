from google.cloud import container_v1beta1

def sample_list_usable_subnetworks():
    if False:
        while True:
            i = 10
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.ListUsableSubnetworksRequest(parent='parent_value')
    page_result = client.list_usable_subnetworks(request=request)
    for response in page_result:
        print(response)