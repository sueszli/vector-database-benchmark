from google.cloud import container_v1beta1

def sample_list_locations():
    if False:
        i = 10
        return i + 15
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.ListLocationsRequest(parent='parent_value')
    response = client.list_locations(request=request)
    print(response)