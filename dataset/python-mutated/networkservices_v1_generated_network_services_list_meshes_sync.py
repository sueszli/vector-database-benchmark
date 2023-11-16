from google.cloud import network_services_v1

def sample_list_meshes():
    if False:
        for i in range(10):
            print('nop')
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.ListMeshesRequest(parent='parent_value')
    page_result = client.list_meshes(request=request)
    for response in page_result:
        print(response)