from google.cloud import network_services_v1

def sample_get_mesh():
    if False:
        while True:
            i = 10
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.GetMeshRequest(name='name_value')
    response = client.get_mesh(request=request)
    print(response)