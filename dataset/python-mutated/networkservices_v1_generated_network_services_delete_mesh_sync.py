from google.cloud import network_services_v1

def sample_delete_mesh():
    if False:
        while True:
            i = 10
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.DeleteMeshRequest(name='name_value')
    operation = client.delete_mesh(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)