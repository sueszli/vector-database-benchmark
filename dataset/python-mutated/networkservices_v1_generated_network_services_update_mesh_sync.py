from google.cloud import network_services_v1

def sample_update_mesh():
    if False:
        return 10
    client = network_services_v1.NetworkServicesClient()
    mesh = network_services_v1.Mesh()
    mesh.name = 'name_value'
    request = network_services_v1.UpdateMeshRequest(mesh=mesh)
    operation = client.update_mesh(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)