from google.cloud import network_services_v1

def sample_create_mesh():
    if False:
        return 10
    client = network_services_v1.NetworkServicesClient()
    mesh = network_services_v1.Mesh()
    mesh.name = 'name_value'
    request = network_services_v1.CreateMeshRequest(parent='parent_value', mesh_id='mesh_id_value', mesh=mesh)
    operation = client.create_mesh(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)