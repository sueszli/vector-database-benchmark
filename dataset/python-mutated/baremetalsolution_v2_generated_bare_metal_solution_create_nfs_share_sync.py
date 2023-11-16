from google.cloud import bare_metal_solution_v2

def sample_create_nfs_share():
    if False:
        i = 10
        return i + 15
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.CreateNfsShareRequest(parent='parent_value')
    operation = client.create_nfs_share(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)