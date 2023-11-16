from google.cloud import bare_metal_solution_v2

def sample_delete_nfs_share():
    if False:
        return 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.DeleteNfsShareRequest(name='name_value')
    operation = client.delete_nfs_share(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)