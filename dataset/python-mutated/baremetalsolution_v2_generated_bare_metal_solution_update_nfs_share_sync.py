from google.cloud import bare_metal_solution_v2

def sample_update_nfs_share():
    if False:
        for i in range(10):
            print('nop')
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.UpdateNfsShareRequest()
    operation = client.update_nfs_share(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)