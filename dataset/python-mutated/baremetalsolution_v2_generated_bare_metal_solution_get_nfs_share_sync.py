from google.cloud import bare_metal_solution_v2

def sample_get_nfs_share():
    if False:
        for i in range(10):
            print('nop')
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.GetNfsShareRequest(name='name_value')
    response = client.get_nfs_share(request=request)
    print(response)