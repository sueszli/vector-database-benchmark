from google.cloud import bare_metal_solution_v2

def sample_evict_volume():
    if False:
        print('Hello World!')
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.EvictVolumeRequest(name='name_value')
    operation = client.evict_volume(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)