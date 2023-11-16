from google.cloud import bare_metal_solution_v2

def sample_evict_lun():
    if False:
        i = 10
        return i + 15
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.EvictLunRequest(name='name_value')
    operation = client.evict_lun(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)