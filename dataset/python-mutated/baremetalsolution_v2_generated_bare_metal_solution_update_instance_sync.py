from google.cloud import bare_metal_solution_v2

def sample_update_instance():
    if False:
        return 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.UpdateInstanceRequest()
    operation = client.update_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)