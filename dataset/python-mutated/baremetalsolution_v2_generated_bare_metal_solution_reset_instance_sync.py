from google.cloud import bare_metal_solution_v2

def sample_reset_instance():
    if False:
        print('Hello World!')
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.ResetInstanceRequest(name='name_value')
    operation = client.reset_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)