from google.cloud import bare_metal_solution_v2

def sample_stop_instance():
    if False:
        while True:
            i = 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.StopInstanceRequest(name='name_value')
    operation = client.stop_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)