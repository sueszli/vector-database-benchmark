from google.cloud import bare_metal_solution_v2

def sample_update_network():
    if False:
        return 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.UpdateNetworkRequest()
    operation = client.update_network(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)