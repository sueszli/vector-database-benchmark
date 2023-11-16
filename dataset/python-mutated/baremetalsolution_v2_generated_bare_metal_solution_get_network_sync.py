from google.cloud import bare_metal_solution_v2

def sample_get_network():
    if False:
        return 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.GetNetworkRequest(name='name_value')
    response = client.get_network(request=request)
    print(response)