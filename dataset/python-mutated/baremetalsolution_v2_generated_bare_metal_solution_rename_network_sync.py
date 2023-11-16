from google.cloud import bare_metal_solution_v2

def sample_rename_network():
    if False:
        while True:
            i = 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.RenameNetworkRequest(name='name_value', new_network_id='new_network_id_value')
    response = client.rename_network(request=request)
    print(response)