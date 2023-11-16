from google.cloud import bare_metal_solution_v2

def sample_get_volume():
    if False:
        return 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.GetVolumeRequest(name='name_value')
    response = client.get_volume(request=request)
    print(response)