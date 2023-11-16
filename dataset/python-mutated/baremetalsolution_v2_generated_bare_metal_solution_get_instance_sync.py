from google.cloud import bare_metal_solution_v2

def sample_get_instance():
    if False:
        i = 10
        return i + 15
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.GetInstanceRequest(name='name_value')
    response = client.get_instance(request=request)
    print(response)