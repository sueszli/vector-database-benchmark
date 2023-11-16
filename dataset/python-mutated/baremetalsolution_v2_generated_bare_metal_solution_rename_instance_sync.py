from google.cloud import bare_metal_solution_v2

def sample_rename_instance():
    if False:
        for i in range(10):
            print('nop')
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.RenameInstanceRequest(name='name_value', new_instance_id='new_instance_id_value')
    response = client.rename_instance(request=request)
    print(response)