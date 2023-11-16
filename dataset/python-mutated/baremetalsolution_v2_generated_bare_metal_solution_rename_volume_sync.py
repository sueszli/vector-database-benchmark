from google.cloud import bare_metal_solution_v2

def sample_rename_volume():
    if False:
        while True:
            i = 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.RenameVolumeRequest(name='name_value', new_volume_id='new_volume_id_value')
    response = client.rename_volume(request=request)
    print(response)