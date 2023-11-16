from google.cloud import bare_metal_solution_v2

def sample_resize_volume():
    if False:
        print('Hello World!')
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.ResizeVolumeRequest(volume='volume_value')
    operation = client.resize_volume(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)