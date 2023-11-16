from google.cloud import bare_metal_solution_v2

def sample_detach_lun():
    if False:
        print('Hello World!')
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.DetachLunRequest(instance='instance_value', lun='lun_value')
    operation = client.detach_lun(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)