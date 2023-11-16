from google.cloud import vmwareengine_v1

def sample_update_subnet():
    if False:
        print('Hello World!')
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.UpdateSubnetRequest()
    operation = client.update_subnet(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)