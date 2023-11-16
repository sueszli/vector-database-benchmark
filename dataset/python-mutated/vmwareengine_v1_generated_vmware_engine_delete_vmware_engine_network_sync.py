from google.cloud import vmwareengine_v1

def sample_delete_vmware_engine_network():
    if False:
        return 10
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.DeleteVmwareEngineNetworkRequest(name='name_value')
    operation = client.delete_vmware_engine_network(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)