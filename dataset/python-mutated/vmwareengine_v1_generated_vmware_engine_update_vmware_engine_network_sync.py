from google.cloud import vmwareengine_v1

def sample_update_vmware_engine_network():
    if False:
        return 10
    client = vmwareengine_v1.VmwareEngineClient()
    vmware_engine_network = vmwareengine_v1.VmwareEngineNetwork()
    vmware_engine_network.type_ = 'LEGACY'
    request = vmwareengine_v1.UpdateVmwareEngineNetworkRequest(vmware_engine_network=vmware_engine_network)
    operation = client.update_vmware_engine_network(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)