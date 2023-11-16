from google.cloud import vmwareengine_v1

def sample_get_vmware_engine_network():
    if False:
        for i in range(10):
            print('nop')
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.GetVmwareEngineNetworkRequest(name='name_value')
    response = client.get_vmware_engine_network(request=request)
    print(response)