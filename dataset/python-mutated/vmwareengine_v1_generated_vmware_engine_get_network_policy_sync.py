from google.cloud import vmwareengine_v1

def sample_get_network_policy():
    if False:
        for i in range(10):
            print('nop')
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.GetNetworkPolicyRequest(name='name_value')
    response = client.get_network_policy(request=request)
    print(response)