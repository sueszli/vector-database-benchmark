from google.cloud import vmwareengine_v1

def sample_delete_network_policy():
    if False:
        print('Hello World!')
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.DeleteNetworkPolicyRequest(name='name_value')
    operation = client.delete_network_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)