from google.cloud import vmwareengine_v1

def sample_update_network_policy():
    if False:
        i = 10
        return i + 15
    client = vmwareengine_v1.VmwareEngineClient()
    network_policy = vmwareengine_v1.NetworkPolicy()
    network_policy.edge_services_cidr = 'edge_services_cidr_value'
    request = vmwareengine_v1.UpdateNetworkPolicyRequest(network_policy=network_policy)
    operation = client.update_network_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)