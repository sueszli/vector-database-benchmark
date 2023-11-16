from google.cloud import vmwareengine_v1

def sample_create_network_policy():
    if False:
        i = 10
        return i + 15
    client = vmwareengine_v1.VmwareEngineClient()
    network_policy = vmwareengine_v1.NetworkPolicy()
    network_policy.edge_services_cidr = 'edge_services_cidr_value'
    request = vmwareengine_v1.CreateNetworkPolicyRequest(parent='parent_value', network_policy_id='network_policy_id_value', network_policy=network_policy)
    operation = client.create_network_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)