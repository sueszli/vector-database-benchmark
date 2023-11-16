from google.cloud import compute_v1

def sample_update_network_interface():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.InstancesClient()
    request = compute_v1.UpdateNetworkInterfaceInstanceRequest(instance='instance_value', network_interface='network_interface_value', project='project_value', zone='zone_value')
    response = client.update_network_interface(request=request)
    print(response)