from google.cloud import compute_v1

def sample_get_effective_firewalls():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.InstancesClient()
    request = compute_v1.GetEffectiveFirewallsInstanceRequest(instance='instance_value', network_interface='network_interface_value', project='project_value', zone='zone_value')
    response = client.get_effective_firewalls(request=request)
    print(response)