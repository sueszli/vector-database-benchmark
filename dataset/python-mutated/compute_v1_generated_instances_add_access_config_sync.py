from google.cloud import compute_v1

def sample_add_access_config():
    if False:
        i = 10
        return i + 15
    client = compute_v1.InstancesClient()
    request = compute_v1.AddAccessConfigInstanceRequest(instance='instance_value', network_interface='network_interface_value', project='project_value', zone='zone_value')
    response = client.add_access_config(request=request)
    print(response)